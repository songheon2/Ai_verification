import copy
import zlib
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple
from Automation.SolverStatus import SolverLimitReached, check_deadline


DEFAULT_TOL = 1e-9

# 누적 반올림 오차 검사 (Reluplex 논문 5절: "Pivot 5,000번마다 오차를 검사하고
# 10^-6을 넘으면 tableau를 복원했다"). 웜 스타트는 자식이 부모 tableau를 그대로
# 물려받으므로, 루트부터 현재 노드까지의 피벗이 한 tableau에 전부 누적된다.
# 그래서 피벗 수는 호출 단위가 아니라 tableau 계보 단위(pivots_since_check)로 센다.
DRIFT_TOL = 1e-6
# 마지막 검사 이후 이만큼 피벗이 쌓였으면 SAT/UNSAT을 선언하기 직전에 검사한다.
# 모든 선언마다 검사하면 LP가 작은 인스턴스에서는 검사 비용이 LP 풀이의 약 30%
# 까지 올라갔다(sat_v18_c87 실측). 논문의 5000보다 훨씬 촘촘한 값이다.
DRIFT_CHECK_AT_EXIT = 200
# 한 호출이 길어지면 끝날 때까지 기다리지 않고 이 피벗 수에서 중간 검사를 한다.
DRIFT_CHECK_EVERY = 1000
# 복원 직후에도 오차가 다시 나면 입력 자체가 수치적으로 나쁜 것이다.
# 무한 복원 대신 수치 실패(UNKNOWN)로 끝낸다.
MAX_RESTORES_PER_CALL = 3


def _scaled_tol(value: float, tol: float = DEFAULT_TOL) -> float:
    """값 크기에 비례한 허용오차 (Simplex/Reluplex 검증이 공유)."""
    return tol if not isfinite(value) else tol * max(1.0, abs(value))


# ─────────────────────────────────────────────
#  자료구조
# ─────────────────────────────────────────────

@dataclass
class Row:
    """
    기저변수 하나에 대응하는 Tableau 행(row).

    의미:
        basic_var = sum(coeffs[xi] * xi)   (비기저변수들의 선형결합)

    예시 (사진):
        s1 = x + y   →  Row(basic_var="s1", coeffs={"x": 1, "y": 1})
        s2 = -2x + y →  Row(basic_var="s2", coeffs={"x": -2, "y": 1})
    """
    basic_var: str               # 기저변수 이름
    coeffs: Dict[str, float]     # 비기저변수 -> 계수


@dataclass
class Bound:
    """변수의 하한(lower)과 상한(upper)"""
    lower: float = 0.0
    upper: float = float('inf')


@dataclass
class SimplexTableau:
    """
    Simplex Tableau 전체 상태.

    - rows   : 기저변수 row 방정식들
    - bounds : 모든 변수(기저+비기저)의 범위
    - assign : 현재 변수 할당값
    - origin_rows : 피벗하기 전의 원래 등식들 (basic_var = sum(coeff*var)).
                    피벗이 쌓이며 생긴 부동소수 오차를 재는 기준이자,
                    오차가 크면 tableau를 다시 세우는 재료다.
    - pivots_since_check : 마지막 오차 검사(또는 새로 만든 뒤) 이후 이 tableau
                    계보에 쌓인 피벗 수. 복사본(웜 스타트 자식)도 물려받는다.
    """
    rows: List[Row]
    bounds: Dict[str, Bound]
    assign: Dict[str, float]
    origin_rows: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)
    pivots_since_check: int = 0

    def __deepcopy__(self, memo: Dict[int, Any]) -> "SimplexTableau":
        # origin_rows는 한 번 만들어지면 바뀌지 않으므로 목록만 새로 만들고
        # 안의 (이름, 계수) 쌍은 부모와 공유한다. 분기·repair마다 tableau를
        # 복사하는데, 원래 등식까지 매번 깊게 복사할 이유가 없다.
        return SimplexTableau(
            rows=copy.deepcopy(self.rows, memo),
            bounds=copy.deepcopy(self.bounds, memo),
            assign=copy.deepcopy(self.assign, memo),
            origin_rows=list(self.origin_rows),
            pivots_since_check=self.pivots_since_check,
        )

    # 기저변수 집합 (빠른 조회)
    @property
    def basic_vars(self) -> List[str]:
        return [r.basic_var for r in self.rows]

    # 비기저변수 집합
    @property
    def nonbasic_vars(self) -> List[str]:
        basic = set(self.basic_vars)
        return [v for v in self.assign if v not in basic]


# ─────────────────────────────────────────────
#  Tableau 구성
# ─────────────────────────────────────────────

# 심플랙스폼을 정의 ( 등식 , 경계 )
def build_tableau(
    # row 방정식들의 list
    row_defs: List[Tuple[str, Dict[str, float]]],
    # 변수 범위 정보 (기저변수·비기저변수 모두 포함)
    bounds: Dict[str, Tuple[float, float]],
) -> SimplexTableau:
    """
    Tableau를 구성합니다.

    Args:
        row_defs : [(기저변수명, {비기저변수명: 계수, ...}), ...]
                   예) [("s1", {"x": 1, "y": 1}),
                        ("s2", {"x": -2, "y": 1})]

        bounds   : {변수명: (lower, upper), ...}
                   기저변수·비기저변수 모두 포함
                   예) {"s1": (0, inf), "s2": (2, inf),
                        "x":  (0, inf), "y": (0, inf)}

    Returns:
        SimplexTableau

    Raises:
        ValueError: 중복 basic variable, bound 누락, row 우변이 다른 basic
            variable을 직접 참조하는 비정규 tableau. 이런 입력은 피벗 규칙의
            전제를 깨서 조용히 잘못된 SAT/UNSAT을 만들 수 있으므로 여기서 막는다.
    """
    basic_names = [name for name, _ in row_defs]
    if len(set(basic_names)) != len(basic_names):
        raise ValueError("tableau의 basic variable 이름은 중복될 수 없습니다")

    missing_bounds = {name for name, _coeffs in row_defs if name not in bounds}
    missing_bounds.update(
        var
        for _name, coeffs in row_defs
        for var, coefficient in coeffs.items()
        if coefficient != 0.0 and var not in bounds
    )
    if missing_bounds:
        missing = ", ".join(sorted(missing_bounds))
        raise ValueError(f"tableau 변수의 bound가 없습니다: {missing}")

    basic_set_input = set(basic_names)
    basic_dependencies = {
        var
        for _name, coeffs in row_defs
        for var, coefficient in coeffs.items()
        if coefficient != 0.0 and var in basic_set_input
    }
    if basic_dependencies:
        variables = ", ".join(sorted(basic_dependencies))
        raise ValueError(
            "row 우변에는 nonbasic variable만 올 수 있습니다: " + variables
        )

    # row_defs를 이용해서 rows를 Row 객체들로 채우기.
    # 정확히 0인 계수는 수식에 영향이 없으므로 저장하지 않는다 (희소 신경망은
    # weight의 대부분이 exact zero라 이 항목들이 tableau를 통째로 부풀린다).
    rows = [Row(basic_var=name,
                coeffs={var: coefficient for var, coefficient in coeffs.items()
                        if coefficient != 0.0})
            for name, coeffs in row_defs]
    # 피벗은 rows의 계수 dict를 제자리에서 고치므로 원래 등식은 따로 복사해 둔다.
    origin_rows = [(row.basic_var, dict(row.coeffs)) for row in rows]

    bound_map: Dict[str, Bound] = {}
    for var, (lo, hi) in bounds.items():
        bound_map[var] = Bound(lower=lo, upper=hi)

    # 초기 할당: 비기저변수는 lower bound, 기저변수는 row로 계산
    all_vars = set(bound_map.keys())
    # basic_set : 기저변수 집합, nonbasic : 비기저변수 집합
    basic_set = {r.basic_var for r in rows}
    nonbasic = all_vars - basic_set

    assign: Dict[str, float] = {}

    # 비기저변수 초기화: lower bound
    for v in nonbasic:
        lo = bound_map[v].lower
        hi = bound_map[v].upper
        if lo == float('-inf') and hi == float('inf'):
            assign[v] = 0.0
        elif lo == float('-inf'):
            assign[v] = min(0.0, hi)
        else:
            assign[v] = lo

    # 기저변수 초기화: row 방정식으로 계산
    for row in rows:
        try:
            assign[row.basic_var] = sum(
                c * assign[nv] for nv, c in row.coeffs.items()
            )
        except (OverflowError, ValueError):
            # overflow나 inf-inf는 유효한 tableau 값이 아니다. NaN으로 남겨두면
            # SAT 직전 일관성 검사가 이를 UNKNOWN으로 승격한다.
            assign[row.basic_var] = float("nan")

    return SimplexTableau(
        rows=rows, bounds=bound_map, assign=assign, origin_rows=origin_rows
    )


# ─────────────────────────────────────────────
#  핵심 연산
# ─────────────────────────────────────────────

def _compute_basic(tableau: SimplexTableau, row: Row) -> float:
    """row 방정식으로 기저변수의 현재값을 계산"""
    try:
        return sum(c * tableau.assign[nv] for nv, c in row.coeffs.items())
    except (OverflowError, ValueError):
        # inf와 -inf의 상쇄나 중간 overflow는 유효한 tableau 값이 아니다.
        # NaN으로 표시하면 SAT 직전 일관성 검사가 이를 UNKNOWN으로 승격한다.
        return float("nan")


def _tableau_is_consistent(
    tableau: SimplexTableau,
    tol: float = DEFAULT_TOL,
    *,
    check_bounds: bool = True,
) -> bool:
    """SAT 반환 직전 tableau의 구조·방정식·(선택적으로) bound를 검증한다.

    피벗을 수백 번 거친 tableau는 나눗셈 오차가 누적되면서 row 식과 assignment가
    서로 어긋나거나 inf/nan이 섞일 수 있다. 그 상태의 assignment를 SAT 모델로
    내보내면 "솔버는 SAT이라는데 실제로는 제약을 만족하지 않는" 결과가 된다.
    여기서 걸리면 SAT이 아니라 수치 실패(UNKNOWN)로 보고한다.
    """
    basic_vars = tableau.basic_vars
    if len(set(basic_vars)) != len(basic_vars):
        return False

    basic_set = set(basic_vars)
    if set(tableau.assign) != set(tableau.bounds):
        return False

    for var, value in tableau.assign.items():
        if not isfinite(value):
            return False
        if check_bounds:
            bound = tableau.bounds[var]
            lower_tol = _scaled_tol(bound.lower, tol)
            upper_tol = _scaled_tol(bound.upper, tol)
            if value < bound.lower - lower_tol or value > bound.upper + upper_tol:
                return False

    for row in tableau.rows:
        if any(var in basic_set or var not in tableau.assign for var in row.coeffs):
            return False
        expected = _compute_basic(tableau, row)
        actual = tableau.assign[row.basic_var]
        scale = max(1.0, abs(expected), abs(actual))
        if not isfinite(expected) or abs(actual - expected) > tol * scale:
            return False

    return True


def _probe_value(var: str) -> float:
    """변수 이름으로 정해지는 [-1, 1) 사이의 고정된 값.

    Python의 hash()는 PYTHONHASHSEED에 따라 실행마다 달라지므로 쓰지 않는다.
    같은 tableau는 언제나 같은 오차 값을 내야 복원 여부가 재현된다.
    """
    return zlib.crc32(var.encode("utf-8")) / 2147483648.0 - 1.0


def _tableau_drift(tableau: SimplexTableau) -> float:
    """현재 tableau의 row 계수가 원래 등식에서 벗어난 정도 (상대 오차의 최댓값).

    `_tableau_is_consistent`는 현재 row와 현재 할당을 비교한다. 그런데 기저변수
    값은 매 피벗마다 현재 row로 다시 계산되므로, row 계수 자체가 틀어져도 그
    검사는 항상 통과한다. 현재 할당을 원래 등식에 넣어 보는 것만으로도 부족하다
    — 값이 0인 비기저변수의 계수 오차가 보이지 않는데, UNSAT 판정은 바로 그
    계수의 부호와 크기로 내려진다.

    그래서 비기저변수에 임의의 값을 넣고 현재 row로 기저변수를 계산한 뒤, 그
    값을 원래 등식에 넣은 잔차를 잰다. row가 정확하다면 비기저변수를 어떤 값으로
    두어도 원래 등식이 성립하므로, 이 잔차는 계수 오차만 반영한다. 현재 할당도
    같은 row로 계산되므로 row가 맞으면 할당도 원래 등식을 만족한다.
    """
    if not tableau.origin_rows:
        return 0.0

    probe: Dict[str, float] = {}
    basic = {row.basic_var for row in tableau.rows}
    for var in tableau.assign:
        if var not in basic:
            probe[var] = _probe_value(var)
    for row in tableau.rows:
        probe[row.basic_var] = sum(c * probe[nv] for nv, c in row.coeffs.items())

    worst = 0.0
    for name, coeffs in tableau.origin_rows:
        try:
            lhs = probe[name]
            terms = [c * probe[var] for var, c in coeffs.items()]
        except KeyError:
            return float("inf")
        rhs = sum(terms)
        if not (isfinite(lhs) and isfinite(rhs)):
            return float("inf")
        scale = max(1.0, abs(lhs), sum(abs(t) for t in terms))
        worst = max(worst, abs(lhs - rhs) / scale)
    return worst


def _restore_tableau(tableau: SimplexTableau) -> bool:
    """원래 등식으로 tableau를 다시 세운다 (논문의 tableau 복원).

    기저는 처음 상태로 돌아가지만, 비기저변수의 값은 지금 값을 그대로 쓴다
    (범위 밖이면 경계로 당긴다). 기저변수는 원래 등식으로 정확히 다시 계산되므로
    지금까지 풀어온 위치에서 오차만 걷어낸 상태로 simplex를 이어갈 수 있다.
    """
    if not tableau.origin_rows:
        return False
    try:
        fresh = build_tableau(
            tableau.origin_rows,
            {var: (b.lower, b.upper) for var, b in tableau.bounds.items()},
        )
    except ValueError:
        return False

    basic = {row.basic_var for row in fresh.rows}
    for var in fresh.assign:
        if var in basic:
            continue
        value = tableau.assign.get(var)
        if value is None or not isfinite(value):
            continue
        bound = fresh.bounds[var]
        fresh.assign[var] = min(max(value, bound.lower), bound.upper)
    for row in fresh.rows:
        fresh.assign[row.basic_var] = _compute_basic(fresh, row)

    tableau.rows = fresh.rows
    tableau.assign.clear()
    tableau.assign.update(fresh.assign)
    tableau.pivots_since_check = 0
    return True


def _tighten_from_row(
    tableau: SimplexTableau, row: Row, tol: float = DEFAULT_TOL
) -> Optional[bool]:
    """row의 우변(비기저변수들의 경계)으로 기저변수의 경계를 좁힌다.

    논문 5절: "각 피벗 뒤에 진입 변수에 대해 경계 강화를 수행했다". 방금 기저로
    들어온 변수의 row 하나만 보므로 row 길이만큼의 비용이다.

    반환: True=좁아짐, False=변화 없음, None=모순(이 row는 만족될 수 없다)
    """
    inf = float("inf")
    lo = hi = 0.0
    lo_inf = hi_inf = False
    for var, c in row.coeffs.items():
        b = tableau.bounds[var]
        t_lo, t_hi = (c * b.lower, c * b.upper) if c > 0 else (c * b.upper, c * b.lower)
        if t_lo == -inf:
            lo_inf = True
        else:
            lo += t_lo
        if t_hi == inf:
            hi_inf = True
        else:
            hi += t_hi
    if lo_inf and hi_inf:
        return False
    # 여러 항을 더하며 생긴 반올림으로 경계가 실제보다 좁아지지 않도록 바깥쪽으로 조금 넓힌다
    implied_lo = -inf if lo_inf else lo - 1e-12 * max(1.0, abs(lo))
    implied_hi = inf if hi_inf else hi + 1e-12 * max(1.0, abs(hi))

    cur = tableau.bounds[row.basic_var]
    new_lo = max(cur.lower, implied_lo)
    new_hi = min(cur.upper, implied_hi)
    if new_lo > new_hi + tol * max(1.0, abs(new_lo), abs(new_hi)):
        return None
    changed = (
        new_lo > cur.lower + _scaled_tol(cur.lower, tol)
        or new_hi < cur.upper - _scaled_tol(cur.upper, tol)
    )
    if changed:
        tableau.bounds[row.basic_var] = Bound(lower=new_lo, upper=new_hi)
    return changed


def _pivot(tableau: SimplexTableau, xi: str, xj: str) -> None:
    """
    피벗: 비기저변수 xi와 기저변수 xj를 교환합니다.

    xj의 row:  xj = ... + a*xi + ...
    →  xi = (xj - ...) / a   (xi가 새 기저변수)
    →  다른 모든 row에서 xi를 새 표현으로 치환

    Args:
        xi : 새로 기저로 들어올 비기저변수
        xj : 기저에서 나갈 기저변수
    """
    # xj의 row 찾기
    # next()는 generator에서 첫 번째 요소를 반환, 없으면 StopIteration 예외 발생
    pivot_row = next(r for r in tableau.rows if r.basic_var == xj)
    a = pivot_row.coeffs[xi]  # 피벗 계수 (0이 아님을 보장)

    # xj의 row:  xj = ... + a*xi + ...
    # ── Step 1: pivot_row를 xi = ... 형태로 변환 ──
    new_coeffs: Dict[str, float] = {}
    for var, c in pivot_row.coeffs.items():
        if var == xi:
            continue
        new_coeffs[var] = -c / a
    new_coeffs[xj] = 1.0 / a  # xj가 새 비기저변수로

    pivot_row.basic_var = xi
    pivot_row.coeffs = new_coeffs

    # ── Step 2: 다른 row들에서 xi를 새 표현으로 치환 ──
    for row in tableau.rows:
        # xi가 이 row에 없거나
        if row.basic_var == xi:
            continue
        # xi가 이 row에 없으면 치환할 필요 없음
        if xi not in row.coeffs:
            continue

        # xi의 계수
        factor = row.coeffs.pop(xi)
        for var, c in new_coeffs.items():
            # get(var, 0.0) : var이 row.coeffs에 없으면 0.0 반환
            # pivot_row와 같은 요소가 있으면 계수 업데이트, 없으면 새로 추가
            row.coeffs[var] = row.coeffs.get(var, 0.0) + factor * c

    # ── Step 3: 할당값 업데이트 ──
    # xi의 새 값은 xj가 경계로 이동한 값에서 결정
    # (update_assign에서 처리하므로 여기선 구조만 바꿈)


def _update_assign(tableau: SimplexTableau, xj: str, new_val: float) -> None:
    """
    비기저변수 xj의 값을 new_val로 변경하고,
    모든 기저변수를 row 방정식으로 재계산합니다.
    """
    tableau.assign[xj] = new_val
    for row in tableau.rows:
        tableau.assign[row.basic_var] = _compute_basic(tableau, row)


# ─────────────────────────────────────────────
#  Tableau 출력 (디버깅)
# ─────────────────────────────────────────────

def _print_tableau(tableau: SimplexTableau, iteration: int) -> None:
    """
    Tableau의 현재 상태를 출력합니다.
    
    Args:
        tableau : SimplexTableau 객체
        iteration : 반복 번호
    """
    print(f"\n[Iteration {iteration}]")
    print("─" * 70)
    
    # 기저변수와 비기저변수 표시
    basic_vars = set(tableau.basic_vars)
    all_vars = set(tableau.assign.keys())
    nonbasic_vars = all_vars - basic_vars
    
    print(f"기저변수: {sorted(basic_vars)}")
    print(f"비기저변수: {sorted(nonbasic_vars)}")
    print()
    
    # 각 row 출력
    for row in tableau.rows:
        xj = row.basic_var
        val = tableau.assign[xj]
        bounds = tableau.bounds[xj]
        
        # row 식 표현
        terms = []
        for var in sorted(row.coeffs.keys()):
            coeff = row.coeffs[var]
            if coeff > 0:
                terms.append(f"+ {coeff:.5f}*{var}")
            else:
                terms.append(f"- {abs(coeff):.5f}*{var}")
        
        expr = " ".join(terms) if terms else "0"
        if expr.startswith("+ "):
            expr = expr[2:]
        
        bounds_str = f"[{bounds.lower:.3f}, {bounds.upper:.3f}]" if bounds.upper != float('inf') else f"[{bounds.lower:.6f}, ∞)"
        in_bounds = "✓" if (bounds.lower <= val + 1e-9 and val <= bounds.upper + 1e-9) else "✗"
        
        print(f"{xj:4s} = {expr:30s}  | 값: {val:8.3f} | 범위: {bounds_str:20s} {in_bounds}")
    
    print()
    print("변수값:")
    for var in sorted(tableau.assign.keys()):
        val = tableau.assign[var]
        bounds = tableau.bounds[var]
        bounds_str = f"[{bounds.lower:.3f}, {bounds.upper:.3f}]" if bounds.upper != float('inf') else f"[{bounds.lower:.6f}, ∞)"
        in_bounds = "✓" if (bounds.lower <= val + 1e-9 and val <= bounds.upper + 1e-9) else "✗"
        print(f"  {var:4s} = {val:8.3f} (범위: {bounds_str:20s}) {in_bounds}")
    
    print("─" * 70)


# ─────────────────────────────────────────────
#  Simplex 메인 알고리즘
#  (Dutertre & de Moura, "A Fast Linear-Arithmetic Solver for DPLL(T)")
# ─────────────────────────────────────────────

def simplex(
    tableau: SimplexTableau,
    max_iter: int = 10000,
    debug: bool = False,
    *,
    deadline: Optional[float] = None,
    report_unknown: bool = False,
    progress: Optional[Any] = None,
    progress_context: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
    tighten_bounds: bool = False,
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Simplex 알고리즘 (Algorithm 3 스타일).

    루프 불변식:
        비기저변수는 항상 [lower, upper] 범위 안에 있음.
        기저변수만 범위를 위반할 수 있음.

    알고리즘:
        1. 범위를 위반한 기저변수 xj를 찾는다 (이름 사전순 최소).
        2. xj의 row에서 피벗 가능한 비기저변수 xi를 찾는다.
            - xj < lj: xi를 올릴 수 있는 변수 (a_ij > 0, xi < u_i)
                      또는 xi를 내릴 수 있는 변수 (a_ij < 0, xi > l_i)
            - xj > uj: 반대 조건
        3. xi를 찾으면 피벗, 못 찾으면 UNSAT.

    수치 안정성 (중요)
    ------------------
    2번에서 후보가 여럿이면 |a_ij|가 가장 큰 것을 고른다(threshold pivoting).
    Bland's rule로 이름 사전순 최초를 고르면 |a|~1e-10짜리 열도 그대로 선택되고,
    _pivot()의 -c/a 에서 오차가 1e10배 증폭된다. 실제로 정수 가중치 모델에서
    몇백 번 피벗한 뒤 row가 통째로 잡음이 되어, 있지도 않은 UNSAT을 선언하고
    Reluplex가 분기 한 번 없이 종료한 사례가 있었다.

    Bland's rule은 BLAND_AFTER 반복 이후 순환 방지 폴백으로만 쓴다. DdM 불변식은
    진입변수 선택과 무관하게 유지되므로, 도중에 Bland로 전환해도 종료성은 보장된다.

    경계 위반 판정과 여유 판정은 절대 EPS가 아니라 값의 크기에 비례한 상대
    허용오차(_tol)를 쓰고, 계수 0 판정도 row 최대계수 대비 상대값(ZERO_REL)을 쓴다.

    누적 오차 복원
    --------------
    tableau 계보에 검사 없이 DRIFT_CHECK_AT_EXIT 피벗 이상 쌓였으면 SAT/UNSAT을
    선언하기 직전에, 한 호출이 길어지면 DRIFT_CHECK_EVERY 피벗에서 원래 등식 대비
    오차(_tableau_drift)를 잰다. DRIFT_TOL을 넘으면 원래 등식으로 tableau를 다시
    세우고(_restore_tableau) 이어서 푼다. 특히 UNSAT은 "피벗할 수 있는 변수가
    없다"는 계수의 부호로 판정하므로, 틀어진 row로 UNSAT을 선언하면 반례가 있는
    영역을 잘라내는 잘못된 증명이 된다.

    피벗마다 경계 강화 (tighten_bounds=True)
    ----------------------------------------
    피벗 직후 새로 기저가 된 변수의 경계를 그 row로 좁힌다(_tighten_from_row).
    좁아진 경계는 tableau.bounds에 남아 호출자가 읽어 갈 수 있다. 범위가
    모순이면 더 피벗할 것 없이 UNSAT이다. 복원할 때는 틀어진 row로 좁혔을 수
    있으므로 이 호출에서 좁힌 경계를 모두 되돌린다.

    Args:
        tableau : SimplexTableau 객체
        max_iter : 최대 반복 횟수 (기본값: 10000)
        debug : True일 때 각 반복 단계마다 tableau를 출력 (기본값: False)

    Returns:
        (assignment, True)  — SAT
        (None, False)       — UNSAT
    """
    EPS = 1e-9
    # row 안에서 "사실상 0"으로 볼 계수의 상대 임계값. 절대값으로 자르면
    # (예전 구현) 정상적인 1e-10짜리 계수가 통째로 버려져서 피벗 후보가
    # 사라지고 없는 UNSAT이 만들어진다.
    ZERO_REL = 1e-12
    # 이 반복 횟수를 넘어가면 진입변수 선택을 순수 Bland's rule로 영구 전환한다.
    # DdM 불변식(비기저변수는 항상 범위 안)은 진입변수를 어떻게 고르든 매 피벗마다
    # 유지되므로, 어느 시점에 Bland로 갈아타도 그때부터는 유효한 상태에서 시작하는
    # 정상 Bland 실행이라 종료가 보장된다.
    BLAND_AFTER = 2 * len(tableau.rows) + 1000

    context = dict(progress_context or {})
    call_id = None
    iteration_count = 0
    pivot_count = 0
    if progress is not None:
        call_id = progress.simplex_start(
            round_index=context.get("round_index"),
            depth=context.get("depth"),
            origin=str(context.get("origin", "simplex")),
            theory_atom_ids=context.get("theory_atom_ids", []),
            row_count=len(tableau.rows),
            variable_count=len(tableau.assign),
        )

    def _finish(result: str) -> None:
        if progress is not None and call_id is not None:
            progress.simplex_end(
                call_id,
                result,
                iterations=iteration_count,
                pivots=pivot_count,
            )

    def _tol(x: float) -> float:
        """절대 EPS 대신 값의 크기에 비례한 허용오차.

        값이 16 정도만 돼도 상대오차 4e-10짜리 부동소수 오차가 절대 1e-9를
        넘어서 '경계 위반'으로 오인된다 — 실제로 이것 때문에 멀쩡한 tableau가
        UNSAT으로 판정된 사례가 있었다.
        """
        if x in (float('inf'), float('-inf')):
            return EPS
        return EPS * max(1.0, abs(x))

    restores = 0
    # 복원하면 이 호출에서 틀어진 row로 좁혔을 수도 있는 경계를 되돌린다
    bounds_at_start = (
        {var: Bound(lower=b.lower, upper=b.upper) for var, b in tableau.bounds.items()}
        if tighten_bounds else None
    )

    def _drift_needs_restore() -> Optional[bool]:
        """오차를 재고 필요하면 복원한다.

        반환: False=오차 정상, True=복원했음(루프를 이어가야 함),
              None=복원할 수 없음(수치 실패로 끝내야 함)
        """
        nonlocal restores
        drift = _tableau_drift(tableau)
        if stats is not None:
            stats["SIMPLEX_DRIFT_CHECKS"] = stats.get("SIMPLEX_DRIFT_CHECKS", 0) + 1
            if drift > stats.get("SIMPLEX_MAX_DRIFT", 0.0):
                stats["SIMPLEX_MAX_DRIFT"] = drift
        if drift <= DRIFT_TOL:
            tableau.pivots_since_check = 0
            return False
        if bounds_at_start is not None:
            tableau.bounds.clear()
            tableau.bounds.update(
                (var, Bound(lower=b.lower, upper=b.upper))
                for var, b in bounds_at_start.items()
            )
        if restores >= MAX_RESTORES_PER_CALL or not _restore_tableau(tableau):
            return None
        restores += 1
        if stats is not None:
            stats["SIMPLEX_RESTORES"] = stats.get("SIMPLEX_RESTORES", 0) + 1
        return True

    def _numerical_failure() -> Tuple[Optional[Dict[str, float]], bool]:
        _finish("NUMERICAL_FAILURE")
        if report_unknown:
            raise SolverLimitReached("SIMPLEX_NUMERICAL_FAILURE")
        return (None, False)

    for iteration in range(max_iter):
        iteration_count = iteration + 1
        try:
            check_deadline(deadline)
        except SolverLimitReached:
            _finish("TIMEOUT")
            raise
        if debug:
            _print_tableau(tableau, iteration)

        use_bland = iteration >= BLAND_AFTER

        # ── 범위 위반 기저변수 찾기 (Bland's rule: 이름 사전순 최소인 변수) ──
        # 이탈변수는 항상 이 전순서로 고른다. 진입변수도 Bland로 고르는 폴백
        # 구간(use_bland)에서는 두 선택이 같은 전순서를 쓰게 되어 anti-cycling
        # 보장이 성립한다. tableau.rows의 list 순서는 피벗마다 바뀌므로
        # "첫 번째로 찾은 row"를 쓰면 그 보장이 깨져서 특히 퇴화(degenerate)된
        # tableau에서 진짜로 순환(cycle)할 수 있다.
        violated_row = None
        for row in tableau.rows:
            xj = row.basic_var
            val = tableau.assign[xj]
            b = tableau.bounds[xj]

            if val < b.lower - _tol(b.lower) or val > b.upper + _tol(b.upper):
                if violated_row is None or xj < violated_row.basic_var:
                    violated_row = row

        if violated_row is None:
            # 모든 기저변수가 범위 안 → SAT.
            # 검사 없이 쌓인 피벗이 많으면 원래 등식 대비 오차부터 본다. 틀어진
            # row로 구한 할당은 원래 문제의 해가 아닐 수 있고, 자식 노드가 이
            # row로 경계 유도를 한다. 복원하면 기저가 바뀌므로 위반 검사부터
            # 다시 한다.
            if tableau.pivots_since_check >= DRIFT_CHECK_AT_EXIT:
                restored = _drift_needs_restore()
                if restored is None:
                    return _numerical_failure()
                if restored:
                    continue
            # 여기까지의 피벗으로 tableau 자체가 깨졌다면(row 식과 assignment
            # 불일치, inf/nan) SAT이라고 내보내면 안 된다 — 수치 실패로 구분한다.
            if not _tableau_is_consistent(tableau):
                return _numerical_failure()
            if progress is not None and call_id is not None:
                progress.simplex_iteration(call_id, iteration, violated_var=None)
            _finish("SAT")
            return (dict(tableau.assign), True)

        # 위반한 기저 변수 xj와 피벗할 비기저변수 xi 탐색
        xj = violated_row.basic_var
        val = tableau.assign[xj]
        b_xj = tableau.bounds[xj]
        going_up = val < b_xj.lower  # True: xj를 올려야 함  False: upper보다 크다는 뜻 → xj를 내려야 함
        if progress is not None and call_id is not None:
            progress.simplex_iteration(
                call_id,
                iteration,
                violated_var=xj,
                value=val,
                lower=b_xj.lower,
                upper=b_xj.upper,
            )

        amax = max((abs(c) for c in violated_row.coeffs.values()), default=0.0)
        zero_thr = max(ZERO_REL * amax, 1e-300)

        def _eligible(xi: str, a: float) -> bool:
            """xi를 진입변수로 썼을 때 xj를 원하는 방향으로 움직일 수 있는가."""
            if abs(a) <= zero_thr:
                return False
            b_xi = tableau.bounds[xi]
            xi_val = tableau.assign[xi]
            if (a > 0) == going_up:
                # xi를 올려야 한다 → upper 쪽에 여유가 있어야 함
                return xi_val < b_xi.upper - _tol(b_xi.upper)
            # xi를 내려야 한다 → lower 쪽에 여유가 있어야 함
            return xi_val > b_xi.lower + _tol(b_xi.lower)

        # ── 피벗 가능한 비기저변수 xi 탐색 ──
        # 기본은 |a| 최대(threshold pivoting). Bland's rule은 |a|~1e-10짜리 열도
        # 태연히 고르는데, _pivot()의 -c/a 에서 오차가 1e10배 증폭되어 몇백 번만
        # 피벗해도 정수 tableau가 잡음 덩어리가 된다.
        pivot_xi = None
        if use_bland:
            for xi in sorted(violated_row.coeffs.keys()):  # Bland's rule
                if _eligible(xi, violated_row.coeffs[xi]):
                    pivot_xi = xi
                    break
        else:
            best = None
            for xi, a in violated_row.coeffs.items():
                if not _eligible(xi, a):
                    continue
                m = abs(a)
                # 동률은 이름 사전순으로 → 실행이 결정론적으로 유지된다
                if best is None or m > best[0] or (m == best[0] and xi < best[1]):
                    best = (m, xi)
            if best is not None:
                pivot_xi = best[1]

        if pivot_xi is None:
            # 피벗 가능한 변수 없음 → UNSAT.
            # 이 판정은 violated_row 계수의 부호에 전적으로 기댄다. 계수가
            # 오차로 틀어졌다면 반례가 있는 영역을 잘라내는 잘못된 UNSAT이
            # 되므로, 검사 없이 쌓인 피벗이 많으면 선언하기 전에 원래 등식
            # 대비 오차를 확인한다.
            if tableau.pivots_since_check >= DRIFT_CHECK_AT_EXIT:
                restored = _drift_needs_restore()
                if restored is None:
                    return _numerical_failure()
                if restored:
                    continue
            _finish("UNSAT")
            return (None, False)

        # ── 피벗 수행 ──
        if progress is not None and call_id is not None:
            progress.simplex_pivot(
                call_id,
                iteration,
                entering=pivot_xi,
                leaving=xj,
                row=xj,
            )
        pivot_count += 1
        # 먼저 xj를 경계로 이동시키는 delta 계산
        a = violated_row.coeffs[pivot_xi]
        target = b_xj.lower if going_up else b_xj.upper
        delta = (target - val) / a  # xj가 target에 도달하도록 xi 변화량

        # xi를 delta만큼 이동 (비기저→기저 교환 전 assign 업데이트)
        _update_assign(tableau, pivot_xi, tableau.assign[pivot_xi] + delta)

        # 구조적 피벗 (row 재작성)
        _pivot(tableau, pivot_xi, xj)

        # fill-in으로 생긴 잡음 계수를 정확히 0으로 지워 누적을 끊는다.
        # (이걸 안 하면 원래 0이어야 할 자리가 1e-10 언저리로 떠다니면서
        #  row가 실제로는 걸지 않는 제약을 거는 것처럼 보이게 된다)
        for row in tableau.rows:
            if not row.coeffs:
                continue
            m = max(abs(c) for c in row.coeffs.values())
            if m == 0.0:
                continue
            thr = ZERO_REL * m
            for v in [v for v, c in row.coeffs.items() if abs(c) <= thr]:
                del row.coeffs[v]

        # 피벗 후 새 비기저변수 xj는 경계값으로 고정
        tableau.assign[xj] = target

        # 기저변수들 재계산
        for row in tableau.rows:
            tableau.assign[row.basic_var] = _compute_basic(tableau, row)

        tableau.pivots_since_check += 1

        # 방금 기저로 들어온 변수(pivot_xi)의 row로 그 경계를 좁힌다.
        # violated_row는 _pivot()이 제자리에서 "pivot_xi = ..." row로 바꿔 두었다.
        if tighten_bounds:
            tightened = _tighten_from_row(tableau, violated_row)
            if tightened is None:
                # 비기저변수를 경계 안에서 어떻게 움직여도 이 row를 만족시킬 수
                # 없다 → UNSAT. 피벗 불가 UNSAT과 똑같이 row 계수에 기대는
                # 판정이므로 같은 오차 검사를 거친다.
                if tableau.pivots_since_check >= DRIFT_CHECK_AT_EXIT:
                    restored = _drift_needs_restore()
                    if restored is None:
                        return _numerical_failure()
                    if restored:
                        continue
                if stats is not None:
                    stats["PIVOT_TIGHTEN_UNSAT"] = stats.get("PIVOT_TIGHTEN_UNSAT", 0) + 1
                _finish("UNSAT")
                return (None, False)
            if tightened and stats is not None:
                stats["PIVOT_TIGHTENED"] = stats.get("PIVOT_TIGHTENED", 0) + 1

        # 긴 호출은 끝날 때까지 기다리지 않고 중간에도 오차를 본다. 틀어진
        # row로 피벗을 계속하면 엉뚱한 방향으로 헤매며 반복만 늘어난다.
        if tableau.pivots_since_check >= DRIFT_CHECK_EVERY:
            if _drift_needs_restore() is None:
                return _numerical_failure()

    # 반복 제한 초과는 논리적 UNSAT이 아니라 결론을 내리지 못한 UNKNOWN이다.
    _finish("ITERATION_LIMIT")
    if report_unknown:
        raise SolverLimitReached("SIMPLEX_ITERATION_LIMIT")
    return (None, False)


# ─────────────────────────────────────────────
#  테스트
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  사진 예시")
    print("  s1 = x + y,   s1 >= 0")
    print("  s2 = -2x + y, s2 >= 2")
    print("  s3 = -10x + y, s3 >= -5"   )
    print("=" * 55)

    row_defs = [
        ("s1", {"x": 1.0,   "y": 1.0}),
        ("s2", {"x": -2.0,  "y": 1.0}),
        ("s3", {"x": -10.0, "y": 1.0}),
    ]
    bounds = {
        "s1": (0.0,        float('inf')),
        "s2": (2.0,        float('inf')),
        "s3": (-5.0,       float('inf')),
        "x":  (-float('inf'),        float('inf')),
        "y":  (-float('inf'),        float('inf')),
    }

    tableau = build_tableau(row_defs, bounds)
    result, sat = simplex(tableau, debug=True)

    if sat:
        print(f"SAT: {result}")
        # 검증
        EPS = 1e-9
        x, y = result['x'], result['y']
        s1 = x + y
        s2 = -2*x + y
        s3 = -10*x + y
        print(f"\n검증:")
        print(f"  s1 = {x:.4f} + {y:.4f} = {s1:.4f} >= 0  → {'✅' if s1 >= -EPS else '❌'}")
        print(f"  s2 = -2*{x:.4f} + {y:.4f} = {s2:.4f} >= 2  → {'✅' if s2 >= 2 - EPS else '❌'}")
        print(f"  s3 = -10*{x:.4f} + {y:.4f} = {s3:.4f} >= -5 → {'✅' if s3 >= -5 - EPS else '❌'}")
    else:
        print("UNSAT")

    # ─── 추가 테스트 ───

    print("\n" + "=" * 55)
    print("  테스트 2: UNSAT 케이스")
    print("  s1 = x,  s1 >= 5")
    print("  s2 = -x, s2 >= -3   (즉 x <= 3)")
    print("  → x >= 5 AND x <= 3 : 불가능")
    print("=" * 55)

    row_defs2 = [
        ("s1", {"x": 1.0}),
        ("s2", {"x": -1.0}),
    ]
    bounds2 = {
        "s1": (5.0, float('inf')),
        "s2": (-3.0, float('inf')),
        "x":  (0.0, float('inf')),
    }
    tableau2 = build_tableau(row_defs2, bounds2)
    result2, sat2 = simplex(tableau2)
    print(f"결과: {'SAT: ' + str(result2) if sat2 else 'UNSAT ✅'}")

    print("\n" + "=" * 55)
    print("  테스트 3: 다변수 연립")
    print("  s1 = x + y,   s1 >= 10")
    print("  s2 = x - y,   s2 >= 0  (x >= y)")
    print("  s3 = -x + 2y, s3 >= 3")
    print("=" * 55)

    row_defs3 = [
        ("s1", {"x": 1.0, "y": 1.0}),
        ("s2", {"x": 1.0, "y": -1.0}),
        ("s3", {"x": -1.0, "y": 2.0}),
    ]
    bounds3 = {
        "s1": (10.0, float('inf')),
        "s2": (0.0,  float('inf')),
        "s3": (3.0,  float('inf')),
        "x":  (0.0,  float('inf')),
        "y":  (0.0,  float('inf')),
    }
    tableau3 = build_tableau(row_defs3, bounds3)
    result3, sat3 = simplex(tableau3)
    if sat3:
        print(f"SAT: {result3}")
        x, y = result3['x'], result3['y']
        EPS = 1e-9
        print(f"  s1={x+y:.3f} >= 10 {'✅' if x+y >= 10-EPS else '❌'}")
        print(f"  s2={x-y:.3f} >= 0  {'✅' if x-y >= 0-EPS else '❌'}")
        print(f"  s3={-x+2*y:.3f} >= 3  {'✅' if -x+2*y >= 3-EPS else '❌'}")
    else:
        print("UNSAT")

    row_defs4 = [
        ("s1", {"x": 1.0, "y": 1.0})
    ]
    bounds4 = {
        "s1": (5.0, float('inf')),
        "x": (-float('inf'), float('inf')),
        "y": (-float('inf'), float('inf')),
    }
    tableau4 = build_tableau(row_defs4, bounds4)
    result4, sat4 = simplex(tableau4)
    print(f"테스트 4 결과: {'SAT: ' + str(result4) if sat4 else 'UNSAT'}")
if __name__ == "__main__":
    main()
