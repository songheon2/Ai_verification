from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    """
    rows: List[Row]
    bounds: Dict[str, Bound]
    assign: Dict[str, float]

    # 기저변수 집합 (빠른 조회)
    @property
    def basic_vars(self) -> List[str]:
        return [r.basic_var for r in self.rows]

    # 비기저변수 집합
    @property
    def nonbasic_vars(self) -> List[str]:
        basic = set(self.basic_vars)
        return [v for v in self.assign if v not in basic]

    def print(self) -> None:
        def fmt_num(x: float) -> str:
            if x == float('inf'):
                return "inf"
            if x == float('-inf'):
                return "-inf"
            return f"{x:.6g}"

        print("[Rows]")
        for row in self.rows:
            if row.coeffs:
                expr = " + ".join(f"{c:.6g}*{v}" for v, c in row.coeffs.items())
            else:
                expr = "0"
            print(f"  {row.basic_var} = {expr}")

        print("[Bounds]")
        for var in sorted(self.bounds.keys()):
            b = self.bounds[var]
            print(f"  {var}: [{fmt_num(b.lower)}, {fmt_num(b.upper)}]")

        print("[Assignment]")
        for var in sorted(self.assign.keys()):
            print(f"  {var} = {fmt_num(self.assign[var])}")

        print()


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
    """
    # row_defs를 이용해서 rows를 Row 객체들로 채우기
    rows = [Row(basic_var=name, coeffs=dict(coeffs))
            for name, coeffs in row_defs]

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
        assign[row.basic_var] = sum(
            c * assign[nv] for nv, c in row.coeffs.items()
        )

    return SimplexTableau(rows=rows, bounds=bound_map, assign=assign)


# ─────────────────────────────────────────────
#  핵심 연산
# ─────────────────────────────────────────────

def _compute_basic(tableau: SimplexTableau, row: Row) -> float:
    """row 방정식으로 기저변수의 현재값을 계산"""
    return sum(c * tableau.assign[nv] for nv, c in row.coeffs.items())


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
#  Simplex 메인 알고리즘
#  (Dutertre & de Moura, "A Fast Linear-Arithmetic Solver for DPLL(T)")
# ─────────────────────────────────────────────

def simplex(tableau: SimplexTableau, max_iter: int = 10000, debug: bool = False) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Simplex 알고리즘 (Algorithm 3 스타일).

    루프 불변식:
        비기저변수는 항상 [lower, upper] 범위 안에 있음.
        기저변수만 범위를 위반할 수 있음.

    알고리즘:
        1. 범위를 위반한 기저변수 xj를 찾는다.
        2. xj의 row에서 피벗 가능한 비기저변수 xi를 찾는다.
            - xj < lj: xi를 올릴 수 있는 변수 (a_ij > 0, xi < u_i)
                      또는 xi를 내릴 수 있는 변수 (a_ij < 0, xi > l_i)
            - xj > uj: 반대 조건
        3. xi를 찾으면 피벗, 못 찾으면 UNSAT.

    Returns:
        (assignment, True)  — SAT
        (None, False)       — UNSAT
    """
    EPS = 1e-9

    for iteration in range(max_iter):
        if debug:
            tableau.print()

        # ── 범위 위반 기저변수 찾기 ──
        violated_row = None
        for row in tableau.rows:
            xj = row.basic_var
            val = tableau.assign[xj]
            b = tableau.bounds[xj]

            if val < b.lower - EPS or val > b.upper + EPS:
                violated_row = row
                break

        if violated_row is None:
            # 모든 기저변수가 범위 안 → SAT
            return (dict(tableau.assign), True)

        # 위반한 기저 변수 xj와 피벗할 비기저변수 xi 탐색
        xj = violated_row.basic_var
        val = tableau.assign[xj]
        b_xj = tableau.bounds[xj]
        going_up = val < b_xj.lower  # True: xj를 올려야 함  False: upper보다 크다는 뜻 → xj를 내려야 함

        # ── 피벗 가능한 비기저변수 xi 탐색 (Bland's rule: 인덱스 최소) ──
        pivot_xi = None

        for xi in sorted(violated_row.coeffs.keys()):  # Bland's rule
            a = violated_row.coeffs[xi]
            b_xi = tableau.bounds[xi]
            xi_val = tableau.assign[xi]

            if going_up:
                # xj < lj → xj를 올려야 함 → LHS를 증가시킬 xi
                # xj를 올리려면 a * xi가 커져야한다
                # a > 0 -> xi를 올려야해서 upper보다 작은지, 
                if a > EPS and xi_val < b_xi.upper - EPS:
                    pivot_xi = xi; break
                # a < 0 -> xi를 내려야해서 lower보다 큰지 확인
                if a < -EPS and xi_val > b_xi.lower + EPS:
                    pivot_xi = xi; break
            else:
                # xj > uj → xj를 내려야 함
                if a < -EPS and xi_val < b_xi.upper - EPS:
                    pivot_xi = xi; break
                if a > EPS and xi_val > b_xi.lower + EPS:
                    pivot_xi = xi; break

        if pivot_xi is None:
            # 피벗 가능한 변수 없음 → UNSAT
            return (None, False)

        if debug:
            print(f"Violated {xj} = {val:.6g}")
            print(f"(bounds [{b_xj.lower:.6g}, {b_xj.upper:.6g}])")
            print(f"pivoting with {pivot_xi}")
            print()

        # ── 피벗 수행 ──
        # 먼저 xj를 경계로 이동시키는 delta 계산
        a = violated_row.coeffs[pivot_xi]
        target = b_xj.lower if going_up else b_xj.upper
        delta = (target - val) / a  # xj가 target에 도달하도록 xi 변화량

        # xi를 delta만큼 이동 (비기저→기저 교환 전 assign 업데이트)
        _update_assign(tableau, pivot_xi, tableau.assign[pivot_xi] + delta)

        # 구조적 피벗 (row 재작성)
        _pivot(tableau, pivot_xi, xj)

        # 피벗 후 새 비기저변수 xj는 경계값으로 고정
        tableau.assign[xj] = target

        # 기저변수들 재계산
        for row in tableau.rows:
            tableau.assign[row.basic_var] = _compute_basic(tableau, row)

    # 반복 제한 초과
    return (None, False)


# ─────────────────────────────────────────────
#  테스트
# ─────────────────────────────────────────────

def test1(debug:bool=False) -> None:
    print("=" * 55)
    print("  테스트 1: 사진 예시")
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
    result, sat = simplex(tableau, debug=debug)

    print(f"결과: {'SAT: ' + str(result) if sat else 'UNSAT'}")


def test2(debug:bool=False) -> None:
    print("=" * 55)
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
    result2, sat2 = simplex(tableau2, debug=debug)
    print(f"결과: {'SAT: ' + str(result2) if sat2 else 'UNSAT'}")

def test3(debug:bool=False) -> None:
    print("=" * 55)
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
    result3, sat3 = simplex(tableau3, debug=debug)
    print(f"결과: {'SAT: ' + str(result3) if sat3 else 'UNSAT'}")

def test4(debug:bool=False) -> None:
    print("=" * 55)
    print("  테스트 4: ")
    print("  s1 = x + y,   s1 >= 5")
    print("=" * 55)

    row_defs4 = [
        ("s1", {"x": 1.0, "y": 1.0})
    ]
    bounds4 = {
        "s1": (5.0, float('inf')),
        "x": (-float('inf'), float('inf')),
        "y": (-float('inf'), float('inf')),
    }
    tableau4 = build_tableau(row_defs4, bounds4)
    result4, sat4 = simplex(tableau4, debug=debug)
    print(f"결과: {'SAT: ' + str(result4) if sat4 else 'UNSAT'}")

def main() -> None:
    debug = True
    test1(debug)
    print()

    test2(debug)
    print()

    test3(debug)
    print()

    test4(debug)
    print()

if __name__ == "__main__":
    main()