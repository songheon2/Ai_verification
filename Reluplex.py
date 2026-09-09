"""Reluplex 알고리즘 구현 (내부적으로 `Simplex.build_tableau`와 `simplex` 사용).

이 모듈은 `reluplex(row_defs, bounds, relus)`를 제공하며,
사진의 Algorithm 4(간단화된 Reluplex)의 재귀적 구현을 따릅니다.
"""
from typing import Any, Dict, List, Tuple, Optional
from Simplex import (
    Bound,
    Row,
    SimplexTableau,
    _compute_basic,
    _pivot,
    build_tableau,
    simplex,
)
from Automation.SolverStatus import SolverLimitReached, check_deadline
from visualization.SolveTrace import SolveTrace, span
import copy
import random
import re

_NEURON_VAR_RE = re.compile(r"^[zh](\d+)_(\d+)(?:_.*)?$")


def _parse_neuron_var(name: str) -> Optional[Tuple[int, int]]:
    match = _NEURON_VAR_RE.match(name)
    return (int(match.group(1)), int(match.group(2))) if match is not None else None

def _express_in_nonbasic(
    tableau: SimplexTableau, terms: Dict[str, float]
) -> Dict[str, float]:
    """terms(변수->계수)를 현재 비기저변수만으로 다시 표현한다.

    tableau의 row 우변은 정의상 전부 비기저변수이므로, terms에 등장하는
    기저변수를 자기 row로 한 번만 치환하면 된다.
    """

    basic_rows = {row.basic_var: row for row in tableau.rows}
    out: Dict[str, float] = {}
    for var, coeff in terms.items():
        coeff = float(coeff)
        row = basic_rows.get(var)
        if row is None:
            out[var] = out.get(var, 0.0) + coeff
        else:
            for nv, cc in row.coeffs.items():
                out[nv] = out.get(nv, 0.0) + coeff * cc
    return {v: c for v, c in out.items() if c != 0.0}


def _sync_bounds(
    tableau: SimplexTableau, bounds_now: Dict[str, Tuple[float, float]]
) -> None:
    """tableau의 bound를 bounds_now에 맞추고 DdM 불변식을 복구한다.

    불변식은 "비기저변수는 항상 범위 안"이다. 범위가 좁아져서 비기저변수가
    밖으로 나가면 새 경계로 당겨온다(DdM의 assertBound와 같은 처리).
    기저변수는 건드리지 않는다 — 범위를 벗어나도 되고, 그건 simplex가
    피벗으로 해결할 일이다. 마지막에 모든 기저변수를 row 식으로 재계산한다.
    """

    basic = {row.basic_var for row in tableau.rows}
    for var, (lo, hi) in bounds_now.items():
        if var not in tableau.assign:
            continue
        tableau.bounds[var] = Bound(lower=lo, upper=hi)
        if var in basic:
            continue
        value = tableau.assign[var]
        if value < lo:
            tableau.assign[var] = lo
        elif value > hi:
            tableau.assign[var] = hi
    for row in tableau.rows:
        tableau.assign[row.basic_var] = _compute_basic(tableau, row)


def _child_tableau(
    parent: SimplexTableau,
    new_rows: List[Tuple[str, Dict[str, float], Tuple[float, float]]],
) -> SimplexTableau:
    """부모의 (이미 푼) tableau를 복사하고 새 제약 row를 기저변수로 얹는다.

    분기할 때마다 build_tableau로 처음부터 다시 만들면 부모가 이미 한 피벗을
    전부 다시 하게 되어 비용이 depth에 비례해 커진다(전체 O(depth^2)).
    자식은 부모와 row 하나 + bound 몇 개만 다르므로, 부모의 basis를 그대로
    물려받고 달라진 부분만 반영한 뒤 이어서 푼다.

    새 row의 우변은 기저변수를 포함할 수 있으므로 _express_in_nonbasic으로
    비기저변수 표현으로 바꿔서 넣는다.
    """

    child = copy.deepcopy(parent)
    for name, terms, (lo, hi) in new_rows:
        row = Row(basic_var=name, coeffs=_express_in_nonbasic(child, terms))
        child.rows.append(row)
        child.bounds[name] = Bound(lower=lo, upper=hi)
        child.assign[name] = _compute_basic(child, row)
    return child


def relu(v: float) -> float:
    """ReLU 함수: 음수일 경우 0, 양수일 경우 자기 자신을 반환."""
    return v if v > 0 else 0.0


def _check_relu_violations(assign: Dict[str, float], relus: List[Tuple[str, str]], tol: float = 1e-9):
    """현재 할당 `assign`에서 ReLU 제약 `relus`가 위반된 (x,y) 쌍들의 리스트를 반환."""
    viol = []
    for x, y in relus:
        if x not in assign or y not in assign:
            viol.append((x, y))
            continue
        if abs(assign[y] - relu(assign[x])) > tol:
            viol.append((x, y))
    return viol


def reluplex(
    row_defs: List[Tuple[str, Dict[str, float]]],
    bounds: Dict[str, Tuple[float, float]],
    relus: List[Tuple[str, str]],
    max_recursion: int = 50,
    simplex_max_iter: int = 10000,
    local_repair_max_iter: int = 10,
    branch_tau: int = 5,
    debug: bool = False,
    *,
    deadline: Optional[float] = None,
    report_unknown: bool = False,
    trace: Optional[SolveTrace] = None,
    split_logger: Optional[Any] = None,
    relu_metadata: Optional[Dict[Tuple[str, str], Tuple[Optional[int], Optional[int]]]] = None,
    progress: Optional[Any] = None,
    progress_context: Optional[Dict[str, Any]] = None,
    branch_rule: str = "violated",
    warm_start: bool = True,
    seed: Optional[int] = 0,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[Dict[str, float]], bool]:

    repair_count: Dict[Tuple[str, str], int] = {}
    base_progress_context = dict(progress_context or {})
    # 전역 random 대신 이 호출 전용 RNG를 쓴다. 예전에는 module-level
    # random.shuffle을 써서 같은 입력도 실행마다 repair 방향 순서가 달라졌고,
    # 그 결과 탐색 경로가 통째로 바뀌어 실행 시간이 3배 이상 흔들렸다
    # (같은 인스턴스가 57초에 끝나기도 하고 200초를 넘기기도 했다).
    # 성능 비교나 버그 재현이 불가능해지므로 기본값을 고정 시드로 둔다.
    # seed=None을 넘기면 예전처럼 비결정적으로 동작한다.
    rng = random.Random(seed)

    def _limit(reason: str) -> Tuple[Optional[Dict[str, float]], bool]:
        if stats is not None:
            # 사유별로 몇 번 걸렸는지 센다. UNKNOWN이 TIMEOUT으로 뭉뚱그려질 때
            # 진짜 병목(깊이 상한인지 Simplex 반복 상한인지)을 구분하는 근거가 된다.
            stats[reason] = stats.get(reason, 0) + 1
        if report_unknown:
            raise SolverLimitReached(reason)
        return None, False

    def _try_repair(
        tableau: SimplexTableau,
        x: str,
        y: str,
        direction: int,
        depth: int,
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        import copy
        check_deadline(deadline)
        t = copy.deepcopy(tableau)

        x_val = t.assign.get(x, 0.0)
        y_val = t.assign.get(y, 0.0)

        target_var = y if direction == 0 else x
        target_val = relu(x_val) if direction == 0 else y_val

        # bounds 범위 확인
        lo = t.bounds[target_var].lower
        hi = t.bounds[target_var].upper
        if target_val < lo - 1e-9 or target_val > hi + 1e-9:
            return None, False

        # [수정된 부분] target_var가 '기저변수'라면 피벗해서 '비기저변수'로 빼내야 함
        if target_var in t.basic_vars:
            pivot_row = next((r for r in t.rows if r.basic_var == target_var), None)
            if pivot_row is not None:
                pivot_col = None
                for nv, c in pivot_row.coeffs.items():
                    if abs(c) > 1e-9:
                        pivot_col = nv
                        break
                if pivot_col is None:
                    return None, False
                _pivot(t, pivot_col, target_var)

        # 값 설정 후 simplex
        t.assign[target_var] = target_val
        for row in t.rows:
            # 예외 없이 모든 기저변수를 수식에 맞게 다시 계산!
            t.assign[row.basic_var] = _compute_basic(t, row)

        with span(trace, "simplex"):
            return simplex(
                t,
                max_iter=simplex_max_iter,
                debug=debug,
                deadline=deadline,
                report_unknown=report_unknown,
                progress=progress,
                progress_context={
                    **base_progress_context,
                    "depth": depth,
                    "origin": "relu_repair",
                },
            )

    def _select_violation(violations: List[Tuple[str, str]]) -> Tuple[str, str]:
        return min(violations, key=lambda p: repair_count.get(p, 0))

    def _rec(
        bounds_now: Dict[str, Tuple[float, float]],
        depth: int,
        current_row_defs: Optional[List[Tuple[str, Dict[str, float]]]] = None,
        warm: Optional[SimplexTableau] = None,
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        check_deadline(deadline)
        
        if current_row_defs is None:
            current_row_defs = row_defs
            
        if depth > max_recursion:
            return _limit("RELUPLEX_RECURSION_LIMIT")

        bounds_now = dict(bounds_now)
        for _, y in relus:
            lo, hi = bounds_now.get(y, (float('-inf'), float('inf')))
            new_lo = max(0.0, lo)
            
            # [수정된 부분] 모순된 제약(하한이 상한보다 큼) 발생 시 즉시 UNSAT 처리
            if new_lo > hi + 1e-9:
                return None, False
                
            bounds_now[y] = (new_lo, hi)

        if warm is not None:
            # 부모의 basis를 물려받고, 좁아진 bound만 반영해 이어서 푼다.
            # bounds_now 기준으로 한 번 맞춰주므로 부모와 어긋날 여지가 없다.
            tableau = warm
            _sync_bounds(tableau, bounds_now)
        else:
            tableau = build_tableau(current_row_defs, bounds_now)
        with span(trace, "simplex", depth=depth):
            sol, sat = simplex(
                tableau,
                max_iter=simplex_max_iter,
                debug=debug,
                deadline=deadline,
                report_unknown=report_unknown,
                progress=progress,
                progress_context={
                    **base_progress_context,
                    "depth": depth,
                    "origin": "reluplex",
                },
            )

        if not sat:
            return None, False

        assign = sol
        violations = _check_relu_violations(assign, relus)
        if not violations:
            return assign, True

        repair_unknown_reason = None
        for _ in range(local_repair_max_iter):
            check_deadline(deadline)
            x, y = _select_violation(violations)
            pair = (x, y)
            repair_count[pair] = repair_count.get(pair, 0) + 1

            best_assign = None
            directions = [0, 1]
            rng.shuffle(directions)
            for direction in directions:
                try:
                    with span(trace, "reluplex_repair", depth=depth, branch_x=x):
                        sol2, sat2 = _try_repair(tableau, x, y, direction, depth)
                except SolverLimitReached as exc:
                    if exc.reason == "TIMEOUT":
                        raise
                    repair_unknown_reason = exc.reason
                    continue
                if not sat2:
                    continue

                violations2 = _check_relu_violations(sol2, relus)
                if not violations2:
                    return sol2, True

                if best_assign is None or len(violations2) < len(_check_relu_violations(best_assign, relus)):
                    best_assign = sol2

            if best_assign is None:
                break

            assign = best_assign
            violations = _check_relu_violations(assign, relus)
            if not violations:
                return assign, True

            if repair_count.get(_select_violation(violations), 0) >= branch_tau:
                break

        # ── 분기 변수(branch_x) 선택 ──
        # 원래 Reluplex 규칙은 "repair로 여러 번 고쳐봤는데 계속 말썽인 ReLU를
        # 분기하라"이다. 그런데 후보를 repair_count에 등록된 쌍 전체에서 골라서
        # 두 가지가 어긋나 있었다:
        #   (a) 지금 위반하지도 않는 ReLU를 분기할 수 있다 — 충돌과 무관한 곳을
        #       쪼개므로 그 분기는 대체로 헛일이다.
        #   (b) 반대로 지금 위반 중인데 repair에 한 번도 안 뽑힌 ReLU는 후보에서
        #       빠진다. 그래서 쪼갤 게 남아있는데도 branch_x=None으로 포기하는
        #       경우가 생긴다.
        # 후보를 "지금 위반 중이면서 아직 고정되지 않은 ReLU"로 좁힌다. 순위는
        # repair_count 내림차순(원 규칙)을 유지하되, 동점은 이름순으로 끊어
        # 실행을 결정적으로 만든다 — 예전에는 repair_count가 금세 포화돼서
        # sorted()가 사실상 삽입 순서(=뉴런 번호순)를 돌려주고 있었다.
        def _splittable(px: str) -> bool:
            lo, hi = bounds_now.get(px, (float('-inf'), float('inf')))
            return lo < 0 and hi > 0

        branch_x = None
        relu_y = None
        if branch_rule == "violated":
            candidates = [p for p in violations if _splittable(p[0])]
            if candidates:
                branch_x, relu_y = min(
                    candidates, key=lambda p: (-repair_count.get(p, 0), p[0])
                )
        if branch_x is None:
            # legacy 경로 겸 폴백: 위반 중인 후보가 하나도 못 쪼개질 때는
            # 예전처럼 repair 이력 전체에서 고른다.
            order = sorted(repair_count, key=lambda p: (-repair_count[p], p[0]))
            for pair in order:
                if _splittable(pair[0]):
                    branch_x, relu_y = pair
                    break


        if branch_x is not None and depth < max_recursion:
            neuron_key = None
            if relu_y is not None and relu_metadata is not None:
                layer_index = relu_metadata.get((branch_x, relu_y))
                if layer_index is not None and None not in layer_index:
                    neuron_key = (int(layer_index[0]), int(layer_index[1]))
            if neuron_key is None:
                neuron_key = _parse_neuron_var(branch_x)

            # 이 with 블록이 열려있는 동안(양쪽 분기가 다 끝날 때까지)이
            # "지금 branch_x에서 split이 진행 중"인 구간이다 — 시각화에서
            # "현재 열려있는 branch_x 스택"을 그대로 이 이벤트들로 재구성한다.
            with span(trace, "reluplex_split", depth=depth, branch_x=branch_x):
                lo, hi = bounds_now.get(branch_x, (float('-inf'), float('inf')))
                layer = neuron_key[0] if neuron_key is not None else None
                index = neuron_key[1] if neuron_key is not None else None

                # 1. x >= 0 분기
                bounds1 = dict(bounds_now)
                bounds1[branch_x] = (max(0.0, lo), hi)
                row_defs1 = list(current_row_defs)
                if relu_y is not None:
                    slack_name = f"relu_slack_{branch_x}_pos_{depth}"
                    row_defs1.append((slack_name, {relu_y: 1.0, branch_x: -1.0}))
                    bounds1[slack_name] = (0.0, 0.0)

                branch_unknown_reason = None
                split_id = None
                if split_logger is not None:
                    # global_split_count 증가 및 '+' 기록: 첫 Reluplex 재귀 호출 직전.
                    split_id = split_logger.begin(branch_x, layer, index)
                warm1 = None
                if warm_start:
                    new_rows1 = []
                    if relu_y is not None:
                        new_rows1.append(
                            (slack_name, {relu_y: 1.0, branch_x: -1.0}, (0.0, 0.0))
                        )
                    warm1 = _child_tableau(tableau, new_rows1)

                try:
                    try:
                        r1, sat1 = _rec(bounds1, depth + 1, row_defs1, warm1)
                    except SolverLimitReached as exc:
                        if exc.reason == "TIMEOUT":
                            raise
                        branch_unknown_reason = exc.reason
                        r1, sat1 = None, False
                    if sat1:
                        return r1, True
                    # 자식이 끝났으므로 참조를 놓아준다 (depth만큼 tableau가
                    # 동시에 살아있지 않도록)
                    warm1 = None

                    # 2. x <= 0 분기
                    bounds2 = dict(bounds_now)
                    bounds2[branch_x] = (lo, min(0.0, hi))
                    row_defs2 = list(current_row_defs)
                    inactive_bounds_conflict = False
                    if relu_y is not None:
                        y_lo, y_hi = bounds2.get(
                            relu_y, (float('-inf'), float('inf'))
                        )
                        inactive_bounds_conflict = max(y_lo, 0.0) > min(y_hi, 0.0) + 1e-9
                        if not inactive_bounds_conflict:
                            bounds2[relu_y] = (max(y_lo, 0.0), min(y_hi, 0.0))

                    if inactive_bounds_conflict:
                        r2, sat2 = None, False
                    else:
                        warm2 = _child_tableau(tableau, []) if warm_start else None
                        try:
                            r2, sat2 = _rec(bounds2, depth + 1, row_defs2, warm2)
                        except SolverLimitReached as exc:
                            if exc.reason == "TIMEOUT":
                                raise
                            branch_unknown_reason = branch_unknown_reason or exc.reason
                            r2, sat2 = None, False
                    if sat2:
                        return r2, True

                    if branch_unknown_reason is not None:
                        return _limit(branch_unknown_reason)
                    return None, False
                finally:
                    if split_logger is not None and split_id is not None:
                        # 두 번째 재귀 호출 이후(또는 조기 반환/예외 시) count 감소.
                        split_logger.end(split_id, branch_x, layer, index)

        if depth >= max_recursion:
            return _limit("RELUPLEX_RECURSION_LIMIT")
        return _limit(repair_unknown_reason or "RELUPLEX_REPAIR_INCONCLUSIVE")

    # [누락되었던 부분 복구] reluplex 함수의 마지막 반환문!
    return _rec(dict(bounds), 0, row_defs)


# ─────────────────────────────────────────────
#  테스트
# ─────────────────────────────────────────────

def main() -> None:
    # ─── Reluplex 테스트 ───
    print("\n" + "=" * 55)
    print("  Reluplex 테스트: x + y >= 5, y = relu(x)")
    row_defs_rel = [("s1", {"x": 1.0, "y": 1.0})]
    bounds_rel = {
        "s1": (5.0, float('inf')),
        "x": (-float('inf'), float('inf')),
        "y": (-float('inf'), float('inf')),
    }
    relus = [("x", "y")]
    try:
        sol_rel, sat_rel = reluplex(row_defs_rel, bounds_rel, relus, debug=True)
        print(f"Reluplex 결과: {'SAT: ' + str(sol_rel) if sat_rel else 'UNSAT'}")
    except Exception as e:
        print(f"Reluplex 테스트 중 오류: {e}")

    print("\n" + "=" * 55)
    print("  Reluplex 테스트: x >= 0, y = relu(x), y < 0 (UNSAT 예제)")

    row_defs_rel = [
        ("c1", {"x": 1.0}),      # x >= 0
    ]

    bounds_rel = {
        "c1": (0.0, float('inf')),   # x >= 0
        "x": (-float('inf'), float('inf')),
        "y": (-float('inf'), -1e-6),  # y < 0
    }

    relus = [("x", "y")]

    try:
        sol_rel, sat_rel = reluplex(row_defs_rel, bounds_rel, relus)
        print(f"Reluplex 결과: {'SAT: ' + str(sol_rel) if sat_rel else 'UNSAT'}")
    except Exception as e:
        print(f"Reluplex 테스트 중 오류: {e}")

    print("\n" + "=" * 55)
    print("  Reluplex 테스트 (SAT): x + y <= 2, y = relu(x)")

    row_defs_rel = [
        ("s1", {"x": 1.0, "y": 1.0}),   # x + y <= 2
    ]

    bounds_rel = {
        "s1": (2, float('inf')),     # x + y <= 2
        "x": (-float('inf'), float('inf')),
        "y": (-float('inf'), float('inf')),
    }

    relus = [("x", "y")]

    try:
        sol_rel, sat_rel = reluplex(row_defs_rel, bounds_rel, relus)
        print(f"Reluplex 결과: {'SAT: ' + str(sol_rel) if sat_rel else 'UNSAT'}")
    except Exception as e:
        print(f"Reluplex 테스트 중 오류: {e}")


    

if __name__ == "__main__":
    main()
