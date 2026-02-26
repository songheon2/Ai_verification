"""Reluplex 알고리즘 구현 (내부적으로 `Simplex.build_tableau`와 `simplex` 사용).

이 모듈은 `reluplex(row_defs, bounds, relus)`를 제공하며,
사진의 Algorithm 4(간단화된 Reluplex)의 재귀적 구현을 따릅니다.
"""
from typing import Dict, List, Tuple, Optional
from Simplex import build_tableau, simplex, _pivot, _compute_basic, SimplexTableau
import random

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
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    간단화된 Reluplex 솔버.

    수정 사항:
        1) ReLU 수리 방향을 양방향(y←relu(x) 또는 x←y)으로 시도
        2) 로컬 수리 실패 시 UNSAT 조기 반환 제거 → 케이스 분기로 넘어감
        3) 위반 선택에 시도 횟수(tau) 카운터 적용하여 루프 방지
        4) 케이스 분기 결과를 명시적으로 체크
    """

    # bounds 합법성 검사: lower > upper인 변수 있으면 즉시 UNSAT
    for var, (lo, hi) in bounds.items():
        if lo > hi:
            # print(f"[Reluplex] 변수 '{var}'의 bounds가 불가능: [{lo}, {hi}]")
            return None, False

    # [수정 3] 각 위반 노드(x,y)에 대한 수리 시도 횟수 카운터
    repair_count: Dict[Tuple[str, str], int] = {}

    def _try_repair(
        tableau: SimplexTableau,
        x: str,
        y: str,
        direction: int,
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        import copy
        # 원본 tableau를 복사해서 수정 (원본 보존)
        t = copy.deepcopy(tableau)

        x_val = t.assign.get(x, 0.0)
        y_val = t.assign.get(y, 0.0)
        target_var = y if direction == 0 else x
        target_val = relu(x_val) if direction == 0 else y_val

        lo = t.bounds[target_var].lower
        hi = t.bounds[target_var].upper
        if target_val < lo - 1e-9 or target_val > hi + 1e-9:
            return None, False

        print(t.assign.get(x), t.assign.get(y, 0.0), direction)
        # 피벗
        if target_var not in t.basic_vars:
            pivot_row = None
            for row in t.rows:
                if target_var in row.coeffs and abs(row.coeffs[target_var]) > 1e-9:
                    pivot_row = row
                    break
            if pivot_row is None:
                return None, False
            _pivot(t, target_var, pivot_row.basic_var)

        # 값 직접 수정 후, 수정된 tableau로 Simplex를 재실행하여
        # 선형 제약(등식·범위)이 유지되는 일관된 해를 얻는다.
        t.assign[target_var] = target_val
        for row in t.rows:
            if row.basic_var != target_var:
                t.assign[row.basic_var] = _compute_basic(t, row)

        
        # 수정된 tableau로 Simplex 실행
        sol_simplex, sat_simplex = simplex(t, max_iter=simplex_max_iter)
        if not sat_simplex:
            return None, False
        return sol_simplex, True
    
    def _select_violation(
        violations: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        """
        [수정 3] tau 카운터 기반 위반 선택.
        시도 횟수가 가장 적은 위반을 먼저 선택 → 특정 노드 집중 방지.
        """
        return min(violations, key=lambda p: repair_count.get(p, 0))

    def _rec(
        bounds_now: Dict[str, Tuple[float, float]],
        depth: int,
        row_defs_now: List[Tuple[str, Dict[str, float]]],
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        if depth > max_recursion:
            return None, False

        # ReLU 출력 변수 y는 반드시 y >= 0 (relu 출력은 항상 비음수)
        bounds_now = dict(bounds_now)
        for _, y in relus:
            lo, hi = bounds_now.get(y, (float('-inf'), float('inf')))
            bounds_now[y] = (max(0.0, lo), hi)
            if max(0.0, lo) > hi:
            # print(f"[Reluplex] 변수 '{var}'의 bounds가 불가능: [{lo}, {hi}]")
                return None, False


        tableau = build_tableau(row_defs_now, bounds_now)
        sol, sat = simplex(tableau, max_iter=simplex_max_iter)
        if not sat:
            # F' 자체가 UNSAT → 진짜 UNSAT
            return None, False

        assign = sol
        violations = _check_relu_violations(assign, relus)
        if not violations:
            return assign, True
        for _ in range(local_repair_max_iter):
            # [수정 3] 시도 횟수 기반으로 위반 선택
            x, y = _select_violation(violations)
            pair = (x, y)
            repair_count[pair] = repair_count.get(pair, 0) + 1

            # [수정 1] 양방향 수리 시도
            best_assign = None
            directions = [0, 1]
            # 수리 방향을 무작위로 섞어서 시도 (편향 방지)
            random.shuffle(directions)
            for direction in directions:
                sol2, sat2 = _try_repair(tableau, x, y, direction)
                # [수정 2] sat2=False는 이 방향이 막힌 것 → 다음 방향 시도
                if not sat2:
                    continue

                violations2 = _check_relu_violations(sol2, relus)
                if not violations2:
                    # 완전히 해결
                    return sol2, True

                # 위반이 줄었으면 더 나은 후보로 기억
                if best_assign is None or len(violations2) < len(_check_relu_violations(best_assign, relus)):
                    best_assign = sol2

            if best_assign is None:
                # 양방향 모두 Simplex 실패 → 케이스 분기로 넘어감
                break

            assign = best_assign
            violations = _check_relu_violations(assign, relus)
            if not violations:
                return assign, True

            # [수정 3] tau 초과 시 케이스 분기 강제
            if repair_count.get(_select_violation(violations), 0) >= branch_tau:
                break

        # 케이스 분기
        # [수정 3] 분기 변수: 수리 시도가 tau를 초과한 변수 중 범위가 0을 포함하는 것
        branch_x = None
        for pair in sorted(repair_count, key=lambda p: -repair_count[p]):
            px, _ = pair
            lo, hi = bounds_now.get(px, (float('-inf'), float('inf')))
            if lo < 0 and hi > 0:
                branch_x = px
                break

        # 분기 변수에 대응하는 relu y 찾기
        relu_y = None
        for px, py in relus:
            if px == branch_x:
                relu_y = py
                break

        if branch_x is not None and depth < max_recursion:
            lo, hi = bounds_now.get(branch_x, (float('-inf'), float('inf')))

            # x >= 0 분기 → y = x 제약 추가
            bounds1 = dict(bounds_now)
            bounds1[branch_x] = (max(0.0, lo), hi)
            row_defs1 = list(row_defs_now)
            if relu_y is not None:
                slack_name = f"relu_slack_{branch_x}_pos"
                row_defs1 = row_defs_now + [(slack_name, {relu_y: 1.0, branch_x: -1.0})]
                bounds1[slack_name] = (0.0, 0.0)  # y - x = 0
            r1, sat1 = _rec(bounds1, depth + 1, row_defs1)
            if sat1:
                return r1, True

            # x <= 0 분기 → y = 0 제약 추가
            bounds2 = dict(bounds_now)
            bounds2[branch_x] = (lo, min(0.0, hi))
            row_defs2 = list(row_defs_now)
            if relu_y is not None:
                bounds2[relu_y] = (0.0, 0.0)  # y = 0으로 고정
            r2, sat2 = _rec(bounds2, depth + 1, row_defs2)
            if sat2:
                return r2, True

            return None, False

        return None, False

    return _rec(dict(bounds), 0, list(row_defs))


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
        sol_rel, sat_rel = reluplex(row_defs_rel, bounds_rel, relus)
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