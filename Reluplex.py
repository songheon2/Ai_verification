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
    branch_tau: int = 5,    debug: bool = False,) -> Tuple[Optional[Dict[str, float]], bool]:

    repair_count: Dict[Tuple[str, str], int] = {}

    def _try_repair(
        tableau: SimplexTableau,
        x: str,
        y: str,
        direction: int,
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        import copy
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

        return simplex(t, max_iter=simplex_max_iter, debug=debug)

    def _select_violation(violations: List[Tuple[str, str]]) -> Tuple[str, str]:
        return min(violations, key=lambda p: repair_count.get(p, 0))

    def _rec(
        bounds_now: Dict[str, Tuple[float, float]], 
        depth: int, 
        current_row_defs: Optional[List[Tuple[str, Dict[str, float]]]] = None
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        
        if current_row_defs is None:
            current_row_defs = row_defs
            
        if depth > max_recursion:
            return None, False

        bounds_now = dict(bounds_now)
        for _, y in relus:
            lo, hi = bounds_now.get(y, (float('-inf'), float('inf')))
            new_lo = max(0.0, lo)
            
            # [수정된 부분] 모순된 제약(하한이 상한보다 큼) 발생 시 즉시 UNSAT 처리
            if new_lo > hi + 1e-9:
                return None, False
                
            bounds_now[y] = (new_lo, hi)

        tableau = build_tableau(current_row_defs, bounds_now)
        sol, sat = simplex(tableau, max_iter=simplex_max_iter, debug=debug)
        
        if not sat:
            return None, False

        assign = sol
        violations = _check_relu_violations(assign, relus)
        if not violations:
            return assign, True

        for _ in range(local_repair_max_iter):
            x, y = _select_violation(violations)
            pair = (x, y)
            repair_count[pair] = repair_count.get(pair, 0) + 1

            best_assign = None
            directions = [0, 1]
            random.shuffle(directions)
            for direction in directions:
                sol2, sat2 = _try_repair(tableau, x, y, direction)
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

        # [누락되었던 부분 복구] 분기 변수(branch_x) 선택 로직!
        branch_x = None
        for pair in sorted(repair_count, key=lambda p: -repair_count[p]):
            px, _ = pair
            lo, hi = bounds_now.get(px, (float('-inf'), float('inf')))
            if lo < 0 and hi > 0:
                branch_x = px
                break

        relu_y = None
        for px, py in relus:
            if px == branch_x:
                relu_y = py
                break

        if branch_x is not None and depth < max_recursion:
            lo, hi = bounds_now.get(branch_x, (float('-inf'), float('inf')))

            # 1. x >= 0 분기
            bounds1 = dict(bounds_now)
            bounds1[branch_x] = (max(0.0, lo), hi)
            row_defs1 = list(current_row_defs)
            if relu_y is not None:
                slack_name = f"relu_slack_{branch_x}_pos_{depth}" 
                row_defs1.append((slack_name, {relu_y: 1.0, branch_x: -1.0}))
                bounds1[slack_name] = (0.0, 0.0) 
            
            r1, sat1 = _rec(bounds1, depth + 1, row_defs1)
            if sat1:
                return r1, True

            # 2. x <= 0 분기
            bounds2 = dict(bounds_now)
            bounds2[branch_x] = (lo, min(0.0, hi))
            row_defs2 = list(current_row_defs)
            if relu_y is not None:
                bounds2[relu_y] = (0.0, 0.0)
            
            r2, sat2 = _rec(bounds2, depth + 1, row_defs2)
            if sat2:
                return r2, True

            return None, False

        return None, False

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