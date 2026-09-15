"""구간 전파로 변수 경계를 좁히고, 안정(stable) ReLU를 분기 없이 확정한다.

Reluplex 논문 4절의 "Tighter Bound Derivation"에 해당한다. 논문은 기저변수
xi의 row에서 pos/neg 계수를 나눠 l(xi)/u(xi)를 유도하는데, 이 저장소는 모든
제약을 `slack = sum(a*v)` + `slack >= b` 형태로 인코딩하므로 두 방향이 모두
필요하다:

    정방향: v들의 경계 -> slack의 경계
    역방향: slack과 나머지 v들의 경계 -> 특정 v의 경계

역방향이 없으면 신경망 등식(z = Wx + b를 두 부등식으로 쪼갠 것)을 타고
입력 박스가 은닉층까지 전파되지 못한다.

왜 필요한가
-----------
이 전까지 이론 솔버는 `y >= 0`(ReLU 출력은 음수가 아님) 하나만 알고 시작했다.
입력 박스조차 slack row에만 들어있고 변수 자신의 bound는 (-inf, inf)여서,
모든 ReLU가 "어느 쪽인지 모름" 상태였다 — 즉 전부 분기 후보였다. 실측에서
mMIMO(982 ReLU)는 실제로 불안정한 게 126개(12.8%)뿐인데도 982개를 모두
쪼갤 대상으로 보고 있었다.
"""

from __future__ import annotations

from math import isfinite
from typing import Dict, Iterable, List, Optional, Tuple

INF = float("inf")
Bounds = Dict[str, Tuple[float, float]]
RowDefs = List[Tuple[str, Dict[str, float]]]


def _term_range(coeff: float, lo: float, hi: float) -> Tuple[float, float]:
    """coeff * v 의 [최솟값, 최댓값]."""
    if coeff > 0:
        return (coeff * lo if lo != -INF else -INF,
                coeff * hi if hi != INF else INF)
    return (coeff * hi if hi != INF else -INF,
            coeff * lo if lo != -INF else INF)


def _tighten(bounds: Bounds, var: str, lo: float, hi: float,
             tol: float) -> Optional[bool]:
    """var의 경계를 [lo, hi]와 교집합한다.

    반환: True=좁아짐, False=변화 없음, None=모순(l > u)
    """
    cur_lo, cur_hi = bounds[var]
    new_lo = cur_lo if lo == -INF else max(cur_lo, lo)
    new_hi = cur_hi if hi == INF else min(cur_hi, hi)
    if new_lo - new_hi > tol * max(1.0, abs(new_lo), abs(new_hi)):
        return None
    changed = False
    scale = max(1.0, abs(new_lo) if isfinite(new_lo) else 1.0,
                abs(new_hi) if isfinite(new_hi) else 1.0)
    if new_lo > cur_lo + tol * scale or (cur_lo == -INF and new_lo != -INF):
        changed = True
    if new_hi < cur_hi - tol * scale or (cur_hi == INF and new_hi != INF):
        changed = True
    if changed:
        bounds[var] = (new_lo, new_hi)
    return changed


def derive_bounds(
    row_defs: RowDefs,
    bounds: Bounds,
    relus: Iterable[Tuple[str, str]],
    *,
    max_passes: int = 20,
    tol: float = 1e-9,
) -> Tuple[Bounds, Dict[Tuple[str, str], str], bool]:
    """구간 전파를 고정점까지 돌린다.

    Returns:
        (좁혀진 bounds, {(x, y): 'active'|'inactive'}, infeasible)

        infeasible=True면 l > u 모순이 나온 것이므로 이 노드는 UNSAT이다.
    """
    b: Bounds = {v: tuple(x) for v, x in bounds.items()}  # type: ignore[misc]
    relu_list = list(relus)
    fixed: Dict[Tuple[str, str], str] = {}

    for _ in range(max_passes):
        changed = False

        # ── 선형 제약 전파 ──
        for name, coeffs in row_defs:
            terms = [(v, c) for v, c in coeffs.items() if c != 0.0 and v in b]
            if not terms or name not in b:
                continue

            # 정방향: sum(a*v) 의 범위로 name을 좁힌다.
            # "나머지 항의 합"을 (전체 - 자기항)으로 빼서 구하면 전체가 무한대일 때
            # -inf - (-inf) 가 되어 계산이 무너진다. 그래서 유한한 항만 따로 더하고
            # 무한대 항이 몇 개인지를 세어, 자기 자신이 유일한 무한대 원인일 때만
            # 나머지 합이 유한하다고 판정한다.
            lo_fin = hi_fin = 0.0      # 유한한 항들의 합
            lo_inf = hi_inf = 0        # -inf / +inf 를 만드는 항의 개수
            tr = {}
            for v, c in terms:
                lo, hi = b[v]
                t_lo, t_hi = _term_range(c, lo, hi)
                tr[v] = (t_lo, t_hi)
                if t_lo == -INF:
                    lo_inf += 1
                else:
                    lo_fin += t_lo
                if t_hi == INF:
                    hi_inf += 1
                else:
                    hi_fin += t_hi

            s_lo = -INF if lo_inf else lo_fin
            s_hi = INF if hi_inf else hi_fin
            r = _tighten(b, name, s_lo, s_hi, tol)
            if r is None:
                return b, fixed, True
            changed |= r

            # 역방향: name과 "나머지 항"으로 각 v를 좁힌다
            n_lo, n_hi = b[name]
            if n_lo == -INF and n_hi == INF:
                continue
            for v, c in terms:
                t_lo, t_hi = tr[v]
                # 나머지 항들의 합 (자기 자신 제외)
                if t_lo == -INF:
                    rest_lo = lo_fin if lo_inf == 1 else -INF
                else:
                    rest_lo = -INF if lo_inf else lo_fin - t_lo
                if t_hi == INF:
                    rest_hi = hi_fin if hi_inf == 1 else INF
                else:
                    rest_hi = INF if hi_inf else hi_fin - t_hi

                # c*v = name - rest
                cv_lo = -INF if (n_lo == -INF or rest_hi == INF) else n_lo - rest_hi
                cv_hi = INF if (n_hi == INF or rest_lo == -INF) else n_hi - rest_lo
                if c > 0:
                    v_lo = -INF if cv_lo == -INF else cv_lo / c
                    v_hi = INF if cv_hi == INF else cv_hi / c
                else:
                    v_lo = -INF if cv_hi == INF else cv_hi / c
                    v_hi = INF if cv_lo == -INF else cv_lo / c
                r = _tighten(b, v, v_lo, v_hi, tol)
                if r is None:
                    return b, fixed, True
                changed |= r

        # ── ReLU 의미 전파 ──
        for x, y in relu_list:
            if x not in b or y not in b:
                continue
            x_lo, x_hi = b[x]
            # y >= 0 이고 y >= x  (relu의 정의)
            r = _tighten(b, y, max(0.0, x_lo if x_lo != -INF else 0.0), INF, tol)
            if r is None:
                return b, fixed, True
            changed |= r
            # y <= max(0, u(x))
            if x_hi != INF:
                r = _tighten(b, y, -INF, max(0.0, x_hi), tol)
                if r is None:
                    return b, fixed, True
                changed |= r

            y_lo, y_hi = b[y]
            if x_lo >= 0.0 and x_lo != -INF:
                fixed[(x, y)] = "active"        # 항상 켜짐 → y = x
                r = _tighten(b, y, x_lo, x_hi, tol)
                if r is None:
                    return b, fixed, True
                changed |= r
                r = _tighten(b, x, y_lo, y_hi, tol)
                if r is None:
                    return b, fixed, True
                changed |= r
            elif x_hi <= 0.0 and x_hi != INF:
                fixed[(x, y)] = "inactive"      # 항상 꺼짐 → y = 0
                r = _tighten(b, y, 0.0, 0.0, tol)
                if r is None:
                    return b, fixed, True
                changed |= r
            elif y_hi <= 0.0 and y_hi != INF:
                # y는 항상 >= 0 이므로 u(y) <= 0 은 곧 y = 0 이다.
                # relu(x) = 0 이려면 x <= 0 이어야 한다.
                fixed[(x, y)] = "inactive"
                r = _tighten(b, x, -INF, 0.0, tol)
                if r is None:
                    return b, fixed, True
                changed |= r
            elif y_lo > 0.0:
                # y가 확실히 양수면 relu는 활성 상태일 수밖에 없다
                fixed[(x, y)] = "active"
                r = _tighten(b, x, y_lo, y_hi, tol)
                if r is None:
                    return b, fixed, True
                changed |= r

        if not changed:
            break

    return b, fixed, False


def apply_fixed_relus(
    row_defs: RowDefs,
    bounds: Bounds,
    fixed: Dict[Tuple[str, str], str],
) -> Tuple[RowDefs, Bounds]:
    """확정된 ReLU를 선형 제약으로 바꿔 넣는다 (분기했을 때와 같은 형태).

    active   → y - x = 0   (slack row, bounds (0,0))
    inactive → y = 0       (bounds만)
    """
    new_rows = list(row_defs)
    new_bounds = dict(bounds)
    for (x, y), state in sorted(fixed.items()):
        if state == "active":
            slack = f"relu_fixed_{x}_on"
            new_rows.append((slack, {y: 1.0, x: -1.0}))
            new_bounds[slack] = (0.0, 0.0)
        else:
            new_bounds[y] = (0.0, 0.0)
    return new_rows, new_bounds
