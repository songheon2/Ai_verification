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


class BoundDeps:
    """경계마다 "어느 깊이의 분기 결정에 기대는가"를 비트마스크로 기록한다.

    충돌 분석과 백점프(논문 4절 "Derived Bounds and Conflict Analysis")에 쓴다.
    비트 i는 깊이 i에서 내린 분기 결정이다. 원래 문제의 경계와 ReLU의 정의
    (y >= 0, y >= x, x <= y)에서 나온 경계는 0이다. 이 저장소의 인코딩에서 row는
    모두 `slack = sum(a*v)` 꼴의 정의라 그 자체로는 아무것도 제한하지 않고, 분기
    결정도 전부 경계(x >= 0, x <= 0, y = 0, slack = 0)로 들어간다. 그래서 경계의
    원인만 따라가면 모순의 원인 분기를 정확히 얻는다. 경계를 좁힐 때마다 그
    계산에 쓰인 경계들의 마스크를 OR해서 새 경계에 붙인다.

    lo / hi : 변수 -> 하한/상한의 마스크 (없으면 0, 0인 항목은 저장하지 않는다)
    conflict: 마지막 모순이 기대는 분기 마스크
    fixed   : 확정한 ReLU (x, y) -> 확정 근거 경계의 마스크
    """

    __slots__ = ("lo", "hi", "conflict", "fixed")

    def __init__(self, lo: Optional[Dict[str, int]] = None,
                 hi: Optional[Dict[str, int]] = None) -> None:
        self.lo: Dict[str, int] = dict(lo) if lo else {}
        self.hi: Dict[str, int] = dict(hi) if hi else {}
        self.conflict = 0
        self.fixed: Dict[Tuple[str, str], int] = {}

    def copy(self) -> "BoundDeps":
        return BoundDeps(self.lo, self.hi)

    def set_lo(self, var: str, mask: int) -> None:
        if mask:
            self.lo[var] = mask
        else:
            self.lo.pop(var, None)

    def set_hi(self, var: str, mask: int) -> None:
        if mask:
            self.hi[var] = mask
        else:
            self.hi.pop(var, None)


def _row_deps(terms: List[Tuple[str, float]], deps: BoundDeps) -> Tuple[int, int]:
    """sum(c*v)의 하한·상한이 기대는 마스크. 하한은 c>0이면 v의 하한, c<0이면 상한으로 만든다."""
    d_lo = d_hi = 0
    for v, c in terms:
        if c > 0:
            d_lo |= deps.lo.get(v, 0)
            d_hi |= deps.hi.get(v, 0)
        else:
            d_lo |= deps.hi.get(v, 0)
            d_hi |= deps.lo.get(v, 0)
    return d_lo, d_hi


def _rest_deps(terms: List[Tuple[str, float]], deps: BoundDeps):
    """i번째 항을 뺀 나머지 항들의 (하한 마스크, 상한 마스크)를 돌려주는 함수.

    OR는 되돌릴 수 없어서 "전체 - 자기"로 못 구하므로 앞쪽·뒤쪽 누적 OR를 쓴다.
    """
    n = len(terms)
    lo_m = [0] * n
    hi_m = [0] * n
    for i, (v, c) in enumerate(terms):
        if c > 0:
            lo_m[i], hi_m[i] = deps.lo.get(v, 0), deps.hi.get(v, 0)
        else:
            lo_m[i], hi_m[i] = deps.hi.get(v, 0), deps.lo.get(v, 0)
    pre_lo, pre_hi = [0] * (n + 1), [0] * (n + 1)
    for i in range(n):
        pre_lo[i + 1] = pre_lo[i] | lo_m[i]
        pre_hi[i + 1] = pre_hi[i] | hi_m[i]
    suf_lo, suf_hi = [0] * (n + 1), [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf_lo[i] = suf_lo[i + 1] | lo_m[i]
        suf_hi[i] = suf_hi[i + 1] | hi_m[i]
    return lambda i: (pre_lo[i] | suf_lo[i + 1], pre_hi[i] | suf_hi[i + 1])


def _term_range(coeff: float, lo: float, hi: float) -> Tuple[float, float]:
    """coeff * v 의 [최솟값, 최댓값]."""
    if coeff > 0:
        return (coeff * lo if lo != -INF else -INF,
                coeff * hi if hi != INF else INF)
    return (coeff * hi if hi != INF else -INF,
            coeff * lo if lo != -INF else INF)


def _tighten(bounds: Bounds, var: str, lo: float, hi: float,
             tol: float, deps: Optional[BoundDeps] = None,
             dep_lo: int = 0, dep_hi: int = 0) -> Optional[bool]:
    """var의 경계를 [lo, hi]와 교집합한다.

    deps가 있으면 새로 바뀐 쪽 경계에 그 원인 마스크(dep_lo / dep_hi)를 붙이고,
    모순이면 모순에 쓰인 두 경계의 마스크를 deps.conflict에 남긴다.

    반환: True=좁아짐, False=변화 없음, None=모순(l > u)
    """
    cur_lo, cur_hi = bounds[var]
    new_lo = cur_lo if lo == -INF else max(cur_lo, lo)
    new_hi = cur_hi if hi == INF else min(cur_hi, hi)
    if new_lo - new_hi > tol * max(1.0, abs(new_lo), abs(new_hi)):
        if deps is not None:
            deps.conflict = (
                (dep_lo if new_lo != cur_lo else deps.lo.get(var, 0))
                | (dep_hi if new_hi != cur_hi else deps.hi.get(var, 0))
            )
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
        if deps is not None:
            if new_lo != cur_lo:
                deps.set_lo(var, dep_lo)
            if new_hi != cur_hi:
                deps.set_hi(var, dep_hi)
    return changed


def derive_bounds(
    row_defs: RowDefs,
    bounds: Bounds,
    relus: Iterable[Tuple[str, str]],
    *,
    max_passes: int = 20,
    tol: float = 1e-9,
    deps: Optional[BoundDeps] = None,
) -> Tuple[Bounds, Dict[Tuple[str, str], str], bool]:
    """구간 전파를 고정점까지 돌린다.

    deps: 주면 경계마다 원인 분기 마스크를 제자리에서 갱신한다(BoundDeps).
        모순이면 deps.conflict, 확정한 ReLU의 근거는 deps.fixed에 남는다.
        원인 계산은 경계가 실제로 바뀔 때만 하므로 대부분의 row에서는 비용이 없다.

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
            cur_lo, cur_hi = b[name]
            # 아래 역방향과 같은 이유로, 좁힐 게 없으면 _tighten을 건너뛴다
            if s_lo > cur_lo or s_hi < cur_hi or cur_lo > cur_hi:
                d_lo, d_hi = _row_deps(terms, deps) if deps is not None else (0, 0)
                r = _tighten(b, name, s_lo, s_hi, tol, deps, d_lo, d_hi)
                if r is None:
                    return b, fixed, True
                changed |= r

            # 역방향: name과 "나머지 항"으로 각 v를 좁힌다
            n_lo, n_hi = b[name]
            if n_lo == -INF and n_hi == INF:
                continue
            rest_deps = None    # 원인 추적용, 처음 필요할 때 만든다
            for i, (v, c) in enumerate(terms):
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
                # 현재 경계보다 좁지 않고 현재 경계에 모순이 없으면 _tighten은
                # 아무것도 바꾸지 못하고 모순도 내지 않는다. 탐색 중 노드마다
                # 유도를 돌리면 호출의 대부분이 이 경우라서 함수 호출 자체를
                # 건너뛴다 (결과는 같고 비용만 준다).
                cur_lo, cur_hi = b[v]
                if v_lo <= cur_lo and v_hi >= cur_hi and cur_lo <= cur_hi:
                    continue
                d_lo = d_hi = 0
                if deps is not None:
                    # c*v = name - rest 이므로 c*v의 하한은 name 하한과 rest 상한,
                    # 상한은 name 상한과 rest 하한으로 만든다. c<0이면 v에서 뒤집힌다.
                    if rest_deps is None:
                        rest_deps = _rest_deps(terms, deps)
                    r_lo, r_hi = rest_deps(i)
                    cv_lo = deps.lo.get(name, 0) | r_hi
                    cv_hi = deps.hi.get(name, 0) | r_lo
                    d_lo, d_hi = (cv_lo, cv_hi) if c > 0 else (cv_hi, cv_lo)
                r = _tighten(b, v, v_lo, v_hi, tol, deps, d_lo, d_hi)
                if r is None:
                    return b, fixed, True
                changed |= r

        # ── ReLU 의미 전파 ──
        # 원인 마스크: relu의 정의에서 바로 나오는 관계(y >= 0, y >= x, x <= y,
        # y <= max(0, u(x)))는 언제나 참이라 쓰인 경계의 마스크만 물려받는다.
        # y > 0 이면 x > 0 이라 x = y 이므로 l(x) >= l(y)도 l(y)의 마스크만 쓴다.
        for x, y in relu_list:
            if x not in b or y not in b:
                continue
            x_lo, x_hi = b[x]
            dxl = dxh = 0
            if deps is not None:
                dxl, dxh = deps.lo.get(x, 0), deps.hi.get(x, 0)
            # y >= 0 이고 y >= x  (relu의 정의)
            r = _tighten(b, y, max(0.0, x_lo if x_lo != -INF else 0.0), INF, tol,
                         deps, dxl if x_lo > 0.0 else 0, 0)
            if r is None:
                return b, fixed, True
            changed |= r
            # y <= max(0, u(x))
            if x_hi != INF:
                r = _tighten(b, y, -INF, max(0.0, x_hi), tol, deps, 0, dxh)
                if r is None:
                    return b, fixed, True
                changed |= r

            y_lo, y_hi = b[y]
            dyl = dyh = 0
            if deps is not None:
                dyl, dyh = deps.lo.get(y, 0), deps.hi.get(y, 0)
            if x_lo >= 0.0 and x_lo != -INF:
                fixed[(x, y)] = "active"        # 항상 켜짐 → y = x
                if deps is not None:
                    deps.fixed[(x, y)] = dxl
                r = _tighten(b, y, x_lo, x_hi, tol, deps, dxl, dxh)
                if r is None:
                    return b, fixed, True
                changed |= r
                r = _tighten(b, x, y_lo, y_hi, tol, deps, dyl, dyh)
                if r is None:
                    return b, fixed, True
                changed |= r
            elif x_hi <= 0.0 and x_hi != INF:
                fixed[(x, y)] = "inactive"      # 항상 꺼짐 → y = 0
                if deps is not None:
                    deps.fixed[(x, y)] = dxh
                r = _tighten(b, y, 0.0, 0.0, tol, deps, 0, dxh)
                if r is None:
                    return b, fixed, True
                changed |= r
            elif y_hi <= 0.0 and y_hi != INF:
                # y는 항상 >= 0 이므로 u(y) <= 0 은 곧 y = 0 이다.
                # relu(x) = 0 이려면 x <= 0 이어야 한다.
                fixed[(x, y)] = "inactive"
                if deps is not None:
                    deps.fixed[(x, y)] = dyh
                r = _tighten(b, x, -INF, 0.0, tol, deps, 0, dyh)
                if r is None:
                    return b, fixed, True
                changed |= r
            elif y_lo > 0.0:
                # y가 확실히 양수면 relu는 활성 상태일 수밖에 없다
                fixed[(x, y)] = "active"
                if deps is not None:
                    deps.fixed[(x, y)] = dyl
                r = _tighten(b, x, y_lo, y_hi, tol, deps, dyl, dyh)
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
