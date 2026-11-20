"""Reluplex 알고리즘 구현 (내부적으로 `Simplex.build_tableau`와 `simplex` 사용).

이 모듈은 `reluplex(row_defs, bounds, relus)`를 제공하며,
사진의 Algorithm 4(간단화된 Reluplex)의 재귀적 구현을 따릅니다.
"""
from contextlib import contextmanager
from math import isfinite
from typing import Any, Dict, List, Tuple, Optional
from Simplex import (
    Bound,
    DEFAULT_TOL,
    DRIFT_TOL, DRIFT_CHECK_EVERY, DRIFT_CHECK_AT_EXIT,
    Row,
    SimplexTableau,
    _compute_basic,
    _pivot,
    _scaled_tol,
    _update_assign,
    build_tableau,
    simplex,
)
from BoundDerivation import BoundDeps, derive_bounds, apply_fixed_relus
from Automation.SolverStatus import SolverLimitReached, check_deadline
from visualization.SolveTrace import SolveTrace, span
import copy
import random
import re
import time
import uuid

_NEURON_VAR_RE = re.compile(r"^[zh](\d+)_(\d+)(?:_.*)?$")

# 논문 4절의 구현 기법 4가지를 켜고 끄는 이름 (기능별 비교 실험용)
FEATURES = ("bound", "backjump", "numeric", "underapprox")
# 기본값 = 전남대_reluplex 최고 설정의 기법 ①②③ (논문 Reluplex와 같은 구성).
# 엔진 기본값(reluplex의 lp_engine="glpk", glpk_extract="basis")과 함께 최고 설정이 된다.
DEFAULT_FEATURES = ("bound", "backjump", "numeric")


def feature_options(features: Any = "default", under_approx_eps: float = 0.1) -> Dict[str, Any]:
    """기능 이름 목록을 reluplex() 인자로 바꾼다.

    bound       — 1. 더 엄밀한 경계 유도: 루트와 노드마다 경계 유도, 피벗마다 진입
                  변수 강화, 1000피벗마다 tableau 전체 강화
    backjump    — 2. 충돌 분석과 백점프
    numeric     — 3. 부동소수 안전장치: 작은 계수로 피벗하지 않기, 누적 오차 검사·복원
    underapprox — 4. 과소근사 (eps = under_approx_eps)

    "default"는 지금 기본값(bound, backjump, numeric), "all"은 넷 다, "none"은 모두 끈다.
    쉼표로 구분한 문자열이나 이름 목록을 받는다.
    """
    if isinstance(features, str):
        key = features.strip().lower()
        if key == "default":
            names = set(DEFAULT_FEATURES)
        elif key == "all":
            names = set(FEATURES)
        elif key in ("none", ""):
            names = set()
        else:
            names = {name.strip() for name in key.split(",") if name.strip()}
    else:
        names = set(features)
    unknown = names - set(FEATURES)
    if unknown:
        raise ValueError(
            f"알 수 없는 기능: {', '.join(sorted(unknown))} (가능: {', '.join(FEATURES)})"
        )
    bound = "bound" in names
    return {
        "derive_bound": bound,
        "derive_in_search": bound,
        "pivot_tighten": bound,
        "full_tighten_every": 1000 if bound else 0,
        "backjump": "backjump" in names,
        "numeric_safeguards": "numeric" in names,
        "under_approx": under_approx_eps if "underapprox" in names else 0.0,
    }


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
    tableau: SimplexTableau,
    bounds_now: Dict[str, Tuple[float, float]],
    deps_now: Optional[BoundDeps] = None,
) -> None:
    """tableau의 bound를 bounds_now에 맞추고 DdM 불변식을 복구한다.

    불변식은 "비기저변수는 항상 범위 안"이다. 범위가 좁아져서 비기저변수가
    밖으로 나가면 새 경계로 당겨온다(DdM의 assertBound와 같은 처리).
    기저변수는 건드리지 않는다 — 범위를 벗어나도 되고, 그건 simplex가
    피벗으로 해결할 일이다. 마지막에 모든 기저변수를 row 식으로 재계산한다.
    deps_now가 있으면 경계의 원인 마스크도 같이 맞춘다(tableau.deps).
    """

    basic = {row.basic_var for row in tableau.rows}
    if deps_now is not None and tableau.deps is None:
        tableau.deps = BoundDeps()
    deps = tableau.deps if deps_now is not None else None
    for var, (lo, hi) in bounds_now.items():
        if var not in tableau.assign:
            continue
        tableau.bounds[var] = Bound(lower=lo, upper=hi)
        if deps is not None:
            deps.set_lo(var, deps_now.lo.get(var, 0))
            deps.set_hi(var, deps_now.hi.get(var, 0))
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
    _append_rows(child, new_rows)
    return child


def _append_rows(
    tableau: SimplexTableau,
    new_rows: List[Tuple[str, Dict[str, float], Tuple[float, float]]],
) -> None:
    """이미 풀던 tableau에 새 제약 row를 제자리에서 기저변수로 얹는다.

    원래 형태(terms)는 origin_rows에도 남긴다. Simplex가 누적 오차를 잴 때와
    tableau를 다시 세울 때 이 원래 등식을 기준으로 삼는다.
    """
    for name, terms, (lo, hi) in new_rows:
        row = Row(basic_var=name, coeffs=_express_in_nonbasic(tableau, terms))
        tableau.rows.append(row)
        tableau.origin_rows.append((name, dict(terms)))
        tableau.bounds[name] = Bound(lower=lo, upper=hi)
        tableau.assign[name] = _compute_basic(tableau, row)


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


def _solution_satisfies(
    row_defs: List[Tuple[str, Dict[str, float]]],
    bounds: Dict[str, Tuple[float, float]],
    relus: List[Tuple[str, str]],
    assign: Optional[Dict[str, float]],
    tol: float = DEFAULT_TOL,
) -> bool:
    """반환할 모델이 현재 하위문제를 실제로 만족하는지 독립적으로 확인한다.

    Simplex의 일관성 검사는 "피벗된 현재 tableau"의 내부 불변식을 보는 것이고,
    이 검사는 "원래 문제 표현"(row_defs/bounds/ReLU 의미)을 직접 본다. 둘은
    대상이 다르므로 한쪽만으로는 tableau 변환 과정의 오류를 잡지 못한다.
    """
    if assign is None:
        return False

    required = set(bounds)
    for basic_var, coeffs in row_defs:
        required.add(basic_var)
        required.update(coeffs)
    for x, y in relus:
        required.update((x, y))

    if any(var not in assign or not isfinite(assign[var]) for var in required):
        return False

    for var, (lower, upper) in bounds.items():
        value = assign[var]
        if value < lower - _scaled_tol(lower, tol) or value > upper + _scaled_tol(upper, tol):
            return False

    for basic_var, coeffs in row_defs:
        try:
            expected = sum(
                coefficient * assign[var]
                for var, coefficient in coeffs.items()
            )
        except (OverflowError, ValueError):
            return False
        actual = assign[basic_var]
        scale = max(1.0, abs(expected), abs(actual))
        if not isfinite(expected) or abs(actual - expected) > tol * scale:
            return False

    return not _check_relu_violations(assign, relus, tol=tol)


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
    stats: Optional[Dict[str, Any]] = None,
    derive_bound: bool = True,
    derive_in_search: bool = True,
    search_derive_rows: str = "both",
    search_derive_passes: int = 20,
    derive_incremental: bool = True,
    pivot_tighten: bool = True,
    full_tighten_every: int = 1000,
    backjump: bool = True,
    check_backjump: bool = False,
    under_approx: float = 0.0,
    under_approx_share: float = 0.3,
    numeric_safeguards: bool = True,
    lp_engine: str = "glpk",
    lp_mode: Optional[str] = None,
    glpk_extract: str = "basis",
    glpk_row_error: str = "auto",
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    local_repair_max_iter: 탐색 노드별 전체 수정 시도 횟수 상한.
        성공 여부와 관계없이 시도마다 1회 소비한다. branch_tau는 뉴런별 시도 상한.
    derive_in_search: 분기로 좁아진 경계로 노드마다 경계 유도를 다시 돌린다
        (논문 5절: "ReLU의 약 10%를 분기하면 경계 유도로 나머지가 전부
        제거되는 경우가 잦았다"). 모순이 나오면 LP를 풀지 않고 그 가지를
        닫고, 새로 확정된 ReLU는 선형 제약으로 바꿔 분기 후보에서 뺀다.
    search_derive_rows: 노드 경계 유도에 쓸 등식.
        "original" — 원래 row(+분기·확정으로 추가된 row). 희소하고 정확하다.
        "tableau"  — 부모가 피벗해 둔 현재 tableau row. 같은 제약의 다른
                     선형결합이라 원래 row에서는 안 나오는 경계가 나온다.
        "both"     — 둘 다.
    search_derive_passes: 노드 경계 유도를 최대 몇 바퀴 돌리나(BoundDerivation의
        max_passes). 20이면 사실상 고정점까지. 줄이면 노드마다 싸지지만 늦은
        바퀴에서 나오는 모순·확정을 놓친다 (4_5/φ3 노드 67개: 4바퀴면 시간 0.54배,
        확정 ReLU 98%, 경계 유도로 찾는 모순 20개 중 12개).
    derive_incremental: 경계 유도의 둘째 바퀴부터 경계가 바뀐 변수가 든 row만
        다시 본다(BoundDerivation incremental). 결과는 같고 시간만 준다 — 기본으로 켬
        (45문제 실험: 노드 수 40문제 모두 같고 시간 0.82배).
    pivot_tighten: Simplex가 피벗할 때마다 진입 변수의 경계를 그 row로 좁힌다
        (논문 5절). LP가 끝나면 좁아진 경계를 노드 경계에 합치고, 그 결과
        새로 확정된 ReLU를 강제한 뒤 LP를 다시 푼다.
    full_tighten_every: LP 한 번 안에서 이 피벗 수마다 tableau의 모든 row로
        경계를 강화한다 (논문 5절 "수천 피벗마다 더 철저한 경계 강화",
        Simplex의 full_tighten_every). pivot_tighten일 때만 쓰인다. 0이면 끈다.
        LP 도중의 row는 LP가 끝났을 때와 다른 선형결합이라, 자식 노드의 경계
        유도가 보지 못하는 경계가 나온다. ACAS Xu 11개(1시간)에서 1000은 끔과
        둘 다 푼 5개 모두 노드가 같거나 줄었고(987 -> 41, 1401 -> 727,
        647 -> 247, 나머지 둘은 실행되지 않아 같음) 나빠진 경우가 없었다. 50은
        새로 2개를 더 풀었지만 1개를 못 풀게 되는 등 들쭉날쭉했다.
    backjump: 충돌 분석과 백점프 (논문 4절 "Derived Bounds and Conflict
        Analysis"). 경계마다 기대는 분기 깊이를 비트마스크로 기록하고
        (BoundDeps), 가지를 닫을 때 그 모순이 실제로 기대는 분기 집합을 위로
        올린다. 첫째 가지의 모순이 이 분기와 무관하면 둘째 가지도 같은 이유로
        UNSAT이므로 풀지 않고 건너뛴다 — 여러 층이 무관하면 그만큼 한꺼번에
        되돌아간다(논문 예: 분기 8, 7, 6을 한 번에 취소).
        기본값은 켠다 (전남대_reluplex 최고 설정, GLPK 코어와 함께). 아래는 자체
        simplex 시절의 실측이다: 모순의 원인이 경로 분기의 67~100%를 차지해
        건너뛴 것이 매번 바로 닫힐 형제 잎 하나뿐이었고, 원인 기록 비용으로 경계
        유도가 33~44% 느려져 ACAS Xu 증명 3개가 10~15% 느려졌다. 켜면
        CONFLICT_BITS / CONFLICT_DEPTH로 "모순이 경로의 몇 할에 기대는가"를 잴 수 있다.
    check_backjump: 테스트용. 건너뛸 가지를 실제로 풀어 보고 SAT이면
        AssertionError를 낸다 (원인 기록이 틀렸는지 검증).
    under_approx: 과소근사의 eps (논문 부록 IV). 0이면 끈다. 켜면 먼저 경계가
        0 근처인 ReLU를 강제로 고정한 좁은 문제를 푼다 — 입력 하한이 -eps 이상이면
        하한을 0으로 올려 활성으로, 입력이나 출력 상한이 eps 이하이면 상한을
        0으로 내려 비활성으로 고정한다(둘 다면 남는 구간이 큰 쪽). 원래 문제의
        일부만 보는 것이라 여기서 찾은 해는 원래 문제의 해(반례)지만, 불가능이
        나와도 원래 문제는 가능할 수 있다. 그래서 반례를 못 찾으면 원래 문제를
        처음부터 다시 푼다.
    under_approx_share: 과소근사 단계에 쓸 시간 비율 (남은 시간 기준).
        deadline이 없으면 과소근사 단계는 끝날 때까지 돈다.
    numeric_safeguards: 부동소수 안전장치 (논문 4절). False면 비교 실험용으로
        끈다 — simplex와 PivotForRelu가 계수 크기를 보지 않고 이름순 첫 후보로
        피벗하고, 누적 오차 검사·복원을 하지 않는다. 판정이 틀릴 수 있다.
    lp_engine: 노드 LP("경계 위반 고치기")를 누가 푸는가. "native"는 자체 simplex.
        "glpk" / "z3"는 lp_mode에 따라 두 가지로 쓴다(LpEngines.py). 기본값 "glpk"
        (코어 방식 + glpk_extract="basis" = 전남대_reluplex 최고 설정).
    lp_mode: 엔진을 어떻게 끼우는가 (lp_engine이 native가 아닐 때).
        "core"   — 카츠 Reluplex의 구조. Reluplex가 자기 tableau를 들고 native
                   경로를 그대로 돌고, Simplex.simplex 자리만 엔진이 맡는다. 그래서
                   ①②③④가 모두 돌고, 엔진 안에서 하던 일(피벗마다 경계 강화, UNSAT의
                   원인, 오차 복원)은 엔진마다 대응 기법으로 한다(LpEngines.GlpkCore,
                   Z3Core). glpk·z3의 기본값.
        "oracle" — 엔진을 계산기처럼: 정의식과 경계만 넘기고 "해가 있나"만 받는다.
                   tableau를 쓰는 기법은 빠지고 repair는 "고칠 값으로 잠깐 고정하고
                   다시 풀기"로 한다. lp_engine="simplex"(자체 simplex를 같은 방식으로
                   쓰는 통제군)는 이 방식뿐이다.
    glpk_extract: lp_engine="glpk", lp_mode="core"일 때 다 푼 LP를 tableau로 가져오는
        방법(LpEngines.GlpkCore). "full"은 기저 row 전부를 GLPK에서 새로 계산해 오고,
        "basis"(기본값)는 카츠 Reluplex처럼 기저 목록만 받아 우리 tableau를 그만큼 피벗한다.
    glpk_row_error: 노드 경계 유도에서 tableau row에 GLPK 잡음 대비 오차 상한을 붙이는
        때(LpEngines.GlpkCore row_error). "auto"는 표 row를 GLPK에서 가져온 tableau만
        ("full"은 늘, "basis"는 우리 피벗만 거쳤으면 안 붙임), "always"는 늘.
    """
    if search_derive_rows not in ("original", "tableau", "both"):
        raise ValueError(f"알 수 없는 search_derive_rows: {search_derive_rows}")
    if lp_engine != "native":
        from LpEngines import CORES, ENGINES, make_core, make_engine
        if lp_engine not in ENGINES:
            raise ValueError(
                f"알 수 없는 lp_engine: {lp_engine} (가능: native, {', '.join(ENGINES)})"
            )
        if lp_mode is None:
            lp_mode = "core" if lp_engine in CORES else "oracle"
        if lp_mode not in ("core", "oracle"):
            raise ValueError(f"알 수 없는 lp_mode: {lp_mode} (가능: core, oracle)")
        if lp_mode == "core" and lp_engine not in CORES:
            raise ValueError(f"lp_mode='core'는 {', '.join(CORES)}만 된다 (native가 곧 simplex의 core다)")

    original_relu_count = len(relus)
    if stats is not None:
        stats['DERIVE_TOTAL_RELUS'] = original_relu_count
        stats['DERIVE_FIXED_RELUS'] = 0
        stats['DERIVE_REMAINING_RELUS'] = original_relu_count
        # 닫은 비율은 이 이론 호출의 탐색 트리 기준이다. 속성에 OR가 있으면
        # DPLL 라운드마다 다른 부분문제로 다시 불리므로 호출마다 새로 센다.
        stats['CLOSED_MEASURE'] = 0.0

    missing_relu_bounds = {
        var for pair in relus for var in pair if var not in bounds
    }
    if missing_relu_bounds:
        missing = ", ".join(sorted(missing_relu_bounds))
        raise ValueError(f"ReLU 변수의 bound가 없습니다: {missing}")

    base_progress_context = dict(progress_context or {})
    # 전역 random 대신 이 호출 전용 RNG를 쓴다. 예전에는 module-level
    # random.shuffle을 써서 같은 입력도 실행마다 repair 방향 순서가 달라졌고,
    # 그 결과 탐색 경로가 통째로 바뀌어 실행 시간이 3배 이상 흔들렸다
    # (같은 인스턴스가 57초에 끝나기도 하고 200초를 넘기기도 했다).
    # 성능 비교나 버그 재현이 불가능해지므로 기본값을 고정 시드로 둔다.
    # seed=None을 넘기면 예전처럼 비결정적으로 동작한다.
    rng = random.Random(seed)

    # ── 충돌 분석 ──
    # 깊이 i의 분기 결정 = 비트 i. _rec이 UNSAT을 돌려줄 때 그 근거가 기대는
    # 분기 마스크를 last_conflict[0]에 남기고, 부모가 바로 읽는다.
    # None은 "증명이 아님"(UNKNOWN을 UNSAT처럼 돌려준 경우)이라 백점프에 쓰지 않는다.
    track_deps = backjump
    last_conflict: List[Optional[int]] = [None]
    # Record whether the returning call detected a conflict itself or only
    # combined child results; the tree must not infer discovery from UNSAT alone.
    last_unsat_origin: List[Optional[str]] = [None]
    # 과소근사 단계에서만 0보다 크다 (_under_approx_phase가 켜고 끈다)
    ua_eps = 0.0
    ua_fixed_total = [0]    # 과소근사 단계에서 실제로 강제 고정한 ReLU 수

    def _full(depth: int) -> int:
        """원인을 모를 때: 깊이 1..depth의 분기 전부."""
        return (1 << (depth + 1)) - 2

    def _limit(reason: str) -> Tuple[Optional[Dict[str, float]], bool]:
        last_conflict[0] = None
        last_unsat_origin[0] = None
        if stats is not None:
            # 사유별로 몇 번 걸렸는지 센다. UNKNOWN이 TIMEOUT으로 뭉뚱그려질 때
            # 진짜 병목(깊이 상한인지 Simplex 반복 상한인지)을 구분하는 근거가 된다.
            stats[reason] = stats.get(reason, 0) + 1
        if report_unknown:
            raise SolverLimitReached(reason)
        return None, False

    def _bump(key: str, amount: float = 1) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + amount

    @contextmanager
    def _lp_timer(repair: bool = False):
        """LP 풀이 한 번의 횟수와 시간 (엔진 비교에서 'LP가 시간의 몇 할인가')."""
        started = time.perf_counter()
        try:
            yield
        finally:
            _bump("LP_CALLS")
            _bump("LP_SECONDS", time.perf_counter() - started)
            if repair:
                _bump("LP_REPAIR_CALLS")

    # lp_engine이 native가 아니면 아래 경계 유도가 끝난 뒤 만든다.
    # engine: oracle 방식의 엔진, core: core 방식에서 Simplex.simplex 자리를 맡는 엔진
    engine = None
    core = None
    previous_lp_model = None

    @contextmanager
    def _external_lp_progress(row_count, variable_count, context, engine_name, owner=None, model=None, bounds=None):
        from Automation.LpTelemetry import ExternalMetrics, observe_bounds
        call_id = None
        result = "ERROR"
        outcome = {}
        failure_details = {}
        collector = None
        if progress is not None:
            call_id = progress.simplex_start(
                round_index=context.get("round_index"),
                depth=context.get("depth"),
                origin=str(context.get("origin", "reluplex")),
                theory_atom_ids=context.get("theory_atom_ids", []),
                row_count=row_count, variable_count=variable_count,
                lp_engine=engine_name, work_counts_available=False,
            )
            # A long external solve may emit no intermediate events.
            if "simplex" in getattr(progress, "enabled_panels", {"simplex"}):
                collector = ExternalMetrics(owner, progress, call_id, context)
                if bounds is not None:
                    progress.simplex_bound_observations(call_id, observe_bounds(model, bounds), source="call_boundary")
            progress.flush()
        try:
            yield outcome
            result = "SAT" if outcome["sat"] else "UNSAT"
        except SolverLimitReached as exc:
            result = exc.reason
            if isinstance(getattr(exc, "details", None), dict):
                failure_details["model_validation"] = exc.details
            raise
        finally:
            if call_id is not None:
                metrics = collector.finish() if collector is not None else {
                    "iterations": None, "pivots": None, "work_counts_available": False}
                progress.simplex_end(
                    call_id, result, iterations=metrics.pop("iterations"), pivots=metrics.pop("pivots"),
                    lp_engine=engine_name, **metrics,
                    **failure_details,
                )

    def _simplex(*args, **kwargs):
        if core is None:
            return simplex(*args, **kwargs)
        # 엔진 코어(LpEngines)는 시각화 기록 콜백을 받지 않는다
        for key in ("mechanism_callback", "numerical_callback", "bound_change_callback"):
            kwargs.pop(key, None)
        tableau = args[0]
        with _external_lp_progress(
            len(tableau.rows), len(tableau.assign),
            dict(kwargs.get("progress_context") or {}), core.name, owner=core,
            model=tableau.assign, bounds=tableau.bounds,
        ) as outcome:
            answer = core.simplex(*args, **kwargs)
            outcome["sat"] = answer[1]
            return answer

    def _engine_solve(
        rows_now: List[Tuple[str, Dict[str, float]]],
        bounds_now: Dict[str, Tuple[float, float]],
        repair: bool = False,
        depth: int = 0,
    ) -> Optional[Dict[str, float]]:
        nonlocal previous_lp_model
        with _lp_timer(repair), _external_lp_progress(
            len(rows_now), len(bounds_now),
            {**base_progress_context, "depth": depth,
             "origin": "relu_repair" if repair else "reluplex"}, lp_engine, owner=engine,
            model=previous_lp_model, bounds=bounds_now,
        ) as outcome:
            # repair는 값을 잠깐 고정하고 푼다: 그때 좁힌 경계는 노드 전체에서 성립하지 않는다
            answer = engine.solve(rows_now, bounds_now, deadline)
            if answer is not None:
                previous_lp_model = answer
            outcome["sat"] = answer is not None
            return answer

    current_branch_id = None
    branch_logging = hasattr(split_logger, "branch_begin")
    round_index = base_progress_context.get("round_index")

    @contextmanager
    def _branch_span(variable, layer, index, *, phase, depth, bound, split_id, split_event):
        nonlocal current_branch_id
        if trace is None and not branch_logging:
            yield None, None
            return
        parent_id = current_branch_id
        log_id = None
        if branch_logging:
            log_id = split_logger.branch_begin(
                variable, layer, index, phase=phase, depth=depth, bound=bound,
                split_id=split_id, parent_branch_id=parent_id,
            )
        branch_id = log_id
        if branch_id is None and trace is not None:
            branch_id = uuid.uuid4().hex
        split_key = split_event.meta.setdefault("split_key", branch_id) if split_event is not None else None
        current_branch_id = branch_id
        try:
            with span(trace, "reluplex_branch", depth=depth, branch_x=variable, phase=phase,
                      layer=layer, index=index, bound=bound, round_index=round_index,
                      branch_id=branch_id, parent_branch_id=parent_id, split_key=split_key) as event:
                yield event, log_id
        finally:
            current_branch_id = parent_id

    record_mechanisms = trace is not None or hasattr(split_logger, "mechanism")

    def _record_mechanism(kind: str, depth: int, **metadata) -> None:
        if not record_mechanisms:
            return
        metadata.update(kind=kind, round_index=round_index, branch_id=current_branch_id)
        with span(trace, "reluplex_mechanism", depth=depth, **metadata):
            pass
        if hasattr(split_logger, "mechanism"):
            split_logger.mechanism(depth=depth, **metadata)

    _record_mechanism("settings", 0, backjump=backjump,
                      derive_in_search=derive_in_search, pivot_tighten=pivot_tighten,
                      full_tighten_every=full_tighten_every, derive_bound=derive_bound,
                      drift_threshold=DRIFT_TOL, drift_check_every=DRIFT_CHECK_EVERY,
                      drift_check_at_exit=DRIFT_CHECK_AT_EXIT,
                      numeric_safeguards=numeric_safeguards,
                      lp_engine=lp_engine, lp_mode=lp_mode or "native",
                      numeric_event_support="native_callbacks" if lp_engine == "native" else "not_exposed_by_external_engine",
                      search_derive_rows=search_derive_rows, warm_start=warm_start,
                      local_repair_max_iter=local_repair_max_iter, branch_tau=branch_tau,
                      underapprox_enabled=under_approx > 0.0, underapprox_epsilon=under_approx)

    def _lp_events(depth: int, **metadata) -> None:
        if any(metadata.get(key, 0) for key in
               ("cycles", "pivot_tightened", "full_tighten", "bound_conflicts")):
            _record_mechanism("simplex", depth, **metadata)

    bound_variables = {}
    if record_mechanisms:
        for x, y in relus:
            key = (relu_metadata or {}).get((x, y)) or _parse_neuron_var(x)
            for var, role in ((x, "z"), (y, "h")):
                bound_variables[var] = {"variable": var, "neuron": x, "role": role,
                                        "layer": key[0] if key else None, "index": key[1] if key else None}

    def _bound_recorder(depth, source):
        pending = []

        def flush():
            if pending:
                _record_mechanism("bound_updates", depth, source=source, changes=list(pending))
                pending.clear()

        def changed(variable, before, after, *, source=source):
            if variable not in bound_variables or before == after:
                return
            pending.append({**bound_variables[variable], "source": source,
                            "before": [value if isfinite(value) else None for value in before],
                            "after": [value if isfinite(value) else None for value in after],
                            "observed_at": time.monotonic()})
            # Batch log writes while retaining every actual change in order.
            if len(pending) >= 64:
                flush()

        return changed, flush

    def _derive_observed(*args, observation_depth, observation_source, **kwargs):
        if not record_mechanisms:
            return derive_bounds(*args, **kwargs)
        changed, flush = _bound_recorder(observation_depth, observation_source)
        try:
            return derive_bounds(*args, on_change=changed, **kwargs)
        finally:
            flush()

    def _simplex_observed(*args, observation_depth, **kwargs):
        if not record_mechanisms:
            return _simplex(*args, **kwargs)
        changed, flush = _bound_recorder(observation_depth, "lp-pivot")
        try:
            return _simplex(*args, bound_change_callback=changed, **kwargs)
        finally:
            flush()

    record_stable = trace is not None or hasattr(split_logger, "stable")

    def _record_stable(fixed, before, after, depth: int, source: str) -> None:
        if not record_stable or not fixed:
            return
        records = []
        for (x, y), state in sorted(fixed.items()):
            key = (relu_metadata or {}).get((x, y)) or _parse_neuron_var(x)
            records.append({
                "variable": x, "output": y, "state": state,
                "layer": key[0] if key else None,
                "index": key[1] if key else None,
                "before": [value if isfinite(value) else None for value in before[x]],
                "after": [value if isfinite(value) else None for value in after[x]],
            })
        metadata = {"source": source, "records": records, "round_index": round_index,
                    "branch_id": current_branch_id}
        with span(trace, "reluplex_stable", depth=depth, **metadata):
            pass
        if hasattr(split_logger, "stable"):
            split_logger.stable(depth=depth, **metadata)

    def _closed(depth: int, kind: str) -> None:
        """UNSAT으로 닫힌 가지(잎)를 센다.

        분기 한 번이 탐색 공간을 반으로 나누므로, 깊이 d에서 닫힌 가지는 전체의
        2^-d 이다. 그 합(CLOSED_MEASURE)이 1이 되면 전체가 닫힌 것이다 — 노드
        수보다 "얼마나 진행됐는가"를 정직하게 보여준다. kind는 무엇이 닫았는지
        (LP=simplex, DERIVE=경계 유도, BOUNDS=경계 모순)를 구분한다.
        """
        _bump("UNSAT_LEAVES")
        _bump(f"UNSAT_LEAVES_{kind}")
        _bump("CLOSED_MEASURE", 2.0 ** -depth)

    def _unsat(
        depth: int, kind: str, mask: Optional[int]
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        """잎을 UNSAT으로 닫고 그 원인 분기 마스크를 부모에게 남긴다.

        mask가 None이면(원인을 모름) 깊이 1..depth 전부에 기댄다고 본다.
        CONFLICT_BITS / CONFLICT_DEPTH는 원인에 실제로 든 분기 수와 경로 길이의
        합이라, 둘의 비가 "모순이 경로의 몇 할에 기대는가"이다.
        """
        _closed(depth, kind)
        _record_mechanism("conflict", depth, source=kind,
                          cause_depths=([i for i in range(1, depth + 1)
                                         if mask & (1 << i)] if mask is not None else None))
        last_unsat_origin[0] = "found"
        if not track_deps:
            last_conflict[0] = None
        elif mask is None:
            last_conflict[0] = _full(depth)
        else:
            if check_backjump and mask & ~_full(depth):
                raise AssertionError(
                    f"원인 마스크에 현재 깊이({depth})보다 깊은 분기가 있다: {mask:b}"
                )
            last_conflict[0] = mask
            _bump("CONFLICT_LEAVES")
            if not mask and depth > 0:
                # 분기 없이도 모순 = 문제 전체가 UNSAT이라는 뜻. 루트에서는 못 보고
                # 깊은 곳에서만 보이면 수치 오차로 생긴 가짜 모순일 수 있어 따로 센다.
                _bump("CONFLICT_EMPTY")
            _bump("CONFLICT_BITS", bin(mask).count("1"))
            _bump("CONFLICT_DEPTH", depth)
        return None, False

    def _validated_sat(
        assign: Optional[Dict[str, float]],
        rows_now: List[Tuple[str, Dict[str, float]]],
        bounds_now: Dict[str, Tuple[float, float]],
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        """SAT 후보를 원래 문제 표현으로 다시 검산한 뒤에만 SAT으로 인정한다."""
        if _solution_satisfies(rows_now, bounds_now, relus, assign):
            return assign, True
        return _limit("RELUPLEX_MODEL_INVALID")

    def _apply_relu_fix(tableau: SimplexTableau, x: str, y: str, depth: int) -> Optional[str]:
        """위반된 ReLU 쌍 하나를 tableau에서 제자리로 고친다.

        논문 3절의 Update_f(y := relu(x))와 Update_b(x := y) 중 하나를 적용한다.
        고칠 변수가 기저변수면 먼저 PivotForRelu로 비기저로 빼낸다. 피벗 열은
        |계수|가 가장 큰 변수로 고른다 — "0이 아닌 첫 계수"로 고르면 1e-9짜리
        계수로도 피벗해서 오차가 1e9배 증폭될 수 있다. 피벗이 필요 없는 쪽을
        먼저 시도하고, 목표값이 변수 범위 밖이면 다른 쪽을 시도한다.

        반환: 적용한 규칙 이름, 어느 쪽으로도 고칠 수 없으면 None.
        수정으로 생긴 경계 위반은 호출자가 simplex로 이어서 고친다.
        """
        basic = {row.basic_var for row in tableau.rows}
        before = [tableau.assign[x], tableau.assign[y]] if record_mechanisms else None
        options = [
            ("update_f", y, relu(tableau.assign[x])),
            ("update_b", x, tableau.assign[y]),
        ]
        rng.shuffle(options)
        options.sort(key=lambda option: option[1] in basic)  # 비기저 쪽 먼저 (안정 정렬)
        for rule, var, target in options:
            bound = tableau.bounds[var]
            if (target < bound.lower - _scaled_tol(bound.lower)
                    or target > bound.upper + _scaled_tol(bound.upper)):
                continue
            target = min(max(target, bound.lower), bound.upper)
            if var in basic:
                row = next(r for r in tableau.rows if r.basic_var == var)
                if not row.coeffs:
                    continue
                if numeric_safeguards:
                    column = min(row.coeffs, key=lambda v: (-abs(row.coeffs[v]), v))
                else:   # 안전장치 끔: 계수 크기와 무관하게 이름순 첫 0 아닌 열
                    column = min(
                        (v for v, c in row.coeffs.items() if c != 0.0),
                        default=next(iter(row.coeffs)),
                    )
                if row.coeffs[column] == 0.0:
                    continue
                _pivot(tableau, column, var)
                tableau.pivots_since_check += 1
                _bump("REPAIR_PIVOTS")
            _update_assign(tableau, var, target)
            _record_mechanism("repair", depth, variable=x, output=y, rule=rule,
                              pivoted=var in basic, before=before,
                              after=[tableau.assign[x], tableau.assign[y]])
            return rule
        _record_mechanism("repair", depth, variable=x, output=y, rule="unfixable",
                          pivoted=False, before=before, after=before)
        return None

    def _enforce_fixed(
        fixed_here: Dict[Tuple[str, str], str],
        depth: int,
        current_row_defs: List[Tuple[str, Dict[str, float]]],
        bounds_now: Dict[str, Tuple[float, float]],
        tableau: Optional[SimplexTableau],
        active: List[Tuple[str, str]],
        deps_now: Optional[BoundDeps] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, float]]], List[Tuple[str, str]]]:
        """확정된 ReLU를 분기했을 때와 같은 형태로 강제하고 active에서 뺀다.

        active: y - x = 0 (slack row), inactive: y = 0 (bound). 두 경우 모두 x의
        부호 경계는 derive_bounds가 이미 좁혀 두었다. bounds_now와 tableau는
        제자리에서 고치고, row 목록과 active는 새로 만들어 돌려준다.
        추가한 경계의 원인은 확정 근거 경계의 원인(deps_now.fixed)이다.
        y >= 0은 ReLU의 정의라 원인이 없다.
        """
        current_row_defs = list(current_row_defs)
        new_rows = []
        for (x, y), state in sorted(fixed_here.items()):
            reason = 0
            if deps_now is not None:
                reason = deps_now.fixed.get((x, y), _full(depth))
            if state == "active":
                slack = f"relu_fixed_{x}_on_d{depth}"
                terms = {y: 1.0, x: -1.0}
                current_row_defs.append((slack, terms))
                bounds_now[slack] = (0.0, 0.0)
                new_rows.append((slack, terms, (0.0, 0.0)))
                if deps_now is not None:
                    deps_now.set_lo(slack, reason)
                    deps_now.set_hi(slack, reason)
            else:
                bounds_now[y] = (0.0, 0.0)
                if deps_now is not None:
                    deps_now.set_lo(y, 0)
                    deps_now.set_hi(y, reason)
        if tableau is not None and new_rows:
            _append_rows(tableau, new_rows)
        return current_row_defs, [p for p in active if p not in fixed_here]

    def _under_approx_fix(
        bounds_now: Dict[str, Tuple[float, float]],
        active: List[Tuple[str, str]],
        deps_now: Optional[BoundDeps],
    ) -> Dict[Tuple[str, str], str]:
        """과소근사 (논문 부록 IV): 경계가 0 근처인 ReLU를 강제로 고정한다.

        입력 x의 하한이 [-eps, 0)이면 하한을 0으로 올려 활성으로, x나 출력 y의
        상한이 (0, eps]이면 x의 상한을 0으로 내려 비활성으로 고정한다. 둘 다
        해당하면 남는 구간이 큰 쪽을 고른다. 버리는 것은 폭 eps 이하의 얇은
        구간뿐이지만 그 ReLU의 분기가 통째로 없어진다. bounds_now는 제자리에서
        고치고, 고정할 쌍을 _enforce_fixed 형식으로 돌려준다.
        """
        inf = float('inf')
        fixed: Dict[Tuple[str, str], str] = {}
        for x, y in active:
            x_lo, x_hi = bounds_now.get(x, (-inf, inf))
            y_hi = bounds_now.get(y, (-inf, inf))[1]
            if not (x_lo < 0.0 < x_hi):
                continue
            near_on = x_lo >= -ua_eps
            near_off = min(x_hi, y_hi) <= ua_eps
            if near_on and (not near_off or x_hi >= -x_lo):
                bounds_now[x] = (0.0, x_hi)
                fixed[(x, y)] = "active"
                if deps_now is not None:
                    reason = deps_now.lo.get(x, 0)
                    deps_now.set_lo(x, reason)
                    deps_now.fixed[(x, y)] = reason
            elif near_off:
                bounds_now[x] = (x_lo, 0.0)
                fixed[(x, y)] = "inactive"
                if deps_now is not None:
                    reason = deps_now.hi.get(x, 0) | deps_now.hi.get(y, 0)
                    deps_now.set_hi(x, reason)
                    deps_now.fixed[(x, y)] = reason
        return fixed

    def _rec(
        bounds_now: Dict[str, Tuple[float, float]],
        depth: int,
        current_row_defs: Optional[List[Tuple[str, Dict[str, float]]]] = None,
        warm: Optional[SimplexTableau] = None,
        active: Optional[List[Tuple[str, str]]] = None,
        deps_now: Optional[BoundDeps] = None,
    ) -> Tuple[Optional[Dict[str, float]], bool]:
        """active: 이 노드에서 아직 켜짐/꺼짐이 정해지지 않은 ReLU.


        분기했거나 경계 유도로 확정된 ReLU는 선형 제약(row/bound)으로 강제되어
        있으므로 위반 검사·repair·분기 대상에서 빠진다. 최종 SAT 검산만은
        _validated_sat이 전체 ReLU로 다시 한다.

        deps_now: bounds_now의 경계마다 기대는 분기 마스크. 이 노드가 소유한다
        (호출자는 새 복사본을 넘긴다). UNSAT을 돌려줄 때는 그 원인 마스크를
        last_conflict[0]에 남긴다.
        """
        check_deadline(deadline)
        
        if current_row_defs is None:
            current_row_defs = row_defs
        if active is None:
            active = relus
            
        if depth > max_recursion:
            return _limit("RELUPLEX_RECURSION_LIMIT")
        _bump("SEARCH_NODES")

        bounds_now = dict(bounds_now)
        if track_deps and deps_now is None:
            deps_now = BoundDeps()      # 루트: 아직 분기가 없다
        for _, y in relus:
            lo, hi = bounds_now.get(y, (float('-inf'), float('inf')))
            new_lo = max(0.0, lo)
            
            # [수정된 부분] 모순된 제약(하한이 상한보다 큼) 발생 시 즉시 UNSAT 처리
            if new_lo > hi + 1e-9:
                return _unsat(
                    depth, "BOUNDS",
                    deps_now.lo.get(y, 0) | deps_now.hi.get(y, 0)
                    if deps_now is not None else None,
                )
                
            bounds_now[y] = (new_lo, hi)
            if deps_now is not None and new_lo != lo:
                deps_now.set_lo(y, 0)   # y >= 0은 ReLU의 정의

        # ── 탐색 중 경계 유도 (논문 5절) ──
        # 루트의 경계 유도는 입력 박스만 보고 한 번 돌았다. 분기로 한 ReLU의
        # 부호가 정해지면 그 사실이 다른 뉴런의 범위를 좁혀서 새로 안정이 되는
        # ReLU가 생긴다. 이 연쇄를 쓰지 않으면 이미 부호가 정해진 ReLU까지
        # 계속 분기하게 된다. 루트(depth 0)는 바로 앞에서 같은 계산을 했으므로
        # 건너뛴다.
        if derive_in_search and depth > 0 and active:
            started = time.perf_counter()
            derive_rows = []
            if search_derive_rows != "tableau" or warm is None:
                derive_rows.extend(current_row_defs)
            if search_derive_rows != "original" and warm is not None:
                if core is not None and core.needs_row_error(warm):
                    # GLPK에서 가져온 표 row: row마다 오차 상한을 붙여 넘긴다
                    _bump("SEARCH_ROW_ERROR_CALLS")
                    derive_rows.extend(core.rows_with_error(warm, bounds_now))
                else:
                    derive_rows.extend((row.basic_var, row.coeffs) for row in warm.rows)
            tightened, fixed_here, infeasible = _derive_observed(
                derive_rows, bounds_now, active, deps=deps_now,
                max_passes=search_derive_passes, incremental=derive_incremental,
                observation_depth=depth, observation_source="search",
            )
            _bump("SEARCH_DERIVE_CALLS")
            _bump("SEARCH_DERIVE_SECONDS", time.perf_counter() - started)
            if record_mechanisms:
                changed = sum(bounds_now[v] != bound for v, bound in tightened.items())
                _record_mechanism("bounds", depth, source="search", changed=changed,
                                  fixed=len(fixed_here), conflict=infeasible,
                                  seconds=time.perf_counter() - started)
            if infeasible:
                # 전파만으로 l > u 모순 — 이 가지는 LP를 풀 필요도 없이 UNSAT
                return _unsat(
                    depth, "DERIVE",
                    deps_now.conflict if deps_now is not None else None,
                )
            _record_stable(fixed_here, bounds_now, tightened, depth, "search")
            bounds_now = tightened
            if fixed_here:
                current_row_defs, active = _enforce_fixed(
                    fixed_here, depth, current_row_defs, bounds_now, warm, active,
                    deps_now,
                )
                _bump("SEARCH_DERIVE_FIXED", len(fixed_here))

        # ── 과소근사 단계에서만: 경계가 0 근처인 ReLU를 강제로 고정 ──
        if ua_eps > 0.0 and active:
            ua_before = {x: bounds_now[x] for x, _ in active} if record_stable else None
            ua_fixed = _under_approx_fix(bounds_now, active, deps_now)
            if ua_fixed:
                _record_stable(ua_fixed, ua_before, bounds_now, depth, "under_approx")
                current_row_defs, active = _enforce_fixed(
                    ua_fixed, depth, current_row_defs, bounds_now, warm, active,
                    deps_now,
                )
                _bump("UA_FIXED", len(ua_fixed))
                ua_fixed_total[0] += len(ua_fixed)

        if engine is not None:
            # ── LP를 바깥 엔진에 맡긴다 (lp_engine, LpEngines.py) ──
            # 엔진은 이 노드의 정의식과 경계만 받아 "만족하는 점이 있는가"를 답한다.
            # tableau를 들여다보는 기법(피벗마다 경계 강화, tableau row로 경계 유도,
            # 부모 tableau 물려받기)은 쓰지 않고, 이어 풀기는 엔진이 알아서 한다.
            with span(trace, "simplex", depth=depth):
                assign = _engine_solve(current_row_defs, bounds_now, depth=depth)
            if assign is None:
                return _unsat(depth, "LP", None)

            violations = _check_relu_violations(assign, active)
            if not violations:
                return _validated_sat(assign, current_row_defs, bounds_now)

            # ── ReLU 위반 수정 (Update_f / Update_b 흉내) ──
            # 자체 simplex는 고친 값에서 출발해 이어서 풀지만, 바깥 엔진은 출발점을
            # 받지 않는다. 그래서 고칠 변수를 목표값으로 잠깐 고정(경계 [t, t])하고
            # 다시 푼다 — 고정은 이번 풀이에만 쓰고 노드 경계에는 남기지 않는다.
            # 고정한 문제가 불가능하면 다른 쪽을 시도하고, 둘 다 안 되면 이 쌍은
            # 분기로만 풀린다(자체 simplex에서 목표값이 범위 밖일 때와 같은 처리).
            # 고칠 쌍 고르기, 횟수 세기, branch_tau는 자체 simplex 경로와 같다.
            repair_count: Dict[Tuple[str, str], int] = {}
            repair_unknown_reason = None
            for _ in range(local_repair_max_iter):
                check_deadline(deadline)
                pair = min(violations, key=lambda p: p[0])
                if repair_count.get(pair, 0) >= branch_tau:
                    break
                repair_count[pair] = repair_count.get(pair, 0) + 1
                x_var, y_var = pair
                options = [
                    ("update_f", y_var, relu(assign[x_var])),
                    ("update_b", x_var, assign[y_var]),
                ]
                rng.shuffle(options)
                fixed_assign = None
                with span(trace, "reluplex_repair", depth=depth, branch_x=x_var):
                    for _rule, var, target in options:
                        lo, hi = bounds_now.get(var, (float('-inf'), float('inf')))
                        if (target < lo - _scaled_tol(lo)
                                or target > hi + _scaled_tol(hi)):
                            continue
                        target = min(max(target, lo), hi)
                        pinned = dict(bounds_now)
                        pinned[var] = (target, target)
                        try:
                            fixed_assign = _engine_solve(current_row_defs, pinned, repair=True, depth=depth)
                        except SolverLimitReached as exc:
                            if exc.reason == "TIMEOUT":
                                raise
                            repair_unknown_reason = exc.reason
                            break
                        if fixed_assign is not None:
                            break
                if repair_unknown_reason is not None:
                    break
                if fixed_assign is None:
                    repair_count[pair] = branch_tau
                    _bump("REPAIR_UNFIXABLE")
                    break
                _bump("REPAIR_STEPS")
                assign = fixed_assign
                violations = _check_relu_violations(assign, active)
                if not violations:
                    return _validated_sat(assign, current_row_defs, bounds_now)
            tableau = None
        else:
            if warm is not None:
                # 부모의 basis를 물려받고, 좁아진 bound만 반영해 이어서 푼다.
                # bounds_now 기준으로 한 번 맞춰주므로 부모와 어긋날 여지가 없다.
                tableau = warm
                _sync_bounds(tableau, bounds_now, deps_now)
            else:
                tableau = build_tableau(current_row_defs, bounds_now)
                if deps_now is not None:
                    tableau.deps = deps_now.copy()
            while True:
                with _lp_timer(), span(trace, "simplex", depth=depth):
                    sol, sat = _simplex_observed(
                        tableau,
                        observation_depth=depth,
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
                        stats=stats,
                        tighten_bounds=pivot_tighten,
                        full_tighten_every=full_tighten_every,
                        mechanism_callback=(lambda **values: _lp_events(depth, **values))
                        if record_mechanisms else None,
                        numerical_callback=(lambda **values: _record_mechanism("numerical", depth, **values))
                        if record_mechanisms else None,
                        numeric_safeguards=numeric_safeguards,
                    )

                if not sat:
                    return _unsat(depth, "LP", tableau.conflict)
                if not pivot_tighten or not active:
                    break

                # 피벗하며 좁힌 경계를 노드 경계에 합친다. 자식은 bounds_now를
                # 물려받으므로 여기서 합쳐야 다음 분기 선택과 자식 노드에 쓰인다.
                # 원인 마스크도 tableau에서 같이 가져온다.
                stable_before = {x: bounds_now[x] for x, _ in active} if record_stable else None
                merged = False
                for var, b in tableau.bounds.items():
                    lo, hi = bounds_now.get(var, (float('-inf'), float('inf')))
                    if b.lower > lo or b.upper < hi:
                        bounds_now[var] = (max(lo, b.lower), min(hi, b.upper))
                        if deps_now is not None:
                            if b.lower > lo:
                                deps_now.set_lo(var, tableau.deps.lo.get(var, 0))
                            if b.upper < hi:
                                deps_now.set_hi(var, tableau.deps.hi.get(var, 0))
                        merged = True
                if not merged:
                    break
                # 좁아진 경계로 부호가 정해진 ReLU가 생겼는지 ReLU 규칙만으로 본다.
                # 정해졌는데 강제하지 않으면 그 ReLU는 위반 중이어도 분기 후보가
                # 아니게 되어(_splittable=False) 탐색이 결론 없이 끝난다. 그래서
                # 강제한 뒤 이 노드의 LP를 다시 푼다.
                tightened, fixed_here, infeasible = _derive_observed(
                    [], bounds_now, active, max_passes=1, deps=deps_now,
                    observation_depth=depth, observation_source="pivot",
                )
                if infeasible:
                    return _unsat(
                        depth, "DERIVE",
                        deps_now.conflict if deps_now is not None else None,
                    )
                _record_stable(fixed_here, stable_before, tightened, depth, "pivot")
                _record_mechanism("bounds", depth, source="pivot", changed=0,
                                  fixed=len(fixed_here), conflict=False)
                bounds_now = tightened
                if not fixed_here:
                    break
                current_row_defs, active = _enforce_fixed(
                    fixed_here, depth, current_row_defs, bounds_now, tableau, active,
                    deps_now,
                )
                _sync_bounds(tableau, bounds_now, deps_now)
                _bump("PIVOT_TIGHTEN_FIXED", len(fixed_here))

            assign = sol
            violations = _check_relu_violations(assign, active)
            if not violations:
                return _validated_sat(assign, current_row_defs, bounds_now)

            # ── ReLU 위반 수정 (논문 3절 Update_b / Update_f / PivotForRelu) ──
            # 노드의 tableau를 제자리에서 고치고 Simplex로 경계를 다시 맞춘다.
            # 성공 여부와 무관하게 시도마다 전체 및 뉴런별 수정 횟수를 늘린다.
            #
            # 예전 구현은 (1) tableau를 통째로 복사해 두 방향 모두 simplex를 끝까지
            # 풀고 할당만 가져왔다 — tableau는 원본 그대로라 다음 수정은 다시 처음
            # 상태에서 시작했다. (2) 횟수를 탐색 전체에서 누적하고 초기화하지 않아,
            # 탐색 초반에는 노드마다 repair를 20번까지 하고 포화된 뒤에는 1번만 했다.
            # ACAS Xu 20개(116초)에서 repair가 시간의 29~52%를 썼다.
            repair_count: Dict[Tuple[str, str], int] = {}
            repair_unknown_reason = None
            for _ in range(local_repair_max_iter):
                check_deadline(deadline)
                pair = min(violations, key=lambda p: p[0])
                if repair_count.get(pair, 0) >= branch_tau:
                    break
                repair_count[pair] = repair_count.get(pair, 0) + 1
                with span(trace, "reluplex_repair", depth=depth, branch_x=pair[0]) as repair_event:
                    if _apply_relu_fix(tableau, *pair, depth) is None:
                        # 어느 쪽으로도 범위 안에서 고칠 수 없다 → 분기로만 풀린다
                        repair_count[pair] = branch_tau
                        _bump("REPAIR_UNFIXABLE")
                        if repair_event is not None:
                            repair_event.meta.update(result="UNFIXABLE")
                        break
                    _bump("REPAIR_STEPS")
                    try:
                        with _lp_timer(repair=True), span(trace, "simplex", depth=depth):
                            sol2, sat2 = _simplex_observed(
                                tableau,
                                observation_depth=depth,
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
                                stats=stats,
                                tighten_bounds=pivot_tighten,
                                full_tighten_every=full_tighten_every,
                                mechanism_callback=(lambda **values: _lp_events(depth, **values))
                                if record_mechanisms else None,
                                numerical_callback=(lambda **values: _record_mechanism("numerical", depth, **values))
                                if record_mechanisms else None,
                                numeric_safeguards=numeric_safeguards,
                            )
                    except SolverLimitReached as exc:
                        if exc.reason == "TIMEOUT":
                            raise
                        repair_unknown_reason = exc.reason
                        if repair_event is not None:
                            repair_event.meta.update(result="UNKNOWN", reason=exc.reason)
                        break
                if not sat2:
                    # 수정은 할당만 바꾸고 제약은 그대로이므로, 여기서 UNSAT이면 이
                    # 노드의 LP 자체가 불가능하다는 뜻이다(피벗 강화로 좁힌 경계 포함).
                    return _unsat(depth, "LP", tableau.conflict)
                assign = sol2
                violations = _check_relu_violations(assign, active)
                if not violations:
                    return _validated_sat(assign, current_row_defs, bounds_now)

        # ── 분기 변수(branch_x) 선택 ──
        # 원래 Reluplex 규칙은 "repair로 여러 번 고쳐봤는데 계속 말썽인 ReLU를
        # 분기하라"이다. 그런데 후보를 repair_count에 등록된 쌍 전체에서 골라서
        # 두 가지가 어긋나 있었다:
        #   (a) 지금 위반하지도 않는 ReLU를 분기할 수 있다 — 충돌과 무관한 곳을
        #       쪼개므로 그 분기는 대체로 헛일이다.
        #   (b) 반대로 지금 위반 중인데 repair에 한 번도 안 뽑힌 ReLU는 후보에서
        #       빠진다. 그래서 쪼갤 게 남아있는데도 branch_x=None으로 포기하는
        #       경우가 생긴다.
        # 후보는 "지금 위반 중이면서 아직 고정되지 않은 ReLU"다.
        # 수정 시도 횟수 내림차순, 동점은 이름순으로 선택한다.
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
            with span(
                trace, "reluplex_split", depth=depth, branch_x=branch_x
            ) as split_trace_event:
                lo, hi = bounds_now.get(branch_x, (float('-inf'), float('inf')))
                layer = neuron_key[0] if neuron_key is not None else None
                index = neuron_key[1] if neuron_key is not None else None
                if split_trace_event is not None:
                    split_trace_event.meta["layer"] = layer
                    split_trace_event.meta["index"] = index

                # 이 분기 결정의 비트. 자식(깊이 depth+1)에서 이 결정으로 생긴
                # 경계는 이 비트를 원인으로 갖는다.
                bit = 1 << (depth + 1)

                # 1. x >= 0 분기
                bounds1 = dict(bounds_now)
                bounds1[branch_x] = (max(0.0, lo), hi)
                deps1 = deps_now.copy() if deps_now is not None else None
                if deps1 is not None:
                    deps1.set_lo(branch_x, bit)
                row_defs1 = list(current_row_defs)
                if relu_y is not None:
                    slack_name = f"relu_slack_{branch_x}_pos_{depth}"
                    row_defs1.append((slack_name, {relu_y: 1.0, branch_x: -1.0}))
                    bounds1[slack_name] = (0.0, 0.0)
                    if deps1 is not None:
                        deps1.set_lo(slack_name, bit)
                        deps1.set_hi(slack_name, bit)

                # 분기한 ReLU는 두 자식에서 모두 제약으로 강제되므로 빠진다
                child_active = [p for p in active if p != (branch_x, relu_y)]

                branch_unknown_reason = None
                c1 = c2 = None      # 두 자식이 UNSAT일 때의 원인 마스크
                split_id = None
                if split_logger is not None:
                    # global_split_count 증가 및 '+' 기록: 첫 Reluplex 재귀 호출 직전.
                    split_id = split_logger.begin(branch_x, layer, index)
                warm1 = None
                if warm_start and engine is None:
                    new_rows1 = []
                    if relu_y is not None:
                        new_rows1.append(
                            (slack_name, {relu_y: 1.0, branch_x: -1.0}, (0.0, 0.0))
                        )
                    warm1 = _child_tableau(tableau, new_rows1)

                try:
                    with _branch_span(
                        branch_x, layer, index, phase="+", depth=depth + 1,
                        bound=f"[{max(0.0, lo)}, {hi}]", split_id=split_id,
                        split_event=split_trace_event,
                    ) as (active_branch_event, active_branch_log_id):
                        try:
                            r1, sat1 = _rec(
                                bounds1, depth + 1, row_defs1, warm1, child_active,
                                deps1,
                            )
                            c1 = last_conflict[0]
                            origin1 = last_unsat_origin[0] if not sat1 else None
                        except SolverLimitReached as exc:
                            if active_branch_event is not None:
                                active_branch_event.meta["result"] = (
                                    "TIMEOUT" if exc.reason == "TIMEOUT" else "UNKNOWN"
                                )
                                active_branch_event.meta["reason"] = exc.reason
                            if active_branch_log_id is not None:
                                split_logger.branch_end(
                                    active_branch_log_id,
                                    result=("TIMEOUT" if exc.reason == "TIMEOUT" else "UNKNOWN"),
                                    reason=exc.reason,
                                )
                                active_branch_log_id = None
                            if exc.reason == "TIMEOUT":
                                raise
                            branch_unknown_reason = exc.reason
                            r1, sat1 = None, False
                        else:
                            if active_branch_event is not None:
                                active_branch_event.meta["result"] = (
                                    "SAT" if sat1 else "UNSAT"
                                )
                                active_branch_event.meta["unsat_origin"] = origin1
                            if active_branch_log_id is not None:
                                split_logger.branch_end(
                                    active_branch_log_id,
                                    result=("SAT" if sat1 else "UNSAT"),
                                    unsat_origin=origin1,
                                )
                                active_branch_log_id = None
                    if sat1:
                        return r1, True
                    # 자식이 끝났으므로 참조를 놓아준다 (depth만큼 tableau가
                    # 동시에 살아있지 않도록)
                    warm1 = None

                    # ── 백점프 (논문 4절 충돌 분석) ──
                    # 첫째 가지를 닫은 모순이 이 분기(bit)에 기대지 않으면, 그
                    # 모순은 둘째 가지에서도 그대로 성립한다(기대는 경계가 모두
                    # 이 분기보다 위에서 정해졌으므로). 그래서 둘째 가지를 풀지
                    # 않고 같은 원인으로 닫는다. 부모도 같은 검사를 하므로 무관한
                    # 층이 여럿이면 한꺼번에 되돌아간다.
                    backjumped = backjump and c1 is not None and not (c1 & bit)

                    # 2. x <= 0 분기
                    bounds2 = dict(bounds_now)
                    bounds2[branch_x] = (lo, min(0.0, hi))
                    deps2 = deps_now.copy() if deps_now is not None else None
                    if deps2 is not None:
                        deps2.set_hi(branch_x, bit)
                    row_defs2 = list(current_row_defs)
                    inactive_bounds_conflict = False
                    if relu_y is not None:
                        y_lo, y_hi = bounds2.get(
                            relu_y, (float('-inf'), float('inf'))
                        )
                        inactive_bounds_conflict = max(y_lo, 0.0) > min(y_hi, 0.0) + 1e-9
                        if not inactive_bounds_conflict:
                            bounds2[relu_y] = (max(y_lo, 0.0), min(y_hi, 0.0))
                            if deps2 is not None:
                                if y_lo < 0.0:
                                    deps2.set_lo(relu_y, 0)     # y >= 0은 ReLU의 정의
                                if y_hi > 0.0:
                                    deps2.set_hi(relu_y, bit)   # y = 0은 이 분기 결정

                    with _branch_span(
                        branch_x, layer, index, phase="-", depth=depth + 1,
                        bound=f"[{lo}, {min(0.0, hi)}]", split_id=split_id,
                        split_event=split_trace_event,
                    ) as (inactive_branch_event, inactive_branch_log_id):
                        if backjumped and not check_backjump:
                            r2, sat2 = None, False
                            c2 = c1
                            _bump("BACKJUMPS")
                            _record_mechanism(
                                "backjump", depth + 1, variable=branch_x,
                                from_depth=depth + 1, target_depth=max(0, c1.bit_length() - 1),
                                cause_depths=[i for i in range(1, depth + 1) if c1 & (1 << i)],
                                skipped_measure=2.0 ** -(depth + 1),
                            )
                            _bump("BACKJUMP_SKIPPED_MEASURE", 2.0 ** -(depth + 1))
                            _bump("CLOSED_MEASURE", 2.0 ** -(depth + 1))
                            if inactive_branch_event is not None:
                                inactive_branch_event.meta["result"] = "UNSAT"
                                inactive_branch_event.meta["reason"] = "BACKJUMP"
                                inactive_branch_event.meta["unsat_origin"] = "skipped"
                            if inactive_branch_log_id is not None:
                                split_logger.branch_end(
                                    inactive_branch_log_id,
                                    result="UNSAT",
                                    reason="BACKJUMP",
                                    unsat_origin="skipped",
                                )
                                inactive_branch_log_id = None
                        elif inactive_bounds_conflict:
                            r2, sat2 = None, False
                            _closed(depth + 1, "BOUNDS")
                            if deps_now is not None:
                                c2 = (
                                    deps_now.lo.get(relu_y, 0)
                                    | deps_now.hi.get(relu_y, 0)
                                    | bit
                                )
                            if inactive_branch_event is not None:
                                inactive_branch_event.meta["result"] = "UNSAT"
                                inactive_branch_event.meta["reason"] = "BOUNDS_CONFLICT"
                                inactive_branch_event.meta["unsat_origin"] = "found"
                            if inactive_branch_log_id is not None:
                                split_logger.branch_end(
                                    inactive_branch_log_id,
                                    result="UNSAT",
                                    reason="BOUNDS_CONFLICT",
                                    unsat_origin="found",
                                )
                                inactive_branch_log_id = None
                        else:
                            warm2 = (
                                _child_tableau(tableau, [])
                                if warm_start and engine is None else None
                            )
                            try:
                                r2, sat2 = _rec(
                                    bounds2, depth + 1, row_defs2, warm2, child_active,
                                    deps2,
                                )
                                c2 = last_conflict[0]
                                origin2 = last_unsat_origin[0] if not sat2 else None
                            except SolverLimitReached as exc:
                                if inactive_branch_event is not None:
                                    inactive_branch_event.meta["result"] = (
                                        "TIMEOUT" if exc.reason == "TIMEOUT" else "UNKNOWN"
                                    )
                                    inactive_branch_event.meta["reason"] = exc.reason
                                if inactive_branch_log_id is not None:
                                    split_logger.branch_end(
                                        inactive_branch_log_id,
                                        result=("TIMEOUT" if exc.reason == "TIMEOUT" else "UNKNOWN"),
                                        reason=exc.reason,
                                    )
                                    inactive_branch_log_id = None
                                if exc.reason == "TIMEOUT":
                                    raise
                                branch_unknown_reason = branch_unknown_reason or exc.reason
                                r2, sat2 = None, False
                            else:
                                if inactive_branch_event is not None:
                                    inactive_branch_event.meta["result"] = (
                                        "SAT" if sat2 else "UNSAT"
                                    )
                                    inactive_branch_event.meta["unsat_origin"] = origin2
                                if inactive_branch_log_id is not None:
                                    split_logger.branch_end(
                                        inactive_branch_log_id,
                                        result=("SAT" if sat2 else "UNSAT"),
                                        unsat_origin=origin2,
                                    )
                                    inactive_branch_log_id = None
                    if backjumped and check_backjump:
                        # 검증 모드: 백점프가 건너뛰었을 가지를 실제로 풀어 봤다
                        if sat2:
                            raise AssertionError(
                                f"백점프가 SAT 가지를 건너뛸 뻔했다: depth={depth}, "
                                f"branch_x={branch_x}, 원인={c1:b}"
                            )
                        _bump(
                            "BACKJUMP_CHECKED" if branch_unknown_reason is None
                            else "BACKJUMP_CHECK_INCONCLUSIVE"
                        )
                    if sat2:
                        return r2, True

                    if branch_unknown_reason is not None:
                        return _limit(branch_unknown_reason)
                    # 이 노드를 닫은 원인: 한쪽 가지의 원인이 이 분기와 무관하면
                    # 그것만으로 충분하고, 아니면 두 원인을 합쳐 이 분기를 뺀다.
                    if not track_deps:
                        conflict = None
                    elif backjumped:
                        conflict = c1
                    elif c2 is not None and not (c2 & bit):
                        conflict = c2
                    elif c1 is not None and c2 is not None:
                        conflict = (c1 | c2) & ~bit
                    else:
                        conflict = _full(depth)
                    last_conflict[0] = conflict
                    last_unsat_origin[0] = "propagated"
                    return None, False
                finally:
                    if split_logger is not None and split_id is not None:
                        # 두 번째 재귀 호출 이후(또는 조기 반환/예외 시) count 감소.
                        split_logger.end(split_id, branch_x, layer, index)

        if depth >= max_recursion:
            return _limit("RELUPLEX_RECURSION_LIMIT")
        return _limit(repair_unknown_reason or "RELUPLEX_REPAIR_INCONCLUSIVE")

    def _under_approx_phase(
        bounds0: Dict[str, Tuple[float, float]],
        rows0: List[Tuple[str, Dict[str, float]]],
    ) -> Tuple[Optional[Dict[str, float]], bool, bool]:
        """과소근사를 켠 채 탐색한다. 반환: (해, sat, 결과가 정확한가).

        보통은 (해, True)만 믿을 수 있다. 다만 탐색 내내 한 번도 강제로 고정하지
        않았다면 원래 탐색과 똑같은 탐색을 끝까지 한 것이므로 UNSAT도 정확하다
        (세 번째 값 True) — 이때는 원래 탐색을 다시 하지 않는다.

        시간은 남은 시간의 under_approx_share만 쓴다. _rec와 simplex는 deadline을
        이 함수의 바깥 변수로 읽으므로, 잠시 앞당겼다가 끝나면 되돌린다.
        """
        nonlocal ua_eps, deadline
        saved_deadline = deadline
        started = time.perf_counter()
        if deadline is not None:
            now = time.monotonic()
            deadline = now + under_approx_share * max(0.0, deadline - now)
        ua_eps = under_approx
        ua_fixed_total[0] = 0
        try:
            model, sat = _rec(bounds0, 0, rows0)
            _bump("UA_RESULT_SAT" if sat else "UA_RESULT_UNSAT")
            return model, sat, ua_fixed_total[0] == 0
        except SolverLimitReached as exc:
            # 과소근사 단계의 시간 초과나 한도는 원래 탐색으로 넘어갈 이유일 뿐이다
            _bump(f"UA_RESULT_{exc.reason}")
            return None, False, False
        finally:
            ua_eps = 0.0
            deadline = saved_deadline
            _bump("UA_SECONDS", time.perf_counter() - started)
            if stats is not None:
                stats["UA_NODES"] = stats.get("SEARCH_NODES", 0)

    # [누락되었던 부분 복구] reluplex 함수의 마지막 반환문!
    # ── 경계 유도 (논문 4절 Tighter Bound Derivation) ──
    # 탐색을 시작하기 전에 구간 전파로 변수 경계를 좁히고, 그 결과 항상 켜짐/
    # 항상 꺼짐이 확정된 ReLU를 선형 제약으로 바꿔 넣는다. 확정된 ReLU는
    # relus 목록에서 빠지므로 위반 검사·repair·분기 대상에서 모두 제외된다.
    # 이게 없으면 모든 ReLU의 pre-activation bound가 (-inf, inf)라
    # _splittable()이 항상 True를 돌려주고, 안정인 뉴런까지 전부 분기 후보가 된다.
    active_row_defs = row_defs
    active_bounds = dict(bounds)
    if derive_bound and relus:
        tightened, fixed, infeasible = _derive_observed(row_defs, bounds, relus,
                                                       incremental=derive_incremental,
                                                       observation_depth=0, observation_source="initial")
        if record_mechanisms:
            ranges = []
            for x, y in relus:
                key = (relu_metadata or {}).get((x, y)) or _parse_neuron_var(x)
                lo, hi = tightened[x]
                ranges.append({"variable": x, "output": y, "layer": key[0] if key else None,
                               "index": key[1] if key else None,
                               "before": [v if isfinite(v) else None for v in bounds[x]],
                               "after": [v if isfinite(v) else None for v in (lo, hi)],
                               "state": "active" if lo >= 0 else "inactive" if hi <= 0 else "unstable"})
            _record_mechanism("bounds", 0, source="initial",
                              changed=sum(bounds[v] != value for v, value in tightened.items()),
                              fixed=len(fixed), conflict=infeasible, ranges=ranges)
        if stats is not None:
            stats['DERIVE_FIXED_RELUS'] = len(fixed)
            stats['DERIVE_REMAINING_RELUS'] = original_relu_count - len(fixed)
        if infeasible:
            # 전파만으로 l > u 모순이 드러났다 — 분기할 것도 없이 UNSAT
            if stats is not None:
                stats['DERIVE_INFEASIBLE'] = stats.get('DERIVE_INFEASIBLE', 0) + 1
            _closed(0, "DERIVE")
            return None, False
        _record_stable(fixed, bounds, tightened, 0, "initial")
        if fixed:
            active_row_defs, active_bounds = apply_fixed_relus(
                row_defs, tightened, fixed
            )
            # relus 자체를 다시 묶는다. _rec와 _check_relu_violations가 이 이름을
            # 클로저로 참조하므로, 확정된 ReLU는 위반 검사·repair·분기에서 모두 빠진다.
            relus = [p for p in relus if p not in fixed]
        else:
            active_bounds = tightened

    # ── 과소근사 단계 (논문 부록 IV) ──
    # 경계가 0 근처인 ReLU를 강제로 고정한 좁은 문제를 남은 시간의 일부 동안
    # 먼저 푼다. 반례를 찾으면 그것이 원래 문제의 반례다. 불가능이 나오거나
    # 시간을 다 쓰면 그 결과는 버리고 원래 문제를 처음부터 푼다. 단, 한 번도
    # 강제로 고정하지 않고 끝났다면 그 탐색이 곧 원래 탐색이므로 결과를 쓴다
    # (sat_relu처럼 0 근처 경계가 없는 문제에서 같은 탐색을 두 번 하지 않는다).
    if lp_engine != "native" and lp_mode == "core":
        # 카츠 구조: native 경로 그대로, Simplex.simplex 자리만 엔진이 맡는다
        core = make_core(lp_engine, full_tighten_every=full_tighten_every,
                         glpk_extract=glpk_extract, glpk_row_error=glpk_row_error)
    elif lp_engine != "native":
        # 엔진은 이 호출 동안 LP 하나를 계속 들고 경계만 바꿔 이어서 푼다.
        # 자체 simplex 엔진(통제군)에는 native 경로와 같은 simplex 설정을 넘기되
        # tableau 결합 기법(피벗마다 경계 강화)은 끈다.
        engine = make_engine(lp_engine, {
            "max_iter": simplex_max_iter,
            "debug": debug,
            "report_unknown": report_unknown,
            "stats": stats,
            "numeric_safeguards": numeric_safeguards,
        })
    try:
        if under_approx > 0.0 and relus:
            model, sat, exact = _under_approx_phase(active_bounds, active_row_defs)
            if sat or exact:
                if exact:
                    _bump("UA_EXACT")
                return model, sat
            if stats is not None:
                stats['CLOSED_MEASURE'] = 0.0
            if seed is not None:
                rng.seed(seed)      # 원래 탐색은 과소근사를 안 켰을 때와 같은 경로로

        return _rec(active_bounds, 0, active_row_defs)
    finally:
        used = engine if engine is not None else core
        if used is not None and lp_engine == "glpk":
            used.close()
        if used is not None and stats is not None:
            stats["LP_ENGINE"] = f"{used.name}-{lp_mode}"
            stats["LP_ENGINE_FAILURES"] = (
                stats.get("LP_ENGINE_FAILURES", 0) + getattr(used, "failures", 0)
            )
            if hasattr(used, "check_seconds"):
                stats["Z3_CHECK_SECONDS"] = (
                    stats.get("Z3_CHECK_SECONDS", 0.0) + used.check_seconds
                )
            if hasattr(used, "pivots"):
                stats["GLPK_PIVOTS"] = stats.get("GLPK_PIVOTS", 0) + used.pivots
            if hasattr(used, "conflicts_found"):
                stats["ENGINE_CONFLICTS_FOUND"] = (
                    stats.get("ENGINE_CONFLICTS_FOUND", 0) + used.conflicts_found
                )


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
