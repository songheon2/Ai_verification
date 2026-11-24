"""Curated implementation reference, not evidence about a historical execution."""
from copy import deepcopy
from math import isfinite


_DPLL_TERMS = {
    "DPLL Round": "Boolean 솔버가 후보를 찾고, 후보가 있으면 이론 솔버로 검사하는 한 차례다. 라운드 시간에는 이론 검사와 그 안의 LP 호출이 포함된다.",
    "selected atoms": "현재 후보가 참·거짓으로 정한 선형 부등식과 ReLU 조건이다. 이 조건들이 해당 라운드의 이론 검사 입력을 구성한다.",
    "Theory Solver Result": "현재 후보의 선형 제약과 ReLU 관계를 함께 검사한 결과다. LP 호출 결과, 최종 검증 판정과 각각 구분한다.",
    "THEORY_CONFLICT": "현재 후보가 이론 제약을 만족할 수 없어 그 조건 조합을 차단하고 다음 후보를 찾는다.",
    "BOOLEAN_UNSAT": "차단 조건을 포함한 Boolean 문제에 남은 후보가 없다. 전체 검증은 UNSAT다.",
    "THEORY_SAT": "현재 후보의 선형 제약과 ReLU 관계를 만족하는 해를 찾았다. 최종 반례 보고에는 VNNLIB 조건 검증이 필요하다.",
    "THEORY_TRIVIAL": "후보에 검사할 이론 조건이 없어 이론 검사를 바로 통과했다.",
    "UNKNOWN": "설정된 한도나 솔버 종료 사유로 판정을 확정하지 못했다. 종료 사유를 함께 확인한다.",
}


def solver_prompt_context():
    dpll = "\n".join(f"{name} · {meaning}" for name, meaning in _DPLL_TERMS.items())
    return f"""검증기 해석에 필요한 기본 정보

DPLL(T)
{dpll}

LP와 Simplex
LP(선형 제약 문제) 호출은 현재 선형 등식·부등식과 변수 경계를 만족할 수 있는지 검사한다. LP SAT 이후에도 ReLU 관계를 검사한다.
z는 ReLU 입력, h는 출력 max(0,z), slack은 부등식을 등식과 변수 경계로 나타내기 위한 보조변수다.
자체 Simplex 피벗은 기저변수를 교체하는 연산이다. GLPK 내부 반복 카운터와 기저 피벗 횟수는 별도 지표다.
호출 경계에서 관측한 위반은 진입 시점의 할당과 경계를 비교한 값이다. 내부 반복의 위반은 솔버가 남긴 샘플 기록을 사용한다.

ReLUplex와 ReLU 탐색
ReLUplex는 DPLL이 선택한 조건을 검사하는 이론 솔버다. 탐색 노드에서 LP로 선형 제약을 풀고, 가능한 할당이 나오면 h = max(0,z)를 검사한다.
ReLU 관계를 위반하면 값을 국소 수정하고 필요시 LP로 선형 제약을 다시 맞춘다. 수정 예산을 소진했거나 관계를 해결하지 못하면 분기 가능한 ReLU를 골라 활성·비활성으로 split한다. 분기 기록만으로 해당 뉴런의 수정 시도 여부를 알 수 없다.
LP나 경계 유도에서 모순을 찾으면 현재 가지를 닫는다. 선형 제약과 ReLU 관계를 모두 만족하는 해를 찾으면 현재 DPLL 후보를 이론 SAT로 반환한다.
분기 방문은 각 가지로 진입한 기록이고 깊이는 루트에서 해당 노드까지의 분기 단계 수다. 열린 상위 가지는 하위 탐색 결과를 기다릴 수 있다.
STABLE은 경계로 ReLU의 활성 또는 비활성 상태가 확정된 것이다.

검증 기법
경계 유도는 제약으로 변수 범위를 좁히고 ReLU 확정이나 모순을 찾는다. GLPK 내부 전체 경계 강화는 별도 계측에서 확인한다.
충돌 분석은 모순에 기여한 분기 조건을 추적하고, 백점프는 그 조건을 바탕으로 탐색 단계를 건너뛴다.
자체 Simplex의 수치 안전장치는 드리프트를 검사하고 필요하면 타블로를 복원한다. 외부 엔진의 수치 처리와 기록 범위는 엔진별로 다르다.
과소근사는 일부 ReLU 상태를 고정한 좁은 문제에서 반례를 먼저 찾는다. 상태를 강제로 고정한 탐색의 UNSAT는 원래 문제의 UNSAT를 확정하는 근거로 쓰지 않는다.

사용자 질문과 관련된 용어만 답변에 사용하고, 실제 기법 적용 여부와 수치는 해당 실행 기록으로 확인하세요."""


_REFERENCE = {
    "dpll": [
        "DPLL이 선택한 조건을 ReLUplex가 검사한다. 이론 충돌로 학습한 절이 다음 Boolean 탐색을 제한한다.",
        "라운드, 이론 검사, LP 호출의 시간은 포함 관계이므로 각 시간을 그대로 더하면 중복 계산된다.",
    ],
    "simplex": [
        "Simplex는 현재 선형 제약과 변수 경계를 만족시키는 할당을 찾는다. ReLU 관계의 만족 여부는 별도로 검사한다.",
        "기저변수의 경계 위반을 선택해 비기저변수 값을 이동하고 피벗한다. 한 변수의 수정이 다른 변수의 값을 바꿀 수 있다.",
        "네이티브 Simplex는 피벗에서 다른 행의 계수를 대입·재작성하고, 할당 갱신과 피벗 후 기저값 재계산에서 전체 행을 순회한다. 외부 LP 엔진의 동작은 runtime_reference에 기록된 엔진을 기준으로 해석한다.",
        "소요시간은 피벗 횟수뿐 아니라 행·계수 수, 경계 강화, 수치 검사/복원 등 호출 안의 작업에 좌우된다. 변수 수가 같으면 그 숫자만으로 특정 호출의 급증을 설명할 수 없다.",
        "호출 시간/피벗 수는 호출 전체를 피벗 수로 나눈 평균 비용 지표이며 순수 피벗 함수의 측정 시간이 아니다.",
        "순환은 같은 기저·할당 상태의 재방문을 감지해 무작위 피벗으로 벗어나려 한다. 피벗이 많다는 이유만으로 순환을 단정하지 않는다.",
        "호출별 행 밀도·세부 함수 소요시간은 현재 로그에 없을 수 있다. 전체 행 처리 비용이라는 구현상 후보와 그 호출에서 실제 발생한 병목을 구분한다.",
    ],
    "relu": [
        "선형 제약을 Simplex로 맞춘 뒤 ReLU 관계 위반을 검사한다. 관계 수정과 선형 제약 복구를 시도한 뒤 필요하면 활성·비활성으로 분기한다.",
        "열린 상위 분기는 하위 탐색 결과를 기다릴 수 있다. 파란 테두리 여러 개가 뉴런들의 병렬 실행을 뜻하지 않는다.",
        "수정 예산과 분기 선택은 구현 버전·설정에 따라 달라질 수 있으므로 현재 참고 설명을 과거 실행에 소급하지 않는다.",
    ],
    "bounds": [
        "등식과 현재 변수 경계로 다른 변수의 하한·상한을 좁힌다. ReLU 입력 하한이 0 이상이면 활성, 상한이 0 이하이면 비활성을 확정할 수 있다.",
        "STABLE 확정은 경계로 상태가 정해진 기록이며 탐색 분기가 아니다. 경계 변경, STABLE 확정, 경계 모순을 구별한다.",
        "루트·탐색 노드의 경계 유도와 피벗 이후 경계 강화는 서로 다른 실행 지점이다. 같은 뉴런의 서로 다른 가지 기록을 하나의 연속 경계로 합치지 않는다.",
    ],
    "tree": [
        "탐색 트리의 각 노드는 한 ReLU에 대한 활성(+) 또는 비활성(-) 가지다. 동일 뉴런도 다른 경로·라운드에서 여러 노드로 등장할 수 있다.",
        "직접 SAT·UNSAT를 발견한 지점과 하위 결과를 전달받아 CLOSED로 종료한 상위 가지를 구별한다. UNSAT 원인 기록의 found·propagated·skipped는 직접 발견·결과 전달·탐색 생략을 뜻한다.",
        "분기 ID와 화면의 노드 번호·Split 번호는 서로 다른 식별자다. 화면 번호를 입증할 자료가 없으면 R번호·레이어·뉴런·분기 조건으로 위치를 설명한다.",
        "열린 상위 가지는 하위 탐색 결과를 기다릴 수 있다. 여러 RUNNING 가지가 병렬 탐색을 의미하지 않는다.",
        "가지의 소요시간은 하위 탐색을 포함하는 중첩 시간이다. 여러 노드의 소요시간을 합쳐 전체 실행시간이나 독점 비용으로 제시하지 않는다.",
        "STABLE 확정은 분기를 추가한 것이 아니다. 확정 이력의 누적 건수와 서로 다른 뉴런 수를 구분한다.",
        "부모 ID·충돌 원인 연결이 기록되지 않았으면 깊이와 시간만으로 정확한 인과·부모 관계를 단정하지 않는다. 라운드·필터·로그 누락 범위를 먼저 고려한다.",
    ],
    "backjump": [
        "충돌에 기여한 분기 조건의 의존성을 분석해 원인과 관계없는 깊이를 건너뛰어 되돌아간다.",
        "일반적인 되돌아감이나 탐색 종료만으로 백점프가 실행됐다고 판단하지 않는다. 설정과 실제 백점프 기록을 구분한다.",
    ],
    "numeric": [
        "행의 계수 크기에 비례한 임계치로 너무 작은 피벗 후보를 피하고, 주기적으로 원래 등식에 대한 드리프트를 확인한다. 필요하면 초기 등식으로 타블로를 복원한다.",
        "피벗 계수나 영점 임계치의 증가는 누적 오차의 측정치가 아니다. 계수의 스케일 변화와 측정된 드리프트·복원 여부를 분리한다.",
        "검사가 활성화됐다는 설정은 드리프트 발생이나 복원을 입증하지 않는다. 특정 Simplex 호출과 연결할 식별자가 없으면 시간·깊이만으로 수치 이벤트를 그 호출의 원인으로 귀속하지 않는다.",
    ],
    "underapprox": [
        "현재 구현은 0 근처 경계의 ReLU 일부를 활성·비활성으로 강제 고정해 좁아진 문제에서 반례를 먼저 찾는다. 모든 변수 경계를 일괄적으로 epsilon만큼 좁히는 구현은 아니다.",
        "입력 하한이 -epsilon 이상이면 0으로 올려 활성으로, 입력 또는 출력 상한이 epsilon 이하이면 입력 상한을 0으로 내려 비활성으로 고정한다. 양쪽이 가능하면 남는 구간이 큰 쪽을 택한다.",
        "그 부분집합에서 찾은 SAT 해는 원래 문제의 해다. 강제 고정이 있었다면 UNSAT만으로 원래 문제의 UNSAT를 결론내리지 않고 원래 탐색으로 돌아간다.",
        "강제 고정이 전혀 없었다면 원래 탐색과 같으므로 그 결과를 재사용할 수 있다. 사용 가능한 시간 중 일부만 사전 탐색에 배정하며 한도 종료도 원래 탐색으로 돌아갈 이유다.",
        "꺼져 있어도 원래 제약의 검증 단계가 빠진 것은 아니다. 사전 반례 탐색을 생략한 것이며 켰을 때의 속도 개선은 비교 실행 없이 보장할 수 없다.",
    ],
}


def implementation_reference(topic):
    if topic in ("all", "mechanisms"):
        names = ("bounds", "backjump", "numeric", "underapprox") if topic == "mechanisms" else tuple(_REFERENCE)
        principles = [f"{name}: {line}" for name in names for line in
                      (_REFERENCE[name][:2] if topic == "all" else _REFERENCE[name])]
    else:
        principles = _REFERENCE.get(topic, _REFERENCE["dpll"])
    return {"topic": topic, "basis": "현재 저장소 구현을 정리한 참고 설명",
            "historical_execution_match": "확인되지 않음: 과거 실행의 구현·활성 상태를 증명하지 않음",
            "principles": deepcopy(principles)}


def feature_configuration(topic, settings, context):
    """Only explicit recorded values; never fill missing flags from today's defaults."""
    if topic in ("all", "mechanisms"):
        return {"state": "per_feature", "features": {
            name: feature_configuration(name, settings, context)
            for name in ("bounds", "backjump", "numeric", "underapprox")}}
    keys = {"underapprox": ("underapprox_enabled", "underapprox_epsilon", "under_approx"),
            "numeric": ("numeric_safeguards",), "backjump": ("backjump",),
            "bounds": ("derive_bound", "derive_in_search", "pivot_tighten")}.get(topic, ())
    if not keys:
        return {"state": "not_applicable", "state_origin": "analysis_derived",
                "meaning": "이 항목은 기법의 켜짐/꺼짐 설정 대상이 아니다. 검증 판정 UNKNOWN을 뜻하지 않는다.",
                "reason": "non_feature_topic", "recorded_settings": [], "run_options": {}}
    observations = []
    for record in settings:
        values = {key: record[key] for key in keys if key in record}
        if values:
            observations.append({"round": record.get("round", record.get("round_index")), "values": values})
    solver = context.get("solver") if isinstance(context.get("solver"), dict) else {}
    options = solver.get("reluplex_options") if isinstance(solver.get("reluplex_options"), dict) else {}
    configured = {key: options[key] for key in keys if key in options}
    invalid_values = []
    def enabled(values):
        if topic == "underapprox":
            if type(values.get("underapprox_enabled")) is bool:
                return values["underapprox_enabled"]
            for key in ("underapprox_epsilon", "under_approx"):
                if type(values.get(key)) in (int, float) and isfinite(values[key]):
                    return values[key] > 0
        flags = [v for v in values.values() if type(v) is bool]
        if topic == "bounds" and not any(flags) and len(flags) < len(keys):
            return None
        return any(flags) if flags else None
    for source, values in [("run_options", configured)] + [
            (f"round:{item['round']}", item["values"]) for item in observations]:
        for key, value in values.items():
            numeric = key in {"underapprox_epsilon", "under_approx"}
            valid = (type(value) in (int, float) and isfinite(value) and value >= 0) if numeric else type(value) is bool
            if not valid:
                invalid_values.append({"source": source, "field": key, "value": value})
    states = {enabled(item["values"]) for item in observations}
    if configured:
        states.add(enabled(configured))
    states.discard(None)
    state = "mixed" if len(states) > 1 else ("enabled" if True in states else "disabled" if False in states else "unknown")
    recorded_keys = set(configured) | {key for item in observations for key in item["values"]}
    required = ("underapprox_enabled",) if topic == "underapprox" else keys
    missing = [key for key in required if key not in recorded_keys]
    reason = ("conflicting_recorded_values" if state == "mixed" else
              "invalid_recorded_values" if invalid_values else
              "no_recorded_feature_fields" if not recorded_keys else
              "partial_feature_fields" if state == "unknown" else "explicit_recorded_values")
    meanings = {
        "enabled": "기록된 설정에서 활성화가 확인됨. 실제 실행·효과는 발생 기록으로 별도 확인한다.",
        "disabled": "기록된 설정에서 비활성화가 확인됨. 검증 실패를 의미하지 않는다.",
        "mixed": "라운드별 설정 또는 실행 옵션과 실제 설정 기록이 서로 다름. 한 상태로 단정하지 않는다.",
        "unknown": "분석기가 기록된 설정만으로 적용 여부를 확정하지 못해 만든 상태다. 솔버가 UNKNOWN 판정을 내렸다는 뜻도, 원시 로그에 state=unknown이 있었다는 뜻도 아니다.",
    }
    return {"state": state, "recorded_settings": observations, "run_options": configured,
            "state_origin": "analysis_derived", "reason": reason,
            "known_fields": sorted(recorded_keys), "missing_fields": missing,
            "invalid_values": invalid_values, "meaning": meanings[state],
            "diagnostic_conclusion": {
                "no_recorded_feature_fields": "분석에 제공된 기록에 해당 기법의 설정 항목이 없어 적용 상태가 미확인으로 표시됐다. 저장 누락인지 선택 범위·수집 한도 때문인지는 기록 범위를 확인한다.",
                "partial_feature_fields": "일부 단계의 설정만 있다. 예를 들어 탐색 중 경계 유도가 꺼졌어도 초기 경계 유도·피벗 강화까지 꺼졌다고 볼 수 없다.",
                "invalid_recorded_values": "설정값의 자료형이나 범위가 기대한 형식과 다르다. 단순 기록 누락과 구별해 저장·전달 경로를 점검해야 한다.",
                "conflicting_recorded_values": "저장 옵션과 라운드 설정이 다르거나 라운드마다 설정이 바뀌었다. 충돌하는 값을 확인해야 한다.",
                "explicit_recorded_values": meanings[state],
            }[reason]}


def runtime_reference(settings, context):
    """Recorded engine identity plus explicitly labelled current implementation knowledge."""
    solver = context.get("solver") if isinstance(context.get("solver"), dict) else {}
    options = solver.get("reluplex_options") if isinstance(solver.get("reluplex_options"), dict) else {}
    runtime_keys = ("lp_engine", "lp_mode", "numeric_event_support")
    recorded = {key: options[key] for key in runtime_keys if key in options}
    observations = [{"round": item.get("round", item.get("round_index")),
                     "values": {key: item[key] for key in runtime_keys if key in item}}
                    for item in settings if any(key in item for key in runtime_keys)]
    variants = {value for value in (solver.get("variant"), context.get("solver_variant")) if isinstance(value, str)}
    engines = {item["values"]["lp_engine"] for item in observations if isinstance(item["values"].get("lp_engine"), str)}
    if isinstance(recorded.get("lp_engine"), str):
        engines.add(recorded["lp_engine"])
    for variant in variants:
        engine, _, mode = variant.partition("-")
        if variant == "simplex-original":
            engines.add("native")
        elif mode.split("-", 1)[0] in ("core", "oracle"):
            engines.add(engine)
    descriptions = {
        "native": "Python 자체 tableau 코어. 수치 안전장치가 켜져 있으면 계수 처리와 주기적 드리프트 검사를 수행한다.",
        "simplex": "기존 Python Simplex를 외부 LP oracle 인터페이스로 호출하는 비교군이다. 네이티브 경로와 tableau 결합·계측 범위가 다를 수 있다.",
        "z3": "선형 제약은 정확한 유리수 연산으로 푼다. Python Simplex의 부동소수 피벗·드리프트 콜백을 실행하는 경로가 아니므로 수치 발생 기록이 없을 수 있다.",
        "glpk": "GLPK의 부동소수 LP 엔진을 쓴다. GLPK 자체 수치 처리와 Python Simplex의 드리프트 콜백은 다르며 core/oracle 및 추출 방식도 구분해야 한다.",
    }
    return {"recorded_engine_candidates": sorted(engines), "recorded_options": recorded,
            "recorded_variants": sorted(variants), "round_observations": observations,
            "identity_state": "conflicting" if len(engines) > 1 else "recorded" if engines else "not_recorded",
            "implementation_basis": "현재 구현 설명이며 과거 실행 코드가 같았다는 증거는 아님",
            "implementation_notes": [descriptions.get(engine, "이 엔진에 대한 구현 참고 설명 없음") for engine in sorted(engines)],
            "missing_identity_meaning": "엔진 기록이 없으면 현재 기본 코어를 과거 실행에 대입하지 않는다."}


def project_structure_reference():
    return {
        "verification_flow": [
            "검증기에서 ONNX 모델과 VNNLIB 속성, LP 엔진 및 제한 시간을 선택한다. 지원하는 ReLU 모델과 속성을 선형 제약과 Boolean 조건으로 변환한다.",
            "DPLL(T)가 Boolean 조건을 선택하고, ReLUplex가 주어진 조건의 선형 제약과 ReLU 관계를 검사한다.",
            "기본 Simplex, GLPK, Z3는 선형 제약의 가능성을 검사하는 경로다. 최종 SAT는 반례가 검증됐을 때 COUNTEREXAMPLE로 보고하고, UNSAT는 VERIFIED로 보고한다. 시간 제한이나 판정 불가 조건은 UNKNOWN으로 남을 수 있다.",
            "외부 LP 엔진의 Core 모드는 ReLUplex 타블로의 선형 풀이에 결합한다. Oracle 모드는 현재 등식과 변수 경계를 넘겨 해가 있는지 확인한다.",
            "실시간과 피드백 시각화는 DPLL 라운드, LP 호출, ReLU 분기와 탐색 트리, Reluplex 기법을 보여 준다. 통합 LLM은 기록 또는 구현 설명으로 질문에 답한다.",
        ],
        "components": implementation_reference("all")["principles"],
    }


def analysis_support(topic, settings, context, event_counts, files):
    """Small causal context, kept separate from potentially huge raw-log summaries."""
    def compact(config):
        return {key: ({name: compact(item) for name, item in value.items()} if key == "features" else value)
                for key, value in config.items() if key not in {"recorded_settings", "run_options"}}
    observed = [{"event": row.get("event"), "kind": row.get("kind"), "count": row["count"]}
                for row in event_counts if row.get("kind") != "settings"]
    solver = context.get("solver") if isinstance(context.get("solver"), dict) else {}
    return {"configuration": compact(feature_configuration(topic, settings, context)),
            "verification_result": {"status": context.get("status", solver.get("status")), "reason": solver.get("reason")},
            "context_provenance": context.get("context_provenance", {}),
            "field_reference": {"derive_bound": "탐색 전 초기 경계 유도", "derive_in_search": "탐색 노드의 경계 유도",
                                "pivot_tighten": "LP 내부 경계 강화 요청 (엔진마다 지원 방식이 다름)",
                                "numeric_safeguards": "자체 Simplex 수치 안전장치 요청 (외부 엔진의 자체 수치 처리와 별개)",
                                "backjump": "원인 의존성에 따른 백점프 요청", "under_approx": "과소근사 epsilon; 0이면 비활성"},
            "runtime_reference": runtime_reference(settings, context),
            "state_domains": {
                "configuration.unknown": "분석기가 계산한 기법 적용 설정의 미확인 상태. 솔버 UNKNOWN 판정과 별개다.",
                "solver.UNKNOWN": "검증에서 SAT/UNSAT을 확정하지 못한 결과. 종료 사유·시간·반복 한도를 확인한다.",
                "relu.unstable": "ReLU 경계가 0을 걸쳐 활성·비활성이 아직 확정되지 않은 상태. 오류가 아니다.",
            },
            "observed_event_counts_in_selection": observed,
            "coverage": [{key: item[key] for key in ("file", "missing", "truncated", "invalid_lines", "records") if key in item} for item in files],
            "interpretation": "적용 설정·실제 발생·기록 가능 여부·측정된 효과를 구분한다. unknown의 생성 주체와 빠진 항목을 짚어 답하고 정보 부족이라는 말만 반복하지 않는다."}
