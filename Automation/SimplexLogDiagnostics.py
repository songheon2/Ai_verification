"""Bounded, snapshot-only Simplex comparisons; peers expose aggregates, not raw logs."""
import json
import math
from statistics import median

from Automation.LogAnalysisKnowledge import implementation_reference
from Automation.LpTelemetry import metric_available


PROFILE_LIMIT = 20000
MECHANISM_KINDS = ("bound_updates", "bounds", "conflict", "backjump", "repair")


def attach_prior_mechanisms(profiles, events):
    """Attach only timing and counts, with no raw events from outside the selection."""
    events = iter(events)
    event = next(events, None)
    latest = {}
    for profile in sorted(profiles, key=lambda p: p.get("start_ns") or 0):
        start = profile.get("start_ns")
        if start is None or profile.get("round") is None:
            continue
        while event is not None and event["time_ns"] <= start:
            latest.setdefault(event.get("round_index", event.get("round")), {})[event["kind"]] = event
            event = next(events, None)
        nearby = latest.get(profile["round"], {})
        profile["prior_mechanisms"] = [
            {"kind": kind, "seconds_before": round((start - item["time_ns"]) / 1e9, 6),
             "source": item.get("source"), "depth": item.get("depth"),
             "changes": len(item.get("changes", [])) if kind == "bound_updates" else item.get("changed")}
            for kind, item in nearby.items() if start - item["time_ns"] <= 10_000_000_000
        ]


def _number(value):
    try:
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def call_profiles(index, args=None, limit=PROFILE_LIMIT, *, include_prior=False):
    args = args or {}
    where, parameters = [], []
    if "call_id" in args:
        if not isinstance(args["call_id"], str) or len(args["call_id"]) > 100:
            raise ValueError("Invalid Simplex call ID")
        where.append("e.call_id=?")
        parameters.append(args["call_id"])
    if "call_number" in args:
        if type(args["call_number"]) is not int or not 1 <= args["call_number"] <= 10**9:
            raise ValueError("call_number must be an explicitly recorded positive call number")
        where.append("COALESCE(CASE WHEN json_valid(e.raw) THEN json_extract(e.raw,'$.call_number') END,"
                     "CASE WHEN json_valid(s.raw) THEN json_extract(s.raw,'$.call_number') END)=?")
        parameters.append(args["call_number"])
    if "round" in args:
        if type(args["round"]) is not int or args["round"] < 1:
            raise ValueError("Invalid round")
        where.append("COALESCE(e.round,s.round)=?")
        parameters.append(args["round"])
    predicate = " AND " + " AND ".join(where) if where else ""
    rows = index.db.execute(
        "SELECT e.file,e.line,e.call_id,COALESCE(e.round,s.round),COALESCE(e.depth,s.depth),"
        "e.duration,e.pivots,e.raw,s.raw FROM events e LEFT JOIN events s ON s.rowid="
        "(SELECT MIN(t.rowid) FROM events t WHERE t.call_id=e.call_id AND t.event='simplex_start') "
        "WHERE e.event IN ('simplex_end','simplex_call') AND (e.call_id IS NULL OR NOT EXISTS "
        "(SELECT 1 FROM events t WHERE t.call_id=e.call_id AND t.event IN ('simplex_end','simplex_call') "
        "AND t.rowid>e.rowid))" + predicate + " ORDER BY e.line LIMIT ?", [*parameters, limit + 1])
    profiles = []
    for file, line, call, rnd, depth, duration, pivots, raw, start in rows:
        end = json.loads(raw)
        start_event = json.loads(start) if start else {}
        metadata = {**start_event, **end}
        profile = {"call_id": call, "round": rnd, "depth": depth,
                   "call_number": metadata.get("call_number") if type(metadata.get("call_number")) is int and metadata["call_number"] > 0 else None,
                   "origin": metadata.get("origin") if isinstance(metadata.get("origin"), str) else None,
                   "seconds": _number(duration), "pivots": _number(pivots) if metric_available(metadata, "pivots") else None,
                   "iterations": _number(metadata.get("iterations")) if metric_available(metadata, "iterations") else None,
                   "work_counts_available": metadata.get("work_counts_available", True),
                   "pivot_count_available": metric_available(metadata, "pivots"),
                   "iteration_count_available": metric_available(metadata, "iterations"),
                   "metrics_source": metadata.get("metrics_source"),
                   "bound_violation_scope": metadata.get("bound_violation_scope"),
                   "model_validation": metadata.get("model_validation"),
                   "glpk_profile": metadata.get("glpk_profile"),
                   "start_ns": start_event.get("time_ns", end.get("start_ns")),
                   "row_count": _number(metadata.get("row_count")),
                   "variable_count": _number(metadata.get("variable_count")),
                   "cycles": _number(metadata.get("cycles")) if metric_available(metadata, "cycles") else None,
                   "result": end.get("result"), "evidence": f"{index.run_id}:{file}:{line}"}
        profile["call_average_seconds_per_pivot"] = (
            profile["seconds"] / profile["pivots"] if profile["seconds"] is not None and profile["pivots"] else None)
        profile["call_average_seconds_per_iteration"] = (
            profile["seconds"] / profile["iterations"] if profile["seconds"] is not None and profile["iterations"] else None)
        profiles.append(profile)
    if include_prior and profiles:
        marks = ','.join('?' for _ in MECHANISM_KINDS)
        rows = index.db.execute("SELECT raw FROM events WHERE event='relu_mechanism' AND time_ns IS NOT NULL "
                                f"AND kind IN ({marks}) ORDER BY time_ns,line", MECHANISM_KINDS)
        attach_prior_mechanisms(profiles, (json.loads(raw) for raw, in rows))
    return {"profiles": profiles[:limit], "limited": len(profiles) > limit}


def _distribution(profiles, field):
    values = [p[field] for p in profiles if p.get(field) is not None]
    return {"samples": len(values), "median": median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def diagnose_simplex(index, args=None, reference=None):
    args = args or {}
    if set(args) - {"run_id", "call_id", "call_number", "round"}:
        raise ValueError("Only Simplex call ID/number and round filters are supported")
    targets = call_profiles(index, args)
    reference = reference if reference is not None else call_profiles(index, include_prior=True)
    history = {p["evidence"]: p.get("prior_mechanisms") for p in reference["profiles"]}
    selected = sorted(targets["profiles"], key=lambda p: p["seconds"] or 0, reverse=True)[:3]
    comparisons = []
    for target in selected:
        peers = [p for p in reference["profiles"] if p["evidence"] != target["evidence"]]
        # Prefer matching phases/rounds; fallbacks are explicitly labelled, never hidden.
        same_round = [p for p in peers if target["round"] is not None and p["round"] == target["round"]]
        same_origin = [p for p in peers if target["origin"] is not None and p["origin"] == target["origin"]]
        matched = [p for p in same_round if target["origin"] is not None and p["origin"] == target["origin"]]
        group, basis = next(((p, name) for p, name in (
            (matched, "same_round_and_origin"), (same_round, "same_round"),
            (same_origin, "same_origin_across_rounds"), (peers, "indexed_run_completed_calls")) if p), ([], "no_peers"))
        target["prior_mechanisms"] = history.get(target["evidence"])
        fields = ("seconds", "iterations", "pivots", "call_average_seconds_per_iteration",
                  "call_average_seconds_per_pivot", "row_count", "variable_count")
        distributions = {field: _distribution(group, field) for field in fields}
        ordered = sorted((p for p in group if p["seconds"] is not None), key=lambda p: p["seconds"])
        middle = len(ordered) // 2
        median_peers = [{key: p[key] for key in ("call_number", "seconds", "iterations", "depth")}
                        for p in ordered[max(0, middle-1):middle+2]]
        ratios = {field: target[field] / distributions[field]["median"]
                  for field in fields if target[field] is not None and distributions[field]["median"]}
        samples = []
        cycle_events = 0
        if target["call_id"]:
            samples = [{"variable": variable, "sampled_selections": count} for variable, count in index.db.execute(
                "SELECT variable,COUNT(*) FROM events WHERE call_id=? AND event='simplex_iteration' "
                "AND variable IS NOT NULL GROUP BY variable ORDER BY COUNT(*) DESC LIMIT 5", (target["call_id"],))]
            cycle_events = index.db.execute("SELECT COUNT(*) FROM events WHERE call_id=? AND event='simplex_cycle'",
                                            (target["call_id"],)).fetchone()[0]
        comparisons.append({"target": target,
            "comparison": {"basis": basis, "calls": len(group), "distribution": distributions,
                           "target_to_peer_median_ratio": ratios, "median_duration_calls": median_peers,
                           "reference_limited": reference["limited"]},
            "sampled_bound_selections": samples, "recorded_cycle_events": cycle_events})
    return {"comparisons": comparisons, "selected_call_list_limited": targets["limited"] or len(targets["profiles"]) > 3,
            "interpretation": [
                "다른 호출은 비교용 집계만 제공된다. 선택 범위 밖의 원시 이벤트는 조회할 수 없다.",
                "호출 시간이 긴 원인을 피벗 총량 증가와 호출 평균 비용 증가로 나눠 비교한다. 수치 재진술은 원인 분석이 아니다.",
                "평균 비용은 전체 호출 시간/피벗 수이며 순수 피벗 함수 시간이 아니다. 피벗 0회 호출은 이 비율에서 제외한다.",
                "항목별 *_count_available을 우선 확인한다. 미수집은 0회가 아니다. call_boundary의 경계 위반은 호출 시점 관측이며 내부 반복 위반 샘플과 합산하지 않는다.",
                "비교군이 작거나 호출 목적·라운드가 다르면 비교의 한계를 밝힌다. 각 지표의 중앙값은 별도로 계산하므로 비율들을 항상 곱셈으로 분해할 수 있는 것은 아니다.",
                "변수 수가 비슷한데 평균 비용이 높으면 변수 수만으로 원인을 단정하지 않는다. 행 밀도·복원·함수별 비용은 기록된 경우에만 실제 원인으로 제시한다.",
                "반복 위반은 샘플 기록이며 실제 전체 위반 횟수가 아니다. 순환 로그 부재도 미발생의 증거는 아니다.",
                "현재 스냅샷에서 완료된 호출끼리의 비교다. 진행 중 호출의 경과시간은 최종 소요시간과 비교하지 않는다.",
                "호출 순번은 명시적인 call_number만 사용하고 call_id의 숫자에 1을 더해 만들지 않는다.",
            ], "implementation_reference": implementation_reference("simplex")}
