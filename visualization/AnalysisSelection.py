"""Click targets for rendered charts, computed from the plotted snapshot only."""
import json
from pathlib import Path

from Automation.LpTelemetry import metric_available
from visualization.AtomicFile import write_atomic_text


def select_axis(figure, axis, entries, *, positions=None, direction="x", label=""):
    geometry = getattr(figure, "_analysis_selection_geometry", [])
    figure._analysis_selection_geometry = geometry
    if entries:
        geometry.append((axis, entries, positions, direction, label))


def select_text(figure, artist, entry, *, label):
    geometry = getattr(figure, "_analysis_text_geometry", [])
    geometry.append((artist, entry, label))
    figure._analysis_text_geometry = geometry


def write_selection_map(figure, image_path, *, cropped=False):
    geometry = getattr(figure, "_analysis_selection_geometry", None)
    texts = getattr(figure, "_analysis_text_geometry", None)
    if geometry is None and texts is None:
        return
    bounds = figure.bbox
    if cropped:
        from matplotlib import rcParams
        bounds = figure.get_tightbbox(figure.canvas.get_renderer()).padded(
            rcParams["savefig.pad_inches"]).transformed(figure.dpi_scale_trans)
    def point(x, y):
        return [(float(x) - bounds.x0) / bounds.width, 1 - (float(y) - bounds.y0) / bounds.height]
    regions, items, item_ids, entry_ids = [], [], {}, {}
    def add_item(entry):
        if id(entry) in entry_ids:
            return entry_ids[id(entry)]
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if key not in item_ids:
            item_ids[key] = len(items)
            items.append(entry)
        entry_ids[id(entry)] = item_ids[key]
        return item_ids[key]
    for artist, entry, label in texts or []:
        area = artist.get_window_extent(figure.canvas.get_renderer())
        if artist.get_clip_on() and artist.axes is not None:
            from matplotlib.transforms import Bbox
            area = Bbox.intersection(area, artist.axes.get_window_extent())
            if area is None:
                continue
        left, top = point(area.x0, area.y1)
        right, bottom = point(area.x1, area.y0)
        regions.append(dict(box=[left, top, right, bottom], direction="x", positions=[],
                            items=[add_item(entry)], label=label))
    for axis, entries, positions, direction, label in geometry or []:
        area = axis.get_window_extent()
        left, top = point(area.x0, area.y1)
        right, bottom = point(area.x1, area.y0)
        indices = [add_item(entry) for entry in entries]
        coordinates = []
        if positions is not None:
            for value in positions:
                x, y = axis.transData.transform((value, 0) if direction == "x" else (0, value))
                coordinates.append(point(x, y)[0 if direction == "x" else 1])
        regions.append(dict(box=[left, top, right, bottom], direction=direction,
                            positions=coordinates, items=indices, label=label))
    path = Path(image_path).with_suffix(".selection.js")
    payload = json.dumps(dict(items=items, regions=regions), ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return write_atomic_text(path, "window.receiveAnalysisMap?.(document.currentScript," + payload + ");")


def solver_selection(figure, axes, feedback, *, simplex):
    duration, work, results, detail = axes.flat
    if simplex:
        from visualization.SolverProgress import LP_ENGINE_LABELS, _constraint_label, _simplex_panel_engine, bound_violation_items

        calls = feedback["simplex_calls"]
        panel_engine = _simplex_panel_engine(feedback)
        entries = []
        for number, call in enumerate(calls, 1):
            scope = {"topic": "simplex"}
            if call.get("call_id"):
                scope["call_id"] = str(call["call_id"])
            if call.get("round") is not None:
                scope["round"] = int(call["round"])
            engine = call.get("lp_engine") or panel_engine
            engine_label = LP_ENGINE_LABELS.get(engine, engine)
            call_label = "LP" if engine and engine != "native" else "Simplex"
            lines = [f"그래프 {number}번째 {call_label} 호출", f"호출 ID: {call.get('call_id', '미기록')}"]
            pivot_label = "Z3 산술 피벗 횟수" if engine == "z3" else "피벗 횟수"
            iteration_label = "GLPK 내부 반복 횟수" if engine == "glpk" else "반복 횟수"
            for key, label in (("round", "DPLL Round"), ("result", "결과"),
                               ("duration_seconds", "소요 시간(초)"), ("pivots", pivot_label),
                               ("iterations", iteration_label), ("depth", "탐색 깊이"),
                               ("row_count", "선형 제약 수"), ("variable_count", "변수 수")):
                if key in {"pivots", "iterations"} and not metric_available(call, key):
                    continue
                if call.get(key) is not None:
                    lines.append(f"{label}: {call[key]}")
            if engine_label:
                lines.append(f"LP 엔진: {engine_label}")
            if engine == "glpk" and metric_available(call, "iterations"):
                lines.append("반복 횟수는 LP 호출 전후 GLPK 내부 카운터의 차이입니다.")
            if engine == "z3" and metric_available(call, "pivots"):
                lines.append("산술 피벗 횟수는 Z3 솔버 통계에서 가져옵니다.")
            if call.get("bound_violation_scope") == "call_boundary":
                lines.append("경계 위반은 LP 호출 진입 시 확인한 할당값을 기준으로 집계합니다.")
            if call.get("result", "RUNNING") == "RUNNING":
                lines.append("진행 중 스냅샷: 소요 시간은 완료 시간이 아닌 현재 경과 시간입니다.")
            entries.append(dict(scope=scope, text="\n".join(lines)))
        for axis, label in ((duration, "Simplex · 호출별 소요 시간"), (work, "Simplex · " + work.get_title())):
            select_axis(figure, axis, entries, positions=range(1, len(calls) + 1), label=label)
        counts = {}
        for call in calls:
            result = call.get("result", "RUNNING")
            counts[result] = counts.get(result, 0) + 1
        select_axis(figure, results, [dict(scope={"topic": "simplex"}, text=f"Simplex 결과 {key}: {value}회")
                                    for key, value in counts.items()], positions=range(len(counts)), label="Simplex · 호출 결과")
        violations, violation_scope = bound_violation_items(feedback)
        violations = list(reversed(violations[:8]))
        count_label = "내부 반복 검사 샘플" if violation_scope == "solver_iteration" else "LP 호출 진입 시 관측"
        detail_label = "경계 위반 반복 변수" if violation_scope == "solver_iteration" else "LP 호출 진입 시 경계 위반"
        select_axis(figure, detail, [dict(scope={"topic": "simplex"}, text=f"{_constraint_label(item)}\n{count_label}: {item.get('count', 0)}회")
                                    for item in violations], positions=range(len(violations)), direction="y", label="Simplex · " + detail_label)
    else:
        rounds = feedback["rounds"][-12:]
        entries = [dict(scope={"topic": "dpll", "round": int(item["round"])}, text="\n".join(
            f"{label}: {item[key]}" for key, label in (("round", "DPLL Round"), ("result", "결과"),
            ("duration_seconds", "소요 시간(초)"), ("simplex_calls", "Simplex 호출 수"),
            ("theory_result", "Theory Solver Result"), ("selected_atoms", "선택된 원자 수")) if item.get(key) is not None)) for item in rounds]
        for axis, label in ((duration, "DPLL · 라운드별 소요 시간"), (work, "DPLL · 라운드별 Simplex 호출 수")):
            select_axis(figure, axis, entries, positions=[item["round"] for item in rounds], label=label)
        from visualization.SolverProgress import _theory_outcome, _compact_theory_flow_lines
        counts = {}
        for item in rounds:
            outcome = _theory_outcome(item)
            if outcome is not None:
                counts[outcome] = counts.get(outcome, 0) + 1
        select_axis(figure, results, [dict(scope={"topic": "dpll"}, text=f"Theory Solver Result · {key}: {value}회")
                                    for key, value in counts.items()], positions=range(len(counts)), direction="y", label="DPLL · Theory Solver Result")
        lines = _compact_theory_flow_lines(feedback, rounds)
        if lines:
            area = detail.texts[0].get_window_extent(figure.canvas.get_renderer())
            centers = [detail.transData.inverted().transform((area.x0, area.y1 - (index + 0.5) * area.height / len(lines)))[1]
                       for index in range(len(lines))]
            select_axis(figure, detail, [dict(scope=entry["scope"], text=entry["text"] + "\n" + line)
                                        for entry, line in zip(entries, lines)], positions=centers, direction="y",
                        label="DPLL · 라운드 및 이론 흐름")


def split_selection(axis, events, positions, round_starts):
    from bisect import bisect_right
    starts = sorted((int(item["time_ns"]), int(item["round"])) for item in round_starts
                    if item.get("time_ns") is not None and item.get("round") is not None)
    stamps = [stamp for stamp, _ in starts]
    entries = []
    for event in events:
        scope = {"topic": "relu"}
        if event.layer is not None and event.index is not None:
            scope.update(layer=event.layer, neuron=event.index)
        index = bisect_right(stamps, event.time_ns) - 1
        if index >= 0:
            scope["round"] = starts[index][1]
        text = f"ReLU 분기 {'시작' if event.event == '+' else '종료'}\nLayer {event.layer if event.layer is not None else '미기록'} · Neuron {event.index if event.index is not None else '미기록'}"
        if index >= 0:
            text += f"\nDPLL Round: {scope['round']}"
        if event.timestamp:
            text += f"\n기록 시각: {event.timestamp}"
        entries.append(dict(scope=scope, text=text))
    select_axis(axis.figure, axis, entries, positions=positions, label="ReLU · 활성 분기 이력")
