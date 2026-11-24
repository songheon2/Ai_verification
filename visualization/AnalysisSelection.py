"""Click targets for rendered charts, computed from the plotted snapshot only."""
import json
from functools import lru_cache
from math import isfinite, sqrt
from pathlib import Path

from Automation.LpTelemetry import metric_available
from visualization.AtomicFile import write_atomic_text


def select_axis(figure, axis, entries, targets, *, label=""):
    geometry = getattr(figure, "_analysis_selection_geometry", [])
    figure._analysis_selection_geometry = geometry
    if entries:
        geometry.append((axis, entries, targets, label))


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
    x_min, y_min = bounds.x0, bounds.y0
    width, height = bounds.width, bounds.height
    def point(x, y):
        return [round((float(x) - x_min) / width, 8),
                round(1 - (float(y) - y_min) / height, 8)]

    def normalized_shape(kind, coords):
        if kind == "box":
            x0, y0, x1, y1 = coords
            return [kind, *point(min(x0, x1), max(y0, y1)), *point(max(x0, x1), min(y0, y1))]
        if kind == "stroke":
            x0, y0, x1, y1, radius = coords
            return [kind, *point(x0, y0), *point(x1, y1),
                    round(radius / width, 8), round(radius / height, 8)]
        return [kind, *[value for x, y in coords for value in point(x, y)]]

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
    for axis, entries, shapes, label in geometry or []:
        area = axis.get_window_extent()
        left, top = point(area.x0, area.y1)
        right, bottom = point(area.x1, area.y0)
        targets = [[index, *normalized_shape(kind, coords)] for index, kind, coords in shapes()]
        regions.append(dict(box=[left, top, right, bottom], targets=targets,
                            items=[add_item(entry) for entry in entries], label=label))
    path = Path(image_path).with_suffix(".selection.js")
    payload = json.dumps(dict(items=items, regions=regions), ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return write_atomic_text(path, "window.receiveAnalysisMap?.(document.currentScript," + payload + ");")


def _bar_shapes(axis, *, labels_cover_bars=False):
    if not axis.containers:
        return
    for index, patch in enumerate(axis.containers[0].patches):
        if patch.get_height() and patch.get_width():
            yield index, "box", patch.get_window_extent().extents
    if labels_cover_bars:
        for label in axis.texts:
            patch = label.get_bbox_patch()
            if patch is not None:
                yield -1, "box", patch.get_window_extent().extents


def _stroke(axis, start, end, width):
    return (*start, *end, width * axis.figure.dpi / 144)


def _cycle_shapes(axis, collection):
    if collection is None:
        return
    # The cycle markers are upward triangles in the PathCollection.
    triangle = collection.get_paths()[0].vertices[:3]
    sizes = collection.get_sizes()
    for index, (x, y) in enumerate(collection.get_offsets()):
        center = axis.transData.transform((x, y))
        scale = sqrt(sizes[index % len(sizes)]) * axis.figure.dpi / 72
        yield int(x) - 1, "triangle", center + triangle * scale


def _duration_shapes(axis):
    import numpy as np

    lines = axis.collections[0]
    width = lines.get_linewidths()[0]
    indices, segments = [], []
    for index, path in enumerate(lines.get_paths()):
        segment = path.vertices
        if len(segment) == 2 and segment[0, 1] != segment[1, 1] and all(isfinite(v) for v in segment[:, 1]):
            indices.append(index)
            segments.append(segment)
    if segments:
        pixels = axis.transData.transform(np.asarray(segments).reshape(-1, 2)).reshape(-1, 2, 2)
        for index, (start, end) in zip(indices, pixels):
            yield index, "stroke", _stroke(axis, start, end, width)
    yield from _cycle_shapes(axis, axis.collections[1] if len(axis.collections) > 1 else None)


def _work_shapes(axis):
    import numpy as np

    line = axis.lines[0]
    # The work curve has gaps wherever the engine did not report a count.
    points = axis.transData.transform(line.get_xydata())
    width = line.get_linewidth()
    finite = np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(finite[:-1] & finite[1:])
    midpoints = (points[indices] + points[indices + 1]) / 2
    for index, middle in zip(indices.tolist(), midpoints):
        yield index, "stroke", _stroke(axis, points[index], middle, width)
        yield index + 1, "stroke", _stroke(axis, middle, points[index + 1], width)
    radius = line.get_markersize() * axis.figure.dpi / 144
    for index in range(0, len(points), line.get_markevery()):
        x, y = points[index]
        if isfinite(x) and isfinite(y):
            yield index, "box", (x - radius, y - radius, x + radius, y + radius)
    yield from _cycle_shapes(axis, axis.collections[0] if axis.collections else None)


def _flow_text_shapes(axis, lines):
    renderer = axis.figure.canvas.get_renderer()
    artist = axis.texts[0]
    area = artist.get_window_extent(renderer)
    rows = [(index, row) for index, line in enumerate(lines) for row in line.split("\n")]
    row_height = area.height / len(rows)
    for row_index, (index, line) in enumerate(rows):
        width, height, _ = renderer.get_text_width_height_descent(line, artist.get_fontproperties(), False)
        middle = area.y1 - (row_index + 0.5) * row_height
        yield index, "box", (area.x0, middle - height / 2, area.x0 + width, middle + height / 2)


@lru_cache(maxsize=8192)
def _simplex_call_entry(number, call_id, call_label_id, round_index, engine, fields,
                        has_iter_count, has_pivot_count, bounds_at_entry, running):
    from visualization.SolverProgress import LP_ENGINE_LABELS

    scope = {"topic": "simplex"}
    if call_id:
        scope["call_id"] = str(call_id)
    if round_index is not None:
        scope["round"] = int(round_index)
    call_label = "LP" if engine and engine != "native" else "Simplex"
    lines = [f"그래프 {number}번째 {call_label} 호출", f"호출 ID: {call_label_id}"]
    lines.extend(f"{label}: {value}" for label, value in fields)
    if engine_label := LP_ENGINE_LABELS.get(engine, engine):
        lines.append(f"LP 엔진: {engine_label}")
    if engine == "glpk" and has_iter_count:
        lines.append("반복 횟수는 LP 호출 전후 GLPK 내부 카운터의 차이입니다.")
    if engine == "z3" and has_pivot_count:
        lines.append("산술 피벗 횟수는 Z3 솔버 통계에서 가져옵니다.")
    if bounds_at_entry:
        lines.append("경계 위반은 LP 호출 진입 시 확인한 할당값을 기준으로 집계합니다.")
    if running:
        lines.append("진행 중 스냅샷: 소요 시간은 완료 시간이 아닌 현재 경과 시간입니다.")
    return dict(scope=scope, text="\n".join(lines))


def solver_selection(figure, axes, feedback, *, simplex, flow_lines=None):
    duration, work, results, detail = axes.flat
    if simplex:
        from visualization.SolverProgress import _constraint_label, _simplex_panel_engine, bound_violation_items

        calls = feedback["simplex_calls"]
        panel_engine = _simplex_panel_engine(feedback)
        entries = []
        for number, call in enumerate(calls, 1):
            engine = call.get("lp_engine") or panel_engine
            available = {field: metric_available(call, field) for field in ("pivots", "iterations")}
            pivot_label = "Z3 산술 피벗 횟수" if engine == "z3" else "피벗 횟수"
            iteration_label = "GLPK 내부 반복 횟수" if engine == "glpk" else "반복 횟수"
            fields = []
            for key, label in (("round", "DPLL Round"), ("result", "결과"),
                               ("duration_seconds", "소요 시간(초)"), ("pivots", pivot_label),
                               ("iterations", iteration_label), ("depth", "탐색 깊이"),
                               ("row_count", "선형 제약 수"), ("variable_count", "변수 수")):
                if not available.get(key, True):
                    continue
                if call.get(key) is not None:
                    fields.append((label, str(call[key])))
            entries.append(_simplex_call_entry(
                number, call.get("call_id"), str(call.get("call_id", "미기록")), call.get("round"), engine, tuple(fields),
                available["iterations"], available["pivots"],
                call.get("bound_violation_scope") == "call_boundary", call.get("result", "RUNNING") == "RUNNING",
            ))
        select_axis(figure, duration, entries, lambda: _duration_shapes(duration), label="Simplex · 호출별 소요 시간")
        select_axis(figure, work, entries, lambda: _work_shapes(work), label="Simplex · " + work.get_title())
        counts = {}
        for call in calls:
            result = call.get("result", "RUNNING")
            counts[result] = counts.get(result, 0) + 1
        select_axis(figure, results, [dict(scope={"topic": "simplex"}, text=f"Simplex 결과 {key}: {value}회")
                                    for key, value in counts.items()], lambda: _bar_shapes(results), label="Simplex · 호출 결과")
        violations, violation_scope = bound_violation_items(feedback)
        violations = list(reversed(violations[:8]))
        count_label = "내부 반복 검사 샘플" if violation_scope == "solver_iteration" else "LP 호출 진입 시 관측"
        detail_label = "경계 위반 반복 변수" if violation_scope == "solver_iteration" else "LP 호출 진입 시 경계 위반"
        select_axis(figure, detail, [dict(scope={"topic": "simplex"}, text=f"{_constraint_label(item)}\n{count_label}: {item.get('count', 0)}회")
                                    for item in violations], lambda: _bar_shapes(detail), label="Simplex · " + detail_label)
    else:
        rounds = feedback["rounds"][-12:]
        entries = [dict(scope={"topic": "dpll", "round": int(item["round"])}, text="\n".join(
            f"{label}: {item[key]}" for key, label in (("round", "DPLL Round"), ("result", "결과"),
            ("duration_seconds", "소요 시간(초)"), ("simplex_calls", "Simplex 호출 수"),
            ("theory_result", "Theory Solver Result"), ("selected_atoms", "선택된 원자 수")) if item.get(key) is not None)) for item in rounds]
        select_axis(figure, duration, entries, lambda: _bar_shapes(duration, labels_cover_bars=True),
                    label="DPLL · 라운드별 소요 시간")
        select_axis(figure, work, entries, lambda: _bar_shapes(work, labels_cover_bars=True),
                    label="DPLL · 라운드별 Simplex 호출 수")
        from visualization.SolverProgress import _theory_outcome, _compact_theory_flow_lines
        counts = {}
        for item in rounds:
            outcome = _theory_outcome(item)
            if outcome is not None:
                counts[outcome] = counts.get(outcome, 0) + 1
        select_axis(figure, results, [dict(scope={"topic": "dpll"}, text=f"Theory Solver Result · {key}: {value}회")
                                    for key, value in counts.items()], lambda: _bar_shapes(results), label="DPLL · Theory Solver Result")
        lines = _compact_theory_flow_lines(feedback, rounds) if flow_lines is None else flow_lines
        if lines:
            select_axis(figure, detail, [dict(scope=entry["scope"], text=entry["text"] + "\n" + line)
                                        for entry, line in zip(entries, lines)], lambda: _flow_text_shapes(detail, lines),
                        label="DPLL · 라운드 및 이론 흐름")


def split_selection(axis, events, xs, ys, round_starts):
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
    def shapes():
        import numpy as np

        width = axis.lines[0].get_linewidth()
        segments = list(zip(xs[1:-1], xs[2:], ys, ys[1:]))
        if not segments:
            return
        points = [point for x, next_x, before, after in segments
                  for point in ((x, before), (x, after), (next_x, after), (x, 0))]
        pixels = axis.transData.transform(np.asarray(points)).reshape(-1, 4, 2)
        for index, ((x, next_x, before, after), (start, corner, end, base)) in enumerate(zip(segments, pixels)):
            if after and x < next_x:
                yield index, "box", (*base, *end)
            if x < next_x:
                yield index, "stroke", _stroke(axis, corner, end, width)
            if before != after:
                yield index, "stroke", _stroke(axis, start, corner, width)

    select_axis(axis.figure, axis, entries, shapes, label="ReLU · 활성 분기 이력")
