"""Build a self-contained, interactive ReLU branch search-tree report."""
from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right
from collections import Counter, defaultdict
from html import escape
import json
import os
from pathlib import Path
import threading
from time import sleep
from typing import Any, Dict, Iterable, List, Optional

from visualization.NetworkLayout import parse_neuron_var
from visualization.SolveTrace import SolveTrace, TraceEvent


_TREE_SCRIPT = Path(__file__).with_name("ReluSearchTreeClient.js").read_text(encoding="utf-8")
_TREE_STYLES = Path(__file__).with_name("ReluSearchTree.css").read_text(encoding="utf-8")


@dataclass
class SearchNode:
    id: int
    order: int
    split_number: Optional[int]
    round_index: Optional[int]
    depth: int
    layer: Optional[int]
    neuron: Optional[int]
    variable: str
    phase: str
    result: str
    branch_result: str
    reason: Optional[str]
    bound: Optional[str]
    duration: float
    completed_at: Optional[float]
    repairs: int
    simplex_calls: int
    prunes_space: bool
    parent_id: Optional[int] = None
    sat_role: Optional[str] = None
    sat_source_id: Optional[int] = None
    unsat_role: Optional[str] = None
    unsat_source_ids: List[int] = field(default_factory=list)
    children: List["SearchNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        children = [child.to_dict() for child in self.children]
        max_depth = max(
            [self.depth] + [int(child["maxDepth"]) for child in children]
        )
        return {
            "id": self.id,
            "order": self.order,
            "splitNumber": self.split_number,
            "roundIndex": self.round_index,
            "depth": self.depth,
            "layer": self.layer,
            "neuron": self.neuron,
            "variable": self.variable,
            "phase": self.phase,
            "result": self.result,
            "branchResult": self.branch_result,
            "parentId": self.parent_id,
            "satRole": self.sat_role,
            "satSourceId": self.sat_source_id,
            "unsatRole": self.unsat_role,
            "unsatSourceIds": self.unsat_source_ids,
            "reason": self.reason,
            "bound": self.bound,
            "duration": self.duration,
            "completedAt": self.completed_at,
            "repairs": self.repairs,
            "simplexCalls": self.simplex_calls,
            "prunesSpace": self.prunes_space,
            "maxDepth": max_depth,
            "subtreeSize": 1 + sum(child["subtreeSize"] for child in children),
            "children": children,
        }


def _inside(candidate: TraceEvent, container: TraceEvent) -> bool:
    container_end = float("inf") if container.t_end is None else container.t_end
    candidate_end = candidate.t_start if candidate.t_end is None else candidate.t_end
    return container.t_start <= candidate.t_start and candidate_end <= container_end


def _count_inside(events: Iterable[TraceEvent], branch: TraceEvent, component: str) -> int:
    end = float("inf") if branch.t_end is None else branch.t_end
    return sum(
        1 for event in events
        if event.component == component and branch.t_start <= event.t_start <= end
    )


def build_relu_search_tree(
    trace: SolveTrace,
    *,
    overall_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert nested ``reluplex_branch`` spans to JSON-friendly tree nodes."""
    branches = sorted(
        (event for event in trace.events if event.component == "reluplex_branch"),
        key=lambda event: event.t_start,
    )
    split_spans = sorted(
        (event for event in trace.events if event.component == "reluplex_split"),
        key=lambda event: event.t_start,
    )
    split_numbers = {id(event): number for number, event in enumerate(split_spans, 1)}
    roots: List[SearchNode] = []
    stack: List[tuple[TraceEvent, SearchNode]] = []
    event_nodes: List[tuple[TraceEvent, SearchNode, str]] = []

    for order, event in enumerate(branches, 1):
        while stack and (
            (event.depth or 0) <= (stack[-1][0].depth or 0)
            or not _inside(event, stack[-1][0])
        ):
            stack.pop()

        parsed = parse_neuron_var(event.branch_x) if event.branch_x else None
        layer = event.meta.get("layer")
        neuron = event.meta.get("index")
        if (layer is None or neuron is None) and parsed is not None:
            layer, neuron = parsed
        raw_result = str(event.meta.get("result", "UNKNOWN")).upper()
        result = raw_result
        split_number = event.meta.get("split_number")
        if split_number is None:
            containing_splits = [
                split for split in split_spans
                if split.branch_x == event.branch_x and _inside(event, split)
            ]
            if containing_splits:
                nearest_split = max(containing_splits, key=lambda split: split.t_start)
                split_number = split_numbers[id(nearest_split)]
        node = SearchNode(
            id=order,
            order=order,
            split_number=(
                None if split_number is None else int(split_number)
            ),
            round_index=(
                None
                if event.meta.get("round_index") is None
                else int(event.meta["round_index"])
            ),
            depth=int(event.depth or 0),
            layer=None if layer is None else int(layer),
            neuron=None if neuron is None else int(neuron),
            variable=str(event.branch_x or "unknown"),
            phase=str(event.meta.get("phase", "?")),
            result=result,
            branch_result=raw_result,
            reason=event.meta.get("reason"),
            bound=event.meta.get("bound"),
            duration=max(0.0, float(event.duration or 0.0)),
            completed_at=None if event.t_end is None else float(event.t_end),
            repairs=_count_inside(trace.events, event, "reluplex_repair"),
            simplex_calls=_count_inside(trace.events, event, "simplex"),
            prunes_space=(raw_result == "UNSAT"),
        )
        if stack:
            node.parent_id = stack[-1][1].id
            stack[-1][1].children.append(node)
        else:
            roots.append(node)
        event_nodes.append((event, node, raw_result))
        stack.append((event, node))

    # A SAT branch either found the solution locally or returned a SAT child.
    # Work from children to parents so every ancestor points to the discovery.
    for _event, node, raw_result in reversed(event_nodes):
        if raw_result != "SAT":
            continue
        sat_child = next(
            (child for child in node.children if child.branch_result == "SAT"),
            None,
        )
        node.sat_role = "found" if sat_child is None else "propagated"
        node.sat_source_id = node.id if sat_child is None else sat_child.sat_source_id

    # UNSAT ancestors can combine multiple conflict leaves. A BACKJUMP leaf
    # reuses an existing conflict without searching, so it is not a discovery.
    for event, node, raw_result in reversed(event_nodes):
        if raw_result != "UNSAT":
            continue
        origin = event.meta.get("unsat_origin")
        if node.reason == "BACKJUMP" or origin == "skipped":
            node.unsat_role = "skipped"
            continue
        unsat_children = [
            child for child in node.children if child.branch_result == "UNSAT"
        ]
        # New logs carry the solver's explicit detection site. The structural
        # fallback keeps reports from older logs readable.
        node.unsat_role = (
            origin if origin in ("found", "propagated")
            else "propagated" if unsat_children else "found"
        )
        node.unsat_source_ids = (
            list(dict.fromkeys(
                source_id
                for child in unsat_children
                for source_id in child.unsat_source_ids
            ))
            if node.unsat_role == "propagated" else [node.id]
        )

    # Intermediate recursive returns are traversal completions, not the final
    # solver verdict. Keep them CLOSED, preserve their branch result, and expose
    # exactly one decisive event.
    for _event, node, raw_result in event_nodes:
        if raw_result in ("SAT", "UNSAT"):
            node.result = "CLOSED"

    final_status = str(overall_status or "").upper()
    if final_status in ("SAT", "UNSAT"):
        decisive = [
            (event, node) for event, node, raw_result in event_nodes
            if raw_result == final_status and event.t_end is not None
        ]
        if decisive:
            _event, decisive_node = max(decisive, key=lambda item: item[0].t_end)
            decisive_node.result = final_status

    return [root.to_dict() for root in roots]


def build_stable_records(trace: SolveTrace) -> List[Dict[str, Any]]:
    """Keep each fixation scoped to its theory call and containing branch."""
    by_depth = defaultdict(list)
    branches = sorted(
        (event for event in trace.events if event.component == "reluplex_branch"),
        key=lambda event: event.t_start,
    )
    for branch_id, event in enumerate(branches, 1):
        by_depth[event.depth or 0].append((event.t_start, event, branch_id))
    starts = {depth: [item[0] for item in items] for depth, items in by_depth.items()}
    records = []
    for event in sorted(
        (event for event in trace.events if event.component == "reluplex_stable"),
        key=lambda event: event.t_start,
    ):
        depth = event.depth or 0
        branch_id = None
        index = bisect_right(starts.get(depth, []), event.t_start) - 1
        if index >= 0:
            _, branch, candidate_id = by_depth[depth][index]
            same_round = (event.meta.get("round_index") is None
                          or branch.meta.get("round_index") is None
                          or event.meta["round_index"] == branch.meta["round_index"])
            if same_round and _inside(event, branch):
                branch_id = candidate_id
        for record in event.meta.get("records", []):
            variable = str(record.get("variable", "unknown"))
            parsed = parse_neuron_var(variable)
            layer, neuron = record.get("layer"), record.get("index")
            if parsed is not None:
                layer = parsed[0] if layer is None else layer
                neuron = parsed[1] if neuron is None else neuron
            records.append({
                "id": len(records) + 1, "variable": variable,
                "layer": layer, "neuron": neuron,
                "result": "STABLE", "state": record.get("state"),
                "source": event.meta.get("source", "search"),
                "before": record.get("before"), "after": record.get("after"),
                "nodeId": branch_id, "branchId": branch_id, "depth": depth,
                "roundIndex": event.meta.get("round_index"),
            })
    return records


def write_relu_search_tree_html(
    trace: SolveTrace,
    output_path: str | Path,
    *,
    overall_status: str,
    overall_reason: Optional[str] = None,
    title: str = "ReLU Search Tree",
    theory_stats: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a standalone HTML report; no server or external library is required."""
    roots = build_relu_search_tree(trace, overall_status=overall_status)
    serialized_nodes: List[Dict[str, Any]] = []

    def collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            serialized_nodes.append(node)
            collect(node["children"])

    collect(roots)
    stable_records = build_stable_records(trace)
    node_by_id = {node["id"]: node for node in serialized_nodes}
    for record in stable_records:
        node = node_by_id.get(record["branchId"])
        if node is not None and record["source"] != "initial":
            node["stableAddedCount"] = node.get("stableAddedCount", 0) + 1
    known_split_numbers = {
        int(node["splitNumber"])
        for node in serialized_nodes
        if node.get("splitNumber") is not None
    }
    closed_count = sum(node["result"] == "CLOSED" for node in serialized_nodes)
    closed_relu_count = len({
        node["variable"]
        for node in serialized_nodes
        if node["result"] == "CLOSED"
    })
    stats = theory_stats or {}
    total_relus = stats.get("DERIVE_TOTAL_RELUS")
    fixed_relus = stats.get("DERIVE_FIXED_RELUS")
    remaining_relus = stats.get("DERIVE_REMAINING_RELUS")
    rounds = [
        int(node["roundIndex"])
        for node in serialized_nodes
        if node.get("roundIndex") is not None
    ]
    metric_round = max(rounds) if rounds else None
    pruned_depths: List[int] = []

    def collect_pruned_frontier(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            if metric_round is not None and node.get("roundIndex") != metric_round:
                collect_pruned_frontier(node["children"])
                continue
            if node.get("prunesSpace") and not node["children"]:
                # A leaf UNSAT prefix closes every still-unassigned activation
                # pattern below it without visiting those individual patterns.
                pruned_depths.append(max(0, int(node["depth"])))
                continue
            collect_pruned_frontier(node["children"])

    collect_pruned_frontier(roots)
    depth_counts = Counter(pruned_depths)
    if isinstance(remaining_relus, int):
        depth_counts = Counter({
            depth: sum(
                count
                for original_depth, count in depth_counts.items()
                if min(original_depth, remaining_relus) == depth
            )
            for depth in {min(value, remaining_relus) for value in depth_counts}
        })
    pruned_ratio = min(
        1.0,
        sum(count * (2.0 ** (-depth)) for depth, count in depth_counts.items()),
    )
    pruned_terms = []
    if isinstance(remaining_relus, int):
        pruned_terms = [
            {
                "depth": depth,
                "count": count,
                "exponent": remaining_relus - depth,
            }
            for depth, count in sorted(depth_counts.items())
        ]
    payload = {
        "title": title,
        "status": str(overall_status).upper(),
        "reason": overall_reason,
        "roots": roots,
        "stableRecords": stable_records,
        "nodeCount": sum(root["subtreeSize"] for root in roots),
        "closedCount": closed_count,
        "closedReluCount": closed_relu_count,
        "boundDerivation": {
            "total": total_relus,
            "fixed": fixed_relus,
            "remaining": remaining_relus,
        },
        "prunedSpace": {
            "totalExponent": remaining_relus,
            "terms": pruned_terms,
            "ratio": pruned_ratio,
            "roundIndex": metric_round,
        },
        "splitCount": (
            len(known_split_numbers)
            if known_split_numbers
            else len(serialized_nodes)
        ),
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    script = _TREE_SCRIPT.replace("__DATA__", data)
    document = _HTML.replace("__SCRIPT__", script).replace("__STYLES__", _TREE_STYLES).replace("__TITLE__", escape(title))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, path)
    return path


def branch_log_path(split_log_path: str | Path) -> Path:
    path = Path(split_log_path)
    return path.with_name(f"{path.stem}_branches.jsonl")


def trace_from_branch_log(path: str | Path) -> SolveTrace:
    """Reconstruct branch spans from an append-only realtime JSONL log."""
    trace = SolveTrace()
    open_events: Dict[str, TraceEvent] = {}
    source = Path(path)
    if not source.exists():
        return trace
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                kind = record["event"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if kind == "relu_stable":
                stamp = int(record["time_ns"]) / 1_000_000_000
                trace.events.append(TraceEvent(
                    component="reluplex_stable", t_start=stamp, t_end=stamp,
                    depth=int(record.get("depth", 0)),
                    meta={key: record.get(key) for key in ("source", "records", "round_index")},
                ))
                continue
            if "branch_id" not in record:
                continue
            branch_id = str(record["branch_id"])
            if kind == "branch_start":
                event = TraceEvent(
                    component="reluplex_branch",
                    t_start=int(record["time_ns"]) / 1_000_000_000,
                    depth=int(record.get("depth", 0)),
                    branch_x=str(record.get("variable", "")),
                    meta={
                        "phase": str(record.get("phase", "?")),
                        "layer": record.get("layer"),
                        "index": record.get("index"),
                        "bound": record.get("bound"),
                        "split_number": record.get("split_number"),
                        "round_index": record.get("round_index"),
                        "result": "RUNNING",
                    },
                )
                trace.events.append(event)
                open_events[branch_id] = event
            elif kind == "branch_end" and branch_id in open_events:
                event = open_events.pop(branch_id)
                event.t_end = int(record["time_ns"]) / 1_000_000_000
                event.meta["result"] = str(record.get("result", "UNKNOWN"))
                if record.get("reason"):
                    event.meta["reason"] = str(record["reason"])
                if record.get("unsat_origin"):
                    event.meta["unsat_origin"] = str(record["unsat_origin"])
    return trace


class RealtimeSearchTreeVisualizer:
    """Coalescing background writer for the realtime interactive tree."""

    def __init__(
        self,
        log_path: str | Path,
        output_path: str | Path,
        *,
        theory_stats: Optional[Dict[str, Any]] = None,
        title: str = "Realtime ReLU Search Tree",
    ) -> None:
        self.log_path = Path(log_path)
        self.output_path = Path(output_path)
        self.status = "RUNNING"
        self.reason: Optional[str] = None
        self.theory_stats = theory_stats
        self.title = title
        self._lock = threading.Lock()
        self._pending = False
        self._worker: Optional[threading.Thread] = None

    def set_overall(self, status: str, reason: Optional[str] = None) -> None:
        with self._lock:
            self.status = str(status).upper()
            self.reason = reason

    def _write_once(self) -> None:
        trace = trace_from_branch_log(self.log_path)
        write_relu_search_tree_html(
            trace,
            self.output_path,
            overall_status=self.status,
            overall_reason=self.reason,
            title=self.title,
            theory_stats=self.theory_stats,
        )

    def _run(self) -> None:
        while True:
            with self._lock:
                self._pending = False
            self._write_once()
            with self._lock:
                if not self._pending:
                    self._worker = None
                    return
            sleep(0.25)

    def request_update(self) -> None:
        with self._lock:
            self._pending = True
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._run,
                name="relu-search-tree-writer",
                daemon=True,
            )
            self._worker.start()

    def flush(self) -> None:
        while True:
            with self._lock:
                worker = self._worker
            if worker is None:
                return
            worker.join()


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__STYLES__</style></head><body>
<header><h1 id="title"></h1><div class="summary"><span id="overall" class="pill"></span><span id="count"></span><span id="reason" class="meta"></span></div>
<div class="metrics"><div class="metric" title="더 분기하지 않고 UNSAT으로 닫힌 말단 접두사가 생략한 활성화 패턴 수를 전체 불안정 ReLU 활성화 패턴 2^n과 비교합니다. 내부 CLOSED 노드는 말단 결과를 합친 것이므로 중복 계산하지 않습니다."><div class="metric-label">닫힘으로 생략한 탐색 공간</div><div id="closedMetric" class="metric-value"></div><div id="closedDetail" class="metric-detail"></div><div class="metric-bar"><i id="closedBar"></i></div></div><div class="metric bound" title="구간 경계 유도로 활성/비활성 상태가 확정되어 탐색에서 제외된 ReLU와 아직 분기가 가능한 불안정 ReLU 수입니다."><div class="metric-label">경계 유도 ReLU</div><div id="boundMetric" class="metric-value"></div><div id="boundDetail" class="metric-detail"></div><div class="metric-bar"><i id="boundBar"></i></div></div></div>
<div class="controls"><button id="defaultView">루트만 보기</button><button id="recentPath" title="루트부터 가장 최근에 추가된 가지까지 펼칩니다">최근 경로보기</button><button id="trackRecentPath" aria-pressed="false" title="켜 두면 새 가지가 추가될 때마다 최근 경로를 자동으로 펼쳐 따라갑니다">최근 경로 추적</button><button id="toggleReportPanels" aria-controls="timelinePanel stablePanel" aria-expanded="true">하단 패널 접기</button><button id="expandAll">모두 펼치기</button><button id="collapseAll">모두 접기</button><select id="statusFilter"><option value="ALL">모든 결과</option><option value="FINAL">최종 결과</option><option value="STABLE">STABLE</option><option value="SAT_FOUND">SAT 발견</option><option value="UNSAT_FOUND">UNSAT 발견</option><option value="CLOSED">CLOSED 전체</option><option value="CLOSED_SAT">CLOSED (SAT)</option><option value="CLOSED_UNSAT">CLOSED (UNSAT)</option><option>RUNNING</option></select><input id="layerFilter" inputmode="numeric" placeholder="Layer 번호"><input id="neuronFilter" inputmode="numeric" placeholder="Neuron 번호"></div></header>
<main><section class="panel"><div id="treeViewport" class="tree-viewport" role="region" aria-label="ReLU 탐색 트리" tabindex="0"><ul id="tree" class="tree"></ul></div><div class="help">트리가 길거나 깊어지면 트리 영역 안에서 위아래·좌우로 스크롤할 수 있습니다.</div><div class="help">L은 Layer(신경망 층), N은 Neuron(해당 층의 뉴런)을 뜻하며 번호는 0부터 시작합니다. 예: L1:N36은 첫 번째 은닉층의 37번째 뉴런입니다.</div>
<div class="help">노드를 클릭해 하위 분기를 펼칩니다. 노드 위에 마우스를 올리면 해·모순 발견 지점, bound, repair, Simplex 호출 수가 표시됩니다.</div>
<div class="help" id="stableHelp">stable: 이 탐색 노드의 조건에서 경계 계산으로 활성(+)·비활성(-)이 새로 확정된 ReLU 뉴런 수입니다.</div></section>
<details class="panel report-panel" id="timelinePanel" open>
<summary><h2 id="timelineTitle">탐색 완료 순서 타임라인</h2></summary>
<div class="panel-content"><div class="controls"><select id="timelineOrder" aria-label="타임라인 정렬"><option value="completion">완료순</option><option value="visit">방문순</option><option value="duration">소요시간순(긴 순)</option></select></div>
<div id="timeline" class="timeline"></div><div class="controls timeline-pager"><button id="timelinePrev">이전</button><span id="timelinePage" class="meta"></span><button id="timelineNext">다음</button></div></div></details>
<details class="panel report-panel" id="stablePanel" open>
<summary><h2>STABLE 확정 뉴런</h2></summary>
<div class="panel-content"><div class="controls"><select id="stableScope" aria-label="STABLE 범위"><option value="additional">추가 STABLE만</option><option value="all">전체 STABLE</option><option value="initial">초기 STABLE만</option></select><select id="stableOrder" aria-label="STABLE 정렬"><option value="newest">최근 확정순</option><option value="neuron">레이어·뉴런순</option><option value="node">STABLE 확정 노드순</option></select><span id="stableCount" class="meta"></span></div>
<div id="stableList" class="stable-list"><table class="stable-table"><thead><tr><th>뉴런</th><th>상태</th><th>확정 결과</th><th>확정 기법</th><th>입력 경계 변화</th><th>STABLE 확정 노드</th></tr></thead><tbody id="stableRows"></tbody></table></div><div class="controls timeline-pager"><button id="stablePrev">이전</button><span id="stablePage" class="meta"></span><button id="stableNext">다음</button></div></div></details>
</main>
<script>__SCRIPT__</script></body></html>'''
