"""Build a self-contained, interactive ReLU branch search-tree report."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import threading
from time import sleep
from typing import Any, Dict, Iterable, List, Optional

from visualization.NetworkLayout import parse_neuron_var
from visualization.SolveTrace import SolveTrace, TraceEvent


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
    document = _HTML.replace("__TITLE__", escape(title)).replace("__DATA__", data)
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
                branch_id = str(record["branch_id"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
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
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2c;--line:#42506b;--text:#e7edf8;--muted:#9daac0;--sat:#35c779;--unsat:#ff626b;--closed:#3b82f6;--unknown:#94a3b8;--timeout:#eab308;--running:#f97316;--other:#8390a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:3;background:#0b1020ee;border-bottom:1px solid #29344b;padding:16px 22px}h1{font-size:20px;margin:0 0 8px}.summary,.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.pill,.badge{border-radius:999px;padding:2px 8px;font-weight:700}.SAT{background:#153f31;color:#7ce7ad}.UNSAT{background:#4b2028;color:#ff9ca3}.CLOSED{background:#172f58;color:#93c5fd}.UNKNOWN{background:#30394a;color:#cbd5e1}.TIMEOUT{background:#4a3b0d;color:#fde047}.RUNNING{background:#51280f;color:#fdba74}.OTHER{background:#283246;color:#b8c4d8}button,select,input{background:#172137;color:var(--text);border:1px solid #35435e;border-radius:6px;padding:6px 9px}button{cursor:pointer}main{padding:18px 22px}.panel{background:var(--panel);border:1px solid #26334b;border-radius:10px;padding:14px;margin-bottom:16px}.panel-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}.panel-heading h2{font-size:16px;margin:0}.tree,.children{list-style:none;margin:0;padding-left:0}.children{margin-left:21px;padding-left:17px;border-left:1px solid var(--line)}details{margin:5px 0}summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:7px;min-height:34px;border-left:var(--weight,3px) solid var(--other);padding:5px 9px;background:#172137;border-radius:6px}summary::-webkit-details-marker{display:none}summary:before{content:'▸';color:var(--muted);width:10px}details[open]>summary:before{content:'▾'}summary.s-SAT{border-color:var(--sat)}summary.s-UNSAT{border-color:var(--unsat)}summary.s-CLOSED{border-color:var(--closed)}summary.s-UNKNOWN{border-color:var(--unknown)}summary.s-TIMEOUT{border-color:var(--timeout)}summary.s-RUNNING{border-color:var(--running)}.label{font-weight:750}.meta,.collapsed{color:var(--muted);font-size:12px}.collapsed{margin-left:auto}.phase{font-family:ui-monospace,monospace;background:#27344c;border-radius:4px;padding:1px 6px}.phase.plus{color:#75d8ff}.phase.minus{color:#dda6ff}.timeline{max-height:330px;overflow:auto}.row{display:grid;grid-template-columns:55px minmax(140px,1fr) 85px 90px;gap:8px;padding:6px 8px;border-bottom:1px solid #243047}.row:hover{background:#19243a}.empty{text-align:center;color:var(--muted);padding:35px}.help{color:var(--muted);font-size:12px;margin-top:8px}@media(max-width:650px){.row{grid-template-columns:45px 1fr 72px}.row .duration{display:none}header,main{padding-left:11px;padding-right:11px}}
.metrics{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:10px;margin:12px 0}.metric{background:#121a2c;border:1px solid #2b3953;border-radius:9px;padding:10px 12px}.metric-label{color:var(--muted);font-size:12px}.metric-value{font-size:18px;font-weight:800;margin-top:2px}.metric-detail{color:var(--muted);font-size:12px;margin-top:2px}.metric-bar{height:5px;background:#26334b;border-radius:999px;overflow:hidden;margin-top:7px}.metric-bar>i{display:block;height:100%;background:var(--closed);border-radius:inherit}.metric.bound .metric-bar>i{background:var(--sat)}@media(max-width:650px){.metrics{grid-template-columns:1fr}}
.badge{white-space:nowrap}.row{grid-template-columns:55px minmax(140px,1fr) 145px 90px}@media(max-width:650px){.row{grid-template-columns:45px minmax(140px,1fr) 145px}}
.SAT_FOUND{background:#153f31;color:#7ce7ad}summary.s-SAT_FOUND{border-color:var(--sat)}
.UNSAT_FOUND{background:#4b2028;color:#ff9ca3}summary.s-UNSAT_FOUND{border-color:var(--unsat)}
</style></head><body>
<header><h1 id="title"></h1><div class="summary"><span id="overall" class="pill"></span><span id="count"></span><span id="reason" class="meta"></span></div>
<div class="metrics"><div class="metric" title="더 분기하지 않고 UNSAT으로 닫힌 말단 접두사가 생략한 활성화 패턴 수를 전체 불안정 ReLU 활성화 패턴 2^n과 비교합니다. 내부 CLOSED 노드는 말단 결과를 합친 것이므로 중복 계산하지 않습니다."><div class="metric-label">닫힘으로 생략한 탐색 공간</div><div id="closedMetric" class="metric-value"></div><div id="closedDetail" class="metric-detail"></div><div class="metric-bar"><i id="closedBar"></i></div></div><div class="metric bound" title="구간 경계 유도로 활성/비활성 상태가 확정되어 탐색에서 제외된 ReLU와 아직 분기가 가능한 불안정 ReLU 수입니다."><div class="metric-label">경계 유도 ReLU</div><div id="boundMetric" class="metric-value"></div><div id="boundDetail" class="metric-detail"></div><div class="metric-bar"><i id="boundBar"></i></div></div></div>
<div class="controls"><button id="defaultView">최종 경로</button><button id="expandAll">모두 펼치기</button><button id="collapseAll">모두 접기</button><select id="statusFilter"><option value="ALL">모든 결과</option><option>SAT</option><option value="SAT_FOUND">SAT 발견</option><option>UNSAT</option><option value="UNSAT_FOUND">UNSAT 발견</option><option value="CLOSED">중간 종료 전체</option><option value="CLOSED_SAT">SAT 중간 종료</option><option value="CLOSED_UNSAT">UNSAT 중간 종료</option><option>RUNNING</option><option>UNKNOWN</option><option>TIMEOUT</option></select><input id="layerFilter" inputmode="numeric" placeholder="Layer 번호"><input id="neuronFilter" inputmode="numeric" placeholder="Neuron 번호"></div></header>
<main><section class="panel"><ul id="tree" class="tree"></ul><div class="help">L은 Layer(신경망 층), N은 Neuron(해당 층의 뉴런)을 뜻하며 번호는 0부터 시작합니다. 예: L1:N36은 첫 번째 은닉층의 37번째 뉴런입니다.</div><div class="help">해를 직접 찾은 가지는 초록색 SAT 발견, 모순을 직접 확인한 가지는 빨간색 UNSAT 발견으로 표시합니다. 하위 결과를 받거나 탐색을 생략해 닫힌 가지는 부모 상태와 관계없이 CLOSED (SAT) 또는 CLOSED (UNSAT)로 표시합니다. +와 −에서 각각 모순을 직접 확인했다면 두 가지 모두 발견으로 표시됩니다. 최종 판정을 대표하는 가지에는 SAT 또는 UNSAT만 표시합니다. 전체 검증 결과는 상단 배지를 확인하세요.</div><div class="help">노드를 클릭해 하위 분기를 펼칩니다. 노드 위에 마우스를 올리면 해·모순 발견 지점, bound, repair, Simplex 호출 수가 표시됩니다.</div></section>
<section class="panel"><div class="panel-heading"><h2 id="timelineTitle">탐색 완료 순서 타임라인</h2><select id="timelineOrder" aria-label="타임라인 정렬"><option value="completion">완료순</option><option value="visit">방문순</option><option value="duration">소요시간순(긴 순)</option></select></div><div id="timeline" class="timeline"></div></section></main>
<script>
const DATA=__DATA__,STATE_PREFIX='relu-search-tree-ui:';let savedState={};try{if(window.name.startsWith(STATE_PREFIX))savedState=JSON.parse(window.name.slice(STATE_PREFIX.length))}catch(_error){savedState={}}let mode=['default','all','none'].includes(savedState.mode)?savedState.mode:'default';
if(window.parent!==window)window.parent.postMessage({type:'relu-search-tree-status',status:DATA.status},'*');
const $=s=>document.querySelector(s), statusClass=s=>['SAT','UNSAT','CLOSED','UNKNOWN','TIMEOUT','RUNNING'].includes(s)?s:'OTHER';
function branchRoleLabel(n){if(n.satRole)return n.satRole==='found'?'SAT 발견':'SAT 전달';if(n.unsatRole)return {found:'UNSAT 발견',propagated:'UNSAT 전달',skipped:'UNSAT 생략'}[n.unsatRole];return ''}
function isFinalVerdict(n){return n.result==='SAT'||n.result==='UNSAT'}
function isClosedResult(n){return n.result==='CLOSED'&&((n.branchResult==='SAT'&&n.satRole!=='found')||(n.branchResult==='UNSAT'&&n.unsatRole!=='found'))}
function resultLabel(n){if(isFinalVerdict(n))return n.result;if(isClosedResult(n))return `CLOSED (${n.branchResult})`;const role=branchRoleLabel(n);if(role)return role;return n.result==='CLOSED'?`CLOSED (${n.branchResult})`:n.result}
function nodeStatusClass(n){if(isFinalVerdict(n))return statusClass(n.result);if(isClosedResult(n))return 'CLOSED';if(n.satRole==='found')return 'SAT_FOUND';if(n.unsatRole==='found')return 'UNSAT_FOUND';return statusClass(n.result==='CLOSED'?n.branchResult:n.result)}
function resultBadge(n){return `<span class="badge ${nodeStatusClass(n)}">${resultLabel(n)}</span>`}
function matchesStatus(n,s){return s==='ALL'||n.result===s||n.branchResult===s||((n.satRole||n.unsatRole)&&s===nodeStatusClass(n))||(n.result==='CLOSED'&&s===`CLOSED_${n.branchResult}`)}
$('#title').textContent=DATA.title;$('#overall').textContent=DATA.status;$('#overall').className='pill '+statusClass(DATA.status);$('#count').textContent=`분기 노드 ${DATA.splitCount.toLocaleString()}개 · 가지 ${DATA.nodeCount.toLocaleString()}개`;$('#reason').textContent=DATA.reason||'';const bd=DATA.boundDerivation||{},ps=DATA.prunedSpace||{},hasBound=Number.isInteger(bd.total)&&Number.isInteger(bd.fixed)&&Number.isInteger(bd.remaining),hasSpace=Number.isInteger(ps.totalExponent),closedPercent=hasSpace?ps.ratio*100:0,percentText=p=>p===0?'0.0':p>=0.01?p.toFixed(2):p.toExponential(2),termText=t=>`${t.count>1?t.count+' × ':''}2^${t.exponent}`,numerator=ps.terms&&ps.terms.length?ps.terms.map(termText).join(' + '):'0';$('#closedMetric').textContent=hasSpace?`(${numerator}) / 2^${ps.totalExponent} = ${percentText(closedPercent)}%`:'계산 중';$('#closedDetail').textContent=hasSpace?`탐색하지 않아도 되는 공간 ${percentText(closedPercent)}% · 남은 공간 ${percentText(Math.max(0,100-closedPercent))}%`:'경계 유도 결과를 기다리는 중';$('#closedBar').style.width=`${Math.min(100,closedPercent)}%`;$('#boundMetric').textContent=hasBound?`${bd.total.toLocaleString()} → ${bd.fixed.toLocaleString()} 고정 → ${bd.remaining.toLocaleString()} 불안정`:'계산 중';$('#boundDetail').textContent=hasBound?`전체 ${bd.total.toLocaleString()}개 중 경계로 ${bd.fixed.toLocaleString()}개 고정`:'경계 유도 결과를 기다리는 중';$('#boundBar').style.width=hasBound&&bd.total?`${bd.fixed/bd.total*100}%`:'0%';
function flat(nodes,out=[]){for(const n of nodes){out.push(n);flat(n.children,out)}return out}const ALL=flat(DATA.roots),LAST_ORDER=ALL.reduce((m,n)=>Math.max(m,n.order),0),TIMELINE=[...ALL].sort((a,b)=>(a.completedAt??Number.MAX_VALUE)-(b.completedAt??Number.MAX_VALUE)),DURATION_TIMELINE=[...ALL].sort((a,b)=>b.duration-a.duration||a.order-b.order);TIMELINE.forEach((n,i)=>n.completionOrder=i+1);DURATION_TIMELINE.forEach((n,i)=>n.durationOrder=i+1);
const NODE_BY_ID=new Map(ALL.map(n=>[n.id,n]));
function matches(n){const s=$('#statusFilter').value,l=$('#layerFilter').value.trim(),v=$('#neuronFilter').value.trim();return matchesStatus(n,s)&&(!l||String(n.layer)===l)&&(!v||String(n.neuron)===v)}
function hasMatch(n){return matches(n)||n.children.some(hasMatch)}
function sourceLabel(source){return `${source.layer==null?source.variable:`L${source.layer}:N${source.neuron}`} (${source.phase}) · 방문 #${source.order}`}
function satSourceLabel(n){const source=NODE_BY_ID.get(n.satSourceId);return source?`해 발견 지점: ${sourceLabel(source)}`:''}
function unsatSourceLabel(n){if(n.unsatRole==='skipped')return '기존 충돌로 탐색 생략 (BACKJUMP)';const sources=(n.unsatSourceIds||[]).map(id=>NODE_BY_ID.get(id)).filter(Boolean);return sources.length?`모순 발견 지점: ${sources.slice(0,8).map(sourceLabel).join(', ')}${sources.length>8?` 외 ${sources.length-8}개`:''}`:''}
function tip(n){return [`Split #${n.splitNumber??'?'}`,`${n.layer==null?n.variable:`Layer ${n.layer}, Neuron ${n.neuron}`} (${n.phase})`,`결과: ${resultLabel(n)}${n.reason?' / '+n.reason:''}`,(isClosedResult(n)||isFinalVerdict(n))?`분기 역할: ${branchRoleLabel(n)}`:'',satSourceLabel(n),unsatSourceLabel(n),`현재 깊이: ${n.depth}`,`하위 탐색 최대 깊이: ${n.maxDepth}`,`bound: ${n.bound||'-'}`,`repair: ${n.repairs}`,`Simplex 호출: ${n.simplexCalls}`,`소요시간: ${n.duration.toFixed(6)}초`].filter(Boolean).join('\n')}
function onLastPath(n){return n.order<=LAST_ORDER&&LAST_ORDER<n.order+n.subtreeSize}
function shouldOpen(n){return mode==='all'||(mode==='default'&&(n.branchResult==='SAT'||n.result==='UNKNOWN'||n.result==='TIMEOUT'||n.result==='RUNNING'||(DATA.status==='UNSAT'&&onLastPath(n))))}
function makeNode(n){const li=document.createElement('li'),d=document.createElement('details'),s=document.createElement('summary');d.dataset.id=n.id;d.open=shouldOpen(n);s.className='s-'+nodeStatusClass(n);s.style.setProperty('--weight',`${Math.min(10,3+Math.log10(1+n.duration*1000)*1.6)}px`);s.title=tip(n);const label=n.layer==null?n.variable:`L${n.layer}:N${n.neuron}`;s.innerHTML=`<span class="label">${label}</span><span class="phase ${n.phase==='+'?'plus':'minus'}">${n.phase}</span><span class="meta">Split #${n.splitNumber??'?'}</span>${resultBadge(n)}<span class="meta">${n.duration.toFixed(3)}s</span>${n.children.length?`<span class="collapsed">${n.subtreeSize-1}개 하위 노드</span>`:''}`;d.appendChild(s);const box=document.createElement('ul');box.className='children';let built=false;const build=()=>{if(built)return;built=true;for(const c of n.children)if(hasMatch(c))box.appendChild(makeNode(c));d.appendChild(box)};if(d.open)build();d.addEventListener('toggle',()=>{if(d.open)build()});li.appendChild(d);return li}
function renderTree(){const root=$('#tree');root.innerHTML='';const visible=DATA.roots.filter(hasMatch);if(!visible.length){root.innerHTML='<li class="empty">기록된 ReLU 분기가 없습니다.</li>';return}for(const n of visible)root.appendChild(makeNode(n))}
function renderTimeline(){const box=$('#timeline'),sortMode=$('#timelineOrder').value;box.innerHTML='';const byCompletion=sortMode==='completion',byDuration=sortMode==='duration';$('#timelineTitle').textContent=byCompletion?'탐색 완료 순서 타임라인':byDuration?'탐색 소요시간순 타임라인':'탐색 방문 순서 타임라인';const source=byCompletion?TIMELINE:byDuration?DURATION_TIMELINE:ALL;const rows=source.filter(matches);if(!rows.length){box.innerHTML='<div class="empty">표시할 탐색 기록이 없습니다.</div>';return}for(const n of rows){const row=document.createElement('div');row.className='row';const label=n.layer==null?n.variable:`L${n.layer}:N${n.neuron}`;const primary=byCompletion?n.completionOrder:byDuration?n.durationOrder:n.order,secondary=byCompletion?`방문 #${n.order}`:byDuration?`방문 #${n.order} · 완료 #${n.completionOrder}`:`완료 #${n.completionOrder}`;row.title=tip(n);row.innerHTML=`<span>#${primary}</span><span>${label} <b class="phase ${n.phase==='+'?'plus':'minus'}">${n.phase}</b> <small class="meta">Split #${n.splitNumber??'?'} · ${secondary}</small></span>${resultBadge(n)}<span class="duration">${n.duration.toFixed(3)}s</span>`;box.appendChild(row)}}
function saveState(){window.name=STATE_PREFIX+JSON.stringify({mode,timelineOrder:$('#timelineOrder').value,status:$('#statusFilter').value,layer:$('#layerFilter').value,neuron:$('#neuronFilter').value})}
if(savedState.status==='SAT_PROPAGATED')savedState.status='CLOSED_SAT';if(['UNSAT_PROPAGATED','UNSAT_SKIPPED'].includes(savedState.status))savedState.status='CLOSED_UNSAT';if(['completion','visit','duration'].includes(savedState.timelineOrder))$('#timelineOrder').value=savedState.timelineOrder;if(['ALL','SAT','SAT_FOUND','UNSAT','UNSAT_FOUND','CLOSED','CLOSED_SAT','CLOSED_UNSAT','RUNNING','UNKNOWN','TIMEOUT'].includes(savedState.status))$('#statusFilter').value=savedState.status;$('#layerFilter').value=savedState.layer||'';$('#neuronFilter').value=savedState.neuron||'';
function render(){renderTree();renderTimeline()}$('#defaultView').onclick=()=>{mode='default';saveState();renderTree()};$('#expandAll').onclick=()=>{mode='all';saveState();renderTree()};$('#collapseAll').onclick=()=>{mode='none';saveState();renderTree()};$('#statusFilter').onchange=()=>{saveState();render()};$('#layerFilter').oninput=()=>{saveState();render()};$('#neuronFilter').oninput=()=>{saveState();render()};$('#timelineOrder').onchange=()=>{saveState();renderTimeline()};render();
</script></body></html>'''
