"""Build a self-contained, interactive ReLU branch search-tree report."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    depth: int
    layer: Optional[int]
    neuron: Optional[int]
    variable: str
    phase: str
    result: str
    reason: Optional[str]
    bound: Optional[str]
    duration: float
    completed_at: Optional[float]
    repairs: int
    simplex_calls: int
    children: List["SearchNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        children = [child.to_dict() for child in self.children]
        return {
            "id": self.id,
            "order": self.order,
            "splitNumber": self.split_number,
            "depth": self.depth,
            "layer": self.layer,
            "neuron": self.neuron,
            "variable": self.variable,
            "phase": self.phase,
            "result": self.result,
            "reason": self.reason,
            "bound": self.bound,
            "duration": self.duration,
            "completedAt": self.completed_at,
            "repairs": self.repairs,
            "simplexCalls": self.simplex_calls,
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
            depth=int(event.depth or 0),
            layer=None if layer is None else int(layer),
            neuron=None if neuron is None else int(neuron),
            variable=str(event.branch_x or "unknown"),
            phase=str(event.meta.get("phase", "?")),
            result=result,
            reason=event.meta.get("reason"),
            bound=event.meta.get("bound"),
            duration=max(0.0, float(event.duration or 0.0)),
            completed_at=None if event.t_end is None else float(event.t_end),
            repairs=_count_inside(trace.events, event, "reluplex_repair"),
            simplex_calls=_count_inside(trace.events, event, "simplex"),
        )
        if stack:
            stack[-1][1].children.append(node)
        else:
            roots.append(node)
        event_nodes.append((event, node, raw_result))
        stack.append((event, node))

    # Intermediate recursive returns are traversal completions, not the final
    # solver verdict. Keep them CLOSED and expose exactly one decisive event.
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
) -> Path:
    """Write a standalone HTML report; no server or external library is required."""
    roots = build_relu_search_tree(trace, overall_status=overall_status)
    payload = {
        "title": title,
        "status": str(overall_status).upper(),
        "reason": overall_reason,
        "roots": roots,
        "nodeCount": sum(root["subtreeSize"] for root in roots),
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
    return trace


class RealtimeSearchTreeVisualizer:
    """Coalescing background writer for the realtime interactive tree."""

    def __init__(self, log_path: str | Path, output_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.output_path = Path(output_path)
        self.status = "RUNNING"
        self.reason: Optional[str] = None
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
            title="Realtime ReLU Search Tree",
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
</style></head><body>
<header><h1 id="title"></h1><div class="summary"><span id="overall" class="pill"></span><span id="count"></span><span id="reason" class="meta"></span></div>
<div class="controls"><button id="defaultView">최종 경로</button><button id="expandAll">모두 펼치기</button><button id="collapseAll">모두 접기</button><select id="statusFilter"><option value="ALL">모든 결과</option><option>SAT</option><option>UNSAT</option><option>CLOSED</option><option>UNKNOWN</option><option>TIMEOUT</option></select><input id="layerFilter" inputmode="numeric" placeholder="Layer 번호"><input id="neuronFilter" inputmode="numeric" placeholder="Neuron 번호"></div></header>
<main><section class="panel"><ul id="tree" class="tree"></ul><div class="help">L은 Layer(신경망 층), N은 Neuron(해당 층의 뉴런)을 뜻하며 번호는 0부터 시작합니다. 예: L1:N36은 첫 번째 은닉층의 37번째 뉴런입니다.</div><div class="help">SAT/UNSAT은 실제 결론이 난 말단 분기이며, CLOSED는 그 하위 분기 탐색을 마치고 되돌아온 내부 노드입니다.</div><div class="help">노드를 클릭해 하위 분기를 펼칩니다. 노드 위에 마우스를 올리면 bound, repair, Simplex 호출 수가 표시됩니다.</div></section>
<section class="panel"><div class="panel-heading"><h2 id="timelineTitle">탐색 완료 순서 타임라인</h2><select id="timelineOrder" aria-label="타임라인 정렬"><option value="completion">완료순</option><option value="visit">방문순</option><option value="duration">소요시간순(긴 순)</option></select></div><div id="timeline" class="timeline"></div></section></main>
<script>
const DATA=__DATA__,STATE_PREFIX='relu-search-tree-ui:';let savedState={};try{if(window.name.startsWith(STATE_PREFIX))savedState=JSON.parse(window.name.slice(STATE_PREFIX.length))}catch(_error){savedState={}}let mode=['default','all','none'].includes(savedState.mode)?savedState.mode:'default';
if(window.parent!==window)window.parent.postMessage({type:'relu-search-tree-status',status:DATA.status},'*');
const $=s=>document.querySelector(s), statusClass=s=>['SAT','UNSAT','CLOSED','UNKNOWN','TIMEOUT','RUNNING'].includes(s)?s:'OTHER';
$('#title').textContent=DATA.title;$('#overall').textContent=DATA.status;$('#overall').className='pill '+statusClass(DATA.status);$('#count').textContent=`분기 ${DATA.nodeCount.toLocaleString()}개`;$('#reason').textContent=DATA.reason||'';
function flat(nodes,out=[]){for(const n of nodes){out.push(n);flat(n.children,out)}return out}const ALL=flat(DATA.roots),LAST_ORDER=ALL.reduce((m,n)=>Math.max(m,n.order),0),TIMELINE=[...ALL].sort((a,b)=>(a.completedAt??Number.MAX_VALUE)-(b.completedAt??Number.MAX_VALUE)),DURATION_TIMELINE=[...ALL].sort((a,b)=>b.duration-a.duration||a.order-b.order);TIMELINE.forEach((n,i)=>n.completionOrder=i+1);DURATION_TIMELINE.forEach((n,i)=>n.durationOrder=i+1);
function matches(n){const s=$('#statusFilter').value,l=$('#layerFilter').value.trim(),v=$('#neuronFilter').value.trim();return (s==='ALL'||n.result===s)&&(!l||String(n.layer)===l)&&(!v||String(n.neuron)===v)}
function hasMatch(n){return matches(n)||n.children.some(hasMatch)}
function tip(n){return [`Split #${n.splitNumber??'?'}`,`${n.layer==null?n.variable:`Layer ${n.layer}, Neuron ${n.neuron}`} (${n.phase})`,`결과: ${n.result}${n.reason?' / '+n.reason:''}`,`bound: ${n.bound||'-'}`,`repair: ${n.repairs}`,`Simplex 호출: ${n.simplexCalls}`,`소요시간: ${n.duration.toFixed(6)}초`].join('\n')}
function onLastPath(n){return n.order<=LAST_ORDER&&LAST_ORDER<n.order+n.subtreeSize}
function shouldOpen(n){return mode==='all'||(mode==='default'&&(n.result==='SAT'||n.result==='UNKNOWN'||n.result==='TIMEOUT'||n.result==='RUNNING'||(DATA.status==='UNSAT'&&onLastPath(n))))}
function makeNode(n){const li=document.createElement('li'),d=document.createElement('details'),s=document.createElement('summary');d.dataset.id=n.id;d.open=shouldOpen(n);s.className='s-'+statusClass(n.result);s.style.setProperty('--weight',`${Math.min(10,3+Math.log10(1+n.duration*1000)*1.6)}px`);s.title=tip(n);const label=n.layer==null?n.variable:`L${n.layer}:N${n.neuron}`;s.innerHTML=`<span class="label">${label}</span><span class="phase ${n.phase==='+'?'plus':'minus'}">${n.phase}</span><span class="meta">Split #${n.splitNumber??'?'}</span><span class="badge ${statusClass(n.result)}">${n.result}</span><span class="meta">${n.duration.toFixed(3)}s</span>${n.children.length?`<span class="collapsed">${n.subtreeSize-1}개 하위 노드</span>`:''}`;d.appendChild(s);const box=document.createElement('ul');box.className='children';let built=false;const build=()=>{if(built)return;built=true;for(const c of n.children)if(hasMatch(c))box.appendChild(makeNode(c));d.appendChild(box)};if(d.open)build();d.addEventListener('toggle',()=>{if(d.open)build()});li.appendChild(d);return li}
function renderTree(){const root=$('#tree');root.innerHTML='';const visible=DATA.roots.filter(hasMatch);if(!visible.length){root.innerHTML='<li class="empty">기록된 ReLU 분기가 없습니다.</li>';return}for(const n of visible)root.appendChild(makeNode(n))}
function renderTimeline(){const box=$('#timeline'),sortMode=$('#timelineOrder').value;box.innerHTML='';const byCompletion=sortMode==='completion',byDuration=sortMode==='duration';$('#timelineTitle').textContent=byCompletion?'탐색 완료 순서 타임라인':byDuration?'탐색 소요시간순 타임라인':'탐색 방문 순서 타임라인';const source=byCompletion?TIMELINE:byDuration?DURATION_TIMELINE:ALL;const rows=source.filter(matches);if(!rows.length){box.innerHTML='<div class="empty">표시할 탐색 기록이 없습니다.</div>';return}for(const n of rows){const row=document.createElement('div');row.className='row';const label=n.layer==null?n.variable:`L${n.layer}:N${n.neuron}`;const primary=byCompletion?n.completionOrder:byDuration?n.durationOrder:n.order,secondary=byCompletion?`방문 #${n.order}`:byDuration?`방문 #${n.order} · 완료 #${n.completionOrder}`:`완료 #${n.completionOrder}`;row.title=tip(n);row.innerHTML=`<span>#${primary}</span><span>${label} <b class="phase ${n.phase==='+'?'plus':'minus'}">${n.phase}</b> <small class="meta">Split #${n.splitNumber??'?'} · ${secondary}</small></span><span class="badge ${statusClass(n.result)}">${n.result}</span><span class="duration">${n.duration.toFixed(3)}s</span>`;box.appendChild(row)}}
function saveState(){window.name=STATE_PREFIX+JSON.stringify({mode,timelineOrder:$('#timelineOrder').value,status:$('#statusFilter').value,layer:$('#layerFilter').value,neuron:$('#neuronFilter').value})}
if(['completion','visit','duration'].includes(savedState.timelineOrder))$('#timelineOrder').value=savedState.timelineOrder;if(['ALL','SAT','UNSAT','CLOSED','UNKNOWN','TIMEOUT'].includes(savedState.status))$('#statusFilter').value=savedState.status;$('#layerFilter').value=savedState.layer||'';$('#neuronFilter').value=savedState.neuron||'';
function render(){renderTree();renderTimeline()}$('#defaultView').onclick=()=>{mode='default';saveState();renderTree()};$('#expandAll').onclick=()=>{mode='all';saveState();renderTree()};$('#collapseAll').onclick=()=>{mode='none';saveState();renderTree()};$('#statusFilter').onchange=()=>{saveState();render()};$('#layerFilter').oninput=()=>{saveState();render()};$('#neuronFilter').oninput=()=>{saveState();render()};$('#timelineOrder').onchange=()=>{saveState();renderTimeline()};render();
</script></body></html>'''
