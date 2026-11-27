"""Build a self-contained, interactive ReLU branch search-tree report."""
from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from html import escape
from math import isfinite
import json
import logging
import os
from pathlib import Path
import threading
from time import monotonic
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from visualization.NetworkLayout import parse_neuron_var
from visualization.AtomicFile import write_atomic_text
from visualization.Fonts import apply_web_font
from visualization.LiveSnapshot import initial_page_data
from visualization.SolveTrace import SolveTrace, TraceEvent
from visualization.SolverMechanisms import mechanism_report_path, write_solver_mechanisms_html


_TREE_SCRIPT = (Path(__file__).with_name("AssetRefresh.js").read_text(encoding="utf-8") + "\n" +
                Path(__file__).with_name("ReluSearchTreeClient.js").read_text(encoding="utf-8"))
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
    branch_id: Optional[str] = None
    completion_order: Optional[int] = None
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
            "branchId": self.branch_id,
            "completionSequence": self.completion_order,
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
    branches = [event for event in trace.events if event.component == "reluplex_branch"]
    split_spans = [event for event in trace.events if event.component == "reluplex_split"]
    split_numbers = {id(event): number for number, event in enumerate(split_spans, 1)}
    split_keys = {event.meta["split_key"]: split_numbers[id(event)]
                  for event in split_spans if event.meta.get("split_key") is not None}
    roots: List[SearchNode] = []
    stack: List[tuple[TraceEvent, SearchNode]] = []
    branch_nodes: Dict[str, SearchNode] = {}
    event_nodes: List[tuple[TraceEvent, SearchNode, str]] = []
    # Mechanism observations add events. Index only work-span timestamps once
    # rather than rescanning the expanded trace twice for every tree node.
    work_starts = {
        component: sorted(event.t_start for event in trace.events if event.component == component)
        for component in ("reluplex_repair", "simplex")
    }

    def count_work(event, component):
        starts = work_starts[component]
        end = float("inf") if event.t_end is None else event.t_end
        return bisect_right(starts, end) - bisect_left(starts, event.t_start)

    for order, event in enumerate(branches, 1):
        has_parent_id = "parent_branch_id" in event.meta
        while stack and (
            (event.depth or 0) <= (stack[-1][0].depth or 0)
            or (not has_parent_id and not _inside(event, stack[-1][0]))
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
            split_number = split_keys.get(event.meta.get("split_key"))
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
            repairs=count_work(event, "reluplex_repair"),
            simplex_calls=count_work(event, "simplex"),
            prunes_space=(raw_result == "UNSAT"),
            branch_id=event.meta.get("branch_id"),
            completion_order=event.meta.get("completion_order"),
        )
        if has_parent_id:
            parent = branch_nodes.get(event.meta["parent_branch_id"])
        else:
            parent = stack[-1][1] if stack else None
        if parent is not None:
            node.parent_id = parent.id
            parent.children.append(node)
        else:
            roots.append(node)
        if node.branch_id is not None:
            branch_nodes[node.branch_id] = node
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
            if all(node.completion_order is not None for _, node in decisive):
                _event, decisive_node = max(decisive, key=lambda item: item[1].completion_order)
            else:
                _event, decisive_node = max(decisive, key=lambda item: item[0].t_end)
            decisive_node.result = final_status

    return [root.to_dict() for root in roots]


def _branch_node_lookup(trace: SolveTrace):
    branches = [event for event in trace.events if event.component == "reluplex_branch"]
    node_by_branch = {event.meta["branch_id"]: node_id for node_id, event in enumerate(branches, 1)
                      if event.meta.get("branch_id") is not None}
    by_depth = defaultdict(list)
    if any(
        event.component in ("reluplex_stable", "reluplex_mechanism") and "branch_id" not in event.meta
        for event in trace.events
    ):
        for node_id, event in enumerate(branches, 1):
            by_depth[event.depth or 0].append((event.t_start, node_id, event))
        for items in by_depth.values():
            items.sort(key=lambda item: (item[0], item[1]))
    starts = {depth: [item[0] for item in items] for depth, items in by_depth.items()}

    def locate(event):
        if "branch_id" in event.meta:
            return node_by_branch.get(event.meta["branch_id"])
        # Compatibility for in-memory traces supplied without branch IDs.
        depth = event.depth or 0
        index = bisect_right(starts.get(depth, []), event.t_start) - 1
        if index >= 0:
            _, node_id, branch = by_depth[depth][index]
            same_round = (event.meta.get("round_index") is None
                          or branch.meta.get("round_index") is None
                          or event.meta["round_index"] == branch.meta["round_index"])
            if same_round and _inside(event, branch):
                return node_id
        return None

    return locate


def build_stable_records(trace: SolveTrace) -> List[Dict[str, Any]]:
    """Keep each fixation scoped to its theory call and containing branch."""
    locate = _branch_node_lookup(trace)
    records = []
    for event in trace.events:
        if event.component != "reluplex_stable":
            continue
        depth = event.depth or 0
        node_id = locate(event)
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
                "nodeId": node_id, "branchId": node_id, "depth": depth,
                "roundIndex": event.meta.get("round_index"),
            })
    return records


def preview_tree(full):
    return {**full, "roots": [{**root, "children": []} for root in full["roots"][:20]],
            "stableRecords": []}


def write_relu_search_tree_html(
    trace: SolveTrace,
    output_path: str | Path,
    *,
    overall_status: str,
    overall_reason: Optional[str] = None,
    title: str = "ReLU Search Tree",
    theory_stats: Optional[Dict[str, Any]] = None,
    mechanism_panels=None,
    live_snapshots=False,
    snapshot_only=False,
    snapshot_version=None,
    model_layer_sizes=None,
    tree_enabled=True,
    live_bound_history=None,
) -> Path:
    """Write an offline report; large histories use an adjacent snapshot file."""
    roots = build_relu_search_tree(trace, overall_status=overall_status)
    serialized_nodes: List[Dict[str, Any]] = []

    def collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            serialized_nodes.append(node)
            collect(node["children"])

    collect(roots)
    stats = theory_stats or {}
    if mechanism_panels is None or mechanism_panels:
        mechanisms = build_mechanism_report(
            trace, serialized_nodes,
            include_bound_history=(live_bound_history is None
                                   and (mechanism_panels is None or "bounds" in mechanism_panels)),
        )
        if live_bound_history is not None:
            mechanisms["boundHistoryFiles"] = live_bound_history.publish(trace)
        # An interrupted LP may not have emitted its final observation yet.
        if mechanisms["available"] and stats:
            for field, stat in (("cycles", "SIMPLEX_CYCLES_DETECTED"),
                                ("pivot_tightened", "PIVOT_TIGHTENED"),
                                ("full_tighten", "SIMPLEX_FULL_TIGHTEN")):
                mechanisms["totals"][field] = max(mechanisms["totals"].get(field, 0), stats.get(stat, 0))
        mechanism_path = mechanism_report_path(output_path)
        write_solver_mechanisms_html(
            mechanisms, mechanism_path, status=str(overall_status).upper(),
            live_snapshots=live_snapshots, snapshot_only=snapshot_only,
            snapshot_version=snapshot_version,
            title=f"Reluplex 검증 기법 · {title}", panels=mechanism_panels,
        )
        if not tree_enabled:
            return mechanism_path
    elif not tree_enabled:
        return Path(output_path)

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
        "modelLayerSizes": list(model_layer_sizes or []),
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
    if live_snapshots:
        payload["snapshotVersion"] = snapshot_version
    data = initial_page_data(
        output_path, payload, "relu-search-tree-snapshot",
        preview_tree,
        live_snapshots=live_snapshots, snapshot_only=snapshot_only,
    )
    if snapshot_only:
        return Path(output_path)
    data = data.replace("</", "<\\/")
    script = _TREE_SCRIPT.replace("__DATA__", data)
    document = _HTML.replace("__SCRIPT__", script).replace("__STYLES__", _TREE_STYLES).replace("__TITLE__", escape(title))
    return write_atomic_text(output_path, apply_web_font(document, output_path))


def branch_log_path(split_log_path: str | Path) -> Path:
    path = Path(split_log_path)
    return path.with_name(f"{path.stem}_branches.jsonl")


def _bound_change(event, change, change_id, node_id, source_item):
    return {**change, "id": change_id, "round_index": event.meta.get("round_index"),
            "depth": event.depth or 0, "nodeId": node_id,
            "sourceLine": event.meta.get("source_line"), "sourceItem": source_item}


def build_mechanism_report(trace: SolveTrace, nodes: List[Dict[str, Any]], *,
                           include_bound_history=True) -> Dict[str, Any]:
    """Locate observations by branch without modifying the tree's nodes."""
    locate = _branch_node_lookup(trace)
    node_by_id = {node["id"]: node for node in nodes}
    events = []
    settings = []
    bound_rounds = []
    bound_history = []
    for event in trace.events:
        if event.component not in {"reluplex_mechanism", "reluplex_stable"}:
            continue
        if event.meta.get("kind") == "bound_updates":
            if include_bound_history:
                node_id = locate(event)
                for item, change in enumerate(event.meta.get("changes", []), 1):
                    bound_history.append(_bound_change(event, change, len(bound_history) + 1, node_id, item))
            continue
        record = dict(event.meta)
        if event.component == "reluplex_stable":
            record["kind"] = "stable"
        for key, value in list(record.items()):
            if isinstance(value, float) and not isfinite(value):
                record[key] = None
                record[key + "_nonfinite"] = True
        record.update(depth=event.depth or 0, nodeId=None)
        if "ranges" in record:
            bound_rounds.append({"round_index": record.get("round_index"), "ranges": record.pop("ranges")})
        if record.get("kind") == "settings":
            settings.append(record)
            continue
        record["nodeId"] = locate(event)
        node = node_by_id.get(record["nodeId"])
        if node is not None and record.get("kind") == "backjump":
            ancestors = []
            parent = node_by_id.get(node.get("parentId"))
            while parent is not None:
                ancestors.append(parent)
                parent = node_by_id.get(parent.get("parentId"))
            record["causeNodeIds"] = [parent["id"] for parent in ancestors
                                      if parent["depth"] in record.get("cause_depths", [])]
            record["targetNodeId"] = next((parent["id"] for parent in ancestors
                                          if parent["depth"] == record.get("target_depth")), None)
        if record.get("kind") == "stable":
            for item, neuron in enumerate(record.get("records", []), 1):
                events.append({**neuron, "kind": "stable", "source": record.get("source"),
                               "round_index": record.get("round_index"), "depth": record["depth"],
                               "nodeId": record["nodeId"], "sequence": len(events) + 1,
                               "sourceLine": record.get("source_line"), "sourceItem": item})
            continue
        record["sequence"] = len(events) + 1
        events.append(record)
    totals = Counter()
    for record in events:
        kind = record.get("kind")
        totals[kind] += 1
        if kind == "bounds":
            totals["bounds_changed"] += record.get("changed", 0)
            totals["fixed"] += record.get("fixed", 0)
        elif kind == "conflict":
            totals["conflict_" + record.get("source", "unknown")] += 1
        elif kind == "repair":
            totals[record.get("rule", "unfixable")] += 1
            totals["repairPivots"] += int(record.get("pivoted", False))
        elif kind == "simplex":
            for key in ("cycles", "pivot_tightened", "full_tighten", "bound_conflicts"):
                totals[key] += record.get(key, 0)
        elif kind == "numerical":
            if record.get("action") == "check":
                totals["drift_checks"] += 1
                totals["restores"] += record.get("outcome") == "restored"
                totals["restore_failures"] += record.get("outcome") == "restore_failed"
                totals["max_drift"] = max(totals["max_drift"], record.get("drift") or 0)
    by_kind = {
        "bounds": [record for record in events if record.get("kind") == "stable"],
        "backjump": [record for record in events if record.get("kind") in {"backjump", "conflict"}],
        "numerical": [record for record in events if record.get("kind") == "numerical"],
        "underapprox": [],
    }
    return {"available": bool(settings or events), "settings": settings,
            "totals": dict(totals), "events": events, "eventsByKind": by_kind,
            "boundRounds": bound_rounds,
            "boundHistory": bound_history,
            "numericalChecks": [r for r in events if r.get("kind") == "numerical" and r.get("action") == "check"][-200:],
            "numericalPivots": [r for r in events if r.get("kind") == "numerical" and r.get("action") == "pivot"][-80:],
            "eventCount": len(events)}


def trace_from_branch_log(path: str | Path) -> SolveTrace:
    """Reconstruct branch spans from an append-only realtime JSONL log."""
    reader = BranchLogReader(path)
    reader.read(include_partial=True)
    return reader.trace


class BranchLogReader:
    """Retain parsed spans and consume only newly appended complete lines."""

    def __init__(self, path):
        self.path = Path(path)
        self.trace = SolveTrace()
        self.open_events: Dict[str, TraceEvent] = {}
        self.offset = 0
        self.line_number = 0
        self.revision = 0
        self._identity = None
        self._anchor = b""

    def read(self, *, include_partial=False):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = self._identity != identity or stat.st_size < self.offset
            if not reset and self.offset:
                stream.seek(self.offset - len(self._anchor))
                reset = stream.read(len(self._anchor)) != self._anchor
            if reset:
                self.trace = SolveTrace()
                self.open_events.clear()
                self.offset = 0
                self.line_number = 0
                self.revision += 1
            self._identity = identity
            stream.seek(self.offset)
            while stream.tell() < stat.st_size:
                line = stream.readline(stat.st_size - stream.tell())
                if not line or (not include_partial and not line.endswith(b"\n")):
                    break
                self.offset = stream.tell()
                self.line_number += 1
                try:
                    record = json.loads(line.decode("utf-8"))
                    if self._consume(record):
                        self.revision += 1
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
            stream.seek(max(0, self.offset - 64))
            self._anchor = stream.read(min(64, self.offset))

    def _open_branch_id(self, depth, round_index):
        branch_id = next(reversed(self.open_events), None)
        if branch_id is not None:
            branch = self.open_events[branch_id]
            branch_round = branch.meta.get("round_index")
            if branch.depth == depth and (round_index is None or branch_round is None or round_index == branch_round):
                return branch_id
        return None

    def _consume(self, record):
        trace, open_events = self.trace, self.open_events
        kind = record["event"]
        if kind not in {"relu_mechanism", "relu_stable", "branch_start", "branch_end"}:
            return False
        stamp_ns = record["monotonic_ns"] if "monotonic_ns" in record else record["time_ns"]
        stamp = int(stamp_ns) / 1_000_000_000
        if kind in {"relu_mechanism", "relu_stable"}:
            depth = int(record.get("depth", 0))
            meta = {key: value for key, value in record.items()
                    if key not in {"event", "timestamp", "time_ns", "monotonic_ns", "depth"}}
            meta["source_line"] = self.line_number
            if "branch_id" not in meta:
                # Older JSONL still identifies open branches by ID and append order.
                meta["branch_id"] = self._open_branch_id(depth, meta.get("round_index"))
            trace.events.append(TraceEvent(
                component="reluplex_mechanism" if kind == "relu_mechanism" else "reluplex_stable",
                t_start=stamp, t_end=stamp, depth=depth, meta=meta,
            ))
            return True
        if not record.get("branch_id"):
            return False
        branch_id = str(record["branch_id"])
        if kind == "branch_start":
            depth = int(record.get("depth", 0))
            round_index = record.get("round_index")
            parent_id = (record["parent_branch_id"] if "parent_branch_id" in record
                         else self._open_branch_id(depth - 1, round_index))
            event = TraceEvent(
                component="reluplex_branch",
                t_start=stamp,
                depth=depth,
                branch_x=str(record.get("variable", "")),
                meta={
                    "branch_id": branch_id,
                    "parent_branch_id": parent_id,
                    "phase": str(record.get("phase", "?")),
                    "layer": record.get("layer"),
                    "index": record.get("index"),
                    "bound": record.get("bound"),
                    "split_number": record.get("split_number"),
                    "round_index": round_index,
                    "result": "RUNNING",
                },
            )
            trace.events.append(event)
            open_events[branch_id] = event
        elif kind == "branch_end" and branch_id in open_events:
            event = open_events.pop(branch_id)
            event.t_end = stamp
            trace.record_branch_completion(event)
            event.meta["result"] = str(record.get("result", "UNKNOWN"))
            if record.get("reason"):
                event.meta["reason"] = str(record["reason"])
            if record.get("unsat_origin"):
                event.meta["unsat_origin"] = str(record["unsat_origin"])
        else:
            return False
        return True


class LiveBoundHistory:
    """Publish each bound change once, ahead of the snapshot that references it."""

    def __init__(self, output_path):
        self.directory = output_path.with_name(f"{output_path.stem}_bound_history")
        self.trace = None
        self.generation = None
        self.cursor = 0
        self.count = 0
        self.files = []

    def publish(self, trace):
        if self.trace is not trace:
            self.trace = trace
            self.generation = uuid4().hex
            self.cursor = self.count = 0
            self.files = []
        if self.cursor == len(trace.events):
            return self.files

        locate = _branch_node_lookup(trace)
        pending = []
        added_files = []

        def write_chunk():
            name = f"{self.generation}-{len(self.files) + len(added_files) + 1}.js"
            data = json.dumps(pending, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
            write_atomic_text(self.directory / name, f"document.currentScript.mechanismBounds={data};")
            added_files.append(f"{self.directory.name}/{name}")
            pending.clear()

        count = self.count
        for event in trace.events[self.cursor:]:
            if event.component == "reluplex_mechanism" and event.meta.get("kind") == "bound_updates":
                node_id = locate(event)
                for item, change in enumerate(event.meta.get("changes", []), 1):
                    count += 1
                    pending.append(_bound_change(event, change, count, node_id, item))
                    if len(pending) == 4096:
                        write_chunk()
        if pending:
            write_chunk()
        self.files.extend(added_files)
        self.cursor = len(trace.events)
        self.count = count
        return self.files


class TreeSnapshotWriter:
    """Own the incremental log reader and published state in the render process."""

    def __init__(self, log_path, output_path, title, mechanism_panels, model_layer_sizes, tree_enabled):
        self.reader = BranchLogReader(log_path)
        self.output_path = Path(output_path)
        self.title = title
        self.mechanism_panels = mechanism_panels
        self.model_layer_sizes = model_layer_sizes
        self.tree_enabled = tree_enabled
        self.published_signature = None
        self.bound_history = (LiveBoundHistory(self.output_path)
                              if mechanism_panels is None or "bounds" in mechanism_panels else None)

    def write(self, status, reason, stats, version):
        self.reader.read()
        signature = (self.reader.revision, status, reason, tuple(
            stats.get(key) for key in (
                "DERIVE_TOTAL_RELUS", "DERIVE_FIXED_RELUS", "DERIVE_REMAINING_RELUS",
                "SIMPLEX_CYCLES_DETECTED", "PIVOT_TIGHTENED", "SIMPLEX_FULL_TIGHTEN",
            )
        ))
        mechanisms_path = mechanism_report_path(self.output_path)
        has_pages = (
            (not self.tree_enabled or self.output_path.is_file())
            and (self.mechanism_panels == () or mechanisms_path.is_file())
        )
        if signature == self.published_signature and has_pages:
            return
        write_relu_search_tree_html(
            self.reader.trace, self.output_path, overall_status=status, overall_reason=reason,
            title=self.title, theory_stats=stats, mechanism_panels=self.mechanism_panels,
            live_snapshots=True, snapshot_version=version,
            snapshot_only=self.published_signature is not None and has_pages and status == "RUNNING",
            model_layer_sizes=self.model_layer_sizes, tree_enabled=self.tree_enabled,
            live_bound_history=self.bound_history,
        )
        self.published_signature = signature


class RealtimeSearchTreeVisualizer:
    """Coalescing background writer for the realtime interactive tree."""

    def __init__(
        self,
        log_path: str | Path,
        output_path: str | Path,
        *,
        theory_stats: Optional[Dict[str, Any]] = None,
        title: str = "Realtime ReLU Search Tree",
        mechanism_panels=None,
        update_interval_seconds: float = 3.0,
        model_layer_sizes=None,
        tree_enabled=True,
        render_in_process=False,
    ) -> None:
        self.log_path = Path(log_path)
        self.output_path = Path(output_path)
        self.status = "RUNNING"
        self.reason: Optional[str] = None
        self.theory_stats = theory_stats
        self.title = title
        self.mechanism_panels = mechanism_panels
        self.model_layer_sizes = list(model_layer_sizes or [])
        self._lock = threading.Lock()
        self._pending = False
        self._worker: Optional[threading.Thread] = None
        self.last_write_error: OSError | RuntimeError | None = None
        self._render_in_process = render_in_process
        self._config = (
            str(self.log_path), str(self.output_path), self.title,
            None if mechanism_panels is None else tuple(mechanism_panels),
            tuple(self.model_layer_sizes), tree_enabled,
        )
        self._local_writer: Optional[TreeSnapshotWriter] = None
        self._snapshot_version = 0
        self._update_interval = max(0.0, float(update_interval_seconds))
        self._last_write_started = float("-inf")
        self._wake = threading.Event()
        self._force = False

    def set_overall(self, status: str, reason: Optional[str] = None) -> None:
        with self._lock:
            self.status = str(status).upper()
            self.reason = reason
            if self.status != "RUNNING":
                self._wake.set()

    def _write_once(self) -> None:
        with self._lock:
            status, reason = self.status, self.reason
        stats = dict(self.theory_stats or {})
        self._snapshot_version += 1
        if self._render_in_process:
            from visualization.RenderProcess import submit_tree_render
            try:
                submit_tree_render(self._config, status, reason, stats, self._snapshot_version).result()
                return
            except (OSError, RuntimeError):
                if status == "RUNNING":
                    raise
                # A child failure must not prevent publication of the final report.
        if self._local_writer is None:
            self._local_writer = TreeSnapshotWriter(*self._config)
        self._local_writer.write(status, reason, stats, self._snapshot_version)

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    delay = (0.0 if self._force or self.status != "RUNNING" else
                             max(0.0, self._last_write_started + self._update_interval - monotonic()))
                self._wake.wait(delay)
                with self._lock:
                    self._pending = False
                    self._force = False
                    self._wake.clear()
                self._last_write_started = monotonic()
                try:
                    self._write_once()
                    self.last_write_error = None
                except (OSError, RuntimeError) as exc:
                    if self.last_write_error is None:
                        logging.getLogger(__name__).warning(
                            "시각화 파일 갱신 실패; 다음 갱신에서 재시도합니다: %s", exc
                        )
                    self.last_write_error = exc
                with self._lock:
                    if not self._pending:
                        self._worker = None
                        return
        finally:
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None

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
                if worker is not None:
                    self._force = True
                    self._wake.set()
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
<div class="controls"><button id="defaultView">루트만 보기</button><button id="recentPath" title="루트부터 가장 최근에 추가된 가지까지 펼칩니다">최근 경로보기</button><button id="trackRecentPath" aria-pressed="false" title="켜 두면 새 가지가 추가될 때마다 최근 경로를 자동으로 펼쳐 따라갑니다">최근 경로 추적</button><button id="toggleReportPanels" aria-controls="timelinePanel stablePanel" aria-expanded="true">하단 패널 접기</button><button id="expandAll">모두 펼치기</button><button id="collapseAll">모두 접기</button><div class="tree-filters" role="group" aria-label="결과 필터 및 정렬"><select id="statusFilter" aria-label="결과 필터"><option value="ALL">모든 결과</option><option value="FINAL">최종 결과</option><option value="STABLE">STABLE</option><option value="SAT_FOUND">SAT 발견</option><option value="UNSAT_FOUND">UNSAT 발견</option><option value="CLOSED">CLOSED 전체</option><option value="CLOSED_SAT">CLOSED (SAT)</option><option value="CLOSED_UNSAT">CLOSED (UNSAT)</option><option>RUNNING</option></select><select id="treeOrder" aria-label="노드 정렬" title="노드순은 트리 구조, 나머지는 현재 범위의 노드를 정렬한 목록으로 표시합니다"><option value="node">노드순</option><option value="layer">레이어순</option><option value="duration">소요 시간순(긴 순)</option><option value="completion">완료 시간순(빠른 순)</option></select></div><div class="ancestor-search" role="group" aria-label="뉴런 트리 검색"><input id="layerFilter" inputmode="numeric" placeholder="Layer 번호"><input id="neuronFilter" inputmode="numeric" placeholder="Neuron 번호"><button id="findAncestors" aria-pressed="false" title="입력한 뉴런의 모든 탐색 위치에서 루트까지의 상위 경로를 보여줍니다">상위 트리 찾기</button><button id="findSubtrees" aria-pressed="false" title="입력한 뉴런이 탐색된 노드와 하위 트리를 보여줍니다">서브 트리 찾기</button><span id="ancestorInfo" class="meta" role="status"></span></div></div></header>
<main>
<section class="panel"><div id="scopeBar" class="controls subtree-controls"><strong id="scopeLabel" aria-live="polite">전체 트리</strong><span id="scopeCount" class="meta"></span><button id="scopeParent" disabled>부모 가지 보기</button><button id="scopeAll" disabled>전체 트리로 돌아가기</button></div><div id="treeViewport" class="tree-viewport" role="region" aria-label="ReLU 탐색 트리" tabindex="0"><ul id="tree" class="tree"></ul></div><div class="help">L은 Layer(신경망 층), N은 Neuron(해당 층의 뉴런)을 뜻하며 번호는 0부터 시작합니다. 예: L1:N36은 첫 번째 은닉층의 37번째 뉴런입니다.</div>
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
