"""Resolve report citations against the evidence sent for this analysis."""
import json
import re

from Automation.LogAnalysisIndex import LOG_FILES
from Automation.LogAnalysisScope import neuron_of


_BASE = r"(?:run|compare):(?:" + "|".join(map(re.escape, LOG_FILES)) + r"):[1-9]\d{0,9}"
_REFERENCE = re.compile(r"\[(" + _BASE + r")(?:#([1-9]\d{0,2}))?\]")
_EVIDENCE = re.compile(r'"(?:evidence|example_evidence)"\s*:\s*"(' + _BASE + r')"')


def _tree_location(record):
    layer, neuron = neuron_of(record)
    round_id, depth = record.get("round_index"), record.get("depth")
    phase = {"+": "활성", "-": "비활성"}.get(record.get("phase"))
    if None in (layer, neuron, round_id, depth, phase):
        return {}
    split = record.get("split_number")
    location = f"R{round_id}의 L{layer} N{neuron} 탐색 트리 · 깊이 {depth}"
    if type(split) is int:
        location += f" · Split #{split}"
    return {"label": location, "round": round_id, "depth": depth, "split": split, "phase": phase}


def restore_tree_citations(result):
    """Attach locations to older saved links using branch starts from their audit."""
    targets = [target for choices in result.get("citation_targets", {}).values() for target in choices
               if target.get("topic") == "tree" and target.get("branchId") and "phase" not in target]
    if not targets:
        return
    branches = {}
    for entry in result.get("queries", []):
        for row in entry.get("response", {}).get("records", []):
            record = row.get("data", {})
            if record.get("event") == "branch_start" and record.get("branch_id"):
                branches[record["branch_id"]] = record
    for target in targets:
        target.update(_tree_location(branches.get(target["branchId"], {})))


def cited_records(report, context, audit):
    if not (references := list(_REFERENCE.finditer(report))):
        return []
    # Only run evidence and read-only tool results count. Questions and past answers do not.
    supplied = set(_EVIDENCE.findall(json.dumps(context.get("runs", []))))
    if context.get("truncated"):
        supplied.update(_EVIDENCE.findall(context.get("preview", "")))
    for entry in audit:
        response = entry["response"]
        supplied.update(_EVIDENCE.findall(json.dumps(response)))
        if isinstance(response, dict) and response.get("truncated"):
            supplied.update(_EVIDENCE.findall(response.get("preview", "")))
    return list(dict.fromkeys((base, int(item) if item else None)
                              for base, item in (match.groups() for match in references)
                              if base in supplied))[:40]


def navigation_targets(index, citations):
    targets = {}
    if not citations:
        return targets
    lines_by_file = {}
    for reference, item in citations:
        run_id, file, line = reference.split(":")
        if run_id == "run":
            lines_by_file.setdefault(file, {}).setdefault(int(line), []).append((reference, item))
    for file, lines in lines_by_file.items():
        marks = ",".join("?" for _ in lines)
        for line, raw in index.db.execute(f"SELECT line,raw FROM events WHERE file=? AND line IN ({marks})",
                                          (file, *lines)):
            record = json.loads(raw)
            event, kind = record.get("event"), record.get("kind")
            round_id = record.get("round_index", record.get("round"))
            branch_id = record.get("branch_id")
            for reference, item in lines[line]:
                target = {"round": round_id, "line": line, "result": record.get("result", record.get("status")),
                          "seconds": record.get("duration_seconds")}
                if event in ("branch_start", "branch_end") and branch_id:
                    start = record
                    if event == "branch_end":
                        row = index.db.execute("SELECT raw FROM events WHERE file=? AND branch_id=? AND event='branch_start' LIMIT 1",
                                               (file, branch_id)).fetchone()
                        start = json.loads(row[0]) if row else {}
                    location = _tree_location(start)
                    if not location:
                        continue
                    target.update(topic="tree", branchId=branch_id, **location)
                elif event == "relu_stable" or kind == "bound_updates":
                    changes = record.get("records" if event == "relu_stable" else "changes") or []
                    selected = [(i, change) for i, change in enumerate(changes, 1)
                                if isinstance(change, dict) and (item is None or i == item)]
                    if len(selected) != 1 or type(round_id) is not int:
                        continue
                    i, change = selected[0]
                    layer, neuron = neuron_of(change)
                    if layer is None or neuron is None or not change.get("variable"):
                        continue
                    targets[reference + (f"#{item}" if item else "")] = [{
                        **target, "topic": "mechanisms", "label": f"R{round_id}의 L{layer} N{neuron} 경계 유도",
                        "stable": event == "relu_stable", "item": change.get("source_item", i),
                        "variable": change["variable"], "role": change.get("role"), "observedAt": change.get("observed_at"),
                        "before": change.get("before"), "after": change.get("after"),
                    }]
                    continue
                elif file == LOG_FILES[0]:
                    simplex = isinstance(event, str) and event.startswith("simplex_")
                    target.update(topic="simplex" if simplex else "dpll", label="Simplex · LP" if simplex else "DPLL · Theory")
                    target["call"] = record.get("call_number")
                    target["callId"] = record.get("call_id")
                elif file == LOG_FILES[2]:
                    target.update(topic="relu", label="ReLU 분기 이력", splitId=record.get("split_id"))
                else:
                    continue
                if item is None:
                    targets[reference] = [target]
    return targets
