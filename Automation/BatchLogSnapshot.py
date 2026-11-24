"""Read compact evidence from the log bytes visible when batch analysis was requested."""
from collections import Counter
import json
import os
from statistics import median

from Automation.LogAnalysisIndex import LOG_FILES, run_log_path
from Automation.LpTelemetry import metric_available
from Automation.SimplexLogDiagnostics import MECHANISM_KINDS, attach_prior_mechanisms

MAX_LOG_WINDOW_BYTES = 1024 * 1024
MAX_SOLVER_LOG_WINDOW_BYTES = 8 * MAX_LOG_WINDOW_BYTES
EVENTS = {"run_context", "round_start", "round_end", "theory_result", "simplex_start", "simplex_call",
          "simplex_end", "solver_end", "branch_start", "branch_end", "relu_stable",
          "relu_mechanism", "+"}


def capture_log_limits(directory):
    limits = []
    for name in LOG_FILES:
        try:
            path = run_log_path(directory, name)
            stat = path.stat()
            if path.is_file() and stat.st_size:
                limits.append((path, stat.st_size, stat.st_dev, stat.st_ino))
        except (OSError, ValueError):
            continue
    return limits


def _read_window(path, size, device, inode):
    try:
        maximum = MAX_SOLVER_LOG_WINDOW_BYTES if path.name == "solver_progress.jsonl" else MAX_LOG_WINDOW_BYTES
        offset = max(0, size - maximum)
        with path.open("rb") as stream:
            if path.resolve().parent != path.parent:
                return None
            actual = os.fstat(stream.fileno())
            if (actual.st_dev, actual.st_ino) != (device, inode) or actual.st_size < size:
                return None
            if offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    stream.readline(size - offset)
            return stream.read(max(0, size - stream.tell())), bool(offset)
    except OSError:
        return None


def _call_comparison(profiles):
    if not profiles:
        return None
    width = max(1, len(profiles) // 4)
    slowest = sorted(profiles, key=lambda item: item["seconds"], reverse=True)[:5]
    for target in slowest:
        if target.get("round") is None or target.get("origin") is None:
            continue
        peers = sorted((p for p in profiles if p is not target and p.get("round") == target["round"]
                        and p.get("origin") == target["origin"]), key=lambda p: p["seconds"])
        counts = [p["iterations"] for p in peers if "iterations" in p]
        middle = len(peers) // 2
        target["same_round_and_origin"] = {
            "calls": len(peers), "iteration_median": median(counts) if counts else None,
            "median_duration_calls": [{key: p.get(key) for key in ("call_number", "iterations", "seconds")}
                                      for p in peers[max(0, middle-1):middle+2]],
        }
    result = {"slowest_calls": slowest}
    if len(profiles) >= 2:
        result["earlier_sample"], result["later_sample"] = (
            {"first_call": group[0].get("call_number"), "last_call": group[-1].get("call_number"),
             "calls": len(group), "median_seconds": round(median(item["seconds"] for item in group), 6),
             "max_seconds": round(max(item["seconds"] for item in group), 6)}
            for group in (profiles[:width], profiles[-width:])
        )
    return result


def summarize_logs(limits):
    summaries = []
    timed_calls = []
    mechanism_events = []
    for path, size, device, inode in limits:
        window = _read_window(path, size, device, inode)
        if window is None:
            continue
        raw, tail_only = window
        counts = Counter()
        mechanisms = Counter()
        latest = {}
        profiles = []
        starts = {}
        pivots = pivot_calls = calls = 0
        for line in raw.split(b"\n")[:-1]:
            try:
                item = json.loads(line)
            except (UnicodeError, ValueError):
                continue
            event = item.get("event") if isinstance(item, dict) else None
            if not isinstance(event, str) or event not in EVENTS:
                continue
            counts[event] += 1
            if event == "relu_mechanism" and isinstance(item.get("kind"), str):
                mechanisms[item["kind"][:60]] += 1
                if item["kind"] in MECHANISM_KINDS and type(item.get("time_ns")) is int:
                    mechanism_events.append(item)
            call_id = item.get("call_id")
            if event == "simplex_start" and isinstance(call_id, str):
                starts[call_id] = {key: item[key] for key in (
                    "call_number", "round", "depth", "origin", "row_count", "variable_count",
                ) if key in item}
                starts[call_id]["start_ns"] = item.get("time_ns")
            if event in ("simplex_call", "simplex_end"):
                calls += 1
                metadata = starts.pop(call_id, {}) if isinstance(call_id, str) else {}
                metadata.update({key: item[key] for key in (
                    "call_number", "round", "depth", "origin", "row_count", "variable_count", "result",
                    "glpk_profile", "start_ns",
                ) if key in item})
                if metric_available(item, "iterations") and type(item.get("iterations")) is int:
                    metadata["iterations"] = item["iterations"]
                duration = item.get("duration_seconds")
                if type(duration) in (int, float) and 0 <= duration < float("inf"):
                    profiles.append({**metadata, "seconds": duration})
                pivot_count = item.get("pivots")
                if type(pivot_count) is int and metric_available(item, "pivots"):
                    pivot_calls += 1
                    pivots += pivot_count
            if event in ("round_end", "theory_result", "solver_end", "simplex_start",
                         "simplex_call", "simplex_end", "branch_end"):
                latest[event] = {key: item[key] for key in (
                    "round", "round_index", "depth", "call_id", "call_number", "status", "result", "reason",
                ) if key in item}
        if counts:
            comparison = _call_comparison(profiles)
            if comparison is not None:
                timed_calls.extend(comparison["slowest_calls"])
            summaries.append({
                "source": path.name, "tail_only": tail_only, "recorded_events": dict(counts),
                "last_event_by_type": latest, "mechanism_counts": dict(mechanisms.most_common(12)),
                "completed_simplex": {"calls": calls,
                                      "seconds": round(sum(item["seconds"] for item in profiles), 6) if profiles else None,
                                      "pivots": pivots if pivot_calls else None,
                                      "pivot_measured_calls": pivot_calls} if calls else None,
                "simplex_timing_comparison": comparison,
            })
    if mechanism_events:
        attach_prior_mechanisms(timed_calls, sorted(mechanism_events, key=lambda item: item["time_ns"]))
    return summaries
