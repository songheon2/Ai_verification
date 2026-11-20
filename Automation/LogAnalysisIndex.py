"""Shared read-only event schema, queries and completed-run indexing."""
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
from Automation.LpTelemetry import metric_available
from visualization.VisualizationMode import LOG_DIRECTORY, resolve_visualization_output


LOG_FILES = ("solver_progress.jsonl", "relu_events_branches.jsonl", "relu_events.jsonl")
MAX_LINE_BYTES = 1024 * 1024
EVENT_SCHEMA = ("CREATE TABLE events (file TEXT, line INTEGER, event TEXT, kind TEXT, "
                "round INTEGER, call_id TEXT, variable TEXT, layer INTEGER, neuron INTEGER, "
                "depth INTEGER, result TEXT, duration REAL, pivots INTEGER, raw TEXT, "
                "branch_id TEXT, time_ns INTEGER, end_offset INTEGER)")


def live_run_fingerprint(directory, run_id):
    return hashlib.sha256((str(directory) + ":" + str(run_id or "legacy")).encode()).hexdigest()


def run_log_path(directory, name):
    organized = directory / LOG_DIRECTORY / name
    path = organized if organized.exists() else directory / name
    if path.resolve().parent != path.parent:
        raise ValueError("Log symlinks outside the selected run are not allowed")
    return path


def create_event_table(db):
    db.execute(EVENT_SCHEMA)
    for name, columns in (("by_event", "event, round"), ("by_call", "call_id"),
                          ("by_neuron", "layer, neuron"), ("by_branch", "branch_id")):
        db.execute(f"CREATE INDEX {name} ON events({columns})")


def insert_event(db, name, line, item, end_offset):
    """Normalize completed and live records identically, without changing raw data."""
    def integer(value):
        return value if type(value) is int and -(2**63) <= value < 2**63 else None

    def numeric(value):
        try:
            return value if type(value) in (int, float) and math.isfinite(value) else None
        except OverflowError:
            return None

    def text(value):
        return value if isinstance(value, str) else None

    db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        name, line, text(item.get("event")), text(item.get("kind")),
        integer(item.get("round", item.get("round_index"))), text(item.get("call_id")),
        text(item.get("variable", item.get("violated_var"))), integer(item.get("layer")),
        integer(item.get("index")), integer(item.get("depth")), text(item.get("result")),
        numeric(item.get("duration_seconds")),
        integer(item.get("pivots")) if metric_available(item, "pivots") else None,
        json.dumps(item, ensure_ascii=False), text(item.get("branch_id")),
        integer(item.get("time_ns")), end_offset))


def safe_data(value):
    """Minimize accidental local path/credential disclosure (not a DLP guarantee)."""
    if isinstance(value, dict):
        result = {str(k): ("[REDACTED]" if re.search(r"api.?key|password|secret|token$", str(k), re.I)
                         else safe_data(v)) for k, v in value.items()}
        if value.get("work_counts_available") is False or any(
                key in value for key in ("pivot_count_available", "iteration_count_available", "cycle_count_available")):
            for key in ("pivots", "iterations", "cycles"):
                if key in result and not metric_available(value, key):
                    result[key] = None
            result["work_counts_meaning"] = "항목별 *_count_available이 우선이다. 미수집 항목의 원시 0은 실제 0회가 아니다."
        return result
    if isinstance(value, (list, tuple)):
        return [safe_data(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        value = re.sub(r"AIza[\w-]{20,}", "[REDACTED]", value)
        if re.match(r"^(?:[A-Za-z]:[\\/]|/|\\\\)", value):
            return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return value


def recorded_context(result, manifest=None):
    """Combine only recorded fields; a mismatched manifest is never evidence."""
    result, manifest = result or {}, manifest or {}
    conflict = bool(result.get("run_id") and manifest.get("run_id") and result["run_id"] != manifest["run_id"])
    if conflict:
        manifest = {}
    merged = {key: result.get(key, manifest.get(key)) for key in
              ("model", "vnnlib", "solver", "status", "note", "solver_variant", "sampling")}
    old_solver = manifest.get("solver") if isinstance(manifest.get("solver"), dict) else {}
    new_solver = result.get("solver") if isinstance(result.get("solver"), dict) else {}
    if old_solver or new_solver:
        merged["solver"] = {**old_solver, **new_solver}
    merged["context_provenance"] = {"result_present": bool(result), "manifest_present": bool(manifest),
                                    "manifest_identity_conflict": conflict,
                                    "meaning": "저장된 값만 결합했다. 현재 코드 기본값으로 채우지 않았다."}
    return safe_data(merged)


class RunIndex:
    """Temporary disk-backed index. Never change the selected run's files."""

    def __init__(self, directory, run_id, max_bytes=256 * 1024 * 1024):
        self.root = resolve_visualization_output(directory).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("--run/--compare must be run directories")
        self.run_id = run_id
        self.temp = tempfile.TemporaryDirectory(prefix="verification-log-analysis-")
        self.db = sqlite3.connect(str(Path(self.temp.name) / "events.sqlite"))
        try:
            result_path = self._file("result.json")
            if not result_path.exists():
                raise ValueError("Completed result.json is required for CLI analysis; use --serve for live runs")
            if result_path.stat().st_size > MAX_LINE_BYTES:
                raise ValueError("result.json exceeds the 1 MiB safety limit")
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if not isinstance(result, dict):
                raise ValueError("result.json must contain an object")
            manifest = {}
            manifest_path = self._file("run_info.json")
            if manifest_path.is_file() and manifest_path.stat().st_size <= MAX_LINE_BYTES:
                try:
                    content = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                    if isinstance(content, dict):
                        manifest = content
                except (ValueError, UnicodeError):
                    pass
            self.result = recorded_context(result, manifest)
            create_event_table(self.db)
            self.files = []
            remaining = max_bytes
            for name in LOG_FILES:
                path = self._file(name)
                if not path.exists():
                    self.files.append({"file": name, "missing": True})
                    continue
                size = path.stat().st_size
                info = {"file": name, "size_bytes": size, "indexed_bytes": 0,
                        "records": 0, "invalid_lines": 0, "truncated": False}
                with path.open("rb") as stream:
                    for number in range(1, 10**12):
                        # Do not materialize oversized lines or read beyond the total budget.
                        allowance = min(MAX_LINE_BYTES + 1, remaining)
                        if allowance <= 0:
                            break
                        raw = stream.readline(allowance)
                        if not raw:
                            break
                        remaining -= len(raw)
                        info["indexed_bytes"] += len(raw)
                        if len(raw) > MAX_LINE_BYTES or (not raw.endswith(b"\n") and stream.tell() < size):
                            info["invalid_lines"] += 1
                            break
                        try:
                            item = json.loads(raw)
                            if not isinstance(item, dict):
                                raise ValueError()
                        except (ValueError, UnicodeError):
                            info["invalid_lines"] += 1
                            continue
                        insert_event(self.db, name, number, item, info["indexed_bytes"])
                        info["records"] += 1
                info["truncated"] = info["indexed_bytes"] < size
                self.files.append(info)
            self.db.commit()
            self.analysis_settings = [safe_data(json.loads(raw)) for (raw,) in self.db.execute(
                "SELECT raw FROM events WHERE event='relu_mechanism' AND kind='settings' ORDER BY rowid DESC LIMIT 20")][::-1]
        except BaseException:
            self.close()
            raise

    def _file(self, name):
        if name in LOG_FILES or name == "relu_events_meta.json":
            return run_log_path(self.root, name)
        path = self.root / name
        if path.resolve().parent != self.root:
            raise ValueError("Log symlinks outside the selected run are not allowed")
        return path

    def close(self):
        self.db.close()
        self.temp.cleanup()

    def summary(self):
        counts = self.db.execute("SELECT file,event,kind,COUNT(*) FROM events GROUP BY file,event,kind").fetchall()
        calls = self.db.execute("SELECT COUNT(*),SUM(duration),SUM(pivots) FROM events "
                                "WHERE event IN ('simplex_end','simplex_call')").fetchone()
        result = {"run_id": self.run_id, "directory_name": self.root.name, "result": self.result,
                "source": f"{self.run_id}:result.json", "files": self.files,
                "indexed_event_counts": [{"file": f, "event": e, "kind": k, "count": n}
                                         for f, e, k, n in counts],
                "indexed_completed_simplex": {"calls": calls[0], "seconds": calls[1], "pivots": calls[2]},
                "limitations": ["Aggregates cover indexed records only; missing logs are not zero events.",
                                "Historical code version/hardware/sampling may be unknown.",
                                "branch_end may require querying branch_start by branch_id."]}
        from Automation.LogAnalysisKnowledge import analysis_support
        result["analysis_support"] = analysis_support("all", self.analysis_settings, self.result,
                                                     result["indexed_event_counts"], self.files)
        return result

    def query(self, args):
        columns = {"file": "file", "event": "event", "kind": "kind", "round": "round",
                   "call_id": "call_id", "variable": "variable", "layer": "layer", "neuron": "neuron",
                   "depth": "depth", "result": "result", "branch_id": "branch_id"}
        predicates, params = [], []
        for key, column in columns.items():
            if key in args:
                if type(args[key]) not in (int, str):
                    raise ValueError("Filters must be strings or integers")
                predicates.append(f"{column}=?")
                params.append(args[key])
        start = max(1, int(args.get("start_line", 1)))
        limit = min(50, max(1, int(args.get("limit", 10))))
        predicates.append("line>=?")
        params.append(start)
        for key, operator in (("after_ns", ">="), ("before_ns", "<=")):
            if key in args:
                predicates.append(f"time_ns{operator}?")
                params.append(int(args[key]))
        rows = self.db.execute("SELECT file,line,raw FROM events WHERE " + " AND ".join(predicates)
                               + " ORDER BY file,line LIMIT ?", [*params, limit]).fetchall()
        return {"run_id": self.run_id, "filters": args, "indexed_scope": self.files,
                "records": [{"evidence": f"{self.run_id}:{file}:{line}", "data": safe_data(json.loads(raw))}
                            for file, line, raw in rows]}

    def rank(self, args):
        limit = min(20, max(1, int(args.get("limit", 10))))
        metric = args.get("metric", "slow_calls")
        if metric in ("slow_calls", "pivot_calls"):
            order = "duration" if metric == "slow_calls" else "pivots"
            rows = self.db.execute(f"SELECT file,line,raw FROM events WHERE event IN "
                                   f"('simplex_end','simplex_call') ORDER BY {order} DESC LIMIT ?", (limit,)).fetchall()
            values = [{"evidence": f"{self.run_id}:{f}:{n}", "data": safe_data(json.loads(raw))} for f, n, raw in rows]
        elif metric in ("split_neurons", "branch_neurons"):
            event = "+" if metric == "split_neurons" else "branch_start"
            rows = self.db.execute("SELECT layer,neuron,variable,COUNT(*),file,MIN(line) "
                                   "FROM events WHERE event=? GROUP BY file,layer,neuron,variable "
                                   "ORDER BY COUNT(*) DESC,layer,neuron LIMIT ?", (event, limit)).fetchall()
            values = [{"layer": layer, "neuron": neuron, "variable": var, "count": count,
                       "example_evidence": f"{self.run_id}:{file}:{line}"}
                      for layer, neuron, var, count, file, line in rows]
        else:
            raise ValueError("Unknown metric")
        return {"run_id": self.run_id, "metric": metric, "indexed_scope": self.files, "ranking": values}

    def simplex_diagnostics(self, args=None):
        from Automation.SimplexLogDiagnostics import diagnose_simplex
        return diagnose_simplex(self, args)
