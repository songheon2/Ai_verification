"""Incremental, read-only JSONL indexing and immutable analysis snapshots.

Runs in the analysis bridge, never in the solver. Source handles are short lived;
an API request reads only its private SQLite snapshot, not growing source logs.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time

from Automation.LogAnalysisIndex import (
    LOG_FILES, MAX_LINE_BYTES, RunIndex, create_event_table, insert_event, live_run_fingerprint, safe_data, recorded_context,
)
from Automation.LogAnalysisScope import MECHANISMS, TOPICS, neuron_of


class FrozenRunIndex(RunIndex):
    def summary(self):
        result = super().summary()
        pending = self.db.execute(
            "SELECT s.call_id,s.round,s.time_ns FROM events s WHERE s.event='simplex_start' "
            "AND NOT EXISTS (SELECT 1 FROM events e WHERE e.call_id=s.call_id "
            "AND e.event IN ('simplex_end','simplex_call')) ORDER BY s.line LIMIT 21").fetchall()
        result.update(source="run:" + self.snapshot["context_source"],
                      snapshot=self.snapshot,
                      sampling=self.sampling,
                      ongoing_simplex={"calls": [{"call_id": call, "round": rnd,
                          "elapsed_at_snapshot_seconds": max(0, (self.snapshot["time_ns"] - stamp) / 1e9)
                          if stamp else None} for call, rnd, stamp in pending[:20]],
                          "list_limited": len(pending) > 20})
        result["limitations"].extend([
            "Frozen at snapshot time; later records are excluded from every tool query.",
            "Ongoing calls have no final result/duration. Elapsed time is not completion time.",
            "Only flushed, newline-terminated records are visible. Compact logs may omit call starts.",
        ])
        return result


class LiveRunIndex(RunIndex):
    def __init__(self, directory, max_bytes=256 * 1024 * 1024):
        self.root = Path(directory).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("--run must be a run directory")
        self.run_id = "run"
        self.max_bytes = max_bytes
        self.temp = tempfile.TemporaryDirectory(prefix="verification-live-index-")
        self.db = sqlite3.connect(str(Path(self.temp.name) / "events.sqlite"), check_same_thread=False)
        self.lock = threading.RLock()
        self.generation = 0
        self.manifest_id = None
        self.has_refreshed = False
        self._reset()
        try:
            self.refresh()
        except BaseException:
            self.close()
            raise

    def _reset(self):
        self.db.execute("DROP TABLE IF EXISTS events")
        create_event_table(self.db)
        self.states = {}
        self.rounds, self.calls, self.neurons, self.settings = set(), [], {}, []
        self.call_set = set()
        self.last_call = None
        self.generation += 1
        self.records = 0

    def _json(self, name):
        path = self._file(name)
        if not path.is_file():
            return {}
        if path.stat().st_size > MAX_LINE_BYTES:
            raise ValueError(f"{name} exceeds the 1 MiB safety limit")
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except (ValueError, UnicodeError):
            return {}  # A legacy writer may be replacing its small metadata file.

    def _remember(self, item):
        rnd = item.get("round", item.get("round_index"))
        if type(rnd) is int and 0 < rnd <= 1000000:
            self.rounds.add(rnd)
        call = item.get("call_id")
        if isinstance(call, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", call):
            self.last_call = call
            if call not in self.call_set and len(self.calls) < 2000:
                self.calls.append(call)
                self.call_set.add(call)
        if item.get("kind") == "settings":
            self.settings = (self.settings + [safe_data(item)])[-20:]
        if item.get("event") not in ("relu_stable", "relu_mechanism", "branch_start", "+"):
            return
        entries = [item]
        for key in ("records", "ranges", "changes"):
            if isinstance(item.get(key), list):
                entries.extend(v for v in item[key] if isinstance(v, dict))
        for record in entries:
            layer, neuron = neuron_of(record)
            if layer is not None and 0 <= layer <= 1000000 and 0 <= neuron <= 1000000:
                self.neurons.setdefault(str(layer), set()).add(neuron)

    def refresh(self, _retry=False):
        """Capture byte limits first; never chase a writer past these boundaries."""
        with self.lock:
            stamp = time.time_ns()
            manifest = self._json("run_info.json")
            manifest_id = manifest.get("run_id")
            epoch = manifest.get("log_epoch_ns", 0)
            epoch = epoch if type(epoch) is int else 0
            # Final context must precede the byte limits. Otherwise a result that
            # appears during collection could label an earlier, incomplete log final.
            result = self._json("result.json")
            if manifest_id and result.get("run_id") != manifest_id:
                result = {}
            metadata = self._file("relu_events_meta.json")
            split_metadata = self._json("relu_events_meta.json") if metadata.is_file() and metadata.stat().st_mtime_ns >= epoch else {}
            changed = self.has_refreshed and manifest_id != self.manifest_id
            limits = {}
            for name in LOG_FILES:
                path = self._file(name)
                if not path.is_file():
                    if name in self.states:
                        changed = True
                    continue
                stat = path.stat()
                if stat.st_mtime_ns < epoch:
                    if name in self.states:
                        changed = True
                    continue  # A reused output folder can still contain previous-run logs.
                identity = (stat.st_dev, stat.st_ino)
                limits[name] = (stat.st_size, identity)
                old = self.states.get(name)
                if old:
                    if identity != old["identity"] or stat.st_size < old["offset"]:
                        changed = True
                    elif old["offset"]:
                        with path.open("rb") as stream:
                            stream.seek(old["offset"] - len(old["anchor"]))
                            if stream.read(len(old["anchor"])) != old["anchor"]:
                                changed = True
            if changed:
                self._reset()
            self.manifest_id = manifest_id
            self.result = recorded_context(result, manifest)
            self.sampling = safe_data(manifest.get("sampling", {}))
            self.split_metadata = split_metadata
            for entry in self.split_metadata.get("round_starts", []):
                if isinstance(entry, dict) and type(entry.get("round")) is int and 0 < entry["round"] <= 1000000:
                    self.rounds.add(entry["round"])
            remaining = self.max_bytes - sum(s["offset"] for s in self.states.values())
            bytes_read = 0
            for name, (limit, identity) in limits.items():
                state = self.states.setdefault(name, {"offset": 0, "line": 0, "records": 0,
                    "invalid_lines": 0, "identity": identity, "anchor": b"", "oversized": False})
                state["size"] = limit
                with self._file(name).open("rb") as stream:
                    stream.seek(state["offset"])
                    while stream.tell() < limit and remaining > 0 and not state["oversized"]:
                        before = stream.tell()
                        raw = stream.readline(min(MAX_LINE_BYTES + 1, limit - before, remaining))
                        bytes_read += len(raw)
                        if len(raw) > MAX_LINE_BYTES:
                            state["oversized"] = True
                            break
                        if not raw.endswith(b"\n"):
                            break  # Leave the incomplete tail for a future refresh.
                        state["offset"] = stream.tell()
                        state["line"] += 1
                        remaining -= len(raw)
                        state["anchor"] = (state["anchor"] + raw)[-64:]
                        try:
                            item = json.loads(raw)
                            if not isinstance(item, dict):
                                raise ValueError()
                        except (ValueError, UnicodeError):
                            state["invalid_lines"] += 1
                            continue
                        insert_event(self.db, name, state["line"], item, state["offset"])
                        self._remember(item)
                        state["records"] += 1
                        self.records += 1
            self.db.commit()
            context_source = "result.json" if result else "run_info.json" if manifest else "available_logs"
            if not self.result.get("model"):
                row = self.db.execute("SELECT raw FROM events WHERE event='run_context' ORDER BY line LIMIT 1").fetchone()
                if row:
                    witness = json.loads(row[0]).get("witness_context") or {}
                    if isinstance(witness, dict):
                        self.result["model"] = safe_data({"name": witness.get("model_name"),
                                                         "layer_sizes": witness.get("model_layer_sizes")})
                        if not manifest and not result:
                            context_source = "solver_progress.jsonl"
            reset_during_read = self._json("run_info.json").get("run_id") != manifest_id
            for name, s in self.states.items():
                path = self._file(name)
                if not path.is_file():
                    reset_during_read = True
                    continue
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) != s["identity"] or stat.st_size < s["offset"]:
                    reset_during_read = True
                elif s["offset"]:
                    with path.open("rb") as stream:
                        stream.seek(s["offset"] - len(s["anchor"]))
                        if stream.read(len(s["anchor"])) != s["anchor"]:
                            reset_during_read = True
            if reset_during_read:
                self._reset()
                self.has_refreshed = False
                if _retry:
                    raise OSError("Logs changed generation while being read; retry after the writer resets them")
                return self.refresh(_retry=True)
            self.files = [{"file": name, "size_bytes": s["size"], "indexed_bytes": s["offset"],
                           "records": s["records"], "invalid_lines": s["invalid_lines"],
                           "truncated": s["offset"] < s["size"], "oversized_line": s["oversized"]}
                          if (s := self.states.get(name)) else {"file": name, "missing": True} for name in LOG_FILES]
            self.fingerprint = live_run_fingerprint(self.root, manifest_id)
            self.identity = self.fingerprint + ":" + str(self.generation)
            self.state = {"identity": self.identity, "time_ns": stamp, "context_source": context_source,
                          "captured_at": datetime.fromtimestamp(stamp / 1e9, timezone.utc).isoformat(),
                          "finished": bool(result) or manifest.get("status") == "CANCELLED",
                          "status": self.result.get("status") or "RUNNING",
                          "indexed_records": self.records, "source_bytes": sum(n for n, _ in limits.values()),
                          "indexed_bytes": sum(s["offset"] for s in self.states.values()),
                          "last_round": max(self.rounds, default=None),
                          "last_call": self.last_call,
                          "bytes_read_this_refresh": bytes_read,
                          "coverage_limited": any(f.get("truncated") or f.get("invalid_lines") for f in self.files)}
            self.has_refreshed = True
            return deepcopy(self.state)

    def options(self):
        return {"topics": TOPICS, "mechanisms": MECHANISMS, "rounds": sorted(self.rounds), "calls": list(self.calls),
                "calls_list_limited": len(self.calls) == 2000,
                "neurons": {layer: sorted(values) for layer, values in self.neurons.items()}}

    def freeze(self):
        with self.lock:
            self.refresh()
            frozen = FrozenRunIndex.__new__(FrozenRunIndex)
            frozen.root, frozen.run_id = self.root, self.run_id
            frozen.temp = tempfile.TemporaryDirectory(prefix="verification-analysis-snapshot-")
            frozen.db = sqlite3.connect(str(Path(frozen.temp.name) / "events.sqlite"), check_same_thread=False)
            try:
                self.db.backup(frozen.db)
                frozen.result, frozen.files = deepcopy(self.result), deepcopy(self.files)
                frozen.snapshot = deepcopy(self.state)
                frozen.snapshot["files"] = deepcopy(self.files)
                frozen.split_metadata = deepcopy(self.split_metadata)
                frozen.analysis_settings = deepcopy(self.settings)
                frozen.sampling = deepcopy(self.sampling)
                return frozen
            except BaseException:
                frozen.close()
                raise
