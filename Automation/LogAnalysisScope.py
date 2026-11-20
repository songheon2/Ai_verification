"""Hard, local data boundaries for item-specific LLM analysis."""
import json
import re
from bisect import bisect_right

from Automation.LogAnalysisIndex import safe_data
from Automation.LogAnalysisKnowledge import feature_configuration, implementation_reference, analysis_support
from Automation.SimplexLogDiagnostics import call_profiles, diagnose_simplex


TOPICS = {"all": "전체", "dpll": "DPLL / Theory", "simplex": "Simplex",
          "relu": "ReLU 분기 이력", "tree": "ReLU 탐색 트리", "mechanisms": "Mechanisms"}
MECHANISMS = {"all": "전체 기법", "bounds": "경계 유도", "backjump": "충돌 분석·백점프",
              "numeric": "부동소수점", "underapprox": "과소근사"}


def analysis_topic(scope):
    if "selections" in scope:
        topics = {analysis_topic(item) for item in scope["selections"]}
        return topics.pop() if len(topics) == 1 else "all"
    if scope["topic"] == "mechanisms":
        return "mechanisms" if scope["mechanism"] == "all" else scope["mechanism"]
    return scope["topic"]


def normalize_scope(value):
    if isinstance(value, dict) and "selections" in value:
        items = value["selections"]
        if set(value) != {"selections"} or not isinstance(items, list) or not 1 <= len(items) <= 20:
            raise ValueError("Select between 1 and 20 items")
        if any(not isinstance(item, dict) or "selections" in item for item in items):
            raise ValueError("Nested selections are not supported")
        normalized = [normalize_scope(item) for item in items]
        unique = {json.dumps(item, sort_keys=True, separators=(",", ":")): item for item in normalized}
        return {"selections": [unique[key] for key in sorted(unique)]}
    if not isinstance(value, dict) or set(value) - {"topic", "mechanism", "round", "call_id", "layer", "neuron", "branch_id"}:
        raise ValueError("Invalid scope fields")
    topic = value.get("topic", "all")
    if not isinstance(topic, str):
        raise ValueError("Invalid analysis topic")
    # Canonicalize old saved analyses/requests without exposing old top-level options.
    if topic in MECHANISMS and topic != "all":
        if value.get("mechanism", topic) != topic:
            raise ValueError("Conflicting mechanism selection")
        scope = {"topic": "mechanisms", "mechanism": topic}
    else:
        scope = {"topic": topic}
        if topic == "mechanisms":
            scope["mechanism"] = value.get("mechanism", "all")
        elif "mechanism" in value:
            raise ValueError("Mechanism selection requires the Mechanisms topic")
    if scope["topic"] not in TOPICS:
        raise ValueError("Unknown analysis topic")
    if scope["topic"] == "mechanisms" and (not isinstance(scope["mechanism"], str) or scope["mechanism"] not in MECHANISMS):
        raise ValueError("Unknown mechanism")
    for key in ("round", "layer", "neuron"):
        if value.get(key) not in (None, ""):
            item = value[key]
            if type(item) is not int or not 0 <= item <= 1000000 or (key == "round" and item == 0):
                raise ValueError(f"Invalid {key}")
            scope[key] = item
    if value.get("call_id"):
        if scope["topic"] != "simplex" or not isinstance(value["call_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value["call_id"]):
            raise ValueError("call_id is only supported for Simplex")
        scope["call_id"] = value["call_id"]
    if "branch_id" in value:
        branch_id = value["branch_id"]
        if scope["topic"] != "tree" or not isinstance(branch_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", branch_id):
            raise ValueError("branch_id is only supported for a tree node")
        scope["branch_id"] = branch_id
    if "neuron" in scope and "layer" not in scope:
        raise ValueError("Select a layer before a neuron")
    if analysis_topic(scope) not in ("relu", "tree", "bounds") and ("layer" in scope or "neuron" in scope):
        raise ValueError("Layer/neuron filters are supported for ReLU and bound derivation")
    return scope


def scope_key(scope):
    return json.dumps(normalize_scope(scope), sort_keys=True, separators=(",", ":"))


def neuron_of(record):
    if type(record.get("layer")) is int and type(record.get("index")) is int:
        return record["layer"], record["index"]
    match = re.match(r"^[zh](\d+)_(\d+)(?:_|$)", str(record.get("variable", record.get("neuron", ""))))
    return (int(match[1]), int(match[2])) if match else (None, None)


def topic_matches(topic, record):
    event, kind = record.get("event", ""), record.get("kind", "")
    if topic == "all":
        return True
    if topic == "mechanisms":
        return event in {"relu_mechanism", "relu_stable"}
    if topic == "dpll":
        return event in {"round_start", "round_end", "theory_selection", "theory_result", "solver_end", "run_context"}
    if topic == "simplex":
        return event.startswith("simplex_") or event == "constraint_detail"
    if topic == "relu":
        return event in {"+", "-", "branch_start", "branch_end"} or (event == "relu_mechanism" and kind == "repair")
    if topic == "tree":
        return event in {"branch_start", "branch_end", "relu_stable"} or (
            event == "relu_mechanism" and kind in {"repair", "conflict", "backjump"})
    if topic == "bounds":
        return event == "relu_stable" or (event == "relu_mechanism" and kind in {"bounds", "bound_updates"})
    kinds = {"backjump": {"conflict", "backjump"}, "numeric": {"numerical"}, "underapprox": {"underapprox"}}
    return (event == "relu_mechanism" and kind in kinds.get(topic, set())) or (
        topic == "underapprox" and event == "relu_stable" and record.get("source") == "under_approx")


def scope_options(index):
    rounds = [r[0] for r in index.db.execute("SELECT DISTINCT round FROM events WHERE round IS NOT NULL ORDER BY round")]
    calls = [r[0] for r in index.db.execute("SELECT DISTINCT call_id FROM events WHERE call_id IS NOT NULL ORDER BY line LIMIT 2000")]
    neurons = {}
    settings = []
    for (raw,) in index.db.execute("SELECT raw FROM events WHERE event IN ('relu_stable','relu_mechanism','branch_start','+')"):
        record = json.loads(raw)
        if record.get("kind") == "settings":
            settings.append(record)
        entries = [record]
        for key in ("records", "ranges", "changes"):
            entries.extend(v for v in record.get(key, []) if isinstance(v, dict))
        for item in entries:
            layer, neuron = neuron_of(item)
            if layer is not None:
                neurons.setdefault(str(layer), set()).add(neuron)
    return {"topics": TOPICS, "mechanisms": MECHANISMS, "rounds": rounds, "calls": calls, "calls_list_limited": len(calls) == 2000,
            "neurons": {layer: sorted(values) for layer, values in neurons.items()}, "settings": safe_data(settings[-20:])}


def _link_legacy_branch_observations(db, branch_id):
    """Use file order and depth to recover ownership in the temporary index."""
    start = db.execute(
        "SELECT file,line,depth,round FROM events WHERE event='branch_start' AND branch_id=? "
        "AND json_type(raw,'$.monotonic_ns') IS NULL LIMIT 1", (branch_id,),
    ).fetchone()
    if start is None:
        return
    file, line, depth, round_id = start
    end = db.execute(
        "SELECT MIN(line) FROM events WHERE file=? AND line>? AND "
        "((event='branch_end' AND branch_id=?) OR (event='branch_start' AND depth<=?))",
        (file, line, branch_id, depth),
    ).fetchone()[0]
    db.execute(
        "UPDATE events SET branch_id=? WHERE file=? AND line>? AND (? IS NULL OR line<?) "
        "AND depth=? AND (? IS NULL OR round IS NULL OR round=?) "
        "AND event IN ('relu_mechanism','relu_stable') AND branch_id IS NULL "
        "AND json_type(raw,'$.branch_id') IS NULL",
        (branch_id, file, line, end, end, depth, round_id, round_id),
    )


class ScopedIndex:
    """Filter a disposable RunIndex, not source files. Tools cannot widen this scope."""
    def __init__(self, index, scope):
        self.index, self.scope = index, normalize_scope(scope)
        self.scopes = self.scope.get("selections", [self.scope])
        self.topic = analysis_topic(self.scope)
        topics = {analysis_topic(item) for item in self.scopes}
        self.has_simplex = bool(topics & {"all", "simplex"})
        self.has_tree = bool(topics & {"all", "tree"})
        self.settings = index.analysis_settings if hasattr(index, "analysis_settings") else scope_options(index)["settings"]
        if all("round" in item for item in self.scopes):
            rounds = {item["round"] for item in self.scopes}
            self.settings = [item for item in self.settings
                             if item.get("round", item.get("round_index")) in rounds]
        db = index.db
        metadata = index._file("relu_events_meta.json")
        if hasattr(index, "split_metadata") or (metadata.is_file() and metadata.stat().st_size <= 1024 * 1024):
            try:
                content = index.split_metadata if hasattr(index, "split_metadata") else json.loads(metadata.read_text(encoding="utf-8"))
                starts = sorted((int(item["time_ns"]), int(item["round"])) for item in
                                content.get("round_starts", []))
                stamps = [stamp for stamp, _ in starts]
                for rowid, stamp in db.execute("SELECT rowid,time_ns FROM events WHERE event IN ('+','-') AND round IS NULL").fetchall():
                    position = bisect_right(stamps, stamp) - 1 if stamp is not None else -1
                    if position >= 0:
                        db.execute("UPDATE events SET round=? WHERE rowid=?", (starts[position][1], rowid))
            except (ValueError, TypeError, KeyError):
                pass  # Missing/corrupt round boundaries cannot be invented.
        db.execute("CREATE INDEX IF NOT EXISTS by_branch ON events(branch_id)")
        # Terminal records often omit metadata. Link only by explicit IDs, never timing.
        for event, start_event, id_column, fields in (
            ("simplex_end", "simplex_start", "call_id", ("round", "depth")),
            ("simplex_iteration", "simplex_start", "call_id", ("round", "depth")),
            ("simplex_bound_observation", "simplex_start", "call_id", ("round", "depth")),
            ("simplex_pivot", "simplex_start", "call_id", ("round", "depth")),
            ("branch_end", "branch_start", "branch_id", ("round", "depth", "variable", "layer", "neuron")),
        ):
            assignments = ",".join(f"{field}=COALESCE({field},(SELECT s.{field} FROM events s WHERE s.{id_column}=e.{id_column} AND s.event=? LIMIT 1))" for field in fields)
            db.execute(f"UPDATE events AS e SET {assignments} WHERE event=?", [*[start_event] * len(fields), event])
        for branch_id in {scope["branch_id"] for scope in self.scopes if "branch_id" in scope}:
            _link_legacy_branch_observations(db, branch_id)
        # Freeze comparison metrics before the destructive scope filter. Raw peer
        # records are still deleted; only bounded aggregate comparisons can leave.
        self.simplex_reference = call_profiles(index) if self.has_simplex else None
        self._simplex_summary = None
        self._tree_summary = None
        last = 0
        while True:
            rows = db.execute("SELECT rowid,round,call_id,layer,neuron,branch_id,raw FROM events WHERE rowid>? ORDER BY rowid LIMIT 32", (last,)).fetchall()
            if not rows:
                break
            for rowid, round_id, call_id, layer, neuron, branch_id, raw in rows:
                last = rowid
                record = json.loads(raw)
                candidates = [scope for scope in self.scopes if (
                                  topic_matches(analysis_topic(scope), record)
                                  or ("branch_id" in scope and record.get("event") == "relu_mechanism"))
                              and ("round" not in scope or round_id == scope["round"])
                              and ("call_id" not in scope or call_id == scope["call_id"])
                              and ("branch_id" not in scope or branch_id == scope["branch_id"])]
                keep = bool(candidates)
                # A selected node can change bounds for neurons other than the one it splits.
                if keep and all("layer" in scope and "branch_id" not in scope for scope in candidates):
                    def matches(item):
                        l, n = neuron_of(item)
                        return any(l == scope["layer"] and ("neuron" not in scope or n == scope["neuron"]) for scope in candidates)
                    collections = [key for key in ("records", "ranges", "changes") if isinstance(record.get(key), list)]
                    if collections:
                        for key in collections:
                            record[key] = [v for v in record[key] if isinstance(v, dict) and matches(v)]
                        keep = any(record[key] for key in collections)
                        # These counts describe the original full batch, not the projected neuron.
                        for key in ("changed", "fixed", "conflict"):
                            record.pop(key, None)
                        record["scope_projection"] = True
                    else:
                        l, n = neuron_of(record)
                        if l is None:
                            l, n = layer, neuron
                        keep = any(l == scope["layer"] and ("neuron" not in scope or n == scope["neuron"]) for scope in candidates)
                if not keep:
                    db.execute("DELETE FROM events WHERE rowid=?", (rowid,))
                else:
                    if branch_id is not None:
                        record.setdefault("branch_id", branch_id)
                    db.execute("UPDATE events SET raw=? WHERE rowid=?", (json.dumps(record, ensure_ascii=False), rowid))
        db.commit()

    def summary(self):
        result = self.index.summary()
        # Only model/property/configuration is shared context. Do not present global stats as scoped.
        context = result.pop("result")
        solver = context.get("solver") or {}
        hidden = {"model"} if self.scope == {"topic": "all"} else {"theory_stats", "model"}
        context["solver"] = {k: v for k, v in solver.items() if k not in hidden}
        result.update(scope=self.scope, run_context_only=context, settings=self.settings,
                      scope_note="Raw queries and selected counts remain restricted to this selection. Simplex comparison aggregates may use other completed calls from the same frozen run, labelled separately. Run context is background only; source file sizes are before filtering.")
        count = sum(row["count"] for row in result["indexed_event_counts"])
        result.update(implementation_reference=implementation_reference(self.topic),
                      feature_configuration=feature_configuration(self.topic, self.settings, context),
                      analysis_support=analysis_support(self.topic, self.settings, context, result["indexed_event_counts"], result["files"]),
                      record_availability={"selected_records": count,
                          "analysis_mode": "records_and_reference" if count else "configuration_and_reference",
                          "meaning": "기록 0건만으로 미적용을 단정하지 않는다. 원리·구현 설명은 기록 없이도 가능하다."})
        if self.has_simplex:
            if self._simplex_summary is None:
                self._simplex_summary = self.simplex_diagnostics()
            result["simplex_analysis"] = self._simplex_summary
        if self.has_tree:
            if self._tree_summary is None:
                from Automation.TreeLogDiagnostics import tree_summary
                self._tree_summary = tree_summary(self.index)
            result["tree_analysis"] = self._tree_summary
        return result

    def query(self, args):
        return {**self.index.query(args), "scope": self.scope}

    def rank(self, args):
        return {**self.index.rank(args), "scope": self.scope}

    def simplex_diagnostics(self, args=None):
        if not self.has_simplex:
            raise ValueError("Simplex diagnostics are only available in the Simplex selection")
        return {**diagnose_simplex(self.index, args, self.simplex_reference), "scope": self.scope}
