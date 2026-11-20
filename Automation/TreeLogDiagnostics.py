"""Summarize the selected frozen branch history, without inferring parent causes."""
import json


def tree_summary(index):
    db = index.db
    counts = {event: count for event, count in db.execute(
        "SELECT event,COUNT(*) FROM events WHERE event IN ('branch_start','branch_end') GROUP BY event")}
    depth = db.execute("SELECT MAX(depth) FROM events WHERE event='branch_start'").fetchone()[0]
    results = [{"result": result, "count": count} for result, count in db.execute(
        "SELECT result,COUNT(*) FROM events WHERE event='branch_end' GROUP BY result")]
    rounds = [{"round": rnd, "started_branches": count, "max_depth": maximum}
              for rnd, count, maximum in db.execute("SELECT round,COUNT(*),MAX(depth) FROM events "
                  "WHERE event='branch_start' GROUP BY round ORDER BY round LIMIT 101")]
    neurons = [{"layer": layer, "neuron": neuron, "started_branches": count}
               for layer, neuron, count in db.execute("SELECT layer,neuron,COUNT(*) FROM events "
                   "WHERE event='branch_start' GROUP BY layer,neuron ORDER BY COUNT(*) DESC LIMIT 10")]
    completed = []
    for file, line, raw, rnd, layer, neuron, branch_depth, started, ended, outcome in db.execute(
        "SELECT s.file,s.line,s.raw,s.round,s.layer,s.neuron,s.depth,s.time_ns,e.time_ns,e.raw "
        "FROM events s JOIN events e ON e.rowid=(SELECT MAX(t.rowid) FROM events t "
        "WHERE t.branch_id=s.branch_id AND t.event='branch_end') "
        "WHERE s.event='branch_start' ORDER BY (e.time_ns-s.time_ns) DESC LIMIT 5"):
        start, end = json.loads(raw), json.loads(outcome)
        completed.append({"round": rnd, "layer": layer, "neuron": neuron, "depth": branch_depth,
            "phase": start.get("phase"), "split_number": start.get("split_number"),
            "inclusive_seconds": (ended - started) / 1e9 if ended is not None and started is not None and ended >= started else None,
            "result": end.get("result"), "reason": end.get("reason"), "unsat_origin": end.get("unsat_origin"),
            "evidence": f"{index.run_id}:{file}:{line}"})
    unmatched = db.execute("SELECT COUNT(*) FROM events s WHERE s.event='branch_start' AND NOT EXISTS "
                          "(SELECT 1 FROM events e WHERE e.event='branch_end' AND e.branch_id=s.branch_id)").fetchone()[0]
    stable_occurrences = 0
    for (raw,) in db.execute("SELECT raw FROM events WHERE event='relu_stable'"):
        records = json.loads(raw).get("records")
        if isinstance(records, list):
            stable_occurrences += sum(isinstance(item, dict) for item in records)
    return {"started_branches": counts.get("branch_start", 0), "end_records": counts.get("branch_end", 0),
            "starts_without_end_record": unmatched, "max_depth": depth, "completed_results": results,
            "rounds": rounds[:100], "round_list_limited": len(rounds) > 100,
            "frequently_branched_neurons": neurons, "longest_completed_branches": completed,
            "stable_confirmation_occurrences": stable_occurrences,
            "limitations": ["Counts cover only the selected snapshot/filter, not the full network by default.",
                "Started branches count active/inactive branch visits, not ReLU splits; one split can have two branch nodes.",
                "An unmatched start has no recorded end; it does not prove the process is still running.",
                "Branch durations include descendants; do not sum them or call them exclusive CPU times.",
                "Branch IDs are not visual node numbers. No parent/causal links are inferred from depth or timing.",
                "Repeated STABLE confirmations may concern the same neuron on different branches."]}
