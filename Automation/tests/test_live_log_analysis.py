"""Live indexing and snapshot isolation; no solver benchmarks or external API."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from Automation.LiveLogAnalysis import LiveRunIndex
from Automation.LogAnalysisScope import ScopedIndex
from Automation.LogAnalysisServer import AnalysisApp
from Automation.tests.test_log_analysis import FakeClient, answer


class LiveLogTests(unittest.TestCase):
    def test_completed_and_live_indexes_share_schema_normalization_and_queries(self):
        from Automation.LogAnalysisIndex import RunIndex
        self.write("result.json", {**self.manifest, "status": "UNSAT"})
        self.append({"event": "simplex_end", "round_index": 2, "call_id": "simplex-7",
                     "duration_seconds": 3.5, "pivots": 12, "result": "SAT"})
        # Corrupt numeric fields stay in raw evidence, but cannot overflow SQLite.
        self.append({"event": "simplex_end", "round": 2, "call_id": "simplex-8",
                     "duration_seconds": 10**400, "pivots": 2**70, "result": "SAT"})
        self.append({"event": "+", "layer": 1, "index": 2}, "relu_events.jsonl")
        live = self.index()
        saved = RunIndex(self.run, "run")
        self.addCleanup(saved.close)
        self.assertEqual(live.db.execute("PRAGMA table_info(events)").fetchall(),
                         saved.db.execute("PRAGMA table_info(events)").fetchall())
        self.assertEqual(live.db.execute("SELECT * FROM events ORDER BY file,line").fetchall(),
                         saved.db.execute("SELECT * FROM events ORDER BY file,line").fetchall())
        self.assertEqual(live.query({"round": 2})["records"], saved.query({"round": 2})["records"])
        self.assertEqual(live.rank({"metric": "slow_calls"})["ranking"],
                         saved.rank({"metric": "slow_calls"})["ranking"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.manifest = {"run_id": "unique-run-a", "status": "RUNNING",
                         "model": {"layer_sizes": [2, 3, 1]},
                         "sampling": {"simplex_detail_stride": 100}}
        self.write("run_info.json", self.manifest)

    def write(self, name, data):
        (self.run / name).write_text(json.dumps(data), encoding="utf-8")

    def append(self, data, name="solver_progress.jsonl"):
        with (self.run / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data) + "\n")

    def index(self, **kwargs):
        index = LiveRunIndex(self.run, **kwargs)
        self.addCleanup(index.close)
        return index

    def freeze(self, index, scope=None):
        frozen = index.freeze()
        self.addCleanup(frozen.close)
        return ScopedIndex(frozen, scope) if scope else frozen

    def test_empty_live_run_does_not_require_final_result(self):
        index = self.index()
        self.assertFalse(index.state["finished"])
        self.assertEqual(index.state["status"], "RUNNING")
        self.assertEqual(self.freeze(index).summary()["result"]["model"]["layer_sizes"], [2, 3, 1])
        self.assertEqual(index.state["indexed_records"], 0)

    def test_older_running_logs_supply_model_context_without_manifest(self):
        (self.run / "run_info.json").unlink()
        self.append({"event": "run_context", "witness_context": {"model_name": "network.onnx",
                                                                 "model_layer_sizes": [5, 50, 5]}})
        index = self.index()
        summary = self.freeze(index, {"topic": "simplex"}).summary()
        self.assertEqual(summary["run_context_only"]["model"]["layer_sizes"], [5, 50, 5])
        self.assertEqual(summary["source"], "run:solver_progress.jsonl")

    def test_partial_tail_is_deferred_and_not_invalid(self):
        path = self.run / "solver_progress.jsonl"
        path.write_bytes(b'{"event":"round_start","round":1}\n{"event":"round_end"')
        index = self.index()
        self.assertEqual(index.state["indexed_records"], 1)
        self.assertEqual(index.files[0]["invalid_lines"], 0)
        with path.open("ab") as stream:
            stream.write(b',"round":1}\n')
        index.refresh()
        self.assertEqual(index.state["indexed_records"], 2)
        self.assertEqual([r["evidence"] for r in index.query({})["records"]],
                         ["run:solver_progress.jsonl:1", "run:solver_progress.jsonl:2"])
        self.assertFalse(index.state["coverage_limited"])

    def test_refresh_only_parses_appended_lines(self):
        self.append({"event": "round_start", "round": 1})
        index = self.index()
        index.refresh()
        self.assertEqual(index.state["bytes_read_this_refresh"], 0)
        new = {"event": "round_start", "round": 2}
        before = (self.run / "solver_progress.jsonl").stat().st_size
        self.append(new)
        index.refresh()
        self.assertEqual(index.state["bytes_read_this_refresh"], (self.run / "solver_progress.jsonl").stat().st_size - before)
        self.assertEqual(index.options()["rounds"], [1, 2])
        self.assertEqual(index.state["indexed_records"], 2)

    def test_frozen_calls_remain_ongoing_when_source_finishes(self):
        self.append({"event": "simplex_start", "round": 1, "call_id": "s1", "time_ns": time.time_ns()-10**9})
        index = self.index()
        selected = self.freeze(index, {"topic": "simplex", "round": 1})
        self.append({"event": "simplex_end", "call_id": "s1", "duration_seconds": 2, "pivots": 4})
        index.refresh()
        summary = selected.summary()
        self.assertEqual(summary["indexed_completed_simplex"]["calls"], 0)
        self.assertEqual(summary["ongoing_simplex"]["calls"][0]["call_id"], "s1")
        self.assertEqual(selected.query({"event": "simplex_end"})["records"], [])
        fresh = self.freeze(index, {"topic": "simplex", "round": 1})
        self.assertEqual(fresh.summary()["ongoing_simplex"]["calls"], [])
        self.assertEqual(fresh.summary()["indexed_completed_simplex"]["calls"], 1)
        self.assertEqual(fresh.summary()["sampling"], self.manifest["sampling"])

    def test_round_metadata_and_neuron_batches_are_frozen(self):
        self.write("relu_events_meta.json", {"round_starts": [{"time_ns": 5, "round": 1}]})
        self.append({"event": "+", "layer": 1, "index": 0, "time_ns": 10}, "relu_events.jsonl")
        index = self.index()
        frozen = self.freeze(index)
        self.write("relu_events_meta.json", {"round_starts": [{"time_ns": 5, "round": 2}]})
        selected = ScopedIndex(frozen, {"topic": "relu", "round": 1})
        self.assertEqual(len(selected.query({})["records"]), 1)
        self.append({"event": "relu_stable", "round_index": 2,
                     "records": [{"variable": "z1_1"}, {"variable": "z1_2"}]}, "relu_events_branches.jsonl")
        index.refresh()
        self.assertEqual(index.options()["neurons"], {"1": [0, 1, 2]})
        selected = self.freeze(index, {"topic": "bounds", "layer": 1, "neuron": 2})
        self.assertNotIn("z1_1", json.dumps(selected.query({})))

    def test_byte_budget_is_explicit_and_never_chases_new_appends(self):
        self.append({"event": "round_start", "round": 1})
        size = (self.run / "solver_progress.jsonl").stat().st_size
        index = self.index(max_bytes=size)
        self.append({"event": "round_start", "round": 2})
        index.refresh()
        self.assertEqual(index.state["indexed_records"], 1)
        self.assertTrue(index.state["coverage_limited"])
        self.assertEqual(index.state["indexed_bytes"], size)

    def test_truncate_and_same_length_rewrite_do_not_mix_runs(self):
        self.append({"event": "round_start", "round": 1})
        index = self.index()
        before = index.identity
        path = self.run / "solver_progress.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"round": 1', '"round": 2'), encoding="utf-8")
        index.refresh()
        self.assertNotEqual(index.identity, before)
        self.assertEqual(index.options()["rounds"], [2])
        path.write_bytes(b"")
        index.refresh()
        self.assertEqual(index.state["indexed_records"], 0)

    def test_final_result_keeps_run_identity_and_stale_result_is_ignored(self):
        self.write("result.json", {"run_id": "previous-run", "status": "VERIFIED"})
        index = self.index()
        self.assertFalse(index.state["finished"])
        identity = index.identity
        self.write("result.json", {**self.manifest, "status": "VERIFIED"})
        index.refresh()
        self.assertTrue(index.state["finished"])
        self.assertEqual(index.state["status"], "VERIFIED")
        self.assertEqual(index.identity, identity)
        self.write("run_info.json", {**self.manifest, "run_id": "next-run"})
        index.refresh()
        self.assertNotEqual(index.identity, identity)
        self.assertFalse(index.state["finished"])

    def test_reused_folder_does_not_read_previous_run_logs_or_round_metadata(self):
        self.append({"event": "round_start", "round": 99})
        self.write("relu_events_meta.json", {"round_starts": [{"time_ns": 1, "round": 99}]})
        old_stamp = time.time_ns() - 10**10
        for name in ("solver_progress.jsonl", "relu_events_meta.json"):
            os.utime(self.run / name, ns=(old_stamp, old_stamp))
        self.write("run_info.json", {**self.manifest, "log_epoch_ns": time.time_ns() - 10**9})
        index = self.index()
        self.assertEqual(index.state["indexed_records"], 0)
        self.assertEqual(index.options()["rounds"], [])

    def test_latest_call_is_not_stuck_at_dropdown_limit(self):
        for number in range(2002):
            self.append({"event": "simplex_start", "call_id": f"s{number}"})
        index = self.index()
        self.assertEqual(len(index.options()["calls"]), 2000)
        self.assertEqual(index.state["last_call"], "s2001")

    def test_completion_appearing_during_capture_cannot_label_earlier_logs_final(self):
        self.append({"event": "simplex_start", "call_id": "s1", "round": 1})
        index = self.index()
        original = index._json
        def finishing_writer(name):
            if name == "result.json":
                self.append({"event": "simplex_end", "call_id": "s1", "duration_seconds": 2})
                self.write("result.json", {**self.manifest, "status": "VERIFIED"})
            return original(name)
        with patch.object(index, "_json", side_effect=finishing_writer):
            snapshot = self.freeze(index, {"topic": "simplex"})
        self.assertTrue(snapshot.summary()["snapshot"]["finished"])
        self.assertEqual(snapshot.summary()["indexed_completed_simplex"]["calls"], 1)

    def test_snapshot_is_fixed_before_worker_and_no_automatic_api_calls(self):
        self.append({"event": "simplex_start", "call_id": "s1", "round": 1})
        entered, release = threading.Event(), threading.Event()
        clients = []
        def factory(progress):
            entered.set()
            release.wait(5)
            client = FakeClient(answer([{"text": "진행 중인 호출입니다."}]))
            clients.append(client)
            return client
        app = AnalysisApp(self.run, output_dir=self.root / "analysis", client_factory=factory)
        try:
            app.info()
            self.assertEqual(clients, [])
            job = app.start({"consent": True, "scope": {"topic": "simplex"}})
            self.assertTrue(entered.wait(5))
            self.append({"event": "simplex_end", "call_id": "s1", "duration_seconds": 1})
            state = app.info()["state"]
            self.assertEqual(state["indexed_records"] - job["snapshot"]["indexed_records"], 1)
            release.set()
            app.executor.shutdown(wait=True)
            result = app.jobs[job["id"]]["result"]
            self.assertFalse(result["snapshot"]["finished"])
            self.assertNotIn('"simplex_end"', json.dumps(clients[0].requests))
            self.assertEqual(result["runs"][0]["indexed_completed_simplex"]["calls"], 0)
            self.assertEqual(result["runs"][0]["ongoing_simplex"]["calls"][0]["call_id"], "s1")
        finally:
            release.set()
            app.close()

    def test_run_reset_clears_cached_result(self):
        self.append({"event": "round_start", "round": 1})
        app = AnalysisApp(self.run, output_dir=self.root / "analysis", client_factory=lambda _: None)
        try:
            app.results["example"] = {"analysis": "old run"}
            self.write("run_info.json", {**self.manifest, "run_id": "next-run"})
            app.info()
            self.assertEqual(app.results, {})
        finally:
            app.close()

    def test_manifest_exists_before_solver_and_final_result_matches(self):
        from Automation.AutoVerify import run_vnnlib_verification
        from Automation.SolverStatus import SolverResult, SolverStatus
        prop = self.root / "test.vnnlib"
        prop.write_text('(declare-const X_0 Real) (declare-const X_1 Real) '
                        '(declare-const Y_0 Real) (assert (>= Y_0 0))', encoding="utf-8")
        def solver(*args, **kwargs):
            manifest = json.loads((self.run / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "RUNNING")
            self.assertFalse((self.run / "result.json").exists())
            return SolverResult(status=SolverStatus.UNSAT, reason="BOOLEAN_UNSAT", rounds=1, elapsed_seconds=0)
        model = Path(__file__).resolve().parents[2] / "xor_network.bin"
        with patch("Automation.AutoVerify.dpll_t_detailed", side_effect=solver):
            result = run_vnnlib_verification(str(model), str(prop), visualization_output_dir=str(self.run))
        manifest = json.loads((self.run / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(result["run_id"], manifest["run_id"])
        self.assertEqual(manifest["status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
