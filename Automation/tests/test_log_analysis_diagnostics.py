"""Deterministic causal-analysis inputs; no Gemini or solver benchmark."""
import json
from pathlib import Path
import tempfile
import unittest

from Automation.LogAnalysisIndex import RunIndex
from Automation.LogAnalysisKnowledge import feature_configuration
from Automation.LogAnalysisScope import ScopedIndex
from Automation.AnalyzeLogs import analyze
from Automation.tests.test_log_analysis import FakeClient, answer


class SimplexDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "result.json").write_text(json.dumps({"status": "UNSAT"}), encoding="utf-8")
        rows = []
        for identifier, number, seconds, pivots, round_id in (
            ("peer-a", 117, 2, 100, 1), ("peer-b", 118, 4, 200, 1),
            ("simplex-119", 120, 18, 300, 1), ("other-round", 121, 100, 100, 2)):
            rows.extend([
                {"event": "simplex_start", "call_id": identifier, "call_number": number,
                 "round": round_id, "depth": 7, "origin": "repair", "row_count": 600, "variable_count": 1292},
                {"event": "simplex_end", "call_id": identifier, "call_number": number,
                 "result": "SAT", "duration_seconds": seconds, "pivots": pivots,
                 "iterations": pivots + 1, "cycles": 0},
            ])
        rows.extend([
            {"event": "simplex_iteration", "call_id": "simplex-119", "violated_var": "z1_2", "iteration": 100},
            {"event": "simplex_iteration", "call_id": "simplex-119", "violated_var": "z1_2", "iteration": 200},
            {"event": "simplex_cycle", "call_id": "simplex-119", "iteration": 201},
        ])
        (self.root / "solver_progress.jsonl").write_text("".join(json.dumps(v) + "\n" for v in rows), encoding="utf-8")

    def index(self):
        index = RunIndex(self.root, "run")
        self.addCleanup(index.close)
        return index

    def test_selected_call_gets_aggregate_peers_without_widening_raw_scope(self):
        selected = ScopedIndex(self.index(), {"topic": "simplex", "call_id": "simplex-119"})
        diagnostic = selected.summary()["simplex_analysis"]["comparisons"][0]
        self.assertEqual(diagnostic["target"]["call_number"], 120)
        self.assertEqual(diagnostic["comparison"]["basis"], "same_round_and_origin")
        self.assertEqual(diagnostic["comparison"]["calls"], 2)
        ratios = diagnostic["comparison"]["target_to_peer_median_ratio"]
        self.assertEqual(ratios["seconds"], 6)
        self.assertEqual(ratios["pivots"], 2)
        self.assertEqual(ratios["call_average_seconds_per_pivot"], 3)
        self.assertEqual(ratios["variable_count"], 1)
        self.assertEqual(diagnostic["sampled_bound_selections"], [{"variable": "z1_2", "sampled_selections": 2}])
        self.assertEqual(diagnostic["recorded_cycle_events"], 1)
        self.assertNotIn("peer-a", json.dumps(selected.summary()))
        self.assertEqual(selected.query({"call_id": "peer-a"})["records"], [])
        self.assertEqual(selected.simplex_diagnostics({"call_number": 117})["comparisons"], [])

    def test_explicit_call_number_tool_and_no_invented_offset(self):
        diagnostic = self.index().simplex_diagnostics({"call_number": 120})["comparisons"][0]
        self.assertEqual(diagnostic["target"]["call_id"], "simplex-119")
        self.assertEqual(self.index().simplex_diagnostics({"call_number": 119})["comparisons"], [])
        selected = ScopedIndex(self.index(), {"topic": "simplex", "call_id": "simplex-119"})
        client = FakeClient(answer([{"functionCall": {"name": "simplex_diagnostics",
            "args": {"run_id": "run", "call_number": 120}}}]), answer([{"text": "호출 평균 비용도 증가했습니다."}]))
        result = analyze({"run": selected}, client, "120번째 호출이 왜 오래 걸렸나요?", 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["queries"][0]["response"]["comparisons"][0]["target"]["call_number"], 120)

    def test_no_peers_zero_pivots_and_ongoing_calls_are_not_fabricated(self):
        path = self.root / "solver_progress.jsonl"
        path.write_text(json.dumps({"event": "simplex_call", "call_id": "only", "duration_seconds": 3,
                                    "pivots": 0}) + "\n" + json.dumps({"event": "simplex_start", "call_id": "ongoing"}) + "\n", encoding="utf-8")
        diagnostic = self.index().simplex_diagnostics()["comparisons"][0]
        self.assertEqual(diagnostic["comparison"]["basis"], "no_peers")
        self.assertEqual(diagnostic["comparison"]["target_to_peer_median_ratio"], {})
        self.assertIsNone(diagnostic["target"]["call_average_seconds_per_pivot"])
        self.assertIsNone(diagnostic["target"]["call_number"])

    def test_external_unmeasured_counts_are_null_not_zero_in_every_analysis_path(self):
        path = self.root / "solver_progress.jsonl"
        detail = {"kind": "equation", "residual": 1.9e-5, "tolerance": 1e-9}
        row = {"event": "simplex_call", "call_id": "external", "duration_seconds": 1,
               "pivots": 0, "iterations": 0, "cycles": 0, "lp_engine": "glpk",
               "work_counts_available": False, "result": "LP_MODEL_INVALID", "model_validation": detail}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        index = self.index()
        diagnostic = index.simplex_diagnostics()["comparisons"][0]["target"]
        self.assertIsNone(diagnostic["pivots"])
        self.assertIsNone(diagnostic["iterations"])
        self.assertIsNone(diagnostic["cycles"])
        self.assertEqual(diagnostic["model_validation"], detail)
        self.assertIsNone(index.summary()["indexed_completed_simplex"]["pivots"])
        data = index.query({"call_id": "external"})["records"][0]["data"]
        self.assertIsNone(data["pivots"])
        self.assertIsNone(data["iterations"])
        self.assertIn("미수집", data["work_counts_meaning"])

    def test_configuration_never_fills_in_missing_flags_from_current_defaults(self):
        self.assertEqual(feature_configuration("underapprox", [], {})["state"], "unknown")
        self.assertEqual(feature_configuration("underapprox", [], {"solver": {"reluplex_options": {"under_approx": 0}}})["state"], "disabled")
        self.assertEqual(feature_configuration("underapprox", [{"underapprox_enabled": True}], {})["state"], "enabled")
        self.assertEqual(feature_configuration("bounds", [{"derive_in_search": False}], {})["state"], "unknown")
        selected = ScopedIndex(self.index(), {"topic": "underapprox"})
        summary = selected.summary()
        self.assertEqual(summary["record_availability"]["selected_records"], 0)
        self.assertEqual(summary["feature_configuration"]["state"], "unknown")
        self.assertTrue(summary["implementation_reference"]["historical_execution_match"])

    def test_tree_summary_distinguishes_inclusive_branch_time_and_missing_end(self):
        rows = [
            {"event": "branch_start", "branch_id": "outer", "round_index": 1, "depth": 1,
             "layer": 1, "index": 2, "phase": "+", "time_ns": 10**9},
            {"event": "branch_start", "branch_id": "child", "round_index": 1, "depth": 2,
             "layer": 2, "index": 3, "phase": "-", "time_ns": 2 * 10**9},
            {"event": "branch_end", "branch_id": "child", "result": "UNSAT", "unsat_origin": "found", "time_ns": 4 * 10**9},
            {"event": "branch_end", "branch_id": "outer", "result": "UNSAT", "unsat_origin": "propagated", "time_ns": 5 * 10**9},
            {"event": "branch_start", "branch_id": "open", "round_index": 2, "depth": 1,
             "layer": 1, "index": 4, "phase": "-", "time_ns": 6 * 10**9},
        ]
        (self.root / "relu_events_branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        selected = ScopedIndex(self.index(), {"topic": "tree"})
        summary = selected.summary()["tree_analysis"]
        self.assertEqual(summary["max_depth"], 2)
        self.assertEqual(summary["starts_without_end_record"], 1)
        branches = summary["longest_completed_branches"]
        self.assertEqual([b["inclusive_seconds"] for b in branches], [4, 2])
        self.assertEqual([b["unsat_origin"] for b in branches], ["propagated", "found"])
        self.assertTrue(all("node_id" not in b and "parent_id" not in b for b in branches))


if __name__ == "__main__":
    unittest.main()
