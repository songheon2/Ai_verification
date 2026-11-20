"""Configuration provenance and explanatory LLM context; no network calls."""
import json
from pathlib import Path
import tempfile
import unittest

from Automation.AnalyzeLogs import analyze, SYSTEM
from Automation.LiveLogAnalysis import LiveRunIndex
from Automation.LogAnalysisIndex import RunIndex, recorded_context
from Automation.LogAnalysisKnowledge import feature_configuration, runtime_reference, analysis_support
from Automation.LogAnalysisScope import ScopedIndex
from Automation.tests.test_log_analysis import FakeClient, answer


class AnalysisConfigurationTests(unittest.TestCase):
    def test_unknown_is_analyzer_generated_and_names_missing_fields(self):
        config = feature_configuration("numeric", [], {"status": "SAT"})
        self.assertEqual(config["state"], "unknown")
        self.assertEqual(config["state_origin"], "analysis_derived")
        self.assertEqual(config["reason"], "no_recorded_feature_fields")
        self.assertEqual(config["missing_fields"], ["numeric_safeguards"])
        self.assertIn("분석에 제공된 기록", config["diagnostic_conclusion"])
        self.assertIn("솔버", config["meaning"])

    def test_non_feature_topics_do_not_fabricate_unknown_feature_state(self):
        for topic in ("simplex", "dpll", "tree", "relu"):
            self.assertEqual(feature_configuration(topic, [], {})["state"], "not_applicable")

    def test_partial_flags_conflicts_and_invalid_types_have_distinct_diagnoses(self):
        config = feature_configuration("bounds", [{"derive_in_search": False}], {})
        self.assertEqual(config["reason"], "partial_feature_fields")
        self.assertEqual(config["missing_fields"], ["derive_bound", "pivot_tighten"])
        config = feature_configuration("backjump", [{"backjump": True}],
                                       {"solver": {"reluplex_options": {"backjump": False}}})
        self.assertEqual(config["state"], "mixed")
        self.assertEqual(config["reason"], "conflicting_recorded_values")
        config = feature_configuration("numeric", [{"numeric_safeguards": "false"}], {})
        self.assertEqual(config["state"], "unknown")
        self.assertEqual(config["reason"], "invalid_recorded_values")
        self.assertEqual(config["invalid_values"][0]["field"], "numeric_safeguards")

    def test_runtime_identity_is_recorded_not_assumed_from_current_defaults(self):
        self.assertEqual(runtime_reference([], {})["identity_state"], "not_recorded")
        runtime = runtime_reference([], {"solver_variant": "z3-core"})
        self.assertEqual(runtime["recorded_engine_candidates"], ["z3"])
        self.assertIn("정확한 유리수", runtime["implementation_notes"][0])
        runtime = runtime_reference([{"lp_engine": "glpk"}], {"solver": {"reluplex_options": {"lp_engine": "z3"}}})
        self.assertEqual(runtime["identity_state"], "conflicting")

    def test_solver_unknown_is_separate_from_configuration_unknown(self):
        support = analysis_support("numeric", [], {"status": "UNKNOWN", "solver": {"reason": "TIMEOUT"}}, [], [])
        self.assertEqual(support["configuration"]["reason"], "no_recorded_feature_fields")
        self.assertEqual(support["verification_result"], {"status": "UNKNOWN", "reason": "TIMEOUT"})
        self.assertIn("configuration.unknown", support["state_domains"])
        self.assertIn("solver.UNKNOWN", support["state_domains"])

    def test_recorded_manifest_context_is_not_current_default_or_other_run(self):
        manifest = {"run_id": "same", "solver_variant": "z3-core", "solver": {"reluplex_options": {"numeric_safeguards": False}}}
        result = {"run_id": "same", "status": "UNKNOWN", "solver": {"reason": "TIMEOUT"}}
        merged = recorded_context(result, manifest)
        self.assertFalse(merged["solver"]["reluplex_options"]["numeric_safeguards"])
        self.assertEqual(merged["solver"]["reason"], "TIMEOUT")
        other = recorded_context({**result, "run_id": "other"}, manifest)
        self.assertNotIn("reluplex_options", other["solver"])
        self.assertTrue(other["context_provenance"]["manifest_identity_conflict"])

    def test_saved_and_live_scoped_support_and_payload_are_enriched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"run_id": "a", "status": "RUNNING", "solver_variant": "z3-core",
                        "solver": {"reluplex_options": {"numeric_safeguards": True, "lp_engine": "z3"}}}
            (root / "run_info.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "result.json").write_text(json.dumps({**manifest, "status": "SAT"}), encoding="utf-8")
            record = {"event": "relu_mechanism", "kind": "settings", "round_index": 1,
                      "numeric_safeguards": True, "lp_engine": "z3", "numeric_event_support": "not_exposed_by_external_engine"}
            (root / "relu_events_branches.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            saved = RunIndex(root, "run")
            live = LiveRunIndex(root)
            frozen = live.freeze()
            try:
                for index in (saved, frozen):
                    selected = ScopedIndex(index, {"topic": "mechanisms", "mechanism": "numeric"})
                    summary = selected.summary()
                    support = summary["analysis_support"]
                    self.assertEqual(support["configuration"]["state"], "enabled")
                    self.assertEqual(support["runtime_reference"]["recorded_engine_candidates"], ["z3"])
                    self.assertEqual(support["observed_event_counts_in_selection"], [])
                    client = FakeClient(answer([{"text": "이 경로는 자체 수치 콜백을 제공하지 않습니다."}]))
                    self.assertEqual(analyze({"run": selected}, client, "수치 기록이 왜 없나요?")["status"], "complete")
                    parts = [json.loads(part["text"]) for part in client.requests[0][0][0]["parts"]]
                    self.assertEqual(parts[-1]["analysis_support"]["configuration"]["state"], "enabled")
            finally:
                saved.close()
                frozen.close()
                live.close()

    def test_causal_support_survives_large_summary_truncation(self):
        support = analysis_support("numeric", [], {}, [], [])
        class LargeSummary:
            def summary(self):
                return {"large": "x" * 100000, "run_id": "run", "analysis_support": support}
        client = FakeClient(answer([{"text": "적용 설정 항목이 저장되지 않았습니다."}]))
        analyze({"run": LargeSummary()}, client, "unknown은 오류인가요?")
        parts = client.requests[0][0][0]["parts"]
        self.assertTrue(json.loads(parts[0]["text"])["truncated"])
        self.assertEqual(json.loads(parts[-1]["text"])["analysis_support"], support)
        self.assertIn("생성 주체·누락 필드", SYSTEM)
