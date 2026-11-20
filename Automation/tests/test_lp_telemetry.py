"""Common LP telemetry, independent of model names and native variable IDs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from Automation.LpTelemetry import ExternalMetrics, metric_available, observe_bounds
from Automation.LogAnalysisIndex import RunIndex, safe_data
from visualization.SolverProgress import (SolverProgressLogger, build_solver_feedback,
                                          read_solver_progress, draw_simplex_panel, bound_violation_items)


class CommonLpTelemetryTests(unittest.TestCase):
    def test_independent_capabilities_and_unknown_engine(self):
        owner = SimpleNamespace(simplex_metrics={"pivots": 999})
        collector = ExternalMetrics(owner, Mock(), "c1", {})
        self.assertEqual(owner.simplex_metrics, {})
        owner.simplex_metrics = {"pivots": 7, "source": "custom-solver"}
        result = collector.finish()
        self.assertTrue(result["pivot_count_available"])
        self.assertFalse(result["iteration_count_available"])
        self.assertTrue(metric_available(result, "pivots"))
        self.assertFalse(result["work_counts_available"])
        unknown = ExternalMetrics(SimpleNamespace(), Mock(), "c2", {}).finish()
        self.assertIsNone(unknown["pivots"])
        self.assertIsNone(unknown["iterations"])

    def test_glpk_reports_iteration_delta_not_fake_pivots(self):
        glp = SimpleNamespace(glp_get_it_cnt=Mock(side_effect=[10, 23]))
        result = ExternalMetrics(SimpleNamespace(glp=glp, lp=object()), Mock(), "c", {}).finish()
        self.assertEqual(result["iterations"], 13)
        self.assertIsNone(result["pivots"])
        self.assertEqual(result["metrics_source"], "glpk-iteration-counter")

    def test_native_callbacks_share_outer_call_and_restore_kwargs(self):
        previous = object()
        owner = SimpleNamespace(simplex_kwargs={"progress": previous})
        logger = Mock()
        collector = ExternalMetrics(owner, logger, "outer-call", {"round_index": 2})
        sink = owner.simplex_kwargs["progress"]
        self.assertEqual(sink.simplex_start(), "outer-call")
        sink.simplex_iteration("anything", 0, violated_var="h7_23")
        sink.simplex_end("anything", "SAT", iterations=5, pivots=4)
        result = collector.finish()
        self.assertEqual(result["pivots"], 4)
        self.assertEqual(result["iterations"], 5)
        logger.simplex_iteration.assert_called_once_with("outer-call", 0, violated_var="h7_23")
        self.assertIs(owner.simplex_kwargs["progress"], previous)
        self.assertNotIn("progress_context", owner.simplex_kwargs)

    def test_bound_observation_uses_only_available_real_values(self):
        bounds = {"x": (0., 1.), "z2_7": (0., 1.), "unseen": (1., 2.)}
        self.assertEqual(observe_bounds(None, bounds), [])
        result = observe_bounds({"x": .5, "z2_7": -2.}, bounds)
        self.assertEqual([item["variable"] for item in result], ["z2_7"])
        self.assertEqual(result[0]["value"], -2.)

    def test_z3_uses_reported_cumulative_arithmetic_pivots(self):
        class Statistics:
            def __init__(self, value):
                self.value = value
            def keys(self):
                return ["arith-pivots"]
            def get_key_value(self, _key):
                return self.value
        owner = SimpleNamespace(name="z3", solver=SimpleNamespace(statistics=Mock(side_effect=[Statistics(8), Statistics(13)])))
        result = ExternalMetrics(owner, Mock(), "c", {}).finish()
        self.assertEqual(result["pivots"], 5)
        self.assertTrue(result["pivot_count_available"])
        self.assertFalse(result["iteration_count_available"])

    def test_batch_observations_retain_all_variables_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.jsonl"
            with SolverProgressLogger(path, compact_simplex_calls=True) as logger:
                call = logger.simplex_start(round_index=3, depth=2, origin="repair", row_count=400, variable_count=800)
                items = [{"variable": f"z9_{i}", "value": -1., "lower": 0., "upper": 1.} for i in range(260)]
                logger.simplex_bound_observations(call, items)
                logger.simplex_end(call, "SAT", pivots=13, iterations=14,
                                   work_counts_available=True, pivot_count_available=True,
                                   iteration_count_available=True, cycle_count_available=False,
                                   metrics_source="test-native-statistics", bound_violation_scope="call_boundary")
            events = read_solver_progress(path)
            batches = [event for event in events if event["event"] == "simplex_bound_observation"]
            self.assertEqual(sum(len(item["observations"]) for item in batches), 260)
            self.assertEqual(len(batches), 3)
            feedback = build_solver_feedback(events)
            self.assertEqual(feedback["counts"]["simplex_pivots"], 13)
            self.assertFalse(feedback["top_violated_constraints"])
            selected, scope = bound_violation_items(feedback)
            self.assertEqual(scope, "call_boundary")
            self.assertTrue(selected)
            self.assertTrue(all(item["observation_scope"] == "call_boundary" for item in selected))

    def test_internal_and_boundary_samples_are_not_merged(self):
        events = [{"event": "simplex_start", "call_id": "native", "round": 1},
                  {"event": "simplex_iteration", "call_id": "native", "iteration": 1, "violated_var": "z3_9"},
                  {"event": "simplex_bound_observation", "call_id": "native", "round": 1,
                   "observations": [{"variable": "z3_9"}] * 5}]
        feedback = build_solver_feedback(events)
        self.assertEqual(feedback["top_violated_constraints"][0]["count"], 1)
        self.assertEqual(feedback["top_observed_bound_constraints"][0]["count"], 5)
        self.assertEqual(bound_violation_items(feedback)[1], "solver_iteration")

    def test_partial_metrics_survive_analysis_queries_and_profiles(self):
        record = {"event": "simplex_call", "call_id": "external", "pivots": 12, "iterations": None,
                  "cycles": None, "work_counts_available": False, "pivot_count_available": True,
                  "iteration_count_available": False, "cycle_count_available": False,
                  "metrics_source": "reported-native", "bound_violation_scope": "call_boundary"}
        self.assertEqual(safe_data(record)["pivots"], 12)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.json").write_text(json.dumps({"status": "SAT"}))
            (root / "solver_progress.jsonl").write_text(json.dumps(record) + "\n")
            index = RunIndex(root, "run")
            try:
                self.assertEqual(index.summary()["indexed_completed_simplex"]["pivots"], 12)
                target = index.simplex_diagnostics()["comparisons"][0]["target"]
                self.assertEqual(target["pivots"], 12)
                self.assertIsNone(target["iterations"])
                self.assertEqual(index.query({"call_id": "external"})["records"][0]["data"]["pivots"], 12)
            finally:
                index.close()

    def test_graphs_use_reported_metrics_and_mark_boundary_observations(self):
        import matplotlib.pyplot as plt
        events = [{"event": "simplex_call", "call_id": "external", "pivots": 17,
                   "iterations": None, "work_counts_available": False, "pivot_count_available": True,
                   "iteration_count_available": False, "bound_violation_scope": "call_boundary", "result": "SAT"},
                  {"event": "simplex_bound_observation", "call_id": "external",
                   "observations": [{"variable": "h4_5"}]}]
        figure, axes = draw_simplex_panel(events)
        try:
            self.assertEqual(list(axes.flat[1].lines[0].get_ydata()), [17])
            self.assertEqual(axes.flat[3].get_title(), "Bound violations at LP call entry")
            self.assertEqual(axes.flat[3].patches[0].get_width(), 1)
        finally:
            plt.close(figure)
        events[0].update(pivots=None, pivot_count_available=False, iterations=8, iteration_count_available=True)
        figure, axes = draw_simplex_panel(events)
        try:
            self.assertEqual(axes.flat[1].get_title(), "Iterations per call")
            self.assertEqual(list(axes.flat[1].lines[0].get_ydata()), [8])
        finally:
            plt.close(figure)
