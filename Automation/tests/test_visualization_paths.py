from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from visualization.RealtimeSplitVisualization import (
    SplitEventLogger,
    split_events_from_trace,
)
from visualization.ReluSearchTree import (
    RealtimeSearchTreeVisualizer,
    branch_log_path,
    build_relu_search_tree,
    trace_from_branch_log,
    write_relu_search_tree_html,
)
from visualization.SolveTrace import SolveTrace, TraceEvent
from visualization.VisualizationMode import (
    create_visualization_run_directory,
    finalize_visualization_run_directory,
)
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


class VisualizationOutputPathTests(unittest.TestCase):
    def test_run_directory_is_grouped_by_model_then_start_time(self):
        started_at = datetime(2026, 9, 18, 3, 4, 5, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run_directory = create_visualization_run_directory(
                "model.onnx", started_at=started_at, output_root=output_root
            )
            local_started_at = started_at.astimezone()
            self.assertEqual(run_directory.parent.name, "model")
            self.assertEqual(
                run_directory.name, f"{local_started_at:%Y%m%d_%H%M%S}~running"
            )

    def test_finalized_directory_stays_in_model_folder(self):
        started_at = datetime(2026, 9, 18, 3, 4, 5, tzinfo=timezone.utc)
        finished_at = datetime(2026, 9, 18, 3, 5, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            staging = create_visualization_run_directory(
                "model.onnx", started_at=started_at, output_root=directory
            )
            model_directory = staging.parent
            finalized = finalize_visualization_run_directory(
                staging,
                "model.onnx",
                started_at=started_at,
                finished_at=finished_at,
            )
            self.assertEqual(finalized.parent, model_directory)
            self.assertEqual(
                finalized.name,
                f"{started_at.astimezone():%Y%m%d_%H%M%S}~"
                f"{finished_at.astimezone():%Y%m%d_%H%M%S}",
            )
            self.assertTrue(finalized.is_dir())


class ReLUSearchTreeTests(unittest.TestCase):
    def test_branch_spans_become_siblings_and_nested_children(self):
        trace = SolveTrace(events=[
            TraceEvent("reluplex_split", 0.9, 7.1, 0, "z1_2"),
            TraceEvent("reluplex_split", 1.9, 4.1, 1, "z2_4"),
            TraceEvent("reluplex_branch", 1.0, 5.0, 1, "z1_2",
                       {"phase": "+", "result": "SAT", "layer": 1, "index": 2}),
            TraceEvent("reluplex_branch", 2.0, 3.0, 2, "z2_4",
                       {"phase": "+", "result": "UNSAT", "layer": 2, "index": 4}),
            TraceEvent("reluplex_branch", 3.1, 4.0, 2, "z2_4",
                       {"phase": "-", "result": "SAT", "layer": 2, "index": 4}),
            TraceEvent("reluplex_branch", 6.0, 7.0, 1, "z1_2",
                       {"phase": "-", "result": "UNSAT", "layer": 1, "index": 2}),
        ])
        roots = build_relu_search_tree(trace)
        self.assertEqual(len(roots), 2)
        self.assertEqual(roots[0]["result"], "CLOSED")
        self.assertEqual(roots[0]["splitNumber"], 1)
        self.assertEqual([child["phase"] for child in roots[0]["children"]], ["+", "-"])
        self.assertEqual(
            [child["splitNumber"] for child in roots[0]["children"]], [2, 2]
        )
        self.assertEqual(roots[0]["children"][0]["result"], "CLOSED")
        self.assertEqual(roots[0]["subtreeSize"], 3)
        self.assertEqual(roots[0]["maxDepth"], 2)

        final_roots = build_relu_search_tree(trace, overall_status="UNSAT")
        self.assertEqual(final_roots[0]["children"][0]["result"], "CLOSED")
        self.assertEqual(final_roots[-1]["result"], "UNSAT")
        final_results = []
        pending = list(final_roots)
        while pending:
            item = pending.pop()
            final_results.append(item["result"])
            pending.extend(item["children"])
        self.assertEqual(final_results.count("UNSAT"), 1)

        nested_unsat = SolveTrace(events=[
            TraceEvent("reluplex_branch", 1.0, 4.0, 1, "z1_1",
                       {"phase": "+", "result": "UNSAT"}),
            TraceEvent("reluplex_branch", 2.0, 3.0, 2, "z2_1",
                       {"phase": "-", "result": "UNSAT"}),
        ])
        nested_roots = build_relu_search_tree(nested_unsat)
        self.assertEqual(nested_roots[0]["result"], "CLOSED")
        self.assertEqual(nested_roots[0]["children"][0]["result"], "CLOSED")
        final_nested_roots = build_relu_search_tree(
            nested_unsat, overall_status="UNSAT"
        )
        self.assertEqual(final_nested_roots[0]["result"], "UNSAT")
        self.assertEqual(
            final_nested_roots[0]["children"][0]["result"], "CLOSED"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested.html"
            write_relu_search_tree_html(
                nested_unsat,
                output,
                overall_status="UNSAT",
                theory_stats={
                    "DERIVE_TOTAL_RELUS": 4,
                    "DERIVE_FIXED_RELUS": 0,
                    "DERIVE_REMAINING_RELUS": 4,
                },
            )
            content = output.read_text(encoding="utf-8")
            # The depth-1 aggregate UNSAT contains the depth-2 terminal UNSAT;
            # only the terminal prefix prunes 2^(4-2), not both prefixes.
            self.assertIn('"exponent": 2', content)
            self.assertIn('"ratio": 0.25', content)

    def test_html_is_self_contained_and_contains_timeline(self):
        trace = SolveTrace(events=[
            TraceEvent("reluplex_branch", 1.0, 2.0, 1, "z3_7",
                       {"phase": "+", "result": "UNSAT", "bound": "[0, 1]"}),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "relu_search_tree.html"
            write_relu_search_tree_html(
                trace,
                output,
                overall_status="UNSAT",
                title="Tree test",
                theory_stats={
                    "DERIVE_TOTAL_RELUS": 27,
                    "DERIVE_FIXED_RELUS": 18,
                    "DERIVE_REMAINING_RELUS": 9,
                },
            )
            content = output.read_text(encoding="utf-8")
            self.assertIn("탐색 완료 순서 타임라인", content)
            self.assertIn('id="timelineOrder"', content)
            self.assertIn('<option value="visit">방문순</option>', content)
            self.assertIn('<option value="duration">소요시간순(긴 순)</option>', content)
            self.assertIn("DURATION_TIMELINE", content)
            self.assertIn("STATE_PREFIX='relu-search-tree-ui:'", content)
            self.assertIn("timelineOrder:$('#timelineOrder').value", content)
            self.assertIn('"phase": "+"', content)
            self.assertIn('id="neuronFilter"', content)
            self.assertIn('id="closedMetric"', content)
            self.assertIn('"closedCount": 0', content)
            self.assertIn('"closedReluCount": 0', content)
            self.assertIn('"total": 27', content)
            self.assertIn('"fixed": 18', content)
            self.assertIn('"remaining": 9', content)
            self.assertIn("닫힘으로 생략한 탐색 공간", content)
            self.assertIn('"totalExponent": 9', content)
            self.assertIn('"exponent": 8', content)
            self.assertIn('"ratio": 0.5', content)
            self.assertIn("탐색하지 않아도 되는 공간", content)
            self.assertIn("경계 유도 ReLU", content)
            self.assertIn('.CLOSED{background:#172f58', content)
            self.assertIn('.TIMEOUT{background:#4a3b0d', content)
            self.assertIn("내부적으로 UNSAT 판정된 중간 분기", content)
            self.assertIn('"splitCount": 1', content)
            self.assertIn("하위 탐색 최대 깊이", content)
            self.assertNotIn("https://", content)


class ReLUFeedbackHistoryTests(unittest.TestCase):
    def test_split_layer_and_neuron_are_preserved_for_lower_graph(self):
        trace = SimpleNamespace(events=[
            SimpleNamespace(
                component="reluplex_split",
                meta={"layer": 2, "index": 17},
                branch_x="z2_17",
                t_start=1.0,
                t_end=2.0,
            )
        ])
        events = split_events_from_trace(trace)
        self.assertEqual(
            [(event.event, event.layer, event.index) for event in events],
            [("+", 2, 17), ("-", 2, 17)],
        )

    def test_variable_name_is_used_when_split_metadata_is_missing(self):
        trace = SimpleNamespace(events=[
            SimpleNamespace(
                component="reluplex_split",
                meta={},
                branch_x="z3_11_vnnlib_23",
                t_start=1.0,
                t_end=2.0,
            )
        ])
        events = split_events_from_trace(trace)
        self.assertEqual(
            [(event.layer, event.index) for event in events],
            [(3, 11), (3, 11)],
        )


class RealtimeSearchTreeTests(unittest.TestCase):
    def test_branch_log_builds_live_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            split_log = Path(directory) / "relu_events.jsonl"
            logger = SplitEventLogger(split_log, reset_log=True)
            parent_split = logger.begin("z1_36", 1, 36)
            parent = logger.branch_begin(
                "z1_36", 1, 36, phase="+", depth=1, bound="[0, 1]",
                split_id=parent_split,
            )
            child_split = logger.begin("z1_12", 1, 12)
            child = logger.branch_begin(
                "z1_12", 1, 12, phase="-", depth=2, bound="[-1, 0]",
                split_id=child_split,
            )
            logger.branch_end(child, result="UNSAT")
            logger.end(child_split, "z1_12", 1, 12)
            logger.branch_end(parent, result="SAT")
            logger.end(parent_split, "z1_36", 1, 36)

            roots = build_relu_search_tree(
                trace_from_branch_log(branch_log_path(split_log))
            )
            self.assertEqual((roots[0]["layer"], roots[0]["neuron"]), (1, 36))
            self.assertEqual(roots[0]["splitNumber"], 1)
            self.assertEqual(roots[0]["children"][0]["splitNumber"], 2)
            self.assertEqual(roots[0]["children"][0]["phase"], "-")

            output = Path(directory) / "relu_search_tree.html"
            theory_stats = {
                "DERIVE_TOTAL_RELUS": 32,
                "DERIVE_FIXED_RELUS": 23,
                "DERIVE_REMAINING_RELUS": 9,
            }
            visualizer = RealtimeSearchTreeVisualizer(
                branch_log_path(split_log),
                output,
                theory_stats=theory_stats,
                title="ReLU Search Tree: test.onnx",
            )
            visualizer.set_overall("SAT", "THEORY_SAT")
            visualizer.request_update()
            visualizer.flush()
            content = output.read_text(encoding="utf-8")
            self.assertIn('"neuron": 36', content)
            self.assertIn('"remaining": 9', content)
            self.assertIn("ReLU Search Tree: test.onnx", content)

    def test_unified_dashboard_embeds_realtime_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "relu.png"
            page = root / "relu_search_tree.html"
            image.touch()
            page.write_text("<html></html>", encoding="utf-8")
            dashboard = root / "realtime.html"
            UnifiedRealtimeDashboard(
                {"relu": image},
                dashboard,
                embedded_pages={"relu-tree": page},
            ).write()
            content = dashboard.read_text(encoding="utf-8")
            self.assertIn("Interactive ReLU Search Tree", content)
            self.assertIn("<iframe", content)
            self.assertIn("auto refresh", content)
            self.assertIn('clearInterval(imageTimer)', content)
            self.assertIn('message.status === "RUNNING"', content)


if __name__ == "__main__":
    unittest.main()
