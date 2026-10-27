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
    def test_explicit_unsat_origin_prevents_false_discovery_in_partial_logs(self):
        for origin in ("found", "propagated", "skipped"):
            with self.subTest(origin=origin):
                trace = SolveTrace(events=[
                    TraceEvent("reluplex_branch", 1, 2, 1, "z1_1",
                               {"phase": "+", "result": "UNSAT", "unsat_origin": origin}),
                ])
                branch = build_relu_search_tree(trace, overall_status="RUNNING")[0]
                self.assertEqual(branch["unsatRole"], origin)
                self.assertEqual(
                    branch["unsatSourceIds"], [branch["id"]] if origin == "found" else []
                )

    def test_live_and_completed_trees_preserve_immediate_parent_for_both_results(self):
        for result in ("SAT", "UNSAT"):
            with self.subTest(result=result):
                trace = SolveTrace(events=[
                    TraceEvent("reluplex_branch", 1, None, 1, "z1_10",
                               {"phase": "+", "result": "RUNNING"}),
                    TraceEvent("reluplex_branch", 2, 9, 2, "z2_20",
                               {"phase": "+", "result": result}),
                    TraceEvent("reluplex_branch", 3, 8, 3, "z3_30",
                               {"phase": "-", "result": result}),
                ])
                root = build_relu_search_tree(trace, overall_status="RUNNING")[0]
                returned = root["children"][0]
                found = returned["children"][0]
                self.assertIsNone(root["parentId"])
                self.assertEqual(returned["parentId"], root["id"])
                self.assertEqual(found["parentId"], returned["id"])
                self.assertEqual(root["result"], "RUNNING")
                role_key = "satRole" if result == "SAT" else "unsatRole"
                self.assertEqual(returned[role_key], "propagated")
                self.assertEqual(found[role_key], "found")

                trace.events[0].t_end = 10
                trace.events[0].meta["result"] = result
                completed = build_relu_search_tree(trace, overall_status=result)[0]
                self.assertEqual(completed["result"], result)
                self.assertEqual(completed["children"][0]["parentId"], root["id"])
                self.assertEqual(completed["children"][0][role_key], "propagated")

    def test_unsat_conflicts_are_distinct_from_returns_and_backjump_skips(self):
        trace = SolveTrace(events=[
            TraceEvent("reluplex_branch", 1, 15, 1, "z1_10",
                       {"phase": "+", "result": "UNSAT"}),
            TraceEvent("reluplex_branch", 2, 9, 2, "z2_20",
                       {"phase": "+", "result": "UNSAT"}),
            TraceEvent("reluplex_branch", 3, 5, 3, "z3_30",
                       {"phase": "+", "result": "UNSAT"}),
            TraceEvent("reluplex_branch", 6, 8, 3, "z3_30",
                       {"phase": "-", "result": "UNSAT", "reason": "BOUNDS_CONFLICT"}),
            TraceEvent("reluplex_branch", 10, 11, 2, "z2_20",
                       {"phase": "-", "result": "UNSAT", "reason": "BACKJUMP"}),
        ])
        root = build_relu_search_tree(trace, overall_status="UNSAT")[0]
        forwarded, skipped = root["children"]
        found = forwarded["children"]
        self.assertEqual(root["result"], "UNSAT")
        self.assertEqual(root["unsatRole"], "propagated")
        self.assertEqual(forwarded["unsatRole"], "propagated")
        source_ids = [branch["id"] for branch in found]
        self.assertEqual(root["unsatSourceIds"], source_ids)
        self.assertEqual(forwarded["unsatSourceIds"], source_ids)
        for branch in found:
            self.assertEqual(branch["unsatRole"], "found")
            self.assertEqual(branch["unsatSourceIds"], [branch["id"]])
        self.assertEqual(skipped["unsatRole"], "skipped")
        self.assertEqual(skipped["unsatSourceIds"], [])
        self.assertTrue(skipped["prunesSpace"])
        self.assertIsNone(root["satRole"])

        trace.events[0].t_end = None
        trace.events[0].meta["result"] = "RUNNING"
        live_root = build_relu_search_tree(trace, overall_status="RUNNING")[0]
        self.assertIsNone(live_root["unsatRole"])
        self.assertEqual(live_root["unsatSourceIds"], [])
        self.assertEqual(live_root["children"][0]["unsatRole"], "propagated")

    def test_sat_discovery_is_distinct_from_ancestor_returns(self):
        trace = SolveTrace(events=[
            TraceEvent("reluplex_branch", 1, 10, 1, "z1_10",
                       {"phase": "+", "result": "SAT"}),
            TraceEvent("reluplex_branch", 2, 3, 2, "z2_20",
                       {"phase": "+", "result": "UNSAT"}),
            TraceEvent("reluplex_branch", 4, 9, 2, "z2_20",
                       {"phase": "-", "result": "SAT"}),
            TraceEvent("reluplex_branch", 5, 8, 3, "z3_30",
                       {"phase": "+", "result": "SAT"}),
        ])
        root = build_relu_search_tree(trace, overall_status="SAT")[0]
        rejected, forwarded = root["children"]
        found = forwarded["children"][0]
        self.assertEqual(root["result"], "SAT")
        self.assertEqual(root["satRole"], "propagated")
        self.assertEqual(forwarded["satRole"], "propagated")
        self.assertEqual(found["satRole"], "found")
        for branch in (root, forwarded, found):
            self.assertEqual(branch["satSourceId"], found["id"])
            self.assertFalse(branch["prunesSpace"])
        self.assertIsNone(rejected["satRole"])
        self.assertIsNone(rejected["satSourceId"])
        self.assertTrue(rejected["prunesSpace"])

        # A live snapshot can already contain the discovery while ancestors
        # are still returning. Do not label those RUNNING branches as SAT.
        trace.events[0].t_end = None
        trace.events[0].meta["result"] = "RUNNING"
        live_root = build_relu_search_tree(trace, overall_status="RUNNING")[0]
        self.assertIsNone(live_root["satRole"])
        self.assertEqual(live_root["result"], "RUNNING")
        self.assertEqual(live_root["children"][1]["satRole"], "propagated")
        self.assertEqual(
            live_root["children"][1]["children"][0]["satRole"], "found"
        )

    def test_sat_found_without_children_and_non_sat_results(self):
        for result in ("SAT", "UNSAT", "UNKNOWN", "TIMEOUT", "RUNNING"):
            with self.subTest(result=result):
                trace = SolveTrace(events=[
                    TraceEvent("reluplex_branch", 1, 2, 1, "z1_1",
                               {"phase": "+", "result": result}),
                ])
                branch = build_relu_search_tree(trace, overall_status=result)[0]
                self.assertEqual(
                    branch["satRole"], "found" if result == "SAT" else None
                )
                self.assertEqual(
                    branch["satSourceId"], branch["id"] if result == "SAT" else None
                )
                self.assertEqual(
                    branch["unsatRole"], "found" if result == "UNSAT" else None
                )
                self.assertEqual(
                    branch["unsatSourceIds"], [branch["id"]] if result == "UNSAT" else []
                )
        self.assertEqual(build_relu_search_tree(SolveTrace(), overall_status="SAT"), [])

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
        self.assertEqual(roots[0]["branchResult"], "SAT")
        self.assertFalse(roots[0]["prunesSpace"])
        self.assertEqual(roots[0]["splitNumber"], 1)
        self.assertEqual([child["phase"] for child in roots[0]["children"]], ["+", "-"])
        self.assertEqual(
            [child["splitNumber"] for child in roots[0]["children"]], [2, 2]
        )
        self.assertEqual(roots[0]["children"][0]["result"], "CLOSED")
        self.assertEqual(
            [child["branchResult"] for child in roots[0]["children"]],
            ["UNSAT", "SAT"],
        )
        self.assertEqual(
            [child["prunesSpace"] for child in roots[0]["children"]],
            [True, False],
        )
        self.assertEqual(roots[0]["subtreeSize"], 3)
        self.assertEqual(roots[0]["maxDepth"], 2)

        final_roots = build_relu_search_tree(trace, overall_status="UNSAT")
        self.assertEqual(final_roots[0]["children"][0]["result"], "CLOSED")
        self.assertEqual(final_roots[-1]["result"], "UNSAT")
        self.assertEqual(final_roots[-1]["branchResult"], "UNSAT")
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
            self.assertIn("SAT 발견", content)
            self.assertIn("SAT 전달", content)
            self.assertIn("CLOSED (UNSAT)", content)
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
    def test_solver_records_direct_conflicts_and_returns_in_trace_and_live_log(self):
        from Reluplex import reluplex

        rows = [
            ("total", {"h1_0": 1.0, "h1_1": 1.0}),
            ("sum_z", {"z1_0": 1.0, "z1_1": 1.0}),
        ]
        bounds = {
            "total": (1.5, 2.0), "sum_z": (-2.0, 0.0),
            "z1_0": (-1.0, 1.0), "z1_1": (-1.0, 1.0),
            "h1_0": (0.0, 1.0), "h1_1": (0.0, 1.0),
        }
        # ReLU outputs cannot sum to 1.5 when both inputs sum to at most 0.
        # Disable tightening/repair here so the proof visits both branch phases.
        with tempfile.TemporaryDirectory() as directory:
            split_log = Path(directory) / "relu_events.jsonl"
            logger = SplitEventLogger(split_log, reset_log=True)
            trace = SolveTrace()
            _, sat = reluplex(
                rows, bounds, [("z1_0", "h1_0"), ("z1_1", "h1_1")],
                trace=trace, split_logger=logger, derive_bound=False,
                derive_in_search=False, pivot_tighten=False,
                local_repair_max_iter=0, warm_start=False, report_unknown=True,
            )
            self.assertFalse(sat)
            feedback_events = [e for e in trace.events if e.component == "reluplex_branch"]
            live_trace = trace_from_branch_log(branch_log_path(split_log))
            feedback = [(e.branch_x, e.meta["phase"], e.meta.get("unsat_origin"))
                        for e in feedback_events]
            live = [(e.branch_x, e.meta["phase"], e.meta.get("unsat_origin"))
                    for e in live_trace.events]
            self.assertEqual(live, feedback)
            self.assertEqual(
                feedback,
                [("z1_0", "+", "propagated"), ("z1_1", "+", "found"),
                 ("z1_1", "-", "found"), ("z1_0", "-", "found")],
            )
            roots = build_relu_search_tree(live_trace, overall_status="UNSAT")
            self.assertEqual(roots[0]["unsatRole"], "propagated")
            self.assertEqual(
                [child["unsatRole"] for child in roots[0]["children"]],
                ["found", "found"],
            )

    def test_live_branch_log_distinguishes_unsat_and_backjump_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            split_log = Path(directory) / "relu_events.jsonl"
            logger = SplitEventLogger(split_log, reset_log=True)
            parent = logger.branch_begin(
                "z1_10", 1, 10, phase="+", depth=1, bound="[0, 1]",
            )
            found = logger.branch_begin(
                "z2_20", 2, 20, phase="+", depth=2, bound="[0, 1]",
            )
            logger.branch_end(found, result="UNSAT", reason="BOUNDS_CONFLICT")
            skipped = logger.branch_begin(
                "z2_20", 2, 20, phase="-", depth=2, bound="[-1, 0]",
            )
            logger.branch_end(skipped, result="UNSAT", reason="BACKJUMP")
            logger.branch_end(parent, result="UNSAT")
            root = build_relu_search_tree(
                trace_from_branch_log(branch_log_path(split_log)),
                overall_status="UNSAT",
            )[0]
            self.assertEqual(root["unsatRole"], "propagated")
            self.assertEqual(
                [child["unsatRole"] for child in root["children"]],
                ["found", "skipped"],
            )
            self.assertEqual(root["unsatSourceIds"], [root["children"][0]["id"]])

    def test_live_branch_log_preserves_discovery_during_sat_return(self):
        with tempfile.TemporaryDirectory() as directory:
            split_log = Path(directory) / "relu_events.jsonl"
            logger = SplitEventLogger(split_log, reset_log=True)
            parent = logger.branch_begin(
                "z1_10", 1, 10, phase="+", depth=1, bound="[0, 1]",
            )
            child = logger.branch_begin(
                "z2_20", 2, 20, phase="-", depth=2, bound="[-1, 0]",
            )
            logger.branch_end(child, result="SAT")
            live_root = build_relu_search_tree(
                trace_from_branch_log(branch_log_path(split_log)),
                overall_status="RUNNING",
            )[0]
            self.assertEqual(live_root["result"], "RUNNING")
            self.assertIsNone(live_root["satRole"])
            self.assertEqual(live_root["children"][0]["satRole"], "found")

            logger.branch_end(parent, result="SAT")
            final_root = build_relu_search_tree(
                trace_from_branch_log(branch_log_path(split_log)),
                overall_status="SAT",
            )[0]
            self.assertEqual(final_root["satRole"], "propagated")
            self.assertEqual(
                final_root["satSourceId"], final_root["children"][0]["id"]
            )

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
            self.assertEqual(roots[0]["branchResult"], "SAT")
            self.assertEqual(roots[0]["children"][0]["branchResult"], "UNSAT")

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
