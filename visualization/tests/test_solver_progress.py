from __future__ import annotations

from pathlib import Path
from time import monotonic
import tempfile
import unittest
from unittest.mock import patch

from Automation.SolverStatus import SolverLimitReached
from DPLL import parse_prop
from DPLL_T import dpll_t_detailed
from Simplex import build_tableau, simplex
from visualization.SolverProgress import (
    _compact_theory_flow_lines,
    SolverProgressLogger,
    SolverProgressPanelVisualizer,
    SolverProgressVisualizer,
    build_solver_feedback,
    draw_dpll_theory_panel,
    draw_simplex_panel,
    draw_solver_dashboard,
    draw_theory_panel,
    read_solver_progress,
    write_solver_feedback,
    write_solver_feedback_panels,
)


class SolverProgressTests(unittest.TestCase):
    def test_round_frames_are_preserved_in_order_while_renderer_is_busy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "progress.jsonl"
            visualizer = SolverProgressPanelVisualizer(
                log_path,
                root / "realtime.png",
                enabled_panels={"dpll_theory"},
                refresh_interval_ms=1,
            )
            rendered_round_events = []

            def capture(_panels=None, events=None):
                if events is not None:
                    rendered_round_events.append(
                        [
                            (item["event"], item.get("round"))
                            for item in events
                            if item["event"] in {"round_start", "round_end"}
                        ]
                    )

            with patch.object(visualizer, "_render_once", side_effect=capture):
                logger = SolverProgressLogger(
                    log_path,
                    update_callback=visualizer.request_update,
                    reset_log=True,
                )
                logger.round_start(1)
                logger.round_end(1, "THEORY_CONFLICT")
                logger.round_start(2)
                logger.round_end(2, "THEORY_SAT")
                visualizer.flush()

            self.assertEqual(
                rendered_round_events,
                [
                    [("round_start", 1)],
                    [("round_start", 1), ("round_end", 1)],
                    [
                        ("round_start", 1),
                        ("round_end", 1),
                        ("round_start", 2),
                    ],
                    [
                        ("round_start", 1),
                        ("round_end", 1),
                        ("round_start", 2),
                        ("round_end", 2),
                    ],
                ],
            )

    def test_simplex_update_does_not_redraw_dpll_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visualizer = SolverProgressPanelVisualizer(
                root / "progress.jsonl", root / "realtime.png"
            )
            rendered_panels = []

            with patch.object(
                visualizer,
                "_render_once",
                side_effect=lambda panels=None, events=None: rendered_panels.append(
                    set(panels or ())
                ),
            ):
                visualizer.request_update(
                    panel="simplex", event="simplex_iteration"
                )
                visualizer.flush()

            self.assertEqual(rendered_panels, [{"simplex"}])

    def test_round_axes_only_show_recorded_integer_rounds(self):
        import matplotlib.pyplot as plt

        events = [
            {"event": "round_start", "round": 1, "time_ns": 1},
            {
                "event": "round_end",
                "round": 1,
                "time_ns": 2,
                "duration_seconds": 0.1,
                "result": "THEORY_SAT",
                "simplex_calls": 1,
            },
        ]

        dashboard_fig, dashboard_axes = draw_solver_dashboard(events)
        panel_fig, panel_axes = draw_dpll_theory_panel(events)
        theory_fig, theory_axes = draw_theory_panel(events)
        simplex_fig, simplex_axes = draw_simplex_panel(events)
        try:
            self.assertEqual(dashboard_axes[0].get_xticks().tolist(), [1])
            self.assertEqual(panel_axes[0, 0].get_xticks().tolist(), [1])
            self.assertEqual(panel_axes[0, 1].get_xticks().tolist(), [1])
            self.assertTrue(
                all(float(tick).is_integer() for tick in panel_axes[0, 1].get_yticks())
            )
            self.assertEqual(panel_axes[1, 0].get_xticks().tolist(), [])
            self.assertEqual(theory_axes[0].get_xticks().tolist(), [])
            self.assertEqual(panel_axes[1, 0].get_xlabel(), "")
            self.assertEqual(theory_axes[0].get_xlabel(), "")
            self.assertEqual(dashboard_axes[0].get_ylim()[0], 0.0)
            self.assertEqual(dashboard_axes[1].get_ylim()[0], 0.0)
            self.assertEqual(panel_axes[0, 0].get_ylim()[0], 0.0)
            self.assertEqual(simplex_axes[0, 0].get_ylim()[0], 0.0)
        finally:
            plt.close(dashboard_fig)
            plt.close(panel_fig)
            plt.close(theory_fig)
            plt.close(simplex_fig)

    def test_theory_flow_uses_compact_summary_instead_of_full_atom_id(self):
        long_atom_id = "very_long_formula_" * 30
        feedback = {
            "simplex_calls": [
                {"call_id": "c1", "round": 1, "result": "SAT"},
            ]
        }
        rounds = [{
            "round": 1,
            "theory_atoms": [
                {"atom_id": long_atom_id, "kind": "InequProp"},
                {"atom_id": "another", "kind": "ReLUProp"},
            ],
            "theory_result": "THEORY_SAT",
        }]

        lines = _compact_theory_flow_lines(feedback, rounds)

        self.assertEqual(len(lines), 1)
        self.assertIn("atoms 2 (InequProp:1, ReLUProp:1)", lines[0])
        self.assertIn("S1:SAT → SAT", lines[0])
        self.assertNotIn(long_atom_id, lines[0])

    def test_theory_result_frequency_excludes_boolean_only_rounds(self):
        import matplotlib.pyplot as plt

        events = [
            {"event": "round_start", "round": 1, "time_ns": 1},
            {
                "event": "theory_selection",
                "round": 1,
                "time_ns": 2,
                "atoms": [{"atom_id": "a1", "kind": "InequProp"}],
            },
            {
                "event": "theory_result",
                "round": 1,
                "time_ns": 3,
                "result": "THEORY_CONFLICT",
            },
            {
                "event": "round_end",
                "round": 1,
                "time_ns": 4,
                "duration_seconds": 0.1,
                "result": "THEORY_CONFLICT",
                "simplex_calls": 1,
            },
            {"event": "round_start", "round": 2, "time_ns": 5},
            {
                "event": "round_end",
                "round": 2,
                "time_ns": 6,
                "duration_seconds": 0.1,
                "result": "BOOLEAN_UNSAT",
                "simplex_calls": 0,
            },
        ]

        fig, axes = draw_dpll_theory_panel(events)
        try:
            labels = [label.get_text() for label in axes[1, 0].get_yticklabels()]
            self.assertEqual(labels, ["CONFLICT"])
        finally:
            plt.close(fig)

    def test_dpll_round_theory_atom_and_simplex_call_are_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            logger = SolverProgressLogger(path, reset_log=True)

            result = dpll_t_detailed(
                parse_prop("ineq(1,x,1) and ineq(-1,x,-2)"),
                progress=logger,
            )
            feedback = build_solver_feedback(read_solver_progress(path))

            self.assertEqual(result.reason, "THEORY_SAT")
            self.assertEqual(feedback["rounds"][0]["result"], "THEORY_SAT")
            self.assertEqual(feedback["rounds"][0]["simplex_calls"], 1)
            call = feedback["simplex_calls"][0]
            self.assertEqual(call["round"], 1)
            self.assertEqual(call["result"], "SAT")
            self.assertEqual(call["theory_atom_ids"], ["a1", "a2"])
            self.assertGreaterEqual(call["iterations"], 1)
            self.assertGreaterEqual(call["pivots"], 1)
            self.assertIn("ineq_slack_0", feedback["top_violated_variables"])
            self.assertIn("x", feedback["top_entering_variables"])

    def test_simplex_unsat_iteration_limit_and_timeout_are_recorded(self):
        cases = [
            ("UNSAT", 10, None, False),
            ("ITERATION_LIMIT", 0, None, True),
            ("TIMEOUT", 10, monotonic() - 1, True),
        ]
        for expected, max_iter, deadline, raises in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "progress.jsonl"
                logger = SolverProgressLogger(path, reset_log=True)
                tableau = build_tableau(
                    [("s", {"x": 1.0})],
                    {"s": (1.0, float("inf")), "x": (0.0, 0.0)},
                )

                if raises:
                    with self.assertRaises(SolverLimitReached):
                        simplex(
                            tableau,
                            max_iter=max_iter,
                            deadline=deadline,
                            report_unknown=True,
                            progress=logger,
                        )
                else:
                    _model, sat = simplex(tableau, max_iter=max_iter, progress=logger)
                    self.assertFalse(sat)

                ends = [
                    event for event in read_solver_progress(path)
                    if event["event"] == "simplex_end"
                ]
                self.assertEqual(len(ends), 1)
                self.assertEqual(ends[0]["result"], expected)

    def test_realtime_dashboard_and_feedback_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "progress.jsonl"
            realtime_path = root / "realtime.png"
            feedback_path = root / "feedback.png"
            visualizer = SolverProgressVisualizer(log_path, realtime_path, live_view=True)
            logger = SolverProgressLogger(log_path, reset_log=True)
            dpll_t_detailed(
                parse_prop("ineq(1,x,1) and ineq(-1,x,-2)"),
                progress=logger,
            )

            visualizer.request_update()
            visualizer.flush()
            feedback = write_solver_feedback(log_path, feedback_path)

            self.assertTrue(realtime_path.exists())
            self.assertTrue(realtime_path.with_suffix(".html").exists())
            self.assertTrue(feedback_path.exists())
            self.assertTrue(feedback_path.with_suffix(".json").exists())
            self.assertEqual(feedback["counts"]["rounds"], 1)
            self.assertEqual(feedback["counts"]["simplex_calls"], 1)

    def test_dpll_and_theory_share_one_panel_separate_from_simplex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "progress.jsonl"
            realtime_base = root / "realtime.png"
            feedback_base = root / "feedback.png"
            logger = SolverProgressLogger(log_path, reset_log=True)
            dpll_t_detailed(
                parse_prop("ineq(1,x,1) and ineq(-1,x,-2)"),
                progress=logger,
            )

            visualizer = SolverProgressPanelVisualizer(log_path, realtime_base)
            visualizer.request_update()
            visualizer.flush()
            feedback, feedback_paths = write_solver_feedback_panels(
                log_path, feedback_base
            )

            self.assertEqual(set(visualizer.panel_paths), {"dpll_theory", "simplex"})
            for path in visualizer.panel_paths.values():
                self.assertTrue(path.exists())
            for path in feedback_paths.values():
                self.assertTrue(path.exists())
            self.assertTrue(feedback_base.with_suffix(".json").exists())
            self.assertEqual(feedback["counts"]["rounds"], 1)

    def test_realtime_visualizer_and_logger_only_run_selected_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "progress.jsonl"
            realtime_base = root / "realtime.png"
            logger = SolverProgressLogger(
                log_path,
                reset_log=True,
                enabled_panels={"dpll_theory"},
            )
            dpll_t_detailed(
                parse_prop("ineq(1,x,1) and ineq(-1,x,-2)"),
                progress=logger,
            )

            visualizer = SolverProgressPanelVisualizer(
                log_path,
                realtime_base,
                enabled_panels={"dpll_theory"},
            )
            visualizer.request_update()
            visualizer.flush()

            self.assertEqual(set(visualizer.panel_paths), {"dpll_theory"})
            self.assertTrue(visualizer.panel_paths["dpll_theory"].exists())
            self.assertFalse(
                realtime_base.with_name(
                    f"{realtime_base.stem}_simplex{realtime_base.suffix}"
                ).exists()
            )
            self.assertFalse(
                any(
                    event["event"].startswith("simplex_")
                    for event in read_solver_progress(log_path)
                )
            )

    def test_feedback_writer_only_renders_selected_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "progress.jsonl"
            feedback_base = root / "feedback.png"
            logger = SolverProgressLogger(log_path, reset_log=True)
            dpll_t_detailed(
                parse_prop("ineq(1,x,1) and ineq(-1,x,-2)"),
                progress=logger,
            )

            _, paths = write_solver_feedback_panels(
                log_path,
                feedback_base,
                enabled_panels={"dpll_theory"},
            )

            self.assertEqual(set(paths), {"dpll_theory"})
            self.assertTrue(paths["dpll_theory"].exists())
            self.assertFalse(
                feedback_base.with_name(
                    f"{feedback_base.stem}_simplex{feedback_base.suffix}"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
