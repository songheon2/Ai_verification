from concurrent.futures import Future
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

from Automation.SolverStatus import SolverLimitReached
from Reluplex import reluplex
from visualization.SolverProgress import (
    SolverProgressLogger, SolverProgressPanelVisualizer, SolverProgressReader,
    build_solver_feedback, draw_simplex_panel, read_solver_progress,
)


class ExternalLpProgressTests(unittest.TestCase):
    def test_realtime_progress_reader_matches_full_replay_across_updates_and_reset(self):
        with tempfile.TemporaryDirectory() as directory, patch("visualization.SolverProgress.time_ns") as now:
            path = Path(directory) / "progress.jsonl"
            reader = SolverProgressReader(path)
            now.return_value = 2_000_000_000
            self.assertEqual(reader.read(), build_solver_feedback([]))
            path.write_bytes(b'{"event":"round_start","round":1,"time_ns":1000000000}\n'
                             b'{"event":"simplex_start","round":1,"call_id":"c1","time_ns":1500000000}\n')
            running = reader.read()
            self.assertEqual(running, build_solver_feedback(read_solver_progress(path)))
            with path.open("ab") as stream:
                stream.write(b'{"event":"simplex_end","call_id":"c1",')
            now.return_value = 3_000_000_000
            self.assertEqual(reader.read()["simplex_calls"][0]["duration_seconds"], 1.5)
            with path.open("ab") as stream:
                stream.write(b'"duration_seconds":1.2,"result":"SAT","iterations":3}\n'
                             b'{"event":"round_end","round":1,"result":"THEORY_SAT","simplex_calls":1}\n')
            self.assertEqual(reader.read(), build_solver_feedback(read_solver_progress(path)))
            self.assertEqual(running["simplex_calls"][0]["result"], "RUNNING")
            path.write_bytes(b'{"event":"round_start","round":2,"time_ns":2500000000}\n' + b'\n' * 512)
            self.assertEqual(reader.read(), build_solver_feedback(read_solver_progress(path)))
            replacement = path.with_name("next.jsonl")
            replacement.write_bytes(b'{"event":"round_start","round":3,"time_ns":2500000000}\n')
            replacement.replace(path)
            self.assertEqual(reader.read(), build_solver_feedback(read_solver_progress(path)))

    def test_simplex_panel_labels_follow_the_reported_engine_and_metric(self):
        from matplotlib import pyplot as plt

        cases = (
            ("native", "", "Pivots per call", "Repeated bound violations", "피벗 횟수: 7"),
            ("simplex", "python-simplex-callbacks", "Pivots per call", "Repeated bound violations", "피벗 횟수: 7"),
            ("glpk", "glpk-iteration-counter", "GLPK iterations per call", "Bound violations at LP call entry", "GLPK 내부 반복 횟수: 12"),
            ("z3", "z3-arithmetic-pivot-counter", "Z3 arithmetic pivots per call", "Bound violations at LP call entry", "Z3 산술 피벗 횟수: 7"),
            ("z3", "unavailable", "Z3 arithmetic pivots per call", "Bound violations at LP call entry", "LP 엔진: Z3"),
        )
        for engine, source, work_title, bounds_title, tooltip in cases:
            with self.subTest(engine=engine, source=source):
                step = {"event": "simplex_call", "call_id": "simplex-1", "result": "SAT",
                        "lp_engine": engine, "metrics_source": source,
                        "pivot_count_available": engine != "glpk" and source != "unavailable",
                        "iteration_count_available": engine != "z3", "pivots": 7, "iterations": 12,
                        "bound_violation_scope": "solver_iteration" if engine in ("native", "simplex") else "call_boundary"}
                fig, axes = draw_simplex_panel([step])
                try:
                    self.assertEqual(axes[0, 1].get_title(), work_title)
                    self.assertEqual(axes[1, 1].get_title(), bounds_title)
                    target = fig._analysis_selection_geometry[1][1][0]["text"]
                    self.assertIn(tooltip, target)
                    if engine == "glpk":
                        self.assertNotIn("피벗 횟수", target)
                        self.assertIn("GLPK iterations: 12", fig.texts[-1].get_text())
                    if source == "unavailable":
                        self.assertIn("No counts reported", [label.get_text() for label in axes[0, 1].texts])
                        self.assertEqual(fig.texts[-1].get_text(), "Calls: 1")
                finally:
                    plt.close(fig)

    def test_unresponsive_render_child_is_reaped_and_next_request_recovers(self):
        from unittest.mock import Mock
        from visualization import RenderProcess

        old = Mock()
        old.is_alive.side_effect = [True, False, False]
        blocked = Mock()
        blocked.poll.return_value = False
        with patch.object(RenderProcess, "_renderer", (old, blocked)):
            with self.assertRaises(TimeoutError):
                RenderProcess._execute_render("solver")
            old.terminate.assert_called_once()
            self.assertIsNone(RenderProcess._renderer)
            replacement, child = Mock(), Mock()
            replacement.poll.return_value = True
            replacement.recv.return_value = None
            context = Mock()
            context.Pipe.return_value = replacement, child
            with patch.object(RenderProcess.multiprocessing, "get_context", return_value=context):
                RenderProcess._execute_render("solver", "log.jsonl")
            context.Process.return_value.start.assert_called_once()
            replacement.send.assert_called_once_with(("solver", ("log.jsonl",)))

    def test_external_validation_failure_records_actual_residual(self):
        from unittest.mock import Mock
        engine = Mock()
        engine.name = "simplex"
        engine.failures = 1
        error = SolverLimitReached("LP_MODEL_INVALID")
        error.details = {"kind": "equation", "residual": 1.9e-5, "tolerance": 1e-9}
        engine.solve.side_effect = error
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "solver.jsonl"
            with SolverProgressLogger(path) as logger, patch("LpEngines.make_engine", return_value=engine):
                logger.round_start(1)
                with self.assertRaises(SolverLimitReached):
                    reluplex([("s", {"x": 1.})], {"x": (0., 1.), "s": (0., 1.)}, [],
                             derive_bound=False, progress=logger, lp_engine="simplex")
            records = read_solver_progress(path)
            end = next(record for record in records if record["event"] == "simplex_end")
            self.assertEqual(end["model_validation"], error.details)

    def test_external_calls_reach_round_and_call_panels(self):
        for engine, modes in (("native", ("core",)), ("simplex", ("oracle",)),
                              ("glpk", ("core", "oracle")), ("z3", ("core", "oracle"))):
            if engine in {"z3", "glpk"}:
                try:
                    import importlib
                    importlib.import_module("z3" if engine == "z3" else "swiglpk")
                except (ImportError, OSError):
                    continue
            for mode in modes:
                for compact in (False, True):
                    for expected in (False, True):
                        with self.subTest(engine=engine, mode=mode, compact=compact,
                                          sat=expected), tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / "progress.jsonl"
                            stats = {}
                            with SolverProgressLogger(path, compact_simplex_calls=compact) as logger:
                                logger.round_start(1)
                                _, sat = reluplex(
                                    [("eq_pos", {"z": 1.0, "x": -1.0}),
                                     ("eq_neg", {"z": -1.0, "x": 1.0})],
                                    {"x": (-2.0, 2.0 if expected else 0.0),
                                     "z": (-2.0, 2.0 if expected else 0.0), "h": (1.0, 2.0),
                                     "eq_pos": (0.0, float("inf")), "eq_neg": (0.0, float("inf"))},
                                    [("z", "h")], progress=logger,
                                    progress_context={"round_index": 1}, stats=stats,
                                    lp_engine=engine, lp_mode=mode,
                                    derive_bound=False, derive_in_search=False,
                                    pivot_tighten=False, local_repair_max_iter=0,
                                )
                                logger.round_end(1, "THEORY_SAT" if sat else "THEORY_UNSAT")
                            self.assertEqual(sat, expected)
                            feedback = build_solver_feedback(read_solver_progress(path))
                            calls = feedback["simplex_calls"]
                            self.assertGreater(len(calls), 0)
                            self.assertEqual(len(calls), stats["LP_CALLS"])
                            self.assertEqual(feedback["rounds"][0]["simplex_calls"], len(calls))
                            self.assertTrue(all(call["result"] in {"SAT", "UNSAT"} for call in calls))
                            self.assertTrue(all(call["duration_seconds"] >= 0 for call in calls))
                            if engine != "native":
                                self.assertTrue(all(call["lp_engine"] == engine for call in calls))
                                if engine == "simplex":
                                    self.assertTrue(all(call["work_counts_available"] for call in calls))
                                    self.assertTrue(all(call["metrics_source"] == "python-simplex-callbacks" for call in calls))
                                else:
                                    self.assertTrue(all(call["work_counts_available"] is False for call in calls))
                                    if engine == "glpk":
                                        self.assertTrue(all(call["iteration_count_available"] for call in calls))

    def test_external_timeout_closes_active_call(self):
        class TimedOutCore:
            name = "z3"
            needs_row_error = False

            def simplex(self, *_args, **_kwargs):
                active = build_solver_feedback(read_solver_progress(path))
                test_case.assertEqual(active["rounds"][0]["simplex_calls"], 1)
                test_case.assertEqual(active["simplex_calls"][0]["result"], "RUNNING")
                test_case.assertFalse(active["simplex_calls"][0]["work_counts_available"])
                raise SolverLimitReached("TIMEOUT")

        test_case = self
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            with SolverProgressLogger(path) as logger, patch("LpEngines.make_core", return_value=TimedOutCore()):
                logger.round_start(1)
                with self.assertRaises(SolverLimitReached):
                    reluplex([("z", {"x": 1.0})],
                             {"x": (-2, 2), "z": (-2, 2), "h": (1, 2)}, [("z", "h")],
                             lp_engine="z3", lp_mode="core", derive_bound=False,
                             derive_in_search=False, progress=logger,
                             progress_context={"round_index": 1})
                self.assertFalse(logger._simplex_started)
            calls = build_solver_feedback(read_solver_progress(path))["simplex_calls"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["result"], "TIMEOUT")

    def test_running_round_retries_after_locked_image_without_new_events(self):
        worker = SolverProgressPanelVisualizer("solver.jsonl", "solver.png")
        worker._round_active = True
        attempts = []

        def render(*_args):
            attempts.append(1)
            future = Future()
            if len(attempts) == 1:
                future.set_exception(PermissionError("image locked"))
            else:
                worker._round_active = False
                future.set_result(None)
            return future

        with patch("visualization.SolverProgress.submit_solver_render", side_effect=render), \
             patch("visualization.SolverProgress.sleep"), warnings.catch_warnings(record=True) as caught:
            worker._run_updates()
        self.assertEqual(len(attempts), 2)
        self.assertTrue(any("재시도" in str(item.message) for item in caught))
        self.assertIsNone(worker._worker)

    def test_logger_round_start_keeps_rendering_without_lp_events(self):
        worker = SolverProgressPanelVisualizer("solver.jsonl", "solver.png")
        rendered = []

        def render(*_args):
            rendered.append(1)
            if len(rendered) == 2:
                with worker._state_lock:
                    worker._round_active = False
            future = Future()
            future.set_result(None)
            return future

        with tempfile.TemporaryDirectory() as tmp, \
             patch("visualization.SolverProgress.submit_solver_render", side_effect=render), \
             patch("visualization.SolverProgress.sleep"):
            with SolverProgressLogger(Path(tmp) / "solver.jsonl", update_callback=worker.request_update) as logger:
                logger.round_start(1)
                worker.flush()
        self.assertEqual(len(rendered), 2)
