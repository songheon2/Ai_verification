"""Regression checks for solver observations and live/feedback parity."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import json
from itertools import count
from unittest.mock import patch

import matplotlib.pyplot as plt

from Automation.tests.test_automation import _random_relu_problem
from Reluplex import reluplex
from Simplex import build_tableau, simplex
from Simplex import DRIFT_CHECK_AT_EXIT, DRIFT_TOL
from BoundDerivation import derive_bounds
from visualization.RealtimeSplitVisualization import SplitEventLogger
from visualization.ReluSearchTree import (
    branch_log_path, build_mechanism_report, build_relu_search_tree, build_stable_records,
    trace_from_branch_log,
)
from visualization.SolveTrace import SolveTrace, span
from visualization.SolverProgress import (
    SolverProgressLogger, build_solver_feedback, draw_simplex_panel,
    read_solver_progress,
)
from visualization.SolverMechanisms import mechanism_report_path
from visualization.ReluSearchTree import write_relu_search_tree_html


def report_for(trace, status):
    roots = build_relu_search_tree(trace, overall_status=status)
    nodes = []
    pending = list(roots)
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node["children"])
    return build_mechanism_report(trace, nodes), nodes


class MechanismObservationTests(unittest.TestCase):
    def test_result_colors_distinguish_conflict_and_timeout_from_running(self):
        from visualization.SolverProgress import _result_color

        expected = {'SAT': '#16a34a', 'UNSAT': '#dc2626', 'RUNNING': '#2563eb',
                    'CONFLICT': '#9333ea', 'TIMEOUT': '#ea580c'}
        for result, color in expected.items():
            self.assertEqual(_result_color(result), color)
            self.assertEqual(_result_color('THEORY_' + result), color)
        self.assertEqual(len(set(expected.values())), 5)
        self.assertEqual(_result_color('UNKNOWN'), '#64748b')

    def test_call_duration_always_explains_sat_and_unsat_colors(self):
        fig, axes = draw_simplex_panel([])
        try:
            legend = axes.flat[0].get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual([label.get_text() for label in legend.get_texts()], ['SAT','UNSAT'])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = legend.get_window_extent(renderer)
            self.assertGreater(bounds.x0, axes.flat[0].get_window_extent(renderer).x1)
            self.assertLess(bounds.x1, axes.flat[1].get_window_extent(renderer).x0)
            self.assertGreater(legend.get_texts()[0].get_window_extent(renderer).y0,
                               legend.get_texts()[1].get_window_extent(renderer).y0)
        finally:
            plt.close(fig)

    def test_occurrence_records_include_all_stable_fixations_only_for_bounds(self):
        trace = SolveTrace()
        with span(trace, 'reluplex_mechanism', kind='bounds', source='search', fixed=0, changed=10):
            pass
        for index in range(70):
            with span(trace, 'reluplex_stable', source='search', depth=2, round_index=1,
                      records=[{'variable': f'z0_{index}', 'state': 'active', 'before': [-1,2], 'after': [0,2]}]):
                pass
            with span(trace, 'reluplex_mechanism', kind='numerical', action='pivot', pivot=index):
                pass
            with span(trace, 'reluplex_mechanism', kind='conflict', source='LP'):
                pass
        report, _ = report_for(trace, 'UNSAT')
        stable = report['eventsByKind']['bounds']
        self.assertEqual(len(stable), 70)
        self.assertTrue(all(record['kind']=='stable' for record in stable))
        self.assertEqual(stable[0]['variable'], 'z0_0')
        self.assertEqual(stable[-1]['variable'], 'z0_69')
        self.assertEqual(len(report['eventsByKind']['numerical']), 70)
        self.assertEqual(len(report['eventsByKind']['backjump']), 70)
        self.assertEqual(len(report['events']), 211)
        self.assertEqual(len({record['sequence'] for record in report['events']}), 211)

    def test_initial_neuron_ranges_match_actual_bound_derivation(self):
        rows, bounds, relus, *_ = _random_relu_problem(0)
        changes = []
        tightened, _fixed, _infeasible = derive_bounds(rows, bounds, relus,
            on_change=lambda variable, before, after: changes.append((variable, before, after)))
        self.assertEqual(derive_bounds(rows, bounds, relus), (tightened, _fixed, _infeasible))
        trace = SolveTrace()
        reluplex(rows, dict(bounds), relus, max_recursion=100, trace=trace)
        report, _ = report_for(trace, 'UNSAT')
        ranges = report['boundRounds'][0]['ranges']
        self.assertEqual(len(ranges), len(relus))
        for record in ranges:
            self.assertEqual(record['after'], list(tightened[record['variable']]))
        self.assertEqual(report['eventsByKind']['underapprox'], [])
        self.assertFalse(report['settings'][0]['underapprox_enabled'])
        self.assertNotIn('repair', report['eventsByKind'])
        self.assertNotIn('cycles', report['eventsByKind'])
        neuron_variables = {variable for pair in relus for variable in pair}
        expected = [(var, [v if abs(v) != float('inf') else None for v in before],
                     [v if abs(v) != float('inf') else None for v in after])
                    for var, before, after in changes if var in neuron_variables]
        actual = [(record['variable'], record['before'], record['after'])
                  for record in report['boundHistory'] if record['source'] == 'initial']
        self.assertEqual(actual, expected)
        self.assertEqual([record['id'] for record in report['boundHistory']], list(range(1,len(report['boundHistory'])+1)))
        self.assertTrue(any(record['role']=='h' for record in report['boundHistory']))
        self.assertTrue(any(record['source']=='search' for record in report['boundHistory']))

    def test_bound_history_is_not_truncated_to_recent_50_events(self):
        trace = SolveTrace()
        for batch in range(3):
            changes = [{"variable": "z0_0", "before": [-i-1, i+1], "after": [-i, i]}
                       for i in range(batch*64,(batch+1)*64)]
            with span(trace,'reluplex_mechanism',kind='bound_updates',depth=1,changes=changes,round_index=1):
                pass
        report, _ = report_for(trace,'SAT')
        self.assertEqual(len(report['boundHistory']),192)
        self.assertEqual(report['boundHistory'][0]['id'],1)
        self.assertEqual(report['boundHistory'][-1]['id'],192)

    def test_numerical_observations_do_not_change_solve_and_show_restoration(self):
        tableau = build_tableau([('s', {'x': 1., 'y': 1.})],
                                {'s': (3., float('inf')), 'x': (0., 1.), 'y': (0., 5.)})
        tableau.rows[0].coeffs['y'] = -1.
        tableau.pivots_since_check = DRIFT_CHECK_AT_EXIT
        baseline = simplex(copy.deepcopy(tableau))
        observed = []
        stats = {}
        actual = simplex(copy.deepcopy(tableau), stats=stats, numerical_callback=lambda **record: observed.append(record))
        self.assertEqual(actual, baseline)
        checks = [record for record in observed if record['action'] == 'check']
        self.assertEqual(len(checks), stats['SIMPLEX_DRIFT_CHECKS'])
        restored = [record for record in checks if record['outcome'] == 'restored']
        self.assertEqual(len(restored), stats['SIMPLEX_RESTORES'])
        self.assertGreater(restored[0]['drift'], DRIFT_TOL)
        self.assertLessEqual(restored[0]['after_drift'], DRIFT_TOL)
        self.assertGreaterEqual(restored[0]['residual'], 0.)
        self.assertLessEqual(restored[0]['after_residual'], 1e-12)

    def test_pivot_sample_records_actual_coefficient_and_small_entry_exclusion(self):
        tableau = build_tableau([('s', {'x': 1., 'y': 1e-15})],
                                {'s': (1., 2.), 'x': (0., 2.), 'y': (0., 2.)})
        samples = []
        _, sat = simplex(tableau, numerical_callback=lambda **record: samples.append(record))
        self.assertTrue(sat)
        pivot = next(record for record in samples if record['action'] == 'pivot')
        self.assertEqual(pivot['coefficient'], 1.)
        self.assertEqual(pivot['zero_threshold'], 1e-12)
        self.assertEqual(pivot['excluded_small'], 1)

    def test_nonfinite_numerical_samples_are_json_safe_not_zero(self):
        trace = SolveTrace()
        with span(trace, 'reluplex_mechanism', kind='numerical', action='check',
                  drift=float('inf'), residual=float('nan'), outcome='restore_failed'):
            pass
        report, _ = report_for(trace, 'UNKNOWN')
        json.dumps(report, allow_nan=False)
        record = report['numericalChecks'][0]
        self.assertIsNone(record['drift'])
        self.assertTrue(record['drift_nonfinite'])
        self.assertEqual(report['totals']['restore_failures'], 1)

    def test_tree_without_mechanisms_does_not_generate_mechanism_report(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory) / 'tree.html'
            write_relu_search_tree_html(SolveTrace(), tree, overall_status='RUNNING', mechanism_panels=[])
            self.assertTrue(tree.exists())
            self.assertFalse(mechanism_report_path(tree).exists())

    def test_report_is_separate_from_the_tree_and_dashboard_embeds_both(self):
        from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard

        rows, bounds, relus, *_ = _random_relu_problem(0)
        trace = SolveTrace()
        _, sat = reluplex(rows, dict(bounds), relus, max_recursion=100, trace=trace)
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory) / "tree.html"
            write_relu_search_tree_html(trace, tree, overall_status="SAT" if sat else "UNSAT")
            report = mechanism_report_path(tree)
            self.assertTrue(report.exists())
            tree_html = tree.read_text(encoding="utf-8")
            self.assertNotIn('id="mechanismRepair"', tree_html)
            self.assertNotIn('mechanism-badges', tree_html)
            self.assertNotIn('"mechanisms":', tree_html)
            report_html = report.read_text(encoding="utf-8")
            self.assertIn('id="mechanismFilter"', report_html)
            self.assertIn('id="numericalValue"', report_html)
            self.assertIn('id="underapproxValue"', report_html)
            self.assertNotIn('id="repairValue"', report_html)
            self.assertNotIn('id="cyclesValue"', report_html)
            dashboard = Path(directory) / "dashboard.html"
            UnifiedRealtimeDashboard(
                {"relu": Path(directory) / "relu.png"}, dashboard,
                embedded_pages={"relu-tree": tree, "solver-mechanisms": report},
            ).write()
            self.assertIn('id="solver-mechanisms-dashboard"', dashboard.read_text(encoding="utf-8"))

    def test_bounds_and_both_repair_rules_match_live_log(self):
        rows, bounds, relus, *_ = _random_relu_problem(0)
        baseline = reluplex(rows, dict(bounds), relus, max_recursion=100)
        trace = SolveTrace()
        with tempfile.TemporaryDirectory() as directory:
            logger = SplitEventLogger(Path(directory) / "relu.jsonl", reset_log=True)
            wall_clock = count(1_800_000_000_000_000_000, -1_000_000)
            with (
                patch("visualization.SolveTrace.monotonic", return_value=100.0),
                patch("visualization.RealtimeSplitVisualization.monotonic_ns", return_value=200_000_000_000),
                patch("visualization.RealtimeSplitVisualization.time_ns", side_effect=wall_clock),
            ):
                observed = reluplex(rows, dict(bounds), relus, max_recursion=100,
                                    trace=trace, split_logger=logger)
            records = [json.loads(line) for line in logger.branch_path.read_text(encoding="utf-8").splitlines()]
            live = trace_from_branch_log(branch_log_path(logger.path))
        active = []
        for record in records:
            if record["event"] == "branch_start":
                self.assertEqual(record.get("parent_branch_id"), active[-1] if active else None)
                active.append(record["branch_id"])
            elif record["event"] == "branch_end":
                self.assertEqual(record["branch_id"], active.pop())
            else:
                self.assertEqual(record["branch_id"], active[-1] if active else None)
        self.assertFalse(active)
        self.assertEqual(observed, baseline)
        report, nodes = report_for(trace, "UNSAT")
        live_report, live_nodes = report_for(live, "UNSAT")
        self.assertEqual(report, live_report)
        self.assertEqual(build_stable_records(trace), build_stable_records(live))
        fields = ("id", "branchId", "parentId", "splitNumber", "completionSequence", "result")
        self.assertEqual(
            [tuple(node[field] for field in fields) for node in nodes],
            [tuple(node[field] for field in fields) for node in live_nodes],
        )
        self.assertGreater(report["totals"]["bounds_changed"], 0)
        self.assertGreater(report["totals"]["update_f"], 0)
        self.assertGreater(report["totals"]["update_b"], 0)
        self.assertTrue(any(event.get("nodeId") is not None for event in report["events"]))
        self.assertFalse(any(node.get("mechanisms") for node in nodes))
        for event in trace.events:
            if event.component != "reluplex_mechanism" or event.meta.get("kind") != "repair":
                continue
            record = event.meta
            if record["rule"] == "update_f":
                self.assertAlmostEqual(record["after"][1], max(0, record["before"][0]))
            elif record["rule"] == "update_b":
                self.assertAlmostEqual(record["after"][0], record["before"][1])

    def test_backjump_links_to_actual_ancestor_and_distinguishes_disabled(self):
        rows, bounds, relus, *_ = _random_relu_problem(36)
        trace = SolveTrace()
        stats = {}
        # 이 문제(seed 36)는 자체 simplex 경로에서 백점프가 일어나는 예라 엔진을 고정한다
        _, sat = reluplex(rows, dict(bounds), relus, max_recursion=100, lp_engine="native",
                          backjump=True, trace=trace, stats=stats)
        self.assertFalse(sat)
        report, nodes = report_for(trace, "UNSAT")
        jumps = [event for event in report["events"] if event["kind"] == "backjump"]
        self.assertEqual(len(jumps), stats["BACKJUMPS"])
        by_id = {node["id"]: node for node in nodes}
        for jump in jumps:
            self.assertLess(jump["target_depth"], jump["from_depth"])
            self.assertEqual(by_id[jump["nodeId"]]["reason"], "BACKJUMP")
            if jump["targetNodeId"] is not None:
                self.assertEqual(by_id[jump["targetNodeId"]]["depth"], jump["target_depth"])
            self.assertTrue(set(jump["cause_depths"]) <= set(range(1, jump["from_depth"])))
        disabled = SolveTrace()
        reluplex(rows, dict(bounds), relus, max_recursion=100, lp_engine="native", backjump=False,
                 trace=disabled)
        off, _ = report_for(disabled, "UNSAT")
        self.assertFalse(off["settings"][0]["backjump"])
        self.assertEqual(off["totals"].get("backjump", 0), 0)

    def test_cycle_event_is_retained_in_both_log_formats_and_plotted(self):
        for compact in (False, True):
            with self.subTest(compact=compact), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "solver.jsonl"
                progress = SolverProgressLogger(path, compact_simplex_calls=compact)
                observed = []
                tableau = build_tableau([("s", {"x": 1.0})],
                                        {"s": (1.0, 2.0), "x": (0.0, 2.0)})
                # Force the repeated-state signal to exercise the rare cycle path.
                with patch("Simplex._var_hash", return_value=0), patch("Simplex._value_hash", return_value=0):
                    _, sat = simplex(tableau, progress=progress,
                                     mechanism_callback=lambda **data: observed.append(data))
                progress.close()
                self.assertTrue(sat)
                self.assertEqual(observed[0]["cycles"], 1)
                feedback = build_solver_feedback(read_solver_progress(path))
                call = feedback["simplex_calls"][0]
                self.assertEqual(call["cycles"], 1)
                self.assertEqual(call["cycle_events"][0]["random_steps"], 100)
                fig, axes = draw_simplex_panel(feedback)
                try:
                    fig.canvas.draw()
                    for axis in axes[0]:
                        self.assertIn("Cycle detected / random pivot", axis.get_legend_handles_labels()[1])
                        bounds = axis.get_legend().get_window_extent(fig.canvas.get_renderer())
                        self.assertGreater(bounds.x0, axis.get_window_extent().x1)
                        self.assertLess(bounds.x1, fig.bbox.x1)
                finally:
                    plt.close(fig)

    def test_legacy_trace_has_no_invented_zero_measurements(self):
        report, _ = report_for(SolveTrace(), "UNKNOWN")
        self.assertFalse(report["available"])


if __name__ == "__main__":
    unittest.main()
