"""Correctness checks for successful, failed and recurring local repairs."""
import unittest
from unittest.mock import patch

from Automation.SolverStatus import SolverLimitReached
from Reluplex import _RepairBudget, reluplex
from Simplex import simplex as real_simplex
from visualization.SolveTrace import SolveTrace


class RepairBudgetTests(unittest.TestCase):
    def test_successes_do_not_consume_failure_budget(self):
        budget = _RepairBudget(10, 100)
        for index in range(20):
            pair = (f'z{index}', f'h{index}')
            self.assertEqual(budget.observe(pair, [], [pair]), (True, False, 0))
        self.assertEqual(budget.failures, 0)
        self.assertEqual(budget.by_pair, {})

    def test_unresolved_repair_is_charged_once(self):
        pair = ('z', 'h')
        budget = _RepairBudget(10, 100)
        self.assertEqual(budget.observe(pair, [pair], [pair]), (False, True, 0))
        self.assertEqual(budget.failures, 1)
        self.assertEqual(budget.by_pair[pair], 1)

    def test_cross_neuron_oscillation_consumes_budget(self):
        a, b = ('za', 'ha'), ('zb', 'hb')
        budget = _RepairBudget(10, 100)
        self.assertEqual(budget.observe(a, [b], [a, b]), (True, False, 0))
        self.assertEqual(budget.observe(b, [a], [b]), (True, False, 1))
        self.assertEqual(budget.observe(a, [b], [a]), (True, False, 1))
        self.assertEqual(budget.failures, 2)
        self.assertEqual(budget.by_pair, {a: 1, b: 1})

    def test_reappearance_is_not_recharged_while_still_violated(self):
        a, b, c = ('za', 'ha'), ('zb', 'hb'), ('zc', 'hc')
        budget = _RepairBudget(10, 100)
        budget.observe(a, [b, c], [a, b, c])
        budget.observe(b, [a, c], [b, c])
        self.assertEqual(budget.observe(c, [a], [a, c]), (True, False, 0))
        self.assertEqual(budget.by_pair[a], 1)

    def test_five_reviolations_with_or_without_current_failure_charge_once(self):
        for current_fails in (False, True):
            with self.subTest(current_fails=current_fails):
                budget = _RepairBudget(10, 100)
                pairs = [(f'z{index}', f'h{index}') for index in range(60, 71)]
                for pair in pairs[:10]:
                    budget.observe(pair, [], [pair])
                remaining = pairs[5:10] + ([pairs[10]] if current_fails else [])
                budget.observe(pairs[10], remaining, [pairs[10]])
                self.assertEqual(budget.failures, 1)
                self.assertTrue(all(budget.by_pair.get(pair, 0)==0 for pair in pairs[:5]))
                self.assertTrue(all(budget.by_pair[pair]==1 for pair in pairs[5:10]))
                self.assertEqual(budget.by_pair.get(pairs[10], 0), int(current_fails))

    def test_new_violation_of_previously_unrepaired_neuron_is_charged(self):
        a, b = ('za', 'ha'), ('zb', 'hb')
        budget = _RepairBudget(10, 100)
        self.assertEqual(budget.observe(a, [b], [a]), (True, False, 1))
        self.assertEqual(budget.failures, 1)
        self.assertEqual(budget.by_pair, {b: 1})

    def test_existing_other_violations_are_not_charged(self):
        a, b = ('za', 'ha'), ('zb', 'hb')
        budget = _RepairBudget(10, 100)
        budget.observe(a, [b], [a, b])
        self.assertEqual(budget.failures, 0)
        self.assertEqual(budget.by_pair, {})

    def test_success_does_not_erase_prior_failures(self):
        a, b = ('za', 'ha'), ('zb', 'hb')
        budget = _RepairBudget(10, 100)
        budget.observe(a, [a], [a])
        budget.observe(a, [], [a])
        budget.observe(b, [a], [b])
        self.assertEqual(budget.by_pair[a], 2)
        self.assertEqual(_RepairBudget(10, 100).by_pair, {})

    @staticmethod
    def independent_relus(count=12):
        pairs = [(f'z1_{index:02}', f'h1_{index:02}') for index in range(count)]
        bounds = {var: bound for z, h in pairs for var, bound in ((z, (-1., 1.)), (h, (1., 1.)))}
        return pairs, bounds

    def test_more_than_ten_distinct_successes_are_repaired_without_splitting(self):
        pairs, bounds = self.independent_relus()
        stats, trace = {}, SolveTrace()
        model, sat = reluplex([], bounds, pairs, derive_bound=False, derive_in_search=False,
                              pivot_tighten=False, max_recursion=0, report_unknown=True,
                              local_repair_max_iter=10, stats=stats, trace=trace)
        self.assertTrue(sat)
        self.assertEqual(stats['REPAIR_ATTEMPTS'], 12)
        self.assertEqual(stats['REPAIR_SUCCESSES'], 12)
        self.assertEqual(stats['REPAIR_FAILURE_BUDGET_USED'], 0)
        self.assertFalse(any(event.component=='reluplex_split' for event in trace.events))
        self.assertTrue(all(model[z]==model[h]==1. for z, h in pairs))

    def test_explicit_attempt_limit_is_still_enforced(self):
        pairs, bounds = self.independent_relus()
        stats = {}
        with self.assertRaises(SolverLimitReached):
            reluplex([], bounds, pairs, derive_bound=False, pivot_tighten=False,
                     max_recursion=0, report_unknown=True, local_repair_max_attempts=2, stats=stats)
        self.assertEqual(stats['REPAIR_ATTEMPTS'], 2)
        self.assertEqual(stats['REPAIR_FAILURE_BUDGET_USED'], 0)
        self.assertEqual(stats['REPAIR_ATTEMPT_LIMIT_REACHED'], 1)

    def test_disabled_repairs_remain_disabled(self):
        pairs, bounds = self.independent_relus()
        stats = {}
        with self.assertRaises(SolverLimitReached):
            reluplex([], bounds, pairs, derive_bound=False, max_recursion=0,
                     report_unknown=True, local_repair_max_iter=0, stats=stats)
        self.assertEqual(stats.get('REPAIR_ATTEMPTS', 0), 0)

    def test_failure_is_checked_after_simplex_not_just_after_update(self):
        pairs, bounds = self.independent_relus(1)
        stats = {}

        def undo_repair(tableau, **options):
            if options.get('progress_context', {}).get('origin')=='relu_repair':
                tableau.assign[pairs[0][0]] = -1.
                return dict(tableau.assign), True
            return real_simplex(tableau, **options)

        with patch('Reluplex.simplex', side_effect=undo_repair), self.assertRaises(SolverLimitReached):
            reluplex([], bounds, pairs, derive_bound=False, pivot_tighten=False,
                     max_recursion=0, report_unknown=True, local_repair_max_iter=2,
                     branch_tau=100, stats=stats)
        self.assertEqual(stats['REPAIR_FAILURES'], 2)
        self.assertEqual(stats['REPAIR_SUCCESSES'], 0)
        self.assertEqual(stats['REPAIR_FAILURE_LIMIT_REACHED'], 1)

    def test_simplex_ripple_charges_one_attempt_in_stats_and_trace(self):
        for current_fails in (False, True):
            with self.subTest(current_fails=current_fails):
                pairs, bounds = self.independent_relus(6)
                stats, trace = {}, SolveTrace()
                attempts = 0

                def reintroduce_violations(tableau, **options):
                    nonlocal attempts
                    model, sat = real_simplex(tableau, **options)
                    if options.get('progress_context', {}).get('origin') == 'relu_repair':
                        attempts += 1
                        if attempts == 6:
                            for z, h in pairs[:5 + int(current_fails)]:
                                tableau.assign[z] = -1.
                            model = dict(tableau.assign)
                    return model, sat

                with patch('Reluplex.simplex', side_effect=reintroduce_violations), \
                     self.assertRaises(SolverLimitReached):
                    reluplex([], bounds, pairs, derive_bound=False, derive_in_search=False,
                             pivot_tighten=False, max_recursion=0, report_unknown=True,
                             local_repair_max_iter=1, stats=stats, trace=trace)
                self.assertEqual(stats['REPAIR_ATTEMPTS'], 6)
                self.assertEqual(stats['REPAIR_REVIOLATIONS'], 5)
                self.assertEqual(stats['REPAIR_FAILURE_BUDGET_USED'], 1)
                self.assertEqual(stats['REPAIR_FAILURE_LIMIT_REACHED'], 1)
                repairs = [event for event in trace.events if event.component == 'reluplex_repair']
                self.assertEqual(repairs[-1].meta['budget_used'], 1)
                self.assertEqual(repairs[-1].meta['failures'], 1)


if __name__=='__main__':
    unittest.main()
