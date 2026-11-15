"""Regression checks for the original node-local repair attempt limits."""
import unittest

from Automation.SolverStatus import SolverLimitReached
from Reluplex import reluplex
from visualization.SolveTrace import SolveTrace


class RepairBudgetTests(unittest.TestCase):
    @staticmethod
    def independent_relus(count=12):
        pairs = [(f'z1_{index:02}', f'h1_{index:02}') for index in range(count)]
        bounds = {var: bound for z, h in pairs
                  for var, bound in ((z, (-1., 1.)), (h, (1., 1.)))}
        return pairs, bounds

    def test_successful_repairs_also_consume_node_attempts(self):
        pairs, bounds = self.independent_relus()
        for limit in (0, 2, 10):
            with self.subTest(limit=limit):
                stats, trace = {}, SolveTrace()
                with self.assertRaises(SolverLimitReached):
                    reluplex([], bounds, pairs, derive_bound=False, derive_in_search=False,
                             pivot_tighten=False, max_recursion=0, report_unknown=True,
                             local_repair_max_iter=limit, stats=stats, trace=trace)
                repairs = [event for event in trace.events if event.component == 'reluplex_repair']
                self.assertEqual(len(repairs), limit)
                self.assertEqual(stats.get('REPAIR_STEPS', 0), limit)

    def test_sufficient_original_attempt_budget_solves_without_splitting(self):
        pairs, bounds = self.independent_relus()
        stats = {}
        model, sat = reluplex([], bounds, pairs, derive_bound=False, derive_in_search=False,
                              pivot_tighten=False, max_recursion=0, report_unknown=True,
                              local_repair_max_iter=12, stats=stats)
        self.assertTrue(sat)
        self.assertEqual(stats['REPAIR_STEPS'], 12)
        self.assertTrue(all(model[z] == model[h] == 1. for z, h in pairs))


if __name__ == '__main__':
    unittest.main()
