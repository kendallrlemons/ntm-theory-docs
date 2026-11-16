import unittest

import numpy as np

from evaluate_fah_allocation import allocate, evaluate_policy, floor_free_shape


class AllocationTests(unittest.TestCase):
    def test_unconstrained_spread_proportional_solution(self):
        n = allocate([1., 2., 3.], 60., 1.)
        np.testing.assert_allclose(n, [10., 20., 30.])
        # A small feasible perturbation increases the true quadratic risk.
        s = np.array([1., 2., 3.])
        perturbed = n + np.array([1., -1., 0.])
        self.assertLess(np.sum(s*s/n), np.sum(s*s/perturbed))

    def test_active_floor_water_filling(self):
        n = allocate([.01, 1., 3.], 10., 2.)
        np.testing.assert_allclose(n, [2., 2., 6.])
        self.assertAlmostEqual(n.sum(), 10.)
        self.assertTrue((n >= 2.).all())

    def test_scale_invariance_including_floor(self):
        weights = np.array([.01, .1, 1., 10.])
        np.testing.assert_allclose(allocate(weights, 20, 2),
                                   allocate(weights * 1e6, 20, 2))
        self.assertAlmostEqual(floor_free_shape([1, 2, 3, 4], weights),
                               floor_free_shape([1, 2, 3, 4], weights * 17))

    def test_budget_gate_and_exact_minimum(self):
        self.assertIsNone(allocate([1., 2.], 3., 2.))
        np.testing.assert_array_equal(allocate([1., 2.], 4., 2.), [2., 2.])
        self.assertEqual(evaluate_policy([1, 2], [1, 2], 20, 18, 2, 1)["status"], "infeasible")

    def test_pilot_charge_and_production_cost_conserved(self):
        result = evaluate_policy([1, 2, 3], [1, 2, 3], 100., 40., 2., 1.)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["production_effort"], 30.)
        self.assertAlmostEqual(result["total_cost"], 100.)
        self.assertAlmostEqual(result["mean_variance_proxy"], np.mean([1/5, 4/10, 9/15]))

    def test_uniform_pilot_overhead_ratio(self):
        s = np.array([1., 2., 4.])
        free = evaluate_policy(s, np.ones(3), 120., 0., 1., 1.)
        paid = evaluate_policy(s, np.ones(3), 120., 40., 1., 1.)
        self.assertAlmostEqual(paid["mean_variance_proxy"] / free["mean_variance_proxy"], 1.5)

    def test_analytic_floor_free_ratio_and_break_even(self):
        s, w = np.array([1., 2., 4.]), np.array([1., 2., 3.])
        factor = floor_free_shape(s, w)
        self.assertLess(factor, 1.)
        cost = 30.
        budget = cost / (1 - factor)
        baseline = evaluate_policy(s, np.ones(3), budget, 0., 1., 1.)
        candidate = evaluate_policy(s, w, budget, cost, 1., 1.)
        self.assertAlmostEqual(candidate["mean_variance_proxy"], baseline["mean_variance_proxy"])

    def test_oracle_is_lower_bound_with_floors(self):
        s = np.array([.1, 2., 10., 30.])
        for budget in (4., 10., 100.):
            oracle = evaluate_policy(s, s, budget, 0., 1., 1.)
            for weights in (np.ones(4), s[::-1], np.array([5., .01, 3., 2.])):
                result = evaluate_policy(s, weights, budget, 0., 1., 1.)
                self.assertGreaterEqual(result["mean_variance_proxy"] + 1e-12,
                                        oracle["mean_variance_proxy"])

    def test_invalid_weights_rejected(self):
        for weights in ([0., 1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.], []):
            with self.assertRaises(ValueError):
                allocate(weights, 10., 1.)


if __name__ == "__main__":
    unittest.main()
