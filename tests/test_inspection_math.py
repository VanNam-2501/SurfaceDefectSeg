"""Algebra, gradients and statistical calibration tests for generic inspection."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inspection_research.probability import (joint_log_probabilities, hierarchical_nll,
                                                HierarchicalHead, entropic_group_risk)
from src.inspection_research.decision import cost_matrix, bayes_actions, select_review_penalty
from src.inspection_research.risk_control import (minimum_defect_units, calibrate_pass_gate,
                                                 calibrate_known_domains, one_sided_binomial_upper)


class ProbabilityTests(unittest.TestCase):
    def test_normalization_for_different_products_and_extreme_logits(self):
        for types in [1, 4, 5, 11]:
            binary = torch.tensor([-1000., -2, 0, 2, 1000.], dtype=torch.float64)
            conditional = torch.arange(5 * types, dtype=torch.float64).reshape(5, types)
            log_q = joint_log_probabilities(binary, conditional)
            self.assertTrue(torch.isfinite(log_q).all())
            self.assertTrue(torch.allclose(log_q.exp().sum(1), torch.ones(5, dtype=torch.float64)))

    def test_hierarchical_nll_equals_binary_plus_conditional(self):
        binary = torch.tensor([1., -1, .5, 2.], dtype=torch.float64)
        types = torch.tensor([[1., 2, 3], [-1., 0, 2], [3., 2, 1], [0., 2, 0]], dtype=torch.float64)
        y = torch.tensor([0, 1, 2, 3])
        expected = F.binary_cross_entropy_with_logits(binary, (y > 0).double(), reduction="none")
        expected[y > 0] += F.cross_entropy(types[y > 0], y[y > 0] - 1, reduction="none")
        self.assertTrue(torch.allclose(hierarchical_nll(binary, types, y, "none"), expected))

    def test_full_softmax_has_exact_hierarchical_factorization(self):
        full = torch.tensor([[1., 0, 3, -2, 1], [0., 5, -1, 3, 2]], dtype=torch.float64)
        binary = torch.logsumexp(full[:, 1:], dim=1) - full[:, 0]
        hierarchy = joint_log_probabilities(binary, full[:, 1:])
        self.assertTrue(torch.allclose(hierarchy, F.log_softmax(full, dim=1), atol=1e-12))

    def test_binary_product_reduces_to_bce(self):
        binary = torch.tensor([-3., 0, 3.], requires_grad=True)
        types = torch.randn(3, 1, requires_grad=True)
        labels = torch.tensor([0, 1, 1])
        actual = hierarchical_nll(binary, types, labels)
        expected = F.binary_cross_entropy_with_logits(binary, labels.float())
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.equal(types.grad, torch.zeros_like(types)))

    def test_shared_head_has_no_product_specific_class_count(self):
        for num_defects in [1, 4, 5]:
            head = HierarchicalHead(16, num_defects)
            binary, types = head(torch.zeros(3, 16))
            self.assertEqual(tuple(binary.shape), (3,))
            self.assertEqual(tuple(types.shape), (3, num_defects))

    def test_group_objective_matches_variational_dual_and_gradient(self):
        losses = torch.tensor([.1, .2, 1.2, 1.0], dtype=torch.float64, requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        prior = torch.tensor([.4, .6], dtype=torch.float64)
        tau = .3
        objective, risks, weights = entropic_group_risk(losses, groups, 2, tau, prior)
        dual = (weights * risks).sum() - tau * (weights * (weights.log() - prior.log())).sum()
        self.assertTrue(torch.allclose(objective, dual))
        objective.backward()
        self.assertTrue(torch.allclose(losses.grad, weights.detach()[groups] / 2))

    def test_group_temperature_limits_and_bounds(self):
        losses = torch.tensor([.1, .3, 1., 2.], dtype=torch.float64)
        groups = torch.tensor([0, 0, 1, 1])
        small, risks, _ = entropic_group_risk(losses, groups, 2, 1e-5)
        large, _, _ = entropic_group_risk(losses, groups, 2, 1e6)
        self.assertAlmostEqual(float(small), float(risks.max()), places=4)
        self.assertAlmostEqual(float(large), float(risks.mean()), places=5)

    def test_missing_group_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            entropic_group_risk(torch.tensor([.1, .2]), torch.tensor([0, 0]), 2)


class DecisionTests(unittest.TestCase):
    def test_binary_bayes_rule_matches_cost_threshold(self):
        p_defect = np.array([.01, .04, .2, .9])
        p = np.column_stack([1 - p_defect, p_defect])
        matrix, actions = cost_matrix(1, false_pass=100, false_reject=5, review=1000)
        chosen, _ = bayes_actions(p, matrix, actions)
        self.assertTrue(np.array_equal(chosen, (p_defect > 5 / 105).astype(int)))

    def test_review_band_has_cost_interpretation(self):
        s = np.array([.001, .01, .03, .5, .7, .9])
        matrix, actions = cost_matrix(1, 100, 5, 2)
        chosen, _ = bayes_actions(np.column_stack([1 - s, s]), matrix, actions)
        self.assertTrue(np.array_equal(chosen, np.array([0, 0, -1, -1, 1, 1])))

    def test_review_fraction_is_nonincreasing_in_penalty(self):
        s = np.linspace(.001, .999, 500)
        p = np.column_stack([1 - s, s])
        matrix, actions = cost_matrix(1, 100, 5, 1)
        rates = [np.mean(bayes_actions(p, matrix, actions, penalty)[0] == -1) for penalty in [0, .5, 1, 2, 10]]
        self.assertTrue(np.all(np.diff(rates) <= 0))

    def test_same_policy_interface_handles_different_class_counts(self):
        for k in [1, 4, 5]:
            p = np.eye(k + 1)
            matrix, actions = cost_matrix(k, 100, 5, 2, wrong_type=5)
            self.assertTrue(np.array_equal(bayes_actions(p, matrix, actions)[0], np.arange(k + 1)))

    def test_pass_guard_only_removes_pass_and_uses_strict_boundary(self):
        p = np.array([[.9, .1], [.875, .125], [.5, .5]])
        matrix, actions = cost_matrix(1, 1, 100, 1000)
        before, _ = bayes_actions(p, matrix, actions)
        after, _ = bayes_actions(p, matrix, actions, pass_threshold=.125)
        self.assertTrue(np.all((after == 0) <= (before == 0)))
        self.assertTrue(np.array_equal(after, [0, -1, -1]))

    def test_development_budget_is_not_claimed_as_population_guarantee(self):
        p = np.array([[.99, .01], [.5, .5], [.1, .9]])
        matrix, actions = cost_matrix(1, 100, 5, 2)
        result = select_review_penalty(p, matrix, actions, 0, [0, 10, 100])
        self.assertEqual(result["status"], "feasible_on_development")
        self.assertEqual(result["selected"]["development_review_rate"], 0)

    def test_invalid_probability_simplex_is_rejected(self):
        matrix, actions = cost_matrix(1, 100, 5, 2)
        with self.assertRaises(ValueError):
            bayes_actions([[.9, .9]], matrix, actions)


class RiskControlTests(unittest.TestCase):
    def test_required_sample_sizes(self):
        self.assertEqual(minimum_defect_units(.01, .05), 299)
        self.assertEqual(minimum_defect_units(.05, .05), 59)
        self.assertGreater(minimum_defect_units(.01, .05, comparisons=8), 299)

    def test_zero_errors_still_has_a_positive_upper_bound(self):
        self.assertGreater(one_sided_binomial_upper(0, 240), .01)
        self.assertLessEqual(one_sided_binomial_upper(0, 299), .01)
        self.assertEqual(one_sided_binomial_upper(0, 0), 1)

    def test_insufficient_calibration_does_not_invent_a_threshold(self):
        result = calibrate_pass_gate(np.linspace(.2, .9, 20), .01, .05)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["threshold"])

    def test_selected_order_statistic_respects_exact_tail(self):
        n, alpha, delta = 400, .05, .05
        result = calibrate_pass_gate(np.linspace(0, 1, n), alpha, delta)
        rank = result["rank"]
        self.assertLessEqual(beta.sf(alpha, rank, n + 1 - rank), delta)
        self.assertGreater(beta.sf(alpha, rank + 1, n - rank), delta)

    def test_calibration_unit_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            calibrate_pass_gate(np.linspace(0, 1, 100), unit_ids=["same"] * 100)

    def test_tied_scores_require_strict_pass(self):
        result = calibrate_pass_gate(np.full(100, .5), .05, .05)
        self.assertEqual(result["comparison"], "strict_less_than")
        self.assertEqual(result["threshold"], .5)
        # A degenerate future distribution at .5 then has zero, rather than 100%, PASS.
        self.assertFalse(.5 < result["threshold"])

    def test_observed_domain_gate_is_minimum_not_average(self):
        values = np.concatenate([np.linspace(.1, .4, 100), np.linspace(.6, .9, 100)])
        domains = np.repeat(["a", "b"], 100)
        result = calibrate_known_domains(values, domains, .1, .05)
        self.assertEqual(result["status"], "calibrated_known_domains_under_assumptions")
        thresholds = [r["threshold"] for r in result["domain_results"].values()]
        self.assertEqual(result["threshold"], min(thresholds))
        self.assertIn("no guarantee for an unseen", result["scope"])

    def test_no_small_domain_is_silently_discarded(self):
        result = calibrate_known_domains(np.linspace(0, 1, 120), ["a"] * 100 + ["b"] * 20, .05, .05)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["threshold"])

    def test_monte_carlo_uniform_defects_matches_violation_control(self):
        # Independent numerical check: under Uniform(0,1), true FNR equals t.
        rng = np.random.default_rng(18)
        n, alpha, delta = 100, .1, .05
        result = calibrate_pass_gate(np.linspace(0, 1, n), alpha, delta)
        samples = rng.uniform(size=(3000, n))
        thresholds = np.sort(samples, axis=1)[:, result["rank"] - 1]
        violation_rate = float(np.mean(thresholds > alpha))
        self.assertLess(violation_rate, delta + .02)


if __name__ == "__main__":
    unittest.main()
