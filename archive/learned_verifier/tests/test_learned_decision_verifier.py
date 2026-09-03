import unittest

import numpy as np

from learned_decision_verifier import (
    choose_binary_threshold,
    choose_triage_thresholds,
    choose_two_specialist_hybrid,
)


class LearnedDecisionVerifierTests(unittest.TestCase):
    def test_binary_threshold_respects_validation_fnr(self) -> None:
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        threshold, metrics = choose_binary_threshold(scores, labels, max_fnr=0.25)
        self.assertGreater(threshold, 0.6)
        self.assertLessEqual(metrics["fnr"], 0.25)
        self.assertEqual(metrics["fpr"], 0.0)

    def test_triage_separates_pass_review_and_defect(self) -> None:
        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.7, 0.4, 0.6, 0.3, 0.1])
        low, high, metrics = choose_triage_thresholds(
            scores, labels, max_alert_fnr=0.34, max_defect_fpr=0.0
        )
        self.assertLessEqual(low, high)
        self.assertLessEqual(metrics["alert_fnr"], 0.34)
        self.assertEqual(metrics["auto_defect_fpr"], 0.0)

    def test_hybrid_never_drops_trusted_consensus(self) -> None:
        labels = np.array([1, 1, 0, 0])
        trusted = np.array([True, False, True, False])
        first = np.array([0.0, 0.9, 0.0, 0.1])
        second = np.array([0.0, 0.2, 0.0, 0.1])
        _, _, metrics = choose_two_specialist_hybrid(
            first, second, labels, trusted, max_fnr=0.0
        )
        self.assertEqual(metrics["fn"], 0)
        # Rescue can add candidates, but it must never remove trusted consensus.
        self.assertGreaterEqual(metrics["fp"], 1)


if __name__ == "__main__":
    unittest.main()
