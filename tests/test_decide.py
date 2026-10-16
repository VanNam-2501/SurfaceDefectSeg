"""Unit tests for Industrial Decision Engine (src/straw_defect_cls/decide.py).

Verifies 100% branch coverage using Python's standard unittest:
- PASS branch ('pass')
- Stage 1 Borderline branch ('borderline_s1')
- Stage 2 Uncertain Confidence branch ('uncertain_s2')
- Stage 2 Out-Of-Distribution branch ('ood_s2')
- Stage 2 Low Margin branch ('low_margin_s2')
- Valid Confirmed Defect branch ('defect')
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.straw_defect_cls.decide import calculate_energy, decide, softmax


class TestDecisionEngine(unittest.TestCase):
    """Test suite covering every decision branch of decide()."""

    def setUp(self) -> None:
        self.base_config = {
            "tau_pass": 0.80,
            "tau_defect": 0.65,
            "tau2": 0.50,
            "margin2": 0.15,
            "e_thr": -2.0,  # Energy >= -2.0 considered anomalous/OOD
            "temperature_stage1": 1.0,
            "temperature_stage2": 1.0,
            "pi_deploy_defect": 0.05,
            "pi_train_defect": 0.05,  # 1:1 ratio for transparent direct test assertions
        }

    def test_branch_pass(self):
        """Test PASS when Stage 1 confidently identifies Good product."""
        logits1 = np.array([5.0, -5.0])
        logits2 = None

        decision, defect_cls, reason = decide(logits1, logits2, self.base_config)

        self.assertEqual(decision, "PASS")
        self.assertIsNone(defect_cls)
        self.assertEqual(reason, "pass")

    def test_branch_borderline_s1(self):
        """Test REVIEW when Stage 1 is uncertain between Good and Defect."""
        logits1 = np.array([0.1, 0.1])
        logits2 = np.array([2.0, 0.0, 0.0, 0.0])

        decision, defect_cls, reason = decide(logits1, logits2, self.base_config)

        self.assertEqual(decision, "REVIEW")
        self.assertIsNone(defect_cls)
        self.assertEqual(reason, "borderline_s1")

    def test_branch_uncertain_s2(self):
        """Test REVIEW when Stage 1 flags defect but Stage 2 top-1 probability < tau2."""
        logits1 = np.array([-5.0, 5.0])
        # High magnitude so energy is low, but flat so top-1 is only ~0.25 < tau2=0.50
        logits2 = np.array([4.0, 3.99, 3.98, 3.97])

        decision, defect_cls, reason = decide(logits1, logits2, self.base_config)

        self.assertEqual(decision, "REVIEW")
        self.assertIsNone(defect_cls)
        self.assertEqual(reason, "uncertain_s2")

    def test_branch_ood_s2(self):
        """Test REVIEW when Stage 2 energy score indicates out-of-distribution sample."""
        logits1 = np.array([-5.0, 5.0])
        # Very small/abnormal logits produce high (anomalous) energy
        logits2 = np.array([-10.0, -10.0, -10.0, -10.0])

        energy = calculate_energy(logits2, temperature=1.0)
        self.assertGreater(energy, self.base_config["e_thr"])

        decision, defect_cls, reason = decide(logits1, logits2, self.base_config)

        self.assertEqual(decision, "REVIEW")
        self.assertIsNone(defect_cls)
        self.assertEqual(reason, "ood_s2")

    def test_branch_low_margin_s2(self):
        """Test REVIEW when Stage 2 top-1 is high but margin between top 2 classes < margin2."""
        logits1 = np.array([-5.0, 5.0])
        logits2 = np.array([5.0, 4.95, -2.0, -2.0])

        decision, defect_cls, reason = decide(logits1, logits2, self.base_config)

        self.assertEqual(decision, "REVIEW")
        self.assertIsNone(defect_cls)
        self.assertEqual(reason, "low_margin_s2")

    def test_branch_valid_defect(self):
        """Test DEFECT when sample passes Stage 1 defect threshold and all 3 Stage 2 guards."""
        logits1 = np.array([-5.0, 5.0])
        logits2 = np.array([6.0, 0.0, 0.0, 0.0])

        decision, defect_cls, reason = decide(logits1, logits2, self.base_config)

        self.assertEqual(decision, "DEFECT")
        self.assertEqual(defect_cls, 0)  # Bent straw
        self.assertEqual(reason, "defect")

    def test_decide_pure_numpy_types(self):
        """Verify decide works with plain python lists and 2D arrays without torch."""
        cfg = {
            "tau_pass": 0.80,
            "tau_defect": 0.65,
            "tau2": 0.50,
            "margin2": 0.15,
            "e_thr": -2.0,
        }
        # Python list input
        d, c, r = decide([10.0, -10.0], None, cfg)
        self.assertEqual(d, "PASS")

        # 2D array input
        d2, c2, r2 = decide(np.array([[10.0, -10.0]]), None, cfg)
        self.assertEqual(d2, "PASS")


if __name__ == "__main__":
    unittest.main()
