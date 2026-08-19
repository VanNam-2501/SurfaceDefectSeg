import unittest

import numpy as np

from decision_policy import apply_decision_policy, border_connected_dark_roi


def policy(models: list[str], votes: int = 1) -> dict:
    return {
        "schema_version": 1,
        "models": {
            name: {
                "pixel_threshold": 0.5,
                "min_component_area_px": 4,
                "tiny_min_area_px": 1,
                "tiny_high_threshold": 0.9,
            }
            for name in models
        },
        "roi": {"border_dark_threshold": -1},
        "ensemble": {
            "defect_votes": votes,
            "agreement_dilation_px": 0,
            "min_consensus_area_px": 1,
        },
    }


class DecisionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((32, 32, 3), 80, dtype=np.uint8)

    def test_border_connected_dark_roi_preserves_internal_dark_region(self) -> None:
        image = np.full((12, 12, 3), 90, dtype=np.uint8)
        image[0, :, :] = 0
        image[:, 0, :] = 0
        image[5:7, 5:7, :] = 0
        roi = border_connected_dark_roi(image, threshold=5)
        self.assertFalse(bool(roi[0, 5]))
        self.assertTrue(bool(roi[5, 5]))
        self.assertTrue(bool(roi[4, 4]))

    def test_single_model_pass_review_defect(self) -> None:
        zero = np.zeros((32, 32), dtype=np.float32)
        passed = apply_decision_policy(self.image, {"unet": zero}, policy(["unet"]))
        self.assertEqual(passed["decision"], "pass")

        tiny = zero.copy()
        tiny[10, 10] = 0.95
        reviewed = apply_decision_policy(self.image, {"unet": tiny}, policy(["unet"]))
        self.assertEqual(reviewed["decision"], "review")

        large = zero.copy()
        large[10:13, 10:13] = 0.8
        defect = apply_decision_policy(self.image, {"unet": large}, policy(["unet"]))
        self.assertEqual(defect["decision"], "defect")

    def test_two_models_require_spatial_agreement(self) -> None:
        first = np.zeros((32, 32), dtype=np.float32)
        second = np.zeros((32, 32), dtype=np.float32)
        first[5:8, 5:8] = 0.8
        second[5:8, 5:8] = 0.85
        agreed = apply_decision_policy(
            self.image,
            {"unet": first, "segformer": second},
            policy(["unet", "segformer"], votes=2),
        )
        self.assertEqual(agreed["decision"], "defect")
        self.assertEqual(agreed["max_spatial_votes"], 2)

        second[:] = 0
        second[20:23, 20:23] = 0.85
        disagreed = apply_decision_policy(
            self.image,
            {"unet": first, "segformer": second},
            policy(["unet", "segformer"], votes=2),
        )
        self.assertEqual(disagreed["decision"], "review")
        self.assertEqual(disagreed["max_spatial_votes"], 1)


if __name__ == "__main__":
    unittest.main()

