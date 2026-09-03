from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND_APP = ROOT / "apps" / "web_demo" / "backend" / "app.py"
_spec = importlib.util.spec_from_file_location("web_demo_backend_app", BACKEND_APP)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load web demo backend: {BACKEND_APP}")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


class WebPolicyTests(unittest.TestCase):
    def test_single_and_pair_use_their_matching_policy_family(self) -> None:
        single = app.current_policy(("unet",))
        pair = app.current_policy(("unet", "vmamba"))
        self.assertTrue(app.is_adaptive_policy(single))
        self.assertFalse(app.is_adaptive_policy(pair))
        self.assertTrue(app.policy_matches_selection(single, ("unet",)))
        self.assertTrue(app.policy_matches_selection(pair, ("unet", "vmamba")))

    def test_adaptive_policy_returns_component_mask(self) -> None:
        image = np.full((16, 16, 3), 200, dtype=np.uint8)
        probability = np.zeros((16, 16), dtype=np.float32)
        probability[4:12, 4:12] = 1.0
        result = app.apply_adaptive_policy(
            image, probability, app.current_policy(("unet",)), "unet"
        )
        self.assertEqual(result["decision"], "defect")
        self.assertEqual(result["mask_pixels"], 64)
        self.assertEqual(result["required_votes"], 1)


if __name__ == "__main__":
    unittest.main()
