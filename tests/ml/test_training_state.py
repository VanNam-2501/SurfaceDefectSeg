from __future__ import annotations

import math
import unittest

from training_state import recover_best_epoch


class RecoverBestEpochTests(unittest.TestCase):
    def test_new_checkpoint_value_takes_precedence(self) -> None:
        history = [{"epoch": 2, "val_positive_dice_0.5": 0.8}]
        self.assertEqual(recover_best_epoch({"best_epoch": 7}, history), 7)

    def test_old_checkpoint_recovers_best_finite_history_row(self) -> None:
        history = [
            {"epoch": 1, "val_positive_dice_0.5": 0.2},
            {"epoch": 2, "val_positive_dice_0.5": 0.8},
            {"epoch": 3, "val_positive_dice_0.5": 0.4},
            {"epoch": 4, "val_positive_dice_0.5": math.nan},
        ]
        self.assertEqual(recover_best_epoch({}, history), 2)

    def test_old_checkpoint_uses_its_saved_best_metric(self) -> None:
        history = [
            {"epoch": 2, "val_positive_dice_0.5": 0.8},
            {"epoch": 3, "val_positive_dice_0.5": 0.80005},
        ]
        self.assertEqual(recover_best_epoch({"best_metric": 0.8}, history), 2)

    def test_missing_validation_history_returns_zero(self) -> None:
        self.assertEqual(recover_best_epoch({}, [{"epoch": 1}]), 0)


if __name__ == "__main__":
    unittest.main()
