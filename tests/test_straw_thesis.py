"""Scientific protocol regression tests; no pretrained downloads or real data."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_TEMP = ROOT / "artifacts" / "verification" / "straw_tests"
TEST_TEMP.mkdir(parents=True, exist_ok=True)

from src.straw_defect_cls.thesis_protocol import export_protocol, load_protocol, split_groups, verify_assignments, digest_object
from src.straw_defect_cls.thesis_evaluation import (choose_policy, decision_metrics, fit_temperature,
                                                   grouped_bootstrap, predict, risk_coverage, wilson)
from src.straw_defect_cls.thesis_runner import jobs, grouped_jobs, verify_freeze


def synthetic_inventory():
    return pd.DataFrame([
        {"sample_id": f"b{b}c{c}i{i}", "group_id": f"b{b}c{c}i{i}", "brand": f"brand{b}",
         "label": c, "class_name": ["good", "bent", "loose", "missing", "pierced"][c],
         "sha256": f"hash{b}-{c}-{i}", "image_path": f"not-needed-{b}-{c}-{i}.jpg"}
        for b in range(3) for c in range(5) for i in range(40)
    ])


class ProtocolTests(unittest.TestCase):
    def test_group_isolation_and_reproducibility(self):
        frame = synthetic_inventory()
        duplicate = frame.iloc[[0]].copy()
        duplicate["sample_id"] = "duplicate"
        frame = pd.concat([frame, duplicate], ignore_index=True)
        a = split_groups(frame, 42)
        b = split_groups(frame, 42)
        self.assertEqual(a.role.tolist(), b.role.tolist())
        self.assertEqual(a.groupby("group_id").role.nunique().max(), 1)
        self.assertEqual(set(a.role), {"train", "val", "calibration", "test"})

    def test_lobo_does_not_use_held_out_brand_in_selection(self):
        frame = split_groups(synthetic_inventory(), 42, "brand1")
        self.assertTrue((frame[frame.role == "test"].brand == "brand1").all())
        self.assertFalse((frame[frame.role != "test"].brand == "brand1").any())
        self.assertEqual(set(frame.role), {"train", "val", "calibration", "test"})

    def test_contradictory_duplicate_labels_block_protocol(self):
        frame = split_groups(synthetic_inventory(), 42)
        frame.loc[1, "group_id"] = frame.loc[0, "group_id"]
        frame.loc[1, "label"] = 4
        with self.assertRaisesRegex(ValueError, "contradictory"):
            verify_assignments(frame)

    def test_manifest_tampering_is_detected(self):
        frame = split_groups(synthetic_inventory(), 42)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            path = Path(directory) / "protocol"
            export_protocol(frame, path, 42, Path(directory))
            load_protocol(path)
            with (path / "test.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Manifest changed"):
                load_protocol(path)

    def test_protocol_cannot_be_overwritten(self):
        frame = split_groups(synthetic_inventory(), 42)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            path = Path(directory) / "protocol"
            export_protocol(frame, path, 42, Path(directory))
            with self.assertRaises(FileExistsError):
                export_protocol(frame, path, 42, Path(directory))


class DecisionTests(unittest.TestCase):
    def test_review_everything_is_not_accuracy_or_coverage(self):
        metrics = decision_metrics(np.arange(5), -np.ones(5, dtype=int))
        self.assertEqual(metrics["fnr"], 0)
        self.assertEqual(metrics["coverage"], 0)
        self.assertEqual(metrics["macro_f1_all"], 0)
        self.assertIsNone(metrics["selective_risk"])
        self.assertGreater(metrics["fnr_95ci"][1], 0)

    def test_deployment_cost_uses_conditional_rates(self):
        # Good is always reviewed; defect is always correctly diagnosed.
        metrics = decision_metrics([0, 1, 2, 3, 4], [-1, 1, 2, 3, 4], pi_deploy=0.05)
        self.assertAlmostEqual(metrics["deployment_review_rate"], 0.95)
        self.assertAlmostEqual(metrics["cost_per_1000"], 1900)

    def test_wrong_defect_type_is_penalized(self):
        metrics = decision_metrics([0, 1, 2, 3, 4], [0, 2, 1, 4, 3])
        self.assertEqual(metrics["fnr"], 0)
        self.assertEqual(metrics["macro_recall_defect"], 0)
        self.assertGreater(metrics["cost_per_1000"], 0)

    def test_temperature_does_not_worsen_calibration_nll(self):
        logits = np.array([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
        fitted = fit_temperature(logits, np.array([0, 1, 0, 1]))
        self.assertLessEqual(fitted["after"]["nll"], fitted["before"]["nll"])

    def test_zero_fnr_does_not_guarantee_one_percent(self):
        self.assertGreater(wilson(0, 240)[1], 0.01)
        self.assertLessEqual(wilson(0, 381)[1], 0.01)

    def test_paired_bootstrap_identical_predictions_is_zero(self):
        y = np.tile(np.arange(5), 5)
        groups = np.arange(len(y)).astype(str)
        result = grouped_bootstrap(y, y, groups, 20, other=y)
        for interval in result["ci95"].values():
            self.assertEqual(interval, [0., 0.])

    def test_bootstrap_keeps_mixed_label_clusters_together(self):
        from src.straw_defect_cls import thesis_evaluation as evaluation
        y = np.array([0, 1, 0, 1, 2, 3, 4])
        groups = np.array(["a", "a", "b", "b", "c", "d", "e"])
        original = evaluation.decision_metrics
        sizes = []
        def observe(labels, *args, **kwargs):
            sizes.append(len(labels))
            self.assertEqual(int((labels == 0).sum()), int((labels == 1).sum()))
            return original(labels, *args, **kwargs)
        with patch.object(evaluation, "decision_metrics", side_effect=observe):
            evaluation.grouped_bootstrap(y, y, groups, 20)
        self.assertEqual(sizes, [len(y)] * 20)

    def test_selective_baseline_and_hard_cascade(self):
        values = {"full5": np.eye(5) * 10, "1": np.array([[10, 0]] + [[0, 10]] * 4),
                  "2": np.concatenate([np.zeros((1, 4)), np.eye(4) * 10])}
        self.assertTrue(np.array_equal(predict(values, {"kind": "hard_cascade"}), np.arange(5)))
        self.assertTrue(np.array_equal(predict(values, {"kind": "full5"}), np.arange(5)))
        self.assertTrue((predict(values, {"kind": "selective_full5", "confidence": 1.000001}) == -1).all())

    def test_infeasible_policy_stays_infeasible(self):
        config = json.loads((ROOT / "configs/straw_thesis.json").read_text())["policy"]
        config.update(pi_train=0.8, max_deployment_review=0.0)
        config["grid"] = {"tau_pass": [1.000001], "tau_defect": [0.5], "confidence": [0.9], "margin": [0.2]}
        logits = {"1": np.zeros((5, 2)), "2": np.zeros((5, 4))}
        policy, _ = choose_policy(logits, np.arange(5), {"1": {"temperature": 1}, "2": {"temperature": 1}}, config)
        self.assertEqual(policy["selection_status"], "infeasible")

    def test_risk_coverage_ends_at_full_coverage(self):
        curve = risk_coverage(np.eye(5), np.arange(5))
        self.assertEqual(curve[-1], {"coverage": 1., "risk": 0.})


class ExecutionTests(unittest.TestCase):
    def test_accumulation_final_short_window_matches_reference(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.straw_defect_cls.train_engine import TrainConfig, train_one_epoch
        torch.manual_seed(7)
        images, labels = torch.randn(6, 3), torch.tensor([0, 1, 2, 3, 4, 0])
        model = torch.nn.Linear(3, 5)
        reference = torch.nn.Linear(3, 5)
        reference.load_state_dict(model.state_dict())
        cfg = TrainConfig(device="cpu", use_amp=False, grad_accum=2, grad_clip=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        other = torch.optim.SGD(reference.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        train_one_epoch(model, loader, criterion, optimizer, torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), cfg)
        for indices in [slice(0, 4), slice(4, 6)]:
            other.zero_grad()
            criterion(reference(images[indices]), labels[indices]).backward()
            other.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_interrupted_training_resume_restores_exact_state(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.straw_defect_cls import train_engine as engine
        images = torch.arange(90, dtype=torch.float32).reshape(30, 3) / 90
        labels = torch.arange(30) % 5
        def loaders(**kwargs):
            data = TensorDataset(images, labels)
            return {"train": DataLoader(data, batch_size=10, shuffle=True), "val": DataLoader(data, batch_size=10)}
        def build(*args, **kwargs):
            return torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 5))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            with patch.object(engine, "build_dataloaders", side_effect=loaders), patch.object(engine, "build_model", side_effect=build), \
                 patch.object(engine, "plot_training_curves"), patch.object(engine, "plot_confusion_matrix"), patch.object(engine, "plot_roc_curves"):
                common = dict(device="cpu", use_amp=False, pretrained=False, epochs=3, num_workers=0,
                              output_dir=directory, resume=True, seed=42)
                engine.run_training(engine.TrainConfig(tag="reference", **common))
                original = engine.train_one_epoch
                count = [0]
                def interrupted(*args, **kwargs):
                    count[0] += 1
                    if count[0] == 2:
                        raise RuntimeError("simulated interruption")
                    return original(*args, **kwargs)
                with patch.object(engine, "train_one_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        engine.run_training(engine.TrainConfig(tag="resumed", **common))
                engine.run_training(engine.TrainConfig(tag="resumed", **common))
                reference = torch.load(Path(directory) / "reference/checkpoints/last.pt", weights_only=False)
                resumed = torch.load(Path(directory) / "resumed/checkpoints/last.pt", weights_only=False)
                self.assertEqual(reference["best_epoch"], resumed["best_epoch"])
                self.assertEqual(reference["scheduler"], resumed["scheduler"])
                for key in reference["model"]:
                    self.assertTrue(torch.equal(reference["model"][key], resumed["model"][key]))

    def test_matrix_has_expected_training_budgets(self):
        config = json.loads((ROOT / "configs/straw_thesis.json").read_text())
        self.assertEqual(len(jobs(config, "core", {})), 27)
        self.assertEqual(len(jobs(config, "extended", {})), 48)
        self.assertEqual(len(jobs(config, "smoke", {})), 3)
        self.assertEqual(len(jobs(config, "core", {"held_out_brand": "brand1"})), 3)
        groups = grouped_jobs(jobs(config, "extended", {}))
        weighted = groups["weighted_resnet18_r224_s42"]
        self.assertEqual(weighted["2"]["setting"], "core")

    def test_test_access_requires_freeze(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            with self.assertRaises(FileNotFoundError):
                verify_freeze(Namespace(output=Path(directory)), {"fingerprint": "fake"})

    def test_corruptions_are_deterministic_by_sample(self):
        from PIL import Image
        import torch
        from src.straw_defect_cls.thesis_inference import transform_image
        image = Image.new("RGB", (80, 60), "gray")
        first = transform_image(image, 32, "noise_2", "sample1")
        second = transform_image(image, 32, "noise_2", "sample1")
        third = transform_image(image, 32, "noise_2", "sample2")
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_vmamba_never_silently_becomes_a_different_architecture(self):
        from src.straw_defect_cls.models import build_model
        with patch.dict(sys.modules, {"vmamba": None}):
            with self.assertRaisesRegex(RuntimeError, "Refusing"):
                build_model("vmamba_t", pretrained=False)


if __name__ == "__main__":
    unittest.main()
