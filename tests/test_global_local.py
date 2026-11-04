"""Scientific-protocol and model tests; passing does not establish accuracy."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.straw_global_local import data, runner
from src.straw_global_local.model import GlobalLocalCNN


class TinyCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.classifier = nn.Linear(4, 5)

    def forward(self, batch):
        return self.classifier(self.encoder(batch["global"].mean((-1, -2))).tanh())


class GlobalLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        temp_root = ROOT / "artifacts" / "verification" / "global_local_test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=temp_root)
        cls.root = Path(cls.temp.name)
        cls.frames = {}
        rng = np.random.default_rng(11)
        for role in data.ROLES:
            rows = []
            for label in range(5):
                sample = f"{role}_{label}"
                path = cls.root / f"{sample}.png"
                Image.fromarray(rng.integers(0, 256, (90, 90, 3), dtype=np.uint8)).save(path)
                rows.append({"sample_id": sample, "group_id": sample, "image_path": str(path),
                             "label": label, "class_name": data.CLASSES[label], "sha256": data.file_hash(path),
                             "width": 90, "height": 90})
            cls.frames[role] = pd.DataFrame(rows)
        cls.protocol, cls.cache = cls.root / "protocol", cls.root / "cache"
        data.save_protocol(cls.frames, cls.protocol, {"smoke_only": False, "historical_exposure": "synthetic fixture"})
        data.create_cache(cls.protocol, cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_groups_cannot_cross_roles(self):
        frames = {k: v.copy() for k, v in self.frames.items()}
        frames["val"].loc[0, "group_id"] = frames["train"].loc[0, "group_id"]
        with self.assertRaisesRegex(ValueError, "group crosses"):
            data.validate_frames(frames)

    def test_label_name_consistency(self):
        frames = {k: v.copy() for k, v in self.frames.items()}
        frames["train"].loc[0, "class_name"] = "missing"
        with self.assertRaisesRegex(ValueError, "Label/name"):
            data.validate_frames(frames)

    def test_manifest_tampering_rejected(self):
        directory = self.root / "tamper"
        data.save_protocol(self.frames, directory, {"smoke_only": True})
        path = directory / "train.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                data.load_protocol(directory)
        finally:
            path.write_bytes(original)

    def test_crop_geometry_covers_frame_with_overlap(self):
        boxes = data.crop_boxes(3000, 3000)
        self.assertEqual(boxes, [(0, 0, 1800, 1800), (1200, 0, 3000, 1800),
                                 (0, 1200, 1800, 3000), (1200, 1200, 3000, 3000)])

    def test_views_independent_of_label(self):
        original = self.frames["train"].copy()
        changed = original.copy()
        changed["label"] = 4 - changed.label
        a = data.ViewDataset(original, self.cache, "B6")[0]
        b = data.ViewDataset(changed, self.cache, "B6")[0]
        self.assertNotEqual(a[1], b[1])
        self.assertTrue(all(torch.equal(a[0][key], b[0][key]) for key in a[0]))

    def test_horizontal_flip_keeps_crop_geometry(self):
        with Image.open(self.frames["train"].iloc[0].image_path) as image:
            arrays = data.render_arrays(image)
            expected = data.tensor_views(data.render_arrays(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)), "B6")
        with patch.object(data.torch, "rand", side_effect=[torch.tensor(.5), torch.tensor(0.)]):
            actual = data.tensor_views(arrays, "B6", augment=True)
        self.assertTrue(torch.allclose(actual["coords"], expected["coords"], atol=1e-6))
        self.assertTrue(torch.allclose(actual["local"], expected["local"], atol=.02))
        self.assertTrue(torch.allclose(actual["global"], expected["global"], atol=.02))

    def test_cache_tampering_rejected(self):
        target = self.cache / "train_0.npz"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "views changed"):
                data.load_cache(self.protocol, self.cache)
        finally:
            target.write_bytes(original)

    def test_all_seven_models_train_without_labels_or_brand_input(self):
        batch = {"global": torch.randn(2, 3, 64, 64), "local": torch.randn(2, 4, 3, 64, 64),
                 "coords": torch.rand(2, 4, 4)}
        for variant in data.VARIANTS:
            with self.subTest(variant=variant):
                model = GlobalLocalCNN(variant, pretrained=False).train()
                logits, weights = model(batch, return_attention=True)
                self.assertEqual(logits.shape, (2, 5))
                nn.functional.cross_entropy(logits, torch.tensor([0, 4])).backward()
                self.assertGreater(float(model.encoder.conv1.weight.grad.norm()), 0)
                if variant == "B6":
                    self.assertGreater(float(model.position_projection.weight.grad.norm()), 0)
                self.assertTrue(torch.isfinite(logits).all())
                if weights is not None:
                    self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2)))

    def test_bn_running_stats_frozen_but_affine_learns(self):
        model = GlobalLocalCNN("B0", pretrained=False).train()
        before = model.encoder.bn1.running_mean.clone()
        model({"global": torch.randn(2, 3, 64, 64)}).sum().backward()
        self.assertTrue(torch.equal(before, model.encoder.bn1.running_mean))
        self.assertIsNotNone(model.encoder.bn1.weight.grad)

    def test_joint_patch_coordinate_permutation_preserves_set_aggregation(self):
        torch.manual_seed(9)
        batch = {"global": torch.randn(1, 3, 64, 64), "local": torch.randn(1, 4, 3, 64, 64),
                 "coords": torch.rand(1, 4, 4)}
        order = [2, 0, 3, 1]
        permuted = {"global": batch["global"], "local": batch["local"][:, order], "coords": batch["coords"][:, order]}
        for variant in ("B5", "B6"):
            model = GlobalLocalCNN(variant, pretrained=False).eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(batch), model(permuted), atol=2e-6))

    def test_missing_coordinates_fail_in_position_model(self):
        model = GlobalLocalCNN("B6", pretrained=False).eval()
        with self.assertRaises(KeyError):
            model({"global": torch.randn(1, 3, 64, 64), "local": torch.randn(1, 4, 3, 64, 64)})

    def test_defect_type_error_is_not_a_missed_defect(self):
        labels = np.arange(5)
        predictions = [0, 2, 0, 3, 4]
        metrics = runner.classification_metrics(labels, np.eye(5)[predictions]*5)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["wrong_defect_type"], 1)
        self.assertEqual(metrics["fnr"], .25)

    def test_identical_predictions_have_zero_paired_interval(self):
        labels = np.tile(np.arange(5), 3)
        result = runner.paired_group_interval(labels, np.arange(15), [labels]*3, [labels]*3, repeats=10)
        self.assertEqual(result["ci95"], [0., 0.])
        self.assertEqual(result["interpretation"], "no clear difference")

    def test_partial_accumulation_matches_explicit_effective_batches(self):
        class Samples(Dataset):
            def __init__(self):
                self.x = torch.arange(21, dtype=torch.float64).reshape(7, 3) / 20
            def __len__(self): return 7
            def __getitem__(self, i): return {"global": self.x[i]}, i % 5, str(i)
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(3, 5).double()
            def forward(self, batch): return self.head(batch["global"])
        torch.manual_seed(3)
        first, samples = Head(), Samples()
        second = copy.deepcopy(first)
        optimizer = torch.optim.SGD(first.parameters(), lr=.05)
        reference = torch.optim.SGD(second.parameters(), lr=.05)
        runner.train_epoch(first, DataLoader(samples, batch_size=2), optimizer,
                           torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), 3)
        for indices in (range(6), range(6, 7)):
            reference.zero_grad()
            nn.functional.cross_entropy(second({"global": samples.x[list(indices)]}), torch.tensor([i % 5 for i in indices])).backward()
            reference.step()
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-12))

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        config = {"purpose": "smoke", "variants": ["B0"], "seeds": [8], "epochs": 2, "patience": 3,
                  "batch_size": 2, "grad_accum": 2, "workers": 0, "encoder_lr": .001,
                  "head_lr": .002, "weight_decay": .0001, "pretrained": False, "amp": False}
        job, spec = {"id": "B0_s8", "variant": "B0", "seed": 8}, {"fingerprint": "fixture"}
        uninterrupted = argparse.Namespace(output=self.root / "uninterrupted", cache=self.cache, max_epochs_this_call=None)
        resumed = argparse.Namespace(output=self.root / "resumed", cache=self.cache, max_epochs_this_call=1)
        with patch.object(runner, "GlobalLocalCNN", TinyCNN):
            runner.train_one(uninterrupted, config, self.frames, spec, job, torch.device("cpu"))
            runner.train_one(resumed, config, self.frames, spec, job, torch.device("cpu"))
            self.assertFalse((resumed.output / "runs/B0_s8/complete.json").exists())
            resumed.max_epochs_this_call = None
            runner.train_one(resumed, config, self.frames, spec, job, torch.device("cpu"))
        a = torch.load(uninterrupted.output / "runs/B0_s8/last.pt", weights_only=False)
        b = torch.load(resumed.output / "runs/B0_s8/last.pt", weights_only=False)
        self.assertEqual(a["best"], b["best"])
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]))

    def test_unequal_effective_batch_rejected(self):
        config = data.read_json(ROOT / "configs/straw_global_local.json")
        config["batches"]["B1"]["grad_accum"] = 8
        with self.assertRaisesRegex(ValueError, "Effective image batch"):
            runner.check_config(config)

    def test_no_test_without_selection_ledger(self):
        args = argparse.Namespace(output=self.root / "not_frozen")
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            runner.load_ledger(args, {}, {})

    def test_review_must_be_explicit_and_attributed(self):
        self.assertFalse(runner.check_review({field: True for field in runner.REVIEW_FIELDS}))
        self.assertFalse(runner.check_review(data.read_json(self.protocol / "review.json")))
        self.assertTrue(runner.check_review({**{field: True for field in runner.REVIEW_FIELDS},
                                           "reviewer": "fixture reviewer", "notes": "test fixture only"}))

    def test_development_pilot_cannot_freeze_for_test(self):
        config = data.read_json(ROOT / "configs/straw_global_local_smoke.json")
        config["purpose"] = "development"
        path = self.root / "development.json"
        data.write_json(path, config)
        args = argparse.Namespace(config=path, protocol=self.protocol, cache=self.cache,
                                  output=self.root / "development_output")
        with self.assertRaisesRegex(ValueError, "Development pilot"):
            runner.freeze(args)
        self.assertFalse((args.output / "freeze.json").exists())

    def test_confirmatory_freeze_requires_actual_review_record(self):
        config = data.read_json(ROOT / "configs/straw_global_local_smoke.json")
        config["purpose"] = "confirmatory"
        path = self.root / "confirmatory.json"
        data.write_json(path, config)
        args = argparse.Namespace(config=path, protocol=self.protocol, cache=self.cache,
                                  output=self.root / "confirmatory_output")
        with self.assertRaisesRegex(ValueError, "Actual label/group"):
            runner.freeze(args)

    def test_family_selection_ignores_test_scores(self):
        config = {"variants": ["B0", "B1", "B5", "B6"], "seeds": [1, 2]}
        scores = {"B0": .4, "B1": .6, "B5": .7, "B6": .65}
        completed = {f"{v}_s{s}": {"validation": {"macro_f1": scores[v]},
                     "test": {"macro_f1": 1-scores[v]}} for v in scores for s in config["seeds"]}
        chosen = runner.choose_families(config, completed)
        self.assertEqual(chosen["baseline"], "B1")
        self.assertEqual(chosen["proposal"], "B5")

    def test_modified_completed_checkpoint_rejected(self):
        job, spec = {"id": "B0_s1", "variant": "B0", "seed": 1}, {"fingerprint": "fixture"}
        folder = self.root / "checkpoint_tamper" / "runs" / job["id"]
        folder.mkdir(parents=True)
        runner.atomic_torch_save(folder / "best.pt", {"weight": torch.ones(1)})
        np.savez_compressed(folder / "val_predictions.npz", logits=np.zeros((5,5)))
        data.write_json(folder / "complete.json", {"experiment": "fixture", "job": job,
            "best_checkpoint": data.file_hash(folder / "best.pt"),
            "val_predictions": data.file_hash(folder / "val_predictions.npz")})
        runner.verify_run(self.root / "checkpoint_tamper", job, spec)
        with (folder / "best.pt").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checkpoint/predictions changed"):
            runner.verify_run(self.root / "checkpoint_tamper", job, spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
