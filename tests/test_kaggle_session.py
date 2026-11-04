"""Session interruption, state transfer and full-sweep orchestration tests."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.straw_global_local import data, runner

spec = importlib.util.spec_from_file_location("kaggle_session_test", ROOT / "scripts/research/kaggle_session.py")
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


class TinyDropoutCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(4, 5)

    def forward(self, batch):
        return self.classifier(self.dropout(self.encoder(batch["global"].mean((-1, -2))).tanh()))


class KaggleSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        base = ROOT / "artifacts/verification/kaggle_session_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=base)
        cls.root = Path(cls.temp.name).resolve()
        frames = {}
        rng = np.random.default_rng(51)
        for role in data.ROLES:
            rows = []
            for label in range(5):
                sample = f"{role}_{label}"
                image = cls.root / f"{sample}.png"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(image)
                rows.append({"sample_id": sample, "group_id": sample, "image_path": str(image),
                             "label": label, "class_name": data.CLASSES[label], "sha256": data.file_hash(image),
                             "width": 32, "height": 32})
            frames[role] = pd.DataFrame(rows)
        cls.protocol, cls.cache = cls.root / "protocol", cls.root / "cache"
        data.save_protocol(frames, cls.protocol, {"smoke_only": True, "historical_exposure": "synthetic only"})
        data.create_cache(cls.protocol, cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, name, variants=None, seeds=None, epochs=2):
        project = self.root / name
        shutil.copytree(self.protocol, project / session.PROTOCOL_REL)
        shutil.copytree(self.cache, project / session.CACHE_REL)
        config = data.read_json(ROOT / "configs/straw_global_local_smoke.json")
        config.update(variants=variants or ["B0", "B5"], seeds=seeds or [8], epochs=epochs, patience=epochs+1,
                      augmentation=True)
        config_path = self.root / f"{name}_config.json"
        data.write_json(config_path, config)
        return argparse.Namespace(config=config_path, protocol=project / session.PROTOCOL_REL,
                                  cache=project / session.CACHE_REL, output=project / session.OUTPUT_REL,
                                  device="cpu", run_id=None, max_epochs_this_call=None)

    def test_resume_zip_matches_uninterrupted_weights_optimizer_and_schedule(self):
        full = self.args("full")
        partial = self.args("partial")
        project = partial.output.parents[2]
        with patch.object(runner, "GlobalLocalCNN", TinyDropoutCNN):
            self.assertTrue(session.train_until_budget(full, 1e12))
            # First epoch completes; the next budget check asks the job to pause.
            ticks = iter([0., 1e9])
            self.assertFalse(session.train_until_budget(partial, 3600, clock=lambda: next(ticks)))
            checkpoint = partial.output / "runs/B0_s8/last.pt"
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved["epoch"], 0)
            self.assertFalse((partial.output / "runs/B0_s8/complete.json").exists())
            data.write_json(project / session.IDENTITY_REL, {"fixture": True})
            snapshot = self.root / "continued.zip"
            manifest = session.make_snapshot(project, snapshot, "fixture-bundle")
            self.assertFalse(any(n.startswith("raw/") for n in manifest["files"]))
            project.rename(self.root / "partial_backup")
            project.mkdir()
            session.restore_snapshot(snapshot, project, "fixture-bundle")
            self.assertTrue(session.train_until_budget(partial, 1e12))
        for job in ("B0_s8", "B5_s8"):
            a = torch.load(full.output / f"runs/{job}/last.pt", weights_only=False)
            b = torch.load(partial.output / f"runs/{job}/last.pt", weights_only=False)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["best"], b["best"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for index, values in a["optimizer"]["state"].items():
                for key, value in values.items():
                    self.assertTrue(torch.equal(value, b["optimizer"]["state"][index][key]))

    def test_sweep_runs_all_21_and_completed_jobs_are_not_retrained(self):
        args = self.args("all21", variants=list(data.VARIANTS), seeds=[42, 43, 44], epochs=1)
        with patch.object(runner, "GlobalLocalCNN", TinyDropoutCNN):
            self.assertTrue(session.train_until_budget(args, 1e12))
            before = {p: data.file_hash(p) for p in args.output.rglob("last.pt")}
            self.assertEqual(len(before), 21)
            with patch.object(runner, "train_one", side_effect=AssertionError("Completed job retrained")):
                self.assertTrue(session.train_until_budget(args, 1e12))
            self.assertEqual(before, {p: data.file_hash(p) for p in before})

    def test_budget_stop_does_not_create_fake_completion(self):
        args = self.args("no_time")
        with patch.object(runner, "train_one", side_effect=AssertionError("No time to train")):
            self.assertFalse(session.train_until_budget(args, 0, clock=lambda: 0))
        self.assertFalse(list(args.output.rglob("complete.json")))

    def fixture_snapshot(self, name):
        project = self.root / name
        data.write_json(project / session.IDENTITY_REL, {"fixture": True})
        data.write_json(project / session.OUTPUT_REL / "progress.json", {"epoch": 7})
        snapshot = self.root / f"{name}.zip"
        session.make_snapshot(project, snapshot, "test-bundle")
        return project, snapshot

    def test_snapshot_tamper_rejected_before_overwriting_progress(self):
        project, snapshot = self.fixture_snapshot("tamper")
        with zipfile.ZipFile(snapshot) as source:
            payload = {n: source.read(n) for n in source.namelist()}
        name = (session.OUTPUT_REL / "progress.json").as_posix()
        payload[name] = b'{"epoch":999}'
        changed = self.root / "tampered.zip"
        with zipfile.ZipFile(changed, "w") as archive:
            for n, value in payload.items():
                archive.writestr(n, value)
        project.rename(self.root / "tamper_old")
        project.mkdir()
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            session.restore_snapshot(changed, project, "test-bundle")
        self.assertFalse((project / session.IDENTITY_REL).exists())

    def test_resume_does_not_overwrite_live_files(self):
        project, snapshot = self.fixture_snapshot("live")
        with self.assertRaisesRegex(FileExistsError, "overwrite live"):
            session.restore_snapshot(snapshot, project, "test-bundle")

    def test_wrong_bundle_and_root_are_rejected(self):
        project, snapshot = self.fixture_snapshot("wrong")
        with self.assertRaisesRegex(ValueError, "bundle/root mismatch"):
            session.restore_snapshot(snapshot, project, "different-bundle")
        with self.assertRaisesRegex(ValueError, "bundle/root mismatch"):
            session.restore_snapshot(snapshot, self.root / "different-root", "test-bundle")

    def test_archive_traversal_is_rejected(self):
        for name in ("../outside", "/absolute", "C:/windows", "x\\y", "./a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                session.safe_name(name)

    def test_resume_detection_requires_unambiguous_selection(self):
        inputs = self.root / "inputs"
        for name in ("old", "new"):
            path = inputs / name / session.RESUME_NAME
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
        with self.assertRaisesRegex(ValueError, "Multiple resume"):
            session.find_resume(inputs, None)
        self.assertIsNone(session.find_resume(inputs, ""))
        self.assertEqual(session.find_resume(inputs, inputs / "new" / session.RESUME_NAME), inputs / "new" / session.RESUME_NAME)

    def test_expanded_resume_dataset_restores(self):
        project, snapshot = self.fixture_snapshot("expanded")
        expanded = self.root / "expanded-input"
        with zipfile.ZipFile(snapshot) as archive:
            archive.extractall(expanded)
        project.rename(self.root / "expanded_old")
        project.mkdir()
        self.assertEqual(session.find_resume(expanded, None), expanded)
        session.restore_snapshot(expanded, project, "test-bundle")
        self.assertEqual(data.read_json(project / session.OUTPUT_REL / "progress.json"), {"epoch": 7})

    def test_epoch_reserve_uses_longest_measured_epoch(self):
        self.assertEqual(session.epoch_reserve([]), 600)
        self.assertEqual(session.epoch_reserve([{"seconds": 800}, {"seconds": 1000}]), 1600)

    def test_bundle_checksums_verified_before_use(self):
        inputs = self.root / "corrupt-input"
        inputs.mkdir()
        path = inputs / "run.py"
        path.write_bytes(b"original")
        record = {"bytes": path.stat().st_size, "sha256": data.file_hash(path)}
        manifest = {"schema": 1, "kind": "straw-kaggle-all-in-one", "files": {"run.py": record}}
        manifest["fingerprint"] = session.canonical_hash(manifest)
        data.write_json(inputs / "bundle_manifest.json", manifest)
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            session.materialize_bundle(inputs, self.root / "corrupt-copy")

    def test_complete_session_pauses_exports_and_auto_restores_next_session(self):
        """Exercise materialize -> prepare -> cache -> pause -> ZIP -> restore -> train/report."""
        bundle, project = self.root / "integration-input", self.root / "integration-project"
        output, inputs = self.root / "integration-output", self.root / "integration-previous"
        bundle.mkdir()
        config = data.read_json(ROOT / "configs/straw_global_local_smoke.json")
        config.update(variants=["B0", "B5"], seeds=[8], epochs=2, patience=3)
        data.write_json(bundle / "configs/straw_global_local.json", config)
        prefix = "Straw Defects in Juice Box"
        _, original = data.load_protocol(self.protocol)
        frames = {role: frame.copy() for role, frame in original.items()}
        for role, frame in frames.items():
            frame["relative_path"] = [f"{prefix}/{role}/{Path(p).name}" for p in frame.image_path]
            for row in frame.itertuples():
                destination = inputs / "dataset-original" / row.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(row.image_path, destination)
        inventory = bundle / "artifacts/straw_thesis/data_v2/iid/inventory.csv"
        inventory.parent.mkdir(parents=True)
        pd.concat(list(frames.values())).to_csv(inventory, index=False)
        records = {p.relative_to(bundle).as_posix(): {"bytes": p.stat().st_size, "sha256": data.file_hash(p)}
                   for p in bundle.rglob("*") if p.is_file()}
        manifest = {"schema": 1, "kind": "straw-kaggle-all-in-one", "files": records, "image_count": 15}
        manifest["fingerprint"] = session.canonical_hash(manifest)
        data.write_json(bundle / "bundle_manifest.json", manifest)
        def prepare_fixture(source, protocol, image_root):
            copied = {role: frame.copy() for role, frame in frames.items()}
            for frame in copied.values():
                frame["image_path"] = [str((image_root / relative).resolve()) for relative in frame.relative_path]
            return data.save_protocol(copied, protocol, {"smoke_only": True, "historical_exposure": "synthetic integration"})
        def bind_fixture(project_root, tree, name):
            # Windows lacks symlink permission; tiny copies stand in for Linux aliases.
            raw = project_root / "raw"
            shutil.copytree(tree, raw / name)
            return raw, None
        with patch.object(data, "prepare", side_effect=prepare_fixture), patch.object(runner, "GlobalLocalCNN", TinyDropoutCNN), patch.object(session, "bind_dataset", side_effect=bind_fixture):
            paused = session.run_session(bundle, project, input_root=inputs, output_root=output,
                                         session_hours=.001, device="cpu", benchmark_enabled=False)
            self.assertEqual(paused["state"], "paused_time_budget")
            self.assertTrue((output / session.RESUME_NAME).is_file())
            previous = inputs / "previous"
            previous.mkdir(parents=True)
            shutil.copy2(output / session.RESUME_NAME, previous / session.RESUME_NAME)
            project.rename(self.root / "integration-project-old")
            # A different Kaggle mount must not invalidate the frozen protocol.
            (inputs / "dataset-original").rename(inputs / "renamed-mount")
            restored = session.run_session(bundle, project, input_root=inputs, output_root=output,
                                           device="cpu", benchmark_enabled=False)
        self.assertEqual(restored["state"], "training_complete")
        self.assertEqual(restored["completed_runs"], 2)
        self.assertEqual(len(restored["sessions"]), 2)
        self.assertTrue((project / session.OUTPUT_REL / "report/validation_runs.csv").is_file())
        self.assertFalse(list((project / session.OUTPUT_REL).rglob("test_predictions.npz")))
        self.assertEqual(data.read_json(project / session.PROTOCOL_REL / "review.json")["labels_reviewed"], False)

    def test_dataset_discovery_with_nested_mount_and_missing_outer_folder(self):
        project = self.root / "path-project"
        inventory = project / "artifacts/straw_thesis/data_v2/iid/inventory.csv"
        inventory.parent.mkdir(parents=True)
        pd.DataFrame({"relative_path": ["Straw Defects in Juice Box/Brand/Good/a.jpg",
                                             "Straw Defects in Juice Box/Brand/Bad/b.jpg"]}).to_csv(inventory, index=False)
        inputs = self.root / "path-inputs"
        tree = inputs / "datasets/account/any-slug/Straw Defects in Juice Box"
        for name in ("Brand/Good/a.jpg", "Brand/Bad/b.jpg"):
            path = tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        found, prefix = session.discover_dataset(project, inputs)
        self.assertEqual(found, tree.resolve())
        self.assertEqual(prefix, "Straw Defects in Juice Box")
        flat = inputs / "flattened-slug"
        tree.rename(flat)
        self.assertEqual(session.discover_dataset(project, inputs)[0], flat.resolve())
        self.assertEqual(session.discover_dataset(project, inputs, flat)[0], flat.resolve())
        duplicate = inputs / "duplicate"
        shutil.copytree(flat, duplicate)
        with self.assertRaisesRegex(ValueError, "Found 2 matching"):
            session.discover_dataset(project, inputs)
        self.assertEqual(session.discover_dataset(project, inputs, flat)[0], flat.resolve())

    def test_dataset_link_uses_input_in_place_and_refuses_existing_directory(self):
        project = self.root / "link-project"
        tree = self.root / "input-images"
        tree.mkdir()
        with patch.object(session.os, "symlink") as symlink:
            root, link = session.bind_dataset(project, tree, "images")
            symlink.assert_called_once_with(str(tree), str(root / "images"), target_is_directory=True)
            self.assertEqual(link, root / "images")
        link.mkdir()
        with self.assertRaises(FileExistsError):
            session.bind_dataset(project, tree, "images")

    def test_wrong_original_with_same_filename_is_rejected_even_with_cache(self):
        _, frames = data.load_protocol(self.protocol)
        image = Path(frames["val"].iloc[0].image_path)
        original = image.read_bytes()
        try:
            image.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Mounted original differs"):
                session.verify_originals(self.protocol)
        finally:
            image.write_bytes(original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
