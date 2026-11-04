"""Regression checks for model/worker/image association and EXIF preprocessing."""
import configparser
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.stream_demo.resident_inference import ResidentInference, raw_rgb_tensor
from apps.stream_demo.serving_contract import CLASSES, load_contract, file_sha256
from src.straw_global_local.data import render_arrays, tensor_views
from src.straw_global_local.deployment import normalize


class StreamContractTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "artifacts/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "artifacts/b1_stream_512"
        self.out.mkdir(parents=True)
        (self.root / "configs").mkdir()
        (self.root / "configs/straw_realtime_stream512.json").write_text(json.dumps({
            "checkpoint_sha256": "pinned-checkpoint", "size": 512, "temperature": 1.0}))
        (self.out / "b1_rawrgb512.onnx").write_bytes(b"fixture graph")
        (self.out / "b1_rtx3050_fp16.engine").write_bytes(b"fixture engine")
        (self.out / "nvinfer.txt").write_text("[property]\ninfer-dims=3;512;512\n")
        config = configparser.ConfigParser(interpolation=None)
        config.read(self.out / "nvinfer.txt")
        canonical = json.dumps({s: dict(config.items(s)) for s in config.sections()}, sort_keys=True).encode()
        context = {"checkpoint_sha256": "pinned-checkpoint",
            "onnx_sha256": file_sha256(self.out / "b1_rawrgb512.onnx"),
            "inference_config_sha256": hashlib.sha256(canonical).hexdigest()}
        (self.out / "manifest.json").write_text(json.dumps({
            "size": 512, "classes": list(CLASSES), "checkpoint_sha256": context["checkpoint_sha256"],
            "onnx_sha256": context["onnx_sha256"], "decision_threshold": None}))
        (self.out / "engine_cache.json").write_text(json.dumps({"context": context,
            "engine_sha256": file_sha256(self.out / "b1_rtx3050_fp16.engine")}))
        self.client = ResidentInference(self.root, "", "", artifact="b1_stream_512")
        self.set_worker("worker-A")
        self.image = self.root / "orientation6.jpg"
        pixels = np.random.default_rng(42).integers(0, 256, (93, 157, 3), dtype=np.uint8)
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(pixels).save(self.image, exif=exif)

    def set_worker(self, generation, **changes):
        value = dict(state="ready", heartbeat=time.time(), generation=generation,
                     contract_sha256=self.client.identity["contract_sha256"])
        value.update(changes)
        (self.client.queue / "runtime.json").write_text(json.dumps(value))

    def test_modified_engine_is_rejected(self):
        (self.out / "b1_rtx3050_fp16.engine").write_bytes(b"another engine")
        with self.assertRaisesRegex(ValueError, "engine bytes"):
            load_contract(self.out)

    def test_modified_class_order_is_rejected(self):
        path = self.out / "manifest.json"
        value = json.loads(path.read_text())
        value["classes"][1:3] = reversed(value["classes"][1:3])
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "class order"):
            load_contract(self.out)

    def test_fresh_unidentified_worker_is_not_ready(self):
        self.set_worker("old-worker", contract_sha256="another-model")
        self.assertEqual(self.client.status()["state"], "error")
        with self.assertRaises(RuntimeError):
            self.client.submit(self.image, "same-event", {})

    def test_exif_resize_and_normalization_match_training(self):
        actual_raw = raw_rgb_tensor(self.image, 512)
        with Image.open(self.image) as source:
            expected = tensor_views(render_arrays(ImageOps.exif_transpose(source).convert("RGB"), "B1"), "B1")["global"][None].numpy()
        actual = normalize(actual_raw.transpose(1, 2, 0).astype(np.uint8))
        np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=0)

    def test_explicit_rotation_is_applied_after_exif(self):
        actual = raw_rgb_tensor(self.image, 512, 90)
        with Image.open(self.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").transpose(Image.Transpose.ROTATE_270)
            expected = np.asarray(image.resize((512, 512), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            raw_rgb_tensor(self.image, 512, 45)

    def test_new_generation_cannot_reuse_old_event_prediction(self):
        first = self.client.submit(self.image, "same-event", {})
        self.assertEqual(first, self.client.submit(self.image, "same-event", {}))
        self.set_worker("worker-B")
        second = self.client.submit(self.image, "same-event", {})
        self.assertNotEqual(first, second)

    def test_windows_sharing_lock_does_not_return_a_false_result(self):
        with patch("apps.stream_demo.resident_inference.read_json_snapshot", side_effect=PermissionError("locked")):
            self.assertEqual(self.client.result("waiting")["status"], "pending")

    def test_latency_does_not_subtract_windows_and_wsl_wall_clocks(self):
        request = self.client.submit(self.image, "clock-drift", {})
        event = json.loads((self.client.queue / f"{request}.event.json").read_text())
        response = dict(status="completed", request_id=request, generation=event["generation"],
            input_sha256=event["input_sha256"], contract_sha256=event["contract_sha256"], completed_at=-1000)
        (self.client.queue / f"{request}.json").write_text(json.dumps(response))
        first = self.client.result(request)["server_event_to_result_ms"]
        self.assertGreaterEqual(first, 0)
        response["completed_at"] = 1e12
        (self.client.queue / f"{request}.json").write_text(json.dumps(response))
        self.assertEqual(self.client.result(request)["server_event_to_result_ms"], first)

    def test_result_from_wrong_input_or_worker_is_rejected(self):
        request = self.client.submit(self.image, "event", {})
        event = json.loads((self.client.queue / f"{request}.event.json").read_text())
        valid = dict(status="completed", request_id=request, generation=event["generation"],
            input_sha256=event["input_sha256"], contract_sha256=event["contract_sha256"])
        for field, wrong in (("generation", "old-worker"), ("input_sha256", "other-pixels"),
                             ("contract_sha256", "other-model")):
            (self.client.queue / f"{request}.json").write_text(json.dumps(dict(valid, **{field: wrong})))
            self.assertEqual(self.client.result(request)["status"], "error", field)
        (self.client.queue / f"{request}.json").write_text(json.dumps(valid))
        self.assertEqual(self.client.result(request)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
