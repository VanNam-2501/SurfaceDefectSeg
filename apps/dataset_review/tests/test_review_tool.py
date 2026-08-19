from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from app import create_app  # noqa: E402
from review_core import DatasetReviewService, ReviewConfig  # noqa: E402


class ReviewToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.tool_dir = root / "tool"
        self.dataset_root = root / "dataset"
        split_dir = self.dataset_root / "dataset_audit" / "splits"
        split_dir.mkdir(parents=True)
        rows_by_split = {}
        for split, label in (("train", 0), ("val", 1), ("test", 0)):
            image_id = f"{'defect' if label else 'good'}_sample_{split}_1234abcd"
            image_path = Path(split) / "images" / f"{image_id}.png"
            absolute_image = self.dataset_root / image_path
            absolute_image.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((16, 16, 3), 90 if label else 140, dtype=np.uint8)
            Image.fromarray(image).save(absolute_image)

            mask_text = ""
            pixels = 0
            components = 0
            if label:
                mask_path = Path("masks") / f"{image_id}.png"
                absolute_mask = self.dataset_root / mask_path
                absolute_mask.parent.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((16, 16), dtype=np.uint8)
                # The real dataset stores binary masks as 0/1. The API must
                # normalize them to visible 0/255 PNGs for the browser.
                mask[4:8, 6:10] = 1
                Image.fromarray(mask).save(absolute_mask)
                mask_text = str(mask_path).replace("\\", "/")
                pixels = 16
                components = 1
            rows_by_split[split] = [
                {
                    "image_id": image_id,
                    "image_path": str(image_path).replace("\\", "/"),
                    "mask_path": mask_text,
                    "label": label,
                    "label_name": "Defect" if label else "Good",
                    "defect_group": "bump" if label else "good",
                    "source_split": split,
                    "height": 16,
                    "width": 16,
                    "defect_pixels": pixels,
                    "defect_ratio": pixels / 256,
                    "num_components": components,
                    "split": split,
                }
            ]
            pd.DataFrame(rows_by_split[split]).to_csv(split_dir / f"{split}.csv", index=False)

        self.results_root = root / "results"
        test_results = self.results_root / "unet" / "run" / "test"
        test_results.mkdir(parents=True)
        test_id = rows_by_split["test"][0]["image_id"]
        pd.DataFrame(
            [
                {
                    "image_id": test_id,
                    "image_score": 1.0,
                    "image_pred": 1,
                    "threshold": 0.5,
                    "predicted_positive_pixels": 20,
                    "pixel_tp": 0,
                    "pixel_fp": 20,
                    "pixel_fn": 0,
                    "positive_dice": "",
                }
            ]
        ).to_csv(test_results / "per_image_metrics.csv", index=False)
        qualitative_dir = test_results / "qualitative"
        qualitative_dir.mkdir()
        Image.fromarray(np.full((16, 64, 3), 180, dtype=np.uint8)).save(
            qualitative_dir / f"high_fp_01_{test_id}.png"
        )
        prediction_dir = test_results.parent / "predictions" / "test" / "probability"
        prediction_dir.mkdir(parents=True)
        probability = np.zeros((16, 16), dtype=np.uint8)
        probability[3:7, 4:9] = 240
        Image.fromarray(probability).save(prediction_dir / f"{test_id}.png")
        pd.DataFrame(
            [
                {
                    "image_id": test_id,
                    "split": "test",
                    "image_score": 240 / 255,
                    "image_pred": 1,
                    "threshold": 0.5,
                    "predicted_positive_pixels": 20,
                    "pixel_tp": 0,
                    "pixel_fp": 20,
                    "pixel_fn": 0,
                    "positive_dice": "",
                    "probability_path": f"predictions/test/probability/{test_id}.png",
                }
            ]
        ).to_csv(test_results.parent / "predictions" / "test" / "manifest.csv", index=False)

        segformer_results = self.results_root / "segformer" / "run" / "test"
        segformer_results.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "image_id": test_id,
                    "image_score": 0.1,
                    "image_pred": 0,
                    "threshold": 0.5,
                    "predicted_positive_pixels": 0,
                    "pixel_tp": 0,
                    "pixel_fp": 0,
                    "pixel_fn": 0,
                    "positive_dice": "",
                }
            ]
        ).to_csv(segformer_results / "per_image_metrics.csv", index=False)
        self.service = DatasetReviewService(
            ReviewConfig(
                tool_dir=self.tool_dir,
                dataset_root=self.dataset_root,
                results_roots=(self.results_root,),
            )
        )
        self.test_id = test_id

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def data_url(mask: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_prediction_candidates_and_api(self) -> None:
        page = self.service.list_items(candidate="false_positive")
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["image_id"], self.test_id)
        self.assertEqual(self.service.list_items(candidate="any")["total"], 1)
        self.assertEqual(self.service.list_items(candidate="none")["total"], 2)
        self.assertEqual(
            self.service.list_items(candidate="any")["total"]
            + self.service.list_items(candidate="none")["total"],
            self.service.list_items(candidate="all")["total"],
        )
        self.assertEqual(self.service.models, ["segformer/run", "unet/run"])
        self.assertEqual(self.service.list_items(candidate="false_positive", model="unet/run")["total"], 1)
        self.assertEqual(self.service.list_items(candidate="false_positive", model="segformer/run")["total"], 0)

        client = TestClient(create_app(self.service))
        self.assertEqual(client.get("/api/health").json()["records"], 3)
        self.assertEqual(client.get(f"/api/items/{self.test_id}").status_code, 200)
        self.assertEqual(client.get(f"/api/items/{self.test_id}/overlay").headers["content-type"], "image/png")
        defect_id = next(image_id for image_id in self.service.records if "defect" in image_id)
        visible_mask = client.get(f"/api/items/{defect_id}/mask")
        visible_array = np.asarray(Image.open(io.BytesIO(visible_mask.content)))
        self.assertEqual(int(visible_array.max()), 255)
        details = client.get(f"/api/items/{self.test_id}").json()
        self.assertEqual(details["qualitative_models"], ["unet/run"])
        self.assertEqual(details["prediction_models"], ["unet/run"])
        preview = client.get(
            f"/api/items/{self.test_id}/qualitative",
            params={"model": "unet/run"},
        )
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["content-type"], "image/png")
        self.assertEqual(
            client.get(
                f"/api/items/{self.test_id}/qualitative",
                params={"model": "segformer/run"},
            ).status_code,
            404,
        )
        for view in ("overlay", "probability", "binary"):
            prediction = client.get(
                f"/api/items/{self.test_id}/prediction",
                params={"model": "unet/run", "view": view},
            )
            self.assertEqual(prediction.status_code, 200)
            self.assertEqual(prediction.headers["content-type"], "image/png")
        self.assertIn("Data Review Studio", client.get("/").text)
        self.assertEqual(client.get("/static/app.js").status_code, 200)

    def test_review_mask_and_non_destructive_export(self) -> None:
        source_image = self.service.get_record(self.test_id)["image_path"]
        source_before = source_image.read_bytes()
        self.service.save_review(
            self.test_id,
            {
                "decision": "relabel_defect",
                "corrected_label": 1,
                "corrected_group": "scratch",
                "tags": ["hidden_defect"],
                "note": "Confirmed hidden defect",
                "hard_negative": False,
                "excluded": False,
                "reviewer": "tester",
            },
        )
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:5, 3:7] = 255
        result = self.service.save_mask_data_url(self.test_id, self.data_url(mask))
        self.assertEqual(result["diagnostics"]["mask"]["pixels"], 12)

        exported = self.service.export("unit_test")
        export_dir = Path(exported["path"])
        test_frame = pd.read_csv(export_dir / "splits" / "test.csv")
        row = test_frame.iloc[0]
        self.assertEqual(int(row["label"]), 1)
        self.assertEqual(row["defect_group"], "scratch")
        self.assertEqual(int(row["defect_pixels"]), 12)
        self.assertTrue(Path(row["mask_path"]).is_file())
        self.assertEqual(source_image.read_bytes(), source_before)
        self.assertFalse((self.dataset_root / "corrected_masks").exists())

        training_root = export_dir / "training_dataset"
        packaged_test = pd.read_csv(training_root / "dataset_audit" / "splits" / "test.csv")
        packaged_test_row = packaged_test.iloc[0]
        self.assertEqual(packaged_test_row["image_path"], f"images/test/{self.test_id}.png")
        self.assertEqual(packaged_test_row["mask_path"], f"masks/test/{self.test_id}.png")
        self.assertTrue((training_root / packaged_test_row["image_path"]).is_file())
        packaged_mask = np.asarray(Image.open(training_root / packaged_test_row["mask_path"]))
        self.assertEqual(int((packaged_mask != 0).sum()), 12)
        self.assertEqual(int(packaged_mask.max()), 255)

        packaged_train = pd.read_csv(training_root / "dataset_audit" / "splits" / "train.csv")
        packaged_good = packaged_train.iloc[0]
        self.assertTrue((training_root / packaged_good["image_path"]).is_file())
        blank_mask = np.asarray(Image.open(training_root / packaged_good["mask_path"]))
        self.assertEqual(int(blank_mask.max()), 0)
        self.assertTrue((training_root / "TRAINING_READY.md").is_file())

    def test_apply_model_prediction_as_review_mask(self) -> None:
        result = self.service.apply_prediction_mask(self.test_id, "unet/run")
        self.assertEqual(result["mask_pixels"], 20)
        review = result["review"]
        self.assertEqual(review["decision"], "fix_mask")
        self.assertEqual(review["corrected_label"], 1)
        self.assertIn("model_mask_accepted", review["tags"])
        self.assertTrue(Path(review["edited_mask_path"]).is_file())
        self.assertEqual(self.service.diagnostics(self.test_id)["mask"]["pixels"], 20)

        client = TestClient(create_app(self.service))
        response = client.post(
            f"/api/items/{self.test_id}/prediction-mask",
            json={"model": "unet/run"},
        )
        self.assertEqual(response.status_code, 200)

    def test_mask_save_removes_large_singleton_spray(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        singleton_points = [(y, x) for y in range(0, 10, 2) for x in range(0, 16, 2)]
        for y, x in singleton_points:
            mask[y, x] = 255
        mask[11:15, 11:15] = 255

        result = self.service.save_mask_data_url(self.test_id, self.data_url(mask))
        self.assertEqual(result["removed_noise_pixels"], 40)
        self.assertEqual(result["diagnostics"]["mask"]["pixels"], 16)
        self.assertEqual(result["diagnostics"]["mask"]["component_count"], 1)
        self.assertEqual(result["diagnostics"]["mask"]["single_pixel_components"], 0)

    def test_uncertain_is_excluded_and_audited(self) -> None:
        train_id = next(item for item in self.service.records if "train" in item)
        self.service.save_review(
            train_id,
            {
                "decision": "uncertain",
                "corrected_label": None,
                "corrected_group": "",
                "tags": ["annotation_ambiguous"],
                "note": "Needs second reviewer",
                "hard_negative": False,
                "excluded": False,
                "reviewer": "tester",
            },
        )
        exported = self.service.export("uncertain")
        clean = pd.read_csv(Path(exported["path"]) / "cleaned_manifest.csv")
        self.assertNotIn(train_id, set(clean["image_id"]))
        audit = pd.read_csv(Path(exported["path"]) / "audit_log.csv")
        row = audit[audit["image_id"] == train_id].iloc[0]
        self.assertEqual(row["decision"], "uncertain")
        self.assertEqual(int(row["excluded"]), 1)


if __name__ == "__main__":
    unittest.main()
