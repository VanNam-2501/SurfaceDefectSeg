"""Ensure synthetic backgrounds preserve white defect cues and source identity."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.data.make_degraded_set import (
    build_degraded_val_set, replace_studio_bg_with_conveyor, studio_background_mask,
)


class DegradedInputTests(unittest.TestCase):
    def test_background_replacement_preserves_internal_white_straw(self):
        image = np.full((64, 64, 3), 255, np.uint8)
        image[10:54, 10:54] = [100, 30, 30]
        image[20:45, 24:28] = 255
        mask = studio_background_mask(image)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[30, 25])
        result = replace_studio_bg_with_conveyor(image)
        np.testing.assert_array_equal(result[10:54, 10:54], image[10:54, 10:54])
        self.assertFalse(np.array_equal(result[0, 0], image[0, 0]))

    def test_gray_background_does_not_turn_a_white_straw_into_background(self):
        image = np.full((64, 64, 3), 180, np.uint8)
        image[15:50, 28:32] = 255
        self.assertFalse(studio_background_mask(image).any())
        np.testing.assert_array_equal(replace_studio_bg_with_conveyor(image), image)

    def test_b1_degradation_applies_exif_and_records_source_without_overwriting(self):
        parent = ROOT / "artifacts/test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            folder = Path(folder)
            source = folder / "Bent.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (80, 120), "red").save(source, exif=exif)
            original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = folder / "test.csv"
            pd.DataFrame([dict(image_path=str(source), sample_id="example", class_name="bent",
                label=1, brand="brand", sha256=original_sha)]).to_csv(manifest, index=False)
            with patch("scripts.data.make_degraded_set.degrade_image", side_effect=lambda image: image):
                frame = build_degraded_val_set(manifest, folder / "generated", exif_transpose=True,
                                               manifest_name="test_degraded.csv")
            row = frame.iloc[0]
            self.assertEqual(row.source_exif_orientation, 6)
            self.assertTrue(row.exif_transpose_applied)
            self.assertEqual(row.source_sha256, original_sha)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_sha)
            with Image.open(row.image_path) as output:
                self.assertEqual(output.size, (120, 80))
                self.assertEqual(output.getexif().get(274, 1), 1)
            self.assertEqual(row.sha256, hashlib.sha256(Path(row.image_path).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
