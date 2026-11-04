"""Regression tests for missed frames, equal-luma color, ROI latching and photo carriers."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts/test_tmp').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from apps.stream_demo.video_tracking import VideoGate, GateConfig, fill_holes
from apps.stream_demo.photo_conveyor import PhotoConveyor
from apps.stream_demo.tracking_math import minimum_assignment


def texture(size=60):
    values = np.random.default_rng(17).integers(160, 255, (size, size), dtype=np.uint8)
    return Image.fromarray(np.repeat(values[:, :, None], 3, axis=2))


class TrackingStabilityTests(unittest.TestCase):
    def gate(self, config=None):
        gate = VideoGate(config or GateConfig(calibration_frames=3))
        for i in range(3):
            gate.process(Image.new('RGB', (320, 180), (110, 110, 110)), (i - 3) * 100000000)
        return gate

    def frame(self, x=None, y=55):
        image = Image.new('RGB', (320, 180), (110, 110, 110))
        if x is not None:
            image.paste(texture(), (x, y))
        return image

    def test_global_assignment_preserves_the_available_match_for_each_track(self):
        # Greedy row order would spend column 0 on row 0 and lose row 1.
        result = minimum_assignment(np.array([[.2, .3], [.25, 10000.]]))
        self.assertEqual(set(result), {(0, 1), (1, 0)})

    def test_same_grayscale_but_different_color_is_detected(self):
        gate = self.gate()
        image = self.frame()
        image.paste(Image.new('RGB', (60, 60), (220, 54, 110)), (200, 50))
        # This color is approximately gray 110; grayscale subtraction alone loses it.
        state = gate.process(image, 0)
        self.assertEqual(state['detected_components'], 1)

    def test_short_dropout_keeps_id_and_never_triggers_from_prediction(self):
        gate = self.gate()
        events = []
        original_id = None
        for i, x in enumerate(range(280, 90, -10)):
            state = gate.process(self.frame(None if i in (9, 10) else x), i * 100000000)
            if i == 8:
                original_id = state['tracks'][0]['id']
            if i in (9, 10):
                self.assertFalse(state['tracks'][0]['observed'])
                self.assertEqual(state['events'], [])
            if i == 11:
                self.assertEqual(state['tracks'][0]['id'], original_id)
            events.extend(state['events'])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['track_id'], original_id)

    def test_irregular_frame_times_keep_velocity_and_id(self):
        gate = self.gate()
        ids, events = set(), []
        for seconds in (0, .1, .2, .31, .48, .72, .85, 1., 1.15, 1.3, 1.5, 1.7, 1.9):
            state = gate.process(self.frame(round(280 - seconds * 90)), round(seconds * 1e9))
            ids.update(t['id'] for t in state['tracks'])
            events.extend(state['events'])
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(events), 1)

    def test_roi_qualification_can_recover_after_the_crossing_frame(self):
        gate = self.gate()
        events = []
        for i, x in enumerate(range(280, 89, -10)):
            # At x=130 the center crosses 160 while the mask temporarily lies below the ROI.
            state = gate.process(self.frame(x, 145 if x == 130 else 100), i * 100000000)
            events.extend(state['events'])
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]['trigger_pts_ns'], events[0]['gate']['crossing_pts_ns'])
        self.assertGreaterEqual(events[0]['gate']['roi_overlap'], .8)

    def test_hole_filling_does_not_connect_separated_products(self):
        mask = np.zeros((50, 80), dtype=bool)
        for x in (5, 45):
            mask[5:40, x:x + 28] = True
            mask[9:36, x + 4:x + 24] = False
        filled = fill_holes(mask)
        self.assertTrue(filled[20, 15])
        self.assertFalse(filled[20, 38])

    def test_carrier_does_not_make_a_flat_photo_pass_quality(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp') as folder:
            root = Path(folder)
            inputs = root / 'artifacts/inspection/conveyor_inputs'
            inputs.mkdir(parents=True)
            image_id = uuid.uuid4().hex
            Image.new('RGB', (220, 220), (110, 110, 110)).save(inputs / (image_id + '.jpg'))
            source = PhotoConveyor(root, [image_id], {'simultaneous': 1, 'speed': 600})
            gate = VideoGate(GateConfig(**source.gate_settings()))
            events = []
            for frame in range(0, source.frame_count, 3):
                stream = io.BytesIO()
                source.render(frame).save(stream, format='JPEG', quality=90)
                with Image.open(io.BytesIO(stream.getvalue())) as opened:
                    state = gate.process(opened.convert('RGB'), round(frame / 30 * 1e9))
                events.extend(state['events'])
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertIn('blurred', event['quality']['reasons'])
            self.assertIn('low_contrast', event['quality']['reasons'])
            self.assertFalse(event['quality']['acceptable'])
            # Carrier borders are excluded from the saved CNN input, not just from the metric.
            pixels = np.asarray(event['image'])
            self.assertGreater(pixels[5:-5, 5:-5].min(), 100)
            self.assertAlmostEqual(event['quality']['sharpness'], 0., delta=1.)

    def test_partial_carrier_keeps_one_observed_track_at_entry_and_exit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp') as folder:
            root = Path(folder)
            inputs = root / 'artifacts/inspection/conveyor_inputs'
            inputs.mkdir(parents=True)
            image_id = uuid.uuid4().hex
            image = Image.new('RGB', (220, 220), (110, 110, 110))
            image.paste(texture(65), (125, 80))
            image.paste(texture(40), (30, 130))
            image.save(inputs / (image_id + '.jpg'))
            source = PhotoConveyor(root, [image_id], {'simultaneous': 1, 'speed': 120})
            gate = VideoGate(GateConfig(**source.gate_settings()))
            events = []
            for frame in range(0, source.frame_count, 3):
                stream = io.BytesIO()
                source.render(frame).save(stream, format='JPEG', quality=90)
                with Image.open(io.BytesIO(stream.getvalue())) as opened:
                    state = gate.process(opened.convert('RGB'), round(frame / 30 * 1e9))
                observed = [t for t in state['tracks'] if t['observed']]
                self.assertLessEqual(len(observed), 1)
                events.extend(state['events'])
            self.assertEqual(len(events), 1)
            self.assertEqual(gate.peak_tracks, 1)


if __name__ == '__main__':
    unittest.main()
