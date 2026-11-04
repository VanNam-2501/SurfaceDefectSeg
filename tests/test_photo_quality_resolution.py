"""Quality regression: small display tiles must not replace source pixels."""
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
import uuid

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts/test_tmp').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from apps.stream_demo.image_quality import assess
from apps.stream_demo.photo_conveyor import PhotoConveyor
from apps.stream_demo.video_tracking import VideoGate, GateConfig
from apps.stream_demo.video_session import VideoSession
from apps.stream_demo.inspection_service import InspectionService


def detail_photo():
    x, y = np.meshgrid(np.arange(640), np.arange(640))
    gray = np.clip(128 + 12 * np.sin(x * 2 * np.pi / 7) + 12 * np.sin(y * 2 * np.pi / 7)
                   + 35 * ((x // 100 + y // 100) % 2), 0, 255).astype('uint8')
    return Image.fromarray(gray).convert('RGB')


class Resident:
    identity = {'contract_sha256': 'fixture', 'engine_sha256': 'fixture-engine'}

    def __init__(self):
        self.calls = 0

    def status(self):
        return {'state': 'ready'}

    def submit(self, *args):
        self.calls += 1
        return uuid.uuid4().hex

    def result(self, request_id):
        return {'status': 'completed', 'class': 'good', 'defective': False, 'contract_sha256': 'fixture'}


class PhotoQualityResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'artifacts/inspection/conveyor_inputs'
        self.inputs.mkdir(parents=True)

    def add_photo(self, image):
        image_id = uuid.uuid4().hex
        image.save(self.inputs / (image_id + '.jpg'), quality=95)
        return image_id

    def test_known_detail_loss_is_prevented_in_all_lane_counts(self):
        sharp = detail_photo()
        # This reproduces the previous false alert; a random-noise fixture alone
        # would not detect the loss of detail introduced by the 128px renderer.
        old_preview_tile = sharp.resize((128, 128), Image.Resampling.BILINEAR)
        self.assertIn('blurred', assess(old_preview_tile)['reasons'])
        image_id = self.add_photo(sharp)
        for lanes in (1, 2, 3):
            source = PhotoConveyor(self.root, [image_id], {'simultaneous': lanes})
            with self.subTest(lanes=lanes):
                self.assertGreaterEqual(source.tile_size, 256)
                quality = assess(source.tiles[0])
                self.assertTrue(quality['acceptable'], quality)
                self.assertEqual(quality['thresholds']['min_sharpness'], 18)

    def test_blurred_dark_and_flat_photos_still_fail_in_every_mode(self):
        sharp = detail_photo()
        for image, reason in ((sharp.filter(ImageFilter.GaussianBlur(12)), 'blurred'),
                              (ImageOps.colorize(sharp.convert('L').point(lambda v: v // 12), 'black', '#ffffff'), 'too_dark'),
                              (Image.new('RGB', (640, 640), '#808080'), 'low_contrast')):
            image_id = self.add_photo(image)
            for lanes in (1, 2, 3):
                source = PhotoConveyor(self.root, [image_id], {'simultaneous': lanes})
                self.assertIn(reason, assess(source.tiles[0])['reasons'])

    def test_mixed_products_keep_quality_and_identity_in_both_directions(self):
        sharp = detail_photo()
        ids = [self.add_photo(im) for im in
               (sharp, sharp.filter(ImageFilter.GaussianBlur(12)), Image.new('RGB', (640, 640), '#080808'))]
        for direction in ('right_to_left', 'left_to_right'):
            source = PhotoConveyor(self.root, ids, {'simultaneous': 3, 'speed': 600}, direction)
            gate = VideoGate(GateConfig(direction=direction, **source.gate_settings()))
            events, frame, step = [], 0, 0
            while frame < source.frame_count:
                state = gate.process(source.render(frame), round(frame / source.fps * 1e9))
                events.extend(state['events'])
                frame += (3, 4, 2, 6, 3, 3)[step % 6]
                step += 1
            with self.subTest(direction=direction):
                self.assertEqual(len(events), 3)
                self.assertEqual(len({e['track_id'] for e in events}), 3)
                self.assertEqual(gate.peak_tracks, 3)
                self.assertEqual(sum(e['quality']['acceptable'] for e in events), 1)
                self.assertEqual(sum('blurred' in e['quality']['reasons'] for e in events), 2)
                self.assertEqual(sum('too_dark' in e['quality']['reasons'] for e in events), 1)
                self.assertTrue(all(min(e['image'].size) >= 250 for e in events))

    def test_display_speed_and_duration_do_not_depend_on_render_resolution(self):
        image_id = self.add_photo(detail_photo())
        for lanes, tile in ((1, 220), (2, 190), (3, 128)):
            for speed in (120, 380, 600):
                source = PhotoConveyor(self.root, [image_id] * 5, {'simultaneous': lanes, 'speed': speed, 'rounds': 2})
                self.assertAlmostEqual(source.travel_seconds, (960 + tile + 60) / speed)
                self.assertEqual(source.description()['expected_products'], 10)
                self.assertAlmostEqual(source.render_speed / source.width, speed / 960)

    def test_session_uses_source_crop_and_records_resolution_provenance(self):
        sharp = detail_photo()
        ids = [self.add_photo(im) for im in
               (sharp, sharp.filter(ImageFilter.GaussianBlur(12)), Image.new('RGB', (640, 640), '#080808'))]
        resident = Resident()
        service = InspectionService(self.root, resident)
        self.addCleanup(service.close)
        session = VideoSession(self.root, resident, service, '', '', 'fixture')
        self.addCleanup(session.stop)
        start = session.start({'kind': 'photos', 'image_ids': ids, 'photos': {'simultaneous': 3, 'speed': 600}})
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            state = session.snapshot()
            if state['status'] == 'completed' and state['gpu_calls']['completed'] == 1:
                break
            self.assertNotEqual(state['status'], 'error', state)
            time.sleep(.1)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(resident.calls, 1)
        self.assertEqual(state['gpu_calls'], {'submitted': 1, 'completed': 1, 'pending': 0, 'failed': 0})
        rows = service.store.query(session_id=start['id'])['items']
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(r['decision'] == 'review' for r in rows), 2)
        for row in rows:
            self.assertEqual(row['metadata']['source_size'], [1920, 1080])
            self.assertEqual(row['metadata']['photo_source_revision'], PhotoConveyor.revision)
            self.assertGreaterEqual(min(row['quality']['width'], row['quality']['height']), 250)
            self.assertEqual(row['quality']['thresholds']['min_sharpness'], 18)
        with Image.open(io.BytesIO(session.frame())) as preview:
            self.assertEqual(preview.size, (960, 540))
        config = next(s['config'] for s in service.store.summary()['sessions'] if s['id'] == start['id'])
        self.assertEqual(config['source']['source_size'], [1920, 1080])


if __name__ == '__main__':
    unittest.main()
