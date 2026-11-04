"""Pixel-bound masks, concurrent ROI objects and real telemetry."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from unittest.mock import patch
import uuid
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts/test_tmp').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from apps.stream_demo.video_tracking import VideoGate, GateConfig
from apps.stream_demo.photo_conveyor import PhotoConveyor, PhotoSettings
from apps.stream_demo.process_metrics import ProcessMetrics
from apps.stream_demo import preview_web


def tile(size=50):
    gray = np.random.default_rng(42).integers(160, 255, (size, size), dtype=np.uint8)
    return Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2))


class ConcurrentTrackingTests(unittest.TestCase):
    def gate(self):
        gate = VideoGate(GateConfig(calibration_frames=3))
        for i in range(3):
            gate.process(Image.new('RGB', (320, 180), (110, 110, 110)), i)
        return gate

    def test_mask_and_opening_show_actual_single_pixel_noise_removal(self):
        gate = self.gate()
        source = Image.new('RGB', (320, 180), (110, 110, 110))
        source.putpixel((30, 30), (255, 255, 255))
        state = gate.process(source, 100000000)
        debug = gate.debug_images()
        self.assertEqual(debug['threshold'].getpixel((30, 30)), (255, 255, 255))
        self.assertEqual(debug['morphology'].getpixel((30, 30)), (0, 0, 0))
        self.assertEqual(state['stage_counts']['grayscale'], 4)
        self.assertEqual(state['stage_counts']['threshold'], 1)
        self.assertEqual(state['stage_counts']['morphology'], 1)
        self.assertEqual(state['detected_components'], 0)

    def test_three_products_cross_in_same_frame_once_each(self):
        gate = self.gate()
        events = []
        for index, x in enumerate(range(280, -60, -7)):
            image = Image.new('RGB', (320, 180), (110, 110, 110))
            for y in (10, 65, 120):
                image.paste(tile(), (x, y))
            events.extend(gate.process(image, index * 100000000)['events'])
        self.assertEqual(len(events), 3)
        self.assertEqual(len({e['track_id'] for e in events}), 3)
        self.assertEqual(len({e['trigger_pts_ns'] for e in events}), 1)
        self.assertEqual(gate.peak_tracks, 3)
        self.assertTrue(all(e['gate']['roi_overlap'] >= .8 for e in events))
        self.assertTrue(all(e['quality']['acceptable'] for e in events))

    def test_photo_conveyor_three_lanes_are_tracked_from_pixels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp') as folder:
            root = Path(folder)
            inputs = root / 'artifacts/inspection/conveyor_inputs'
            inputs.mkdir(parents=True)
            ids = [uuid.uuid4().hex for _ in range(3)]
            for image_id in ids:
                tile(160).save(inputs / (image_id + '.jpg'))
            source = PhotoConveyor(root, ids, {'simultaneous': 3, 'speed': 360})
            gate = VideoGate(GateConfig(**source.gate_settings()))
            events = []
            for frame in range(0, source.frame_count, 3):
                events.extend(gate.process(source.render(frame), round(frame / 30 * 1e9))['events'])
            self.assertEqual(len(events), 3)
            self.assertEqual(gate.peak_tracks, 3)
            self.assertTrue(all(e['quality']['acceptable'] for e in events))

    def test_photo_settings_reject_path_and_nonfinite_or_unbounded_controls(self):
        for spec in ({'simultaneous': 4}, {'speed': float('nan')}, {'speed': 0}, {'rounds': 30}, {'unknown_option': True}):
            with self.assertRaises(ValueError):
                PhotoSettings.parse(spec)
        with self.assertRaises(ValueError):
            PhotoConveyor(ROOT, ['../../secret'])

    def test_process_memory_is_measured_and_scope_is_explicit(self):
        sample = ProcessMetrics().snapshot()
        self.assertGreater(sample['rss_mib'], 0)
        self.assertGreaterEqual(sample['peak_rss_mib'], sample['rss_mib'])
        self.assertIn('Excludes browser', sample['scope'])


class FakeResident:
    identity = {'contract_sha256': 'fixture', 'engine_sha256': 'fixture-engine', 'checkpoint_sha256': 'fixture-checkpoint'}
    def __init__(self, *args, **kwargs):
        self.calls = 0
    def status(self):
        return {'state': 'ready', 'product_calls': self.calls, 'warmup_calls': 0}
    def hardware_status(self):
        return {'telemetry_available': False}
    def submit(self, *args):
        self.calls += 1
        return uuid.uuid4().hex
    def result(self, request_id):
        return {'status': 'completed', 'class': 'good', 'defective': False, 'contract_sha256': 'fixture'}


class ConcurrentHTTPTests(unittest.TestCase):
    def test_uploaded_photos_concurrent_gate_debug_and_gpu_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp') as folder:
            root = Path(folder)
            with patch.object(preview_web, 'ROOT', root), patch.object(preview_web, 'ResidentInference', FakeResident):
                handler = preview_web.make_handler(root / 'preview', '', '')
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            def get(path):
                with urlopen(base + path) as response:
                    return json.load(response)
            def post(path, data):
                raw = data if isinstance(data, bytes) else json.dumps(data).encode()
                with urlopen(Request(base + path, data=raw, method='POST')) as response:
                    return json.load(response)
            try:
                raw = io.BytesIO()
                tile(180).save(raw, format='PNG')
                ids = [post('/api/conveyor/images', raw.getvalue())['image_id'] for _ in range(3)]
                selected = [{'image_id': image_id, 'name': f'photo-{i}.png'} for i, image_id in enumerate(ids)]
                self.assertEqual(post('/api/conveyor/selection', {'items': selected}), {'saved': 3})
                self.assertEqual([item['image_id'] for item in get('/api/conveyor/selection')['items']], ids)
                self.assertEqual(json.loads((root / 'artifacts/inspection/conveyor_selection.json').read_text(encoding='utf-8')), selected)
                self.assertEqual(handler.workspace_api.service.resident.calls, 0)
                state = post('/api/video/start', {'kind': 'photos', 'image_ids': ids, 'photos': {'simultaneous': 3, 'speed': 600}})
                session_id = state['id']
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    state = get('/api/video')
                    if state['status'] == 'completed' and state['gpu_calls']['completed'] == 3:
                        break
                    self.assertNotEqual(state['status'], 'error', state)
                    time.sleep(.1)
                self.assertEqual(state['status'], 'completed')
                self.assertEqual(state['gpu_calls'], {'submitted': 3, 'completed': 3, 'pending': 0, 'failed': 0})
                self.assertEqual(state['peak_simultaneous_tracks'], 3)
                self.assertEqual(sum(state['counts'].values()), 3)
                with urlopen(base + state['debug']['urls']['morphology']) as response:
                    self.assertEqual(Image.open(io.BytesIO(response.read())).size, (320, 180))
                with urlopen(base + state['debug']['urls']['montage']) as response:
                    self.assertEqual(Image.open(io.BytesIO(response.read())).size, (640, 636))
                with self.assertRaises(HTTPError):
                    urlopen(base + state['debug']['urls']['gray'].replace(session_id, '0' * 32))
                self.assertGreater(get('/api/runtime')['process_hardware']['rss_mib'], 0)
            finally:
                handler.workspace_api.close()
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
