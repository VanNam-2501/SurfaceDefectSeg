"""Portable runtime commands, engine cache invalidation and removed web routes."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts/test_tmp').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'apps/stream_demo'))
from apps.stream_demo.runtime_backend import host_command
from apps.stream_demo.container_boot import build_key, cached_engine_valid
from apps.stream_demo.serving_contract import file_sha256
from apps.stream_demo import preview_web
from apps.stream_demo.performance_monitor import PerformanceMonitor


class FakeResident:
    identity = {'contract_sha256': 'fixture', 'engine_sha256': 'fixture', 'checkpoint_sha256': 'fixture'}
    def __init__(self, *args, **kwargs): pass
    def status(self): return dict(state='ready', **self.identity)
    def hardware_status(self): return {}


class PortableRuntimeTests(unittest.TestCase):
    def test_native_command_never_invokes_windows_or_docker(self):
        with patch.dict('os.environ', {'STRAW_RUNTIME': 'native'}):
            self.assertEqual(host_command('unused', 'nvidia-smi', '--query'), ['nvidia-smi', '--query'])

    def test_windows_backend_keeps_explicit_distro(self):
        with patch.dict('os.environ', {'STRAW_RUNTIME': ''}):
            self.assertEqual(host_command('fixture', 'bash', 'capture.sh'),
                ['wsl.exe', '-d', 'fixture', '-u', 'root', '--', 'bash', 'capture.sh'])

    def test_engine_is_invalidated_for_gpu_driver_runtime_or_model_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp') as folder:
            out = Path(folder)
            (out / 'b1_rawrgb512.onnx').write_bytes(b'fixture-onnx')
            (out / 'manifest.json').write_text(json.dumps({'size': 512, 'onnx_sha256': file_sha256(out / 'b1_rawrgb512.onnx')}))
            key = build_key(out, 'GPU A, driver A, uuid A', '10.fixture')
            engine = out / 'b1_rtx3050_fp16.engine'
            engine.write_bytes(b'fixture-engine')
            (out / 'target_engine.json').write_text(json.dumps({'build_key': key, 'engine_sha256': file_sha256(engine)}))
            self.assertTrue(cached_engine_valid(out, key))
            for field in ('gpu_identity', 'tensorrt_version', 'onnx_sha256', 'precision'):
                changed = dict(key, **{field: 'changed'})
                self.assertFalse(cached_engine_valid(out, changed), field)
            engine.write_bytes(b'corrupted-engine')
            self.assertFalse(cached_engine_valid(out, key))
            (out / 'target_engine.json').write_text('null')
            self.assertFalse(cached_engine_valid(out, key))
            (out / 'b1_rawrgb512.onnx').write_bytes(b'wrong-model')
            with self.assertRaises(ValueError): build_key(out, 'GPU A', '10.fixture')

    def test_http_has_no_research_ui_or_endpoints_and_keeps_origin_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/test_tmp') as folder:
            root = Path(folder)
            with patch.object(preview_web, 'ROOT', root), patch.object(preview_web, 'ResidentInference', FakeResident):
                handler = preview_web.make_handler(root / 'preview', '', '')
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            try:
                with urlopen(base + '/') as response: html = response.read().decode()
                self.assertNotIn('data-nav="research"', html)
                self.assertNotIn('data-view="research"', html)
                performance = html.split('<section data-view="performance"', 1)[1].split('</section>', 1)[0]
                video = html.split('<section data-view="video"', 1)[1].split('</section>', 1)[0]
                self.assertIn('id="cpuChart"', performance)
                self.assertNotIn('id="cpuSteps"', video)
                with urlopen(base + '/performance') as response: self.assertEqual(response.status, 200)
                with urlopen(base + '/api/performance') as response: first = json.load(response)
                time.sleep(1.1)
                with urlopen(base + '/api/performance') as response: later = json.load(response)
                self.assertGreater(len(later['samples']), len(first['samples']))
                self.assertEqual(later['sample_interval_seconds'], 1)
                for route in ('/research', '/api/research', '/api/research/files/fixture', '/research_report.html'):
                    with self.assertRaises(HTTPError) as caught: urlopen(base + route)
                    self.assertEqual(caught.exception.code, 404)
                with patch.dict('os.environ', {'STRAW_RUNTIME': 'native'}):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(Request(base + '/api/video/stop', data=b'{}', headers={'Origin': 'https://other.example'}))
                    self.assertEqual(caught.exception.code, 403)
            finally:
                handler.workspace_api.close()
                server.shutdown()
                server.server_close()
                thread.join()

    def test_performance_counts_only_observed_tracks_and_does_not_repeat_cpu_frames(self):
        class Video:
            lock = threading.Lock()
            state = {'id': 'fixture', 'status': 'running', 'analysis_steps': 12, 'host_gate_ms': 49,
                'tracks': [{'confirmed': True, 'observed': True}, {'confirmed': True, 'observed': False},
                           {'confirmed': False, 'observed': True}]}
        class Store:
            def session_gpu_counts(self, session_id):
                return {'submitted': 7, 'completed': 6, 'pending': 1, 'failed': 0}
        class Metrics:
            def snapshot(self): return {'rss_mib': 56.9, 'cpu_one_core_percent': 15}
        sampler = PerformanceMonitor(Video(), Store(), Metrics(), limit=3)
        sampler.close()
        sampler.sample_once()
        point = sampler.snapshot()['samples'][-1]
        self.assertEqual(point['active_tracks'], 1)
        self.assertEqual(point['predicted_tracks'], 1)
        self.assertEqual(point['gpu_pending'], 1)
        self.assertEqual(point['ram_mib'], 56.9)
        self.assertIsNone(point['cpu_frame_ms'])
        for _ in range(8): sampler.sample_once()
        self.assertEqual(len(sampler.snapshot()['samples']), 3)
        sampler.video.state = dict(sampler.video.state, id='new-session', analysis_steps=1, host_gate_ms=21)
        sampler.sample_once()
        self.assertEqual(len(sampler.snapshot()['samples']), 1)
        self.assertEqual(sampler.snapshot()['samples'][0]['cpu_frame_ms'], 21)


if __name__ == '__main__':
    unittest.main()
