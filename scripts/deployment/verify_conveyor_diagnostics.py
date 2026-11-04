"""Exercise real GPU calls for multiple products simultaneously inside the ROI."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8033')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/verification/upgrade_v3_2/concurrent_gpu.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def call(path, body=None, raw=None):
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        with urlopen(Request(args.url + path, data=data), timeout=20) as response:
            return json.load(response)
    before = call('/api/runtime')
    assert before['state'] == 'ready'
    assert before['process_hardware']['rss_mib'] > 0
    samples = json.loads((ROOT / 'demo_samples/manifest.json').read_text(encoding='utf-8'))['cases'][:3]
    ids = [call('/api/conveyor/images', raw=(ROOT / row['path']).read_bytes())['image_id'] for row in samples]
    cases = []
    try:
        for name, simultaneous, direction in (
                ('three_parallel', 3, 'right_to_left'),
                ('two_parallel_reverse', 2, 'left_to_right')):
            old = call('/api/runtime')
            session = call('/api/video/start', {'kind': 'photos', 'image_ids': ids,
                'photos': {'simultaneous': simultaneous, 'speed': 360}, 'gate': {'direction': direction}})
            print('Running:', name, session['id'], flush=True)
            benchmark = None
            if name == 'three_parallel':
                benchmark = subprocess.Popen([sys.executable, str(ROOT / 'scripts/deployment/benchmark_preview.py'),
                    '--url', args.url, '--output', str(args.output.parent / 'preview_cadence.json')], stdout=subprocess.DEVNULL)
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                state = call('/api/video')
                if state['status'] == 'error':
                    raise RuntimeError(state.get('error', state))
                if state['status'] == 'completed' and not state.get('gpu_calls', {}).get('pending'):
                    break
                time.sleep(.15)
            else:
                raise TimeoutError(name)
            if benchmark:
                assert benchmark.wait(timeout=8) == 0
            history = call('/api/history?session_id=' + session['id'])['items']
            new = call('/api/runtime')
            assert len(history) == 3 and len({r['track_id'] for r in history}) == 3, (name, history)
            assert old['generation'] == new['generation'] == before['generation']
            expected_gpu = 3
            assert state['gpu_calls']['submitted'] == state['gpu_calls']['completed'] == expected_gpu, (name, state['gpu_calls'], [(r['decision'], r['quality']) for r in history])
            assert new['product_calls'] - old['product_calls'] == expected_gpu
            assert state['peak_simultaneous_tracks'] >= simultaneous
            if simultaneous == 3:
                assert len({r['metadata']['trigger_pts_ns'] for r in history}) == 1
            assert all(r['metadata']['gate']['roi_overlap'] >= .8 for r in history)
            assert state['stage_counts']['grayscale'] == state['analysis_steps']
            assert 0 < state['stage_counts']['morphology'] < state['analysis_steps']
            for stage in ('threshold', 'morphology', 'objects', 'montage'):
                with urlopen(args.url + state['debug']['urls'][stage], timeout=5) as response:
                    data = response.read()
                with Image.open(io.BytesIO(data)) as image:
                    assert image.size == ((640, 636) if stage == 'montage' else (320, 180))
                if stage == 'montage':
                    (args.output.parent / (name + '_steps.jpg')).write_bytes(data)
            cases.append({'name': name, 'status': 'PASS', 'session_id': session['id'], 'products': len(history),
                'peak_simultaneous_tracks': state['peak_simultaneous_tracks'], 'gpu_calls': state['gpu_calls'],
                'worker_product_calls_delta': new['product_calls'] - old['product_calls'],
                'analysis_steps': state['analysis_steps'], 'stage_counts': state['stage_counts'],
                'rows': history})
            print('PASS:', name, state['gpu_calls'], flush=True)
        result = {'status': 'PASS', 'scope': 'Functional tests on rendered validation-photo tiles; no real camera or multi-object accuracy claim',
            'engine_sha256': before['engine_sha256'], 'worker_generation': before['generation'],
            'preview': json.loads((args.output.parent / 'preview_cadence.json').read_text(encoding='utf-8')),
            'process_hardware': new['process_hardware'], 'cases': cases}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'PASS', 'cases': len(cases), 'preview_fps': result['preview']['received_fps']}, indent=2))
    finally:
        call('/api/video/stop', {})


if __name__ == '__main__':
    main()
