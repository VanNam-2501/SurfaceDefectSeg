"""Verify a complete selected-photo replay against its schedule, outside the tracker.

Reference positions are used only here to evaluate returned crop metadata.
No reference coordinates or object IDs are sent to VideoGate.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.photo_conveyor import PhotoConveyor


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8035')
    parser.add_argument('--selection', type=Path, default=ROOT / 'artifacts/inspection/conveyor_selection.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/inspection/tracking_audit/live_gpu.json')
    parser.add_argument('--public-summary', type=Path)
    parser.add_argument('--speed', type=int, default=600)
    parser.add_argument('--simultaneous', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--direction', choices=['right_to_left', 'left_to_right'], default='right_to_left')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--expect-reviews', type=int, help='Fail if the number of quality reviews differs')
    args = parser.parse_args()
    rows = json.loads(args.selection.read_text(encoding='utf-8'))
    assert 1 <= len(rows) <= 50
    def call(path, body=None, raw=None):
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        with urlopen(Request(args.url + path, data=data), timeout=20) as response:
            return json.load(response)
    before = call('/api/runtime')
    assert before['state'] == 'ready'
    ids = [call('/api/conveyor/images', raw=(ROOT / 'artifacts/inspection/conveyor_inputs' / (row['image_id'] + '.jpg')).read_bytes())['image_id'] for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    try:
        for simultaneous in args.simultaneous:
            spec = {'simultaneous': simultaneous, 'speed': args.speed, 'rounds': args.rounds}
            reference = PhotoConveyor(ROOT, [r['image_id'] for r in rows], spec, args.direction)
            expected = reference.expected_products
            old = call('/api/runtime')
            session = call('/api/video/start', {'kind': 'photos', 'image_ids': ids, 'photos': spec,
                'gate': {'direction': args.direction}})
            print(f'Started {len(rows)} photos, {simultaneous} simultaneous, {args.speed}px/s', flush=True)
            benchmark = None
            preview_path = args.output.parent / f'preview_{simultaneous}.json'
            benchmark = subprocess.Popen([sys.executable, str(ROOT / 'scripts/deployment/benchmark_preview.py'),
                '--url', args.url, '--output', str(preview_path)], stdout=subprocess.DEVNULL)
            deadline = time.monotonic() + reference.duration + 30
            next_progress = 10
            while time.monotonic() < deadline:
                state = call('/api/video')
                if state['status'] == 'error':
                    raise RuntimeError(state.get('error', state))
                count = sum(state.get('counts', {}).values())
                if count >= next_progress:
                    print(f'  {simultaneous} simultaneous: {count}/{expected} records', flush=True)
                    next_progress += 10
                if state['status'] == 'completed' and not state.get('gpu_calls', {}).get('pending'):
                    break
                time.sleep(.25)
            else:
                raise TimeoutError('Replay did not finish')
            assert benchmark.wait(timeout=8) == 0
            history = call('/api/history?session_id=' + session['id'] + '&limit=500')['items']
            assigned, crop_scores = defaultdict(list), []
            for row in history:
                meta = row['metadata']
                elapsed = meta['source_pts_ns'] / 1e9 - reference.initial_seconds
                group = int(elapsed / reference.travel_seconds)
                assert 0 <= group < len(reference.groups)
                x = reference.width - elapsed % reference.travel_seconds * reference.render_speed
                if args.direction == 'left_to_right':
                    x = reference.width - reference.tile_size - x
                x = round(x)
                boxes = [(x, reference.positions[lane], x + reference.tile_size, reference.positions[lane] + reference.tile_size)
                    for lane in range(len(reference.groups[group]))]
                scores = [iou(meta['crop_box'], box) for box in boxes]
                lane = max(range(len(scores)), key=scores.__getitem__)
                assigned[group * simultaneous + lane].append(row['track_id'])
                crop_scores.append(scores[lane])
                assert row['identity']['contract_sha256'] == before['contract_sha256']
            assert len(history) == expected and len(assigned) == expected, (len(history), len(assigned))
            assert all(len(values) == 1 for values in assigned.values())
            assert min(crop_scores) >= .85, min(crop_scores)
            assert not any('too_small' in r['quality']['reasons'] or 'roi_crossing_unconfirmed' in r['quality']['reasons'] for r in history)
            assert all(r['decision'] in ('good', 'defect', 'review') for r in history)
            gpu = state['gpu_calls']
            reviews = sum(r['decision'] == 'review' for r in history)
            if args.expect_reviews is not None:
                assert reviews == args.expect_reviews, (reviews, args.expect_reviews)
            assert gpu['submitted'] == gpu['completed'] == expected - reviews
            assert not gpu['failed'] and not gpu['pending']
            assert state['peak_simultaneous_tracks'] <= simultaneous, state['peak_simultaneous_tracks']
            new = call('/api/runtime')
            assert new['generation'] == before['generation'] and new['product_calls'] - old['product_calls'] == gpu['completed']
            preview = json.loads(preview_path.read_text(encoding='utf-8'))
            case = {'status': 'PASS', 'simultaneous': simultaneous, 'direction': args.direction,
                'speed_preview_pixels_per_second': args.speed, 'rounds': args.rounds,
                'reference_products': expected, 'records': len(history),
                'missed_products': 0, 'extra_records': 0, 'partial_crops': 0, 'minimum_crop_iou': min(crop_scores),
                'gpu_calls': gpu, 'quality_reviews': reviews, 'quality_reasons': dict(Counter(reason for r in history for reason in r['quality']['reasons'])),
                'preview_received_fps': preview['received_fps'], 'peak_simultaneous_tracks': state['peak_simultaneous_tracks'],
                'source': reference.description(), 'quality_thresholds': state['settings']['quality'],
                'sharpness_range': [min(r['quality']['sharpness'] for r in history), max(r['quality']['sharpness'] for r in history)]}
            cases.append(case)
            print(json.dumps(case), flush=True)
        result = {'status': 'PASS', 'scope': 'Functional rendered-photo tracking test on selected local images; not a camera or classification accuracy benchmark',
            'engine_sha256': before['engine_sha256'], 'cases': cases}
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        if args.public_summary:
            args.public_summary.parent.mkdir(parents=True, exist_ok=True)
            args.public_summary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    finally:
        call('/api/video/stop', {})


if __name__ == '__main__':
    main()
