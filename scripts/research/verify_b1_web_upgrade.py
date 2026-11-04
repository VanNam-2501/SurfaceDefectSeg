"""Replay the deployment JPEG proxy frames through the actual FastAPI handler.

This exercises upload decoding, GPU inference and the active decision policy.
It is a static validation replay, not a live camera validation.
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['STRAW_WEB_CONFIG'] = 'straw_realtime_web.json'
from apps.web_demo.backend.b1_app import app
from src.straw_global_local.data import CLASSES
from src.straw_global_local.postprocess import counts


def main():
    cases = json.loads((ROOT/'artifacts/b1_deployment/frame640/index.json').read_text())['cases']
    predictions, elapsed, examples = [], [], []
    with TestClient(app) as client:
        health = client.get('/health').json()
        for j, case in enumerate(cases):
            path = ROOT/'artifacts/b1_deployment/input_cache/512_q90'/f"{case['sample_id']}.jpg"
            start = time.perf_counter()
            with path.open('rb') as image:
                response = client.post('/predict', files={'file': ('frame.jpg', image, 'image/jpeg')},
                                       data={'frame_id': str(j)})
            elapsed.append((time.perf_counter()-start)*1000)
            response.raise_for_status()
            data = response.json()
            predictions.append(CLASSES.index(data['class']))
            if CLASSES.index(data['class']) != case['label']:
                examples.append({'sample_id': case['sample_id'], 'expected': CLASSES[case['label']],
                                 'predicted': data['class'], 'raw': data['raw_model_class'],
                                 'defect_score': data['defect_score']})
    labels = np.array([r['label'] for r in cases])
    report = {'source': 'static validation JPEG512 q90 proxy, FastAPI TestClient in process',
              'checkpoint_sha256': health['checkpoint_sha256'], 'size': health['size'],
              'threshold': health['decision_threshold'], 'n': len(cases),
              **counts(labels, np.array(predictions)),
              'latency_ms': {'p50': float(np.percentile(elapsed, 50)),
                             'p95': float(np.percentile(elapsed, 95)),
                             'p99': float(np.percentile(elapsed, 99))},
              'errors': examples}
    target = ROOT/'straw_resume/error_analysis/b1_postprocess/web512_api_replay.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
