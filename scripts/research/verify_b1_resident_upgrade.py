"""Probe the active TensorRT resident ROI worker with labeled static frames."""
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.resident_inference import ResidentInference
from src.straw_global_local.data import CLASSES


def main():
    cases = json.loads((ROOT/'artifacts/b1_deployment/frame640/index.json').read_text())['cases'][:10]
    resident = ResidentInference(ROOT, '', '', artifact='b1_stream_512')
    if resident.status()['state'] != 'ready':
        raise RuntimeError('TensorRT resident worker is not ready')
    rows = []
    for case in cases:
        request_id = resident.submit(ROOT/'artifacts/b1_deployment/frame640'/f"{case['sample_id']}.jpg",
                                     f"upgrade-{uuid.uuid4().hex}", {'sample_id':case['sample_id']})
        deadline = time.monotonic() + 30
        response = resident.result(request_id)
        while response.get('status') == 'pending' and time.monotonic() < deadline:
            time.sleep(.02)
            response = resident.result(request_id)
        if response.get('status') != 'completed':
            raise RuntimeError(f'Resident inference failed: {response}')
        rows.append({'sample_id':case['sample_id'], 'expected':CLASSES[case['label']],
                     'predicted':response['class'],
                     'tensor_ms':response['tensor_transfer_inference_ms'],
                     'event_to_result_ms':response['server_event_to_result_ms']})
    report = {'n':len(rows), 'correct':sum(r['expected']==r['predicted'] for r in rows),
              'tensor_p50_ms':float(np.median([r['tensor_ms'] for r in rows])),
              'tensor_p95_ms':float(np.percentile([r['tensor_ms'] for r in rows],95)),
              'event_to_result_p95_ms':float(np.percentile([r['event_to_result_ms'] for r in rows],95)),
              'rows':rows}
    target=ROOT/'artifacts/b1_stream_512/resident_probe.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
