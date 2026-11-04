"""Measure the running local B1 web endpoint with deployment JPEG payloads."""
import json
from pathlib import Path
import time

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    cases = json.loads((ROOT/'artifacts/b1_deployment/frame640/index.json').read_text())['cases'][:30]
    timings=[]
    with httpx.Client(base_url='http://127.0.0.1:8032',timeout=10) as client:
        health=client.get('/health').json()
        if health['size']!=512 or health['decision_threshold'] is None:
            raise RuntimeError('The active web endpoint is not the verified B1 512 policy')
        for j,case in enumerate(cases):
            payload=(ROOT/'artifacts/b1_deployment/input_cache/512_q90'/f"{case['sample_id']}.jpg").read_bytes()
            start=time.perf_counter()
            response=client.post('/predict',files={'file':('frame.jpg',payload,'image/jpeg')},data={'frame_id':str(j)})
            elapsed=(time.perf_counter()-start)*1000
            response.raise_for_status()
            data=response.json()
            if data['frame_id']!=j:
                raise RuntimeError('Frame ID changed in HTTP response')
            timings.append(elapsed)
    report={'source':'local HTTP localhost, static JPEG512 validation proxy, 1 client',
            'health':health,'n':len(timings),
            'roundtrip_p50_ms':float(np.percentile(timings,50)),
            'roundtrip_p95_ms':float(np.percentile(timings,95)),
            'roundtrip_p99_ms':float(np.percentile(timings,99)),
            'deadline_200ms_misses':int(sum(t>200 for t in timings))}
    target=ROOT/'artifacts/b1_deployment/web512_http_probe.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
