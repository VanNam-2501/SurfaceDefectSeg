"""Paced HTTP load against the local B1 app. No camera or browser UI assumed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import httpx
import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/b1_deployment'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--clients',type=int,default=1);parser.add_argument('--rate',type=float,default=10)
    args=parser.parse_args()
    frame=pd.read_csv(ROOT/'artifacts/straw_global_local/data_v1/val.csv').groupby('label',sort=True).head(4)
    url='http://127.0.0.1:8032'
    with httpx.Client(timeout=30,trust_env=False) as client:
        health=client.get(url+'/health');health.raise_for_status()
        size=health.json()['size']
        payloads=[(OUT/f'input_cache/{size}_q90'/f'{i}.jpg').read_bytes() for i in frame.sample_id]
        process=psutil.Process(health.json()['pid'])
        initial_rss=process.memory_info().rss
        for i in range(10):
            reply=client.post(url+'/predict',files={'file':('warm.jpg',payloads[i%len(payloads)],'image/jpeg')},data={'frame_id':i})
            reply.raise_for_status()
    def worker(index):
        records=[];started=time.perf_counter();deadline=started+args.seconds;next_at=started;i=0
        with httpx.Client(timeout=5,trust_env=False) as client:
            while time.perf_counter()<deadline:
                before=time.perf_counter();frame_id=index*1_000_000+i
                try:
                    response=client.post(url+'/predict',files={'file':('frame.jpg',payloads[i%len(payloads)],'image/jpeg')},data={'frame_id':frame_id})
                    elapsed=(time.perf_counter()-before)*1000
                    data=response.json()
                    if response.status_code==200: assert data['frame_id']==frame_id,'Mismatched frame ID'
                    records.append(dict(status=response.status_code,latency_ms=elapsed,
                        server_ms=data.get('timings_ms',{}).get('server_request'),client=index,
                        server_rss_mib=process.memory_info().rss/2**20))
                except Exception as exc:
                    records.append(dict(status='error',error=type(exc).__name__,client=index))
                i+=1;next_at+=1/args.rate
                delay=next_at-time.perf_counter()
                if delay>0:time.sleep(delay)
                else:next_at=time.perf_counter() # no artificial catch-up burst
        return records,time.perf_counter()-started
    with ThreadPoolExecutor(max_workers=args.clients) as executor:
        outputs=list(executor.map(worker,range(args.clients)))
    records=[row for rows,_ in outputs for row in rows];success=[r for r in records if r['status']==200]
    latency=np.asarray([r['latency_ms'] for r in success]);duration=max(d for _,d in outputs)
    report=dict(seconds=duration,clients=args.clients,target_rate_per_client=args.rate,requests=len(records),
        successful=len(success),busy_429=sum(r['status']==429 for r in records),errors=sum(r['status'] not in (200,429) for r in records),
        successful_per_second=len(success)/duration,deadline_ms=200,
        deadline_misses=sum(r['latency_ms']>200 for r in success),
        http_ms={f'p{q}':float(np.percentile(latency,q)) for q in (50,95,99)} if len(latency) else {},
        server_initial_rss_mib=initial_rss/2**20,server_final_rss_mib=process.memory_info().rss/2**20,
        server_peak_sampled_rss_mib=max((r.get('server_rss_mib',0) for r in records),default=0),
        health=health.json(),note='HTTP request/response with existing JPEG bytes. Excludes camera, browser encode and rendering. No training may run concurrently.')
    report['per_client']=[{'client':i,'successful':sum(r['client']==i and r['status']==200 for r in records),
        'busy_429':sum(r['client']==i and r['status']==429 for r in records)} for i in range(args.clients)]
    target=OUT/f'http_{args.clients}client.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(records).to_csv(target.with_suffix('.csv'),index=False)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
