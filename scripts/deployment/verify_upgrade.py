"""Functional replay validation of GPU decode, tracking, quality review and history."""
from pathlib import Path
import argparse
import json
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8033')
    parser.add_argument('--rtsp')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/verification/upgrade_v2/video_file.json')
    args=parser.parse_args()
    def call(path,body=None,raw=None):
        data=raw if raw is not None else json.dumps(body).encode() if body is not None else None
        with urlopen(Request(args.url+path,data=data,headers={'Content-Type':'application/json'}),timeout=20) as response:
            return json.load(response)
    health=call('/health')
    if not health['runtime_verified']:raise RuntimeError('TensorRT is not ready')
    before=call('/api/runtime')
    spec={'kind':'rtsp','uri':args.rtsp} if args.rtsp else {'kind':'demo'}
    if args.upload:
        upload=call('/api/video/upload',raw=(ROOT/'demo_samples/conveyor_replay.mp4').read_bytes())
        spec={'kind':'file','upload_id':upload['upload_id']}
    session=call('/api/video/start',spec)
    deadline=time.monotonic()+65
    states=[]
    try:
        while time.monotonic()<deadline:
            state=call('/api/video')
            if not states or states[-1]!=state['status']:
                states.append(state['status']);print('Video state:',state['status'],flush=True)
            if state['status']=='error':raise RuntimeError(state.get('error','Video failed'))
            counts=state.get('counts',{})
            if args.rtsp and sum(counts.values())>=5 and not counts.get('pending'):
                state=call('/api/video/stop',{});break
            if state['status']=='completed' and not counts.get('pending'):break
            time.sleep(.25)
        else:raise TimeoutError('Replay session did not finish')
        history=call('/api/history?session_id='+session['id'])
        rows=history['items']
        if len(rows)!=5 or len({r['track_id'] for r in rows})!=5:
            raise AssertionError(f'Expected 5 unique products, found {len(rows)}')
        reviews=[r for r in rows if r['decision']=='review']
        classified=[r for r in rows if r['decision'] in ('good','defect')]
        if len(reviews)!=2 or len(classified)!=3:raise AssertionError('Expected three GPU inspections and two quality reviews')
        if not any('too_dark' in r['quality']['reasons'] for r in reviews):raise AssertionError('Dark sample not reviewed')
        if not any('blurred' in r['quality']['reasons'] for r in reviews):raise AssertionError('Blurred sample not reviewed')
        if any(r['identity']['contract_sha256']!=health['contract_sha256'] for r in rows):raise AssertionError('Wrong engine identity')
        if any(r['metadata']['source_pts_ns']>r['metadata']['trigger_pts_ns'] for r in rows):raise AssertionError('Selected frame is newer than its trigger')
        after=call('/api/runtime')
        result={'status':'PASS','scope':'Synthetic validation-photo conveyor; functional test only, not camera accuracy',
            'source_kind':'file_upload' if args.upload else 'rtsp_replay' if args.rtsp else 'file_replay','session_id':session['id'],
            'products':len(rows),'gpu_inspections':len(classified),'quality_reviews':len(reviews),
            'states':states,'frames_received':state['frames_received'],
            'worker_generation_unchanged':before['generation']==after['generation'],
            'product_calls_delta':after['product_calls']-before['product_calls'],
            'engine_sha256':health['engine_sha256'],'rows':rows}
        assert result['worker_generation_unchanged'] and result['product_calls_delta']==3
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    finally:call('/api/video/stop',{})


if __name__=='__main__':main()
