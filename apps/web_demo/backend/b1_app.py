"""Dedicated single-stage B1 demo. Run one worker; never silently use other weights."""
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,UnidentifiedImageError
import io
import torch
from fastapi import FastAPI,File,UploadFile,HTTPException,Request,Form
from fastapi.responses import HTMLResponse,FileResponse
from pydantic import BaseModel,Field
from src.straw_global_local.data import CLASSES,file_hash
from src.straw_global_local.deployment import TorchRuntime,decode,softmax
from src.straw_global_local.postprocess import predict_from_probabilities


@asynccontextmanager
async def lifespan(app):
    config_name=os.environ.get('STRAW_WEB_CONFIG','straw_realtime.json')
    if config_name not in ('straw_realtime.json','straw_realtime_web.json'):
        raise RuntimeError('Unsupported B1 web configuration')
    config=json.loads((ROOT/'configs'/config_name).read_text(encoding='utf-8'))
    checkpoint=ROOT/config['checkpoint']
    if file_hash(checkpoint)!=config['checkpoint_sha256']:
        raise RuntimeError('Deployment checkpoint differs from validated checkpoint')
    policy=None
    if config.get('decision_policy'):
        policy=json.loads((ROOT/config['decision_policy']).read_text(encoding='utf-8'))
        if (policy.get('schema')!=1 or policy.get('variant')!='B1' or
            policy.get('checkpoint_sha256')!=config['checkpoint_sha256'] or
            policy.get('input_size')!=config['size'] or
            policy.get('temperature')!=config['temperature']):
            raise RuntimeError('Decision policy does not match the active B1 checkpoint or preprocessing')
    torch.set_num_threads(4)
    app.state.config=config
    app.state.decision_policy=policy
    # Warm and serve on the same dedicated thread. CUDA/cuDNN first-use costs
    # must not be deferred to an arbitrary request thread after readiness.
    app.state.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='b1-inference')
    def initialize():
        runtime=TorchRuntime(checkpoint,size=config['size'],precision=config['precision'],
            channels_last=config['channels_last'],autotune=config['autotune'],device=config['device'])
        for _ in range(10): runtime.run(np.zeros((1,3,config['size'],config['size']),np.float32))
        return runtime
    app.state.runtime=await asyncio.get_running_loop().run_in_executor(app.state.worker,initialize)
    # A single owner performs GPU inference; reject overflow instead of building a stale queue.
    app.state.busy=asyncio.Lock()
    app.state.inflight=None
    app.state.admitted=0
    app.state.checkpoint_hash=file_hash(checkpoint)
    yield
    app.state.worker.shutdown(wait=True)
    del app.state.runtime


app=FastAPI(title='B1 Straw Inspection',lifespan=lifespan)


@app.middleware('http')
async def timing(request,call_next):
    request.state.started=time.perf_counter()
    return await call_next(request)


@app.get('/health')
def health():
    cfg=app.state.config
    return {'ready':True,'model':'B1','pid':os.getpid(),'checkpoint_sha256':app.state.checkpoint_hash,
        'size':cfg['size'],'precision':cfg['precision'],'device':cfg['device'],
        'decision_threshold':app.state.decision_policy['threshold'] if app.state.decision_policy else None,
        'classes':CLASSES,'camera_jpeg_quality':cfg['camera_jpeg_quality'],'camera_target_fps':cfg['camera_target_fps'],
        'max_waiting_requests':1,'queue_wait_budget_ms':50}


def inspect(raw):
    cfg=app.state.config
    before=time.perf_counter()
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width*opened.height>16_000_000: raise ValueError('Image is too large')
        x=decode(raw,cfg['size'])
    except (ValueError,UnidentifiedImageError,OSError) as exc:
        raise HTTPException(422,'Invalid image or image exceeds 16 million pixels') from exc
    ready=time.perf_counter()
    logits=app.state.runtime.run(x)
    if not np.isfinite(logits).all(): raise HTTPException(503,'Nonfinite model output; prediction unavailable')
    inferred=time.perf_counter()
    probability=softmax(logits,cfg['temperature'])[0]
    raw_index=int(probability.argmax())
    policy=getattr(app.state,'decision_policy',None)
    index=int(predict_from_probabilities(probability[None],policy['threshold'])[0]) if policy else raw_index
    result={'class':CLASSES[index],'defective':index!=0,
        'raw_model_class':CLASSES[raw_index], 'defect_score':float(1-probability[0]),
        'decision_threshold':policy['threshold'] if policy else None,
        'probabilities':dict(zip(CLASSES,probability.tolist())),
        'confidence_calibrated':False,
        'timings_ms':{'preprocess':1000*(ready-before),'infer_with_transfer':1000*(inferred-ready)}}
    result['timings_ms']['postprocess']=1000*(time.perf_counter()-inferred)
    return result


@app.post('/predict')
async def predict(request:Request,file:UploadFile=File(...),frame_id:int=Form(0)):
    pending=getattr(app.state,'inflight',None)
    # At most one active request and one waiter. No await occurs between
    # capacity check and reservation, so event-loop admission is atomic.
    if getattr(app.state,'admitted',0)>=2 or (not app.state.busy.locked() and pending is not None and not pending.done()):
        raise HTTPException(429,'Inference capacity full; send a newer frame later')
    app.state.admitted=getattr(app.state,'admitted',0)+1
    acquired=False
    queue_started=time.perf_counter()
    try:
        try:
            async with asyncio.timeout(.05):
                await app.state.busy.acquire()
            acquired=True
            pending=getattr(app.state,'inflight',None)
            if pending is not None and not pending.done():
                # A cancelled owner can leave a GPU kernel running. Do not put
                # another executor job behind it without a bounded wait.
                remaining=.05-(time.perf_counter()-queue_started)
                if remaining<=0:raise asyncio.TimeoutError
                try:await asyncio.wait_for(asyncio.shield(pending),timeout=remaining)
                except asyncio.TimeoutError:raise
                except Exception:pass
        except asyncio.TimeoutError as exc:
            raise HTTPException(429,'Inference wait exceeded 50 ms; send a fresh frame') from exc
        queue_ms=1000*(time.perf_counter()-queue_started)
        raw=await file.read(4*1024*1024+1)
        if len(raw)>4*1024*1024: raise HTTPException(413,'Upload limit is 4 MiB; resize before sending')
        future=asyncio.get_running_loop().run_in_executor(app.state.worker,inspect,raw)
        app.state.inflight=future
        # A disconnected client must not make an already-running GPU job look
        # idle and accumulate stale work in ThreadPoolExecutor's internal queue.
        result=await asyncio.shield(future)
        result['frame_id']=frame_id
        result['timings_ms']['queue_wait']=queue_ms
        result['timings_ms']['server_request']=1000*(time.perf_counter()-request.state.started)
        return result
    finally:
        if acquired:app.state.busy.release()
        app.state.admitted-=1


@app.get('/',response_class=HTMLResponse)
def home():
    return (Path(__file__).with_name('b1_demo.html')).read_text(encoding='utf-8')


@app.get('/example')
def example():
    path=ROOT/'data/straw_defects_raw/Straw Defects in Juice Box/Chocolait/Bent/Bent (4).jpg'
    if not path.exists(): raise HTTPException(404,'Validation example is not available')
    return FileResponse(path,media_type='image/jpeg')


@app.get('/validation-check',response_class=HTMLResponse)
def validation_page():
    return Path(__file__).with_name('b1_validation.html').read_text(encoding='utf-8')


def validation_index():
    path=ROOT/'artifacts/b1_deployment/frame640/index.json'
    if not path.exists():raise HTTPException(404,'Run prepare_browser_replay.py first')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['manifest_sha256']!=file_hash(ROOT/'artifacts/straw_global_local/data_v1/val.csv'):
        raise HTTPException(409,'Validation manifest changed')
    return data


@app.get('/validation-cases')
def validation_cases():
    data=validation_index()
    return {'cases':[{'sample_id':r['sample_id'],'url':r['url']} for r in data['cases']],
        'note':data['note']}


@app.get('/validation-frame/{sample_id}')
def validation_frame(sample_id:str):
    data=validation_index();allowed={r['sample_id'] for r in data['cases']}
    if sample_id not in allowed:raise HTTPException(404,'Unknown validation frame')
    return FileResponse(ROOT/'artifacts/b1_deployment/frame640'/f'{sample_id}.jpg',media_type='image/jpeg')


class ReplayFrame(BaseModel):
    sample_id:str
    prediction:str
    client_ms:float=Field(ge=0)
    http_ms:float=Field(ge=0)
    encode_ms:float=Field(ge=0)


class ReplayResults(BaseModel):
    frames:list[ReplayFrame]=Field(min_length=300,max_length=300)


@app.post('/validation-results')
def validation_results(body:ReplayResults):
    from sklearn.metrics import confusion_matrix,f1_score
    cases=validation_index()['cases'];expected={r['sample_id']:r['label'] for r in cases}
    ids=[r.sample_id for r in body.frames]
    if len(set(ids))!=300 or set(ids)!=set(expected):raise HTTPException(422,'Expected all 300 unique validation IDs')
    if any(r.prediction not in CLASSES for r in body.frames):raise HTTPException(422,'Unknown class')
    y=np.array([expected[r.sample_id] for r in body.frames]);pred=np.array([CLASSES.index(r.prediction) for r in body.frames])
    defect=y!=0
    report={'source':'static 640px validation replay through actual browser canvas/JPEG; not real camera',
        'checkpoint_sha256':app.state.checkpoint_hash,'size':app.state.config['size'],'n':300,
        'accuracy':float(np.mean(y==pred)),'macro_f1':float(f1_score(y,pred,average='macro',labels=list(range(5)),zero_division=0)),
        'fn':int(np.sum(defect&(pred==0))),'fp':int(np.sum(~defect&(pred!=0))),
        'wrong_defect_type':int(np.sum(defect&(pred!=0)&(y!=pred))),
        'confusion':confusion_matrix(y,pred,labels=list(range(5))).tolist(),
        'timings_ms':{field:{f'p{q}':float(np.percentile([getattr(r,field) for r in body.frames],q)) for q in (50,95,99)} for field in ('client_ms','http_ms','encode_ms')},
        'deadline_200ms_misses':sum(r.client_ms>200 for r in body.frames),
        'frames':[r.model_dump() for r in body.frames]}
    (ROOT/'artifacts/b1_deployment/browser_replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return {key:value for key,value in report.items() if key!='frames'}
