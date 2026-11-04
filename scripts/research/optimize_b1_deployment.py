"""Validation-only deployment study: exact input, JPEG, FP16, ONNX and INT8.

Run with --phase gpu, --phase onnx or --phase all. Writes independent results;
never changes original checkpoints, splits, thresholds or the existing web app.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from src.straw_global_local.data import file_hash
from src.straw_global_local.deployment import TensorB1,TorchRuntime,ORTRuntime,packet,decode,measure
from src.straw_global_local.runner import classification_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/b1_deployment'
CHECKPOINT=ROOT/'artifacts/local_b1_check/B1_s42/best.pt'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def source(row): return ROOT/'data/straw_defects_raw'/row.relative_path


def evaluate(name,runtime,frame,payloads,size,metadata):
    folder=OUT/name; folder.mkdir(parents=True,exist_ok=True)
    saved=[]
    start=time.perf_counter()
    for i,raw in enumerate(payloads):
        saved.append(runtime.run(decode(raw,size))[0])
        if (i+1)%100==0: print(f'{name}: val {i+1}/{len(frame)}',flush=True)
    logits=np.asarray(saved)
    result=classification_metrics(frame.label.to_numpy(),logits)
    np.savez_compressed(folder/'val_predictions.npz',logits=logits,labels=frame.label.to_numpy(),sample_ids=frame.sample_id.to_numpy(dtype=str))
    runtime.size=size
    # Balanced selection, identical validation IDs across all candidates.
    indices=frame.reset_index().groupby('label',sort=True).head(4)['index'].tolist()
    timings=measure(runtime,[payloads[i] for i in indices],repeats=60,warmup=20)
    result.update(metadata)
    result.update(timings)
    result.update(name=name,size=size,n=len(frame),role='val',mean_payload_kib=float(np.mean([len(p) for p in payloads])/1024),
                  evaluation_seconds=time.perf_counter()-start)
    write(folder/'result.json',result)
    print(name, result['macro_f1'],result['fn'],result['fp'],result['server_p95_ms'],flush=True)
    return result


def export_and_quantize(size,train,quality):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader,quantize_static,QuantType,QuantFormat,CalibrationMethod
    from onnxruntime.quantization.shape_inference import quant_pre_process
    target=OUT/'models'; target.mkdir(exist_ok=True)
    temporary=OUT/'tmp';temporary.mkdir(exist_ok=True)
    tempfile.tempdir=str(temporary)
    fp=target/f'b1_{size}_fp32.onnx'
    prepared=target/f'b1_{size}_prepared.onnx'
    quant=target/f'b1_{size}_int8.onnx'
    if not fp.exists():
        model=TensorB1(CHECKPOINT).eval()
        torch.onnx.export(model,torch.zeros(1,3,size,size),str(fp),opset_version=17,
            input_names=['image'],output_names=['logits'],dynamo=False)
        onnx.checker.check_model(onnx.load(str(fp)))
    reference=TensorB1(CHECKPOINT).eval()
    x=decode(packet(source(train.iloc[0]),size,quality),size)
    with torch.inference_mode(): expected=reference(torch.from_numpy(x)).numpy()
    actual=ORTRuntime(fp).run(x)
    parity=float(np.max(np.abs(expected-actual)))
    assert np.allclose(expected,actual,atol=2e-4,rtol=2e-4),f'ONNX parity failed: {parity}'
    write(target/f'parity_{size}.json',{'max_absolute_logit_error':parity,'checked':'one training image; full val evaluated separately'})
    if not quant.exists():
        # Torch export already folds eval Conv/BN. Avoid device-specific compilation
        # during preprocessing; quantized and float runtimes are measured separately.
        if not prepared.exists(): quant_pre_process(str(fp),str(prepared),skip_symbolic_shape=True,skip_optimization=True)
        # 40 original train images per class, disjoint from validation/test.
        calibration=train.groupby('label',sort=True).head(40)
        write(target/f'calibration_{size}.json',{'sample_ids':calibration.sample_id.tolist(),'role':'train','quality':quality})
        class Reader(CalibrationDataReader):
            def __init__(self): self.rows=iter(calibration.itertuples())
            def get_next(self):
                row=next(self.rows,None)
                return None if row is None else {'image':decode(packet(source(row),size,quality),size)}
        quantize_static(str(prepared),str(quant),Reader(),quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,
            calibrate_method=CalibrationMethod.MinMax)
    return fp,quant


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['gpu','onnx','all'],default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/'artifacts/straw_global_local/data_v1/val.csv')
    train=pd.read_csv(ROOT/'artifacts/straw_global_local/data_v1/train.csv')
    spec={'checkpoint_sha256':file_hash(CHECKPOINT),'val_manifest_sha256':file_hash(ROOT/'artifacts/straw_global_local/data_v1/val.csv'),
          'torch':torch.__version__,'python':platform.python_version(),'torch_threads':4,'seed':42,
          'test_used_for_selection':False,'input_note':'PIL browser proxy; must also verify actual browser canvas output.'}
    write(OUT/'study.json',spec)
    for row in frame.itertuples(): assert file_hash(source(row))==row.sha256,'Original input mismatch'
    exact=[source(row).read_bytes() for row in frame.itertuples()]
    # Store JPEG proxy inputs once; client cost is measured separately from server.
    payloads={}
    for size in ((512,384) if args.phase in ('gpu','all') else (512,)):
        started=time.perf_counter()
        payloads[size]=[packet(source(row),size,90) for row in frame.itertuples()]
        write(OUT/f'input_{size}.json',{'average_client_proxy_ms':1000*(time.perf_counter()-started)/len(frame),
             'quality':90,'size':size,'note':'Offline PIL resize+JPEG proxy. Excludes camera and network. Not browser latency.'})
    candidates=[]
    if args.phase in ('gpu','all'):
        configs=[('torch_fp32_exact',512,'fp32',False,exact),
                 ('torch_fp32_jpeg512',512,'fp32',False,payloads[512]),
                 ('torch_fp16_jpeg512',512,'fp16',False,payloads[512]),
                 ('torch_fp16_cl_jpeg512',512,'fp16',True,payloads[512]),
                 ('torch_fp16_cl_jpeg384',384,'fp16',True,payloads[384])]
        for name,size,precision,channels,raw in configs:
            runtime=TorchRuntime(CHECKPOINT,size,precision,channels,autotune=True)
            candidates.append(evaluate(name,runtime,frame,raw,size,dict(runtime='pytorch_cuda',precision=precision,channels_last=channels,autotune=True)))
            del runtime; gc.collect(); torch.cuda.empty_cache()
    if args.phase in ('onnx','all'):
        for size in (512,):
            fp,quant=export_and_quantize(size,train,90)
            for precision,path in [('fp32',fp),('int8',quant)]:
                runtime=ORTRuntime(path)
                candidates.append(evaluate(f'onnx_{precision}_jpeg{size}',runtime,frame,payloads[size],size,
                    dict(runtime='onnx_cpu',precision=precision,model_mib=path.stat().st_size/2**20)))
                del runtime; gc.collect()
    all_results=[json.loads(p.read_text(encoding='utf-8')) for p in OUT.glob('*/result.json')]
    flat=[{k:v for k,v in r.items() if np.isscalar(v) or v is None} for r in all_results]
    pd.DataFrame(flat).to_csv(OUT/'summary.csv',index=False)
    print('DONE: '+str(OUT),flush=True)


if __name__=='__main__': main()
