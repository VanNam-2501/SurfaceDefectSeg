"""Freeze a validation-selected web config and materialize FP16 weights.

Fixed serving seed 42. Never select the best seed on test or validation.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import torch
from src.straw_global_local.data import file_hash
from src.straw_global_local.deployment import TensorB1

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/b1_deployment'


def main():
    baseline=json.loads((OUT/'torch_fp32_exact/result.json').read_text())
    timings=pd.read_csv(OUT/'final_idle_benchmarks.csv').set_index('name')
    recovered=OUT/'recovery_finetune_384_s42_safety'
    quality=json.loads((recovered/'fp16_val.json').read_text())
    safe=quality['macro_f1']>=baseline['macro_f1']-.005 and quality['fn']<=baseline['fn'] and quality['fp']<=baseline['fp']
    faster=timings.loc['finetuned384_gpu_fp16_cl','server_p95_ms']<min(timings.loc['gpu_fp32','server_p95_ms'],timings.loc['gpu_fp16_cl','server_p95_ms'])
    cfg=json.loads((ROOT/'configs/straw_realtime.json').read_text())
    source=recovered/'best_eligible.pt'
    if safe and faster:
        model=TensorB1(source).eval().half()
        target=OUT/'models/b1_web384_fp16.pt'
        torch.save({'model':model.state_dict(),'source_sha256':file_hash(source),'validation':quality},target)
        reloaded=TensorB1(target).eval().half()
        assert all(torch.equal(value,reloaded.state_dict()[key]) for key,value in model.state_dict().items()),'FP16 checkpoint roundtrip differs'
        cfg.update(checkpoint=target.relative_to(ROOT).as_posix(),checkpoint_sha256=file_hash(target),size=384,precision='fp16',channels_last=True,
            note='Seed 42 fixed; recovered 384 JPEG q90, FP16 artifact validated by state roundtrip and validation. Exploratory deployment study; no new test selection.')
        selected='recovered384_fp16'
    else:
        source=ROOT/'artifacts/local_b1_check/B1_s42/best.pt'
        half=timings.loc['gpu_fp16_cl','server_p95_ms']<timings.loc['gpu_fp32','server_p95_ms']
        cfg.update(checkpoint=source.relative_to(ROOT).as_posix(),checkpoint_sha256=file_hash(source),size=512,
            precision='fp16' if half else 'fp32',channels_last=bool(half))
        selected='original512'
    (ROOT/'configs/straw_realtime.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    ledger={'selected':selected,'fixed_seed':42,'validation_only':True,'test_used':False,'candidate384_safe':safe,
        'candidate384_faster_in_pilot':bool(faster),'max_f1_drop':.005,'max_fn':baseline['fn'],'max_fp':baseline['fp'],
        'config':cfg,'source_sha256':file_hash(source),
        'note':'Quality constraints are validation engineering criteria, not statistical noninferiority proof. Browser/HTTP checks follow this freeze.'}
    (OUT/'selection.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    print(json.dumps(ledger,indent=2),flush=True)


if __name__=='__main__':main()
