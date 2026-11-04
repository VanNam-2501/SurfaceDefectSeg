"""Run the selected B1 web checkpoint on validation JPEG proxy inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.deployment import TorchRuntime, decode, softmax

cfg = json.loads((ROOT/'configs/straw_realtime.json').read_text(encoding='utf-8'))
cases = json.loads((ROOT/'artifacts/b1_deployment/frame640/index.json').read_text(encoding='utf-8'))['cases']
torch.set_num_threads(4)
runtime = TorchRuntime(ROOT/cfg['checkpoint'], size=cfg['size'], precision=cfg['precision'],
                       channels_last=cfg['channels_last'], device=cfg['device'])
logits = []
for j,case in enumerate(cases):
    raw = (ROOT/'artifacts/b1_deployment/input_cache/384_q90'/f"{case['sample_id']}.jpg").read_bytes()
    logits.append(runtime.run(decode(raw, cfg['size']))[0])
    if (j+1)%100 == 0:
        print('Processed', j+1, flush=True)
z = np.asarray(logits)
p = softmax(z, cfg['temperature'])
y = np.array([case['label'] for case in cases])
score = 1-p[:,0]
print('Argmax errors/FN/FP:',int(sum(p.argmax(1)!=y)),int(sum((y!=0)&(p.argmax(1)==0))),int(sum((y==0)&(p.argmax(1)!=0))), flush=True)
print('Good maximum score:',float(score[y==0].max()),'Defect minimum score:',float(score[y!=0].min()), flush=True)
print('FN cases:',[(cases[i]['sample_id'],float(score[i])) for i in np.where((y!=0)&(p.argmax(1)==0))[0]],flush=True)
np.savez_compressed(Path(__file__).parent/'b1_web384_val_logits.npz',logits=z,labels=y,
                    sample_ids=np.array([case['sample_id'] for case in cases]))
