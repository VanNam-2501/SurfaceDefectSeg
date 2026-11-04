"""Exploratory manual-ROI probe on error cases only; not a performance evaluation."""
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.straw_global_local.data import render_arrays, tensor_views
from src.straw_global_local.model import GlobalLocalCNN
OUT=Path(__file__).parent
BASE=ROOT/'straw_resume/artifacts/straw_global_local'
constants={}
for node in ast.parse((OUT/'analyze_cases.py').read_text(encoding='utf-8')).body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('IDS','BOXES'):
        constants[node.targets[0].id]=ast.literal_eval(node.value)
rows={r['sample_id']:r for r in csv.DictReader((BASE/'data_v1/test.csv').open(encoding='utf-8'))}
zooms={}
for sid,box in zip(constants['IDS'],constants['BOXES']):
    row=rows[sid]
    with Image.open(ROOT/'data/straw_defects_raw'/row['relative_path']) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB')
    w,h=image.size
    crop=image.crop(tuple(round(v*(w if j%2==0 else h)) for j,v in enumerate(box)))
    side=max(crop.size)
    bg=tuple(int(v) for v in np.asarray(image.crop((0,0,40,40))).mean(axis=(0,1)))
    square=Image.new('RGB',(side,side),bg)
    square.paste(crop,((side-crop.width)//2,(side-crop.height)//2))
    zooms[sid]=render_arrays(square)
torch.set_num_threads(4)
classes=['good','bent','loose','missing','pierced']
records=[]
for variant in ('B1','B6'):
    for seed in (42,43,44):
        model=GlobalLocalCNN(variant,pretrained=False).eval()
        ckpt=torch.load(BASE/f'confirmatory_kaggle_v1/runs/{variant}_s{seed}/best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'])
        for number,sid in enumerate(constants['IDS'],1):
            batch={k:v[None] for k,v in tensor_views(zooms[sid],variant).items()}
            with torch.inference_mode():
                p=model(batch)[0].softmax(0).numpy()
            records.append({'case':number,'sample_id':sid,'variant':variant,'seed':seed,
                'label':rows[sid]['class_name'],'prediction':classes[int(p.argmax())],
                'pred_probability':float(p.max())})
        print('Probe complete',variant,seed,flush=True)
with (OUT/'manual_zoom_probe.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
for number in range(1,11):
    print(number,{v:[r['prediction'] for r in records if r['case']==number and r['variant']==v] for v in ('B1','B6')},flush=True)
