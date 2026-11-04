"""Read-only case diagnostics; visual zooms are manual inspection regions, not labels."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'straw_resume/artifacts/straw_global_local'
OUT = Path(__file__).parent
CLASSES = ['good', 'bent', 'loose', 'missing', 'pierced']
IDS = [
    '2111bd90d1573ba7fcb9', '4ca8398053bf7212e86e',
    '7371773a4b79f5da705a', '7e70ee7ef5a71a0a0d49',
    '86be61c858692d3d577f', '9fe49ea541a582d8fd3c',
    'b51e257986f8709546a8', 'bd338d974748090dd2e1',
    'c088784d1b58b132df6d', 'cd73a826ea98b5004542',
]
BOXES = [(.12,.35,.9,.99),(.47,.08,.86,.86),(.24,.01,1,.95),
         (.12,.01,.72,.92),(0,0,.88,.99),(.1,.04,.97,.96),
         (0,.01,.98,1),(.26,.06,.78,.98),(0,.23,.9,1),(.32,.05,.75,.9)]
rows = list(csv.DictReader((BASE/'data_v1/test.csv').open(encoding='utf-8')))
indices = {r['sample_id']: i for i,r in enumerate(rows)}
saved = {}
for variant in ('B1','B6'):
    for seed in (42,43,44):
        p = BASE/f'confirmatory_kaggle_v1/runs/{variant}_s{seed}/test_predictions.npz'
        with np.load(p, allow_pickle=False) as d:
            assert list(d['sample_ids']) == [r['sample_id'] for r in rows]
            assert list(d['labels']) == [int(r['label']) for r in rows]
            logits = d['logits'].astype(float)
        probs = np.exp(logits-logits.max(1,keepdims=True))
        saved[(variant,seed)] = probs/probs.sum(1,keepdims=True)

import torch
from src.straw_global_local.model import GlobalLocalCNN
from src.straw_global_local.data import tensor_views
torch.set_num_threads(4)
attention = {}
for seed in (42,43,44):
    model = GlobalLocalCNN('B6', pretrained=False).eval()
    checkpoint = torch.load(BASE/f'confirmatory_kaggle_v1/runs/B6_s{seed}/best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    for sid in IDS:
        with np.load(BASE/f'cache_v1/{sid}.npz',allow_pickle=False) as d:
            batch = {k:v[None] for k,v in tensor_views(d,'B6').items()}
        with torch.inference_mode():
            logits, weights = model(batch, return_attention=True)
        i = indices[sid]
        attention[(sid,seed)] = {'weights': weights[0].tolist(),
            'cpu_prediction':CLASSES[int(logits[0].argmax())],
            'saved_prediction':CLASSES[int(saved[('B6',seed)][i].argmax())]}
    print('Attention complete',seed,flush=True)

font_path = 'C:/Windows/Fonts/arial.ttf'
font = ImageFont.truetype(font_path,20)
small = ImageFont.truetype(font_path,17)
records=[]
for number,(sid,box) in enumerate(zip(IDS,BOXES),1):
    i=indices[sid]; row=rows[i]
    path=ROOT/'data/straw_defects_raw'/row['relative_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],sid
    with Image.open(path) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB')
    w,h=image.size
    zoom=image.crop(tuple(round(v*(w if j%2==0 else h)) for j,v in enumerate(box)))
    page=Image.new('RGB',(1420,900),'white'); draw=ImageDraw.Draw(page)
    draw.text((15,10),f'{number:02d} | {row["brand"]} | label={row["class_name"]} | {sid}',fill='black',font=font)
    lines=[]
    for variant in ('B1','B6'):
        parts=[]
        for seed in (42,43,44):
            p=saved[(variant,seed)][i]; pred=int(p.argmax())
            parts.append(f's{seed}: {CLASSES[pred]} ({p[pred]:.1%})')
            records.append({'case':number,'sample_id':sid,'brand':row['brand'],
                'label':row['class_name'],'variant':variant,'seed':seed,
                'prediction':CLASSES[pred],'pred_probability':p[pred],
                'true_probability':p[int(row['label'])], 'defect_probability':1-p[0],
                'relative_path':row['relative_path'],
                'group_id':row['group_id']})
        lines.append(variant+' | '+' | '.join(parts))
    for j,line in enumerate(lines): draw.text((15,40+j*26),line,fill='black',font=small)
    zoom.thumbnail((640,735));page.paste(zoom,((650-zoom.width)//2,110+(735-zoom.height)//2))
    draw.text((15,858),'Zoom from original; manual inspection region, not a model input.',fill='black',font=small)
    with np.load(BASE/f'cache_v1/{sid}.npz',allow_pickle=False) as d:
        page.paste(Image.fromarray(d['global224']),(680,125))
        draw.text((680,100),'B6 global 224',fill='black',font=small)
        for k,(x,y) in enumerate(((925,125),(1175,125),(925,430),(1175,430))):
            page.paste(Image.fromarray(d['local'][k]),(x,y))
            draw.text((x,y-25),('top left','top right','bottom left','bottom right')[k],fill='black',font=small)
            values='/'.join(f'{attention[(sid,s)]["weights"][k]:.2f}' for s in (42,43,44))
            draw.text((x,y+230),'weight '+values,fill='black',font=small)
    draw.text((680,735),'Weights: seed 42 / 43 / 44.',fill='black',font=small)
    draw.text((680,765),'Attention weights do not establish causal explanations.',fill='black',font=small)
    page.save(OUT/f'case_{number:02d}.png')
with (OUT/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
(OUT/'attention.json').write_text(json.dumps({f'{sid}_s{seed}':v for (sid,seed),v in attention.items()},indent=2),encoding='utf-8')
print('Wrote 10 case pages, 60 prediction records, and attention diagnostics.',flush=True)
print('CPU/saved prediction mismatches:',sum(v['cpu_prediction']!=v['saved_prediction'] for v in attention.values()),flush=True)
