"""Prepare a label-independent 640px static-frame proxy for browser validation.

This is a replay of validation photos, not real camera/video-domain validation.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from PIL import Image,ImageOps
from src.straw_global_local.data import file_hash

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/b1_deployment/frame640'
OUT.mkdir(parents=True,exist_ok=True)
manifest=ROOT/'artifacts/straw_global_local/data_v1/val.csv'
frame=pd.read_csv(manifest)
cases=[]
for i,row in enumerate(frame.itertuples()):
    source=ROOT/'data/straw_defects_raw'/row.relative_path
    assert file_hash(source)==row.sha256
    target=OUT/f'{row.sample_id}.jpg'
    if not target.exists():
        with Image.open(source) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB').resize((640,640),Image.Resampling.BILINEAR)
            image.save(target,format='JPEG',quality=95)
    cases.append({'sample_id':row.sample_id,'label':int(row.label),'class_name':row.class_name,
        'url':f'/validation-frame/{row.sample_id}','sha256':file_hash(target)})
    if (i+1)%50==0:print('Prepared frame proxies',i+1,'/',len(frame),flush=True)
(OUT/'index.json').write_text(json.dumps({'manifest_sha256':file_hash(manifest),'cases':cases,
    'note':'Static 640x640 JPEG95 proxies; not a camera dataset. Browser then renders/encodes with actual deploy size and JPEG quality.'},indent=2),encoding='utf-8')
print('Prepared '+str(OUT),flush=True)
