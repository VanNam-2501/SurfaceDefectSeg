"""Reproduce B1 locally and benchmark batch-one inference without retraining."""
import copy
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageOps
from src.straw_global_local.data import CLASSES, file_hash, render_arrays, tensor_views
from src.straw_global_local.model import GlobalLocalCNN
from src.straw_global_local.runner import classification_metrics as metrics

OUT = ROOT / 'artifacts/local_b1_check'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def main():
    archive = Path(r'E:\Dowload\straw_resume.zip')
    prefix = 'artifacts/straw_global_local/confirmatory_kaggle_v1/runs/'
    with zipfile.ZipFile(archive) as z:
        for seed in (42, 43, 44):
            for name in ('best.pt', 'complete.json', 'val_predictions.npz'):
                target = OUT / f'B1_s{seed}' / name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    target.write_bytes(z.read(f'{prefix}B1_s{seed}/{name}'))
        for role in ('val', 'test'):
            local = ROOT / f'artifacts/straw_global_local/data_v1/{role}.csv'
            remote = pd.read_csv(z.open(f'artifacts/straw_global_local/data_v1/{role}.csv'))
            frame = pd.read_csv(local)
            assert frame.sample_id.tolist() == remote.sample_id.tolist(), 'Split order differs'
            assert frame.sha256.tolist() == remote.sha256.tolist(), 'Image hashes differ'
    frame = pd.read_csv(ROOT / 'artifacts/straw_global_local/data_v1/test.csv')
    env = dict(python=platform.python_version(), torch=torch.__version__,
               cpu=platform.processor(), logical_cpus=os.cpu_count(), threads=torch.get_num_threads(),
               ram_gib=psutil.virtual_memory().total / 2**30,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               note='Laptop measurements; not a dedicated edge device. Batch=1.')
    save('environment.json', env)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    for seed in (42, 43, 44):
        checkpoint = OUT / f'B1_s{seed}/best.pt'
        marker = json.loads((checkpoint.parent / 'complete.json').read_text())
        assert file_hash(checkpoint) == marker['best_checkpoint'], 'Checkpoint hash mismatch'
        model = GlobalLocalCNN('B1', pretrained=False).eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
        models.append(model.to(device))
    logits = [[] for _ in models]
    half_model = copy.deepcopy(models[0]).half() if device == 'cuda' else None
    half_logits = []
    print(f'Evaluating {len(frame)} test images on {device}; original images hash-checked.', flush=True)
    with torch.inference_mode():
        for i, row in enumerate(frame.itertuples()):
            path = ROOT / 'data/straw_defects_raw' / row.relative_path
            assert file_hash(path) == row.sha256, f'Image changed: {path}'
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
                batch = tensor_views(render_arrays(image, 'B1'), 'B1')
            x = {'global': batch['global'][None].to(device)}
            for j, model in enumerate(models):
                logits[j].append(model(x).float().cpu().numpy()[0])
            if half_model is not None:
                half_logits.append(half_model({'global': x['global'].half()}).float().cpu().numpy()[0])
            if (i+1) % 50 == 0:
                print(f'Test {i+1}/{len(frame)}', flush=True)
    labels = frame.label.to_numpy()
    rows = []
    for seed, scores in zip((42, 43, 44), logits):
        scores = np.asarray(scores)
        result = metrics(labels, scores)
        save(f'B1_s{seed}_test.json', result)
        rows.append(dict(seed=seed, precision='FP32', **{k:v for k,v in result.items() if np.isscalar(v)}))
        pred = scores.argmax(1)
        export = frame[['sample_id','relative_path','class_name','label']].copy()
        export['prediction'] = [CLASSES[p] for p in pred]
        for c, name in enumerate(CLASSES):
            export[f'logit_{name}'] = scores[:, c]
        export.to_csv(OUT / f'B1_s{seed}_predictions.csv', index=False)
        export[labels != pred].to_csv(OUT / f'B1_s{seed}_errors.csv', index=False)
    if half_model is not None:
        result = metrics(labels, np.asarray(half_logits))
        save('B1_s42_fp16_test.json', result)
        rows.append(dict(seed=42, precision='FP16', **{k:v for k,v in result.items() if np.isscalar(v)}))
        save('fp16_agreement.json', {'n':len(labels), 'changed_predictions':int(np.sum(np.argmax(logits[0],1) != np.argmax(half_logits,1)))})
    pd.DataFrame(rows).to_csv(OUT / 'test_summary.csv', index=False)
    # Threshold survey uses SAVED VALIDATION predictions exclusively. No test tuning.
    val = np.load(OUT / 'B1_s42/val_predictions.npz')
    p = torch.tensor(val['logits']).softmax(1).numpy()
    defect_score = 1-p[:,0]
    threshold_rows = []
    for tau in np.linspace(0, 1, 101):
        pred = np.where(defect_score >= tau, p[:,1:].argmax(1)+1, 0)
        y = val['labels']; defect = y != 0
        threshold_rows.append(dict(threshold=tau, fn=int(np.sum(defect & (pred==0))),
            fp=int(np.sum(~defect & (pred!=0))), good_to_bent=int(np.sum(~defect & (pred==1))),
            fnr=float(np.mean(pred[defect]==0)), fpr=float(np.mean(pred[~defect]!=0))))
    pd.DataFrame(threshold_rows).to_csv(OUT / 'validation_threshold_survey.csv', index=False)
    # Use validation images for all speed comparisons; same workload across devices.
    valframe = pd.read_csv(ROOT / 'artifacts/straw_global_local/data_v1/val.csv')
    samples = valframe.groupby('label', sort=True).head(4).to_dict('records')
    benches = []
    targets = [('cpu','FP32'), ('cuda','FP32'), ('cuda','FP16')] if device == 'cuda' else [('cpu','FP32')]
    for target, precision in targets:
        model = copy.deepcopy(models[0]).to(target).eval()
        if precision == 'FP16': model.half()
        total, forward = [], []
        if target == 'cuda': torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for i in range(110):
                row = samples[i % len(samples)]
                sync(target); start = time.perf_counter()
                with Image.open(ROOT / 'data/straw_defects_raw' / row['relative_path']) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                    batch = tensor_views(render_arrays(image, 'B1'), 'B1')
                x = batch['global'][None].to(target)
                if precision == 'FP16': x = x.half()
                sync(target); before = time.perf_counter()
                model({'global':x}); sync(target); end = time.perf_counter()
                if i >= 10:
                    total.append(1000*(end-start)); forward.append(1000*(end-before))
        benches.append(dict(device=target,precision=precision,warmup=10,repeats=100,batch=1,
            p50_end_to_end_ms=float(np.median(total)),p95_end_to_end_ms=float(np.quantile(total,.95)),
            p50_forward_ms=float(np.median(forward)),p95_forward_ms=float(np.quantile(forward,.95)),
            process_rss_mib=psutil.Process().memory_info().rss/2**20,
            note='RSS includes evaluator and resident models; not isolated deployment memory.'))
        print(benches[-1], flush=True)
        pd.DataFrame(benches).to_csv(OUT / 'benchmarks.csv', index=False)
        del model
    print('COMPLETE: '+str(OUT), flush=True)


if __name__ == '__main__': main()
