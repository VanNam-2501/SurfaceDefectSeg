"""Validation-only horizontal-flip TTA probe for frozen B1 512 checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.straw_global_local.model import GlobalLocalCNN
from src.straw_global_local.data import tensor_views

BASE = Path('straw_resume/artifacts/straw_global_local')
RUNS = BASE / 'confirmatory_kaggle_v1/runs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)


def probability(logits):
    z = np.asarray(logits, dtype=np.float64)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def metrics(labels, pred):
    y = np.asarray(labels)
    return {'macro_f1': float(f1_score(y, pred, labels=range(5), average='macro', zero_division=0)),
            'errors': int(np.sum(y != pred)),
            'fn': int(np.sum((y != 0) & (pred == 0))),
            'fp': int(np.sum((y == 0) & (pred != 0))),
            'wrong_defect_type': int(np.sum((y != 0) & (pred != 0) & (pred != y)))}


def flipped_logits(model, ids, batch_size=4):
    output = []
    for start in range(0, len(ids), batch_size):
        arrays = []
        for sid in ids[start:start + batch_size]:
            with np.load(BASE / 'cache_v1' / f'{sid}.npz', allow_pickle=False) as values:
                arrays.append(tensor_views(values, 'B1')['global'].flip(-1))
        tensor = torch.stack(arrays).to(DEVICE)
        with torch.inference_mode():
            result = model({'global': tensor}).float().cpu().numpy()
        output.append(result)
    return np.concatenate(output)


summary = []
for seed in (42, 43, 44):
    run = RUNS / f'B1_s{seed}'
    with np.load(run / 'val_predictions.npz', allow_pickle=False) as d:
        ids, labels, original = d['sample_ids'], d['labels'], d['logits'].copy()
    model = GlobalLocalCNN('B1', pretrained=False).eval().to(DEVICE)
    model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=False)['model'])
    flip = flipped_logits(model, ids)
    np.savez_compressed(Path(__file__).parent / f'b1_s{seed}_val_flip_logits.npz',
                        sample_ids=ids, labels=labels, logits=flip)
    original_p, flip_p = probability(original), probability(flip)
    candidates = {
        'original': original_p.argmax(1),
        'flip_only': flip_p.argmax(1),
        'mean_probabilities': ((original_p + flip_p) / 2).argmax(1),
        'mean_logits': ((original.astype(float) + flip.astype(float)) / 2).argmax(1),
    }
    item = {'seed': seed, 'device': DEVICE,
            'metrics': {name: metrics(labels, pred) for name, pred in candidates.items()},
            'changed_ids': {name: [str(ids[i]) for i in np.flatnonzero(pred != candidates['original'])]
                            for name, pred in candidates.items() if name != 'original'}}
    summary.append(item)
    print(json.dumps(item), flush=True)
(Path(__file__).parent / 'b1_flip_validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
