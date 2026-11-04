"""Exploratory operating-point analysis. Select thresholds using validation only."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE.parent / 'artifacts/straw_global_local/confirmatory_kaggle_v1/runs'


def load(run, split):
    with np.load(run / f'{split}_predictions.npz', allow_pickle=False) as saved:
        labels = saved['labels'].copy()
        z = saved['logits'].astype(float)
    probs = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return labels, probs


def metrics(labels, predictions):
    return {
        'fn': int(np.sum((labels != 0) & (predictions == 0))),
        'fp': int(np.sum((labels == 0) & (predictions != 0))),
        'wrong_defect_type': int(np.sum((labels != 0) & (predictions != 0) & (labels != predictions))),
        'total_5class_errors': int(np.sum(labels != predictions)),
    }


records = []
for variant in ('B1', 'B6'):
    for seed in (42, 43, 44):
        run = RUNS / f'{variant}_s{seed}'
        yv, pv = load(run, 'val')
        scores = 1 - pv[:, 0]
        minimum_defect = float(scores[yv != 0].min())
        lower = scores[scores < minimum_defect]
        # Largest zero-FN validation operating region; midpoint of its lower boundary gap.
        # This minimizes validation false positives subject to zero validation misses.
        threshold = (minimum_defect + (float(lower.max()) if len(lower) else 0.0)) / 2
        yt, pt = load(run, 'test')

        def decide(p):
            return np.where(1 - p[:, 0] >= threshold, p[:, 1:].argmax(1) + 1, 0)

        records.append({
            'variant': variant, 'seed': seed, 'threshold': threshold,
            'val_before': metrics(yv, pv.argmax(1)),
            'val_after': metrics(yv, decide(pv)),
            'test_before': metrics(yt, pt.argmax(1)),
            'test_after': metrics(yt, decide(pt)),
        })

result = {
    'scope': 'Exploratory analysis after inspecting test errors; not a preregistered test result.',
    'score': 'sum of defect softmax probabilities = 1 - P(good); not calibrated risk',
    'selection': 'Validation only: zero validation FN, then minimum validation FP; midpoint in boundary gap.',
    'decision': 'score >= threshold => defect, subtype = argmax of four defect probabilities; otherwise good.',
    'limitations': [
        'Zero validation misses is an operating constraint, not a future guarantee.',
        'Thresholds are checkpoint dependent and must be selected again after retraining.',
        'Previously exposed test and small sample size require a new independent evaluation.',
        'Subtype mistakes can persist or new ones can appear when converting good to defect.',
    ],
    'runs': records,
}
(HERE / 'postprocessing_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
for row in records:
    print(row['variant'], row['seed'], 'threshold', round(row['threshold'], 7),
          'val after', row['val_after'], 'test before', row['test_before'],
          'test after', row['test_after'])
