"""Fit a conservative B1 512 decision rule using validation predictions only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.postprocess import (
    apply_policy, counts, file_sha256, probabilities,
    predict_from_probabilities, select_zero_fp_threshold,
)


def evaluate(labels, predictions):
    result = counts(labels, predictions)
    result['macro_f1'] = float(f1_score(labels, predictions, labels=range(5), average='macro', zero_division=0))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=ROOT/'straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1/runs')
    parser.add_argument('--protocol', type=Path, default=ROOT/'straw_resume/artifacts/straw_global_local/data_v1/protocol.json')
    parser.add_argument('--out', type=Path, default=ROOT/'straw_resume/error_analysis/b1_postprocess')
    parser.add_argument('--evaluate-exposed-test', action='store_true', help='Exploratory report only; never fits policy')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in (42, 43, 44):
        run = args.runs / f'B1_s{seed}'
        val_path = run/'val_predictions.npz'
        with np.load(val_path, allow_pickle=False) as saved:
            val_labels = saved['labels'].copy()
            val_logits = saved['logits'].copy()
        threshold, selection = select_zero_fp_threshold(val_labels, probabilities(val_logits))
        checkpoint = run/'best.pt'
        policy = {
            'schema': 1, 'variant': 'B1', 'seed': seed, 'input_size': 512,
            'checkpoint_sha256': file_sha256(checkpoint),
            'validation_predictions_sha256': file_sha256(val_path),
            'protocol_fingerprint': protocol['fingerprint'],
            'temperature': 1.0, 'threshold': threshold,
            'selection': selection,
            'score': '1 - P(good)',
            'defect_subtype': 'argmax among bent, loose, missing, pierced',
        }
        (args.out/f'B1_s{seed}.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
        record = {'seed': seed, 'threshold': threshold,
                  'validation_original': evaluate(val_labels, predict_from_probabilities(probabilities(val_logits), None)),
                  'validation_selected': evaluate(val_labels, apply_policy(val_logits, policy, checkpoint)[0])}
        if args.evaluate_exposed_test:
            with np.load(run/'test_predictions.npz', allow_pickle=False) as saved:
                test_labels = saved['labels'].copy()
                test_logits = saved['logits'].copy()
            record['previously_exposed_test_original'] = evaluate(test_labels, predict_from_probabilities(probabilities(test_logits), None))
            record['previously_exposed_test_selected'] = evaluate(test_labels, apply_policy(test_logits, policy, checkpoint)[0])
        summary.append(record)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
