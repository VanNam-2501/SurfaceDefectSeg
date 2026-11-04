"""Aggregate completed fold reports; never confuse brand variation with seed variation."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Directory containing lobo_<brand> result folders')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-folds', type=int, default=8)
    args = parser.parse_args()
    parts, configs, brands = [], set(), set()
    for path in sorted(args.root.glob('*/report/all_results.csv')):
        run = path.parent.parent
        status = json.loads((path.parent / 'status.json').read_text())
        if status['status'] != 'COMPLETED INTERNAL BENCHMARK':
            raise ValueError(f'Incomplete/smoke fold: {run}')
        experiment = json.loads((run / 'experiment.json').read_text())
        # Only fold protocols identify LOBO results; require a sibling protocol record exported by runner.
        context = json.loads((run / 'protocol_context.json').read_text())
        brand = context.get('held_out_brand')
        if not brand or brand in brands:
            raise ValueError(f'Missing or repeated held-out brand in {run}')
        brands.add(brand)
        configs.add(json.dumps(experiment['config'], sort_keys=True))
        part = pd.read_csv(path)
        part['held_out_brand'] = brand
        parts.append(part)
    if len(parts) != args.expected_folds or len(configs) != 1:
        raise ValueError(f'Expected {args.expected_folds} unique folds with identical config; found {len(parts)} folds and {len(configs)} configs')
    frame = pd.concat(parts, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'lobo_per_brand.csv', index=False)
    metrics = ['fnr', 'fpr', 'macro_f1_all', 'macro_recall_defect', 'review_rate', 'deployment_review_rate', 'cost_per_1000']
    summary = frame.groupby(['condition', 'variant'])[metrics].agg(['mean', 'std', 'min', 'max', 'count'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(args.output / 'lobo_mean_std_across_brands.csv')
    print(f'Aggregated {len(brands)} held-out brands. SD is across brands, not random seeds.')


if __name__ == '__main__':
    main()
