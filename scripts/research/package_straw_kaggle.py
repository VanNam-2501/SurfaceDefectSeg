"""Package only straw thesis sources; no datasets, old checkpoints or credentials."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/packages/straw_thesis_source.zip')
    args = parser.parse_args()
    files = list((ROOT / 'src/straw_defect_cls').glob('*.py'))
    files += [ROOT / 'configs/straw_thesis.json', ROOT / 'requirements/straw-thesis.txt',
              ROOT / 'scripts/research/aggregate_straw_lobo.py',
              ROOT / 'tests/test_straw_thesis.py', ROOT / 'tests/test_decide.py',
              ROOT / 'docs/STRAW_THESIS.md', ROOT / 'experiments/notebooks/straw_thesis_kaggle.ipynb']
    files += [ROOT / 'docs/STRAW_RESEARCH_REVIEW.md']
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'Created {args.output}: {len(files)} source files, {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
