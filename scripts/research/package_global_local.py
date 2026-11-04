"""Source-only Kaggle bundle; frozen CSV metadata included, no images/checkpoints."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def source_files():
    files = list((ROOT / "src/straw_global_local").glob("*.py"))
    files += list((ROOT / "configs").glob("straw_global_local*.json"))
    files += [ROOT / p for p in ["src/straw_defect_cls/__init__.py", "src/straw_defect_cls/thesis_protocol.py",
              "requirements/straw-thesis.txt", "tests/test_global_local.py", "scripts/research/package_global_local.py",
              "scripts/research/summarize_global_local_pilots.py",
              "docs/STRAW_GLOBAL_LOCAL_RUNBOOK.md", "docs/STRAW_COMPLETION_REQUIREMENTS.md",
              "docs/STRAW_GLOBAL_LOCAL_VERIFICATION.md",
              "docs/STRAW_RESEARCH_PLAN.md", "docs/STRAW_RESEARCH_SYNTHESIS.md",
              "experiments/notebooks/straw_global_local_kaggle.ipynb"]]
    source_protocol = ROOT / "artifacts/straw_thesis/data_v2/iid"
    files += [source_protocol / name for name in ["protocol.json", "inventory.csv", "train.csv", "val.csv", "calibration.csv", "test.csv"]]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/packages/straw_global_local_source.zip")
    args = parser.parse_args()
    files = source_files()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"{args.output}: {len(set(files))} files, {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
