"""Dataset preparation script for Juice Box Straw Defect Classification.

Scans the dataset (from zip or folder), parses brand & defect class hierarchy,
and creates unified stratified splits for:
- Stage 1: Triage (Good vs Defective)
- Stage 2: Diagnosis (Bent, Loose, Missing, Pierced)
- 5-class Full baseline
"""
import argparse
import sys
import zipfile
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.straw_defect_cls.dataset import split_dataset


def extract_if_zip(source_path: Path, extract_dir: Path) -> Path:
    """Extract zip archive if source points to a .zip file."""
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        print(f"Extracting {source_path} to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path, "r") as z:
            z.extractall(extract_dir)
        print("Extraction complete.")
        return extract_dir
    return source_path


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Hierarchical Straw Defect Classification")
    parser.add_argument(
        "--source",
        type=str,
        default=str(repo_root / "archive.zip"),
        help="Path to raw dataset folder or archive.zip",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "data" / "straw_defects_splits"),
        help="Path to output split CSVs and statistics",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")

    args = parser.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        # Fallback check
        alt_raw = repo_root / "data" / "straw_defects_raw"
        if alt_raw.exists():
            src_path = alt_raw
        else:
            raise FileNotFoundError(f"Source path {src_path} does not exist.")

    # Auto extract if zip
    extract_target = repo_root / "data" / "straw_defects_raw"
    final_data_dir = extract_if_zip(src_path, extract_target)

    # Perform stratified split
    out_dir = Path(args.output)
    stats = split_dataset(
        root_dir=final_data_dir,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("\n[SUCCESS] Hierarchical dataset splits ready at:", out_dir)


if __name__ == "__main__":
    main()

