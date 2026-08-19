"""Kaggle launcher for the frozen three-model training protocol.

Run this script from a writable copy of the project in ``/kaggle/working``.
The dataset itself may stay read-only under ``/kaggle/input``.  The launcher
only writes experiment evidence and checkpoints beneath ``--output-dir``.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    train_script: str
    result_folder: str
    batch_size: int
    grad_accum: int


SPECS = {
    "unet": ModelSpec("unet", "train_unet.py", "unet_r18", batch_size=2, grad_accum=2),
    "segformer": ModelSpec("segformer", "train_segformer.py", "segformer_b0", batch_size=2, grad_accum=2),
    "vmamba": ModelSpec("vmamba", "train_vmamba.py", "vmamba_t_s2l5", batch_size=1, grad_accum=4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate U-Net, SegFormer and VMamba on Kaggle."
    )
    parser.add_argument("--dataset-root", required=True, help="training_dataset or 3CAD-ANI root")
    parser.add_argument("--model", choices=["all", *SPECS], default="all")
    parser.add_argument("--output-dir", default="/kaggle/working/results")
    parser.add_argument("--run-name", default="cleaned_v1_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Physical batch override. 0 uses the model-safe default.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=0,
        help="Gradient accumulation override. 0 uses the model-safe default.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--val-tile-batch-size", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-protocol-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    return parser.parse_args()


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    training_root = Path(__file__).resolve().parent
    project_root = training_root.parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    splits_root = dataset_root / "dataset_audit" / "splits"
    split_paths = {
        "train": splits_root / "train.csv",
        "val": splits_root / "val.csv",
        "test": splits_root / "test.csv",
    }
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    for name, path in split_paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name} split: {path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(SPECS.values()) if args.model == "all" else [SPECS[args.model]]

    # A unique run name prevents accidental overwriting of a completed Kaggle
    # output when a notebook is resumed or re-run.
    for spec in selected:
        run_dir = output_dir / spec.result_folder / args.run_name
        if run_dir.exists():
            raise SystemExit(
                f"Run already exists: {run_dir}\n"
                "Choose a new --run-name; use the underlying train_*.py with "
                "--resume only when intentionally resuming that exact run."
            )

    print(f"Project root : {project_root}")
    print(f"Dataset root : {dataset_root}")
    print(f"Output root  : {output_dir}")
    print("Models       : " + ", ".join(spec.key for spec in selected))

    if not args.skip_protocol_check:
        run(
            [
                sys.executable,
                str(project_root / "scripts" / "verification" / "check_protocol.py"),
                "--dataset-root", str(dataset_root),
                "--train-csv", str(split_paths["train"]),
                "--val-csv", str(split_paths["val"]),
                "--test-csv", str(split_paths["test"]),
                "--save", str(output_dir / f"protocol_check_{args.run_name}.json"),
            ],
            project_root,
            args.dry_run,
        )

    # VMamba's custom CUDA selective-scan must be validated before spending
    # GPU quota on U-Net/SegFormer when all models were requested.
    if any(spec.key == "vmamba" for spec in selected):
        run(
            [sys.executable, str(project_root / "tests" / "ml" / "test_vmamba_runtime.py")],
            project_root,
            args.dry_run,
        )

    for spec in selected:
        print(f"\n===== TRAIN {spec.key.upper()} =====", flush=True)
        batch_size = args.batch_size or spec.batch_size
        grad_accum = args.grad_accum or spec.grad_accum
        if batch_size < 1 or grad_accum < 1:
            raise SystemExit("--batch-size and --grad-accum must be positive, or 0 for defaults.")
        train_args = [
            sys.executable,
            str(training_root / spec.train_script),
            "--dataset-root", str(dataset_root),
            "--train-csv", str(split_paths["train"]),
            "--val-csv", str(split_paths["val"]),
            "--test-csv", str(split_paths["test"]),
            "--output-dir", str(output_dir),
            "--run-name", args.run_name,
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(batch_size),
            "--grad-accum", str(grad_accum),
            "--num-workers", str(args.num_workers),
            "--patch-size", str(args.patch_size),
            "--stride", str(args.stride),
            "--val-tile-batch-size", str(args.val_tile_batch_size),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--early-stopping-min-delta", str(args.early_stopping_min_delta),
            "--lr-patience", str(args.lr_patience),
            "--lr-factor", str(args.lr_factor),
            "--val-every", str(args.val_every),
            "--device", "cuda",
        ]
        run(train_args, project_root, args.dry_run)

        checkpoint = output_dir / spec.result_folder / args.run_name / "checkpoints" / "best.pt"
        if not args.dry_run and not checkpoint.is_file():
            raise SystemExit(f"Best checkpoint was not produced: {checkpoint}")

        evaluation_args = [
            sys.executable,
            str(project_root / "scripts" / "evaluation" / "evaluate_model.py"),
            "--model", spec.key,
            "--checkpoint", str(checkpoint),
            "--dataset-root", str(dataset_root),
            "--train-csv", str(split_paths["train"]),
            "--val-csv", str(split_paths["val"]),
            "--test-csv", str(split_paths["test"]),
            "--tile-size", str(args.patch_size),
            "--stride", str(args.stride),
            "--tile-batch-size", str(args.val_tile_batch_size),
            "--device", "cuda",
        ]
        print(f"\n===== VALIDATION / THRESHOLD SELECTION: {spec.key.upper()} =====", flush=True)
        run([*evaluation_args, "--split", "val"], project_root, args.dry_run)

        if not args.skip_test:
            print(f"\n===== TEST (FROZEN VALIDATION THRESHOLD): {spec.key.upper()} =====", flush=True)
            run([*evaluation_args, "--split", "test"], project_root, args.dry_run)

    print(f"\nCompleted. Results are in: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
