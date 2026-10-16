"""Leave-One-Brand-Out Cross-Validation (LOBO-CV) (Phase 4).

Protocol:
- 8-Fold Cross-Validation across all 8 beverage brands:
  ['Choco', 'Chocolait', 'Chuckie', 'Dutchmill Purple', 'Dutchmill Yellow', 'Moo', 'Nutriboost', 'Selecta']
- In each fold:
  * Held-out brand = Test Set (250 images: 50 per class)
  * Remaining 7 brands = Training Set (1,750 images: 350 per class)
- Evaluates ResNet-18 in both architectures:
  1. Two-Stage Cascade (Stage 1 + Stage 2 via industrial decide.py)
  2. Single-Stage Baseline (full5)
- Outputs:
  * Brand confusion matrices in ``submission/figures/lobo_cm_*.png``
  * Summary table in ``submission/tables/lobo_summary.csv``

Usage:
    .venv/Scripts/python.exe experiments/lobo_cv.py --smoke
    .venv/Scripts/python.exe experiments/lobo_cv.py --epochs 5
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.straw_defect_cls.dataset import (
    FULL_CLASS_NAMES,
    StrawDefectDataset,
    get_train_transform,
    get_val_transform,
)
from src.straw_defect_cls.decide import decide
from src.straw_defect_cls.losses import CrossEntropyWithLabelSmoothing
from src.straw_defect_cls.metrics import compute_audit_metrics, plot_confusion_matrix, wilson_score_interval
from src.straw_defect_cls.models import build_model


BRANDS = [
    "Choco",
    "Chocolait",
    "Chuckie",
    "Dutchmill Purple",
    "Dutchmill Yellow",
    "Moo",
    "Nutriboost",
    "Selecta",
]


# ──────────────────────────────────────────────────────────────────────────────
# LOBO Split Builder
# ──────────────────────────────────────────────────────────────────────────────
def create_lobo_splits(
    full_csv: Path,
    held_out_brand: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create train and test CSVs for a specific Leave-One-Brand-Out fold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(full_csv)

    brand_col = "brand" if "brand" in df.columns else "brand_name"

    test_df = df[df[brand_col].str.lower() == held_out_brand.lower()].copy()
    train_df = df[df[brand_col].str.lower() != held_out_brand.lower()].copy()

    safe_name = held_out_brand.lower().replace(" ", "_")
    train_csv = output_dir / f"train_lobo_{safe_name}.csv"
    test_csv = output_dir / f"test_lobo_{safe_name}.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, test_csv


# ──────────────────────────────────────────────────────────────────────────────
# Fast Model Trainer for LOBO Folds
# ──────────────────────────────────────────────────────────────────────────────
def train_lobo_model(
    model_name: str,
    train_csv: Path,
    num_classes: int,
    task: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    is_stage1: bool = False,
    is_stage2: bool = False,
    max_batches: Optional[int] = None,
) -> nn.Module:
    """Train model on a LOBO training split."""
    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)

    train_ds = StrawDefectDataset(csv_file=train_csv, task=task, transform=get_train_transform(224))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    if is_stage1:
        # Class-weighted CE penalizing false negatives
        weight = torch.tensor([1.0, 3.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif is_stage2:
        criterion = CrossEntropyWithLabelSmoothing(num_classes=4, smoothing=0.1).to(device)
    else:
        criterion = CrossEntropyWithLabelSmoothing(num_classes=5, smoothing=0.05).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    model.train()
    for _ in range(epochs):
        for b_idx, (imgs, lbls) in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outs = model(imgs)
                loss = criterion(outs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main LOBO Routine
# ──────────────────────────────────────────────────────────────────────────────
def run_lobo_cross_validation(
    brands: Optional[List[str]] = None,
    epochs: int = 4,
    batch_size: int = 32,
    smoke: bool = False,
) -> pd.DataFrame:
    """Execute 8-Fold Leave-One-Brand-Out Cross-Validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if brands is None:
        brands = BRANDS

    if smoke:
        brands = [brands[0]]
        epochs = 1

    tables_dir = REPO_ROOT / "submission" / "tables"
    figures_dir = REPO_ROOT / "submission" / "figures"
    lobo_splits_dir = REPO_ROOT / "data" / "lobo_splits"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Master dataset splits to combine into all 2000 images
    splits_dir = REPO_ROOT / "data" / "straw_defects_splits"
    train_m = pd.read_csv(splits_dir / "train.csv")
    val_m = pd.read_csv(splits_dir / "val.csv")
    test_m = pd.read_csv(splits_dir / "test.csv")
    all_df = pd.concat([train_m, val_m, test_m], ignore_index=True)
    master_csv = lobo_splits_dir / "all_samples.csv"
    lobo_splits_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(master_csv, index=False)

    # Threshold configuration
    thr_file = REPO_ROOT / "configs" / "thresholds.json"
    cfg = json.loads(thr_file.read_text(encoding="utf-8")) if thr_file.exists() else {
        "tau_pass": 0.85, "tau_defect": 0.50, "tau2": 0.50, "margin2": 0.15, "e_thr": -2.0
    }

    print("\n" + "="*70)
    print("PHASE 4: LEAVE-ONE-BRAND-OUT (LOBO) CROSS-VALIDATION")
    print(f"Folds: {len(brands)} | Epochs per model: {epochs} | Device: {device}")
    print("="*70)

    summary_records = []

    for fold_idx, brand in enumerate(brands, 1):
        print(f"\n--- [Fold {fold_idx}/{len(brands)}] Held-Out Brand: {brand} ---")
        train_csv, test_csv = create_lobo_splits(master_csv, brand, lobo_splits_dir)

        max_b = 3 if smoke else None

        # ──────────────────────────────────────────────────────────────────────
        # 1. Train & Evaluate Single-Stage Baseline (full5)
        # ──────────────────────────────────────────────────────────────────────
        print(f"  Training full5 baseline for {brand}...")
        m_full5 = train_lobo_model(
            "resnet18", train_csv, num_classes=5, task="full",
            epochs=epochs, batch_size=batch_size, device=device,
            max_batches=max_b,
        )

        test_ds = StrawDefectDataset(csv_file=test_csv, task="full", transform=get_val_transform(224))
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        preds_full5, true_lbls = [], []
        with torch.no_grad():
            for t_idx, (imgs, lbls) in enumerate(test_loader):
                if smoke and t_idx >= 2:
                    break
                imgs = imgs.to(device)
                outs = m_full5(imgs)
                preds = outs.argmax(dim=1).cpu().numpy().tolist()
                preds_full5.extend(preds)
                true_lbls.extend(lbls.numpy().tolist())

        y_true = np.array(true_lbls)
        p_full5 = np.array(preds_full5)
        dec_full5 = np.where(p_full5 == 0, "PASS", "DEFECT")

        m_res_full5 = compute_audit_metrics(y_true, dec_full5, p_full5, cost_matrix=cfg.get("cost_matrix"))

        summary_records.append({
            "fold": fold_idx,
            "held_out_brand": brand,
            "architecture": "Single-Stage Baseline (full5)",
            "fnr": m_res_full5["fnr"],
            "fnr_ci_95": f"[{m_res_full5['fnr_ci_low']:.4f}, {m_res_full5['fnr_ci_high']:.4f}]",
            "fpr": m_res_full5["fpr"],
            "macro_recall_defect": m_res_full5["macro_recall_defect"],
            "review_rate": m_res_full5["review_rate"],
            "cost_per_1000": m_res_full5["expected_cost_per_1000"],
        })

        # Save confusion matrix for full5
        cm_full5 = confusion_matrix(y_true, p_full5, labels=range(5))
        safe_brand = brand.lower().replace(" ", "_")
        plot_confusion_matrix(cm_full5, FULL_CLASS_NAMES, figures_dir / f"lobo_cm_{safe_brand}_full5.png")

        # ──────────────────────────────────────────────────────────────────────
        # 2. Train & Evaluate Two-Stage Hierarchical Cascade
        # ──────────────────────────────────────────────────────────────────────
        print(f"  Training Two-Stage Cascade (Stage 1 + Stage 2) for {brand}...")
        m_s1 = train_lobo_model(
            "resnet18", train_csv, num_classes=2, task="stage1",
            epochs=epochs, batch_size=batch_size, device=device, is_stage1=True,
            max_batches=max_b,
        )
        m_s2 = train_lobo_model(
            "resnet18", train_csv, num_classes=4, task="stage2",
            epochs=epochs, batch_size=batch_size, device=device, is_stage2=True,
            max_batches=max_b,
        )

        dec_cascade, preds_cascade = [], []
        with torch.no_grad():
            for t_idx, (imgs, lbls) in enumerate(test_loader):
                if smoke and t_idx >= 2:
                    break
                imgs = imgs.to(device)
                out1 = m_s1(imgs).cpu().numpy()
                out2 = m_s2(imgs).cpu().numpy()
                for i in range(len(lbls)):
                    dec, d_cls, rsn = decide(out1[i], out2[i], cfg)
                    dec_cascade.append(dec)
                    if dec == "PASS":
                        preds_cascade.append(0)
                    elif dec == "DEFECT":
                        preds_cascade.append(int(d_cls) + 1)
                    else:
                        preds_cascade.append(-1)

        p_casc = np.array(preds_cascade)
        m_res_casc = compute_audit_metrics(y_true, dec_cascade, p_casc, cost_matrix=cfg.get("cost_matrix"))

        summary_records.append({
            "fold": fold_idx,
            "held_out_brand": brand,
            "architecture": "Two-Stage Hierarchical",
            "fnr": m_res_casc["fnr"],
            "fnr_ci_95": f"[{m_res_casc['fnr_ci_low']:.4f}, {m_res_casc['fnr_ci_high']:.4f}]",
            "fpr": m_res_casc["fpr"],
            "macro_recall_defect": m_res_casc["macro_recall_defect"],
            "review_rate": m_res_casc["review_rate"],
            "cost_per_1000": m_res_casc["expected_cost_per_1000"],
        })

        # Save confusion matrix for cascade (valid predictions only, map -1 review to diagonal or ignore)
        valid_mask = p_casc >= 0
        cm_casc = confusion_matrix(y_true[valid_mask], p_casc[valid_mask], labels=range(5))
        plot_confusion_matrix(cm_casc, FULL_CLASS_NAMES, figures_dir / f"lobo_cm_{safe_brand}_cascade.png")

        print(
            f"  [Summary {brand}] Cascade FNR: {m_res_casc['fnr']*100:.2f}% | "
            f"Full5 FNR: {m_res_full5['fnr']*100:.2f}% | Review: {m_res_casc['review_rate']*100:.1f}%"
        )

    summary_df = pd.DataFrame(summary_records)
    out_csv = tables_dir / "lobo_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved LOBO-CV summary table to: {out_csv}")
    print(f"[DONE] Saved confusion matrices to: {figures_dir}")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Leave-One-Brand-Out Cross-Validation.")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-fold 1-epoch smoke test")
    args = parser.parse_args()

    run_lobo_cross_validation(epochs=args.epochs, batch_size=args.batch_size, smoke=args.smoke)


if __name__ == "__main__":
    main()
