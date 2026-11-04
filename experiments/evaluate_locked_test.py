"""Audit-Grade Evaluation on Locked Test Set (Phase 4).

Protocol:
- Evaluated ONCE on locked test set (both clean test and factory-degraded test).
- Compares Two-Stage Hierarchical Cascade vs Single-Stage 5-Class Baseline (full5).
- Uses industrial post-processing from configs/thresholds.json:
  * Temperature scaling
  * Bayes prior shift (pi_deploy = 5%)
  * Neyman-Pearson tau_pass gate
  * Multi-barrier Stage 2 guards
- Calculates:
  * False Negative Rate (FNR) with Wilson 95% Confidence Interval
  * False Positive Rate (FPR)
  * Per-class empirical recall
  * Review rate (operator queue)
  * Reweighted deployment accuracy under pi_deploy = 5%
  * Expected operational cost per 1,000 cartons ($100:10:1 cost matrix)
  * Inference latency (ms/carton)

Outputs:
  - CSV report: ``submission/tables/test_results.csv``
  - Markdown table: ``submission/tables/test_results.md``

Usage:
    .venv/Scripts/python.exe experiments/evaluate_locked_test.py
    .venv/Scripts/python.exe experiments/evaluate_locked_test.py --backbone resnet18 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.data.make_degraded_set import degrade_image, degraded_filename
from src.straw_defect_cls.dataset import StrawDefectDataset, get_val_transform
from src.straw_defect_cls.decide import decide
from src.straw_defect_cls.metrics import compute_audit_metrics
from src.straw_defect_cls.models import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def ensure_degraded_test_set(test_csv: Path, output_dir: Path, seed: int = 42) -> Path:
    """Generate degraded test set if not already created."""
    out_csv = output_dir / "test_degraded.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        paths = existing["image_path"].astype(str)
        expected_count = len(pd.read_csv(test_csv))
        if len(existing) == expected_count and paths.nunique() == len(existing) and all(Path(p).is_file() for p in paths):
            return out_csv
        raise ValueError(f"Existing degraded set has duplicate or missing image files: {out_csv}")

    print(f"\n[INFO] Generating factory-degraded test set at {output_dir}...")
    import random
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(test_csv)
    records = []

    for idx, row in df_test.iterrows():
        orig_p = Path(str(row["image_path"]))
        if not orig_p.exists():
            orig_p = REPO_ROOT / orig_p
        if not orig_p.exists():
            continue

        pil_im = Image.open(orig_p).convert("RGB")
        deg_np = degrade_image(np.asarray(pil_im))

        out_img_p = images_dir / degraded_filename(orig_p, row)
        Image.fromarray(deg_np).save(out_img_p, quality=88)

        rec = dict(row)
        rec["image_path"] = str(out_img_p)
        rec["is_degraded"] = True
        records.append(rec)

    deg_df = pd.DataFrame(records)
    if len(deg_df) != len(df_test) or deg_df["image_path"].nunique() != len(deg_df):
        raise RuntimeError(
            f"Generated degraded test set is incomplete or has path collisions: "
            f"rows={len(deg_df)}/{len(df_test)}, unique_paths={deg_df['image_path'].nunique()}"
        )
    deg_df.to_csv(out_csv, index=False)
    print(f"[DONE] Degraded test set created: {len(deg_df)} samples at {out_csv}")
    return out_csv


def measure_model_latency(model: nn.Module, device: torch.device, image_size: int = 224) -> float:
    """Measure single-carton inference latency in ms."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = ((time.perf_counter() - t0) / 30.0) * 1000.0
    return round(lat, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Procedures
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_cascade_on_set(
    model_s1: nn.Module,
    model_s2: nn.Module,
    csv_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Evaluate Two-Stage Hierarchical Cascade using industrial decision engine."""
    model_s1.eval()
    model_s2.eval()

    # Legacy stage checkpoints predate EXIF normalization; match their training input.
    ds = StrawDefectDataset(csv_file=csv_path, task="full", transform=get_val_transform(image_size), exif_transpose=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_y_true: List[int] = []
    all_decisions: List[str] = []
    all_pred_classes: List[int] = []

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        out1 = model_s1(imgs).cpu().numpy()
        out2 = model_s2(imgs).cpu().numpy()

        for i in range(len(lbls)):
            decision, defect_cls, reason = decide(out1[i], out2[i], cfg)
            all_decisions.append(decision)
            all_y_true.append(int(lbls[i].item()))

            if decision == "PASS":
                pred_c = 0
            elif decision == "DEFECT":
                pred_c = int(defect_cls) + 1  # Map 0..3 to 1..4 (Bent, Loose, Missing, Pierced)
            else:
                pred_c = -1  # Review queue

            all_pred_classes.append(pred_c)

    lat_s1 = measure_model_latency(model_s1, device, image_size=image_size)
    lat_s2 = measure_model_latency(model_s2, device, image_size=image_size)
    # Stage 1 evaluates 100%, Stage 2 evaluates on suspected defects (~20% in val/test)
    total_lat = round(lat_s1 + 0.20 * lat_s2, 2)

    metrics = compute_audit_metrics(
        y_true=all_y_true,
        decisions=all_decisions,
        predicted_classes=all_pred_classes,
        cost_matrix=cfg.get("cost_matrix"),
        pi_deploy=float(cfg.get("pi_deploy_defect", 0.05)),
        latency_ms=total_lat,
    )
    return metrics


@torch.inference_mode()
def evaluate_full5_on_set(
    model_full5: nn.Module,
    csv_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Evaluate Single-Stage 5-Class Baseline Model."""
    model_full5.eval()
    # Legacy full5 checkpoint predates EXIF normalization; match training input.
    ds = StrawDefectDataset(csv_file=csv_path, task="full", transform=get_val_transform(image_size), exif_transpose=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_y_true: List[int] = []
    all_decisions: List[str] = []
    all_pred_classes: List[int] = []

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        outs = model_full5(imgs).cpu().numpy()
        preds = np.argmax(outs, axis=1)

        for i in range(len(lbls)):
            p_cls = int(preds[i])
            decision = "PASS" if p_cls == 0 else "DEFECT"
            all_decisions.append(decision)
            all_pred_classes.append(p_cls)
            all_y_true.append(int(lbls[i].item()))

    lat = measure_model_latency(model_full5, device, image_size=image_size)

    metrics = compute_audit_metrics(
        y_true=all_y_true,
        decisions=all_decisions,
        predicted_classes=all_pred_classes,
        cost_matrix=cfg.get("cost_matrix"),
        pi_deploy=float(cfg.get("pi_deploy_defect", 0.05)),
        latency_ms=lat,
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main Audit Evaluation Function
# ──────────────────────────────────────────────────────────────────────────────
def run_locked_test_evaluation(
    backbone: str = "resnet18",
    seed: int = 42,
    resolution: int = 224,
) -> pd.DataFrame:
    """Execute single-pass locked test set evaluation on clean and degraded conditions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = REPO_ROOT / "experiments" / "runs"
    tables_dir = REPO_ROOT / "submission" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_clean_csv = REPO_ROOT / "data" / "straw_defects_splits" / "test.csv"
    test_degraded_dir = REPO_ROOT / "data" / "test_degraded_v2"
    test_degraded_csv = ensure_degraded_test_set(test_clean_csv, test_degraded_dir, seed=seed)

    # Load thresholds and post-processing configs
    thr_file = REPO_ROOT / "configs" / "thresholds.json"
    if not thr_file.exists():
        raise FileNotFoundError(f"Missing {thr_file}. Run Phase 3 calibration first.")
    cfg = json.loads(thr_file.read_text(encoding="utf-8"))

    # Load Models
    tag_s1 = f"{backbone}_stage1_r{resolution}_seed{seed}"
    tag_s2 = f"{backbone}_stage2_r{resolution}_seed{seed}"
    tag_full5 = f"{backbone}_stagefull5_r{resolution}_seed{seed}"

    ckpt_s1 = runs_dir / tag_s1 / "checkpoints" / "best.pt"
    ckpt_s2 = runs_dir / tag_s2 / "checkpoints" / "best.pt"
    ckpt_full5 = runs_dir / tag_full5 / "checkpoints" / "best.pt"

    print("\n" + "="*70)
    print("PHASE 4: AUDIT-GRADE EVALUATION ON LOCKED TEST SET")
    print(f"Backbone: {backbone} | Resolution: {resolution} | Seed: {seed}")
    print("="*70)

    # Load Cascade Models
    model_s1 = build_model(backbone, num_classes=2, pretrained=False).to(device)
    model_s1.load_state_dict(torch.load(ckpt_s1, map_location=device, weights_only=False)["model"])

    model_s2 = build_model(backbone, num_classes=4, pretrained=False).to(device)
    model_s2.load_state_dict(torch.load(ckpt_s2, map_location=device, weights_only=False)["model"])

    # Load Full5 Baseline
    model_full5 = build_model(backbone, num_classes=5, pretrained=False).to(device)
    model_full5.load_state_dict(torch.load(ckpt_full5, map_location=device, weights_only=False)["model"])

    # Run evaluations
    eval_matrix = [
        ("Two-Stage Cascade", "Clean", model_s1, model_s2, test_clean_csv, True),
        ("Two-Stage Cascade", "Factory Degraded", model_s1, model_s2, test_degraded_csv, True),
        ("Single-Stage Baseline (full5)", "Clean", model_full5, None, test_clean_csv, False),
        ("Single-Stage Baseline (full5)", "Factory Degraded", model_full5, None, test_degraded_csv, False),
    ]

    records = []
    for arch_name, condition, m1, m2, data_path, is_cascade in eval_matrix:
        print(f"\nEvaluating {arch_name} on [{condition} Test Set]...")
        if is_cascade:
            m = evaluate_cascade_on_set(m1, m2, data_path, cfg, device, image_size=resolution)
        else:
            m = evaluate_full5_on_set(m1, data_path, cfg, device, image_size=resolution)

        rec = {
            "architecture": arch_name,
            "condition": condition,
            "fnr": m["fnr"],
            "fnr_ci_95": f"[{m['fnr_ci_low']:.4f}, {m['fnr_ci_high']:.4f}]",
            "fpr": m["fpr"],
            "recall_good": m["recalls"].get("good", 0.0),
            "recall_bent": m["recalls"].get("bent", 0.0),
            "recall_loose": m["recalls"].get("loose", 0.0),
            "recall_missing": m["recalls"].get("missing", 0.0),
            "recall_pierced": m["recalls"].get("pierced", 0.0),
            "macro_recall_defect": m["macro_recall_defect"],
            "review_rate": m["review_rate"],
            "reweighted_acc_deploy": m["reweighted_accuracy"],
            "cost_per_1000_cartons": m["expected_cost_per_1000"],
            "latency_ms": m["latency_ms"],
        }
        records.append(rec)
        print(
            f"  -> FNR: {m['fnr']*100:.2f}% (CI: [{m['fnr_ci_low']*100:.2f}%, {m['fnr_ci_high']*100:.2f}%]) | "
            f"Review Rate: {m['review_rate']*100:.1f}% | Cost/1000: ${m['expected_cost_per_1000']:.1f} | Latency: {m['latency_ms']:.1f}ms"
        )

    res_df = pd.DataFrame(records)
    out_csv = tables_dir / "test_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved test results to: {out_csv}")

    # Generate Markdown Table ready for thesis inclusion
    md_content = "# Bảng Kết Quả Đánh Giá Trên Tập Kiểm Thử Khóa (Locked Test Set)\n\n"
    md_content += res_df.to_markdown(index=False)
    md_content += "\n\n*Ghi chú: Đánh giá duy nhất một lần (single-pass locked test). Ảnh degraded được dựng từ từng dòng test bằng tên file có thương hiệu và mã băm để tránh ghi đè; ngưỡng cascade được hiệu chuẩn trên `val_degraded_v2`. Chi phí dự tính trên 1.000 hộp đồ uống dựa trên ma trận chi phí c_FN:c_REVIEW:c_FP = 100:10:1 và tỉ lệ lỗi sản xuất pi_deploy = 5%.*\n"
    out_md = tables_dir / "test_results.md"
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[DONE] Saved markdown report table to: {out_md}")

    return res_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on locked test set.")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--resolution", type=int, default=224, help="Resolution")
    args = parser.parse_args()

    run_locked_test_evaluation(backbone=args.backbone, seed=args.seed, resolution=args.resolution)


if __name__ == "__main__":
    main()
