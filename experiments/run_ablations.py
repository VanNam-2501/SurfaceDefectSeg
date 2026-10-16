"""Ablation Baseline & Model Comparison Experiment Runner (Phase 2).

Executes the benchmark matrix:
- 3 Backbones: ResNet-18 (CNN), Swin-T (Transformer), VMamba-T (State-Space Model)
- 2 Architectures: Hierarchical Cascade (Stage 1 + Stage 2) vs Full 5-Class Baseline (full5)
- 3 Seeds: 42, 43, 44 @ resolution 224
- High-Resolution check: ResNet-18 Cascade @ resolution 384
- Metrics aggregated:
  * fnr: False Negative Rate (leaked defects)
  * fnr_ci_low, fnr_ci_high: Wilson 95% confidence interval
  * macro_recall_defect: Macro recall across the 4 defect categories (Bent, Loose, Missing, Pierced)
  * review_rate: Proportion of items routed to human operator review
  * ms_per_img: Inference latency per image in milliseconds

Outputs:
  - CSV report: ``experiments/ablation_summary.csv``

Usage:
  .venv/Scripts/python.exe experiments/run_ablations.py
  .venv/Scripts/python.exe experiments/run_ablations.py --smoke
  .venv/Scripts/python.exe experiments/run_ablations.py --epochs 15 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.straw_defect_cls.dataset import StrawDefectDataset, get_val_transform
from src.straw_defect_cls.metrics import wilson_score_interval
from src.straw_defect_cls.models import build_model
from src.straw_defect_cls.train_engine import TrainConfig, run_training


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def load_trained_model(
    model_name: str,
    stage: str,
    num_classes: int,
    resolution: int,
    seed: int,
    runs_dir: Path,
    device: torch.device,
) -> Optional[nn.Module]:
    """Load best checkpoint for a trained stage model."""
    tag = f"{model_name}_stage{stage}_r{resolution}_seed{seed}"
    ckpt_path = runs_dir / tag / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        return None

    try:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"[WARN] Failed to load checkpoint {ckpt_path}: {exc}")
        return None


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    num_warmup: int = 5,
    num_iters: int = 25,
) -> float:
    """Benchmark single-image inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # GPU warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start

    ms_per_img = (total_time / num_iters) * 1000.0
    return round(ms_per_img, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Procedures
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_full5_baseline(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate 5-class single-stage baseline model."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Defect detection FNR (True defect is classes 1, 2, 3, 4; Good is 0)
    is_true_defect = y_true > 0
    total_defects = int(np.sum(is_true_defect))
    false_negatives = int(np.sum(is_true_defect & (y_pred == 0)))
    fnr = float(false_negatives / max(1, total_defects))

    ci_low, ci_high = wilson_score_interval(false_negatives, total_defects, confidence=0.95)

    # Macro recall across defect classes (1: Bent, 2: Loose, 3: Missing, 4: Pierced)
    defect_recalls = []
    for defect_cls in [1, 2, 3, 4]:
        cls_mask = y_true == defect_cls
        if np.sum(cls_mask) > 0:
            rec = float(np.sum((y_true == defect_cls) & (y_pred == defect_cls)) / np.sum(cls_mask))
            defect_recalls.append(rec)
    macro_recall_defect = float(np.mean(defect_recalls)) if defect_recalls else 0.0

    return {
        "fnr": round(fnr, 4),
        "fnr_ci_low": round(ci_low, 4),
        "fnr_ci_high": round(ci_high, 4),
        "macro_recall_defect": round(macro_recall_defect, 4),
        "review_rate": 0.0,  # Standard baseline has no review mechanism
    }


@torch.inference_mode()
def evaluate_cascade_architecture(
    model_s1: nn.Module,
    model_s2: nn.Module,
    val_csv: Path,
    image_size: int,
    device: torch.device,
    tau_pass: float = 0.80,
    tau_defect: float = 0.65,
    tau2: float = 0.50,
    margin2: float = 0.15,
) -> Dict[str, float]:
    """Evaluate two-stage hierarchical cascade pipeline."""
    model_s1.eval()
    model_s2.eval()

    val_ds = StrawDefectDataset(csv_file=val_csv, task="full", transform=get_val_transform(image_size))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    total_samples = len(val_ds)
    all_decisions: list[str] = []
    all_final_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)

        # Stage 1: Good vs Defect
        out_s1 = model_s1(images)
        p1 = torch.softmax(out_s1, dim=1).cpu().numpy()

        # Stage 2: 4 Defect classes
        out_s2 = model_s2(images)
        p2 = torch.softmax(out_s2, dim=1).cpu().numpy()

        for i in range(len(labels)):
            lbl = labels[i].item()
            prob_good = p1[i, 0]
            prob_defect = p1[i, 1]

            # Stage 2 probabilities
            p2_sorted = np.sort(p2[i])[::-1]
            top1_p2 = p2_sorted[0]
            top2_p2 = p2_sorted[1] if len(p2_sorted) > 1 else 0.0
            margin = top1_p2 - top2_p2
            top1_class_s2 = int(np.argmax(p2[i])) + 1  # Map to 1..4

            # Decision Logic conforming to industrial specification:
            # 1. PASS if p1[Good] >= tau_pass
            if prob_good >= tau_pass:
                decision = "PASS"
                pred_cls = 0
            # 2. DEFECT if p1[Defect] >= tau_defect AND passes Stage 2 guard
            elif (prob_defect >= tau_defect) and (top1_p2 >= tau2) and (margin >= margin2):
                decision = "DEFECT"
                pred_cls = top1_class_s2
            # 3. Otherwise REVIEW
            else:
                decision = "REVIEW"
                pred_cls = -1  # Review queue

            all_decisions.append(decision)
            all_final_preds.append(pred_cls)
            all_labels.append(lbl)

    y_true = np.array(all_labels)
    y_pred = np.array(all_final_preds)
    decisions = np.array(all_decisions)

    # Metrics computation
    is_true_defect = y_true > 0
    total_defects = int(np.sum(is_true_defect))

    # Leaked defects: True defect incorrectly classified as PASS (pred_cls == 0)
    false_negatives = int(np.sum(is_true_defect & (y_pred == 0)))
    fnr = float(false_negatives / max(1, total_defects))
    ci_low, ci_high = wilson_score_interval(false_negatives, total_defects, confidence=0.95)

    # Review rate
    review_count = int(np.sum(decisions == "REVIEW"))
    review_rate = float(review_count / max(1, total_samples))

    # Macro recall across defect classes among diagnosed defects
    defect_recalls = []
    for defect_cls in [1, 2, 3, 4]:
        cls_mask = y_true == defect_cls
        if np.sum(cls_mask) > 0:
            rec = float(np.sum((y_true == defect_cls) & (y_pred == defect_cls)) / np.sum(cls_mask))
            defect_recalls.append(rec)
    macro_recall_defect = float(np.mean(defect_recalls)) if defect_recalls else 0.0

    return {
        "fnr": round(fnr, 4),
        "fnr_ci_low": round(ci_low, 4),
        "fnr_ci_high": round(ci_high, 4),
        "macro_recall_defect": round(macro_recall_defect, 4),
        "review_rate": round(review_rate, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Experiment Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_all_ablations(
    backbones: List[str],
    seeds: List[int],
    epochs: int = 15,
    batch_size: int = 32,
    smoke: bool = False,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Run full matrix of ablation baselines and generate summary table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = REPO_ROOT / "experiments" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_file = REPO_ROOT / "experiments" / "ablation_summary.csv"
    val_csv = REPO_ROOT / "data" / "straw_defects_splits" / "val.csv"

    # Load thresholds from configs/thresholds.json
    thr_file = REPO_ROOT / "configs" / "thresholds.json"
    tau_pass = 0.80
    tau_defect = 0.65
    tau2 = 0.50
    margin2 = 0.15
    if thr_file.exists():
        try:
            thr_data = json.loads(thr_file.read_text(encoding="utf-8"))
            tau_pass = float(thr_data.get("tau_pass", tau_pass))
            tau_defect = float(thr_data.get("tau_defect", tau_defect))
            tau2 = float(thr_data.get("tau2", tau2))
            margin2 = float(thr_data.get("margin2", margin2))
        except Exception:
            pass

    if smoke:
        backbones = [backbones[0]]
        seeds = [seeds[0]]
        epochs = 1

    records: List[Dict[str, Any]] = []

    # 1. Standard Resolution @224 Grid
    for backbone in backbones:
        for seed in seeds:
            # ──────────────────────────────────────────────────────────────────
            # A. Full 5-Class Baseline Model
            # ──────────────────────────────────────────────────────────────────
            full5_tag = f"{backbone}_stagefull5_r224_seed{seed}"
            full5_ckpt = runs_dir / full5_tag / "checkpoints" / "best.pt"

            if not full5_ckpt.exists() or not skip_existing:
                print(f"\n>>> Training baseline: {full5_tag}")
                try:
                    cfg = TrainConfig(
                        model_name=backbone,
                        stage="full5",
                        image_size=224,
                        seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        tag=full5_tag,
                    )
                    run_training(cfg)
                except Exception as exc:
                    print(f"[WARN] Failed training {full5_tag}: {exc}")
                    continue

            # Load and evaluate full5
            model_full5 = load_trained_model(backbone, "full5", 5, 224, seed, runs_dir, device)
            if model_full5 is not None:
                val_ds = StrawDefectDataset(csv_file=val_csv, task="full", transform=get_val_transform(224))
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                res_full5 = evaluate_full5_baseline(model_full5, val_loader, device)
                latency = measure_inference_latency(model_full5, device, image_size=224)

                records.append({
                    "backbone": backbone,
                    "arch": "full5",
                    "resolution": 224,
                    "seed": seed,
                    "fnr": res_full5["fnr"],
                    "fnr_ci_low": res_full5["fnr_ci_low"],
                    "fnr_ci_high": res_full5["fnr_ci_high"],
                    "macro_recall_defect": res_full5["macro_recall_defect"],
                    "review_rate": res_full5["review_rate"],
                    "ms_per_img": latency,
                })
                print(f"[Result] {backbone} full5 @224 s{seed} -> FNR={res_full5['fnr']:.4f} Latency={latency:.1f}ms")

            # ──────────────────────────────────────────────────────────────────
            # B. Two-Stage Cascade Architecture (Stage 1 + Stage 2)
            # ──────────────────────────────────────────────────────────────────
            s1_tag = f"{backbone}_stage1_r224_seed{seed}"
            s2_tag = f"{backbone}_stage2_r224_seed{seed}"
            s1_ckpt = runs_dir / s1_tag / "checkpoints" / "best.pt"
            s2_ckpt = runs_dir / s2_tag / "checkpoints" / "best.pt"

            # Train Stage 1 if needed
            if not s1_ckpt.exists() or not skip_existing:
                print(f"\n>>> Training Stage 1: {s1_tag}")
                try:
                    cfg_s1 = TrainConfig(
                        model_name=backbone,
                        stage="1",
                        image_size=224,
                        seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        tag=s1_tag,
                    )
                    run_training(cfg_s1)
                except Exception as exc:
                    print(f"[WARN] Failed Stage 1 {s1_tag}: {exc}")
                    continue

            # Train Stage 2 if needed
            if not s2_ckpt.exists() or not skip_existing:
                print(f"\n>>> Training Stage 2: {s2_tag}")
                try:
                    cfg_s2 = TrainConfig(
                        model_name=backbone,
                        stage="2",
                        image_size=224,
                        seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        tag=s2_tag,
                    )
                    run_training(cfg_s2)
                except Exception as exc:
                    print(f"[WARN] Failed Stage 2 {s2_tag}: {exc}")
                    continue

            # Load and evaluate cascade
            m_s1 = load_trained_model(backbone, "1", 2, 224, seed, runs_dir, device)
            m_s2 = load_trained_model(backbone, "2", 4, 224, seed, runs_dir, device)
            if m_s1 is not None and m_s2 is not None:
                res_cascade = evaluate_cascade_architecture(
                    m_s1, m_s2, val_csv, 224, device,
                    tau_pass=tau_pass, tau_defect=tau_defect, tau2=tau2, margin2=margin2,
                )
                lat_s1 = measure_inference_latency(m_s1, device, image_size=224)
                lat_s2 = measure_inference_latency(m_s2, device, image_size=224)
                # Weighted latency: Stage 1 is run 100% of the time, Stage 2 run on defects (~20% in val)
                cascade_lat = round(lat_s1 + 0.20 * lat_s2, 2)

                records.append({
                    "backbone": backbone,
                    "arch": "cascade",
                    "resolution": 224,
                    "seed": seed,
                    "fnr": res_cascade["fnr"],
                    "fnr_ci_low": res_cascade["fnr_ci_low"],
                    "fnr_ci_high": res_cascade["fnr_ci_high"],
                    "macro_recall_defect": res_cascade["macro_recall_defect"],
                    "review_rate": res_cascade["review_rate"],
                    "ms_per_img": cascade_lat,
                })
                print(f"[Result] {backbone} cascade @224 s{seed} -> FNR={res_cascade['fnr']:.4f} Latency={cascade_lat:.1f}ms")

    # 2. ResNet-18 Cascade @384 High-Resolution Experiment
    if not smoke:
        r384_seed = seeds[0]
        s1_tag_384 = f"resnet18_stage1_r384_seed{r384_seed}"
        s2_tag_384 = f"resnet18_stage2_r384_seed{r384_seed}"
        s1_ckpt_384 = runs_dir / s1_tag_384 / "checkpoints" / "best.pt"
        s2_ckpt_384 = runs_dir / s2_tag_384 / "checkpoints" / "best.pt"

        if not s1_ckpt_384.exists() or not skip_existing:
            print(f"\n>>> Training Stage 1 @384: {s1_tag_384}")
            try:
                cfg_s1_384 = TrainConfig(
                    model_name="resnet18", stage="1", image_size=384,
                    seed=r384_seed, epochs=epochs, batch_size=max(8, batch_size // 2), tag=s1_tag_384,
                )
                run_training(cfg_s1_384)
            except Exception as exc:
                print(f"[WARN] Failed Stage 1 @384: {exc}")

        if not s2_ckpt_384.exists() or not skip_existing:
            print(f"\n>>> Training Stage 2 @384: {s2_tag_384}")
            try:
                cfg_s2_384 = TrainConfig(
                    model_name="resnet18", stage="2", image_size=384,
                    seed=r384_seed, epochs=epochs, batch_size=max(8, batch_size // 2), tag=s2_tag_384,
                )
                run_training(cfg_s2_384)
            except Exception as exc:
                print(f"[WARN] Failed Stage 2 @384: {exc}")

        m_s1_384 = load_trained_model("resnet18", "1", 2, 384, r384_seed, runs_dir, device)
        m_s2_384 = load_trained_model("resnet18", "2", 4, 384, r384_seed, runs_dir, device)
        if m_s1_384 is not None and m_s2_384 is not None:
            res_cascade_384 = evaluate_cascade_architecture(
                m_s1_384, m_s2_384, val_csv, 384, device,
                tau_pass=tau_pass, tau_defect=tau_defect, tau2=tau2, margin2=margin2,
            )
            lat_s1_384 = measure_inference_latency(m_s1_384, device, image_size=384)
            lat_s2_384 = measure_inference_latency(m_s2_384, device, image_size=384)
            cascade_lat_384 = round(lat_s1_384 + 0.20 * lat_s2_384, 2)

            records.append({
                "backbone": "resnet18",
                "arch": "cascade",
                "resolution": 384,
                "seed": r384_seed,
                "fnr": res_cascade_384["fnr"],
                "fnr_ci_low": res_cascade_384["fnr_ci_low"],
                "fnr_ci_high": res_cascade_384["fnr_ci_high"],
                "macro_recall_defect": res_cascade_384["macro_recall_defect"],
                "review_rate": res_cascade_384["review_rate"],
                "ms_per_img": cascade_lat_384,
            })
            print(f"[Result] resnet18 cascade @384 s{r384_seed} -> FNR={res_cascade_384['fnr']:.4f} Latency={cascade_lat_384:.1f}ms")

    # Save summary dataframe
    df_summary = pd.DataFrame(records)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_file, index=False)
    print(f"\n[DONE] Ablation summary compiled with {len(df_summary)} configurations.")
    print(f"Saved to: {summary_file}")
    return df_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete ablation benchmark.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test mode")
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain existing runs")
    args = parser.parse_args()

    run_all_ablations(
        backbones=["resnet18", "swin_t", "vmamba_t"],
        seeds=[42, 43, 44],
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoke=args.smoke,
        skip_existing=not args.force_retrain,
    )


if __name__ == "__main__":
    main()
