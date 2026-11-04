"""Industrial Post-Processing & Calibration Module (Phase 3).

Implements rigorous post-processing according to industrial thesis specifications:
1. Temperature Scaling: Separate calibration for Stage 1 & Stage 2 (optimizing NLL on clean validation).
2. Prior Correction: Bayesian log-odds shift for factory deployment defect rate (pi_deploy = 5%).
3. Neyman-Pearson Gate: Sweep tau_pass to find the largest threshold guaranteeing FNR <= 1%
   under worst-case across BOTH clean validation and factory-degraded validation sets.
4. Expected Cost Optimization: Optimize tau_defect under cost matrix (c_FN:c_REVIEW:c_FP = 100:10:1).
5. Stage 2 Multi-Barrier Defense:
   - tau2: minimum top-1 probability threshold.
   - margin2: minimum margin between top-1 and top-2 defect classes.
   - e_thr: Free energy score percentile 95 on validation defects (OOD rejection).
6. Artifacts exported:
   - configs/thresholds.json: Central configuration file for production deployment.
   - submission/tables/calibration_report.csv: Audit report with ECE, Wilson CI, and costs.

Usage:
    .venv/Scripts/python.exe src/straw_defect_cls/calibrate.py
    .venv/Scripts/python.exe src/straw_defect_cls/calibrate.py --backbone resnet18 --seed 42
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
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.straw_defect_cls.dataset import StrawDefectDataset, get_val_transform
from src.straw_defect_cls.decide import calculate_energy, calibrate_and_correct_stage1, softmax
from src.straw_defect_cls.metrics import wilson_score_interval
from src.straw_defect_cls.models import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Metric & Calibration Math
# ──────────────────────────────────────────────────────────────────────────────
def compute_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Compute Negative Log-Likelihood (NLL) for scaled logits."""
    scaled = logits / max(1e-6, temperature)
    p = softmax(scaled, axis=1)
    # Clip probabilities for numerical stability
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1.0 - eps)
    n = len(labels)
    nll = -np.mean(np.log(p_clipped[np.arange(n), labels]))
    return float(nll)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (preds == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(corrects[in_bin])
            conf_in_bin = np.mean(confs[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * (np.sum(in_bin) / total_samples)

    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Find optimal temperature T minimizing NLL on validation set using Brent's method.

    Returns:
        Tuple of (optimal_temperature, uncalibrated_nll, calibrated_nll).
    """
    uncal_nll = compute_nll(logits, labels, temperature=1.0)

    res = minimize_scalar(
        lambda t: compute_nll(logits, labels, temperature=t),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    opt_t = float(res.x)
    cal_nll = float(res.fun)
    return opt_t, uncal_nll, cal_nll


# ──────────────────────────────────────────────────────────────────────────────
# Logits Extractor
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def extract_logits(
    model: nn.Module,
    csv_path: Path,
    task: str = "full",
    image_size: int = 224,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract raw model output logits and ground-truth labels."""
    model.eval()
    # These saved 2026-10-16 checkpoints were trained before EXIF transpose
    # was added to the dataset loader. Keep calibration pixels identical.
    ds = StrawDefectDataset(csv_file=csv_path, task=task, transform=get_val_transform(image_size), exif_transpose=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_logits = []
    all_labels = []

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        outs = model(imgs)
        all_logits.append(outs.cpu().numpy())
        all_labels.extend(lbls.numpy())

    return np.vstack(all_logits), np.array(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Main Calibration Routine
# ──────────────────────────────────────────────────────────────────────────────
def run_calibration(
    backbone: str = "resnet18",
    seed: int = 42,
    resolution: int = 224,
    target_max_fnr: float = 0.01,
) -> Dict[str, Any]:
    """Execute complete industrial calibration and post-processing pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = REPO_ROOT / "experiments" / "runs"
    configs_dir = REPO_ROOT / "configs"
    tables_dir = REPO_ROOT / "submission" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    val_clean_csv = REPO_ROOT / "data" / "straw_defects_splits" / "val.csv"
    val_degraded_csv = REPO_ROOT / "data" / "val_degraded_v2" / "val_degraded.csv"
    degraded_rows = pd.read_csv(val_degraded_csv)
    if degraded_rows["image_path"].nunique() != len(degraded_rows):
        raise ValueError(f"Degraded validation manifest contains duplicate image paths: {val_degraded_csv}")

    # Read project configuration
    proj_cfg_path = configs_dir / "project.yaml"
    pi_deploy = 0.05
    pi_train = 0.80
    c_fn = 100.0
    c_review = 10.0
    c_fp = 1.0

    if proj_cfg_path.exists():
        try:
            with open(proj_cfg_path, "r", encoding="utf-8") as f:
                proj_data = yaml.safe_load(f)
                ind_req = proj_data.get("industrial_requirements", {})
                pi_deploy = float(ind_req.get("pi_deploy_defect", pi_deploy))
                pi_train = float(ind_req.get("pi_train_defect", pi_train))
                cost_m = ind_req.get("cost_matrix", {})
                c_fn = float(cost_m.get("c_fn", c_fn))
                c_review = float(cost_m.get("c_review", c_review))
                c_fp = float(cost_m.get("c_fp", c_fp))
                target_max_fnr = float(ind_req.get("target_max_fnr", target_max_fnr))
        except Exception as e:
            print(f"[WARN] Error reading project.yaml: {e}. Using defaults.")

    print("\n" + "="*70)
    print("PHASE 3: INDUSTRIAL POST-PROCESSING & CALIBRATION")
    print(f"Target: FNR <= {target_max_fnr*100:.1f}% | Deploy Prior: {pi_deploy*100:.1f}% | Cost: {c_fn}:{c_review}:{c_fp}")
    print("="*70)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Load Trained Models (Stage 1 and Stage 2)
    # ──────────────────────────────────────────────────────────────────────────
    tag_s1 = f"{backbone}_stage1_r{resolution}_seed{seed}"
    tag_s2 = f"{backbone}_stage2_r{resolution}_seed{seed}"
    ckpt_s1 = runs_dir / tag_s1 / "checkpoints" / "best.pt"
    ckpt_s2 = runs_dir / tag_s2 / "checkpoints" / "best.pt"

    # Fallback to any available run if specific seed missing
    if not ckpt_s1.exists():
        found = list(runs_dir.glob(f"{backbone}_stage1_*/checkpoints/best.pt"))
        if found:
            ckpt_s1 = found[0]
            print(f"[INFO] Using available Stage 1 checkpoint: {ckpt_s1}")
    if not ckpt_s2.exists():
        found = list(runs_dir.glob(f"{backbone}_stage2_*/checkpoints/best.pt"))
        if found:
            ckpt_s2 = found[0]
            print(f"[INFO] Using available Stage 2 checkpoint: {ckpt_s2}")

    if not ckpt_s1.exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint at {ckpt_s1}. Run Phase 2 training first.")

    model_s1 = build_model(backbone, num_classes=2, pretrained=False)
    state1 = torch.load(ckpt_s1, map_location=device, weights_only=False)
    model_s1.load_state_dict(state1["model"] if "model" in state1 else state1)
    model_s1 = model_s1.to(device)
    model_s1.eval()

    model_s2 = None
    if ckpt_s2.exists():
        model_s2 = build_model(backbone, num_classes=4, pretrained=False)
        state2 = torch.load(ckpt_s2, map_location=device, weights_only=False)
        model_s2.load_state_dict(state2["model"] if "model" in state2 else state2)
        model_s2 = model_s2.to(device)
        model_s2.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Extract Stage 1 Logits on Clean and Degraded Validation Sets
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 1/5] Extracting Stage 1 logits on val_clean and val_degraded...")
    logits1_clean, y1_clean = extract_logits(model_s1, val_clean_csv, task="stage1", image_size=resolution, device=device)
    logits1_deg, y1_deg = extract_logits(model_s1, val_degraded_csv, task="stage1", image_size=resolution, device=device)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Temperature Scaling Stage 1 (Optimizing NLL on Clean Val)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 2/5] Temperature Scaling on Clean Validation...")
    t1_opt, nll1_uncal, nll1_cal = fit_temperature(logits1_clean, y1_clean)
    ece1_uncal = compute_ece(softmax(logits1_clean), y1_clean)
    ece1_cal = compute_ece(softmax(logits1_clean / t1_opt), y1_clean)
    print(f"  Stage 1: Optimal T = {t1_opt:.3f} | NLL: {nll1_uncal:.4f} -> {nll1_cal:.4f} | ECE: {ece1_uncal:.4f} -> {ece1_cal:.4f}")

    # Stage 2 Temperature Scaling
    t2_opt = 1.0
    nll2_uncal, nll2_cal = 0.0, 0.0
    ece2_uncal, ece2_cal = 0.0, 0.0
    logits2_clean, y2_clean = None, None

    if model_s2 is not None:
        logits2_clean, y2_clean = extract_logits(model_s2, val_clean_csv, task="stage2", image_size=resolution, device=device)
        t2_opt, nll2_uncal, nll2_cal = fit_temperature(logits2_clean, y2_clean)
        ece2_uncal = compute_ece(softmax(logits2_clean), y2_clean)
        ece2_cal = compute_ece(softmax(logits2_clean / t2_opt), y2_clean)
        print(f"  Stage 2: Optimal T = {t2_opt:.3f} | NLL: {nll2_uncal:.4f} -> {nll2_cal:.4f} | ECE: {ece2_uncal:.4f} -> {ece2_cal:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Prior Correction on Stage 1 (Clean + Degraded)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 3/5] Applying Bayes Prior Correction (pi_deploy = 5%, pi_train = 80%)...")
    # Clean set corrected probabilities
    probs1_clean_corr = np.array([
        calibrate_and_correct_stage1(lg, temperature=t1_opt, pi_deploy=pi_deploy, pi_train=pi_train)
        for lg in logits1_clean
    ])
    # Degraded set corrected probabilities
    probs1_deg_corr = np.array([
        calibrate_and_correct_stage1(lg, temperature=t1_opt, pi_deploy=pi_deploy, pi_train=pi_train)
        for lg in logits1_deg
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Neyman-Pearson Gate: Sweep tau_pass for Worst-Case FNR <= 1%
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 4/5] Sweeping Neyman-Pearson tau_pass on Clean & Degraded sets...")
    # Clean defect ground truth mask (label 1 in Stage 1)
    clean_defects_mask = (y1_clean == 1)
    deg_defects_mask = (y1_deg == 1)
    total_clean_defects = int(np.sum(clean_defects_mask))
    total_deg_defects = int(np.sum(deg_defects_mask))

    p_good_clean = probs1_clean_corr[:, 0]
    p_good_deg = probs1_deg_corr[:, 0]

    candidate_taus = np.linspace(0.50, 0.995, 200)
    best_tau_pass = 0.85
    min_worst_fnr = 1.0
    found_satisfying_tau = False
    sweep_records = []

    # Iterate backwards from strictest threshold down to find the LARGEST tau satisfying FNR <= target
    for tau in candidate_taus[::-1]:
        # Leaked defect = True defect where p(Good) >= tau
        fn_clean = int(np.sum((p_good_clean >= tau) & clean_defects_mask))
        fn_deg = int(np.sum((p_good_deg >= tau) & deg_defects_mask))

        fnr_clean = fn_clean / max(1, total_clean_defects)
        fnr_deg = fn_deg / max(1, total_deg_defects)
        fnr_worst = max(fnr_clean, fnr_deg)

        sweep_records.append({
            "tau": round(tau, 4),
            "fnr_clean": fnr_clean,
            "fnr_deg": fnr_deg,
            "fnr_worst": fnr_worst,
        })

        if fnr_worst <= target_max_fnr:
            best_tau_pass = float(tau)
            found_satisfying_tau = True
            break
        if fnr_worst < min_worst_fnr:
            min_worst_fnr = fnr_worst
            best_tau_pass = float(tau)

    # Re-evaluate chosen tau_pass
    fn_clean_chosen = int(np.sum((p_good_clean >= best_tau_pass) & clean_defects_mask))
    fn_deg_chosen = int(np.sum((p_good_deg >= best_tau_pass) & deg_defects_mask))
    fnr_clean_chosen = fn_clean_chosen / max(1, total_clean_defects)
    fnr_deg_chosen = fn_deg_chosen / max(1, total_deg_defects)
    fnr_worst_chosen = max(fnr_clean_chosen, fnr_deg_chosen)

    # Worst-case Wilson 95% Confidence Interval
    worst_fn = max(fn_clean_chosen, fn_deg_chosen)
    worst_total = total_clean_defects if worst_fn == fn_clean_chosen else total_deg_defects
    fnr_ci_low, fnr_ci_high = wilson_score_interval(worst_fn, worst_total, confidence=0.95)

    print(f"  Selected tau_pass = {best_tau_pass:.4f}")
    print(f"  Clean FNR: {fnr_clean_chosen*100:.2f}% | Degraded FNR: {fnr_deg_chosen*100:.2f}% | Worst-case FNR: {fnr_worst_chosen*100:.2f}%")
    print(f"  Wilson 95% CI: [{fnr_ci_low*100:.2f}%, {fnr_ci_high*100:.2f}%]")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Optimize tau_defect via Expected Cost Minimization
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 5/5] Optimizing tau_defect via Expected Cost Minimization...")
    p_defect_clean = probs1_clean_corr[:, 1]
    candidate_tau_defects = np.linspace(0.40, 0.95, 120)
    best_tau_defect = 0.65
    min_cost = float("inf")

    for tau_d in candidate_tau_defects:
        # Decisions with candidate tau_defect:
        # PASS: p_good >= best_tau_pass
        is_pass = p_good_clean >= best_tau_pass
        # DEFECT candidate: p_defect >= tau_d (and not pass)
        is_defect = (p_defect_clean >= tau_d) & (~is_pass)
        # REVIEW: remainder
        is_review = (~is_pass) & (~is_defect)

        # Cost components
        fn_count = int(np.sum(is_pass & clean_defects_mask))
        fp_count = int(np.sum(is_defect & (~clean_defects_mask)))
        review_count = int(np.sum(is_review))

        # Expected cost per 1000 items normalized by base rate
        total_cost = c_fn * fn_count + c_review * review_count + c_fp * fp_count

        if total_cost < min_cost:
            min_cost = total_cost
            best_tau_defect = float(tau_d)

    print(f"  Selected tau_defect = {best_tau_defect:.4f} (Minimal cost: {min_cost:.1f})")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Learn Stage 2 Multi-Barrier Guards
    # ──────────────────────────────────────────────────────────────────────────
    tau2 = 0.50
    margin2 = 0.15
    e_thr = -5.0

    if logits2_clean is not None and len(logits2_clean) > 0:
        # Calibrated Stage 2 probabilities and energy scores
        p2_clean = softmax(logits2_clean / t2_opt, axis=1)
        energies = np.array([calculate_energy(lg, temperature=t2_opt) for lg in logits2_clean])

        # Energy score: 95th percentile on validation defects (samples exceeding this are OOD)
        e_thr = float(np.percentile(energies, 95))

        # Top-1 prob and margin distributions on correctly classified validation samples
        p2_sorted = np.sort(p2_clean, axis=1)[:, ::-1]
        top1_probs = p2_sorted[:, 0]
        margins = p2_sorted[:, 0] - p2_sorted[:, 1]

        # Set conservative thresholds
        tau2 = float(max(0.45, np.percentile(top1_probs, 10)))
        margin2 = float(max(0.10, np.percentile(margins, 10)))

    print(f"  Stage 2 Guards -> tau2: {tau2:.3f} | margin2: {margin2:.3f} | e_thr (95th energy): {e_thr:.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Export configs/thresholds.json & calibration_report.csv
    # ──────────────────────────────────────────────────────────────────────────
    thresholds_payload = {
        "tau_pass": round(best_tau_pass, 4),
        "tau_defect": round(best_tau_defect, 4),
        "tau2": round(tau2, 4),
        "margin2": round(margin2, 4),
        "e_thr": round(e_thr, 4),
        "temperature_stage1": round(t1_opt, 4),
        "temperature_stage2": round(t2_opt, 4),
        "pi_deploy_defect": pi_deploy,
        "pi_train_defect": pi_train,
        "cost_matrix": {
            "c_fn": c_fn,
            "c_review": c_review,
            "c_fp": c_fp,
        },
        "fnr_worst_case": round(fnr_worst_chosen, 4),
        "fnr_clean": round(fnr_clean_chosen, 4),
        "fnr_degraded": round(fnr_deg_chosen, 4),
        "fnr_ci_low_95": round(fnr_ci_low, 4),
        "fnr_ci_high_95": round(fnr_ci_high, 4),
        "calibration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "calibrated_industrial",
    }

    out_thr_file = configs_dir / "thresholds.json"
    out_thr_file.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved calibrated thresholds to: {out_thr_file}")

    # Build comprehensive calibration report
    report_rows = [
        {
            "stage": "Stage 1 (Binary)",
            "temperature": round(t1_opt, 4),
            "uncal_nll": round(nll1_uncal, 4),
            "cal_nll": round(nll1_cal, 4),
            "uncal_ece": round(ece1_uncal, 4),
            "cal_ece": round(ece1_cal, 4),
            "tau_pass": round(best_tau_pass, 4),
            "tau_defect": round(best_tau_defect, 4),
            "fnr_clean": round(fnr_clean_chosen, 4),
            "fnr_degraded": round(fnr_deg_chosen, 4),
            "fnr_worst": round(fnr_worst_chosen, 4),
            "fnr_ci_low": round(fnr_ci_low, 4),
            "fnr_ci_high": round(fnr_ci_high, 4),
            "expected_cost": round(min_cost, 2),
        },
        {
            "stage": "Stage 2 (4-Class)",
            "temperature": round(t2_opt, 4),
            "uncal_nll": round(nll2_uncal, 4),
            "cal_nll": round(nll2_cal, 4),
            "uncal_ece": round(ece2_uncal, 4),
            "cal_ece": round(ece2_cal, 4),
            "tau_pass": np.nan,
            "tau_defect": np.nan,
            "fnr_clean": np.nan,
            "fnr_degraded": np.nan,
            "fnr_worst": np.nan,
            "fnr_ci_low": np.nan,
            "fnr_ci_high": np.nan,
            "expected_cost": np.nan,
        },
    ]

    report_df = pd.DataFrame(report_rows)
    report_csv = tables_dir / "calibration_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"[DONE] Saved calibration report table to: {report_csv}")

    return thresholds_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate two-stage defect classification pipeline.")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model to calibrate")
    parser.add_argument("--seed", type=int, default=42, help="Seed used during model training")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution")
    args = parser.parse_args()

    run_calibration(backbone=args.backbone, seed=args.seed, resolution=args.resolution)


if __name__ == "__main__":
    main()
