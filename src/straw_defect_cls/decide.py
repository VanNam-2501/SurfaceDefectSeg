"""Industrial Decision Engine for Two-Stage Straw Defect Classification.

Pure NumPy implementation:
- No torch runtime dependency during decision evaluation.
- Full temperature scaling & deployment prior adjustment.
- Neyman-Pearson PASS gate (worst-case FNR control).
- Multi-barrier Stage 2 defense:
  1. Top-1 probability >= tau2
  2. Margin (top1 - top2) >= margin2
  3. Energy score < e_thr (Out-of-Distribution rejection)
- Output decision: PASS, REVIEW, or DEFECT with explicit reason string.

Reasons:
  - 'pass': Stage 1 confidence in Good >= tau_pass
  - 'borderline_s1': Stage 1 confidence is ambiguous (p_defect < tau_defect and p_good < tau_pass)
  - 'uncertain_s2': Stage 2 top-1 probability < tau2
  - 'ood_s2': Stage 2 energy score >= e_thr (anomalous / OOD sample)
  - 'low_margin_s2': Stage 2 margin (top1 - top2) < margin2 (high confusion between defect types)
  - 'defect': Passed Stage 1 defect threshold and all 3 Stage 2 guards
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def calculate_energy(logits: np.ndarray, temperature: float = 1.0) -> float:
    """Calculate Free Energy Score for out-of-distribution detection (Liu et al., NeurIPS 2020).

    E(x) = -T * log(sum(exp(logits / T)))
    Lower (more negative) energy indicates in-distribution samples.
    Higher (less negative or positive) energy indicates OOD or anomaly.

    Args:
        logits: Unnormalized logits vector of shape (K,).
        temperature: Temperature scaling factor.

    Returns:
        Scalar energy score.
    """
    scaled = np.asarray(logits, dtype=np.float64) / max(1e-6, temperature)
    m = np.max(scaled)
    # Log-sum-exp trick to prevent overflow
    lse = m + np.log(np.sum(np.exp(scaled - m)))
    return float(-temperature * lse)


def calibrate_and_correct_stage1(
    logits1: np.ndarray,
    temperature: float = 1.0,
    pi_deploy: float = 0.05,
    pi_train: float = 0.80,
) -> np.ndarray:
    """Apply temperature scaling and deployment prior correction to binary Stage 1 logits.

    Formula:
        z_good_cal = z_good / T + log((1 - pi_deploy) / (1 - pi_train))
        z_defect_cal = z_defect / T + log(pi_deploy / pi_train)

    Args:
        logits1: Array-like of length 2 [logit_good, logit_defect].
        temperature: Learned temperature factor.
        pi_deploy: Defect base rate in factory deployment (e.g. 0.05 = 5%).
        pi_train: Defect base rate in training distribution (e.g. 0.80 = 80%).

    Returns:
        Calibrated probabilities array [p_good, p_defect].
    """
    z = np.asarray(logits1, dtype=np.float64).flatten()
    assert z.shape[0] == 2, f"Stage 1 logits must have length 2, got shape {z.shape}"

    # Temperature scaling
    z_scaled = z / max(1e-6, temperature)

    # Prior shift log-odds correction (Bayes Rule)
    # Clip probabilities to prevent div-by-zero or log(0)
    p_dep_def = np.clip(pi_deploy, 1e-6, 1.0 - 1e-6)
    p_dep_good = 1.0 - p_dep_def
    p_trn_def = np.clip(pi_train, 1e-6, 1.0 - 1e-6)
    p_trn_good = 1.0 - p_trn_def

    offset_good = np.log(p_dep_good / p_trn_good)
    offset_defect = np.log(p_dep_def / p_trn_def)

    corrected_logits = np.array([
        z_scaled[0] + offset_good,
        z_scaled[1] + offset_defect,
    ])
    return softmax(corrected_logits)


def decide(
    logits1: Union[np.ndarray, list],
    logits2: Optional[Union[np.ndarray, list]],
    cfg: Dict[str, Any],
) -> Tuple[str, Optional[int], str]:
    """Hierarchical industrial decision function.

    Pure NumPy execution without PyTorch dependencies.

    Decision flow:
    1. Calibrate Stage 1 logits with Temperature Scaling and Deployment Prior Correction.
    2. If p1[Good] >= tau_pass:
           -> ("PASS", None, "pass")
    3. If p1[Defect] < tau_defect:
           -> ("REVIEW", None, "borderline_s1")
    4. Otherwise (Defect candidate), check Stage 2 guards:
       a. Energy Score >= e_thr:
           -> ("REVIEW", None, "ood_s2")
       b. Top-1 probability < tau2:
           -> ("REVIEW", None, "uncertain_s2")
       c. Top-1 - Top-2 margin < margin2:
           -> ("REVIEW", None, "low_margin_s2")
    5. If all guards pass:
           -> ("DEFECT", defect_class, "defect")

    Args:
        logits1: Stage 1 unnormalized logits (shape (2,) or (1, 2)).
        logits2: Stage 2 unnormalized logits (shape (4,) or (1, 4)) or None.
        cfg: Configuration dictionary with threshold and calibration parameters:
             - tau_pass (float): PASS probability threshold.
             - tau_defect (float): Minimum Stage 1 defect probability to trigger diagnosis.
             - tau2 (float): Stage 2 minimum top-1 probability guard.
             - margin2 (float): Stage 2 minimum top1 - top2 margin guard.
             - e_thr (float): Maximum allowed Stage 2 energy score before OOD rejection.
             - temperature_stage1 (float): Temperature parameter for Stage 1 (default: 1.0).
             - temperature_stage2 (float): Temperature parameter for Stage 2 (default: 1.0).
             - pi_deploy_defect (float): Factory defect prior rate (default: 0.05).
             - pi_train_defect (float): Training set defect prior rate (default: 0.80).

    Returns:
        Tuple of (decision, defect_class, reason):
        - decision: 'PASS' | 'REVIEW' | 'DEFECT'
        - defect_class: int in [0, 3] (0: Bent, 1: Loose, 2: Missing, 3: Pierced) if DEFECT, else None.
        - reason: 'pass' | 'borderline_s1' | 'uncertain_s2' | 'ood_s2' | 'low_margin_s2' | 'defect'
    """
    # Extract configuration parameters with default fallbacks
    tau_pass = float(cfg.get("tau_pass", 0.80))
    tau_defect = float(cfg.get("tau_defect", 0.65))
    tau2 = float(cfg.get("tau2", 0.50))
    margin2 = float(cfg.get("margin2", 0.15))
    e_thr = float(cfg.get("e_thr", -5.0))
    t1 = float(cfg.get("temperature_stage1", 1.0))
    t2 = float(cfg.get("temperature_stage2", 1.0))
    pi_deploy = float(cfg.get("pi_deploy_defect", 0.05))
    pi_train = float(cfg.get("pi_train_defect", 0.80))

    # Step 1: Calibrate & prior-correct Stage 1
    p1 = calibrate_and_correct_stage1(
        logits1=np.asarray(logits1, dtype=np.float64).flatten(),
        temperature=t1,
        pi_deploy=pi_deploy,
        pi_train=pi_train,
    )
    p_good, p_defect = float(p1[0]), float(p1[1])

    # Step 2: Neyman-Pearson Gate -> PASS
    if p_good >= tau_pass:
        return "PASS", None, "pass"

    # Step 3: Stage 1 borderline screening
    if p_defect < tau_defect:
        return "REVIEW", None, "borderline_s1"

    # Step 4: Stage 2 verification
    if logits2 is None:
        return "REVIEW", None, "uncertain_s2"

    z2 = np.asarray(logits2, dtype=np.float64).flatten()
    assert z2.shape[0] == 4, f"Stage 2 logits must have length 4, got shape {z2.shape}"

    # Temperature scaling on Stage 2
    z2_scaled = z2 / max(1e-6, t2)
    p2 = softmax(z2_scaled)

    # 4a. Energy Score Guard (OOD / Anomaly check)
    energy = calculate_energy(z2, temperature=t2)
    if energy >= e_thr:
        return "REVIEW", None, "ood_s2"

    # 4b. Confidence Guard (Top-1 probability check)
    sorted_indices = np.argsort(p2)[::-1]
    top1_idx = int(sorted_indices[0])
    top1_prob = float(p2[top1_idx])
    top2_prob = float(p2[sorted_indices[1]])

    if top1_prob < tau2:
        return "REVIEW", None, "uncertain_s2"

    # 4c. Margin Guard (Confusion between top 2 defect classes)
    margin = top1_prob - top2_prob
    if margin < margin2:
        return "REVIEW", None, "low_margin_s2"

    # Step 5: All guards passed -> Validated defect diagnosis
    return "DEFECT", top1_idx, "defect"
