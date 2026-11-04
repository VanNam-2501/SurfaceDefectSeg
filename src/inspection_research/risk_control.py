"""Finite-sample PASS control for a fixed score and independent defective units.

Scores are oriented so that larger means more likely defective. Strict comparison
at the order statistic is deliberate: ties must not admit calibration duplicates.
These functions cannot verify statistical independence or distribution stability.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta


ASSUMPTIONS = [
    "Scoring function and preprocessing fixed independently of this calibration set.",
    "Calibration scores represent independent defective products from the target conditional distribution.",
    "Future defective scores have the same conditional distribution; arbitrary domain shift is not covered.",
    "PASS uses score < threshold, never <=; additional gates may only shrink the PASS set.",
]


def _levels(alpha, delta):
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be strictly between zero and one")


def minimum_defect_units(alpha: float, delta: float, comparisons: int = 1) -> int:
    """Minimum n for a nontrivial first order-statistic gate, using delta/comparisons."""
    _levels(alpha, delta)
    if not isinstance(comparisons, int) or comparisons < 1:
        raise ValueError("comparisons must be a positive integer")
    return math.ceil(math.log(delta / comparisons) / math.log1p(-alpha))


def one_sided_binomial_upper(events: int, trials: int, delta: float = 0.05) -> float:
    if not 0 < delta < 1 or not 0 <= events <= trials or not isinstance(events, int) or not isinstance(trials, int):
        raise ValueError("Invalid binomial counts or confidence level")
    if trials == 0 or events == trials:
        return 1.0
    if events == 0:
        return float(-np.expm1(np.log(delta) / trials))
    return float(beta.ppf(1 - delta, events + 1, trials - events))


def calibrate_pass_gate(defect_scores, alpha=0.05, delta=0.05, unit_ids=None):
    """Choose k by (n,alpha,delta), then t=score_(k); no score-dependent rank search.

    For continuous scores F_D(t) ~ Beta(k,n+1-k). With ties, using F_D(t-)
    and strict PASS is conservative. The bound is high-probability over the
    calibration sample, not a guarantee for each individual product.
    """
    _levels(alpha, delta)
    values = np.asarray(defect_scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Defect probabilities must form a finite vector in [0,1]")
    if unit_ids is not None:
        ids = np.asarray(unit_ids)
        if ids.shape != values.shape or len(np.unique(ids)) != len(ids):
            raise ValueError("Use one independent defect observation per unit; duplicate unit IDs found")
    n = len(values)
    result = {"method": "NP strict order-statistic PASS gate", "alpha": alpha, "delta": delta,
              "n_defect_units": n, "minimum_units": minimum_defect_units(alpha, delta),
              "assumptions": ASSUMPTIONS, "unit_ids_checked_unique": unit_ids is not None}
    if n < result["minimum_units"]:
        return dict(result, status="insufficient_data", threshold=None, rank=None,
                    note="Do not claim risk control from empirical FNR=0. A no-PASS fallback is a separate operational choice.")
    ranks = np.arange(1, n + 1)
    upper = beta.ppf(1 - delta, ranks, n + 1 - ranks)
    admissible = np.flatnonzero(upper <= alpha)
    if len(admissible) == 0:
        return dict(result, status="insufficient_data", threshold=None, rank=None)
    index = int(admissible[-1])
    return dict(result, status="calibrated_under_assumptions", threshold=float(np.sort(values)[index]),
                rank=index + 1, population_fnr_upper=float(upper[index]), comparison="strict_less_than")


def calibrate_known_domains(defect_scores, domains, alpha=0.05, delta=0.05, unit_ids=None):
    """Union bound over observed domains; take min threshold for one global PASS gate.

    Covers mixtures of those same conditional domains, not a new unseen domain.
    If one domain lacks evidence, the result is insufficient; no domain is dropped.
    """
    scores, domains = np.asarray(defect_scores), np.asarray(domains)
    if scores.ndim != 1 or domains.shape != scores.shape or len(scores) == 0:
        raise ValueError("Need aligned nonempty scores and domain IDs")
    keys = np.unique(domains)
    ids = np.asarray(unit_ids) if unit_ids is not None else None
    if ids is not None and (ids.shape != scores.shape or len(np.unique(ids)) != len(ids)):
        raise ValueError("Independent unit IDs must be globally unique")
    calibrated = {}
    for key in keys:
        selected = domains == key
        calibrated[str(key)] = calibrate_pass_gate(scores[selected], alpha, delta / len(keys),
                                                  ids[selected] if ids is not None else None)
    feasible = all(item["status"] == "calibrated_under_assumptions" for item in calibrated.values())
    return {"status": "calibrated_known_domains_under_assumptions" if feasible else "insufficient_data",
            "alpha": alpha, "delta": delta, "domain_results": calibrated,
            "threshold": min(item["threshold"] for item in calibrated.values()) if feasible else None,
            "scope": "Observed conditional domains and their mixtures; no guarantee for an unseen domain."}
