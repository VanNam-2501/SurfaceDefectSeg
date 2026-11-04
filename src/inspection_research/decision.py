"""One expected-cost decision rule shared by all products and classifiers."""
from __future__ import annotations

import numpy as np


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or p.shape[0] == 0 or not np.isfinite(p).all():
        raise ValueError("Expected a finite, nonempty Nx(K+1) probability matrix")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    return p


def cost_matrix(num_defects: int, false_pass: float, false_reject: float,
                review: float, wrong_type: float = 0):
    """Rows: REVIEW, defect 1..K, PASS. Columns: good, defect 1..K.

    A fixed tie convention favors REVIEW, then a defect, then PASS. A review
    row assumes a specified human-review cost, not automatic human perfection.
    Supply a custom matrix if review errors vary by true class.
    """
    values = np.array([false_pass, false_reject, review, wrong_type], dtype=float)
    if num_defects < 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Require K>=1 and finite nonnegative costs")
    matrix = np.zeros((num_defects + 2, num_defects + 1), dtype=float)
    matrix[0] = review
    for defect in range(1, num_defects + 1):
        matrix[defect, 0] = false_reject
        matrix[defect, 1:] = wrong_type
        matrix[defect, defect] = 0
    matrix[-1, 1:] = false_pass
    actions = np.array([-1, *range(1, num_defects + 1), 0])
    return matrix, actions


def bayes_actions(probabilities, matrix, actions, review_penalty=0.0, pass_threshold=None):
    """Minimize posterior expected cost; optionally apply a strict calibrated PASS gate.

    A denied PASS goes to REVIEW. It never becomes an unsupported defect diagnosis.
    The returned action can lose empirical review-budget feasibility after gating.
    """
    p = validate_probabilities(probabilities)
    matrix, actions = np.asarray(matrix, dtype=float), np.asarray(actions)
    if matrix.shape != (len(actions), p.shape[1]) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Invalid action cost matrix")
    if set(actions.tolist()) != {-1, *range(p.shape[1])} or len(actions) != p.shape[1] + 1:
        raise ValueError("Actions must contain REVIEW=-1 and all labels 0..K exactly once")
    if review_penalty < 0 or not np.isfinite(review_penalty):
        raise ValueError("Review penalty must be finite and nonnegative")
    expected = p @ matrix.T
    expected[:, actions == -1] += review_penalty
    chosen = actions[np.argmin(expected, axis=1)].copy()
    if pass_threshold is not None:
        if not np.isfinite(pass_threshold) or not 0 <= pass_threshold <= 1:
            raise ValueError("PASS threshold must be a finite score in [0,1]")
        score = 1 - p[:, 0]
        chosen[(chosen == 0) & ~(score < pass_threshold)] = -1
    return chosen, expected


def select_review_penalty(probabilities, matrix, actions, budget, grid, weights=None):
    """Development-set tuning only. This does not certify a population budget."""
    p = validate_probabilities(probabilities)
    if not 0 <= budget <= 1:
        raise ValueError("Review budget must be in [0,1]")
    weights = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(p),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid development weights")
    records = []
    for penalty in sorted(set(grid)):
        chosen, _ = bayes_actions(p, matrix, actions, penalty)
        rate = float(np.average(chosen == -1, weights=weights))
        records.append({"review_penalty": float(penalty), "development_review_rate": rate})
    feasible = [record for record in records if record["development_review_rate"] <= budget]
    return {"status": "feasible_on_development" if feasible else "infeasible_on_development",
            "selected": feasible[0] if feasible else None, "search": records,
            "note": "Budget can change on calibration/test or after applying a PASS guard."}
