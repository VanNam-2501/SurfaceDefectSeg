"""Shared decision/metric implementation for every thesis comparison."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z = NormalDist().inv_cdf(0.975)
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    margin = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float(max(0, center - margin)), float(min(1, center + margin))]


def probability_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = softmax(logits / temperature, axis=1)
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        chosen = (confidence > low) & (confidence <= high)
        if chosen.any():
            ece += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
    nll = np.mean(logsumexp(logits / temperature, axis=1) - (logits / temperature)[np.arange(len(labels)), labels])
    brier = np.mean(np.sum((probabilities - np.eye(logits.shape[1])[labels]) ** 2, axis=1))
    return {"nll": float(nll), "ece_15bins": float(ece), "brier": float(brier)}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    if len(labels) == 0 or not np.isfinite(logits).all():
        raise ValueError("Calibration requires finite, nonempty logits")
    result = minimize_scalar(lambda log_t: probability_metrics(logits, labels, np.exp(log_t))["nll"],
                             bounds=(-3, 3), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    temperature = float(np.exp(result.x))
    before = probability_metrics(logits, labels)
    after = probability_metrics(logits, labels, temperature)
    # T=1 is always an admissible fallback.
    if after["nll"] > before["nll"]:
        temperature, after = 1.0, before
    return {"temperature": temperature, "before": before, "after": after}


def discrimination_metrics(logits, labels):
    probabilities = softmax(logits, axis=1)
    y = np.asarray(labels, dtype=int)
    one_hot = np.eye(logits.shape[1])[y]
    if len(np.unique(y)) != logits.shape[1]:
        return {"auroc": None, "average_precision": None, "reason": "missing classes"}
    if logits.shape[1] == 2:
        return {"auroc": float(roc_auc_score(y, probabilities[:, 1])),
                "average_precision": float(average_precision_score(y, probabilities[:, 1]))}
    return {"auroc": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
            "average_precision": float(average_precision_score(one_hot, probabilities, average="macro"))}


def stage_probabilities(logits1, logits2, policy):
    z1 = np.asarray(logits1) / policy.get("temperature_stage1", 1.0)
    if policy.get("prior_correction", False):
        train, deploy = policy["pi_train"], policy["pi_deploy"]
        if not (0 < train < 1 and 0 < deploy < 1):
            raise ValueError("Prior rates must be strictly between zero and one")
        z1 = z1 + np.log([(1 - deploy) / (1 - train), deploy / train])
    p1 = softmax(z1, axis=1)
    p2 = softmax(np.asarray(logits2) / policy.get("temperature_stage2", 1.0), axis=1)
    return p1, p2


def predict(logits: dict, policy: dict) -> np.ndarray:
    """Return 0=PASS, 1..4=diagnosed DEFECT, -1=REVIEW."""
    kind = policy["kind"]
    if kind in {"full5", "selective_full5"}:
        p = softmax(logits["full5"] / policy.get("temperature_full5", 1.0), axis=1)
        predictions = p.argmax(1)
        if kind == "selective_full5":
            predictions[p.max(1) < policy["confidence"]] = -1
        return predictions
    if kind == "hard_cascade":
        return np.where(logits["1"].argmax(1) == 0, 0, logits["2"].argmax(1) + 1)
    p1, p2 = stage_probabilities(logits["1"], logits["2"], policy)
    predictions = np.full(len(p1), -1, dtype=int)
    passed = p1[:, 0] >= policy["tau_pass"]
    diagnosed = (p1[:, 1] >= policy["tau_defect"]) & ~passed
    if policy.get("guards", True):
        ordered = np.sort(p2, axis=1)
        diagnosed &= (ordered[:, -1] >= policy["confidence"]) & ((ordered[:, -1] - ordered[:, -2]) >= policy["margin"])
    predictions[passed] = 0
    predictions[diagnosed] = p2[diagnosed].argmax(1) + 1
    return predictions


def decision_metrics(labels, predictions, pi_deploy=0.05, costs=None) -> dict:
    y, pred = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if len(y) == 0 or y.shape != pred.shape or not np.isin(y, range(5)).all() or not np.isin(pred, range(-1, 5)).all():
        raise ValueError("Invalid labels or predictions")
    costs = costs or {"fn": 100.0, "review": 2.0, "fp": 5.0, "misdiagnosis": 5.0}
    defect, good, accepted = y > 0, y == 0, pred >= 0
    n_defect, n_good = int(defect.sum()), int(good.sum())
    fn = int(((pred == 0) & defect).sum())
    fp = int(((pred > 0) & good).sum())
    fnr, fpr = fn / max(n_defect, 1), fp / max(n_good, 1)
    review_good = float((pred[good] == -1).mean()) if n_good else 0.0
    review_defect = float((pred[defect] == -1).mean()) if n_defect else 0.0
    wrong_type = float(((pred[defect] > 0) & (pred[defect] != y[defect])).mean()) if n_defect else 0.0
    review_deploy = (1 - pi_deploy) * review_good + pi_deploy * review_defect
    cost = 1000 * (costs["fn"] * pi_deploy * fnr + costs["fp"] * (1 - pi_deploy) * fpr
                   + costs["review"] * review_deploy + costs["misdiagnosis"] * pi_deploy * wrong_type)
    recalls = {str(c): float((pred[y == c] == c).mean()) if (y == c).any() else None for c in range(5)}
    return {
        "n": len(y), "n_defect": n_defect, "n_good": n_good, "fn": fn, "fp": fp,
        "fnr": fnr if n_defect else None, "fnr_95ci": wilson(fn, n_defect),
        "fpr": fpr if n_good else None, "fpr_95ci": wilson(fp, n_good),
        "coverage": float(accepted.mean()), "review_rate": float((~accepted).mean()),
        "review_rate_good": review_good, "review_rate_defect": review_defect,
        "deployment_review_rate": review_deploy, "cost_per_1000": float(cost),
        "cost_units": "assumed relative cost, not monetary loss",
        "selective_risk": float((pred[accepted] != y[accepted]).mean()) if accepted.any() else None,
        "accuracy_all": float((pred == y).mean()),
        "macro_f1_all": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
        "macro_recall_defect": float(np.mean([recalls[str(c)] for c in range(1, 5) if recalls[str(c)] is not None])),
        "per_class_recall": recalls,
        "confusion_5x6": confusion_matrix(y, pred, labels=[0, 1, 2, 3, 4, -1])[:5].tolist(),
    }


def choose_policy(logits: dict, labels, calibration: dict, config: dict, variant: str = "triage") -> tuple[dict, list[dict]]:
    """Finite preregistered validation search, minimize cost subject to FNR/review.

    Infeasible constraints stay explicitly infeasible; no safety guarantee is made.
    """
    base = {"kind": "triage", "temperature_stage1": calibration["1"]["temperature"],
            "temperature_stage2": calibration["2"]["temperature"],
            "pi_train": config["pi_train"], "pi_deploy": config["pi_deploy"],
            "prior_correction": variant != "no_prior", "guards": variant != "no_guards"}
    if variant == "no_temperature":
        base.update(temperature_stage1=1.0, temperature_stage2=1.0)
    candidates = []
    if variant == "selective_full5":
        candidates = [{"kind": variant, "temperature_full5": calibration["full5"]["temperature"], "confidence": float(c)}
                      for c in config["grid"]["full5_confidence"]]
    else:
        for passed in config["grid"]["tau_pass"]:
            for defective in config["grid"]["tau_defect"]:
                # Avoid overlapping gates (PASS has priority but overlap obscures interpretation).
                if passed + defective < 1:
                    continue
                for confidence in config["grid"]["confidence"] if base["guards"] else [0.0]:
                    for margin in config["grid"]["margin"] if base["guards"] else [0.0]:
                        candidates.append(dict(base, tau_pass=passed, tau_defect=defective, confidence=confidence, margin=margin))
    records, ranked = [], []
    for index, candidate in enumerate(candidates):
        pred = predict(logits, candidate)
        m = decision_metrics(labels, pred, config["pi_deploy"], config["costs"])
        fn_violation = max(0, m["fnr"] - config["max_fnr"])
        review_violation = max(0, m["deployment_review_rate"] - config["max_deployment_review"])
        feasible = fn_violation <= 1e-12 and review_violation <= 1e-12
        # Prefer feasibility, then constraint violations (safety first), cost, and empirical review.
        rank = (not feasible, fn_violation, review_violation, m["cost_per_1000"], m["review_rate"], index)
        ranked.append((rank, candidate, m))
        records.append({"candidate": index, **candidate, "feasible": feasible,
                        **{key: m[key] for key in ["fnr", "fpr", "coverage", "review_rate", "deployment_review_rate", "cost_per_1000"]}})
    if not ranked:
        raise ValueError("Policy grid is empty")
    rank, selected, metrics = min(ranked, key=lambda entry: entry[0])
    selected = dict(selected, selection_status="feasible_empirical" if not rank[0] else "infeasible",
                    validation_metrics=metrics, constraint_basis="empirical validation rates, not a statistical safety guarantee")
    return selected, records


def risk_coverage(logits, labels, temperature=1.0) -> list[dict]:
    probs = softmax(logits / temperature, axis=1)
    order = np.argsort(-probs.max(1), kind="stable")
    errors = (probs.argmax(1)[order] != labels[order]).astype(float)
    return [{"coverage": (i + 1) / len(labels), "risk": float(value)}
            for i, value in enumerate(np.cumsum(errors) / np.arange(1, len(labels) + 1))]


def grouped_bootstrap(labels, predictions, groups, n_bootstrap=1000, seed=42, other=None, pi_deploy=0.05, costs=None) -> dict:
    """Paired group bootstrap within true classes; respects exact-duplicate clusters."""
    y, p, g = np.asarray(labels), np.asarray(predictions), np.asarray(groups)
    rng = np.random.default_rng(seed)
    # Candidate clusters may contain several labels. Keep each entire cluster
    # in one composition stratum; never resample it independently per class.
    by_composition = {}
    for key in np.unique(g):
        block = np.flatnonzero(g == key)
        composition = tuple(np.unique(y[block]).tolist())
        by_composition.setdefault(composition, []).append(block)
    strata = list(by_composition.values())
    keys = ["fnr", "macro_f1_all", "review_rate", "cost_per_1000"]
    values = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = np.concatenate([np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for blocks in strata])
        a = decision_metrics(y[idx], p[idx], pi_deploy, costs)
        b = decision_metrics(y[idx], np.asarray(other)[idx], pi_deploy, costs) if other is not None else None
        for key in keys:
            if a[key] is not None and (b is None or b[key] is not None):
                values[key].append(a[key] - (b[key] if b else 0))
    return {"method": "label-composition-stratified paired group bootstrap" if other is not None else "label-composition-stratified group bootstrap",
            "replicates": n_bootstrap, "ci95": {k: np.quantile(v, [0.025, 0.975]).tolist() if v else None for k, v in values.items()}}
