"""Describe completed development pilots only. Never reads Test predictions."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.straw_global_local.data import load_protocol, read_json, write_json, file_hash
from src.straw_global_local.runner import classification_metrics, jobs, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifacts/straw_global_local/data_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/verification/global_local_pilot")
    args = parser.parse_args()
    protocol, frames = load_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, errors, histories, provenance = [], [], {}, []
    for experiment in args.experiment:
        spec = read_json(experiment / "experiment.json")
        if spec["config"]["purpose"] != "development" or spec["protocol"] != protocol["fingerprint"]:
            raise ValueError("Only development pilots with the same protocol are supported")
        for job in jobs(spec["config"]):
            marker = verify_run(experiment, job, spec)
            folder = experiment / "runs" / job["id"]
            with np.load(folder / "val_predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["sample_ids"], frames["val"].sample_id.astype(str).to_numpy()) or not np.array_equal(saved["labels"], frames["val"].label.to_numpy()):
                    raise ValueError("Validation alignment mismatch")
                metrics = classification_metrics(saved["labels"], saved["logits"])
                predictions = saved["logits"].argmax(1)
            if metrics != marker["validation"]:
                raise ValueError("Validation metrics do not match checkpoint")
            history = read_json(folder / "history.json")
            rows.append({"variant": job["variant"], "seed": job["seed"], "epochs": marker["epochs_completed"],
                "best_epoch": marker["best_epoch"], "val_macro_f1": metrics["macro_f1"],
                "val_accuracy": metrics["accuracy"], "val_fn": metrics["fn"], "val_fp": metrics["fp"],
                "val_wrong_type": metrics["wrong_defect_type"], "training_loop_seconds": history[-1]["seconds"]})
            wrong = frames["val"].copy()
            wrong["predicted_label"] = predictions
            wrong["variant"], wrong["seed"] = job["variant"], job["seed"]
            errors.append(wrong[wrong.label != predictions])
            histories[job["id"]] = history
            provenance.append({"experiment": str(experiment), "fingerprint": spec["fingerprint"],
                               "job": job, "checkpoint": marker["best_checkpoint"]})
    table = pd.DataFrame(rows).sort_values(["variant", "seed"])
    table.to_csv(args.output / "validation_pilot_summary.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(args.output / "validation_error_cases.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["val_macro_f1"] for r in history], marker="o", label=name)
        axes[1].plot(epochs, [r["val_ce"] for r in history], marker="o", label=name)
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("Validation cross-entropy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.25)
        axis.legend()
    fig.suptitle("DEVELOPMENT PILOT ONLY - one seed, no main Test evaluation")
    fig.tight_layout()
    fig.savefig(args.output / "validation_learning_curves.png", dpi=160)
    plt.close(fig)
    write_json(args.output / "pilot_summary.json", {"status": "DEVELOPMENT ONLY, NOT THESIS RESULTS",
        "protocol": protocol["fingerprint"], "counts": protocol["counts"], "test_evaluated": False,
        "rows": table.to_dict("records"), "provenance": provenance, "summary_script": file_hash(__file__),
        "limits": ["One seed and three epochs do not establish superiority.",
                   "Validation was used for development and checkpoint selection.",
                   "Training-loop time is not end-to-end inference latency or total project cost."]})
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
