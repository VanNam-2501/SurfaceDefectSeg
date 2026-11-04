"""Export thesis tables, seed summaries, error lists and scientific figures."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .thesis_protocol import digest_file, write_json


def generate_report(output: Path) -> None:
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    expected_groups = sorted(set(job["group"] for job in plan["jobs"]))
    expected_conditions = ["clean", "noise_2"] if experiment["smoke_only"] else experiment["config"]["conditions"]
    missing = [f"{group}/{condition}" for group in expected_groups for condition in expected_conditions
               if not (output / "test" / group / condition / "complete.json").exists()]
    rows, errors = [], []
    for path in sorted((output / "test").glob("*/*/metrics.json")):
        directory = path.parent
        marker = directory / "complete.json"
        if not marker.exists():
            continue
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if completed["freeze"] != ledger["fingerprint"]:
            raise ValueError("Mixed freeze ledgers in report")
        for filename, claimed in completed["files"].items():
            if digest_file(directory / filename) != claimed:
                raise ValueError(f"Test result changed: {directory / filename}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for record in metrics:
            family, seed = record["group"].rsplit("_s", 1)
            row = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
            row.update(family=family, seed=int(seed), fnr_ci_low=record["fnr_95ci"][0], fnr_ci_high=record["fnr_95ci"][1])
            row.update({f"recall_{label}": value for label, value in record["per_class_recall"].items()})
            rows.append(row)
        if directory.name == "clean":
            predictions = pd.read_csv(directory / "predictions.csv")
            for variant in ["full5", "selective_full5", "triage"]:
                wrong = predictions[predictions[variant] != predictions.label].copy()
                wrong["variant"], wrong["group"] = variant, directory.parent.name
                wrong["predicted"] = wrong[variant]
                errors.append(wrong[["sample_id", "brand", "label", "predicted", "variant", "group"]])
    if not rows:
        raise ValueError("No completed test results")
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(report / "all_results.csv", index=False)
    metrics_columns = ["fnr", "fpr", "macro_f1_all", "macro_recall_defect", "review_rate", "deployment_review_rate", "coverage", "selective_risk", "cost_per_1000"]
    aggregated = frame.groupby(["family", "condition", "variant"])[metrics_columns].agg(["mean", "std", "count"])
    aggregated.columns = ["_".join(column) for column in aggregated.columns]
    aggregated.to_csv(report / "seed_mean_std.csv")
    if errors:
        pd.concat(errors, ignore_index=True).to_csv(report / "error_analysis.csv", index=False)
    frame.groupby(["group", "variant"])[metrics_columns].agg(["mean", "max"]).to_csv(report / "stress_summary.csv")

    clean = frame[(frame.condition == "clean") & frame.variant.isin(["full5", "selective_full5", "hard_cascade", "triage"])]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, title in zip(axes, ["fnr", "review_rate", "macro_f1_all"], ["Defects incorrectly passed", "Manual review fraction", "Macro F1 (review = unclassified)"]):
        means = clean.groupby(["family", "variant"])[metric].mean().unstack()
        stds = clean.groupby(["family", "variant"])[metric].std().unstack().fillna(0)
        means.plot.bar(ax=ax, yerr=stds, capsize=2, rot=35)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=7)
    figure.tight_layout()
    figure.savefig(report / "comparison.png", dpi=180)
    plt.close(figure)

    triage = frame[frame.variant == "triage"]
    heat = triage.pivot_table(index="family", columns="condition", values="fnr", aggfunc="mean")
    figure, ax = plt.subplots(figsize=(max(8, len(heat.columns) * 0.65), max(3, len(heat.index) * 0.6)))
    mesh = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.05, float(heat.max().max())))
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=50, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Frozen triage policy: FNR under synthetic stress")
    figure.colorbar(mesh, ax=ax)
    figure.tight_layout()
    figure.savefig(report / "robustness.png", dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 5))
    for path in sorted((output / "test").glob("*/clean/risk_coverage.csv")):
        curve = pd.read_csv(path)
        ax.plot(curve.coverage, curve.risk, label=path.parent.parent.name, alpha=0.75)
    ax.set(xlabel="Coverage", ylabel="Selective classification error", title="Full5 risk-coverage (post-hoc descriptive curve)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(report / "risk_coverage.png", dpi=180)
    plt.close(figure)

    status = "SMOKE ONLY — NOT THESIS RESULTS" if experiment["smoke_only"] else "INCOMPLETE" if missing else "COMPLETED INTERNAL BENCHMARK"
    selected = ledger["selected_on_validation"]
    summary = {
        "status": status, "missing_test_groups_conditions": missing,
        "expected_training_runs": len(plan["jobs"]), "completed_training_runs": len(list((output / "runs").glob("*/complete.json"))),
        "experiment": experiment["fingerprint"], "selected_on_validation": selected,
        "historical_exposure": experiment["historical_exposure"],
        "limitations": ["Exact duplicate grouping does not identify every acquisition session or near duplicate.",
                        "Synthetic corruptions do not establish real factory robustness.",
                        "FNR point estimates are not statistical safety guarantees.",
                        "Deployment prevalence and costs are assumptions, not measured operating conditions.",
                        "Seed standard deviations and sampling confidence intervals quantify different uncertainty sources."],
    }
    write_json(report / "status.json", summary)
    selected_columns = ["group", "variant", "fnr", "fnr_ci_low", "fnr_ci_high", "review_rate", "deployment_review_rate", "macro_f1_all", "cost_per_1000"]
    text = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Straw thesis results</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ccc;padding:6px}}img{{max-width:100%}}.status{{background:#fff1c2;padding:14px}}li{{margin:8px 0}}</style>
<h1>Straw defect thesis benchmark</h1><p class="status">{html.escape(status)}</p>
<p>Validation-selected candidate: <b>{html.escape(selected)}</b>. All preregistered comparisons are reported.</p>
<p>{html.escape(experiment['historical_exposure'])}</p>
<p>Missing group/condition pairs: {len(missing)}. Review is counted as unclassified in macro F1 and recall.</p>
<img src="comparison.png" alt="Baseline and triage comparison"><img src="robustness.png" alt="Synthetic stress FNR"><img src="risk_coverage.png" alt="Risk coverage">
<h2>Clean test results per seed</h2>{clean[selected_columns].to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Limits of the evidence</h2><ul>{''.join('<li>' + html.escape(item) + '</li>' for item in summary['limitations'])}</ul>
<p><a href="all_results.csv">All results</a> · <a href="seed_mean_std.csv">Mean and standard deviation</a> · <a href="error_analysis.csv">Error analysis</a> · <a href="status.json">Completion and evidence status</a></p></html>"""
    (report / "index.html").write_text(text, encoding="utf-8")
    print(f"REPORT {status}: {report / 'index.html'}")
