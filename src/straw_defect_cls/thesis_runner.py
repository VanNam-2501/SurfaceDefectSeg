"""Reproducible CLI: prepare -> plan -> train -> calibrate -> freeze -> test -> report.

Test access requires a frozen selection ledger. A smoke run can exercise every
step, but its results are permanently labelled smoke and cannot be a thesis result.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .thesis_protocol import (CLASSES, ROLES, digest_file, digest_object, export_protocol,
                              load_protocol, prepare, split_groups, write_json)
from .thesis_evaluation import (choose_policy, decision_metrics, fit_temperature,
                                grouped_bootstrap, predict, probability_metrics, risk_coverage, discrimination_metrics)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ["triage", "selective_full5", "no_prior", "no_temperature", "no_guards"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_fingerprint():
    return digest_object({p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def jobs(config, suite, protocol):
    smoke = suite == "smoke"
    lobo = protocol.get("held_out_brand") is not None
    backbones = ["resnet18"] if smoke or lobo else config["backbones"]
    seeds = config["seeds"][:1] if smoke or lobo else config["seeds"]
    settings = [("core", config["core_resolution"], "safe", 1.0, backbones, config["stages"])]
    if suite == "extended" and not lobo:
        ext = config["extensions"]
        settings += [
            ("highres", ext["high_resolution"]["resolution"], "safe", 1.0, [ext["high_resolution"]["backbone"]], config["stages"]),
            ("noaug", config["core_resolution"], "none", 1.0, [ext["augmentation"]["backbone"]], config["stages"]),
            ("weighted", config["core_resolution"], "safe", ext["class_weight"]["defect_weight"], [ext["class_weight"]["backbone"]], ["1"]),
        ]
    result = []
    for setting, resolution, augmentation, weight, architectures, stages in settings:
        for backbone in architectures:
            for seed in seeds:
                for stage in stages:
                    group = f"{setting}_{backbone}_r{resolution}_s{seed}"
                    result.append({"id": f"{group}_stage{stage}", "group": group, "setting": setting,
                                   "backbone": backbone, "seed": seed, "stage": stage, "resolution": resolution,
                                   "augmentation": augmentation, "stage1_defect_weight": weight})
    return result


def open_experiment(args):
    config = read_json(args.config)
    protocol = load_protocol(args.protocol)
    if protocol["smoke_only"] and args.suite != "smoke":
        raise ValueError("A smoke subset cannot be used for a thesis training suite")
    spec = {"schema": 1, "config": config, "suite": args.suite,
            "protocol_fingerprint": protocol["fingerprint"], "source_fingerprint": source_fingerprint(),
            "smoke_only": args.suite == "smoke", "historical_exposure": protocol["historical_exposure"]}
    spec["fingerprint"] = digest_object(spec)
    path = args.output / "experiment.json"
    if path.exists() and read_json(path) != spec:
        raise ValueError("Experiment config, code, or protocol changed. Use a new output directory.")
    if not path.exists():
        write_json(path, spec)
    write_json(args.output / "protocol_context.json", protocol)
    job_list = jobs(config, args.suite, protocol)
    write_json(args.output / "plan.json", {"training_runs": len(job_list), "jobs": job_list,
                                           "test_conditions": config["conditions"], "smoke_only": spec["smoke_only"]})
    return config, protocol, job_list, spec


def checkpoint_for(output, job):
    return output / "runs" / job["id"] / "checkpoints" / "best.pt"


def train(args):
    import torch
    from .train_engine import TrainConfig, run_training
    config, protocol, job_list, spec = open_experiment(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Pass --device cpu explicitly for CPU execution.")
    # Verify actual source files before training, not only CSV integrity.
    load_protocol(args.protocol, check_images=True)
    selected = [job for job in job_list if not args.run_id or job["id"] in args.run_id]
    if not selected or (args.run_id and set(args.run_id) - {j["id"] for j in selected}):
        raise ValueError("Unknown or empty --run-id selection")
    for job in selected:
        run_dir = args.output / "runs" / job["id"]
        marker = run_dir / "complete.json"
        checkpoint = checkpoint_for(args.output, job)
        if marker.exists():
            complete = read_json(marker)
            if complete["experiment"] != spec["fingerprint"] or not checkpoint.exists() or complete["checkpoint_sha256"] != digest_file(checkpoint):
                raise ValueError(f"Completed run integrity check failed: {job['id']}")
            print(f"SKIP completed {job['id']}", flush=True)
            continue
        params = dict(config["training"])
        params.update(config["batches"][job["backbone"]])
        if args.suite == "smoke":
            params.update(epochs=1, batch_size=4, grad_accum=1, num_workers=0, pretrained=False)
        if args.workers is not None:
            params["num_workers"] = args.workers
        if job["resolution"] == 384:
            params["batch_size"] = max(1, params["batch_size"] // 2)
            params["grad_accum"] *= 2
        params.update(model_name=job["backbone"], stage=job["stage"], image_size=job["resolution"],
                      seed=job["seed"], augmentation=job["augmentation"], preprocessing="resize",
                      stage1_defect_weight=job["stage1_defect_weight"],
                      train_csv=str((args.protocol / "train.csv").resolve()), val_csv=str((args.protocol / "val.csv").resolve()),
                      masks_csv="", output_dir=str((args.output / "runs").resolve()), tag=job["id"],
                      device=args.device, resume=True, protocol_fingerprint=protocol["fingerprint"])
        run_training(TrainConfig(**params))
        summary = read_json(run_dir / "summary.json")
        if summary["completion_reason"] == "max_epochs" and summary["epochs_completed"] != params["epochs"]:
            raise ValueError("Incomplete run cannot be marked complete")
        write_json(marker, {"experiment": spec["fingerprint"], "checkpoint_sha256": digest_file(checkpoint), "job": job})


def cached_logits(args, spec, job, role, condition="clean"):
    from .thesis_inference import infer
    frame = pd.read_csv(args.protocol / f"{role}.csv")
    checkpoint = checkpoint_for(args.output, job)
    marker_path = checkpoint.parent.parent / "complete.json"
    if not marker_path.exists():
        raise ValueError(f"Training not completed: {job['id']}")
    marker = read_json(marker_path)
    checkpoint_hash = digest_file(checkpoint)
    if marker["experiment"] != spec["fingerprint"] or marker["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint/protocol mismatch")
    provenance = {"experiment": spec["fingerprint"], "checkpoint": checkpoint_hash,
                  "manifest": digest_file(args.protocol / f"{role}.csv"), "role": role, "condition": condition}
    path = args.output / "cache" / job["id"] / f"{role}_{condition}.npz"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        saved_metadata = read_json(metadata)
        if saved_metadata.get("provenance") == provenance and saved_metadata.get("sha256") == digest_file(path):
            with np.load(path, allow_pickle=False) as saved:
                if np.array_equal(saved["sample_ids"], frame.sample_id.to_numpy()):
                    return saved["logits"]
    print(f"INFER {job['id']} {role} {condition}", flush=True)
    values = infer(checkpoint, frame, condition, args.device, batch_size=4 if args.suite == "smoke" else 16)
    if values.shape != (len(frame), 2 if job["stage"] == "1" else 4 if job["stage"] == "2" else 5) or not np.isfinite(values).all():
        raise ValueError("Invalid model logits")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=values, sample_ids=frame.sample_id.to_numpy(dtype=str))
    write_json(metadata, {"provenance": provenance, "sha256": digest_file(path)})
    return values


def grouped_jobs(job_list):
    groups = {}
    for job in job_list:
        groups.setdefault(job["group"], {})[job["stage"]] = job
    # Weight ablation only retrains Stage 1, keeping the same core Stage 2/full5.
    for name, stage_jobs in groups.items():
        first = next(iter(stage_jobs.values()))
        if first["setting"] == "weighted":
            core = f"core_{first['backbone']}_r{first['resolution']}_s{first['seed']}"
            stage_jobs.update({stage: groups[core][stage] for stage in ["2", "full5"]})
        if set(stage_jobs) != {"1", "2", "full5"}:
            raise ValueError(f"Missing stages for {name}")
    return groups


def calibrate(args):
    config, protocol, job_list, spec = open_experiment(args)
    if (args.output / "freeze.json").exists():
        raise ValueError("Experiment is frozen. Start a new experiment to recalibrate.")
    load_protocol(args.protocol, check_images=True)
    cal = pd.read_csv(args.protocol / "calibration.csv")
    val = pd.read_csv(args.protocol / "val.csv")
    train_frame = pd.read_csv(args.protocol / "train.csv")
    selection_rows = []
    for group, stage_jobs in grouped_jobs(job_list).items():
        if args.group and group not in args.group:
            continue
        cal_logits = {stage: cached_logits(args, spec, job, "calibration") for stage, job in stage_jobs.items()}
        val_logits = {stage: cached_logits(args, spec, job, "val") for stage, job in stage_jobs.items()}
        masks = {"1": np.ones(len(cal), dtype=bool), "2": cal.label.to_numpy() > 0,
                 "full5": np.ones(len(cal), dtype=bool)}
        labels = {"1": (cal.label.to_numpy() > 0).astype(int), "2": cal.label.to_numpy() - 1, "full5": cal.label.to_numpy()}
        calibration = {stage: fit_temperature(cal_logits[stage][masks[stage]], labels[stage][masks[stage]]) for stage in stage_jobs}
        for stage in stage_jobs:
            selected = val.label.to_numpy() > 0 if stage == "2" else np.ones(len(val), dtype=bool)
            truth = (val.label.to_numpy() > 0).astype(int) if stage == "1" else val.label.to_numpy() - 1 if stage == "2" else val.label.to_numpy()
            calibration[stage]["validation_before"] = probability_metrics(val_logits[stage][selected], truth[selected])
            calibration[stage]["validation_after"] = probability_metrics(val_logits[stage][selected], truth[selected], calibration[stage]["temperature"])
        policy_config = dict(config["policy"], pi_train=float((train_frame.label > 0).mean()))
        bundle = {"experiment": spec["fingerprint"], "group": group, "calibration": calibration, "policies": {},
                  "checkpoints": {stage: digest_file(checkpoint_for(args.output, job)) for stage, job in stage_jobs.items()},
                  "policy_config": policy_config}
        policies = {"full5": {"kind": "full5"}, "hard_cascade": {"kind": "hard_cascade"}}
        for variant in VARIANTS:
            chosen, search = choose_policy(val_logits, val.label.to_numpy(), calibration, policy_config, variant)
            policies[variant] = chosen
            scan_path = args.output / "selection" / group / f"{variant}_search.csv"
            scan_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(search).to_csv(scan_path, index=False)
        for variant, policy in policies.items():
            m = decision_metrics(val.label.to_numpy(), predict(val_logits, policy), policy_config["pi_deploy"], policy_config["costs"])
            policies[variant]["validation_metrics"] = m
            selection_rows.append({"group": group, "variant": variant, "status": policy.get("selection_status", "baseline"),
                                   **{k: m[k] for k in ["fnr", "fpr", "review_rate", "deployment_review_rate", "cost_per_1000", "macro_f1_all"]}})
        bundle["policies"] = policies
        write_json(args.output / "selection" / group / "policies.json", bundle)
        pd.DataFrame(risk_coverage(val_logits["full5"], val.label.to_numpy(), calibration["full5"]["temperature"])).to_csv(args.output / "selection" / group / "risk_coverage.csv", index=False)
        print(f"CALIBRATED {group}: {policies['triage']['selection_status']}", flush=True)
    # Merge all groups so sharded calibration does not erase earlier selections.
    all_rows = []
    for path in sorted((args.output / "selection").glob("*/policies.json")):
        bundle = read_json(path)
        for name, policy in bundle["policies"].items():
            m = policy["validation_metrics"]
            all_rows.append({"group": bundle["group"], "variant": name, "status": policy.get("selection_status", "baseline"),
                             **{k: m[k] for k in ["fnr", "fpr", "review_rate", "deployment_review_rate", "cost_per_1000", "macro_f1_all"]}})
    if not all_rows:
        raise ValueError("No calibrated groups")
    pd.DataFrame(all_rows).to_csv(args.output / "validation_summary.csv", index=False)


def freeze(args):
    config, protocol, job_list, spec = open_experiment(args)
    load_protocol(args.protocol, check_images=True)
    groups = grouped_jobs(job_list)
    selected = {}
    for group in groups:
        path = args.output / "selection" / group / "policies.json"
        bundle = read_json(path)
        if bundle["experiment"] != spec["fingerprint"]:
            raise ValueError("Selection experiment mismatch")
        for stage, job in groups[group].items():
            if digest_file(checkpoint_for(args.output, job)) != bundle["checkpoints"][stage]:
                raise ValueError("Checkpoint changed after calibration")
        selected[group] = digest_file(path)
    # Select deployment candidate from validation only; retain all preregistered comparisons.
    validation = pd.read_csv(args.output / "validation_summary.csv")
    triage = validation[validation.variant == "triage"].copy()
    triage["family"] = triage.group.str.replace(r"_s\d+$", "", regex=True)
    triage["infeasible"] = (triage.status != "feasible_empirical").astype(int)
    # Choose a family using all preregistered seeds, then the smallest seed for deployment.
    families = triage.groupby("family")[["infeasible", "fnr", "deployment_review_rate", "cost_per_1000"]].mean()
    families = families.sort_values(["infeasible", "fnr", "cost_per_1000", "deployment_review_rate"])
    best_family = families.index[0]
    best = triage[triage.family == best_family].sort_values("group").iloc[0]
    ledger = {"experiment": spec["fingerprint"], "policy_sha256": selected,
              "selected_on_validation": str(best.group), "selection_status": str(best.status),
              "selected_family": best_family, "selection_rule": "family mean across all seeds, then smallest seed; no test-based selection",
              "smoke_only": args.suite == "smoke", "all_conditions": config["conditions"],
              "historical_exposure": protocol["historical_exposure"]}
    ledger["fingerprint"] = digest_object(ledger)
    path = args.output / "freeze.json"
    if path.exists() and read_json(path) != ledger:
        raise ValueError("Freeze ledger already exists with different contents")
    write_json(path, ledger)
    print(json.dumps(ledger, indent=2))


def verify_freeze(args, spec):
    ledger = read_json(args.output / "freeze.json")
    claimed = ledger.pop("fingerprint")
    if digest_object(ledger) != claimed or ledger["experiment"] != spec["fingerprint"]:
        raise ValueError("Invalid freeze ledger")
    ledger["fingerprint"] = claimed
    for group, expected in ledger["policy_sha256"].items():
        if digest_file(args.output / "selection" / group / "policies.json") != expected:
            raise ValueError("Policy changed after freezing")
    return ledger


def evaluate_test(args):
    config, protocol, job_list, spec = open_experiment(args)
    ledger = verify_freeze(args, spec)
    load_protocol(args.protocol, check_images=True)
    test = pd.read_csv(args.protocol / "test.csv")
    conditions = ["clean", "noise_2"] if args.suite == "smoke" else config["conditions"]
    group_map = grouped_jobs(job_list)
    for group, stage_jobs in group_map.items():
        if args.group and group not in args.group:
            continue
        bundle = read_json(args.output / "selection" / group / "policies.json")
        for stage, job in stage_jobs.items():
            if digest_file(checkpoint_for(args.output, job)) != bundle["checkpoints"][stage]:
                raise ValueError("Checkpoint changed after frozen calibration")
        for condition in conditions:
            destination = args.output / "test" / group / condition
            marker = destination / "complete.json"
            if marker.exists():
                saved = read_json(marker)
                if saved["freeze"] != ledger["fingerprint"]:
                    raise ValueError("Test was run under a different freeze")
                for filename, expected in saved["files"].items():
                    if digest_file(destination / filename) != expected:
                        raise ValueError("Test output changed after completion")
                print(f"SKIP tested {group} {condition}", flush=True)
                continue
            values = {stage: cached_logits(args, spec, job, "test", condition) for stage, job in stage_jobs.items()}
            destination.mkdir(parents=True, exist_ok=True)
            rows, predictions, bootstraps = [], test[["sample_id", "group_id", "brand", "label"]].copy(), {}
            for name, policy in bundle["policies"].items():
                pred = predict(values, policy)
                predictions[name] = pred
                m = decision_metrics(test.label, pred, config["policy"]["pi_deploy"], config["policy"]["costs"])
                # Record the whole metric schema, including confusion counts and CIs.
                rows.append({"group": group, "condition": condition, "variant": name, "smoke_only": args.suite == "smoke", **m})
            if condition == "clean":
                diagnostics = {}
                for stage in stage_jobs:
                    mask = test.label.to_numpy() > 0 if stage == "2" else np.ones(len(test), dtype=bool)
                    truth = (test.label.to_numpy() > 0).astype(int) if stage == "1" else test.label.to_numpy() - 1 if stage == "2" else test.label.to_numpy()
                    diagnostics[stage] = {"before": probability_metrics(values[stage][mask], truth[mask]),
                                          "after": probability_metrics(values[stage][mask], truth[mask], bundle["calibration"][stage]["temperature"]),
                                          "discrimination": discrimination_metrics(values[stage][mask], truth[mask])}
                write_json(destination / "calibration_diagnostics.json", diagnostics)
                iterations = 20 if args.suite == "smoke" else config["bootstrap_replicates"]
                bootstraps = grouped_bootstrap(test.label, predictions.triage, test.group_id, iterations,
                                              other=predictions.full5, pi_deploy=config["policy"]["pi_deploy"], costs=config["policy"]["costs"])
                pd.DataFrame(risk_coverage(values["full5"], test.label.to_numpy(), bundle["calibration"]["full5"]["temperature"])).to_csv(destination / "risk_coverage.csv", index=False)
                # Prior and cost sensitivity: never retune a policy on test.
                sensitivity = []
                for prior in [0.01, 0.05, 0.10, 0.20]:
                    for fn_cost in [20, 100, 500]:
                        for name in ["full5", "selective_full5", "triage"]:
                            costs = dict(config["policy"]["costs"], fn=fn_cost)
                            m = decision_metrics(test.label, predictions[name], prior, costs)
                            sensitivity.append({"pi_deploy": prior, "fn_cost": fn_cost, "variant": name,
                                                "cost_per_1000": m["cost_per_1000"], "deployment_review_rate": m["deployment_review_rate"]})
                pd.DataFrame(sensitivity).to_csv(destination / "sensitivity.csv", index=False)
            write_json(destination / "metrics.json", rows)
            write_json(destination / "paired_bootstrap.json", bootstraps)
            predictions.to_csv(destination / "predictions.csv", index=False)
            files = {p.name: digest_file(p) for p in destination.iterdir() if p.is_file() and p.name != "complete.json"}
            write_json(marker, {"freeze": ledger["fingerprint"], "files": files})
            print(f"TESTED {group} {condition}", flush=True)


def make_smoke(source: Path, output: Path):
    metadata = load_protocol(source)
    frames = []
    for role in ROLES:
        frame = pd.read_csv(source / f"{role}.csv")
        sampled = frame.groupby("label", group_keys=False).head(4 if role == "train" else 2)
        frames.append(sampled.assign(role=role))
    return export_protocol(pd.concat(frames, ignore_index=True), output, metadata["seed"], Path(metadata["source_root"]), smoke=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke-data", "lobo-data", "plan", "train", "calibrate", "freeze", "test", "benchmark", "report"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "straw_thesis.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifacts" / "straw_thesis" / "data_v2" / "iid")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "straw_defects_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "straw_thesis" / "core_v1")
    parser.add_argument("--suite", choices=["core", "extended", "smoke"], default="core")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-id", action="append", help="Train specific job IDs (repeat to shard jobs)")
    parser.add_argument("--group", action="append", help="Calibrate/test specific group IDs")
    args = parser.parse_args()
    args.output, args.protocol = args.output.resolve(), args.protocol.resolve()
    if args.command == "prepare":
        print(json.dumps(prepare(args.data_root, args.output), indent=2))
    elif args.command == "smoke-data":
        print(json.dumps(make_smoke(args.protocol, args.output), indent=2))
    elif args.command == "lobo-data":
        meta = load_protocol(args.protocol)
        frame = pd.read_csv(args.protocol / "inventory.csv")
        for brand in sorted(frame.brand.unique()):
            name = brand.lower().replace(" ", "_")
            destination = args.output / name
            if (destination / "protocol.json").exists():
                load_protocol(destination)
                continue
            export_protocol(split_groups(frame, 42, brand), destination, 42, Path(meta["source_root"]), brand)
            print(f"LOBO protocol: {destination}", flush=True)
    elif args.command == "plan":
        _, _, job_list, spec = open_experiment(args)
        print(json.dumps({"runs": len(job_list), "groups": list(grouped_jobs(job_list)), "experiment": spec["fingerprint"]}, indent=2))
    elif args.command == "train":
        train(args)
    elif args.command == "calibrate":
        calibrate(args)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "test":
        evaluate_test(args)
    elif args.command == "benchmark":
        from .thesis_inference import benchmark
        _, _, job_list, _ = open_experiment(args)
        for job in job_list:
            if not args.run_id or job["id"] in args.run_id:
                write_json(args.output / "benchmarks" / f"{job['id']}.json", benchmark(checkpoint_for(args.output, job), args.device))
    elif args.command == "report":
        from .thesis_report import generate_report
        _, _, _, spec = open_experiment(args)
        verify_freeze(args, spec)
        generate_report(args.output)


if __name__ == "__main__":
    main()
