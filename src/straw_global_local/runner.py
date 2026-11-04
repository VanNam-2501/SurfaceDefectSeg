"""prepare -> cache -> plan -> train -> benchmark -> freeze -> test -> report.

Training selects checkpoints on validation. Test inference requires an immutable
ledger, complete runs, and actual review evidence (except explicitly marked smoke).
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from .data import (CLASSES, VARIANTS, ViewDataset, create_cache, digest, file_hash,
                   load_cache, load_protocol, prepare, read_json, render_arrays,
                   tensor_views, write_json)
from .model import GlobalLocalCNN

REVIEW_FIELDS = ("labels_reviewed", "related_groups_reviewed", "dataset_rights_recorded", "related_work_reviewed")
REQUIRED_CORE = list(VARIANTS)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment(device):
    import torchvision
    import sklearn
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "determinism": "warn_only: nondeterministic operations are reported, not claimed impossible"}


def source_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def check_config(config):
    if config.get("purpose") not in ("smoke", "development", "confirmatory"):
        raise ValueError("Declare purpose: smoke/development/confirmatory")
    if not config["variants"] or len(set(config["variants"])) != len(config["variants"]) or set(config["variants"]) - set(VARIANTS):
        raise ValueError("Invalid variants")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Declare unique seeds")
    for key in ("epochs", "batch_size", "grad_accum", "patience"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config.get("workers", 0) < 0 or config["encoder_lr"] <= 0 or config["head_lr"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid training parameters")
    effective = config["batch_size"] * config["grad_accum"]
    for variant, value in config.get("batches", {}).items():
        if variant not in VARIANTS or any(not isinstance(value.get(k), int) or value[k] < 1 for k in ("batch_size", "grad_accum")):
            raise ValueError("Invalid per-variant batch settings")
        if value["batch_size"] * value["grad_accum"] != effective:
            raise ValueError("Effective image batch must match across variants")


def jobs(config):
    return [{"id": f"{variant}_s{seed}", "variant": variant, "seed": seed}
            for variant in config["variants"] for seed in config["seeds"]]


def open_experiment(args, verify_cache=False):
    config = read_json(args.config)
    check_config(config)
    protocol, frames, cached = load_cache(args.protocol, args.cache, verify=verify_cache)
    if protocol["context"]["smoke_only"] and config["purpose"] != "smoke":
        raise ValueError("Smoke data cannot support development/confirmatory claims")
    spec = {"schema": 1, "config": config, "protocol": protocol["fingerprint"],
            "cache": cached["fingerprint"], "source": source_hash(), "classes": CLASSES}
    spec["fingerprint"] = digest(spec)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "experiment.json"
    if target.exists() and read_json(target) != spec:
        raise ValueError("Code/config/protocol/cache changed: use a new experiment output")
    if not target.exists():
        write_json(target, spec)
    write_json(args.output / "plan.json", {"jobs": jobs(config), "run_count": len(jobs(config)),
        "purpose": config["purpose"], "test_requires_freeze": True,
        "claim": "Planned experiments, not evidence of accuracy improvement"})
    return config, protocol, frames, spec


def device_for(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: explicitly use --device cpu")
    return device


def selected_jobs(config, requested):
    result = [j for j in jobs(config) if not requested or j["id"] in requested]
    if not result or requested and set(requested) - {j["id"] for j in result}:
        raise ValueError("Unknown run ID")
    return result


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def classification_metrics(labels, logits):
    labels, logits = np.asarray(labels, dtype=int), np.asarray(logits, dtype=float)
    if logits.shape != (len(labels), 5) or not np.isfinite(logits).all():
        raise ValueError("Expected finite aligned N x 5 logits")
    shifted = logits - logits.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(1, keepdims=True)
    predictions = logits.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=list(range(5)), zero_division=0)
    defect = labels != 0
    binary_predictions = predictions != 0
    fn = int(np.sum(defect & ~binary_predictions))
    fp = int(np.sum(~defect & binary_predictions))
    return {"n": len(labels), "accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(f1.mean()), "confusion": confusion_matrix(labels, predictions, labels=list(range(5))).tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(CLASSES)},
            "n_defect": int(defect.sum()), "n_good": int((~defect).sum()), "fn": fn, "fp": fp,
            "fnr": fn / int(defect.sum()) if defect.any() else None,
            "fpr": fp / int((~defect).sum()) if (~defect).any() else None,
            "wrong_defect_type": int(np.sum(defect & binary_predictions & (predictions != labels))),
            "defect_auroc": float(roc_auc_score(defect, 1-probabilities[:, 0])) if len(set(defect)) == 2 else None,
            "defect_average_precision": float(average_precision_score(defect, 1-probabilities[:, 0])) if defect.any() else None,
            "ce": float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]))}


def train_epoch(model, loader, optimizer, scaler, device, grad_accum):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, seen = 0., 0
    for index, (batch, labels, _) in enumerate(loader):
        batch, labels = move_batch(batch, device), labels.to(device)
        window_start = (index // grad_accum) * grad_accum * loader.batch_size
        denominator = min(grad_accum * loader.batch_size, len(loader.dataset) - window_start)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = nn.functional.cross_entropy(model(batch), labels, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        scaler.scale(loss / denominator).backward()
        if (index + 1) % grad_accum == 0 or index + 1 == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach())
        seen += len(labels)
    return loss_sum / seen


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    values, labels, ids = [], [], []
    for batch, target, samples in loader:
        values.append(model(move_batch(batch, device)).float().cpu().numpy())
        labels.extend(target.numpy().tolist())
        ids.extend(samples)
    return np.concatenate(values), np.array(labels), np.array(ids, dtype=str)


def atomic_torch_save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state()}


def restore_rng(value, generator):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])
    generator.set_state(value["loader"])


def train_one(args, config, frames, spec, job, device):
    folder = args.output / "runs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    marker, best_path, last_path = folder / "complete.json", folder / "best.pt", folder / "last.pt"
    if marker.exists():
        verify_run(args.output, job, spec)
        print(f"SKIP completed {job['id']}", flush=True)
        return
    if (args.output / "freeze.json").exists():
        raise ValueError("Cannot mutate training after freeze")
    seed_all(job["seed"])
    generator = torch.Generator().manual_seed(job["seed"] + 10000)
    params = config.get("batches", {}).get(job["variant"], {})
    batch_size, accumulation = params.get("batch_size", config["batch_size"]), params.get("grad_accum", config["grad_accum"])
    workers = config.get("workers", 0)
    train_loader = DataLoader(ViewDataset(frames["train"], args.cache, job["variant"], config.get("augmentation", True)),
                              batch_size=batch_size, shuffle=True, generator=generator, num_workers=workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(ViewDataset(frames["val"], args.cache, job["variant"]), batch_size=batch_size,
                            num_workers=workers, generator=torch.Generator().manual_seed(job["seed"] + 20000))
    model = GlobalLocalCNN(job["variant"], config.get("pretrained", True), config.get("freeze_bn", True)).to(device)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": config["encoder_lr"]},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": config["head_lr"]}],
        weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.get("amp", True))
    history, first_epoch, best, bad, elapsed = [], 0, (-1., float("-inf")), 0, 0.
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["experiment"] != spec["fingerprint"] or saved["job"] != job:
            raise ValueError("Resume provenance mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, first_epoch, best, bad, elapsed = saved["history"], saved["epoch"]+1, tuple(saved["best"]), saved["bad"], saved["elapsed"]
        restore_rng(saved["rng"], generator)
    start = time.perf_counter()
    completed_this_call = 0
    for epoch in range(first_epoch, config["epochs"]):
        if bad >= config["patience"]:
            break
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, accumulation)
        logits, labels, ids = infer(model, val_loader, device)
        metrics = classification_metrics(labels, logits)
        if not np.isfinite(metrics["ce"]):
            raise RuntimeError("Nonfinite validation loss")
        ranking = (metrics["macro_f1"], -metrics["ce"])
        if ranking > best:
            best, bad = ranking, 0
            atomic_torch_save(best_path, {"model": model.state_dict(), "job": job, "experiment": spec["fingerprint"],
                                         "epoch": epoch, "validation": metrics})
            np.savez_compressed(folder / "val_predictions.npz", logits=logits, labels=labels, sample_ids=ids)
        else:
            bad += 1
        history.append({"epoch": epoch+1, "train_ce": train_loss, "val_ce": metrics["ce"],
                        "val_macro_f1": metrics["macro_f1"], "seconds": elapsed + time.perf_counter()-start})
        scheduler.step()
        atomic_torch_save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "epoch": epoch, "best": best, "bad": bad, "elapsed": elapsed + time.perf_counter()-start,
            "rng": rng_state(generator), "experiment": spec["fingerprint"], "job": job})
        write_json(folder / "history.json", history)
        print(f"{job['id']} epoch {epoch+1}/{config['epochs']} trainCE={train_loss:.4f} valF1={metrics['macro_f1']:.4f} valCE={metrics['ce']:.4f}", flush=True)
        completed_this_call += 1
        if args.max_epochs_this_call and completed_this_call >= args.max_epochs_this_call:
            break
    finished = len(history) >= config["epochs"] or bad >= config["patience"]
    if finished:
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        write_json(folder / "complete.json", {"experiment": spec["fingerprint"], "job": job,
            "best_checkpoint": file_hash(best_path), "val_predictions": file_hash(folder / "val_predictions.npz"),
            "validation": saved["validation"], "best_epoch": saved["epoch"]+1,
            "epochs_completed": len(history), "reason": "patience" if bad >= config["patience"] else "max_epochs",
            "environment": environment(device), "parameter_count": sum(p.numel() for p in model.parameters())})


def verify_run(output, job, spec):
    folder = output / "runs" / job["id"]
    marker = read_json(folder / "complete.json")
    if marker["experiment"] != spec["fingerprint"] or marker["job"] != job:
        raise ValueError("Completed run provenance mismatch")
    if marker["best_checkpoint"] != file_hash(folder / "best.pt") or marker["val_predictions"] != file_hash(folder / "val_predictions.npz"):
        raise ValueError("Completed checkpoint/predictions changed")
    return marker


def train(args):
    config, _, frames, spec = open_experiment(args, verify_cache=True)
    device = device_for(args)
    for job in selected_jobs(config, args.run_id):
        train_one(args, config, frames, spec, job, device)


def choose_families(config, completed):
    scores = {variant: float(np.mean([completed[f"{variant}_s{seed}"]["validation"]["macro_f1"] for seed in config["seeds"]]))
              for variant in config["variants"]}
    baselines = [v for v in ("B0", "B1") if v in scores]
    proposals = [v for v in ("B3", "B4", "B5", "B6") if v in scores]
    if not baselines or not proposals:
        raise ValueError("Freeze needs a whole-image baseline and a global-local candidate")
    # Iteration order is a predefined deterministic tie break favoring simpler variants.
    return {"baseline": max(baselines, key=lambda v: scores[v]),
            "proposal": max(proposals, key=lambda v: scores[v]), "validation_mean_f1": scores}


def check_review(review):
    return all(review.get(field) is True for field in REVIEW_FIELDS) and bool(review.get("reviewer", "").strip()) and bool(review.get("notes", "").strip())


def freeze(args):
    config, protocol, _, spec = open_experiment(args, verify_cache=True)
    if config["purpose"] == "development":
        raise ValueError("Development pilot has no test access; start a confirmatory experiment")
    review = read_json(args.protocol / "review.json")
    if config["purpose"] != "smoke" and not check_review(review):
        raise ValueError("Actual label/group/rights/related-work review evidence is required in review.json; do not fabricate completion")
    completed = {j["id"]: verify_run(args.output, j, spec) for j in jobs(config)}
    ledger = {"experiment": spec["fingerprint"], "protocol": protocol["fingerprint"],
              "review": review, "selection": choose_families(config, completed),
              "runs": {key: {"checkpoint": m["best_checkpoint"], "val_predictions": m["val_predictions"]} for key, m in completed.items()},
              "purpose": config["purpose"], "rule": "mean validation macro-F1 across declared seeds; test never selects family"}
    ledger["fingerprint"] = digest(ledger)
    path = args.output / "freeze.json"
    if path.exists() and read_json(path) != ledger:
        raise ValueError("Selection/review is already frozen")
    write_json(path, ledger)
    print(json.dumps(ledger["selection"], indent=2), flush=True)


def load_ledger(args, config, spec):
    if not (args.output / "freeze.json").exists():
        raise ValueError("Test access requires a frozen selection ledger")
    ledger = read_json(args.output / "freeze.json")
    saved = ledger.pop("fingerprint")
    if digest(ledger) != saved or ledger["experiment"] != spec["fingerprint"]:
        raise ValueError("Selection ledger changed or mismatched")
    ledger["fingerprint"] = saved
    if ledger["review"] != read_json(args.protocol / "review.json"):
        raise ValueError("Review evidence changed after freeze")
    for job in jobs(config):
        marker = verify_run(args.output, job, spec)
        if ledger["runs"][job["id"]]["checkpoint"] != marker["best_checkpoint"]:
            raise ValueError("Checkpoint changed after selection")
    return ledger


def test(args):
    config, _, frames, spec = open_experiment(args, verify_cache=True)
    ledger = load_ledger(args, config, spec)
    device = device_for(args)
    for job in jobs(config):
        folder = args.output / "runs" / job["id"]
        target, marker = folder / "test_predictions.npz", folder / "test.json"
        if marker.exists():
            record = read_json(marker)
            if record["freeze"] != ledger["fingerprint"] or record["predictions"] != file_hash(target):
                raise ValueError("Frozen test output changed")
            print(f"SKIP test {job['id']}", flush=True)
            continue
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        model = GlobalLocalCNN(job["variant"], pretrained=False, freeze_bn=config.get("freeze_bn", True)).to(device)
        model.load_state_dict(checkpoint["model"])
        loader = DataLoader(ViewDataset(frames["test"], args.cache, job["variant"]), batch_size=4,
                            num_workers=config.get("workers", 0))
        logits, labels, ids = infer(model, loader, device)
        if not np.array_equal(ids, frames["test"].sample_id.astype(str).to_numpy()):
            raise ValueError("Test sample alignment changed")
        np.savez_compressed(target, logits=logits, labels=labels, sample_ids=ids)
        write_json(marker, {"freeze": ledger["fingerprint"], "predictions": file_hash(target),
                            "metrics": classification_metrics(labels, logits)})
        print(f"TEST {job['id']} complete ({config['purpose']})", flush=True)


def paired_group_interval(labels, group_ids, first_predictions, second_predictions, repeats=1000, seed=2026):
    labels, group_ids = np.asarray(labels), np.asarray(group_ids)
    first_predictions, second_predictions = np.asarray(first_predictions), np.asarray(second_predictions)
    if first_predictions.shape != second_predictions.shape or first_predictions.ndim != 2 or first_predictions.shape[1] != len(labels):
        raise ValueError("Expected aligned seed x image predictions")
    groups = [np.flatnonzero(group_ids == group) for group in np.unique(group_ids)]
    rng = np.random.default_rng(seed)
    def delta(indices):
        return float(np.mean([f1_score(labels[indices], a[indices], labels=range(5), average="macro", zero_division=0)
                          - f1_score(labels[indices], b[indices], labels=range(5), average="macro", zero_division=0)
                          for a, b in zip(first_predictions, second_predictions)]))
    observed = delta(np.arange(len(labels)))
    samples = [delta(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(repeats)]
    lo, hi = np.quantile(samples, [.025, .975])
    return {"delta_macro_f1": observed, "ci95": [float(lo), float(hi)], "repeats": repeats,
            "interpretation": "positive evidence" if lo > 0 else "negative evidence" if hi < 0 else "no clear difference",
            "scope": "paired image-group sampling conditional on trained seeds; not 3x independent test images"}


def benchmark(args):
    config, _, frames, spec = open_experiment(args, verify_cache=True)
    device = device_for(args)
    samples = frames["val"].head(5)
    for job in selected_jobs(config, args.run_id):
        if not args.run_id and job["seed"] != config["seeds"][0]:
            continue
        marker = verify_run(args.output, job, spec)
        model = GlobalLocalCNN(job["variant"], pretrained=False, freeze_bn=config.get("freeze_bn", True)).to(device).eval()
        folder = args.output / "runs" / job["id"]
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["model"])
        total_times, forward_times = [], []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for index in range(args.repeats + 5):
            row = samples.iloc[index % len(samples)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with Image.open(row.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                batch = tensor_views(render_arrays(image, job["variant"]), job["variant"])
            batch = move_batch({key: value[None] for key, value in batch.items()}, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before_forward = time.perf_counter()
            with torch.inference_mode():
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            if index >= 5:
                total_times.append((end-start)*1000)
                forward_times.append((end-before_forward)*1000)
        write_json(folder / "benchmark.json", {"checkpoint": marker["best_checkpoint"], "experiment": spec["fingerprint"],
            "environment": environment(device), "samples": "first 5 validation images, cyclic", "warmup": 5,
            "repeats": args.repeats, "batch": 1, "includes": "original image decode + view generation + normalization + transfer + forward",
            "end_to_end_ms": {"p50": float(np.median(total_times)), "p95": float(np.quantile(total_times, .95))},
            "forward_ms": {"p50": float(np.median(forward_times)), "p95": float(np.quantile(forward_times, .95))},
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "parameters": marker["parameter_count"]})
        print(f"BENCHMARK {job['id']}", flush=True)


def report(args):
    config, protocol, frames, spec = open_experiment(args)
    folder = args.output / "report"
    folder.mkdir(parents=True, exist_ok=True)
    completed, records, missing = {}, [], []
    for job in jobs(config):
        path = args.output / "runs" / job["id"] / "complete.json"
        if not path.exists():
            missing.append(job["id"])
            continue
        marker = verify_run(args.output, job, spec)
        completed[job["id"]] = marker
        records.append({"run": job["id"], "variant": job["variant"], "seed": job["seed"],
                        "best_epoch": marker["best_epoch"], **{f"val_{key}": marker["validation"][key] for key in ("macro_f1", "accuracy", "fnr", "fpr")}})
    pd.DataFrame(records).to_csv(folder / "validation_runs.csv", index=False)
    review = read_json(args.protocol / "review.json")
    ready = {"purpose_confirmatory": config["purpose"] == "confirmatory", "not_smoke_data": not protocol["context"]["smoke_only"],
             "all_core_variants": set(REQUIRED_CORE).issubset(config["variants"]), "at_least_3_seeds": len(config["seeds"]) >= 3,
             "human_review_recorded": check_review(review), "all_training_complete": not missing,
             "selection_frozen": (args.output / "freeze.json").exists(), "all_test_outputs_verified": False,
             "cost_measured": False}
    comparison, tests = None, []
    if ready["selection_frozen"]:
        ledger = load_ledger(args, config, spec)
        predicted = {}
        for job in jobs(config):
            run_folder = args.output / "runs" / job["id"]
            target, marker_path = run_folder / "test_predictions.npz", run_folder / "test.json"
            if not marker_path.exists():
                continue
            marker = read_json(marker_path)
            if marker["freeze"] != ledger["fingerprint"] or marker["predictions"] != file_hash(target):
                raise ValueError("Report found modified test artifacts")
            with np.load(target, allow_pickle=False) as saved:
                if not np.array_equal(saved["sample_ids"], frames["test"].sample_id.astype(str).to_numpy()) or not np.array_equal(saved["labels"], frames["test"].label.to_numpy()):
                    raise ValueError("Test sample/label alignment mismatch")
                predicted[job["id"]] = saved["logits"].argmax(1)
                recomputed = classification_metrics(saved["labels"], saved["logits"])
            if recomputed != marker["metrics"]:
                raise ValueError("Stored test metric mismatch")
            tests.append({"variant": job["variant"], "seed": job["seed"], **{k: marker["metrics"][k] for k in ("macro_f1", "accuracy", "fnr", "fpr", "fn", "fp", "wrong_defect_type")}})
        ready["all_test_outputs_verified"] = len(tests) == len(jobs(config))
        if ready["all_test_outputs_verified"]:
            baseline, proposal = ledger["selection"]["baseline"], ledger["selection"]["proposal"]
            comparison = paired_group_interval(frames["test"].label.to_numpy(), frames["test"].group_id.to_numpy(),
                [predicted[f"{proposal}_s{s}"] for s in config["seeds"]],
                [predicted[f"{baseline}_s{s}"] for s in config["seeds"]], repeats=args.bootstrap)
            comparison.update({"baseline": baseline, "proposal": proposal})
    benchmarks = []
    for variant in config["variants"]:
        path = args.output / "runs" / f"{variant}_s{config['seeds'][0]}" / "benchmark.json"
        if path.exists():
            value = read_json(path)
            expected = completed.get(f"{variant}_s{config['seeds'][0]}", {})
            if value["experiment"] != spec["fingerprint"] or value["checkpoint"] != expected.get("best_checkpoint"):
                raise ValueError("Benchmark provenance mismatch")
            benchmarks.append({"variant": variant, "p50_end_to_end_ms": value["end_to_end_ms"]["p50"],
                               "p95_end_to_end_ms": value["end_to_end_ms"]["p95"], "parameters": value["parameters"]})
    ready["cost_measured"] = len(benchmarks) == len(config["variants"])
    pd.DataFrame(benchmarks).to_csv(folder / "benchmarks.csv", index=False)
    pd.DataFrame(tests).to_csv(folder / "test_runs.csv", index=False)
    if tests:
        pd.DataFrame(tests).groupby("variant").agg({k: ["mean", "std"] for k in ("macro_f1", "accuracy", "fnr", "fpr")}).to_csv(folder / "test_summary.csv")
    status = "READY_FOR_INTERNAL_THESIS_ANALYSIS" if all(ready.values()) else "NOT_READY_FOR_THESIS_CONCLUSIONS"
    evidence = {"status": status, "checks": ready, "missing_runs": missing, "primary_comparison": comparison,
                "limits": ["Readiness is evidence completeness, not university approval or proof of novelty.",
                           "Positive improvement is not required for an honest research conclusion.",
                           "Image groups are inherited heuristics; no independent factory test.",
                           "Annotation/review entries are human attestations, not machine-verified truth."],
                "experiment": spec["fingerprint"]}
    write_json(folder / "readiness.json", evidence)
    lines = ["# Báo cáo thí nghiệm CNN toàn cục–cục bộ", "", f"**{status}**", "",
             f"Purpose: {config['purpose']}. Train/Val/Test: {protocol['counts']}.", "",
             "| Điều kiện bằng chứng | Đã có |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in ready.items()]
    lines += ["", "## So sánh chính đã chọn bằng Validation", "", json.dumps(comparison, ensure_ascii=False, indent=2),
              "", "## Giới hạn", ""] + [f"- {item}" for item in evidence["limits"]]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "cache", "plan", "train", "freeze", "test", "benchmark", "report"])
    parser.add_argument("--config", type=Path, default=Path("configs/straw_global_local.json"))
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/straw_global_local/data_v1"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/straw_global_local/cache_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/straw_global_local/confirmatory_v1"))
    parser.add_argument("--source-protocol", type=Path, default=Path("artifacts/straw_thesis/data_v2/iid"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--smoke-per-class", type=int)
    parser.add_argument("--cache-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--max-epochs-this-call", type=int)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.smoke_per_class is not None and args.smoke_per_class < 1 or args.repeats < 1 or args.bootstrap < 1 or args.cache_workers < 1:
        parser.error("Counts must be positive")
    if args.max_epochs_this_call is not None and args.max_epochs_this_call < 1:
        parser.error("--max-epochs-this-call must be positive")
    torch.set_num_threads(2)
    if args.action == "prepare":
        print(json.dumps(prepare(args.source_protocol, args.protocol, args.image_root, args.smoke_per_class), indent=2))
    elif args.action == "cache":
        result = create_cache(args.protocol, args.cache, args.cache_workers)
        print(f"Verified cache: {len(result['files'])} original images")
    elif args.action == "plan":
        open_experiment(args)
        print(json.dumps(read_json(args.output / "plan.json"), indent=2))
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()
