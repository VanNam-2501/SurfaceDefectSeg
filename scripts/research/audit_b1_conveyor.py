"""Compare the pinned B1 checkpoint and resident TensorRT on identical inputs.

Replays the frozen clean test and the user's exact 30 legacy Bent JPEGs. A
lossless clockwise rotation of those same JPEG pixels isolates orientation;
it does not recreate the factory simulation or tune the model on test data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.resident_inference import ResidentInference, raw_rgb_tensor
from src.straw_global_local.data import CLASSES, file_hash, load_protocol, render_arrays, tensor_views
from src.straw_global_local.deployment import TensorB1, normalize

USER_PREDICTIONS = {
    1: "pierced", 2: "bent", 3: "pierced", 5: "loose", 6: "bent", 7: "bent",
    8: "bent", 9: "bent", 11: "bent", 12: "bent", 13: "good", 14: "bent",
    16: "loose", 18: "bent", 19: "bent", 21: "pierced", 22: "missing",
    24: "bent", 25: "bent", 26: "good", 27: "bent", 29: "bent", 30: "bent",
    31: "bent", 32: "bent", 33: "pierced", 34: "bent", 35: "pierced",
    40: "loose", 41: "pierced",
}


def metrics(rows, key):
    labels = np.array([r["label"] for r in rows])
    predicted = np.array([CLASSES.index(r[key]) for r in rows])
    return {
        "n": len(rows), "correct": int((labels == predicted).sum()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=list(range(5)), average="macro", zero_division=0)),
        "fn": int(((labels != 0) & (predicted == 0)).sum()),
        "fp": int(((labels == 0) & (predicted != 0)).sum()),
        "confusion": confusion_matrix(labels, predicted, labels=list(range(5))).tolist(),
    }


def make_cases(protocol, output, diagnostic_csv=None, include_baseline=True):
    metadata, frames = load_protocol(protocol)
    cases = [dict(subset="clean_test", sample_id=str(r.sample_id), image_path=str(r.image_path),
                  label=int(r.label), expected_sha256=str(r.sha256)) for r in frames["test"].itertuples()] if include_baseline else []
    legacy_sources = pd.read_csv(ROOT / "data/straw_defects_splits/test.csv")
    last_source = {Path(r.image_path).stem: Path(r.image_path) for r in legacy_sources.itertuples()}
    corrected = output / "upright_legacy30"
    corrected.mkdir(parents=True, exist_ok=True)
    rotation_rows = []
    for number, reported in (USER_PREDICTIONS.items() if include_baseline else []):
        path = ROOT / "data/test_degraded/images" / f"degraded_Bent ({number}).jpg"
        source = last_source[f"Bent ({number})"]
        with Image.open(source) as original:
            orientation = int(original.getexif().get(274, 1))
        if orientation != 6:
            raise ValueError(f"Cannot infer the legacy correction for EXIF={orientation}: {source}")
        with Image.open(path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError("Legacy JPEG is no longer the metadata-stripped input")
            upright = image.convert("RGB").transpose(Image.Transpose.ROTATE_270)
        target = corrected / f"upright_Bent_{number:02d}.png"
        if target.exists():
            with Image.open(target) as previous:
                if not np.array_equal(np.asarray(previous), np.asarray(upright)):
                    raise ValueError(f"Corrected diagnostic pixels changed: {target}")
        else:
            upright.save(target)
        common = dict(sample_id=f"legacy_Bent_{number:02d}", label=1, reported_prediction=reported)
        cases.append(dict(common, subset="legacy_uploaded30", image_path=str(path), expected_sha256=file_hash(path)))
        cases.append(dict(common, subset="legacy_rotated30", image_path=str(target), expected_sha256=file_hash(target)))
        rotation_rows.append(dict(source_image=str(source), original_exif=orientation,
            legacy_path=str(path), legacy_sha256=file_hash(path), corrected_path=str(target),
            corrected_sha256=file_hash(target), clockwise_rotation_degrees=90))
    (output / "rotation_manifest.json").write_text(json.dumps(rotation_rows, indent=2), encoding="utf-8")
    if diagnostic_csv:
        frame = pd.read_csv(diagnostic_csv)
        recipe = json.loads(diagnostic_csv.with_name("generation.json").read_text(encoding="utf-8"))
        if recipe["source_manifest_sha256"] != metadata["manifests"]["test.csv"]:
            raise ValueError("Diagnostic set is not derived from the same frozen test role")
        if set(frame.sample_id) != set(frames["test"].sample_id) or not frame.exif_transpose_applied.all():
            raise ValueError("Incomplete or unoriented diagnostic set")
        originals = frames["test"].set_index("sample_id")
        for row in frame.itertuples():
            original = originals.loc[row.sample_id]
            if (int(row.label) != int(original.label) or row.class_name != original.class_name
                    or row.source_sha256 != original.sha256
                    or str(row.sha256) != recipe["outputs"][str(row.sample_id)]):
                raise ValueError(f"Diagnostic label/source/output identity changed: {row.sample_id}")
        cases += [dict(subset="b1_synthetic_degraded_test", sample_id=str(r.sample_id),
            image_path=str(r.image_path), label=int(r.label), expected_sha256=str(r.sha256)) for r in frame.itertuples()]
    return metadata, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifacts/straw_global_local/data_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-csv", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-diagnostic", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("Use a new audit directory to preserve previous evidence")
    protocol, cases = make_cases(args.protocol, args.output_dir, args.diagnostic_csv, include_baseline=not args.only_diagnostic)
    if args.only_diagnostic:
        if args.diagnostic_csv is None:
            raise ValueError("--only-diagnostic requires --diagnostic-csv")
        cases = [case for case in cases if case["subset"] == "b1_synthetic_degraded_test"]
    resident = ResidentInference(ROOT, "", "", artifact="b1_stream_512")
    state = resident.status()
    if state["state"] != "ready":
        raise RuntimeError(f"The verified 512 worker is not ready: {state}")
    config512 = json.loads((ROOT / "configs/straw_realtime_stream512.json").read_text())
    config384 = json.loads((ROOT / "configs/straw_realtime.json").read_text())
    for config in (config512, config384):
        if file_hash(ROOT / config["checkpoint"]) != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint file differs from its pinned SHA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = TensorB1(ROOT / config512["checkpoint"]).eval().cuda()
    legacy_model = TensorB1(ROOT / config384["checkpoint"]).eval().cuda()
    rows = []
    trace_path = args.output_dir / "predictions.jsonl"
    if trace_path.exists():
        if not args.resume:
            raise FileExistsError("Partial audit exists; use --resume to validate and continue")
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    inventory = {(case["subset"], case["sample_id"]): case for case in cases}
    completed = set()
    for row in rows:
        key = row["subset"], row["sample_id"]
        case = inventory[key]
        if key in completed or row["contract_sha256"] != resident.identity["contract_sha256"]:
            raise ValueError("Partial audit is duplicated or belongs to another engine")
        if row["expected_sha256"] != case["expected_sha256"] or file_hash(case["image_path"]) != case["expected_sha256"]:
            raise ValueError("Partial audit input bytes changed")
        completed.add(key)
    print(f"Resuming {len(rows)}/{len(cases)} verified rows", flush=True)
    with torch.inference_mode(), trace_path.open("a", encoding="utf-8") as trace:
        for index, case in enumerate(cases, 1):
            if (case["subset"], case["sample_id"]) in completed:
                continue
            path = Path(case["image_path"])
            if file_hash(path) != case["expected_sha256"]:
                raise ValueError(f"Input bytes changed: {path}")
            raw = raw_rgb_tensor(path, 512)
            x = normalize(raw.transpose(1, 2, 0).astype(np.uint8))
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                training = tensor_views(render_arrays(image, "B1"), "B1")["global"][None].numpy()
            preprocess_error = float(np.max(np.abs(x - training)))
            if preprocess_error > 3e-7:
                raise ValueError(f"Serving preprocess differs from training: {preprocess_error}")
            expected = model(torch.from_numpy(x).cuda()).softmax(1).cpu().numpy()[0]
            request_id = resident.submit(path, f"audit-{uuid.uuid4().hex}",
                {"subset": case["subset"], "sample_id": case["sample_id"]})
            deadline = time.monotonic() + 30
            response = resident.result(request_id)
            while response.get("status") == "pending" and time.monotonic() < deadline:
                time.sleep(.02)
                response = resident.result(request_id)
            if response.get("status") != "completed":
                raise RuntimeError(f"TensorRT request failed: {response}")
            input_sha = hashlib.sha256(raw.tobytes()).hexdigest()
            if response["input_sha256"] != input_sha:
                raise ValueError("TensorRT received pixels different from the PyTorch reference")
            actual = np.array([response["probabilities"][c] for c in CLASSES])
            row = dict(case, pytorch=CLASSES[int(expected.argmax())], tensorrt=response["class"],
                pytorch_probabilities=expected.tolist(), tensorrt_probabilities=actual.tolist(),
                probability_max_abs_error=float(np.max(np.abs(expected - actual))),
                preprocessing_max_abs_error=preprocess_error, input_sha256=input_sha,
                contract_sha256=response["contract_sha256"], generation=response["generation"],
                tensor_ms=response["tensor_transfer_inference_ms"])
            if case["subset"].startswith("legacy_"):
                small = raw_rgb_tensor(path, 384)
                small = normalize(small.transpose(1, 2, 0).astype(np.uint8))
                p384 = legacy_model(torch.from_numpy(small).cuda()).softmax(1).cpu().numpy()[0]
                row.update(pytorch384=CLASSES[int(p384.argmax())], pytorch384_probabilities=p384.tolist())
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            if index % 25 == 0 or index == len(cases):
                print(f"Audited {index}/{len(cases)} inputs", flush=True)
    subsets = defaultdict(list)
    for row in rows:
        subsets[row["subset"]].append(row)
    summaries = {}
    for name, values in subsets.items():
        summaries[name] = {
            "pytorch512": metrics(values, "pytorch"), "tensorrt512": metrics(values, "tensorrt"),
            "changed_predictions": sum(r["pytorch"] != r["tensorrt"] for r in values),
            "probability_max_abs_error": max(r["probability_max_abs_error"] for r in values),
        }
        if name.startswith("legacy_"):
            summaries[name]["pytorch384"] = metrics(values, "pytorch384")
            if name == "legacy_uploaded30":
                summaries[name]["pytorch384_matches_user_table"] = sum(r["pytorch384"] == r["reported_prediction"] for r in values)
    summary = {
        "schema": 1, "purpose": "pipeline diagnosis and deployment parity; no test tuning",
        "identity": resident.identity, "protocol": protocol,
        "reference_checkpoint_sha256": file_hash(ROOT / config512["checkpoint"]),
        "legacy_checkpoint_sha256": file_hash(ROOT / config384["checkpoint"]),
        "legacy30_caveat": "Single-class user batch; legacy source filenames collided across brands. Rotation is a diagnostic, not new held-out evidence.",
        "subsets": summaries, "n": len(rows),
        "changed_predictions": sum(r["pytorch"] != r["tensorrt"] for r in rows),
        "probability_max_abs_error": max(r["probability_max_abs_error"] for r in rows),
        "preprocessing_max_abs_error": max(r["preprocessing_max_abs_error"] for r in rows),
    }
    clean = summaries.get("clean_test", {}).get("pytorch512")
    summary["clean_test_reproduced"] = (clean["n"] == 299 and clean["correct"] == 297 and clean["fn"] == 1 and clean["fp"] == 0) if clean else None
    summary["parity_pass"] = summary["changed_predictions"] == 0 and summary["probability_max_abs_error"] < .01
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "changed_predictions", "probability_max_abs_error", "clean_test_reproduced", "parity_pass", "subsets")}, indent=2))
    if summary["clean_test_reproduced"] is False or not summary["parity_pass"]:
        raise RuntimeError("Audit did not reproduce the clean test or deployment parity; inspect the evidence")


if __name__ == "__main__":
    main()
