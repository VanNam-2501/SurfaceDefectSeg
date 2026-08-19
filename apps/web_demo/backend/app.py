"""Local API for real inference with the three trained segmentation models."""
from __future__ import annotations

import base64
import asyncio
import importlib.util
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(
    os.environ.get(
        "SEGMENTATION_PROJECT_ROOT",
        REPO_ROOT / "src" / "threecad_segmentation",
    )
).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fullres_eval import predict_full_image  # noqa: E402
from model_factory import build_model  # noqa: E402
from decision_policy import (  # noqa: E402
    apply_decision_policy,
    border_connected_dark_roi,
    load_decision_policy,
)
from learned_decision_verifier import image_features, probability_features  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS = {
    "unet": Path(os.environ.get("UNET_CHECKPOINT", REPO_ROOT / "artifacts" / "checkpoints" / "final" / "unet_best.pt")),
    "segformer": Path(os.environ.get("SEGFORMER_CHECKPOINT", REPO_ROOT / "artifacts" / "checkpoints" / "final" / "segformer_best.pt")),
    "vmamba": Path(os.environ.get("VMAMBA_CHECKPOINT", REPO_ROOT / "artifacts" / "checkpoints" / "final" / "vmamba_best.pt")),
}
POLICY_PATH = Path(
    os.environ.get(
        "DECISION_POLICY",
        REPO_ROOT / "artifacts" / "reports" / "final" / "decision_and_test_audit" / "spatial" / "unet_vmamba" / "policy" / "decision_policy.json",
    )
).resolve()
LEARNED_POLICY_PATH = Path(
    os.environ.get(
        "LEARNED_VERIFIER_POLICY",
        REPO_ROOT / "artifacts" / "reports" / "final" / "decision_and_test_audit" / "hybrid_pairs" / "unet_vmamba" / "learned_policy.json",
    )
).resolve()
LEARNED_MODELS_DIR = Path(
    os.environ.get(
        "LEARNED_VERIFIER_MODELS",
        REPO_ROOT / "artifacts" / "reports" / "final" / "decision_and_test_audit" / "hybrid_pairs" / "unet_vmamba" / "models",
    )
).resolve()
LOADED: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
POLICY_CACHE: tuple[int, dict[str, Any]] | None = None
LEARNED_POLICY_CACHE: tuple[int, dict[str, Any]] | None = None
VERIFIER_CACHE: dict[str, dict[str, Any]] = {}
INFERENCE_LOCK = asyncio.Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

app = FastAPI(title="Aluminum Surface Lab Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def current_policy() -> dict[str, Any] | None:
    global POLICY_CACHE
    if not POLICY_PATH.is_file():
        POLICY_CACHE = None
        return None
    modified = POLICY_PATH.stat().st_mtime_ns
    if POLICY_CACHE is None or POLICY_CACHE[0] != modified:
        POLICY_CACHE = (modified, load_decision_policy(POLICY_PATH))
    return POLICY_CACHE[1]


def current_learned_policy() -> dict[str, Any] | None:
    global LEARNED_POLICY_CACHE
    if not LEARNED_POLICY_PATH.is_file():
        LEARNED_POLICY_CACHE = None
        return None
    modified = LEARNED_POLICY_PATH.stat().st_mtime_ns
    if LEARNED_POLICY_CACHE is None or LEARNED_POLICY_CACHE[0] != modified:
        policy = json.loads(LEARNED_POLICY_PATH.read_text(encoding="utf-8"))
        if policy.get("type") != "cross_validated_learned_verifier":
            raise ValueError(f"Unsupported learned verifier: {LEARNED_POLICY_PATH}")
        if "hybrid_fusion" not in policy.get("branches", {}):
            raise ValueError("Learned verifier has no hybrid_fusion branch")
        LEARNED_POLICY_CACHE = (modified, policy)
    return LEARNED_POLICY_CACHE[1]


def load_verifier_bundle(name: str) -> dict[str, Any]:
    if name not in VERIFIER_CACHE:
        path = LEARNED_MODELS_DIR / f"{name}_folds.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Verifier bundle missing: {path}")
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("branch") != name or not bundle.get("estimators"):
            raise ValueError(f"Invalid verifier bundle: {path}")
        VERIFIER_CACHE[name] = bundle
    return VERIFIER_CACHE[name]


def learned_verifier_status() -> dict[str, Any]:
    try:
        policy = current_learned_policy()
        if policy is None:
            return {"available": False, "ready": False, "reason": "Policy missing."}
        hybrid = policy["branches"]["hybrid_fusion"]
        missing = []
        for name, settings in hybrid["rescue"].items():
            if settings["score_strategy"] == "learned_hgb":
                path = LEARNED_MODELS_DIR / f"{name}_folds.pkl"
                if not path.is_file():
                    missing.append(str(path))
        return {
            "available": True,
            "ready": not missing,
            "reason": "" if not missing else "Missing verifier bundle(s): " + ", ".join(missing),
            "path": str(LEARNED_POLICY_PATH),
            "models_dir": str(LEARNED_MODELS_DIR),
            "type": "hybrid_fully_automatic",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"available": True, "ready": False, "reason": str(exc)}


def verifier_features(
    source: np.ndarray,
    probability: np.ndarray,
    model: str,
    feature_size: int,
    roi_threshold: int,
) -> dict[str, float]:
    original_height, original_width = source.shape[:2]
    scale = min(1.0, float(feature_size) / max(original_height, original_width))
    target_width = max(1, int(round(original_width * scale)))
    target_height = max(1, int(round(original_height * scale)))
    resized = source
    if (target_height, target_width) != (original_height, original_width):
        resized = cv2.resize(
            source, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
    resized_probability = probability
    if probability.shape != (target_height, target_width):
        resized_probability = cv2.resize(
            probability,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    roi = border_connected_dark_roi(resized, threshold=roi_threshold)
    gray, gradient, features = image_features(resized, roi)
    features.update(
        probability_features(
            model, resized_probability, gray, gradient, roi
        )
    )
    return features


def verifier_score(
    source: np.ndarray,
    probability: np.ndarray,
    model: str,
    strategy: str,
    learned_policy: dict[str, Any],
    roi_threshold: int,
) -> float:
    features = verifier_features(
        source,
        probability,
        model,
        int(learned_policy["calibration"].get("feature_size", 256)),
        roi_threshold,
    )
    if strategy != "learned_hgb":
        if strategy not in features:
            raise ValueError(f"Verifier feature is unavailable: {strategy}")
        return float(features[strategy])
    bundle = load_verifier_bundle(model)
    columns = bundle["feature_columns"]
    try:
        vector = np.asarray([[features[column] for column in columns]], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"Verifier feature is unavailable: {exc.args[0]}") from exc
    return float(
        np.mean(
            [
                estimator.predict_proba(vector)[0, 1]
                for estimator in bundle["estimators"]
            ]
        )
    )


def apply_learned_hybrid(
    source: np.ndarray,
    probabilities: dict[str, np.ndarray],
    spatial: dict[str, Any],
    spatial_policy: dict[str, Any],
    learned_policy: dict[str, Any],
) -> dict[str, Any]:
    hybrid = learned_policy["branches"]["hybrid_fusion"]
    scores: dict[str, float] = {}
    rescued: list[str] = []
    if spatial["decision"] != "defect":
        for name, settings in hybrid["rescue"].items():
            threshold = float(settings["threshold"])
            if threshold > 1.0:
                scores[name] = 0.0
                continue
            score = verifier_score(
                source,
                probabilities[name],
                name,
                str(settings["score_strategy"]),
                learned_policy,
                int(spatial_policy.get("roi", {}).get("border_dark_threshold", 5)),
            )
            scores[name] = score
            if score >= threshold:
                rescued.append(name)

    if spatial["decision"] == "defect":
        mask = np.asarray(spatial["mask"], dtype=bool)
        level = "defect"
        reason = "U-Net and SegFormer agree spatially"
    elif rescued:
        masks = [
            np.asarray(spatial["analyses"][name]["candidate_mask"], dtype=bool)
            for name in rescued
        ]
        mask = np.logical_or.reduce(masks)
        if not mask.any():
            mask = np.logical_or.reduce(
                [probabilities[name] >= 0.5 for name in rescued]
            )
        level = "defect"
        reason = "specialist verifier confirmed: " + ", ".join(rescued)
    else:
        mask = np.zeros(source.shape[:2], dtype=bool)
        level = "pass"
        reason = "no spatial consensus and no specialist verifier confirmation"
    return {
        "level": level,
        "reason": reason,
        "mask": mask,
        "scores": scores,
        "rescued_models": rescued,
    }

def to_data_url(array: np.ndarray) -> str:
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

def runtime_issue(name: str) -> str:
    if name == "segformer" and importlib.util.find_spec("transformers") is None:
        return "SegFormer runtime is not installed."
    if name == "vmamba":
        candidates = []
        if os.environ.get("VMAMBA_REPO"):
            candidates.append(Path(os.environ["VMAMBA_REPO"]).expanduser())
        candidates.extend([ROOT / "third_party" / "VMamba", Path("/content/TTTN/third_party/VMamba")])
        if not any((candidate / "vmamba.py").is_file() for candidate in candidates):
            return "VMamba runtime is not installed."
        if not torch.cuda.is_available():
            return "VMamba requires the configured CUDA runtime."
    return ""


def status() -> dict[str, dict[str, Any]]:
    policy = current_policy()
    policy_models = set(policy["models"]) if policy else set(CHECKPOINTS)
    result = {}
    for name, path in CHECKPOINTS.items():
        issue = "" if path.is_file() else "Checkpoint missing."
        if not issue:
            issue = runtime_issue(name)
        result[name] = {
            "available": not issue,
            "checkpoint": str(path),
            "reason": issue,
            "policy_compatible": name in policy_models,
        }
    return result


def load_model(name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name in LOADED:
        return LOADED[name]
    model_status = status()[name]
    if not model_status["available"]:
        raise HTTPException(422, f"{name} is unavailable: {model_status['reason']}")
    checkpoint = CHECKPOINTS[name]
    state = torch.load(checkpoint, map_location="cpu")
    if "model" not in state:
        raise HTTPException(422, f"Invalid checkpoint for {name}: expected a 'model' state dictionary.")
    model = build_model(name, pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval().to(DEVICE)
    config = state.get("config", {})
    LOADED[name] = (model, config)
    return LOADED[name]


@app.get("/health")
def health() -> dict[str, Any]:
    policy = current_policy()
    learned = learned_verifier_status()
    model_status = status()
    policy_ready = bool(
        policy
        and all(model_status.get(name, {}).get("available", False) for name in policy["models"])
    )
    return {
        "device": str(DEVICE),
        "project_root": str(ROOT),
        "models": model_status,
        "policy": {
            "available": policy is not None,
            "ready": policy_ready,
            "path": str(POLICY_PATH),
            "models": list(policy["models"]) if policy else [],
            "targets": policy.get("targets", {}) if policy else {},
            "mode": "hybrid_fully_automatic" if learned.get("ready") else "spatial_triage",
            "learned_verifier": learned,
        },
    }


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    models: str = Form("unet,segformer,vmamba"),
    threshold: float | None = Form(None),
) -> dict[str, Any]:
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise HTTPException(422, "Threshold must be between 0 and 1.")
    selected = [item.strip().lower() for item in models.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(CHECKPOINTS))
    if invalid:
        raise HTTPException(422, f"Unsupported model(s): {', '.join(invalid)}")
    if not selected:
        raise HTTPException(422, "Select at least one model.")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than the 20 MB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as uploaded:
            width, height = uploaded.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds the 25-megapixel safety limit.")
            source = np.asarray(ImageOps.exif_transpose(uploaded).convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(422, "Upload a valid PNG or JPEG image.") from exc

    policy = current_policy()
    if policy:
        expected_models = set(policy["models"])
        if set(selected) != expected_models:
            raise HTTPException(
                422,
                "Frozen policy requires exactly these models: "
                + ", ".join(policy["models"]),
            )

    results = []
    probabilities: dict[str, np.ndarray] = {}
    elapsed_by_model: dict[str, float] = {}
    # One process normally owns one GPU. Serialize requests to avoid concurrent
    # model loads/forwards exhausting its memory.
    async with INFERENCE_LOCK:
        for name in selected:
            model, config = load_model(name)
            started = time.perf_counter()
            probability, _ = predict_full_image(
                model=model,
                image=source,
                device=DEVICE,
                data_mode=config.get("data_mode", "patch"),
                tile_size=int(config.get("patch_size", 512)),
                stride=int(config.get("stride", 256)),
                tile_batch_size=int(config.get("val_tile_batch_size", 4)),
                amp=bool(config.get("amp", True)),
            )
            probabilities[name] = probability
            elapsed_by_model[name] = time.perf_counter() - started

    if policy:
        combined = apply_decision_policy(source, probabilities, policy)
        learned_policy = current_learned_policy()
        learned_status = learned_verifier_status()
        hybrid = (
            apply_learned_hybrid(source, probabilities, combined, policy, learned_policy)
            if learned_policy is not None and learned_status.get("ready")
            else None
        )
        for name in selected:
            analysis = combined["analyses"][name]
            mask = np.asarray(analysis["candidate_mask"], dtype=bool)
            overlay = source.copy()
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.asarray([229, 109, 57])).astype(np.uint8)
            results.append({
                "model": name,
                "threshold": float(analysis["threshold"]),
                "min_component_area_px": int(analysis["min_component_area_px"]),
                "state": (
                    "strong" if analysis["has_strong"]
                    else "review" if analysis["has_candidate"]
                    else "pass"
                ),
                "strong_component_count": int(analysis["strong_component_count"]),
                "review_component_count": int(analysis["review_component_count"]),
                "elapsed_seconds": elapsed_by_model[name],
                "mask_png": to_data_url((mask.astype(np.uint8) * 255)),
                "overlay_png": to_data_url(overlay),
            })
        decision_level = hybrid["level"] if hybrid else combined["decision"]
        decision_reason = hybrid["reason"] if hybrid else combined["reason"]
        decision_mask = np.asarray(hybrid["mask"] if hybrid else combined["mask"], dtype=bool)
        color = {
            "pass": np.asarray([67, 145, 104]),
            "review": np.asarray([232, 168, 55]),
            "defect": np.asarray([213, 68, 62]),
        }[decision_level]
        decision_overlay = source.copy()
        decision_overlay[decision_mask] = (
            0.35 * decision_overlay[decision_mask] + 0.65 * color
        ).astype(np.uint8)
        decision = {
            "level": decision_level,
            "reason": decision_reason,
            "required_votes": int(combined["required_votes"]),
            "max_spatial_votes": int(combined["max_spatial_votes"]),
            "strong_models": combined["strong_models"],
            "candidate_models": combined["candidate_models"],
            "mask_pixels": int(combined["mask_pixels"]),
            "mask_png": to_data_url(decision_mask.astype(np.uint8) * 255),
            "overlay_png": to_data_url(decision_overlay),
            "verifier_scores": hybrid["scores"] if hybrid else {},
            "rescued_models": hybrid["rescued_models"] if hybrid else [],
        }
        mode = "learned_hybrid_policy" if hybrid else "calibrated_policy"
    else:
        cutoff = 0.5 if threshold is None else float(threshold)
        legacy_masks = []
        for name in selected:
            mask = probabilities[name] >= cutoff
            legacy_masks.append(mask)
            overlay = source.copy()
            overlay[mask] = (
                0.45 * overlay[mask] + 0.55 * np.asarray([229, 109, 57])
            ).astype(np.uint8)
            results.append({
                "model": name,
                "threshold": cutoff,
                "min_component_area_px": 1,
                "state": "strong" if mask.any() else "pass",
                "strong_component_count": int(mask.any()),
                "review_component_count": 0,
                "elapsed_seconds": elapsed_by_model[name],
                "mask_png": to_data_url(mask.astype(np.uint8) * 255),
                "overlay_png": to_data_url(overlay),
            })
        decision_mask = np.logical_or.reduce(legacy_masks)
        decision_level = "defect" if decision_mask.any() else "pass"
        decision = {
            "level": decision_level,
            "reason": "legacy threshold fallback; calibrate decision_policy.json",
            "required_votes": 1,
            "max_spatial_votes": int(decision_mask.any()),
            "strong_models": [
                name for name in selected if bool((probabilities[name] >= cutoff).any())
            ],
            "candidate_models": [
                name for name in selected if bool((probabilities[name] >= cutoff).any())
            ],
            "mask_pixels": int(decision_mask.sum()),
            "mask_png": to_data_url(decision_mask.astype(np.uint8) * 255),
            "overlay_png": to_data_url(source),
        }
        mode = "legacy_threshold"
    return {
        "mode": mode,
        "policy_path": str(POLICY_PATH) if policy else "",
        "decision": decision,
        "results": results,
    }
