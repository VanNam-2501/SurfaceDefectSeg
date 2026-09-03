"""Local API for real inference with the three trained segmentation models."""
from __future__ import annotations

import base64
import asyncio
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
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
from adaptive_component_policy import (  # noqa: E402
    PolicyConfig,
    accepted_components,
    components_for_threshold,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS = {
    "unet": Path(os.environ.get("UNET_CHECKPOINT", REPO_ROOT / "artifacts" / "checkpoints" / "final" / "unet_best.pt")),
    "segformer": Path(os.environ.get("SEGFORMER_CHECKPOINT", REPO_ROOT / "artifacts" / "checkpoints" / "final" / "segformer_best.pt")),
    "vmamba": Path(os.environ.get("VMAMBA_CHECKPOINT", REPO_ROOT / "artifacts" / "checkpoints" / "final" / "vmamba_best.pt")),
}
MODEL_ORDER = ("unet", "segformer", "vmamba")
SUPPORTED_SELECTIONS = (
    ("unet",),
    ("segformer",),
    ("vmamba",),
    ("unet", "segformer"),
    ("unet", "vmamba"),
    ("segformer", "vmamba"),
)
DEFAULT_SELECTION = ("unet", "vmamba")
RAW_THRESHOLDS = {
    "unet": 0.49,
    "segformer": 0.66,
    "vmamba": 0.51,
}
ADAPTIVE_POLICY_PATH = Path(
    os.environ.get(
        "ADAPTIVE_POLICY",
        REPO_ROOT / "artifacts" / "reports" / "final" / "decision_and_test_audit" / "adaptive_single" / "adaptive_component_policy.json",
    )
).resolve()
SPATIAL_POLICY_ROOT = Path(
    os.environ.get(
        "DECISION_POLICY_ROOT",
        REPO_ROOT / "artifacts" / "reports" / "final" / "decision_and_test_audit" / "spatial",
    )
).resolve()
POLICY_OVERRIDE_PATH = (
    Path(os.environ["DECISION_POLICY"]).resolve()
    if os.environ.get("DECISION_POLICY")
    else None
)
LOADED: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
POLICY_CACHE: dict[Path, tuple[int, dict[str, Any]]] = {}
RUNTIME_ISSUE_CACHE: dict[str, str] = {}
INFERENCE_LOCK = asyncio.Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ADAPTIVE_POLICY_TYPE = "adaptive_component_evidence"

app = FastAPI(title="Aluminum Surface Lab Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def selection_id(selected: tuple[str, ...]) -> str:
    return "_".join(selected)


def policy_path_for(selected: tuple[str, ...]) -> Path:
    if len(selected) == 1:
        return ADAPTIVE_POLICY_PATH
    generated = SPATIAL_POLICY_ROOT / selection_id(selected) / "policy" / "decision_policy.json"
    if POLICY_OVERRIDE_PATH and POLICY_OVERRIDE_PATH.is_file():
        override = load_decision_policy(POLICY_OVERRIDE_PATH)
        if set(override["models"]) == set(selected):
            return POLICY_OVERRIDE_PATH
    return generated


def is_adaptive_policy(policy: Mapping[str, Any]) -> bool:
    return policy.get("type") == ADAPTIVE_POLICY_TYPE


def policy_matches_selection(policy: Mapping[str, Any], selected: tuple[str, ...]) -> bool:
    policy_models = set(policy.get("models", {}))
    selected_models = set(selected)
    return (
        selected_models.issubset(policy_models)
        if is_adaptive_policy(policy)
        else selected_models == policy_models
    )


def current_policy(selected: tuple[str, ...]) -> dict[str, Any] | None:
    path = policy_path_for(selected)
    if not path.is_file():
        POLICY_CACHE.pop(path, None)
        return None
    modified = path.stat().st_mtime_ns
    cached = POLICY_CACHE.get(path)
    if cached is None or cached[0] != modified:
        if len(selected) == 1:
            policy = json.loads(path.read_text(encoding="utf-8"))
            if not is_adaptive_policy(policy):
                raise ValueError(f"Unsupported adaptive policy: {path}")
        else:
            policy = load_decision_policy(path)
        cached = (modified, policy)
        POLICY_CACHE[path] = cached
    return cached[1]



def to_data_url(array: np.ndarray) -> str:
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def apply_adaptive_policy(
    image: np.ndarray,
    probability: np.ndarray,
    policy: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Apply the frozen component rules used by the Adaptive single-model report."""
    settings = policy["models"][model]["config"]
    config = PolicyConfig(
        low_threshold=float(settings["low_threshold"]),
        min_area=int(settings["min_area_px"]),
        peak_threshold=float(settings["peak_threshold"]),
        min_persistent_area=int(settings["min_persistent_area_px"]),
        min_local_contrast=float(settings["min_local_contrast"]),
    )
    roi = border_connected_dark_roi(
        image, threshold=int(policy.get("roi_border_dark_threshold", 5))
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    frame = components_for_threshold(
        np.asarray(probability, dtype=np.float32), gray, roi, config.low_threshold
    )
    accepted = accepted_components(frame, config)
    binary = ((probability >= config.low_threshold) & roi).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.isin(labels, np.flatnonzero(accepted) + 1)
    analysis = {
        "threshold": config.low_threshold,
        "min_component_area_px": config.min_area,
        "tiny_high_threshold": config.peak_threshold,
        "strong_mask": mask,
        "review_mask": np.zeros(mask.shape, dtype=bool),
        "candidate_mask": mask,
        "has_strong": bool(mask.any()),
        "has_candidate": bool(mask.any()),
        "strong_component_count": int(accepted.sum()),
        "review_component_count": 0,
        "candidate_pixels": int(mask.sum()),
        "components": [],
    }
    return {
        "decision": "defect" if mask.any() else "pass",
        "reason": "Adaptive component rules accepted one or more components" if mask.any() else "no component passed the Adaptive component rules",
        "mask": mask,
        "consensus_mask": mask,
        "models_used": [model],
        "required_votes": 1,
        "max_spatial_votes": int(mask.any()),
        "strong_models": [model] if mask.any() else [],
        "candidate_models": [model] if mask.any() else [],
        "mask_pixels": int(mask.sum()),
        "analyses": {model: analysis},
        "roi": roi,
    }

def runtime_issue(name: str) -> str:
    if name in RUNTIME_ISSUE_CACHE:
        return RUNTIME_ISSUE_CACHE[name]
    issue = ""
    if name == "segformer" and importlib.util.find_spec("transformers") is None:
        issue = "SegFormer runtime is not installed."
    elif name == "vmamba":
        candidates = []
        if os.environ.get("VMAMBA_REPO"):
            candidates.append(Path(os.environ["VMAMBA_REPO"]).expanduser())
        candidates.extend([ROOT / "third_party" / "VMamba", Path("/content/TTTN/third_party/VMamba")])
        repository = next((candidate for candidate in candidates if (candidate / "vmamba.py").is_file()), None)
        if repository is None:
            issue = "VMamba runtime is not installed."
        elif not torch.cuda.is_available():
            issue = "VMamba requires the configured CUDA runtime."
        else:
            try:
                importlib.import_module("selective_scan_cuda")
                if str(repository) not in sys.path:
                    sys.path.insert(0, str(repository))
                vmamba_module = importlib.import_module("vmamba")
                if not getattr(vmamba_module, "WITH_SELECTIVESCAN_MAMBA", False):
                    issue = "VMamba selective-scan CUDA extension is not active."
            except Exception as exc:  # pragma: no cover - depends on optional GPU runtime
                issue = f"VMamba runtime check failed: {exc.__class__.__name__}."
    RUNTIME_ISSUE_CACHE[name] = issue
    return issue


def status() -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in CHECKPOINTS.items():
        issue = "" if path.is_file() else "Checkpoint missing."
        if not issue:
            issue = runtime_issue(name)
        result[name] = {
            "available": not issue,
            "checkpoint": str(path),
            "reason": issue,
            "policy_compatible": any(name in selection for selection in SUPPORTED_SELECTIONS),
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
    model_status = status()
    modes = []
    for name, cutoff in RAW_THRESHOLDS.items():
        ready = model_status[name]["available"]
        modes.append({
            "id": f"raw_{name}",
            "models": [name],
            "available": True,
            "models_available": ready,
            "ready": ready,
            "reason": "" if ready else model_status[name]["reason"],
            "path": "",
            "targets": {"threshold": cutoff},
            "threshold": cutoff,
            "mode": "raw_segmentation",
        })
    for selected in SUPPORTED_SELECTIONS:
        policy = current_policy(selected)
        models_available = all(model_status[name]["available"] for name in selected)
        compatible = bool(policy and policy_matches_selection(policy, selected))
        ready = compatible and models_available
        if not policy:
            reason = "Frozen validation policy is missing."
        elif not compatible:
            reason = "Policy model list does not match this mode."
        elif not models_available:
            unavailable = [name for name in selected if not model_status[name]["available"]]
            reason = "Unavailable model(s): " + ", ".join(unavailable)
        else:
            reason = ""
        targets = policy.get("targets", {}) if policy else {}
        if policy and is_adaptive_policy(policy):
            targets = {"max_alert_fnr": policy.get("validation_target_max_fnr")}
        modes.append({
            "id": selection_id(selected),
            "models": list(selected),
            "available": policy is not None,
            "models_available": models_available,
            "ready": ready,
            "reason": reason,
            "path": str(policy_path_for(selected)),
            "targets": targets,
            "mode": "adaptive_single_model" if len(selected) == 1 else "spatial_pair_ensemble",
        })
    default_mode = next(
        (
            mode
            for mode in modes
            if mode["id"] == selection_id(DEFAULT_SELECTION) and mode["ready"]
        ),
        next((mode for mode in modes if mode["ready"]), modes[0]),
    )
    return {
        "device": str(DEVICE),
        "project_root": str(ROOT),
        "models": model_status,
        "default_mode": default_mode["id"],
        "modes": modes,
        "policy": default_mode,
    }


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    models: str = Form("unet,segformer,vmamba"),
    decision_mode: str = Form("calibrated"),
    threshold: float | None = Form(None),
) -> dict[str, Any]:
    decision_mode = decision_mode.strip().lower()
    if decision_mode not in {"calibrated", "raw"}:
        raise HTTPException(422, "decision_mode must be calibrated or raw.")
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise HTTPException(422, "Threshold must be between 0 and 1.")
    selected = [item.strip().lower() for item in models.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(CHECKPOINTS))
    if invalid:
        raise HTTPException(422, f"Unsupported model(s): {', '.join(invalid)}")
    if not selected:
        raise HTTPException(422, "Select at least one model.")
    if len(selected) != len(set(selected)):
        raise HTTPException(422, "Each model may be selected only once.")
    normalized = tuple(name for name in MODEL_ORDER if name in selected)
    if normalized not in SUPPORTED_SELECTIONS:
        raise HTTPException(422, "Select one model or one supported two-model ensemble.")
    selected = list(normalized)
    if decision_mode == "raw" and len(selected) != 1:
        raise HTTPException(422, "Raw baseline supports one model at a time.")
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

    policy = None if decision_mode == "raw" else current_policy(normalized)
    if decision_mode == "calibrated":
        if not policy:
            raise HTTPException(
                422,
                f"Frozen policy is missing for mode: {selection_id(normalized)}",
            )
        if not policy_matches_selection(policy, normalized):
            raise HTTPException(
                422,
                "Frozen policy does not match the selected inference mode.",
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
        if is_adaptive_policy(policy):
            combined = apply_adaptive_policy(source, probabilities[selected[0]], policy, selected[0])
        else:
            combined = apply_decision_policy(source, probabilities, policy)
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
        decision_level = combined["decision"]
        decision_reason = combined["reason"]
        decision_mask = np.asarray(combined["mask"], dtype=bool)
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
        }
        mode = (
            "adaptive_single_model"
            if is_adaptive_policy(policy)
            else "spatial_pair_ensemble"
        )
    else:
        name = selected[0]
        cutoff = RAW_THRESHOLDS[name] if threshold is None else float(threshold)
        mask = probabilities[name] >= cutoff
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
        decision_level = "defect" if mask.any() else "pass"
        decision_color = (
            np.asarray([213, 68, 62])
            if decision_level == "defect"
            else np.asarray([67, 145, 104])
        )
        decision_overlay = source.copy()
        decision_overlay[mask] = (
            0.35 * decision_overlay[mask] + 0.65 * decision_color
        ).astype(np.uint8)
        decision = {
            "level": decision_level,
            "reason": (
                f"Raw {name} probability map at threshold {cutoff:.2f}; "
                "no component filtering or spatial policy."
            ),
            "required_votes": 1,
            "max_spatial_votes": int(mask.any()),
            "strong_models": [name] if mask.any() else [],
            "candidate_models": [name] if mask.any() else [],
            "mask_pixels": int(mask.sum()),
            "mask_png": to_data_url(mask.astype(np.uint8) * 255),
            "overlay_png": to_data_url(decision_overlay),
        }
        mode = "raw_segmentation"
    return {
        "mode": mode,
        "selection": (
            f"raw_{selection_id(normalized)}"
            if decision_mode == "raw"
            else selection_id(normalized)
        ),
        "policy_path": str(policy_path_for(normalized)) if policy else "",
        "decision": decision,
        "results": results,
    }
