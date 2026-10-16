"""Explainable AI (XAI) Suite for Juice Box Straw Defect Classification (Phase 5).

Provides counterfactual, causal, and geometric interpretability tools:
1. Object-Level Occlusion Attribution (Primary Industrial Proof):
   Quantifies Delta confidence when occluding:
   - Straw region (plastic target)
   - Beverage carton printing region (brand graphics)
   - Background region (studio backdrop)
   Proves the decision is causally driven by the straw, not the brand graphics.
2. 14x14 Grid Occlusion Sensitivity Mapping:
   Coarse sliding-patch counterfactual perturbation map.
3. Straw Geometry Feature Extraction:
   - Principal component orientation angle (PCA axis angle in degrees)
   - Max kink/bend curvature angle
   - Protrusion flag (straw extends outside carton boundaries)
4. Case-Based Nearest Neighbor Retrieval (k=3):
   Cosine similarity in penultimate feature space with overlay masks.
5. Adebayo Sanity Check:
   Cascading parameter randomization from classifier down to early conv layers.
6. Web/API integration:
   All functions provide JSON-serializable outputs and base64 encoded PNGs.

Run:
    .venv/Scripts/python.exe src/straw_defect_cls/explain.py
"""
from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.straw_defect_cls.dataset import FULL_CLASS_NAMES, get_val_transform
from src.straw_defect_cls.models import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Helper Utilities: Image Conversions & Base64 Encoders
# ──────────────────────────────────────────────────────────────────────────────
def np_image_to_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encode an RGB numpy array (H, W, 3) uint8 to a base64 data string."""
    pil_img = Image.fromarray(np.clip(img_rgb, 0, 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def fig_to_base64(fig: plt.Figure) -> str:
    """Encode a Matplotlib figure to a base64 data string."""
    buffered = io.BytesIO()
    fig.savefig(buffered, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a 2D normalized heatmap [0, 1] onto an RGB image (H, W, 3)."""
    h, w = img_rgb.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_norm = np.clip(heat_resized, 0.0, 1.0)
    heat_colored = cv2.applyColorMap(np.uint8(255 * heat_norm), colormap)
    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    blended = np.uint8(alpha * heat_colored + (1.0 - alpha) * img_rgb)
    return blended


def load_rgb_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Load image into RGB uint8 numpy array."""
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
        return np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        return image_input.copy()
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def load_binary_mask(mask_input: Union[str, Path, np.ndarray, Image.Image], target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load binary mask into boolean/uint8 2D array matching target_shape (H, W)."""
    if isinstance(mask_input, (str, Path)):
        mask_np = cv2.imread(str(mask_input), cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask_input, Image.Image):
        mask_np = np.array(mask_input.convert("L"))
    elif isinstance(mask_input, np.ndarray):
        mask_np = mask_input.squeeze()
    else:
        raise ValueError(f"Unsupported mask input type: {type(mask_input)}")

    if target_shape is not None and mask_np.shape[:2] != target_shape:
        mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return (mask_np > 127).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Object-Level Occlusion Attribution (Primary Industrial Proof)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_for_inference(img_rgb: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Transform numpy RGB uint8 image into normalized PyTorch tensor (3, H, W)."""
    pil_im = Image.fromarray(np.clip(img_rgb, 0, 255).astype(np.uint8))
    tf = get_val_transform(image_size)
    return tf(pil_im)


def segment_carton_regions(img_rgb: np.ndarray, straw_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Segment image into carton printing area and studio background area.

    In beverage packaging studio inspection:
    - Background: studio white backdrop (RGB luminance > 230).
    - Carton Printing: non-white carton area excluding the straw region.
    - Background Mask: backdrop area excluding the straw region.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    is_carton = gray < 230

    # Ensure straw region is excluded from carton printing
    straw_bool = (straw_mask > 0)
    carton_printing = is_carton & (~straw_bool)
    background = (~is_carton) & (~straw_bool)

    return carton_printing.astype(np.uint8), background.astype(np.uint8)


def object_attribution(
    model: nn.Module,
    img: Union[str, Path, np.ndarray, Image.Image],
    mask: Union[str, Path, np.ndarray, Image.Image],
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
    gray_fill_val: int = 128,
) -> Dict[str, Any]:
    """Evaluate counterfactual drop in confidence across 3 distinct physical regions:

    1. Straw Region (the active plastic component)
    2. Carton Printing Region (brand artwork, graphics, nutrition tables)
    3. Background Region (white studio backdrop)

    A valid causal model must display high Delta confidence when the straw is occluded,
    and minimal to zero Delta confidence when the carton printing is occluded.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    img_rgb = load_rgb_image(img)
    straw_mask = load_binary_mask(mask, target_shape=img_rgb.shape[:2])
    carton_mask, bg_mask = segment_carton_regions(img_rgb, straw_mask)

    # Prepare occluded variations with neutral gray fill (128)
    img_straw_occ = img_rgb.copy()
    img_straw_occ[straw_mask > 0] = gray_fill_val

    img_carton_occ = img_rgb.copy()
    img_carton_occ[carton_mask > 0] = gray_fill_val

    img_bg_occ = img_rgb.copy()
    img_bg_occ[bg_mask > 0] = gray_fill_val

    def _predict_prob(im: np.ndarray) -> np.ndarray:
        t = preprocess_for_inference(im, 224).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)
            probs = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
        return probs

    p_base = _predict_prob(img_rgb)
    p_straw_occ = _predict_prob(img_straw_occ)
    p_carton_occ = _predict_prob(img_carton_occ)
    p_bg_occ = _predict_prob(img_bg_occ)

    if target_class is None:
        target_class = int(np.argmax(p_base))

    conf_base = float(p_base[target_class])
    delta_straw = float(conf_base - p_straw_occ[target_class])
    delta_carton = float(conf_base - p_carton_occ[target_class])
    delta_bg = float(conf_base - p_bg_occ[target_class])

    # Causal attribution ratio: straw delta relative to total perturbation
    total_delta = max(delta_straw, 0.0) + abs(delta_carton) + abs(delta_bg) + 1e-6
    straw_attribution_ratio = float(max(delta_straw, 0.0) / total_delta)

    # Verification verdict
    # Pass if straw occlusion drops confidence substantially and carton drop is minor
    verdict = "STRAW_CAUSAL_VALIDATED" if (delta_straw > 0.20 and delta_carton < 0.15) else "BRAND_INFLUENCED_WARNING"

    return {
        "target_class_idx": target_class,
        "target_class_name": FULL_CLASS_NAMES[target_class] if target_class < len(FULL_CLASS_NAMES) else str(target_class),
        "baseline_confidence": conf_base,
        "delta_conf_straw": delta_straw,
        "delta_conf_carton": delta_carton,
        "delta_conf_bg": delta_bg,
        "straw_attribution_ratio": straw_attribution_ratio,
        "verdict": verdict,
        "overlays_base64": {
            "original": np_image_to_base64(img_rgb),
            "straw_occluded": np_image_to_base64(img_straw_occ),
            "carton_occluded": np_image_to_base64(img_carton_occ),
            "bg_occluded": np_image_to_base64(img_bg_occ),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. 14x14 Grid Occlusion Sensitivity Mapping
# ──────────────────────────────────────────────────────────────────────────────
def occlusion_map(
    model: nn.Module,
    img: Union[str, Path, np.ndarray, Image.Image],
    grid: int = 14,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    gray_fill_val: int = 128,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute 14x14 grid counterfactual occlusion sensitivity map.

    Returns:
        (raw_heatmap_2d, blended_overlay_rgb, result_metadata)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    img_rgb = load_rgb_image(img)
    h, w = img_rgb.shape[:2]
    val_tf = get_val_transform(224)

    # Baseline probability
    t_base = preprocess_for_inference(img_rgb, 224).unsqueeze(0).to(device)
    with torch.no_grad():
        p_base = F.softmax(model(t_base), dim=1).cpu().numpy().squeeze(0)

    if target_class is None:
        target_class = int(np.argmax(p_base))
    conf_base = float(p_base[target_class])

    # Generate 14x14 grid patches
    cell_h = h / grid
    cell_w = w / grid
    occluded_tensors = []
    cell_coords = []

    for r in range(grid):
        for c in range(grid):
            y1 = int(round(r * cell_h))
            y2 = int(round((r + 1) * cell_h))
            x1 = int(round(c * cell_w))
            x2 = int(round((c + 1) * cell_w))

            im_occ = img_rgb.copy()
            im_occ[y1:y2, x1:x2] = gray_fill_val
            t_occ = preprocess_for_inference(im_occ, 224)
            occluded_tensors.append(t_occ)
            cell_coords.append((r, c))

    # Batched inference
    n_patches = len(occluded_tensors)
    prob_drops = np.zeros((grid, grid), dtype=np.float32)

    with torch.no_grad():
        for b_start in range(0, n_patches, batch_size):
            batch = torch.stack(occluded_tensors[b_start : b_start + batch_size]).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, target_class].cpu().numpy()
            for idx, p_occ in enumerate(probs):
                r_idx, c_idx = cell_coords[b_start + idx]
                # Attribution = Drop in confidence when region is removed
                prob_drops[r_idx, c_idx] = max(conf_base - p_occ, 0.0)

    # Normalize heatmap to [0, 1]
    p_max = float(prob_drops.max())
    p_min = float(prob_drops.min())
    if p_max > p_min:
        norm_map = (prob_drops - p_min) / (p_max - p_min)
    else:
        norm_map = prob_drops

    # Create overlay
    overlay_rgb = overlay_heatmap(img_rgb, norm_map, alpha=0.55, colormap=cv2.COLORMAP_JET)

    meta = {
        "grid": grid,
        "target_class": target_class,
        "target_class_name": FULL_CLASS_NAMES[target_class] if target_class < len(FULL_CLASS_NAMES) else str(target_class),
        "baseline_confidence": conf_base,
        "max_drop": p_max,
        "overlay_base64": np_image_to_base64(overlay_rgb),
    }

    return norm_map, overlay_rgb, meta


# ──────────────────────────────────────────────────────────────────────────────
# 3. Straw Geometric Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────
def geometry_features(
    mask: Union[str, Path, np.ndarray, Image.Image],
    carton_margin_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Extract physical geometric features from the straw segmentation mask.

    Computes:
    - PCA Principal Orientation Angle (degrees [-90, 90])
    - Kink Curvature Angle (deviation from straight 180-deg line)
    - Protrusion Flag (straw extends beyond packaging margin)
    - Aspect ratio, contour area, and solidity.
    """
    mask_np = load_binary_mask(mask)
    h, w = mask_np.shape[:2]

    # Handle missing straw case
    num_pts = int(np.sum(mask_np > 0))
    if num_pts < 30:
        return {
            "straw_present": False,
            "pca_axis_angle_deg": 0.0,
            "kink_angle_deg": 0.0,
            "flag_straw_outside_edge": False,
            "aspect_ratio": 0.0,
            "area_pixels": num_pts,
            "solidity": 0.0,
            "geometry_diagnosis": "No straw detected (Missing).",
        }

    # 1. PCA Principal Axis Angle
    pts = np.argwhere(mask_np > 0)  # (N, 2) in (y, x)
    pts_xy = pts[:, [1, 0]].astype(np.float32)  # (x, y)
    mean, eigenvectors = cv2.PCACompute(pts_xy, mean=None)
    primary_vector = eigenvectors[0]  # (vx, vy)
    pca_angle = float(math.degrees(math.atan2(primary_vector[1], primary_vector[0])))
    # Normalize to [-90, 90]
    if pca_angle > 90:
        pca_angle -= 180
    elif pca_angle < -90:
        pca_angle += 180

    # 2. Kink Angle (Contour Polygon Deviation)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon=0.015 * peri, closed=True)

    max_kink_deg = 0.0
    pts_poly = approx.reshape(-1, 2)
    n_poly = len(pts_poly)
    if n_poly >= 3:
        for i in range(n_poly):
            p_prev = pts_poly[(i - 1) % n_poly]
            p_curr = pts_poly[i]
            p_next = pts_poly[(i + 1) % n_poly]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-4 and norm2 > 1e-4:
                cos_ang = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle = math.degrees(math.acos(cos_ang))
                # Straight line bend is 180 - angle
                kink = abs(180.0 - angle)
                if kink > max_kink_deg and kink < 170.0:  # avoid acute loop turns
                    max_kink_deg = kink

    # 3. Protrusion Flag (touches carton margins)
    x, y, bw, bh = cv2.boundingRect(cnt)
    margin_y = int(h * carton_margin_ratio)
    margin_x = int(w * carton_margin_ratio)
    protrudes = (y <= margin_y) or ((y + bh) >= (h - margin_y)) or (x <= margin_x) or ((x + bw) >= (w - margin_x))

    # 4. Aspect Ratio & Solidity
    rect = cv2.minAreaRect(cnt)
    dim1, dim2 = rect[1]
    aspect_ratio = float(max(dim1, dim2) / (min(dim1, dim2) + 1e-6))
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(cv2.contourArea(cnt) / (hull_area + 1e-6))

    # Diagnostic synthesis
    if max_kink_deg >= 35.0:
        diagnosis = f"Significant kink detected ({max_kink_deg:.1f}° deviation -> BENT candidate)."
    elif protrudes or aspect_ratio < 4.0:
        diagnosis = f"Packaging border overlap/displacement ({aspect_ratio:.1f} aspect ratio -> LOOSE candidate)."
    else:
        diagnosis = f"Straight, properly affixed straw (PCA axis: {pca_angle:.1f}°)."

    return {
        "straw_present": True,
        "pca_axis_angle_deg": round(pca_angle, 2),
        "kink_angle_deg": round(max_kink_deg, 2),
        "flag_straw_outside_edge": bool(protrudes),
        "aspect_ratio": round(aspect_ratio, 2),
        "area_pixels": int(num_pts),
        "solidity": round(solidity, 3),
        "geometry_diagnosis": diagnosis,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Case-Based Nearest Neighbor Retrieval (k=3)
# ──────────────────────────────────────────────────────────────────────────────
class CaseBasedRetriever:
    """Gallery-based nearest neighbor explanation using penultimate layer embeddings."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.gallery_metadata: List[Dict[str, Any]] = []

    def extract_embedding(self, img_rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 512-dim embedding before the final linear classifier."""
        t = preprocess_for_inference(img_rgb, 224).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "backbone"):
                feat = self.model.backbone(t)
            else:
                feat = self.model(t)
            feat = feat.view(feat.size(0), -1).cpu().numpy().squeeze(0)

        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-7)

    def build_gallery_from_csv(self, csv_path: Union[str, Path], max_samples_per_class: int = 15) -> None:
        """Build reference case gallery from dataset CSV."""
        df = pd.read_csv(csv_path)
        embeddings = []
        metadata = []

        for c_idx, c_name in enumerate(FULL_CLASS_NAMES):
            sub_df = df[df["class_name"].str.lower() == c_name.lower()].head(max_samples_per_class)
            for _, row in sub_df.iterrows():
                p = Path(row["image_path"])
                if not p.exists():
                    continue
                try:
                    im = load_rgb_image(p)
                    emb = self.extract_embedding(im)
                    embeddings.append(emb)
                    metadata.append({
                        "image_path": str(p),
                        "class_idx": c_idx,
                        "class_name": c_name,
                        "brand": row.get("brand", "Unknown"),
                    })
                except Exception:
                    continue

        if embeddings:
            self.gallery_embeddings = np.array(embeddings, dtype=np.float32)
            self.gallery_metadata = metadata

    def retrieve(
        self,
        img_rgb: np.ndarray,
        k: int = 3,
        target_class: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k nearest reference cases using Cosine Similarity."""
        if self.gallery_embeddings is None or len(self.gallery_metadata) == 0:
            return []

        query_emb = self.extract_embedding(img_rgb)
        sims = self.gallery_embeddings @ query_emb

        indices = np.argsort(-sims)
        results = []

        for idx in indices:
            meta = self.gallery_metadata[idx]
            if target_class is not None and meta["class_idx"] != target_class:
                continue

            sim_score = float(sims[idx])
            try:
                ref_im = load_rgb_image(meta["image_path"])
                thumb = cv2.resize(ref_im, (160, 160))
                b64 = np_image_to_base64(thumb)
            except Exception:
                b64 = ""

            results.append({
                "rank": len(results) + 1,
                "similarity": round(sim_score, 4),
                "class_name": meta["class_name"],
                "brand": meta["brand"],
                "image_path": meta["image_path"],
                "thumbnail_base64": b64,
            })
            if len(results) >= k:
                break

        return results


# ──────────────────────────────────────────────────────────────────────────────
# 5. Adebayo Sanity Check (Parameter Randomization Test)
# ──────────────────────────────────────────────────────────────────────────────
def sanity_check_adebayo(
    model: nn.Module,
    img: Union[str, Path, np.ndarray, Image.Image],
    device: Optional[torch.device] = None,
    grid: int = 14,
) -> Dict[str, Any]:
    """Perform cascading model parameter randomization (Adebayo et al., NeurIPS 2018).

    Tests whether the explanation map genuinely depends on learned weights:
    - Step 0: Trained Model -> baseline occlusion map
    - Step 1: Randomize classification head (`classifier`) -> measure Spearman correlation rho
    - Step 2: Randomize layer4 (top conv) -> measure rho
    - Step 3: Randomize entire model -> measure rho

    A passing sanity check requires rho to collapse towards zero as weights are randomized.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_rgb = load_rgb_image(img)

    # Backup original state dict
    original_state = copy.deepcopy(model.state_dict())

    try:
        # Step 0: Baseline attribution map
        base_map, _, _ = occlusion_map(model, img_rgb, grid=grid, device=device)
        base_flat = base_map.flatten()

        records = [{"stage": "Trained Baseline", "correlation_rho": 1.0, "p_value": 0.0}]

        def _safe_spearman(base: np.ndarray, pert: np.ndarray) -> Tuple[float, float]:
            if np.std(pert) < 1e-5 or np.std(base) < 1e-5:
                # Flat map indicates complete destruction of learned saliency -> correlation is 0.0
                return 0.0, 1.0
            r, p = spearmanr(base, pert)
            if np.isnan(r):
                return 0.0, 1.0
            return float(r), float(p)

        # Step 1: Re-initialize classifier head
        if hasattr(model, "classifier"):
            for m in model.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        map_head, _, _ = occlusion_map(model, img_rgb, grid=grid, device=device)
        rho_head, p_head = _safe_spearman(base_flat, map_head.flatten())
        records.append({
            "stage": "Randomize Classifier Head",
            "correlation_rho": round(rho_head, 4),
            "p_value": p_head,
        })

        # Step 2: Re-initialize layer4 (top conv block)
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            for m in model.backbone.layer4.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        map_l4, _, _ = occlusion_map(model, img_rgb, grid=grid, device=device)
        rho_l4, p_l4 = _safe_spearman(base_flat, map_l4.flatten())
        records.append({
            "stage": "Randomize Layer 4 (Cascading)",
            "correlation_rho": round(rho_l4, 4),
            "p_value": p_l4,
        })

        # Step 3: Randomize entire network
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        map_all, _, _ = occlusion_map(model, img_rgb, grid=grid, device=device)
        rho_all, p_all = _safe_spearman(base_flat, map_all.flatten())
        records.append({
            "stage": "Randomize Entire Network",
            "correlation_rho": round(rho_all, 4),
            "p_value": p_all,
        })

        # Verification criterion: Final correlation must be < 0.35
        passed = (rho_all < 0.35) and (rho_l4 < float(records[0]["correlation_rho"]))

        return {
            "passed": passed,
            "verdict": "ADEBAYO_SANITY_CHECK_PASSED" if passed else "FAILED_EDGE_DETECTOR_WARNING",
            "stages": records,
        }

    finally:
        # Restore intact trained weights
        model.load_state_dict(original_state)
        model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive Batch Evaluation & Report Generation
# ──────────────────────────────────────────────────────────────────────────────
def run_phase5_xai_suite() -> None:
    """Execute complete Phase 5 XAI analysis and produce audit artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f"PHASE 5: EXPLAINABLE AI (XAI) & CAUSAL AUDITING [Device: {device}]")
    print("=" * 70)

    tables_dir = REPO_ROOT / "submission" / "tables"
    figures_dir = REPO_ROOT / "submission" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained models
    model_s1_ckpt = REPO_ROOT / "experiments" / "runs" / "resnet18_stage1_r224_seed42" / "checkpoints" / "best.pt"
    model_s2_ckpt = REPO_ROOT / "experiments" / "runs" / "resnet18_stage2_r224_seed42" / "checkpoints" / "best.pt"
    model_f5_ckpt = REPO_ROOT / "experiments" / "runs" / "resnet18_stagefull5_r224_seed42" / "checkpoints" / "best.pt"

    m_full5 = build_model("resnet18", num_classes=5, pretrained=False).to(device)
    if model_f5_ckpt.exists():
        ckpt = torch.load(model_f5_ckpt, map_location=device)
        st = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        m_full5.load_state_dict(st)
        print(f"[Loaded] Full5 ResNet-18 checkpoint from: {model_f5_ckpt}")
    m_full5.eval()

    # 2. Load pseudo-masks
    masks_csv = REPO_ROOT / "data" / "masks" / "masks.csv"
    if not masks_csv.exists():
        print(f"[ERROR] Masks CSV not found at: {masks_csv}")
        return
    masks_df = pd.read_csv(masks_csv)

    # ──────────────────────────────────────────────────────────────────────────
    # Task 1: Object-Level Occlusion Attribution Table
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Computing Object-Level Occlusion Attribution across classes...")
    attribution_records = []
    geometry_records = []

    for idx, row in masks_df.iterrows():
        img_p = Path(row["image_path"])
        mask_p = Path(row["mask_path"])
        if not (img_p.exists() and mask_p.exists()):
            continue

        c_name = row["class_name"].lower()
        brand = row["brand"]
        c_idx = FULL_CLASS_NAMES.index(c_name) if c_name in FULL_CLASS_NAMES else 0

        # Object Attribution
        attr = object_attribution(m_full5, img_p, mask_p, target_class=c_idx, device=device)
        attribution_records.append({
            "image": img_p.name,
            "brand": brand,
            "class_name": c_name,
            "baseline_conf": round(attr["baseline_confidence"], 4),
            "delta_straw": round(attr["delta_conf_straw"], 4),
            "delta_carton": round(attr["delta_conf_carton"], 4),
            "delta_bg": round(attr["delta_conf_bg"], 4),
            "straw_attribution_ratio": round(attr["straw_attribution_ratio"], 4),
            "verdict": attr["verdict"],
        })

        # Geometry Features
        geom = geometry_features(mask_p)
        geom["image"] = img_p.name
        geom["class_name"] = c_name
        geom["brand"] = brand
        geom["baseline_conf"] = round(attr["baseline_confidence"], 4)
        geometry_records.append(geom)

    attr_df = pd.DataFrame(attribution_records)
    summary_attr = attr_df.groupby("class_name").agg({
        "baseline_conf": "mean",
        "delta_straw": "mean",
        "delta_carton": "mean",
        "delta_bg": "mean",
        "straw_attribution_ratio": "mean",
    }).reset_index()

    out_attr_csv = tables_dir / "xai_object_attribution.csv"
    summary_attr.to_csv(out_attr_csv, index=False)
    print(f"[DONE] Saved Object Attribution summary table to: {out_attr_csv}")
    print(summary_attr.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────────
    # Task 2: 14x14 Occlusion Sensitivity Map Overlay Figures
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Generating 14x14 Occlusion Maps for representative defect classes...")
    for c_idx, c_name in enumerate(FULL_CLASS_NAMES):
        sample_row = masks_df[masks_df["class_name"].str.lower() == c_name.lower()].iloc[0]
        img_p = Path(sample_row["image_path"])
        _, overlay_rgb, meta = occlusion_map(m_full5, img_p, grid=14, target_class=c_idx, device=device)

        out_fig = figures_dir / f"xai_occlusion_{c_name.lower()}.png"
        cv2.imwrite(str(out_fig), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        print(f"  Saved {c_name} occlusion map to: {out_fig} (Base Conf: {meta['baseline_confidence']:.2f}, Max Drop: {meta['max_drop']:.2f})")

    # ──────────────────────────────────────────────────────────────────────────
    # Task 3: Geometry Scatter Plot (Confidence vs Kink Angle)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Generating Geometry Feature Scatter Plot...")
    geom_df = pd.DataFrame(geometry_records)
    plt.figure(figsize=(8, 6))

    colors = {"good": "#2ca02c", "bent": "#d62728", "loose": "#ff7f0e", "missing": "#1f77b4", "pierced": "#9467bd"}
    for c_name, group in geom_df.groupby("class_name"):
        plt.scatter(
            group["kink_angle_deg"],
            group["baseline_conf"],
            label=c_name.capitalize(),
            color=colors.get(c_name.lower(), "gray"),
            alpha=0.75,
            edgecolors="none",
            s=45,
        )

    plt.title("Straw Geometry vs Model Confidence (ResNet-18)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Max Kink Angle (Degrees Deviation from Straight)", fontsize=11)
    plt.ylabel("Model Output Confidence", fontsize=11)
    plt.axvline(x=35.0, color="#d62728", linestyle="--", alpha=0.6, label="Bent Threshold (35°)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    scatter_png = figures_dir / "xai_confidence_vs_angle_scatter.png"
    plt.savefig(scatter_png, dpi=200)
    plt.close()
    print(f"[DONE] Saved Geometry Scatter Plot to: {scatter_png}")

    # ──────────────────────────────────────────────────────────────────────────
    # Task 4: Adebayo Sanity Check
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Executing Adebayo Parameter Randomization Sanity Check...")
    test_img = Path(masks_df.iloc[0]["image_path"])
    sanity_results = sanity_check_adebayo(m_full5, test_img, device=device)
    print(f"  Sanity Check Verdict: {sanity_results['verdict']}")
    for st in sanity_results["stages"]:
        print(f"    - {st['stage']}: Spearman rho = {st['correlation_rho']} (p={st['p_value']:.4e})")

    print("\n[DONE] Phase 5 XAI Suite successfully completed.")


if __name__ == "__main__":
    run_phase5_xai_suite()
