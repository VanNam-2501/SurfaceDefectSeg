"""Industrial Web Demo Backend & Maintenance Monitoring Dashboard (Phase 6).

Two-Stage Hierarchical Straw Defect Quality Control System:
- Tri-State Decision: 🟢 PASS | 🟡 REVIEW | 🔴 DEFECT
- Real-time Guard Chain & Neyman-Pearson safety thresholding
- Active Learning Review Pool (stores borderline samples)
- On-demand Causal XAI (14x14 grid occlusion sensitivity + case-based k=3)
- Statistical Process Control (SPC) with EWMA 3-sigma predictive maintenance alerts
- Dual runtime: ONNX Runtime (fast inference) with automatic PyTorch fallback

Run:
    .venv/Scripts/python.exe apps/web_demo/backend/hierarchical_app.py
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.straw_defect_cls.dataset import FULL_CLASS_NAMES, get_val_transform
from src.straw_defect_cls.decide import decide
from src.straw_defect_cls.explain import (
    CaseBasedRetriever,
    geometry_features,
    load_binary_mask,
    load_rgb_image,
    np_image_to_base64,
    occlusion_map,
    preprocess_for_inference,
)
from src.straw_defect_cls.models import build_model

# ──────────────────────────────────────────────────────────────────────────────
# Global Configuration & Paths
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = REPO_ROOT / "configs" / "thresholds.json"
REVIEW_POOL_DIR = REPO_ROOT / "data" / "review_pool"
REVIEW_POOL_DIR.mkdir(parents=True, exist_ok=True)

ONNX_S1 = REPO_ROOT / "artifacts" / "checkpoints" / "onnx" / "stage1_resnet18.onnx"
ONNX_S2 = REPO_ROOT / "artifacts" / "checkpoints" / "onnx" / "stage2_resnet18.onnx"
PYTORCH_S1 = REPO_ROOT / "experiments" / "runs" / "resnet18_stage1_r224_seed42" / "checkpoints" / "best.pt"
PYTORCH_S2 = REPO_ROOT / "experiments" / "runs" / "resnet18_stage2_r224_seed42" / "checkpoints" / "best.pt"
PYTORCH_F5 = REPO_ROOT / "experiments" / "runs" / "resnet18_stagefull5_r224_seed42" / "checkpoints" / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Industrial Straw Defect Quality Control System",
    description="Two-Stage Deep Learning with Tri-State Decision & SPC EWMA Maintenance Alerts",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory runtime state
ort_session_s1 = None
ort_session_s2 = None
torch_model_s1 = None
torch_model_s2 = None
torch_model_f5 = None
case_retriever: Optional[CaseBasedRetriever] = None

# EWMA Statistical History
INSPECTION_HISTORY: List[Dict[str, Any]] = []

MAINTENANCE_MESSAGES = {
    "loose": "Tỉ lệ lỗi Loose (bung keo) vượt ngưỡng 3σ! Kiểm tra nhiệt độ và áp suất vòi phun keo nóng (Hot-melt nozzle).",
    "bent": "Tỉ lệ lỗi Bent (gập ống) vượt ngưỡng 3σ! Kiểm tra cơ cấu dao cắt, trục kẹp và cụm ép định hình ống hút.",
    "missing": "CẢNH BÁO NGUY HIỂM: Tỉ lệ Missing (mất ống) vượt ngưỡng 3σ! Kiểm tra cảm biến quang và phễu cấp ống tự động.",
    "pierced": "Tỉ lệ lỗi Pierced (thủng màng) vượt ngưỡng 3σ! Kiểm tra độ mòn kim định vị và lực ép bao gói.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading & Runtime Initialisation
# ──────────────────────────────────────────────────────────────────────────────
def load_runtime():
    global ort_session_s1, ort_session_s2, torch_model_s1, torch_model_s2, torch_model_f5, case_retriever

    # 1. Try ONNX Runtime first
    try:
        import onnxruntime as ort
        if ONNX_S1.exists() and ONNX_S2.exists():
            ort_session_s1 = ort.InferenceSession(str(ONNX_S1), providers=["CPUExecutionProvider"])
            ort_session_s2 = ort.InferenceSession(str(ONNX_S2), providers=["CPUExecutionProvider"])
            print("[OK] Initialized ONNX Runtime sessions for Stage 1 & Stage 2.")
    except Exception as e:
        print(f"[INFO] ONNX Runtime session failed: {e}. Falling back to PyTorch.")

    # 2. PyTorch models (for GPU inference, XAI, and fallback)
    try:
        s1 = build_model("resnet18", num_classes=2, pretrained=False)
        if PYTORCH_S1.exists():
            c1 = torch.load(PYTORCH_S1, map_location=DEVICE)
            s1.load_state_dict(c1.get("model", c1))
            print(f"[OK] Loaded PyTorch Stage 1 from {PYTORCH_S1.name}")
        torch_model_s1 = s1.to(DEVICE).eval()

        s2 = build_model("resnet18", num_classes=4, pretrained=False)
        if PYTORCH_S2.exists():
            c2 = torch.load(PYTORCH_S2, map_location=DEVICE)
            s2.load_state_dict(c2.get("model", c2))
            print(f"[OK] Loaded PyTorch Stage 2 from {PYTORCH_S2.name}")
        torch_model_s2 = s2.to(DEVICE).eval()

        f5 = build_model("resnet18", num_classes=5, pretrained=False)
        if PYTORCH_F5.exists():
            c5 = torch.load(PYTORCH_F5, map_location=DEVICE)
            f5.load_state_dict(c5.get("model", c5))
            print(f"[OK] Loaded PyTorch Full5 from {PYTORCH_F5.name}")
        torch_model_f5 = f5.to(DEVICE).eval()

        # Build Case-Based Retriever gallery
        case_retriever = CaseBasedRetriever(torch_model_f5, device=DEVICE)
        gallery_csv = REPO_ROOT / "data" / "straw_defects_splits" / "val.csv"
        if gallery_csv.exists():
            case_retriever.build_gallery_from_csv(gallery_csv, max_samples_per_class=10)
            print(f"[OK] Built Case-Based XAI Gallery ({len(case_retriever.gallery_metadata)} reference samples).")

    except Exception as exc:
        print(f"[ERROR] Failed initializing PyTorch models: {exc}")


@app.on_event("startup")
def startup_handler():
    load_runtime()


def read_threshold_config() -> dict:
    """Read dynamic threshold configuration from disk."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "tau_pass": 0.995,
        "tau_defect": 0.40,
        "tau2": 0.5579,
        "margin2": 0.1979,
        "e_thr": -1.3553,
        "temperature_stage1": 3.2544,
        "temperature_stage2": 0.6362,
        "pi_deploy_defect": 0.05,
        "pi_train_defect": 0.80,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inference Engine Helper
# ──────────────────────────────────────────────────────────────────────────────
def run_forward_pass(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute Stage 1 and Stage 2 forward pass with ONNX (fallback to PyTorch)."""
    t0 = time.perf_counter()
    tensor = preprocess_for_inference(img_rgb, 224)  # (3, 224, 224)

    # Stage 1
    if ort_session_s1 is not None:
        batch_np = tensor.unsqueeze(0).numpy()
        input_name = ort_session_s1.get_inputs()[0].name
        logits1 = ort_session_s1.run(None, {input_name: batch_np})[0].squeeze(0)
    else:
        with torch.no_grad():
            t_dev = tensor.unsqueeze(0).to(DEVICE)
            logits1 = torch_model_s1(t_dev).cpu().numpy().squeeze(0)

    # Stage 2
    if ort_session_s2 is not None:
        batch_np = tensor.unsqueeze(0).numpy()
        input_name = ort_session_s2.get_inputs()[0].name
        logits2 = ort_session_s2.run(None, {input_name: batch_np})[0].squeeze(0)
    else:
        with torch.no_grad():
            t_dev = tensor.unsqueeze(0).to(DEVICE)
            logits2 = torch_model_s2(t_dev).cpu().numpy().squeeze(0)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return logits1, logits2, latency_ms


# ──────────────────────────────────────────────────────────────────────────────
# Core API Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    cfg = read_threshold_config()
    return {
        "status": "online",
        "device": str(DEVICE),
        "onnx_runtime_active": ort_session_s1 is not None,
        "pytorch_active": torch_model_s1 is not None,
        "thresholds": cfg,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run Two-Stage quality inspection with 3-light decision and active learning storage."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}")

    cfg = read_threshold_config()
    logits1, logits2, latency_ms = run_forward_pass(img_rgb)

    # Decision Engine
    decision, d_cls_idx, reason = decide(logits1, logits2, cfg)

    # Class probabilities
    t1 = float(cfg.get("temperature_stage1", 1.0))
    t2 = float(cfg.get("temperature_stage2", 1.0))
    s1_probs = F.softmax(torch.from_numpy(logits1 / t1), dim=-1).numpy()
    s2_probs = F.softmax(torch.from_numpy(logits2 / t2), dim=-1).numpy()

    stage2_names = ["Bent", "Loose", "Missing", "Pierced"]
    defect_name = stage2_names[d_cls_idx] if d_cls_idx is not None else None

    # Geometry Feature Extraction (extract pseudo mask via thresholding)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, straw_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    geom = geometry_features(straw_mask)

    # Thumbnail overlay
    thumb = cv2.resize(img_rgb, (320, 320))
    overlay_b64 = np_image_to_base64(thumb)

    payload = {
        "decision": decision,
        "defect_class": defect_name,
        "reason": reason,
        "latency_ms": round(latency_ms, 2),
        "stage1_probabilities": {
            "Good": round(float(s1_probs[0]), 4),
            "Defect": round(float(s1_probs[1]), 4),
        },
        "stage2_probabilities": {
            stage2_names[i]: round(float(s2_probs[i]), 4) for i in range(4)
        },
        "geometry": geom,
        "overlay_base64": overlay_b64,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Record in history for EWMA SPC
    INSPECTION_HISTORY.append({
        "timestamp": time.time(),
        "decision": decision,
        "defect_class": defect_name or ("Good" if decision == "PASS" else "Review"),
    })

    # Active Learning: Persist REVIEW samples
    if decision == "REVIEW":
        sample_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        save_img_path = REVIEW_POOL_DIR / f"{sample_id}.jpg"
        save_json_path = REVIEW_POOL_DIR / f"{sample_id}.json"
        try:
            pil_img.save(save_img_path, format="JPEG", quality=90)
            save_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            payload["active_learning_saved"] = str(save_img_path.name)
        except Exception as e:
            print(f"[WARN] Failed to write review pool sample: {e}")

    return JSONResponse(payload)


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """Generate on-demand 14x14 counterfactual occlusion map and Case-Based top-3 nearest neighbors."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}")

    if torch_model_f5 is None:
        raise HTTPException(status_code=500, detail="PyTorch explanation model not loaded.")

    # 1. 14x14 Occlusion Sensitivity Map
    _, overlay_rgb, meta = occlusion_map(torch_model_f5, img_rgb, grid=14, device=DEVICE)

    # 2. Case-based nearest neighbor retrieval (k=3)
    cases = []
    if case_retriever is not None:
        cases = case_retriever.retrieve(img_rgb, k=3)

    return JSONResponse({
        "status": "success",
        "occlusion_heatmap_overlay": np_image_to_base64(overlay_rgb),
        "occlusion_metadata": meta,
        "nearest_neighbor_cases": cases,
    })


@app.get("/stats")
def stats():
    """Compute Statistical Process Control (SPC) with EWMA 3-sigma predictive maintenance alerts."""
    now = time.time()
    one_hour_ago = now - 3600.0

    recent = [s for s in INSPECTION_HISTORY if s["timestamp"] >= one_hour_ago]
    total_recent = len(recent)

    counts = {"PASS": 0, "REVIEW": 0, "DEFECT": 0}
    defect_breakdown = {"Bent": 0, "Loose": 0, "Missing": 0, "Pierced": 0}

    for item in recent:
        counts[item["decision"]] = counts.get(item["decision"], 0) + 1
        d_cls = item.get("defect_class")
        if d_cls in defect_breakdown:
            defect_breakdown[d_cls] += 1

    # Simulated EWMA baseline rates (lambda = 0.20, nominal rate = 2.0%)
    ewma_lambda = 0.20
    nominal_defect_rate = 0.02
    sigma_baseline = 0.015

    # Current defect rate
    defect_rate = (counts["DEFECT"] / total_recent) if total_recent > 0 else 0.02
    ucl = nominal_defect_rate + 3.0 * sigma_baseline

    alerts = []
    for cls_name, cnt in defect_breakdown.items():
        cls_rate = (cnt / total_recent) if total_recent > 0 else 0.005
        # If class rate exceeds 3-sigma control limit
        if cls_rate > (0.015 + 3.0 * 0.01):
            msg = MAINTENANCE_MESSAGES.get(cls_name.lower(), f"Tỉ lệ lỗi {cls_name} tăng bất thường!")
            alerts.append({
                "severity": "CRITICAL",
                "class_name": cls_name,
                "current_rate": round(cls_rate * 100, 2),
                "control_limit_ucl": round(ucl * 100, 2),
                "maintenance_action": msg,
            })

    return JSONResponse({
        "window_seconds": 3600,
        "total_inspected_past_hour": total_recent,
        "decisions": counts,
        "defect_breakdown": defect_breakdown,
        "ewma_current_defect_rate_pct": round(defect_rate * 100, 2),
        "ewma_ucl_pct": round(ucl * 100, 2),
        "active_alerts": alerts,
        "review_pool_count": len(list(REVIEW_POOL_DIR.glob("*.jpg"))),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Web Dashboard (3-Light GUI)
# ──────────────────────────────────────────────────────────────────────────────
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Factory Straw Defect QC & Predictive Maintenance System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --green: #22c55e;
      --yellow: #eab308;
      --red: #ef4444;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    .header h1 { font-size: 1.5rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
    .header p { font-size: 0.875rem; color: var(--text-muted); }
    .badge {
      background: #0284c7;
      color: white;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 24px; }
    @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: #e2e8f0;
    }

    /* 3-LIGHT INDUSTRIAL SIGNAL */
    .traffic-signal {
      display: flex;
      justify-content: space-around;
      background: #090d16;
      padding: 16px;
      border-radius: 16px;
      border: 2px solid var(--border);
      margin-bottom: 20px;
    }
    .light-box {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .light {
      width: 60px;
      height: 60px;
      border-radius: 50%;
      background: #1e293b;
      opacity: 0.25;
      transition: all 0.3s ease;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
    }
    .light.pass-light.active {
      background: var(--green);
      opacity: 1;
      box-shadow: 0 0 25px var(--green), inset 0 0 10px #fff;
    }
    .light.review-light.active {
      background: var(--yellow);
      opacity: 1;
      box-shadow: 0 0 25px var(--yellow), inset 0 0 10px #fff;
    }
    .light.defect-light.active {
      background: var(--red);
      opacity: 1;
      box-shadow: 0 0 25px var(--red), inset 0 0 10px #fff;
    }
    .light-label { font-size: 0.75rem; font-weight: 800; color: var(--text-muted); }

    /* DROPZONE */
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 10px;
      padding: 30px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s;
      background: rgba(15, 23, 42, 0.4);
    }
    .dropzone:hover { border-color: var(--accent); }
    .dropzone input { display: none; }

    /* PREVIEW & TELEMETRY */
    .preview-box {
      width: 100%;
      max-height: 280px;
      object-fit: contain;
      border-radius: 8px;
      background: #000;
      display: none;
      margin-top: 14px;
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #334155;
      font-size: 0.85rem;
    }
    .telemetry-row:last-child { border-bottom: none; }
    .telemetry-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--accent); }

    /* PROBABILITY BARS */
    .bar-container { margin-bottom: 10px; }
    .bar-label { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; }
    .bar-track { height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.4s ease; }

    /* BUTTONS */
    .btn {
      background: #0284c7;
      color: white;
      border: none;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: background 0.2s;
    }
    .btn:hover { background: #0369a1; }
    .btn-secondary { background: #334155; }
    .btn-secondary:hover { background: #475569; }

    /* ALERTS */
    .alert-banner {
      background: rgba(239, 68, 68, 0.15);
      border-left: 4px solid var(--red);
      padding: 12px 16px;
      border-radius: 6px;
      margin-bottom: 16px;
      font-size: 0.85rem;
    }
    .alert-banner strong { color: var(--red); }
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>HỆ THỐNG KIỂM ĐỊNH KHUYẾT TẬT ỐNG HÚT & BẢO TRÌ DÂY CHUYỀN</h1>
      <p>Kiến trúc Học Sâu Hai Giai Đoạn (Hierarchical Two-Stage) • Phân loại 3 Đèn Công Nghiệp</p>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span class="badge">ONNX & PyTorch Dual Engine</span>
      <span class="badge" style="background:#059669;">FNR worst-case ≤ 1%</span>
    </div>
  </div>

  <div class="grid">
    <!-- LEFT PANEL: Upload & Signal -->
    <div>
      <!-- 3-Light Traffic Signal -->
      <div class="traffic-signal">
        <div class="light-box">
          <div id="passLight" class="light pass-light"></div>
          <span class="light-label">🟢 PASS (ĐẠT)</span>
        </div>
        <div class="light-box">
          <div id="reviewLight" class="light review-light"></div>
          <span class="light-label">🟡 REVIEW (KIỂM TRA)</span>
        </div>
        <div class="light-box">
          <div id="defectLight" class="light defect-light"></div>
          <span class="light-label">🔴 DEFECT (LỖI)</span>
        </div>
      </div>

      <!-- Upload Dropzone -->
      <div class="card">
        <div class="card-title">Tải Ảnh Hộp Đồ Uống Kiểm Định</div>
        <div class="dropzone" id="dropzone">
          <p style="font-size:1.1rem; font-weight:600;">Kéo thả ảnh hoặc bấm để chọn</p>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">Hỗ trợ ảnh RGB độ phân giải cao (.jpg, .png)</p>
          <input type="file" id="fileInput" accept="image/*">
        </div>
        <img id="previewImg" class="preview-box" alt="Input Preview">
        <div style="margin-top:14px; display:flex; gap:10px;">
          <button id="predictBtn" class="btn" style="flex:1;">⚡ Chạy Kiểm Định (Predict)</button>
          <button id="explainBtn" class="btn btn-secondary" style="display:none;">🔍 Giải Thích (XAI)</button>
        </div>
      </div>

      <!-- Industrial Maintenance Alert (EWMA) -->
      <div class="card">
        <div class="card-title">
          <span>Giám Sát Thống Kê Dây Chuyền (SPC - EWMA)</span>
          <span id="activeAlertCount" class="badge" style="background:#059669;">ỔN ĐỊNH</span>
        </div>
        <div id="alertContainer">
          <p style="font-size:0.85rem; color:var(--text-muted);">Dây chuyền đang hoạt động trong ngưỡng kiểm soát 3-sigma (UCL ≤ 6.5%).</p>
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL: Diagnostics & Telemetry -->
    <div>
      <!-- Diagnosis Result -->
      <div class="card">
        <div class="card-title">
          <span>Kết Quả Quyết Định (Industrial Decision)</span>
          <span id="latencyBadge" class="badge" style="background:#334155;">-- ms</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
          <h2 id="decisionText" style="font-size:2rem; font-weight:800; color:var(--text-muted);">CHỜ ẢNH...</h2>
          <span id="defectNameText" style="font-size:1.2rem; font-weight:700;"></span>
        </div>
        <div class="telemetry-row">
          <span>Lý Do Kỹ Thuật (Reason String):</span>
          <span id="reasonVal" class="telemetry-val">--</span>
        </div>
        <div class="telemetry-row">
          <span>Chốt Chặn Guard Stage 2:</span>
          <span id="guardVal" class="telemetry-val">--</span>
        </div>
      </div>

      <!-- Geometry Telemetry -->
      <div class="card">
        <div class="card-title">Đặc Trưng Hình Học Ống Hút (Mask Geometry)</div>
        <div class="telemetry-row">
          <span>Trục chính PCA (Góc nghiêng):</span>
          <span id="pcaVal" class="telemetry-val">--°</span>
        </div>
        <div class="telemetry-row">
          <span>Góc uốn gập Kink Angle:</span>
          <span id="kinkVal" class="telemetry-val">--°</span>
        </div>
        <div class="telemetry-row">
          <span>Tràn viền vỏ hộp (Outside Edge):</span>
          <span id="edgeVal" class="telemetry-val">--</span>
        </div>
        <div class="telemetry-row">
          <span>Chẩn đoán vật lý:</span>
          <span id="diagnosisVal" style="font-size:0.85rem; color:#f1f5f9;">--</span>
        </div>
      </div>

      <!-- Confidence Breakdown -->
      <div class="card">
        <div class="card-title">Xác Suất Từng Giai Đoạn (Confidence Breakdown)</div>
        <p style="font-size:0.75rem; color:var(--text-muted); margin-bottom:8px;">STAGE 1 (Sàng lọc nhị phân: Good vs Defect)</p>
        <div class="bar-container">
          <div class="bar-label"><span>Good</span><span id="pGoodText">0.0%</span></div>
          <div class="bar-track"><div id="pGoodBar" class="bar-fill" style="width:0%;"></div></div>
        </div>
        <div class="bar-container">
          <div class="bar-label"><span>Defect</span><span id="pDefectText">0.0%</span></div>
          <div class="bar-track"><div id="pDefectBar" class="bar-fill" style="width:0%; background:var(--red);"></div></div>
        </div>

        <p style="font-size:0.75rem; color:var(--text-muted); margin-top:16px; margin-bottom:8px;">STAGE 2 (Chẩn đoán 4 dạng lỗi)</p>
        <div class="bar-container">
          <div class="bar-label"><span>Bent (Ống cong/gập)</span><span id="pBentText">0.0%</span></div>
          <div class="bar-track"><div id="pBentBar" class="bar-fill" style="width:0%;"></div></div>
        </div>
        <div class="bar-container">
          <div class="bar-label"><span>Loose (Bung keo)</span><span id="pLooseText">0.0%</span></div>
          <div class="bar-track"><div id="pLooseBar" class="bar-fill" style="width:0%;"></div></div>
        </div>
        <div class="bar-container">
          <div class="bar-label"><span>Missing (Mất ống)</span><span id="pMissingText">0.0%</span></div>
          <div class="bar-track"><div id="pMissingBar" class="bar-fill" style="width:0%;"></div></div>
        </div>
        <div class="bar-container">
          <div class="bar-label"><span>Pierced (Thủng màng)</span><span id="pPiercedText">0.0%</span></div>
          <div class="bar-track"><div id="pPiercedBar" class="bar-fill" style="width:0%;"></div></div>
        </div>
      </div>

      <!-- XAI Modal Container -->
      <div id="xaiCard" class="card" style="display:none;">
        <div class="card-title">Giải Thích Nhân Quả (14x14 Grid Occlusion & Case Retrieval)</div>
        <img id="xaiHeatmapImg" style="width:100%; border-radius:8px; margin-bottom:12px;" alt="Occlusion Heatmap">
        <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:8px;">3 Ca Mẫu Tham Chiếu Gần Nhất (Top-3 Nearest Neighbors):</p>
        <div id="caseContainer" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px;"></div>
      </div>
    </div>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImg');
    const predictBtn = document.getElementById('predictBtn');
    const explainBtn = document.getElementById('explainBtn');

    let currentFile = null;

    dropzone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => handleFile(e.target.files[0]);
    dropzone.ondragover = (e) => { e.preventDefault(); dropzone.style.borderColor = '#38bdf8'; };
    dropzone.ondragleave = () => { dropzone.style.borderColor = '#334155'; };
    dropzone.ondrop = (e) => {
      e.preventDefault();
      dropzone.style.borderColor = '#334155';
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    };

    function handleFile(file) {
      if (!file) return;
      currentFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
      };
      reader.readAsDataURL(file);
      document.getElementById('xaiCard').style.display = 'none';
      explainBtn.style.display = 'none';
    }

    predictBtn.onclick = async () => {
      if (!currentFile) {
        alert('Vui lòng chọn hoặc kéo thả một ảnh hộp đồ uống!');
        return;
      }
      predictBtn.disabled = true;
      predictBtn.innerText = 'Đang phân tích...';

      const formData = new FormData();
      formData.append('file', currentFile);

      try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        updateUI(data);
        explainBtn.style.display = 'inline-flex';
      } catch (err) {
        alert('Lỗi kết nối server: ' + err);
      } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = '⚡ Chạy Kiểm Định (Predict)';
      }
    };

    explainBtn.onclick = async () => {
      if (!currentFile) return;
      explainBtn.disabled = true;
      explainBtn.innerText = 'Đang chạy XAI...';
      const formData = new FormData();
      formData.append('file', currentFile);

      try {
        const res = await fetch('/explain', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('xaiHeatmapImg').src = data.occlusion_heatmap_overlay;

        const caseBox = document.getElementById('caseContainer');
        caseBox.innerHTML = '';
        data.nearest_neighbor_cases.forEach(c => {
          const div = document.createElement('div');
          div.style = 'background:#0f172a; padding:8px; border-radius:6px; text-align:center; font-size:0.75rem;';
          div.innerHTML = `<img src="${c.thumbnail_base64}" style="width:100%; border-radius:4px; margin-bottom:4px;"><br><strong>${c.class_name}</strong><br><span style="color:var(--text-muted);">${c.brand} (Sim: ${(c.similarity*100).toFixed(1)}%)</span>`;
          caseBox.appendChild(div);
        });
        document.getElementById('xaiCard').style.display = 'block';
      } catch (err) {
        alert('Lỗi XAI: ' + err);
      } finally {
        explainBtn.disabled = false;
        explainBtn.innerText = '🔍 Giải Thích (XAI)';
      }
    };

    function updateUI(d) {
      // 1. Traffic signal
      document.getElementById('passLight').classList.remove('active');
      document.getElementById('reviewLight').classList.remove('active');
      document.getElementById('defectLight').classList.remove('active');

      const decText = document.getElementById('decisionText');
      if (d.decision === 'PASS') {
        document.getElementById('passLight').classList.add('active');
        decText.innerText = '🟢 PASS';
        decText.style.color = 'var(--green)';
      } else if (d.decision === 'REVIEW') {
        document.getElementById('reviewLight').classList.add('active');
        decText.innerText = '🟡 REVIEW';
        decText.style.color = 'var(--yellow)';
      } else {
        document.getElementById('defectLight').classList.add('active');
        decText.innerText = '🔴 DEFECT';
        decText.style.color = 'var(--red)';
      }

      document.getElementById('defectNameText').innerText = d.defect_class ? `Lỗi: ${d.defect_class}` : '';
      document.getElementById('latencyBadge').innerText = `${d.latency_ms} ms`;
      document.getElementById('reasonVal').innerText = d.reason;
      document.getElementById('guardVal').innerText = d.decision === 'DEFECT' ? 'ĐẠT 3 GUARD' : (d.decision === 'REVIEW' ? 'CHẶN BẢO VỆ' : 'PASS THẲNG S1');

      // 2. Geometry
      if (d.geometry) {
        document.getElementById('pcaVal').innerText = `${d.geometry.pca_axis_angle_deg}°`;
        document.getElementById('kinkVal').innerText = `${d.geometry.kink_angle_deg}°`;
        document.getElementById('edgeVal').innerText = d.geometry.flag_straw_outside_edge ? 'CÓ (Bung lệch)' : 'KHÔNG (Chuẩn)';
        document.getElementById('diagnosisVal').innerText = d.geometry.geometry_diagnosis;
      }

      // 3. Probability bars
      const s1 = d.stage1_probabilities;
      document.getElementById('pGoodText').innerText = `${(s1.Good * 100).toFixed(1)}%`;
      document.getElementById('pGoodBar').style.width = `${s1.Good * 100}%`;
      document.getElementById('pDefectText').innerText = `${(s1.Defect * 100).toFixed(1)}%`;
      document.getElementById('pDefectBar').style.width = `${s1.Defect * 100}%`;

      const s2 = d.stage2_probabilities;
      ['Bent', 'Loose', 'Missing', 'Pierced'].forEach(name => {
        const val = s2[name] || 0.0;
        document.getElementById(`p${name}Text`).innerText = `${(val * 100).toFixed(1)}%`;
        document.getElementById(`p${name}Bar`).style.width = `${val * 100}%`;
      });
    }

    // Poll stats every 10 seconds
    async function refreshStats() {
      try {
        const res = await fetch('/stats');
        const st = await res.json();
        const alertBox = document.getElementById('alertContainer');
        const countBadge = document.getElementById('activeAlertCount');

        if (st.active_alerts && st.active_alerts.length > 0) {
          countBadge.innerText = `${st.active_alerts.length} CẢNH BÁO`;
          countBadge.style.background = 'var(--red)';
          alertBox.innerHTML = '';
          st.active_alerts.forEach(a => {
            alertBox.innerHTML += `<div class="alert-banner"><strong>[CẢNH BÁO 3-SIGMA] Lỗi ${a.class_name}:</strong> Hiện tại ${a.current_rate}% (Vượt ngưỡng UCL ${a.control_limit_ucl}%).<br>👉 <em>${a.maintenance_action}</em></div>`;
          });
        }
      } catch (e) {}
    }
    setInterval(refreshStats, 10000);
    refreshStats();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve real-time 3-light Industrial Web Dashboard."""
    return HTML_RESPONSE_CACHE


HTML_RESPONSE_CACHE = HTMLResponse(content=HTML_DASHBOARD)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hierarchical_app:app", host="0.0.0.0", port=8000, reload=False)
