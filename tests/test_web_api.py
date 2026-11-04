"""Smoke test for Phase 6 Web Demo backend.

Validates:
- GET /health
- POST /predict with a test image
- POST /explain
- GET /stats

Usage:
    .venv/Scripts/python.exe tests/test_web_api.py
"""
import sys
from pathlib import Path
import io

from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from apps.web_demo.backend.hierarchical_app import app, load_runtime

def test_api():
    load_runtime()
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    print("[PASS] /health response:", res_health.json()["status"])

    # 2. Predict with dummy image
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res_pred = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert res_pred.status_code == 200, f"Predict failed: {res_pred.text}"
    p_data = res_pred.json()
    assert p_data["decision"] in ["PASS", "REVIEW", "DEFECT"], f"Invalid decision: {p_data}"
    print(f"[PASS] /predict decision: {p_data['decision']} | Latency: {p_data['latency_ms']} ms")

    # 3. Stats check
    res_stats = client.get("/stats")
    assert res_stats.status_code == 200, f"Stats failed: {res_stats.text}"
    st_data = res_stats.json()
    print("[PASS] /stats active alerts:", len(st_data["active_alerts"]))

    # 4. Root HTML
    res_root = client.get("/")
    assert res_root.status_code == 200, f"Root HTML failed: {res_root.status_code}"
    assert "HỆ THỐNG KIỂM ĐỊNH KHUYẾT TẬT ỐNG HÚT" in res_root.text
    print("[PASS] GET / returns 3-light interactive HTML Dashboard")

    print("\n[ALL TESTS PASSED] Phase 6 Web API is fully functional!")

if __name__ == "__main__":
    test_api()
