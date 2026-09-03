FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements /app/requirements
COPY apps/web_demo/backend/requirements.txt /app/apps/web_demo/backend/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       torch==2.11.0 torchvision==0.26.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
       -r /app/apps/web_demo/backend/requirements.txt

COPY src /app/src
COPY apps/web_demo/backend /app/apps/web_demo/backend
COPY artifacts/checkpoints/final/unet_best.pt /app/artifacts/checkpoints/final/unet_best.pt
COPY artifacts/checkpoints/final/segformer_best.pt /app/artifacts/checkpoints/final/segformer_best.pt
COPY artifacts/reports/final/decision_and_test_audit/adaptive_single/adaptive_component_policy.json /app/artifacts/reports/final/decision_and_test_audit/adaptive_single/adaptive_component_policy.json
COPY artifacts/reports/final/decision_and_test_audit/spatial/unet_segformer/policy/decision_policy.json /app/artifacts/reports/final/decision_and_test_audit/spatial/unet_segformer/policy/decision_policy.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app:app", "--app-dir", "/app/apps/web_demo", "--host", "0.0.0.0", "--port", "8000"]

