"""Fast local web application for large-scale dataset review and correction."""
from __future__ import annotations

import argparse
import io
import threading
import webbrowser
from pathlib import Path
from typing import Any, Literal

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from review_core import DECISIONS, ISSUE_TAGS, DatasetReviewService, ReviewConfig


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
STATIC_DIR = TOOL_DIR / "static"


class ReviewPayload(BaseModel):
    image_id: str
    decision: str
    corrected_label: int | None = None
    corrected_group: str = ""
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    hard_negative: bool = False
    excluded: bool = False
    reviewer: str = ""


class BulkReviewPayload(BaseModel):
    image_ids: list[str] = Field(min_length=1, max_length=500)
    decision: str
    corrected_label: int | None = None
    corrected_group: str = ""
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    hard_negative: bool = False
    excluded: bool = False
    reviewer: str = ""


class MaskPayload(BaseModel):
    data_url: str


class PredictionMaskPayload(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class ExportPayload(BaseModel):
    name: str = ""


def png_response(array: np.ndarray) -> Response:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return Response(buffer.getvalue(), media_type="image/png")


def create_app(service: DatasetReviewService) -> FastAPI:
    app = FastAPI(
        title="Aluminum Dataset Review Studio",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.service = service
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def svc() -> DatasetReviewService:
        return app.state.service

    def record_or_404(image_id: str) -> dict[str, Any]:
        try:
            return svc().get_record(image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        groups = sorted({item["defect_group"] for item in svc().records.values()})
        return {
            "summary": svc().summary(),
            "groups": groups,
            "models": svc().models,
            "decisions": sorted(DECISIONS),
            "issue_tags": sorted(ISSUE_TAGS),
            "dataset_root": str(svc().config.dataset_root),
            "database_path": str(svc().config.database_path),
            "exports_dir": str(svc().config.exports_dir),
        }

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return svc().summary()

    @app.get("/api/items")
    def items(
        split: str = "all",
        label: str = "all",
        group: str = "all",
        review_status: str = "all",
        decision: str = "all",
        candidate: str = "all",
        model: str = "all",
        min_score: float | None = Query(default=None, ge=0.0, le=1.0),
        search: str = "",
        sort: Literal["priority", "score", "area", "id", "updated"] = "priority",
        descending: bool = True,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        try:
            return svc().list_items(
                split=split,
                label=label,
                group=group,
                review_status=review_status,
                decision=decision,
                candidate=candidate,
                model=model,
                min_score=min_score,
                search=search,
                sort=sort,
                descending=descending,
                offset=offset,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/items/{image_id}")
    def item_details(image_id: str) -> dict[str, Any]:
        try:
            return svc().details(image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/items/{image_id}/image")
    def source_image(image_id: str) -> FileResponse:
        record = record_or_404(image_id)
        return FileResponse(record["image_path"])

    @app.get("/api/items/{image_id}/mask")
    def source_mask(
        image_id: str,
        version: Literal["current", "original"] = "current",
    ) -> Response:
        record = record_or_404(image_id)
        mask_path = record["mask_path"] if version == "original" else svc().current_mask_path(image_id)
        if mask_path is not None and mask_path.is_file():
            # Source masks can be encoded as 0/1. Returning that PNG directly
            # makes foreground value 1 look black in the browser even though
            # the overlay correctly treats it as foreground. Normalize only
            # the display response to 0/255; the source file remains untouched.
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
            if mask.ndim == 3:
                mask = np.any(mask != 0, axis=2)
            else:
                mask = mask != 0
            return png_response(mask.astype(np.uint8) * 255)
        return png_response(np.zeros((record["height"], record["width"]), dtype=np.uint8))

    @app.get("/api/items/{image_id}/overlay")
    def source_overlay(
        image_id: str,
        version: Literal["current", "original"] = "current",
    ) -> Response:
        record = record_or_404(image_id)
        with Image.open(record["image_path"]) as image:
            original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask_path = record["mask_path"] if version == "original" else svc().current_mask_path(image_id)
        if mask_path is None:
            return png_response(original)
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        if mask.ndim == 3:
            mask = np.any(mask != 0, axis=2)
        else:
            mask = mask != 0
        if mask.shape != original.shape[:2]:
            raise HTTPException(
                status_code=422,
                detail=f"Mask size {mask.shape} does not match image size {original.shape[:2]}",
            )
        overlay = original.copy()
        color = np.asarray([255, 78, 88], dtype=np.float32)
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * color).astype(np.uint8)
        return png_response(overlay)

    @app.get("/api/items/{image_id}/qualitative")
    def qualitative(image_id: str, model: str | None = None) -> FileResponse:
        record = record_or_404(image_id)
        available = sorted(
            name
            for name in record["predictions"]
            if (name, image_id) in svc().qualitative_assets
        )
        selected_model = model or (available[0] if available else None)
        asset = svc().qualitative_assets.get((selected_model, image_id)) if selected_model else None
        if asset is None or not asset.is_file():
            detail = (
                f"No qualitative preview for model '{model}' and this sample"
                if model
                else "No qualitative model image for this sample"
            )
            raise HTTPException(status_code=404, detail=detail)
        return FileResponse(asset, media_type="image/png")

    @app.get("/api/items/{image_id}/prediction")
    def model_prediction(
        image_id: str,
        model: str,
        view: Literal["overlay", "probability", "binary"] = "overlay",
    ) -> Response:
        record = record_or_404(image_id)
        asset = svc().prediction_assets.get((model, image_id))
        if asset is None or not asset.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"No full prediction map for model '{model}' and this sample",
            )
        with Image.open(asset) as image:
            encoded = np.asarray(image)
        if encoded.ndim == 3:
            encoded = encoded[..., 0]
        scale = 65535.0 if encoded.dtype == np.uint16 else 255.0
        probability = np.clip(encoded.astype(np.float32) / scale, 0.0, 1.0)
        threshold = float(record["predictions"].get(model, {}).get("threshold", 0.5))
        binary = probability >= threshold

        if view == "binary":
            return png_response(binary.astype(np.uint8) * 255)
        if view == "probability":
            heatmap_bgr = cv2.applyColorMap(
                np.rint(probability * 255.0).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )
            return png_response(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))

        with Image.open(record["image_path"]) as image:
            original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if probability.shape != original.shape[:2]:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction size {probability.shape} does not match image size {original.shape[:2]}",
            )
        overlay = original.copy()
        color = np.asarray([255, 215, 0], dtype=np.float32)
        overlay[binary] = (0.30 * overlay[binary] + 0.70 * color).astype(np.uint8)
        return png_response(overlay)

    @app.post("/api/reviews")
    def save_review(payload: ReviewPayload) -> dict[str, Any]:
        try:
            review = svc().save_review(payload.image_id, payload.model_dump())
            return {"ok": True, "review": review, "summary": svc().summary()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reviews/bulk")
    def save_bulk_review(payload: BulkReviewPayload) -> dict[str, Any]:
        saved = []
        base = payload.model_dump(exclude={"image_ids"})
        for image_id in dict.fromkeys(payload.image_ids):
            try:
                saved.append(svc().save_review(image_id, dict(base) | {"image_id": image_id}))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "saved": len(saved), "summary": svc().summary()}

    @app.post("/api/items/{image_id}/mask")
    def save_mask(image_id: str, payload: MaskPayload) -> dict[str, Any]:
        try:
            return {"ok": True} | svc().save_mask_data_url(image_id, payload.data_url)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/items/{image_id}/prediction-mask")
    def apply_prediction_mask(image_id: str, payload: PredictionMaskPayload) -> dict[str, Any]:
        try:
            return {"ok": True} | svc().apply_prediction_mask(image_id, payload.model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export")
    def export(payload: ExportPayload) -> dict[str, Any]:
        try:
            result = svc().export(payload.name)
            return {"ok": True} | result
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "records": len(svc().records)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and correct segmentation datasets safely")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "3cad_ani",
        help="Dataset root containing dataset_audit/splits",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        default=[],
        help="Model results directory; may be supplied more than once",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = DatasetReviewService(
        ReviewConfig(
            tool_dir=TOOL_DIR,
            dataset_root=args.dataset_root,
            results_roots=tuple(args.results_root),
        )
    )
    app = create_app(service)
    if args.open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
