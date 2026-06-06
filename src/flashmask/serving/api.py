"""FastAPI service for text-region detection.

Endpoints:
    GET  /health   -> liveness + whether weights are loaded
    POST /predict  -> upload an image, get detected text regions (JSON)

The model is loaded once at startup from ``FLASHMASK_DETECTOR_ONNX`` (default
``models/detector.onnx``). Weights are not committed; if absent, the service
starts and ``/health`` reports ``model_loaded: false`` so orchestrators get a
clear signal instead of a crash.

Run with:  uvicorn flashmask.serving.api:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from flashmask.config import paths

logger = logging.getLogger(__name__)
_state: dict = {"pipeline": None}


def _detector_path() -> Path:
    return Path(os.environ.get("FLASHMASK_DETECTOR_ONNX", paths.models / "detector.onnx"))


def _classifier_path() -> Path:
    return Path(os.environ.get("FLASHMASK_CLASSIFIER_ONNX", paths.models / "text_classifier.onnx"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    det = _detector_path()
    if det.exists():
        from flashmask.inference.pipeline import TextMaskPipeline

        _state["pipeline"] = TextMaskPipeline(det, _classifier_path())
        logger.info("Loaded detector from %s", det)
    else:
        logger.warning("Detector weights not found at %s; /predict will return 503.", det)
    yield
    _state["pipeline"] = None


app = FastAPI(title="flashmask", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["pipeline"] is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    pipeline = _state["pipeline"]
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"detail": f"No detector weights at {_detector_path()}. See models/README.md."},
        )
    buf = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"detail": "Could not decode image."})
    regions = pipeline.run(img)
    return JSONResponse({"count": len(regions), "regions": [r.to_dict() for r in regions]})
