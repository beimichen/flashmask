"""Smoke tests for the FastAPI service (no weights required)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from flashmask.serving.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:  # triggers lifespan (model load attempt)
        yield c


def test_health_reports_status_and_model_flag(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_predict_without_weights_returns_503(client):
    # Weights are not committed, so /predict should fail cleanly rather than crash.
    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(buf, format="PNG")
    resp = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert resp.status_code in (200, 503)
    if resp.status_code == 503:
        assert "models/README" in resp.json()["detail"]
