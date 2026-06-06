"""Tests for model-in-the-loop pre-label source selection."""

from __future__ import annotations

from flashmask.labeling.prelabel import should_use_model


def test_uses_doctr_when_no_model(tmp_path):
    assert should_use_model(None) is False
    assert should_use_model(tmp_path / "detector.onnx") is False


def test_uses_model_once_weights_exist(tmp_path):
    weights = tmp_path / "detector.onnx"
    weights.write_bytes(b"placeholder")
    assert should_use_model(weights) is True
