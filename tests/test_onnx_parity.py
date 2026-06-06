"""Parity test: the ONNX export must detect the same boxes as the .pt model.

Skipped automatically unless the optional [train] stack is installed AND trained
weights + a parity image are present (they are not committed). To run it:

    export FLASHMASK_DETECTOR_PT=models/best.pt
    export FLASHMASK_PARITY_IMAGE=data/sample/images/sample_diagram_00.png
    uv run --extra train pytest tests/test_onnx_parity.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_weights

_HAS_TRAIN_STACK = all(importlib.util.find_spec(m) for m in ("torch", "ultralytics"))
_WEIGHTS = Path(os.environ.get("FLASHMASK_DETECTOR_PT", "models/best.pt"))
_IMAGE = Path(os.environ.get("FLASHMASK_PARITY_IMAGE", "data/sample/images/sample_diagram_00.png"))


@pytest.mark.skipif(
    not _HAS_TRAIN_STACK, reason="optional [train] stack (torch/ultralytics) not installed"
)
@pytest.mark.skipif(not _WEIGHTS.exists(), reason=f"no weights at {_WEIGHTS} (not committed)")
@pytest.mark.skipif(not _IMAGE.exists(), reason=f"no parity image at {_IMAGE}")
def test_pt_and_onnx_agree():
    from flashmask.modeling.export import check_parity, export_onnx

    onnx_path = export_onnx(_WEIGHTS)
    result = check_parity(_WEIGHTS, onnx_path, _IMAGE, tol=5)
    assert result["passed"], result
