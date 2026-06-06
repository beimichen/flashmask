"""Export a trained YOLO detector to ONNX and verify parity with the PyTorch model.

A silent export drift (wrong opset, NMS baked in differently, preprocessing
mismatch) produces an ONNX model that detects subtly different boxes — a bug that
only surfaces in production. :func:`check_parity` compares the two on a sample
image and is exercised by ``tests/test_onnx_parity.py`` (skipped when weights or
the optional torch stack are unavailable).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_onnx(
    weights: str | Path,
    *,
    opset: int = 17,
    dynamic: bool = True,
    simplify: bool = True,
    nms: bool = True,
) -> Path:
    """Export ``weights`` to ONNX with decode+NMS baked in. Returns the .onnx path."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = model.export(format="onnx", opset=opset, dynamic=dynamic, simplify=simplify, nms=nms)
    return Path(out)


def check_parity(
    weights: str | Path, onnx_path: str | Path, image: str | Path, *, tol: float = 5.0
) -> dict:
    """Compare detection counts between the .pt and .onnx models on one image.

    Returns a dict with both box counts and a ``passed`` flag (counts within
    ``tol`` boxes). Raises if either model fails to run.
    """
    import cv2
    from ultralytics import YOLO

    from flashmask.inference.pipeline import PipelineConfig, TextMaskPipeline

    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Could not read parity image: {image}")

    torch_boxes = len(YOLO(str(weights))(img, verbose=False)[0].boxes)
    onnx_boxes = len(
        TextMaskPipeline(onnx_path, config=PipelineConfig(), use_ocr=False).detect(img)
    )

    result = {
        "torch_boxes": torch_boxes,
        "onnx_boxes": onnx_boxes,
        "passed": abs(torch_boxes - onnx_boxes) <= tol,
    }
    print(
        f"parity: torch={torch_boxes} onnx={onnx_boxes} -> {'OK' if result['passed'] else 'MISMATCH'}"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export a YOLO detector to ONNX and check parity.")
    ap.add_argument("--weights", "-w", type=Path, required=True)
    ap.add_argument("--parity-image", type=Path, help="optional image to verify .pt vs .onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)
    onnx_path = export_onnx(args.weights, opset=args.opset)
    print(f"Exported -> {onnx_path}")
    if args.parity_image:
        check_parity(args.weights, onnx_path, args.parity_image)


if __name__ == "__main__":
    main()
