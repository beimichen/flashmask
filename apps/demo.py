"""Gradio demo: upload a diagram, see its text regions detected and masked.

Loads the ONNX detector (and optional classifier) from ``models/``. Weights are
not committed — train via the pipeline or drop a ``detector.onnx`` into
``models/`` (see models/README.md). Without weights the UI still launches and
explains what's missing, so the demo never hard-crashes.

Run:  python apps/demo.py        (or: just demo)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from flashmask.config import paths

DETECTOR = Path(os.environ.get("FLASHMASK_DETECTOR_ONNX", paths.models / "detector.onnx"))
CLASSIFIER = Path(
    os.environ.get("FLASHMASK_CLASSIFIER_ONNX", paths.models / "text_classifier.onnx")
)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None and DETECTOR.exists():
        from flashmask.inference.pipeline import TextMaskPipeline

        _pipeline = TextMaskPipeline(DETECTOR, CLASSIFIER)
    return _pipeline


def detect_and_mask(image: np.ndarray):
    """Return (annotated, masked) RGB images for the Gradio gallery."""
    if image is None:
        return None, None
    pipeline = _get_pipeline()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if pipeline is None:
        note = image.copy()
        cv2.putText(
            note,
            "No detector.onnx in models/ - see models/README.md",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 0, 0),
            2,
        )
        return note, image

    regions = pipeline.run(bgr)
    annotated = bgr.copy()
    for r in regions:
        cv2.rectangle(annotated, (r.x, r.y), (r.x + r.width, r.y + r.height), (0, 200, 0), 2)
    masked = pipeline.mask(bgr, regions)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)


def build_demo() -> gr.Blocks:
    sample_dir = paths.data_sample / "images"
    examples = [str(p) for p in sorted(sample_dir.glob("*.png"))] if sample_dir.exists() else None
    with gr.Blocks(title="flashmask") as demo:
        gr.Markdown(
            "# flashmask\nDetect text regions in a diagram and mask them to make a study flashcard."
        )
        with gr.Row():
            inp = gr.Image(label="Diagram", type="numpy")
            with gr.Column():
                out_annotated = gr.Image(label="Detected regions")
                out_masked = gr.Image(label="Masked (flashcard)")
        gr.Button("Detect & mask", variant="primary").click(
            detect_and_mask, inputs=inp, outputs=[out_annotated, out_masked]
        )
        if examples:
            gr.Examples(examples=examples, inputs=inp)
    return demo


if __name__ == "__main__":
    build_demo().launch()
