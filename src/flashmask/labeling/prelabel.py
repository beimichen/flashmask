"""Pick how to pre-label an image: the trained detector if it exists, else DocTR.

This closes the label -> train -> label loop (a model-in-the-loop data flywheel):
the first pass uses DocTR OCR because no model exists yet; once a detector has
been trained, the labeling tool loads it and seeds boxes from the model's own
predictions, so each annotation round *refines the latest model* instead of
starting from scratch. Corrected labels feed the next fine-tune.
"""

from __future__ import annotations

from pathlib import Path

from flashmask.config import paths
from flashmask.labeling.region import Region


def should_use_model(detector_path: str | Path | None) -> bool:
    """True when a trained detector exists and should drive pre-labeling."""
    return bool(detector_path) and Path(detector_path).exists()


def prelabel(
    image_path: str | Path,
    *,
    detector_path: str | Path | None = None,
    doctr_conf: float = 0.5,
    line_tol: int = 10,
    horiz_tol: int = 10,
) -> tuple[list[Region], str]:
    """Pre-label one image. Returns ``(regions, source)`` with source 'model'|'doctr'.

    Uses the detector at ``detector_path`` (default ``models/detector.onnx``) when
    present; otherwise falls back to DocTR OCR + word clustering.
    """
    detector_path = detector_path if detector_path is not None else paths.models / "detector.onnx"
    if should_use_model(detector_path):
        return _prelabel_with_model(image_path, detector_path), "model"
    return _prelabel_with_doctr(image_path, doctr_conf, line_tol, horiz_tol), "doctr"


def _prelabel_with_model(image_path: str | Path, detector_path: str | Path) -> list[Region]:
    import cv2

    from flashmask.inference.pipeline import TextMaskPipeline

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    # Detector-only (no classifier/OCR gate): for labeling we want every candidate
    # box to refine, i.e. maximise recall and let the annotator prune.
    pipe = TextMaskPipeline(detector_path, classifier_path=None, use_ocr=False)
    return [
        Region(x=b["x"], y=b["y"], width=b["width"], height=b["height"], id=f"r{i}")
        for i, b in enumerate(pipe.detect(img), start=1)
    ]


def _prelabel_with_doctr(
    image_path: str | Path, conf: float, line_tol: int, horiz_tol: int
) -> list[Region]:
    from flashmask.labeling.doctr_assist import cluster_words, run_doctr

    words = run_doctr(str(image_path), conf)
    return cluster_words(words, line_tol, horiz_tol)
