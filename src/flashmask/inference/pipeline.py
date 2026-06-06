"""End-to-end text-region detection & masking pipeline (ONNX, CPU-friendly).

Stages: YOLO detector -> (optional) ResNet18 text/non-text filter -> OCR -> regex
word filter -> mask. Designed to run from ONNX exports with no torch dependency,
so it powers both the Gradio demo and the FastAPI service.

If the classifier weights are absent (they are not committed — see
``models/README.md``), the classifier gate is skipped and a warning is logged;
the detector + OCR + regex path still runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from flashmask.inference.postprocess import (
    filter_nested_boxes,
    is_word_like,
    letterbox,
    nms,
)
from flashmask.labeling.region import Region

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class PipelineConfig:
    """Thresholds for the detection/filter pipeline (tuned defaults from the run)."""

    img_size: int = 960
    conf_threshold: float = 0.20
    nms_iou: float = 0.10
    pad_pct_h: float = 0.05
    pad_pct_v: float = 0.125
    cls_input_size: int = 256
    cls_conf_thresh: float = 0.50
    ocr_override_conf_thresh: float = 0.60
    large_box_area_pct: float = 0.008
    upscale_height_thresh: int = 64
    target_height: int = 48


class TextMaskPipeline:
    """Detect text regions in a diagram image and (optionally) mask them."""

    def __init__(
        self,
        detector_path: str | Path,
        classifier_path: str | Path | None = None,
        config: PipelineConfig | None = None,
        *,
        use_ocr: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.cfg = config or PipelineConfig()
        providers = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(detector_path), providers=providers)
        self.det_in = self.det.get_inputs()[0].name

        self.cls = None
        if classifier_path and Path(classifier_path).exists():
            self.cls = ort.InferenceSession(str(classifier_path), providers=providers)
            self.cls_in = self.cls.get_inputs()[0].name
        elif classifier_path:
            logger.warning(
                "Classifier weights not found at %s; skipping the filter.", classifier_path
            )

        self.ocr = None
        if use_ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self.ocr = RapidOCR()
            except Exception as exc:  # noqa: BLE001
                logger.warning("RapidOCR unavailable (%s); text filtering disabled.", exc)

    # ── detection ────────────────────────────────────────────────────────────
    def detect(self, img: np.ndarray) -> list[dict]:
        """Run the detector and return padded, NMS-ed, nested-filtered boxes."""
        h0, w0 = img.shape[:2]
        padded, r, (dx, dy) = letterbox(img, (self.cfg.img_size, self.cfg.img_size), scaleup=False)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        out = np.asarray(self.det.run(None, {self.det_in: blob})[0])
        if out.ndim == 3:
            out = out[0]

        boxes, scores = [], []
        for det in out:
            x1, y1, x2, y2, conf = det[:5]
            x0, y0 = (x1 - dx) / r, (y1 - dy) / r
            w, h = (x2 - dx) / r - x0, (y2 - dy) / r - y0
            boxes.append((int(x0), int(y0), int(w), int(h)))
            scores.append(float(conf))

        kept = nms(boxes, scores, self.cfg.conf_threshold, self.cfg.nms_iou)
        preds = []
        for k in kept:
            x, y, w, h = boxes[k]
            px, py = w * self.cfg.pad_pct_h, h * self.cfg.pad_pct_v
            x1 = int(max(0, x - px))
            y1 = int(max(0, y - py))
            x2 = int(min(w0, x + w + px))
            y2 = int(min(h0, y + h + py))
            preds.append(
                {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "det_conf": scores[k]}
            )

        nested = filter_nested_boxes([(p["x"], p["y"], p["width"], p["height"]) for p in preds])
        nested_set = set(nested)
        return [p for p in preds if (p["x"], p["y"], p["width"], p["height"]) in nested_set]

    # ── classifier ───────────────────────────────────────────────────────────
    def classify(self, crop: np.ndarray) -> float:
        """Probability that ``crop`` is text (1.0 if no classifier is loaded)."""
        if self.cls is None:
            return 1.0
        padded, _, _ = letterbox(crop, (self.cfg.cls_input_size, self.cfg.cls_input_size))
        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((x - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        logits = np.asarray(self.cls.run(None, {self.cls_in: x})[0])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return float((exp / exp.sum(axis=1, keepdims=True))[0, 1])

    # ── OCR ──────────────────────────────────────────────────────────────────
    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        if self.ocr is None or crop.size == 0:
            return "", 0.0
        if crop.shape[0] < self.cfg.upscale_height_thresh:
            scale = self.cfg.target_height / crop.shape[0]
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), self.cfg.target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        result, _ = self.ocr(crop, use_det=False, use_cls=False, use_rec=True)
        if not result:
            return "", 0.0
        text, score = result[0][0], result[0][1]
        return text, float(score) * 100

    # ── full gauntlet ────────────────────────────────────────────────────────
    def run(self, img: np.ndarray) -> list[Region]:
        """Return accepted text regions after detection, classification and filtering."""
        cfg, img_area = self.cfg, img.shape[0] * img.shape[1]
        accepted: list[Region] = []
        for i, b in enumerate(self.detect(img), start=1):
            crop = img[b["y"] : b["y"] + b["height"], b["x"] : b["x"] + b["width"]]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            cls_conf = self.classify(crop)
            is_large = (b["width"] * b["height"]) / img_area > cfg.large_box_area_pct

            if is_large:
                if cls_conf < cfg.cls_conf_thresh:
                    continue
                text, _ = self.recognize(crop)
            else:
                if cls_conf < cfg.ocr_override_conf_thresh:
                    continue
                text, ocr_conf = self.recognize(crop)
                if self.ocr is not None and not is_word_like(text, ocr_conf):
                    continue
            accepted.append(
                Region(
                    x=b["x"], y=b["y"], width=b["width"], height=b["height"], id=f"r{i}", text=text
                )
            )
        return accepted

    @staticmethod
    def mask(
        img: np.ndarray, regions: list[Region], color: tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """Return a copy of ``img`` with each region filled (text obfuscated)."""
        out = img.copy()
        for r in regions:
            cv2.rectangle(out, (r.x, r.y), (r.x + r.width, r.y + r.height), color, thickness=-1)
        return out
