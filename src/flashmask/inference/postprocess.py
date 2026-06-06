"""Pure post-processing helpers for the detection pipeline.

Kept free of model/session state so they unit-test directly: letterbox resize,
NMS, nested-box suppression, and the heuristic OCR text filter that rejects
detections which are not real words (measurements, chemical formulae, equations,
stray symbols). These heuristics are why the pipeline masks *words* rather than
every detected glyph.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

# A detection box as (x, y, w, h) in pixels.
Box = tuple[int, int, int, int]

MIN_ALPHANUM_LENGTH = 2
THREE_LETTER_OCR_CONF_THRESH = 85.0

_MEASUREMENT_RE = re.compile(r"^\d+(?:[.-]\d+)?\s*(?:nm|mm|cm|m|km|µm|g|kg|ml|l)$", re.IGNORECASE)
_CHEM_RE = re.compile(r"^([A-Z][a-z]?\d*)+$")
_EQUATION_RE = re.compile(r"^\s*[a-zA-Z]\s*[+\-=*/<>]+\s*[a-zA-Z0-9]\s*$")


def letterbox(
    img: np.ndarray,
    new_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize+pad to ``new_size`` keeping aspect ratio. Returns (img, ratio, (dw, dh))."""
    h, w = img.shape[:2]
    new_w, new_h = new_size
    r = min(new_w / w, new_h / h)
    if not scaleup:
        r = min(r, 1.0)
    unpad_w, unpad_h = round(w * r), round(h * r)
    dw, dh = (new_w - unpad_w) / 2, (new_h - unpad_h) / 2
    if (w, h) != (unpad_w, unpad_h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def nms(boxes: list[Box], scores: list[float], conf_thresh: float, iou_thresh: float) -> list[int]:
    """Non-maximum suppression. Returns kept indices (empty if none survive)."""
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes([list(b) for b in boxes], list(scores), conf_thresh, iou_thresh)
    if keep is None or len(keep) == 0:
        return []
    return [int(i) for i in np.array(keep).flatten()]


def filter_nested_boxes(boxes: list[Box]) -> list[Box]:
    """Drop any box fully contained inside a strictly larger one."""
    kept: list[Box] = []
    for i, bi in enumerate(boxes):
        xi, yi, wi, hi = bi
        nested = False
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            xj, yj, wj, hj = bj
            inside = xi >= xj and yi >= yj and xi + wi <= xj + wj and yi + hi <= yj + hj
            if inside and wi * hi < wj * hj:
                nested = True
                break
        if not nested:
            kept.append(bi)
    return kept


def is_word_like(text: str, ocr_conf: float) -> bool:
    """True if ``text`` looks like a real word worth masking (not symbols/formulae)."""
    return not _is_invalid_text(text, ocr_conf)


def _is_invalid_text(text: str, ocr_conf: float) -> bool:
    trimmed = text.strip()
    alnum = "".join(filter(str.isalnum, trimmed))

    if len(alnum) < MIN_ALPHANUM_LENGTH:
        return True
    if not any(c.isalpha() for c in trimmed) and any(c.isdigit() for c in trimmed):
        return True
    if _MEASUREMENT_RE.fullmatch(trimmed):
        return True
    # A chemical-formula pattern is only kept if it's a 3+ letter all-caps acronym.
    if _CHEM_RE.fullmatch(alnum) and not (alnum.isupper() and len(alnum) >= 3):
        return True
    if len(alnum) <= 4 and _EQUATION_RE.fullmatch(trimmed):
        return True
    return len(alnum) == 3 and ocr_conf < THREE_LETTER_OCR_CONF_THRESH
