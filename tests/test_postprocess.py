"""Unit tests for the pure post-processing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from flashmask.inference.postprocess import (
    filter_nested_boxes,
    is_word_like,
    letterbox,
    nms,
)


def test_letterbox_returns_requested_shape_and_ratio():
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    out, ratio, (dw, dh) = letterbox(img, (640, 640))
    assert out.shape[:2] == (640, 640)
    assert ratio == pytest.approx(640 / 600)  # limited by the wider dimension
    assert dh > dw  # padding added on the (shorter) vertical axis


def test_letterbox_no_upscale_keeps_ratio_below_one():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, ratio, _ = letterbox(img, (640, 640), scaleup=False)
    assert ratio == 1.0


def test_nms_suppresses_overlapping_lower_score_box():
    boxes = [(0, 0, 100, 100), (5, 5, 100, 100)]
    keep = nms(boxes, [0.9, 0.8], conf_thresh=0.1, iou_thresh=0.5)
    assert keep == [0]


def test_nms_empty_input():
    assert nms([], [], 0.1, 0.5) == []


def test_filter_nested_boxes_drops_inner_box():
    outer = (0, 0, 100, 100)
    inner = (10, 10, 20, 20)
    assert filter_nested_boxes([outer, inner]) == [outer]


@pytest.mark.parametrize(
    "text,conf,expected",
    [
        ("Mitochondrion", 99.0, True),
        ("DNA", 99.0, True),  # 3-letter all-caps acronym is kept
        ("x", 99.0, False),  # too short
        ("12", 99.0, False),  # digits only
        ("5 km", 99.0, False),  # measurement
        ("a = b", 99.0, False),  # short equation
        ("cat", 50.0, False),  # 3 letters below the OCR confidence floor
        ("cat", 95.0, True),  # same word, confident OCR
    ],
)
def test_is_word_like(text, conf, expected):
    assert is_word_like(text, conf) is expected
