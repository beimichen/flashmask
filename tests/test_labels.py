"""Tests for label conversion, cleaning, and the Region model."""

from __future__ import annotations

import pytest

from flashmask.data.labels import box_to_yolo_line, clean_metadata, visible_boxes
from flashmask.labeling.region import Region


def test_box_to_yolo_line_normalizes_center_and_size():
    line = box_to_yolo_line({"x": 10, "y": 20, "width": 30, "height": 40}, 100, 200)
    assert line == "0 0.250000 0.200000 0.300000 0.200000"


def test_box_to_yolo_line_rejects_bad_dims():
    with pytest.raises(ValueError):
        box_to_yolo_line({"x": 0, "y": 0, "width": 1, "height": 1}, 0, 100)


def test_clean_metadata_drops_small_boxes_keeps_entry():
    meta = {
        "a.png": {"boxes": [{"width": 5, "height": 5}, {"width": 40, "height": 40}]},
    }
    cleaned = clean_metadata(meta, min_width=10, min_height=10)
    assert len(cleaned["a.png"]["boxes"]) == 1
    assert cleaned["a.png"]["boxes"][0]["width"] == 40


def test_visible_boxes_filters_invisible():
    entry = {"boxes": [{"id": 1, "visible": True}, {"id": 2, "visible": False}, {"id": 3}]}
    assert [b["id"] for b in visible_boxes(entry)] == [1, 3]


def test_region_to_yolo_matches_box_helper():
    r = Region(x=10, y=20, width=30, height=40)
    assert r.to_yolo(100, 200) == box_to_yolo_line(r.to_dict(), 100, 200)


def test_region_dict_roundtrip_ignores_unknown_keys():
    data = {"x": 1, "y": 2, "width": 3, "height": 4, "text": "hi", "bogus": "ignored"}
    r = Region.from_dict(data)
    assert (r.x, r.y, r.width, r.height, r.text) == (1, 2, 3, 4, "hi")
    assert r.area == 12
