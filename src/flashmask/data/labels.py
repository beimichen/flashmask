"""Label utilities: review-metadata <-> YOLO format, cleaning, and cropping.

The labeling tool stores annotations as a JSON map::

    { "<image_path>": {"status": ..., "boxes": [{"x","y","width","height","visible",...}]} }

These helpers are pure functions (no globals, no hardcoded paths) so they are
unit-tested directly — the conversion is the part most worth getting right.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2

Metadata = Mapping[str, Mapping[str, Any]]


def box_to_yolo_line(box: Mapping[str, float], img_w: int, img_h: int, class_id: int = 0) -> str:
    """Convert one ``{x, y, width, height}`` pixel box to a normalized YOLO line.

    Raises ``ValueError`` on non-positive image dimensions.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img_w}x{img_h}")
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def visible_boxes(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Boxes from a metadata entry, dropping any explicitly marked invisible."""
    return [b for b in entry.get("boxes", []) if b.get("visible", True)]


def clean_metadata(metadata: Metadata, min_width: int = 10, min_height: int = 10) -> dict[str, Any]:
    """Drop boxes smaller than ``min_width`` x ``min_height`` from every entry.

    Image entries are kept even if they end up with no boxes (negatives are
    useful training signal).
    """
    cleaned: dict[str, Any] = {}
    for img_path, entry in metadata.items():
        kept = [
            b
            for b in entry.get("boxes", [])
            if b.get("width", 0) >= min_width and b.get("height", 0) >= min_height
        ]
        cleaned[img_path] = {**entry, "boxes": kept}
    return cleaned


def write_yolo_labels(
    metadata: Metadata,
    image_paths: Iterable[str],
    labels_dir: Path,
    *,
    read_image=cv2.imread,
) -> int:
    """Write one ``<stem>.txt`` YOLO label file per image. Returns box count.

    ``read_image`` is injectable so tests can avoid touching the filesystem.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for img_path in image_paths:
        entry = metadata[img_path]
        im = read_image(str(img_path))
        if im is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h, w = im.shape[:2]
        lines = [box_to_yolo_line(b, w, h) for b in visible_boxes(entry)]
        (labels_dir / f"{Path(img_path).stem}.txt").write_text("\n".join(lines))
        total += len(lines)
    return total


def crop_regions(image, boxes: Iterable[Mapping[str, float]], pad_frac: float = 0.1) -> list[Any]:
    """Crop each box from ``image`` with fractional padding (for classifier data)."""
    h_img, w_img = image.shape[:2]
    crops = []
    for box in boxes:
        x, y = int(box.get("x", 0)), int(box.get("y", 0))
        w, h = int(box.get("width", 0)), int(box.get("height", 0))
        px, py = round(pad_frac * w), round(pad_frac * h)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(w_img, x + w + px), min(h_img, y + h + py)
        crops.append(image[y1:y2, x1:x2])
    return crops


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_metadata(metadata: Metadata, path: Path) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
