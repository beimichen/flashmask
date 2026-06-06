"""A self-contained text-region box.

This replaces the ``Region`` type that previously lived in the (private) flashcard
application backend, so the labeling tools and dataset code have no external app
dependency. Coordinates are pixel values in the source image's frame.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Region:
    """An axis-aligned text bounding box, in absolute pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    id: str | None = None
    text: str = ""
    group: str | None = None
    visible: bool = True

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return ``(x1, y1, x2, y2)`` corners."""
        return self.x, self.y, self.x + self.width, self.y + self.height

    def to_yolo(self, img_w: int, img_h: int, class_id: int = 0) -> str:
        """Serialize to a YOLO label line: ``class cx cy w h`` (normalized 0-1)."""
        cx = (self.x + self.width / 2) / img_w
        cy = (self.y + self.height / 2) / img_h
        nw = self.width / img_w
        nh = self.height / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "group": self.group,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        """Build a Region from a metadata dict, tolerating extra/missing keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
