"""Build an Ultralytics-style YOLO dataset from review metadata.

Produces the standard layout::

    <out>/images/{train,val,test}/...   <out>/labels/{train,val,test}/...   <out>/data.yaml

using the leakage-safe :func:`flashmask.data.splits.split_by_scene` so that
augmented variants and multi-page documents never straddle splits.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml

from flashmask.config import paths as project_paths
from flashmask.data.labels import load_metadata, write_yolo_labels
from flashmask.data.splits import split_by_scene


def _resolve(image_path: str) -> str:
    """Resolve a metadata key: absolute paths as-is, relative against the repo root."""
    p = Path(image_path)
    return str(p if p.is_absolute() else project_paths.root / p)


def build_yolo_dataset(
    metadata_path: Path,
    out_dir: Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    class_names: list[str] | None = None,
) -> Path:
    """Materialize images + YOLO labels + ``data.yaml``. Returns the data.yaml path."""
    metadata: dict[str, Any] = load_metadata(metadata_path)
    out_dir = Path(out_dir)
    splits = split_by_scene(metadata.keys(), ratios, metadata=metadata)
    # Re-key metadata by resolved absolute path so committed (relative) sample
    # metadata and absolute real-data metadata both work on any machine.
    abs_metadata = {_resolve(k): v for k, v in metadata.items()}

    for split, image_paths in splits.items():
        if not image_paths:
            continue
        img_out = out_dir / "images" / split
        img_out.mkdir(parents=True, exist_ok=True)
        abs_paths = [_resolve(p) for p in image_paths]
        for src in abs_paths:
            dst = img_out / Path(src).name
            if not dst.exists():
                shutil.copy(src, dst)
        write_yolo_labels(
            abs_metadata, abs_paths, out_dir / "labels" / split, read_image=cv2.imread
        )

    data_yaml = out_dir / "data.yaml"
    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": class_names or ["text"],
    }
    if splits.get("test"):
        cfg["test"] = "images/test"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return data_yaml
