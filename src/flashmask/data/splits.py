"""Leakage-safe train/val/test splitting.

Two failure modes inflate reported metrics on this dataset, and both are avoided
here:

1. **Near-duplicate leakage.** Synthetic generation produces several augmented
   variants of the same background (``bg_orig``, ``bg_jitter``, ``bg_blur`` ...),
   and a scanned PDF yields many similar pages. If variants of one source land in
   both train and val, val mAP is optimistically biased. We split by *scene key*
   (whole groups go to one split), never by individual image.

2. **Synthetic contamination of the test set.** Headline numbers must reflect
   real-world performance, so the held-out test split is drawn from **real images
   only**; synthetic data is confined to train/val.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

# Suffixes appended by the synthetic generator / augmentation passes.
_VARIANT_SUFFIX = re.compile(r"_(orig|jitter|blur|squish|aug\d*|var\d*)(_\d+)?$", re.IGNORECASE)
# Trailing page / index markers, e.g. "_page_3", "_p2", "-12".
_PAGE_SUFFIX = re.compile(r"[_-](page[_-]?\d+|p\d+|\d+)$", re.IGNORECASE)


def scene_key(image_path: str) -> str:
    """Group key identifying the *source scene* an image was derived from.

    Variants and page indices are stripped so all derivatives of one source map
    to the same key.
    """
    stem = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    prev = None
    while prev != stem:
        prev = stem
        stem = _VARIANT_SUFFIX.sub("", stem)
        stem = _PAGE_SUFFIX.sub("", stem)
    return stem.strip(" _-").lower()


def is_synthetic(image_path: str, entry: Mapping[str, Any] | None = None) -> bool:
    """Heuristic: synthetic images come from the generator (tagged path/metadata)."""
    if entry is not None and entry.get("source") == "synthetic":
        return True
    p = image_path.lower()
    return "synthetic" in p or bool(_VARIANT_SUFFIX.search(image_path.rsplit(".", 1)[0]))


def _bucket(key: str) -> float:
    """Deterministic value in [0, 1) for a key — stable across runs and machines."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_by_scene(
    image_paths: Iterable[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
    real_only_test: bool = True,
) -> dict[str, list[str]]:
    """Partition images into train/val/test with no scene leakage across splits.

    All images sharing a :func:`scene_key` are assigned to the same split via a
    hash bucket (deterministic, seed-free, order-independent). When
    ``real_only_test`` is set, synthetic scenes are never placed in the test
    split — they are redirected to train.
    """
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    train_r, val_r, _ = ratios

    groups: dict[str, list[str]] = defaultdict(list)
    for p in image_paths:
        groups[scene_key(p)].append(p)

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key, members in groups.items():
        b = _bucket(key)
        if b < train_r:
            split = "train"
        elif b < train_r + val_r:
            split = "val"
        else:
            split = "test"

        if split == "test" and real_only_test:
            entry = (metadata or {}).get(members[0])
            if any(is_synthetic(m, entry) for m in members):
                split = "train"
        out[split].extend(members)

    return out
