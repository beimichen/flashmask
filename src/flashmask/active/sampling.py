"""Uncertainty acquisition functions for active learning.

Pure, model-free scoring so it unit-tests directly. Given the per-detection
confidences the detector produced on an image (and optionally the text
classifier's probabilities for the same boxes), return a scalar "how informative
is this image to label next" — higher means more uncertain.

Strategies
----------
- ``entropy``      mean binary entropy of box confidences. Maximal when the
                   detector sits at p≈0.5 on its boxes (classic least-confidence
                   acquisition). Default; needs no classifier.
- ``margin``       share of boxes whose confidence falls in a borderline band
                   (e.g. 0.20–0.50) — images full of "maybe text" detections.
- ``disagreement`` mean absolute gap between the detector's confidence and the
                   text classifier's probability on each box. Surfaces images
                   where the two heads contradict each other. Needs a classifier.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-9
STRATEGIES = ("entropy", "margin", "disagreement")


def binary_entropy(p: float) -> float:
    """Binary Shannon entropy in bits: 1.0 at p=0.5, ~0 at p∈{0,1}."""
    p = min(max(p, _EPS), 1 - _EPS)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def score_detections(
    confidences: Sequence[float],
    cls_probs: Sequence[float] | None = None,
    *,
    strategy: str = "entropy",
    band: tuple[float, float] = (0.20, 0.50),
    empty_image_score: float = 0.0,
) -> float:
    """Score one image's informativeness from its detection confidences.

    ``confidences`` and ``cls_probs`` (when given) must be aligned per box and
    already filtered to the detections you care about. An image with no
    detections returns ``empty_image_score`` (default 0.0, so blank/no-text
    images don't dominate; raise it to deliberately surface possible
    false-negatives).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if not confidences:
        return empty_image_score

    if strategy == "entropy":
        return _mean([binary_entropy(c) for c in confidences])
    if strategy == "margin":
        lo, hi = band
        return _mean([1.0 if lo <= c <= hi else 0.0 for c in confidences])
    # disagreement
    if cls_probs is None or len(cls_probs) != len(confidences):
        raise ValueError("disagreement strategy needs classifier probs aligned with confidences")
    return _mean([abs(c - p) for c, p in zip(confidences, cls_probs)])
