"""Tests for active-learning uncertainty scoring."""

from __future__ import annotations

import math

import pytest

from flashmask.active.sampling import binary_entropy, score_detections


def test_binary_entropy_peaks_at_half():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-6)
    assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-6)
    assert binary_entropy(0.1) == pytest.approx(binary_entropy(0.9))  # symmetric


def test_entropy_strategy_ranks_uncertain_above_confident():
    confident = score_detections([0.97, 0.95, 0.98], strategy="entropy")
    uncertain = score_detections([0.5, 0.45, 0.55], strategy="entropy")
    assert uncertain > confident
    assert uncertain == pytest.approx(1.0, abs=0.05)


def test_empty_image_returns_empty_score():
    assert score_detections([], strategy="entropy") == 0.0
    assert score_detections([], strategy="entropy", empty_image_score=0.7) == 0.7


def test_margin_strategy_counts_borderline_fraction():
    # band default (0.20, 0.50): two of three boxes are borderline.
    assert score_detections([0.3, 0.45, 0.9], strategy="margin") == pytest.approx(2 / 3)


def test_disagreement_strategy_measures_head_gap():
    # both boxes: detector 0.9 vs classifier 0.1 -> gap 0.8
    score = score_detections([0.9, 0.9], [0.1, 0.1], strategy="disagreement")
    assert score == pytest.approx(0.8)
    # agreeing heads -> ~0 disagreement (mean of |0.9-0.88|, |0.8-0.82| = 0.02)
    assert score_detections([0.9, 0.8], [0.88, 0.82], strategy="disagreement") == pytest.approx(
        0.02
    )


def test_disagreement_requires_aligned_cls_probs():
    with pytest.raises(ValueError):
        score_detections([0.9, 0.8], None, strategy="disagreement")
    with pytest.raises(ValueError):
        score_detections([0.9, 0.8], [0.1], strategy="disagreement")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        score_detections([0.5], strategy="bogus")


def test_entropy_value_matches_formula():
    # single box at 0.9 -> entropy = -(.9 log2 .9 + .1 log2 .1)
    expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    assert score_detections([0.9], strategy="entropy") == pytest.approx(expected)
