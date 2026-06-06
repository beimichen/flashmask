"""Tests for the union-find word clustering used in OCR-assisted labeling."""

from __future__ import annotations

from flashmask.labeling.doctr_assist import cluster_words


def _word(text, x1, y1, x2, y2):
    return {"text": text, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_adjacent_words_merge_into_one_region():
    words = [_word("Hello", 0, 0, 50, 20), _word("World", 55, 0, 110, 20)]
    regions = cluster_words(words, line_tol=10, horiz_tol=10)
    assert len(regions) == 1
    assert regions[0].text == "Hello World"
    assert regions[0].xyxy == (0, 0, 110, 20)


def test_distant_words_stay_separate():
    words = [_word("Top", 0, 0, 40, 20), _word("Bottom", 0, 400, 60, 420)]
    regions = cluster_words(words, line_tol=10, horiz_tol=10)
    assert len(regions) == 2


def test_empty_input_returns_no_regions():
    assert cluster_words([], 10, 10) == []
