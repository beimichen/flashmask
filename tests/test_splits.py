"""Tests for leakage-safe, group-aware dataset splitting."""

from __future__ import annotations

from flashmask.data.splits import scene_key, split_by_scene


def test_scene_key_strips_variants_and_pages():
    assert scene_key("/x/bgA_orig_0.png") == "bga"
    assert scene_key("/x/bgA_jitter_12.png") == "bga"
    assert scene_key("/x/docB_page_3.png") == "docb"
    assert scene_key("docB_page_10.png") == "docb"


def test_no_scene_leaks_across_splits():
    paths = [
        "bgA_orig_0.png",
        "bgA_jitter_1.png",
        "bgA_blur_2.png",
        "docB_page_1.png",
        "docB_page_2.png",
        "docB_page_3.png",
        "photoC.png",
        "photoD.png",
        "photoE.png",
        "photoF.png",
    ]
    splits = split_by_scene(paths, (0.6, 0.2, 0.2))

    location = {}
    for split, members in splits.items():
        for m in members:
            location[scene_key(m)] = location.get(scene_key(m), set()) | {split}
    # every scene key lands in exactly one split
    assert all(len(s) == 1 for s in location.values())


def test_synthetic_scenes_excluded_from_test_split():
    # All synthetic (variant-tagged) — none may appear in the real-only test split.
    paths = [f"bg{i}_jitter_{i}.png" for i in range(40)]
    splits = split_by_scene(paths, (0.6, 0.2, 0.2), real_only_test=True)
    assert splits["test"] == []


def test_split_is_deterministic():
    paths = [f"img_{i}.png" for i in range(50)]
    assert split_by_scene(paths) == split_by_scene(paths)
