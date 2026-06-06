"""OCR-assisted pre-labeling: turn DocTR word detections into region boxes.

The annotator runs this to get a first-pass set of boxes, then corrects them in
the Streamlit tool — far faster than drawing every box by hand. The clustering
(union-find over adjacent words) is a pure function and is unit-tested; only
:func:`run_doctr` needs the optional ``[label]`` extra.
"""

from __future__ import annotations

import re
from typing import Any

from flashmask.labeling.region import Region

Word = dict[str, Any]  # {"text", "x1", "y1", "x2", "y2"}


def run_doctr(image_path: str, conf_threshold: float = 0.5) -> list[Word]:
    """Detect words with DocTR; return pixel-space word boxes above the threshold."""
    import cv2
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True)
    result = model(DocumentFile.from_images(image_path))
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    words: list[Word] = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                txt = word.value.strip()
                if word.confidence < conf_threshold:
                    continue
                if re.fullmatch(r"[\d\W]+", txt) and re.search(r"\d", txt):
                    continue  # pure numbers/symbols are not flashcard-worthy text
                (xmin, ymin), (xmax, ymax) = word.geometry
                words.append(
                    {
                        "text": txt,
                        "x1": int(xmin * w),
                        "y1": int(ymin * h),
                        "x2": int(xmax * w),
                        "y2": int(ymax * h),
                    }
                )
    return words


def cluster_words(words: list[Word], line_tol: int = 10, horiz_tol: int = 10) -> list[Region]:
    """Merge adjacent words into region boxes via union-find. Pure / testable."""
    parent = list(range(len(words)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, a in enumerate(words):
        for j in range(i + 1, len(words)):
            b = words[j]
            vert_overlap = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
            horiz_gap = max(0, b["x1"] - a["x2"], a["x1"] - b["x2"])
            if vert_overlap > 0 and horiz_gap <= horiz_tol:
                union(i, j)
                continue
            horiz_overlap = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
            vert_gap = max(0, b["y1"] - a["y2"], a["y1"] - b["y2"])
            if horiz_overlap > 0 and vert_gap <= line_tol:
                union(i, j)

    groups: dict[int, list[Word]] = {}
    for i in range(len(words)):
        groups.setdefault(find(i), []).append(words[i])

    regions: list[Region] = []
    for idx, cluster in enumerate(groups.values(), start=1):
        cluster.sort(key=lambda w: w["x1"])
        x1 = min(w["x1"] for w in cluster)
        y1 = min(w["y1"] for w in cluster)
        x2 = max(w["x2"] for w in cluster)
        y2 = max(w["y2"] for w in cluster)
        regions.append(
            Region(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
                id=f"r{idx}",
                text=" ".join(w["text"] for w in cluster),
            )
        )
    return regions
