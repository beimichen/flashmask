# Dataset Card — flashmask text regions

## Overview

A single-class **text-region detection** dataset over diagram/figure/document
images, used to train the flashmask detector. Each image has axis-aligned boxes
around regions of printed text.

## Composition

| Subset | ~Size | Source | Labels |
|--------|-------|--------|--------|
| Real | ~1,000 images, ~12.5k boxes | Diagrams/figures from scientific PDFs (arXiv, PMC, CORE) and slide/screenshot captures | DocTR-assisted + manual correction |
| Synthetic | generated on demand | Vocabulary text rendered onto real backgrounds | Exact (generation-time ground truth) |

Average ≈ 12–13 text boxes per real image.

## Collection & labeling

1. **Acquire** PDFs (`flashmask.data.scrape`) and render pages to images
   (`flashmask.data.pdf_to_image`).
2. **Pre-label** with DocTR OCR, clustering words into regions via union-find
   (`flashmask.labeling.doctr_assist`).
3. **Correct** boxes in the Streamlit tool (`apps/label_tool.py`); annotations are
   stored as a review-metadata JSON:
   ```json
   {"<image>": {"status": "...", "boxes": [{"x","y","width","height","visible",...}]}}
   ```
4. **Clean** (`flashmask.data.labels.clean_metadata` drops sub-10px boxes) and
   **convert** to YOLO format with leakage-safe splits
   (`flashmask.data.dataset` + `flashmask.data.splits`).

## Splits

`flashmask.data.splits.split_by_scene` partitions by **scene key** (all augmented
variants / pages of one source stay in the same split) and keeps the **test split
real-only**, so reported metrics are not inflated by near-duplicate leakage or
synthetic contamination.

## Availability & licensing

- The image corpus is **not redistributed** here — source PDFs carry their own
  licenses and many captures are not ours to share. The repo ships a tiny set of
  **synthetic placeholder** images (`just sample`) so the pipeline/demo/tests run
  on a fresh clone.
- Reproduce the real corpus with the scraping + labeling pipeline above, or point
  the dataset builder at your own labelled images.

## Known biases

Skewed toward English-language STEM diagrams and two-column paper layouts;
under-represents handwriting, dense tables, and non-Latin scripts.
