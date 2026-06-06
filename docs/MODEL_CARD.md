# Model Card — flashmask text-region detector

## Model summary

A single-class object detector that localizes **text regions** in diagram,
textbook and scientific-paper images so the text can be masked to generate study
flashcards. Built on **Ultralytics YOLO (small, 960px)** and trained in two
phases (synthetic pretraining → real-data fine-tuning). A lightweight ResNet18
text/non-text classifier and an OCR + regex filter sit downstream to suppress
false positives at inference time.

- **Task:** object detection (1 class: `text`)
- **Architecture:** YOLOv5s-u backbone, 960×960 input
- **Frameworks:** PyTorch / Ultralytics (training), ONNX Runtime (inference)
- **License:** MIT

## Intended use

- **In scope:** detecting printed/rendered text regions in diagrams, figures,
  slides, and document scans for downstream masking/obfuscation.
- **Out of scope:** OCR/text *recognition* (handled separately by RapidOCR),
  handwriting, scene text in natural photos, and any use requiring guaranteed
  redaction (see Limitations — this is best-effort masking, not a security
  control).

## Training data

- **Real data:** ~1,000 human-labelled diagram images (OCR-assisted labeling with
  DocTR, then manual correction). See the [Dataset Card](DATASET_CARD.md).
- **Synthetic data:** vocabulary text rendered onto document/figure backgrounds in
  varied fonts, sizes, rotations and photometric/geometric augmentations, used for
  phase-1 pretraining.
- Synthetic data is confined to train/val; real-world performance is measured on
  held-out **real** images only.

## Training procedure

Two-phase schedule (exact hyperparameters in
[`configs/train/`](../configs/train/)):

1. **Phase 1 — pretrain on synthetic** with heavy augmentation (mosaic, mixup,
   HSV, random erasing) to learn the general appearance of text regions.
2. **Phase 2 — fine-tune on real** with light augmentation and a frozen stem
   (SGD, lr0 0.01, 100 epochs, early-stopping patience 30, AMP).

## Evaluation results

Measured on the held-out **real** validation split (best checkpoint, epoch 87),
at 960px:

| Metric | Value |
|--------|-------|
| Precision | **0.917** |
| Recall | **0.883** |
| mAP@0.50 | **0.929** |
| mAP@0.50:0.95 | **0.717** |

Curves and the confusion matrix are in [`reports/figures/`](../reports/figures/).

> Provenance: these are the actual metrics from the surviving two-phase run
> (`reports/results.csv`). The original run used an 80/20 real train/val split;
> the current code additionally carves out a **real-only test split**
> (`flashmask.data.splits`) for future runs to avoid val-set optimism. Numbers
> from a fresh training run will vary slightly (PyTorch is not bit-reproducible
> across hardware).

### Recommended inference settings

`imgsz=960`, `conf≈0.20–0.40`, `nms_iou≈0.10`. Thresholds are intentionally
permissive: over-masking a little is cheaper than leaking answer text on a
flashcard. Defaults live in
[`configs/inference/onnx_pipeline.yaml`](../configs/inference/onnx_pipeline.yaml).

## Limitations & ethical considerations

- **Not a redaction guarantee.** Missed detections leave text visible; this is a
  study aid, not a security/privacy control.
- **Sim-to-real gap.** Synthetic pretraining helps but does not perfectly match
  real diagram typography/layout.
- **Domain skew.** Training data skews toward STEM diagrams and papers; expect
  weaker performance on very different layouts (handwriting, dense tables, non-
  Latin scripts).
- **Downstream OCR errors** can cause the regex filter to wrongly keep/drop a box.
