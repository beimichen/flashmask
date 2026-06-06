# Models

Trained weights are **not committed** to this repository (they are large binary
artifacts and don't belong in git history). This directory is otherwise empty by
design — `.gitignore` keeps `*.pt` / `*.onnx` out.

## What goes here

| File | Produced by | Used by |
|------|-------------|---------|
| `best.pt` | `flashmask.modeling.train_detector` (phase 2) | evaluation, ONNX export |
| `detector.onnx` | `flashmask export -w models/best.pt` | inference pipeline, API, demo |
| `text_classifier.onnx` | `flashmask train-classifier` | false-positive filter (optional) |

## How to obtain weights

**Option A — train them** (needs the dataset and the `[train]` extra):

```bash
just train-pretrain                 # phase 1 on synthetic data
just train-finetune                 # phase 2 on real data
just export runs/finetune_real/weights/best.pt   # -> models/detector.onnx
```

**Option B — drop in your own.** Place a YOLO `best.pt` here and run
`flashmask export -w models/best.pt`, or copy an existing `detector.onnx` into
this folder. The demo (`just demo`) and API (`just serve`) auto-detect it; until
then they run in a no-model mode that explains what's missing.

> To make the public Gradio/Hugging Face Spaces demo fully live, host
> `detector.onnx` on the Hugging Face Hub and have the app download it at
> startup — see the README "Limitations & next steps".
