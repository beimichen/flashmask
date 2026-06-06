"""Evaluate a trained detector and report the headline metrics.

Validates on the dataset's **test** split (real images only — see
``flashmask.data.splits``) so the reported numbers reflect real-world
performance rather than synthetic-inflated validation scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def evaluate(
    weights: str | Path,
    data_yaml: str | Path,
    *,
    imgsz: int = 960,
    conf: float = 0.25,
    iou: float = 0.5,
    split: str = "test",
) -> dict:
    """Run validation and return a metrics dict (precision/recall/mAP50/mAP50-95)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou, split=split)
    box = metrics.box
    out = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
    print("\n".join(f"  {k:>10}: {v:.4f}" for k, v in out.items()))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a YOLO text detector.")
    ap.add_argument("--weights", "-w", type=Path, required=True)
    ap.add_argument("--data", "-d", type=Path, required=True, help="path to data.yaml")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args(argv)
    evaluate(
        args.weights, args.data, imgsz=args.imgsz, conf=args.conf, iou=args.iou, split=args.split
    )


if __name__ == "__main__":
    main()
