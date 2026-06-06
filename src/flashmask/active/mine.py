"""Mine an unlabeled image pool for the most informative images to label next.

Runs the trained detector over a pool of (already diagram-filtered) images,
scores each by uncertainty (see :mod:`flashmask.active.sampling`), ranks them,
and selects the top-K. With ``--stage`` it copies the winners into the labeling
queue and seeds the review-metadata with the model's own boxes, so the labeling
tool opens them pre-labelled for correction — closing the
label -> train -> mine -> label flywheel.

    flashmask active mine --pool data/interim/images --top-k 50 --stage

This is an acquisition/selection step, not a retrainer: you review the selected
batch in the labeling tool, then run `just train-finetune` on the enlarged set.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from flashmask.active.sampling import STRATEGIES, score_detections
from flashmask.config import paths

logger = logging.getLogger(__name__)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class ImageScore:
    path: str
    score: float
    n_boxes: int
    mean_conf: float

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "score": round(self.score, 6),
            "n_boxes": self.n_boxes,
            "mean_conf": round(self.mean_conf, 4),
        }


@dataclass
class MineConfig:
    pool_dir: Path
    detector_path: Path = field(default_factory=lambda: paths.models / "detector.onnx")
    classifier_path: Path | None = None
    top_k: int = 50
    strategy: str = "entropy"
    conf_floor: float = 0.05
    band: tuple[float, float] = (0.20, 0.50)
    empty_image_score: float = 0.0
    stage: bool = False
    stage_dir: Path = field(default_factory=lambda: paths.data_raw / "to_label")
    metadata_path: Path = field(
        default_factory=lambda: paths.data_processed / "review_metadata.json"
    )
    report_path: Path | None = None


def _list_images(pool_dir: Path) -> list[Path]:
    return sorted(p for p in Path(pool_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible (portable metadata keys)."""
    try:
        return str(path.resolve().relative_to(paths.root))
    except ValueError:
        return str(path)


def mine(cfg: MineConfig) -> list[ImageScore]:
    """Score the pool and return the top-K most uncertain images (ranked)."""
    import cv2

    from flashmask.inference.pipeline import TextMaskPipeline

    if not Path(cfg.detector_path).exists():
        raise FileNotFoundError(
            f"No detector at {cfg.detector_path}. Train one first (see models/README.md)."
        )

    pipe = TextMaskPipeline(cfg.detector_path, cfg.classifier_path, use_ocr=False)
    strategy = cfg.strategy
    if strategy == "disagreement" and pipe.cls is None:
        logger.warning("disagreement needs a classifier; none loaded — using 'entropy'.")
        strategy = "entropy"

    images = _list_images(cfg.pool_dir)
    if not images:
        raise FileNotFoundError(f"No images found in pool {cfg.pool_dir}")
    logger.info("Scoring %d images with strategy=%s", len(images), strategy)

    scored: list[ImageScore] = []
    detections: dict[str, list[dict]] = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = [b for b in pipe.detect(img) if b["det_conf"] >= cfg.conf_floor]
        confs = [b["det_conf"] for b in boxes]
        cls_probs = None
        if strategy == "disagreement":
            cls_probs = [
                pipe.classify(img[b["y"] : b["y"] + b["height"], b["x"] : b["x"] + b["width"]])
                for b in boxes
            ]
        score = score_detections(
            confs,
            cls_probs,
            strategy=strategy,
            band=cfg.band,
            empty_image_score=cfg.empty_image_score,
        )
        scored.append(
            ImageScore(str(img_path), score, len(boxes), sum(confs) / len(confs) if confs else 0.0)
        )
        detections[str(img_path)] = boxes

    scored.sort(key=lambda s: s.score, reverse=True)
    selected = scored[: cfg.top_k]

    if cfg.stage:
        _stage(selected, detections, cfg)
    if cfg.report_path:
        _write_report(scored, selected, strategy, cfg)
    return selected


def _stage(selected: list[ImageScore], detections: dict[str, list[dict]], cfg: MineConfig) -> None:
    """Copy selected images into the label queue and seed model pre-labels."""
    cfg.stage_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(cfg.metadata_path.read_text()) if cfg.metadata_path.exists() else {}
    for item in selected:
        src = Path(item.path)
        dst = cfg.stage_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        boxes = [
            {
                "id": f"r{i}",
                "text": "",
                "x": b["x"],
                "y": b["y"],
                "width": b["width"],
                "height": b["height"],
                "group": None,
                "visible": True,
            }
            for i, b in enumerate(detections[item.path], start=1)
        ]
        metadata[_rel(dst)] = {"status": "pending", "source": "active_mining", "boxes": boxes}
    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Staged %d images into %s", len(selected), cfg.stage_dir)


def _write_report(
    scored: list[ImageScore], selected: list[ImageScore], strategy: str, cfg: MineConfig
) -> None:
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    report = {
        "strategy": strategy,
        "pool": str(cfg.pool_dir),
        "scored": len(scored),
        "selected": len(selected),
        "top": [s.as_dict() for s in selected],
    }
    Path(cfg.report_path).write_text(json.dumps(report, indent=2))  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Mine an image pool for the most informative images.")
    ap.add_argument("--pool", type=Path, required=True, help="directory of unlabeled images")
    ap.add_argument("--detector", type=Path, default=paths.models / "detector.onnx")
    ap.add_argument("--classifier", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--strategy", choices=STRATEGIES, default="entropy")
    ap.add_argument("--conf-floor", type=float, default=0.05)
    ap.add_argument("--band", type=float, nargs=2, default=(0.20, 0.50), metavar=("LO", "HI"))
    ap.add_argument("--empty-image-score", type=float, default=0.0)
    ap.add_argument(
        "--stage", action="store_true", help="copy winners into the label queue + pre-label"
    )
    ap.add_argument("--report", type=Path, default=paths.reports / "active_mining.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    selected = mine(
        MineConfig(
            pool_dir=args.pool,
            detector_path=args.detector,
            classifier_path=args.classifier,
            top_k=args.top_k,
            strategy=args.strategy,
            conf_floor=args.conf_floor,
            band=tuple(args.band),
            empty_image_score=args.empty_image_score,
            stage=args.stage,
            report_path=args.report,
        )
    )
    print(f"\nTop {len(selected)} by uncertainty ({args.strategy}):")
    for rank, s in enumerate(selected[:20], start=1):
        print(
            f"  {rank:>3}. {s.score:.4f}  ({s.n_boxes} boxes, mean_conf {s.mean_conf:.2f})  {Path(s.path).name}"
        )
    print(f"\nFull report -> {args.report}" + ("  | staged to label queue" if args.stage else ""))


if __name__ == "__main__":
    main()
