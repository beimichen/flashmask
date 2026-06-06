"""Synthetic training data: render vocabulary text onto background images.

Pastes randomly styled (font, size, rotation, colour-for-contrast) single- and
multi-line text onto real backgrounds, with photometric/geometric augmentation
variants per background, and emits matching review-metadata boxes. Ported and
tidied from the original ``generate_synthetic_data_improved.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flashmask.config import paths, set_seed

DISTANCE_THRESHOLD = 20  # min px gap between a multiline block and other blocks


class SyntheticDataGenerator:
    """Render labelled synthetic text-on-background images."""

    MIN_ASPECT_RATIO = 0.2
    MAX_ASPECT_RATIO = 5.0
    FONT_SIZE_MIN_PCT = 0.025
    FONT_SIZE_MAX_PCT = 0.055

    def __init__(
        self,
        keywords_file: Path,
        backgrounds_dir: Path,
        fonts_dir: Path,
        min_labels: int = 3,
        max_labels: int = 12,
    ) -> None:
        self.background_paths = self._load_backgrounds(backgrounds_dir)
        self.font_paths = list(Path(fonts_dir).glob("*.ttf")) + list(Path(fonts_dir).glob("*.otf"))
        if not self.font_paths:
            raise FileNotFoundError(f"No font files in {fonts_dir}")
        self.vocabulary = [
            line.strip()
            for line in Path(keywords_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.vocabulary:
            raise ValueError(f"No keywords in {keywords_file}")
        self.min_labels, self.max_labels = min_labels, max_labels

    def _load_backgrounds(self, backgrounds_dir: Path) -> list[Path]:
        valid = []
        for p in Path(backgrounds_dir).glob("*.*"):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h and self.MIN_ASPECT_RATIO <= w / h <= self.MAX_ASPECT_RATIO:
                valid.append(p)
        if not valid:
            raise FileNotFoundError(f"No valid backgrounds in {backgrounds_dir}")
        return valid

    @staticmethod
    def _overlaps(new_box: Mapping[str, float], existing: list[Mapping[str, float]]) -> bool:
        x1, y1 = new_box["x"], new_box["y"]
        x2, y2 = x1 + new_box["width"], y1 + new_box["height"]
        for b in existing:
            bx1, by1 = b["x"], b["y"]
            bx2, by2 = bx1 + b["width"], by1 + b["height"]
            if not (x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1):
                return True
        return False

    @staticmethod
    def _jitter(bg: np.ndarray) -> np.ndarray:
        img = cv2.convertScaleAbs(
            bg, alpha=random.choice([0.8, 1.2]), beta=random.choice([-30, 30])
        )
        gamma = random.choice([0.9, 1.1])
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def _blur(bg: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(bg, (5, 5), 0)

    @staticmethod
    def _squish(bg: np.ndarray) -> np.ndarray:
        h, w = bg.shape[:2]
        canvas = np.full_like(bg, [int(c) for c in bg.mean(axis=(0, 1))])
        if random.choice(["vertical", "horizontal"]) == "vertical":
            new_h = int(h * 0.7)
            canvas[(h - new_h) // 2 : (h - new_h) // 2 + new_h, :] = cv2.resize(bg, (w, new_h))
        else:
            new_w = int(w * 0.7)
            canvas[:, (w - new_w) // 2 : (w - new_w) // 2 + new_w] = cv2.resize(bg, (new_w, h))
        return canvas

    def _render_one(self, aug: np.ndarray) -> tuple[Image.Image, list[dict[str, Any]]]:
        h, w = aug.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)).convert("RGBA")
        boxes: list[dict[str, Any]] = []
        for _ in range(random.randint(self.min_labels, self.max_labels)):
            if random.random() < 0.4:
                text = "\n".join(random.choices(self.vocabulary, k=random.randint(2, 5)))
            else:
                text = random.choice(self.vocabulary)

            font_size = max(
                1, int(random.uniform(self.FONT_SIZE_MIN_PCT, self.FONT_SIZE_MAX_PCT) * h)
            )
            font = ImageFont.truetype(str(random.choice(self.font_paths)), size=font_size)
            bbox = ImageDraw.Draw(pil).multiline_textbbox((0, 0), text, font=font, spacing=4)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = int(0.1 * max(tw, th))
            tw_p, th_p = tw + 2 * pad, th + 2 * pad

            p = random.random()
            angle = (
                random.choice([90, -90])
                if p < 0.05
                else random.uniform(30, 45) * random.choice([-1, 1])
                if p < 0.15
                else 0
            )
            self._place_text(pil, aug, text, font, (pad, tw_p, th_p), angle, boxes)
        return pil, boxes

    def _place_text(self, pil, aug, text, font, dims, angle, boxes) -> None:
        pad, tw_p, th_p = dims
        h, w = aug.shape[:2]
        for _ in range(20):
            layer = Image.new("RGBA", (tw_p, th_p), (0, 0, 0, 0))
            ImageDraw.Draw(layer).multiline_text(
                (pad, pad), text, font=font, fill=(255, 255, 255, 255), spacing=4
            )
            rotated = layer.rotate(angle, expand=True) if angle else layer
            rt_w, rt_h = rotated.size

            x, y = random.randint(0, max(0, w - rt_w)), random.randint(0, max(0, h - rt_h))
            new_box = {"x": x, "y": y, "width": rt_w, "height": rt_h}
            if self._overlaps(new_box, boxes) or x + rt_w > w or y + rt_h > h:
                continue
            if "\n" in text and self._too_close(new_box, boxes):
                continue

            region = aug[y : y + rt_h, x : x + rt_w]
            avg = int(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean()) if region.size else 0
            colour = (255, 255, 255, 255) if avg < 127 else (0, 0, 0, 255)
            layer = Image.new("RGBA", (tw_p, th_p), (0, 0, 0, 0))
            ImageDraw.Draw(layer).multiline_text(
                (pad, pad), text, font=font, fill=colour, spacing=4
            )
            rotated = layer.rotate(angle, expand=True) if angle else layer

            pil.paste(rotated, (x, y), rotated)
            boxes.append(
                {
                    "id": f"r{len(boxes) + 1}",
                    "text": text.replace(chr(10), " "),
                    **new_box,
                    "group": None,
                    "visible": True,
                }
            )
            return

    @staticmethod
    def _too_close(new_box: Mapping[str, float], boxes: list[Mapping[str, float]]) -> bool:
        for b in boxes:
            dx = min(
                abs(new_box["x"] - b["x"]),
                abs(new_box["x"] + new_box["width"] - b["x"] - b["width"]),
            )
            dy = min(
                abs(new_box["y"] - b["y"]),
                abs(new_box["y"] + new_box["height"] - b["y"] - b["height"]),
            )
            if dx < DISTANCE_THRESHOLD and dy < DISTANCE_THRESHOLD:
                return True
        return False

    def generate(self, output_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate 4 augmented variants per background; return the metadata map."""
        metadata = dict(metadata or {})
        img_dir = Path(output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for bg_path in self.background_paths:
            raw = cv2.imread(str(bg_path))
            if raw is None:
                continue
            variants = {
                "orig": raw,
                "jitter": self._jitter(raw),
                "blur": self._blur(raw),
                "squish": self._squish(raw),
            }
            for name, aug in variants.items():
                pil, boxes = self._render_one(aug)
                out = img_dir / f"{bg_path.stem}_{name}_{count}.png"
                pil.convert("RGB").save(out)
                metadata[str(out.resolve())] = {
                    "status": "Accept as-is",
                    "source": "synthetic",
                    "boxes": boxes,
                }
                count += 1
        return metadata


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic text-on-background data.")
    ap.add_argument("--keywords", type=Path, default=paths.data_external / "keywords.txt")
    ap.add_argument("--backgrounds", type=Path, required=True)
    ap.add_argument("--fonts", type=Path, default=paths.fonts)
    ap.add_argument("--out", type=Path, default=paths.data_processed / "synthetic")
    ap.add_argument("--min-labels", type=int, default=3)
    ap.add_argument("--max-labels", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    gen = SyntheticDataGenerator(
        args.keywords, args.backgrounds, args.fonts, args.min_labels, args.max_labels
    )
    meta = gen.generate(args.out)
    out_json = args.out / "synthetic_metadata.json"
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"Generated {len(meta)} images; metadata -> {out_json}")


if __name__ == "__main__":
    main()
