"""Generate a tiny set of PLACEHOLDER diagram images + labels for demos/tests.

The original training data is not redistributable, so this synthesizes a handful
of simple labelled diagrams (boxes, arrows, text) purely so that ``just demo``,
the test suite, and ``train_detector data=sample`` can run end-to-end on a fresh
clone. These are obviously synthetic and are NOT the real dataset.

Run:  python scripts/make_sample_images.py
"""

from __future__ import annotations

import json
import random

from PIL import Image, ImageDraw, ImageFont

from flashmask.config import paths, set_seed

WORDS = [
    "Mitochondrion",
    "Nucleus",
    "Cytoplasm",
    "Membrane",
    "Ribosome",
    "Input",
    "Hidden Layer",
    "Output",
    "Encoder",
    "Decoder",
    "Gradient",
    "Loss",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("Roboto-Medium.ttf", "OpenSans-Regular.ttf"):
        path = paths.fonts / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_image(rng: random.Random) -> tuple[Image.Image, list[dict]]:
    w, h = 800, 600
    img = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    boxes: list[dict] = []

    for i in range(rng.randint(4, 7)):
        label = rng.choice(WORDS)
        font = _font(rng.randint(20, 34))
        x, y = rng.randint(30, w - 220), rng.randint(30, h - 80)
        tb = draw.textbbox((x, y), label, font=font)
        bw, bh = tb[2] - tb[0], tb[3] - tb[1]
        # a decorative box/ellipse near the label so it reads like a diagram
        draw.rectangle([x - 12, y - 10, x + bw + 12, y + bh + 10], outline=(90, 90, 90), width=2)
        draw.text((x, y), label, fill=(20, 20, 20), font=font)
        boxes.append(
            {
                "id": f"r{i + 1}",
                "text": label,
                "x": x - 12,
                "y": y - 10,
                "width": bw + 24,
                "height": bh + 20,
                "group": None,
                "visible": True,
            }
        )
    # a couple of connecting arrows for flavour
    for _ in range(3):
        draw.line(
            [rng.randint(0, w), rng.randint(0, h), rng.randint(0, w), rng.randint(0, h)],
            fill=(150, 150, 150),
            width=2,
        )
    return img, boxes


def main(n: int = 4) -> None:
    set_seed(123)
    rng = random.Random(123)
    out_dir = paths.data_sample / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}
    for i in range(n):
        img, boxes = make_image(rng)
        path = out_dir / f"sample_diagram_{i:02d}.png"
        img.save(path)
        metadata[str(path)] = {"status": "Accept as-is", "source": "placeholder", "boxes": boxes}

    meta_path = paths.data_sample / "sample_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {n} placeholder diagrams to {out_dir} and metadata to {meta_path}")


if __name__ == "__main__":
    main()
