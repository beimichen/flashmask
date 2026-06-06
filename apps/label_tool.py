"""Streamlit model-in-the-loop labeling tool (backend-free rewrite).

Pre-label text regions, correct them on a canvas, and save to a review-metadata
JSON consumed by the dataset builder. Pre-labeling uses the **trained detector**
once one exists at ``models/detector.onnx`` (refining the model's own
predictions), and falls back to **DocTR OCR** the first time — closing the
label -> train -> label flywheel. No dependency on any private app backend.

Run:  streamlit run apps/label_tool.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from flashmask.config import paths
from flashmask.labeling.prelabel import prelabel, should_use_model
from flashmask.labeling.region import Region

IMAGE_DIR = Path("data/raw/to_label")
METADATA_PATH = Path("data/processed/review_metadata.json")
DETECTOR_PATH = paths.models / "detector.onnx"
MAX_DISPLAY_WIDTH = 800

st.set_page_config(page_title="flashmask labeling", layout="wide")

images = sorted(p for p in IMAGE_DIR.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
if not images:
    st.warning(f"No images found in {IMAGE_DIR}. Put images there to label.")
    st.stop()

metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
if "idx" not in st.session_state:
    st.session_state.idx = 0
idx = st.session_state.idx
img_path = images[idx]
st.sidebar.markdown(f"**Image {idx + 1}/{len(images)}** — `{img_path.name}`")

using_model = should_use_model(DETECTOR_PATH)
st.sidebar.caption(
    f"Pre-labeling source: **{'trained detector' if using_model else 'DocTR OCR'}**"
    + ("" if using_model else f"  (no model at `{DETECTOR_PATH}` yet)")
)

conf = st.slider("Min OCR confidence", 0.0, 1.0, 0.5, 0.01)
line_tol = st.slider("Vertical gap tolerance (px)", 0, 100, 10)
horiz_tol = st.slider("Horizontal gap tolerance (px)", 0, 200, 10)

if "regions" not in st.session_state or st.session_state.get("regions_for") != str(img_path):
    saved = metadata.get(str(img_path), {}).get("boxes")
    st.session_state.regions = [Region.from_dict(b) for b in saved] if saved else []
    st.session_state.regions_for = str(img_path)

if st.button("▶ Pre-label with model" if using_model else "▶ Pre-label with DocTR (OCR)"):
    regions, source = prelabel(
        str(img_path),
        detector_path=DETECTOR_PATH,
        doctr_conf=conf,
        line_tol=line_tol,
        horiz_tol=horiz_tol,
    )
    st.session_state.regions = regions
    st.toast(f"Pre-labeled {len(regions)} regions via {source}")

pil = Image.fromarray(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
scale = min(1.0, MAX_DISPLAY_WIDTH / pil.width)
disp = pil.resize((int(pil.width * scale), int(pil.height * scale)))
initial = [
    {
        "type": "rect",
        "left": r.x * scale,
        "top": r.y * scale,
        "width": r.width * scale,
        "height": r.height * scale,
        "fill": "rgba(0,0,0,0.3)",
        "stroke": "red",
    }
    for r in st.session_state.regions
]
canvas = st_canvas(
    background_image=disp,
    drawing_mode="rect",
    stroke_width=2,
    height=disp.height,
    width=disp.width,
    initial_drawing={"objects": initial},
    key=f"c{idx}",
)

col_prev, col_next = st.columns(2)
if col_prev.button("← Previous") and idx > 0:
    st.session_state.idx -= 1
    st.rerun()
if col_next.button("Save & Next →"):
    boxes = []
    for i, obj in enumerate(canvas.json_data.get("objects", []), start=1):
        boxes.append(
            Region(
                x=int(obj["left"] / scale),
                y=int(obj["top"] / scale),
                width=int(obj["width"] / scale),
                height=int(obj["height"] / scale),
                id=f"r{i}",
            ).to_dict()
        )
    metadata[str(img_path)] = {"status": "Save edits", "boxes": boxes}
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    st.session_state.idx = min(idx + 1, len(images) - 1)
    st.rerun()
