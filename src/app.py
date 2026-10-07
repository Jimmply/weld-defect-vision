"""Streamlit demo — upload a weld image, see defect boxes + confidence."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


@st.cache_resource
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(weights_path: str):
    """Lazy-load YOLOv8 — only when the user first uploads an image."""
    from ultralytics import YOLO
    return YOLO(weights_path)


def main() -> None:
    cfg = load_config()
    st.set_page_config(
        page_title=cfg["streamlit"]["page_title"],
        page_icon=cfg["streamlit"]["page_icon"],
        layout="wide",
    )
    st.title(f"{cfg['streamlit']['page_icon']} {cfg['streamlit']['page_title']}")
    st.caption("YOLOv8 instance segmentation of weld radiograph defects. Upload an image to see defect bounding boxes, masks, and confidence scores.")

    weights = REPO_ROOT / "models" / "best.pt"
    if not weights.exists():
        st.warning("No trained weights found at `models/best.pt`. Run `python scripts/train.py` first, or drop a trained `.pt` file in `models/`.")
        return

    with st.sidebar:
        conf = st.slider("Confidence threshold", 0.0, 1.0, cfg["streamlit"]["default_conf"], 0.05)
        iou = st.slider("IoU threshold", 0.0, 1.0, cfg["streamlit"]["default_iou"], 0.05)

    uploaded = st.file_uploader("Upload a weld image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a weld radiograph (PNG / JPG) to run detection.")
        return

    model = load_model(str(weights))
    img_bytes = uploaded.read()

    import numpy as np
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_bytes))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(img, use_column_width=True)

    results = model.predict(np.array(img), conf=conf, iou=iou, verbose=False)
    result = results[0]

    with col2:
        st.subheader("Detections")
        annotated = result.plot()
        st.image(annotated, use_column_width=True)

    if result.boxes is None or len(result.boxes) == 0:
        st.info("No defects detected above the current confidence threshold.")
        return

    st.subheader("Per-defect summary")
    rows = []
    for i, box in enumerate(result.boxes):
        cls_idx = int(box.cls[0])
        rows.append({
            "idx": i,
            "class": result.names.get(cls_idx, str(cls_idx)),
            "confidence": float(box.conf[0]),
            "bbox_xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()],
        })
    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
