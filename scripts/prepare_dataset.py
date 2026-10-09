"""Convert the Kaggle welding-defect dataset into the YOLOv8 training layout.

Input (flat, under ``data/raw/images_bbox_mask/``):
  <session>.rf.<hash>.jpg   # image
  <session>.rf.<hash>.txt   # YOLO bbox labels: ``class cx cy w h`` (normalized)
  <session>.rf.<hash>.png   # segmentation mask (ignored for the bbox detection path)

Output (``data/processed/``):
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
  data.yaml                       # YOLOv8 config

Splitting is weld-session-stratified using the Roboflow source stem — two
augmentations of the same source image can never land in different buckets.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_split import SplitConfig, roboflow_session_from_filename, session_stratified_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(REPO_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def prepare(raw_dir: Path, out_dir: Path, split_cfg: SplitConfig, class_names: list[str]) -> dict:
    """Build the YOLOv8 train/val/test layout. Returns per-bucket counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for bucket in ("train", "val", "test"):
        (out_dir / "images" / bucket).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / bucket).mkdir(parents=True, exist_ok=True)

    images = sorted(f.name for f in raw_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
    logger.info("found %d images in %s", len(images), raw_dir)
    if not images:
        raise FileNotFoundError(f"no images under {raw_dir}")

    split = session_stratified_split(
        images, config=split_cfg, session_fn=roboflow_session_from_filename
    )

    counts = {}
    for bucket, files in split.items():
        n_copied = 0
        n_skipped_no_label = 0
        for img_name in files:
            img_src = raw_dir / img_name
            lbl_src = raw_dir / (Path(img_name).stem + ".txt")
            if not lbl_src.exists():
                n_skipped_no_label += 1
                continue
            shutil.copy2(img_src, out_dir / "images" / bucket / img_name)
            shutil.copy2(lbl_src, out_dir / "labels" / bucket / lbl_src.name)
            n_copied += 1
        counts[bucket] = {"copied": n_copied, "skipped_no_label": n_skipped_no_label}
        logger.info("bucket %-5s : copied=%d  skipped_no_label=%d",
                    bucket, n_copied, n_skipped_no_label)

    # Write YOLOv8 data.yaml
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    logger.info("wrote %s (%d classes)", out_dir / "data.yaml", len(class_names))

    return counts


if __name__ == "__main__":
    cfg = load_config()
    raw = REPO_ROOT / cfg["data"]["raw_dir"] / "images_bbox_mask"
    out = REPO_ROOT / cfg["data"]["processed_dir"]
    split_cfg = SplitConfig(
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        random_seed=cfg["split"]["random_seed"],
    )
    counts = prepare(raw, out, split_cfg, class_names=cfg["classes"])
    total = sum(c["copied"] for c in counts.values())
    logger.info("DONE: %d total image+label pairs under %s", total, out)
