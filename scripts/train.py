"""Train YOLOv8 instance segmentation on the prepared weld-defect dataset.

Prerequisite: `python scripts/prepare_dataset.py` has been run and produced
``data/processed/data.yaml`` with train/val/test image + label pointers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(REPO_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--device", default=None, help="Override config device (cpu or 0).")
    parser.add_argument("--data", default=None, help="Override path to data.yaml.")
    args = parser.parse_args()

    cfg = load_config()
    t = cfg["training"]
    data_yaml = args.data or str(REPO_ROOT / cfg["data"]["processed_dir"] / "data.yaml")

    from ultralytics import YOLO

    logger.info("loading YOLO weights: %s", t["model"])
    model = YOLO(t["model"])

    logger.info("training on %s for %d epochs", data_yaml, args.epochs or t["epochs"])
    model.train(
        data=data_yaml,
        epochs=args.epochs or t["epochs"],
        imgsz=t["image_size"],
        batch=args.batch or t["batch_size"],
        patience=t["patience"],
        optimizer=t["optimizer"],
        lr0=t["lr0"],
        lrf=t["lrf"],
        weight_decay=t["weight_decay"],
        warmup_epochs=t["warmup_epochs"],
        device=args.device or t["device"],
        project=str(REPO_ROOT / "runs"),
        name="train",
    )

    best = REPO_ROOT / "runs" / "train" / "weights" / "best.pt"
    out = REPO_ROOT / "models" / "best.pt"
    if best.exists():
        import shutil
        shutil.copy2(best, out)
        logger.info("saved best weights -> %s", out)


if __name__ == "__main__":
    main()
