"""Download the welding-defect segmentation dataset from Kaggle via kagglehub.

Requires:
1. `pip install kagglehub`
2. A Kaggle token at ~/.kaggle/access_token (chmod 600)
   or KAGGLE_API_TOKEN env var set.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_kaggle_auth() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    legacy_json = kaggle_dir / "kaggle.json"
    new_token = kaggle_dir / "access_token"
    env_new = "KAGGLE_API_TOKEN" in os.environ
    env_old = "KAGGLE_KEY" in os.environ and "KAGGLE_USERNAME" in os.environ
    if not (legacy_json.exists() or new_token.exists() or env_new or env_old):
        logger.error(
            "Kaggle credentials missing. Provide ONE of:\n"
            "  a) ~/.kaggle/access_token containing your KGAT_... token (chmod 600), OR\n"
            "  b) ~/.kaggle/kaggle.json with {\"username\":..., \"key\":...} (chmod 600), OR\n"
            "  c) env var KAGGLE_API_TOKEN=KGAT_..., OR\n"
            "  d) env vars KAGGLE_USERNAME and KAGGLE_KEY."
        )
        sys.exit(1)
    if new_token.exists() and "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = new_token.read_text().strip()
        logger.info("loaded KAGGLE_API_TOKEN from %s", new_token)


def download(dataset: str) -> Path:
    import kagglehub
    logger.info("downloading dataset '%s' via kagglehub", dataset)
    path = kagglehub.dataset_download(dataset)
    logger.info("kagglehub returned: %s", path)
    return Path(path)


def copy_to_raw(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if target.exists():
            logger.info("  %s already present, skipping", f.name)
            continue
        if f.is_dir():
            shutil.copytree(f, target)
        else:
            shutil.copy2(f, target)
        logger.info("  copied %s", f.name)


def unzip_archives(dst: Path) -> None:
    for zip_path in sorted(dst.glob("*.zip")):
        logger.info("unzipping %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dst)
        zip_path.unlink()


if __name__ == "__main__":
    cfg = load_config()
    check_kaggle_auth()
    kagglehub_path = download(cfg["data"]["kaggle_dataset"])
    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    copy_to_raw(kagglehub_path, raw_dir)
    unzip_archives(raw_dir)
    logger.info("done. raw at %s", raw_dir)
