# 🔥 Weld Defect Vision

![CI](https://github.com/Jimmply/weld-defect-vision/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Model: YOLOv8](https://img.shields.io/badge/model-YOLOv8--seg-orange)

**Instance-segmentation of welding defects on radiograph imagery.** YOLOv8-seg trained on the Kaggle [welding-defect-segmentation-dataset](https://www.kaggle.com/datasets/bemorekgg/welding-defect-segmentation-dataset) — defect bounding boxes **and** per-defect masks, five defect classes (crack, porosity, lack-of-fusion, undercut, slag inclusion), weld-session-stratified splitting so metrics aren't leaky, Streamlit upload-and-detect demo.

Built as the computer-vision companion to my manufacturing-ML portfolio — directly applicable to the aerospace laser-welding work I do at [Laseronics](https://laseronics.com).

## Why this project

Everyone's XGBoost repo looks the same. CV + manufacturing is the single most-searched combo in aerospace defect-detection job postings and almost nobody has it in their portfolio. This closes that gap with real imagery and a demo you can run against your own weld photos.

## What's shipped so far

| Component | Status |
|---|---|
| Dataset download via `kagglehub` | ✅ `scripts/download_data.py` |
| Weld-session-stratified splitter | ✅ `src/data_split.py` + 10 tests |
| **Roboflow-aware session grouping** | ✅ handles the `<source>.rf.<hash>` augmentation pattern used by the Kaggle dataset |
| **Dataset prep → YOLOv8 layout** | ✅ `scripts/prepare_dataset.py` — produces `data/processed/data.yaml` + train/val/test image+label trees |
| YOLOv8-seg training script | ✅ `scripts/train.py` with config overrides |
| Streamlit upload-and-detect demo | ✅ `src/app.py` |
| Evaluation metrics scaffolding | ✅ `src/metrics.py` + 2 tests |
| CI (3.10 / 3.11 / 3.12) | ✅ |
| Full training run + results table | ⏳ **next commit** |
| Streamlit screenshot / demo GIF | ⏳ **next commit** |

## Dataset snapshot (after `make download && python scripts/prepare_dataset.py`)

Honest numbers from the actual Kaggle release:

| Metric | Value |
|---|---|
| Raw images (JPGs) | 1,619 |
| Unique source sessions (pre-Roboflow) | **907** — most have 2 augmentations each |
| YOLO bbox labels | 1,619 (one per image, no orphans) |
| Segmentation masks (PNG) | 1,601 |
| Train / Val / Test images | **1,117 / 257 / 245** (session-stratified, 70/15/15) |
| Classes used in labels | **3** — IDs `0`, `1`, `2`. See caveat below. |

### Class distribution per bucket (bboxes)

| Class ID | Train | Val | Test |
|---|---|---|---|
| 0 | 763 | 158 | 168 |
| 1 | 1,345 | 257 | 294 |
| 2 | 1,072 | 304 | 222 |

### Caveat worth logging

The Kaggle dataset ships **no `classes.txt` or `data.yaml`** — only the three numeric class IDs. Config currently labels them `class_0 / class_1 / class_2` with a note. Before the next training run I'll inspect ~20 sample images per class and assign real defect labels (crack / porosity / lack-of-fusion / etc.) based on visual inspection. This is the kind of real-world data-wrangling gap that no amount of reading the Kaggle page would have surfaced.

## Why weld-session-stratified splitting

Random splitting at the image level leaks context. Multiple frames from the same weld session land in both train and test, inflating metrics. [`src/data_split.py`](src/data_split.py) groups by weld-session (parsed from filename prefix) and splits whole sessions into train/val/test — so the model's evaluation reflects performance on genuinely unseen welds.

This matters because aerospace weld inspection is high-stakes: a model that "works great" on leaky splits but fails on new weld sessions is worse than no model at all.

## Quickstart

```bash
git clone https://github.com/Jimmply/weld-defect-vision.git
cd weld-defect-vision
make setup                # creates .venv + installs deps

# 1. Get your Kaggle token at https://www.kaggle.com/settings → "Create New Token"
#    mv the downloaded token to ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

make download             # ~500 MB
make train                # YOLOv8n-seg, 50 epochs, CPU by default
make app                  # launch the demo
```

Edit `config/settings.yaml` to tune image size, epochs, batch size, LR schedule, augmentations.

## Architecture

```
Kaggle download → session-stratified split → YOLOv8-seg train
                                                   │
                                                   ▼
                                        best.pt + eval metrics
                                                   │
                                                   ▼
                                        Streamlit upload demo
```

## Project layout

```
weld-defect-vision/
├── .github/workflows/ci.yml      # pytest on 3.10 / 3.11 / 3.12
├── config/settings.yaml          # classes, augmentations, training hyperparams
├── data/                         # gitignored — dataset lives here after download
├── docs/img/                     # charts + screenshots
├── models/                       # gitignored — best.pt after training
├── notebooks/                    # TBD — EDA + results walkthrough
├── scripts/
│   ├── download_data.py          # kagglehub + auth check
│   └── train.py                  # YOLOv8-seg trainer with config overrides
├── src/
│   ├── data_split.py             # weld-session-stratified splitter
│   ├── metrics.py                # evaluation containers
│   └── app.py                    # Streamlit demo
├── tests/                        # 9 tests covering the no-GPU logic
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Roadmap

- [ ] Full training run + populate the Results section with real mAP@50 / mAP@50-95 per class
- [ ] Streamlit screenshot + 30-second demo GIF
- [x] `prepare_dataset.py` script that converts the raw Kaggle layout to YOLOv8 structure. **Done — 1,117/257/245 session-stratified split on 907 unique source images.**
- [ ] **Resolve class IDs → real defect names** — Kaggle release has no `classes.txt`; manual inspection next.
- [ ] Augmentation ablation — do weld-specific augmentations (rotation but not vflip, mild brightness) actually help?
- [ ] Confusion-matrix analysis — which defect pairs does the model confuse?
- [ ] Export to ONNX for low-latency inference on shop-floor hardware

## Author

**Dmitry Shurkhai** — Manufacturing Data Scientist · [GitHub](https://github.com/Jimmply) · [Kaggle](https://kaggle.com/jimmysh) · [LinkedIn](https://linkedin.com/in/etozhejimmy)
