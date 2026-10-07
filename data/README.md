# Data directory

Weld-defect imagery + annotations land here after running:

```bash
python scripts/download_data.py
```

The Kaggle dataset [`bemorekgg/welding-defect-segmentation-dataset`](https://www.kaggle.com/datasets/bemorekgg/welding-defect-segmentation-dataset) contains:

- Welding radiograph images (JPG/PNG)
- YOLO-format bounding boxes
- SAM-generated segmentation masks (from the earlier `Welding Defect Dataset v2`)

Downloaded contents are **gitignored** — only the directory structure is tracked.

## Processed layout (what `scripts/prepare_dataset.py` produces)

```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml           # YOLOv8 config pointing to the above
```

## Why weld-session-stratified splitting

Random splitting at the image level leaks context — multiple images from the same weld session end up in both train and test, inflating metrics. We split by weld-session (filename prefix heuristic) so the model is evaluated on genuinely unseen welds.
