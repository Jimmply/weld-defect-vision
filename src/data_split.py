"""Weld-session-stratified train/val/test split.

Random splitting over image files leaks across weld sessions — if frames 1-10
of session A end up in train and frames 11-20 in test, the model's "test"
performance is really memorization. This module groups images by weld-session
(parsed from filename prefix) and splits whole sessions into train/val/test.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


SESSION_PATTERN = re.compile(r"^([a-zA-Z0-9_-]+?)(?:_\d+)?\.(?:png|jpg|jpeg)$", re.IGNORECASE)

# Roboflow augmented filenames look like  <source>.rf.<hash>.<ext>
# where <source> is the real image id and <hash> changes per augmentation.
ROBOFLOW_SPLIT = re.compile(r"\.rf\.[0-9a-f]+", re.IGNORECASE)


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_seed: int = 42


def session_from_filename(filename: str) -> str:
    """Extract the weld-session identifier from an image filename.

    Heuristic: everything before the final ``_<digits>.<ext>`` suffix is the session.
    Falls back to the full stem if no ``_<digits>`` suffix is present.
    """
    match = SESSION_PATTERN.match(filename)
    if match is None:
        return Path(filename).stem
    return match.group(1)


def roboflow_session_from_filename(filename: str) -> str:
    """Extract the source image stem from a Roboflow-augmented filename.

    Roboflow saves augmentations as ``<source>.rf.<hash>.<ext>`` where the
    same ``<source>`` can appear multiple times (one per augmentation). We
    group by ``<source>`` so augmented variants of the same real image don't
    get split across train / val / test.
    """
    stem = Path(filename).stem
    parts = ROBOFLOW_SPLIT.split(stem, maxsplit=1)
    return parts[0] if parts else stem


def group_by_session(
    filenames: list[str],
    session_fn=session_from_filename,
) -> dict[str, list[str]]:
    """Return ``{session_id: [file1, file2, ...]}`` for a list of image filenames."""
    groups: dict[str, list[str]] = defaultdict(list)
    for f in filenames:
        groups[session_fn(f)].append(f)
    return dict(groups)


def session_stratified_split(
    filenames: list[str],
    config: SplitConfig | None = None,
    session_fn=session_from_filename,
) -> dict[str, list[str]]:
    """Split so that no weld session is in more than one of train/val/test.

    Returns ``{"train": [...], "val": [...], "test": [...]}``.
    Pass ``session_fn=roboflow_session_from_filename`` for Roboflow-augmented data.
    """
    config = config or SplitConfig()
    assert abs(config.train_frac + config.val_frac + config.test_frac - 1.0) < 1e-6, \
        "split fractions must sum to 1.0"

    import random
    rng = random.Random(config.random_seed)

    groups = group_by_session(filenames, session_fn=session_fn)
    sessions = sorted(groups.keys())
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = int(n * config.train_frac)
    n_val = int(n * config.val_frac)

    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train:n_train + n_val]
    test_sessions = sessions[n_train + n_val:]

    out = {"train": [], "val": [], "test": []}
    for s in train_sessions:
        out["train"].extend(groups[s])
    for s in val_sessions:
        out["val"].extend(groups[s])
    for s in test_sessions:
        out["test"].extend(groups[s])

    logger.info("split: %d sessions / %d images -> train=%d val=%d test=%d",
                len(sessions), len(filenames), len(out["train"]), len(out["val"]), len(out["test"]))
    return out
