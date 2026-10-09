"""Tests for the weld-session-stratified train/val/test split."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_split import (
    SplitConfig,
    group_by_session,
    roboflow_session_from_filename,
    session_from_filename,
    session_stratified_split,
)


def test_session_from_filename_strips_numeric_suffix():
    assert session_from_filename("session_01_42.jpg") == "session_01"
    assert session_from_filename("weld_A2B_7.png") == "weld_A2B"
    assert session_from_filename("single_session.jpg") == "single_session"


def test_session_from_filename_handles_mixed_case_extensions():
    assert session_from_filename("abc_1.JPG") == "abc"
    assert session_from_filename("def_123.Png") == "def"


def test_group_by_session_collects_all_frames():
    files = ["sess1_0.jpg", "sess1_1.jpg", "sess1_2.jpg", "sess2_0.jpg", "sess2_1.jpg"]
    groups = group_by_session(files)
    assert set(groups["sess1"]) == {"sess1_0.jpg", "sess1_1.jpg", "sess1_2.jpg"}
    assert set(groups["sess2"]) == {"sess2_0.jpg", "sess2_1.jpg"}


def test_split_no_session_appears_in_more_than_one_bucket():
    # 10 sessions × 5 frames each
    files = [f"sess{i:02d}_{j}.jpg" for i in range(10) for j in range(5)]
    split = session_stratified_split(files, SplitConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=0))

    def sessions_in(bucket: list[str]) -> set[str]:
        return {session_from_filename(f) for f in bucket}

    train_s = sessions_in(split["train"])
    val_s = sessions_in(split["val"])
    test_s = sessions_in(split["test"])

    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)


def test_split_preserves_total_frame_count():
    files = [f"sess{i}_{j}.jpg" for i in range(20) for j in range(3)]
    split = session_stratified_split(files)
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(files)


def test_split_is_deterministic_given_seed():
    files = [f"sess{i}_0.jpg" for i in range(50)]
    split1 = session_stratified_split(files, SplitConfig(random_seed=7))
    split2 = session_stratified_split(files, SplitConfig(random_seed=7))
    assert split1 == split2


def test_split_fractions_must_sum_to_one():
    files = ["a_0.jpg"]
    with pytest.raises(AssertionError):
        session_stratified_split(files, SplitConfig(train_frac=0.8, val_frac=0.1, test_frac=0.2))


def test_roboflow_session_strips_augmentation_hash():
    # Real Kaggle filenames from bemorekgg/welding-defect-segmentation-dataset
    assert roboflow_session_from_filename(
        "0e538f39-1d30436c-c23_jpg.rf.05ff78024b481ae4fada24ef439ebd97.jpg"
    ) == "0e538f39-1d30436c-c23_jpg"
    assert roboflow_session_from_filename(
        "116e2e9b-6b433071-download_2_jpeg_jpg.rf.476154d28d24d7450321e4ae5b581f2e.png"
    ) == "116e2e9b-6b433071-download_2_jpeg_jpg"


def test_roboflow_session_passthrough_on_non_roboflow_names():
    # Should not crash on filenames without an .rf. segment
    assert roboflow_session_from_filename("plain_image.jpg") == "plain_image"


def test_split_with_roboflow_session_fn_keeps_augmentations_together():
    # Two source images, each with 3 augmentation variants
    files = [
        f"source_A_jpg.rf.{h}.jpg" for h in ["aaaaaaaa", "bbbbbbbb", "cccccccc"]
    ] + [
        f"source_B_jpg.rf.{h}.jpg" for h in ["dddddddd", "eeeeeeee", "ffffffff"]
    ]
    split = session_stratified_split(
        files,
        SplitConfig(train_frac=0.5, val_frac=0.5, test_frac=0.0, random_seed=0),
        session_fn=roboflow_session_from_filename,
    )

    def source_of(fname: str) -> str:
        return roboflow_session_from_filename(fname)

    for bucket in ("train", "val", "test"):
        sources_in_bucket = {source_of(f) for f in split[bucket]}
        # Each source either entirely in or entirely out of the bucket
        for src in sources_in_bucket:
            source_files = [f for f in files if source_of(f) == src]
            assert all(f in split[bucket] for f in source_files), \
                f"source {src!r} was split across buckets"
