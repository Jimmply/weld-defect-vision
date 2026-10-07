"""Tests for the evaluation-metrics containers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import ClassMetrics, RunSummary


def test_class_metrics_as_dict_rounds_to_four_decimals():
    cm = ClassMetrics("crack", precision=0.123456, recall=0.789012, map50=0.5, map50_95=0.3, support=42)
    d = cm.as_dict()
    assert d["precision"] == 0.1235
    assert d["recall"] == 0.7890
    assert d["support"] == 42
    assert d["class"] == "crack"


def test_run_summary_markdown_table_has_header_and_rows():
    rs = RunSummary(
        overall_map50=0.65,
        overall_map50_95=0.42,
        per_class=[
            ClassMetrics("crack", 0.8, 0.7, 0.75, 0.5, 100),
            ClassMetrics("porosity", 0.6, 0.55, 0.57, 0.3, 50),
        ],
    )
    md = rs.as_markdown_table()
    assert "| Class |" in md
    assert "crack" in md
    assert "porosity" in md
    assert md.count("\n") >= 3  # header + separator + at least 2 rows
