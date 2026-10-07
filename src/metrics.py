"""Instance-segmentation evaluation metrics for weld defects.

Thin wrappers around Ultralytics' built-in validation + convenience helpers for
per-class breakdown and confusion-matrix reporting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassMetrics:
    name: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    support: int

    def as_dict(self) -> dict:
        return {
            "class": self.name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "mAP50": round(self.map50, 4),
            "mAP50-95": round(self.map50_95, 4),
            "support": self.support,
        }


@dataclass
class RunSummary:
    overall_map50: float
    overall_map50_95: float
    per_class: list[ClassMetrics]

    def as_markdown_table(self) -> str:
        header = "| Class | Precision | Recall | mAP@50 | mAP@50-95 | Support |\n"
        header += "|---|---|---|---|---|---|\n"
        rows = []
        for c in self.per_class:
            rows.append(f"| {c.name} | {c.precision:.3f} | {c.recall:.3f} | "
                        f"{c.map50:.3f} | {c.map50_95:.3f} | {c.support} |")
        return header + "\n".join(rows)
