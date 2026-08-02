#!/usr/bin/env python3
"""Validate and summarize human annotations for the zero-Doppler review queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/zero_doppler_human_review_summary_v1"
REVIEW_STATUSES = {"pending", "reviewed", "needs_more_context", "unavailable"}
VISIBLE_PATTERNS = {
    "unreviewed",
    "near_zero_doppler_peak",
    "multiple_peaks",
    "broad_structure",
    "edge_peak",
    "no_clear_pattern",
}
INDEPENDENT_EVIDENCE_SOURCE = "independent_scene_record"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a manually reviewed zero-Doppler queue."
    )
    parser.add_argument(
        "--reviewed-queue",
        type=Path,
        required=True,
        help="A separately saved CSV; never overwrite review_queue.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip()


def validate_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold",
        "sample_id",
        "review_priority",
        "review_status",
        "visible_pattern",
        "physical_class",
        "evidence_source",
        "review_note",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"review queue missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("review queue must contain at least one row")
    normalized = frame.copy()
    for column in (
        "sample_id",
        "review_priority",
        "review_status",
        "visible_pattern",
        "physical_class",
        "evidence_source",
        "review_note",
    ):
        normalized[column] = normalized_text(normalized, column)
    if normalized.duplicated(["fold", "sample_id"]).any():
        raise ValueError("review queue has duplicate (fold, sample_id) rows")

    errors: list[str] = []
    invalid_status = sorted(set(normalized["review_status"]) - REVIEW_STATUSES)
    if invalid_status:
        errors.append(f"invalid review_status values: {invalid_status}")
    invalid_patterns = sorted(set(normalized["visible_pattern"]) - VISIBLE_PATTERNS)
    if invalid_patterns:
        errors.append(f"invalid visible_pattern values: {invalid_patterns}")

    reviewed = normalized["review_status"].eq("reviewed")
    incomplete = normalized["review_status"].isin({"needs_more_context", "unavailable"})
    named_physical = normalized["physical_class"].ne("unknown")
    if (reviewed & normalized["visible_pattern"].eq("unreviewed")).any():
        errors.append("reviewed rows must record a visible_pattern")
    if (reviewed & normalized["review_note"].eq("")).any():
        errors.append("reviewed rows must contain a review_note")
    if (incomplete & normalized["review_note"].eq("")).any():
        errors.append("needs_more_context/unavailable rows must explain the gap")
    if (incomplete & named_physical).any():
        errors.append("incomplete rows must keep physical_class as unknown")
    if (named_physical & normalized["evidence_source"].ne(INDEPENDENT_EVIDENCE_SOURCE)).any():
        errors.append(
            "named physical_class requires evidence_source=independent_scene_record"
        )
    if (named_physical & normalized["review_note"].eq("")).any():
        errors.append("named physical_class requires a review_note")
    if (normalized["review_status"].eq("pending") & named_physical).any():
        errors.append("pending rows must keep physical_class as unknown")
    if errors:
        raise ValueError("; ".join(errors))
    return normalized


def summarize_reviews(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    status_counts = frame["review_status"].value_counts().sort_index()
    priority_summary = (
        frame.groupby(["review_priority", "review_status", "visible_pattern"], dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values(["review_priority", "review_status", "visible_pattern"])
    )
    named_labels = frame[frame["physical_class"].ne("unknown")].copy()
    named_labels = named_labels[
        [
            "fold",
            "sample_id",
            "review_priority",
            "physical_class",
            "evidence_source",
            "review_note",
        ]
    ].sort_values(["fold", "sample_id"])
    pending = int(status_counts.get("pending", 0))
    summary = {
        "status": "COMPLETE" if pending == 0 else "INCOMPLETE",
        "queue_count": int(len(frame)),
        "review_status_counts": {key: int(value) for key, value in status_counts.items()},
        "reviewed_count": int(status_counts.get("reviewed", 0)),
        "pending_count": pending,
        "named_physical_label_count": int(len(named_labels)),
        "claim_boundary": (
            "human visible-pattern review only; named physical labels require an "
            "independent scene record and do not establish calibrated polarimetry "
            "or blind-test model evidence"
        ),
    }
    return summary, priority_summary, named_labels


def make_readme(summary: dict[str, Any]) -> str:
    return f"""# Zero-Doppler Human Review Summary V1

Status: `{summary['status']}`. The submitted queue contains {summary['queue_count']} rows,
with {summary['reviewed_count']} reviewed and {summary['pending_count']} pending.

This output reports human-visible patterns only. It is not a physical background
taxonomy and cannot promote consumed development evidence to a blind-test claim.

## Validation rules

- `reviewed` needs a non-default visible pattern and a review note.
- `needs_more_context` and `unavailable` need a note explaining the gap.
- `physical_class=unknown` is the default.
- Any named physical class needs both a note and
  `evidence_source=independent_scene_record`.

## Files

- `priority_pattern_summary.csv`: review completion and visible-pattern counts;
- `named_physical_labels.csv`: only rows with an independently supported physical label;
- `summary.json`: validation outcome and claim boundary.
"""


def audit_reviews(
    *, reviewed_queue: Path, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    reviewed_queue = resolve_path(reviewed_queue)
    output_dir = resolve_path(output_dir)
    if not reviewed_queue.is_file():
        raise FileNotFoundError(f"reviewed queue not found: {reviewed_queue}")
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = validate_reviews(pd.read_csv(reviewed_queue, encoding="utf-8-sig"))
    summary, priority_summary, named_labels = summarize_reviews(frame)
    summary["reviewed_queue_sha256"] = sha256_file(reviewed_queue)
    priority_summary.to_csv(
        output_dir / "priority_pattern_summary.csv", index=False, encoding="utf-8-sig"
    )
    named_labels.to_csv(
        output_dir / "named_physical_labels.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(make_readme(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = audit_reviews(
        reviewed_queue=args.reviewed_queue,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("Zero-Doppler human review audit: PASS")
    print(f"status={summary['status']}")
    print(f"reviewed_count={summary['reviewed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
