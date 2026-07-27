#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_roi_bc_dpg_joint_tables_v1 import (  # noqa: E402
    FOLDS,
    build_joint_tables,
    display_path,
    ensure_writable_output,
    resolve_output_dir,
)


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "data_audit"
    / "final_roi_bc_dpg_joint_v2_base_threshold"
)

MODE_COLUMNS = {
    "bc_dpg_v3": ("bc_false_alarm", "bc_correct"),
    "roi_baseline": ("roi_base_false_alarm", "roi_base_correct"),
    "roi_power_control": ("roi_power_false_alarm", "roi_power_correct"),
    "roi_ri4": ("roi_ri4_false_alarm", "roi_ri4_correct"),
}

COMPARISONS = {
    "roi_power_control": ("roi_power_false_alarm", "roi_power_correct"),
    "roi_ri4": ("roi_ri4_false_alarm", "roi_ri4_correct"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed-threshold six-fold ROI and BC-DPG audit."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generated files in an existing output directory to be replaced.",
    )
    return parser.parse_args()


def detection_metrics(
    frame: pd.DataFrame,
    fold_label: str,
    model: str,
) -> dict[str, object]:
    false_alarm_column, correct_column = MODE_COLUMNS[model]
    labels = frame["target_present"].astype(int)
    background_count = int(labels.eq(0).sum())
    target_count = int(labels.eq(1).sum())
    false_alarm_count = int(frame[false_alarm_column].astype(bool).sum())
    correct_count = int(frame[correct_column].astype(bool).sum())
    return {
        "fold": fold_label,
        "model": model,
        "background_samples": background_count,
        "target_samples": target_count,
        "false_alarms": false_alarm_count,
        "pfa": false_alarm_count / background_count if background_count else float("nan"),
        "correct_detections": correct_count,
        "joint_pd": correct_count / target_count if target_count else float("nan"),
    }


def complementarity_metrics(
    frame: pd.DataFrame,
    fold_label: str,
    comparison: str,
) -> dict[str, object]:
    roi_fa_column, roi_correct_column = COMPARISONS[comparison]
    labels = frame["target_present"].astype(int)
    background = labels.eq(0)
    target = labels.eq(1)

    bc_fa = frame["bc_false_alarm"].astype(bool) & background
    roi_fa = frame[roi_fa_column].astype(bool) & background
    bc_correct = frame["bc_correct"].astype(bool) & target
    roi_correct = frame[roi_correct_column].astype(bool) & target

    return {
        "fold": fold_label,
        "comparison": comparison,
        "bc_false_alarms": int(bc_fa.sum()),
        "roi_false_alarms": int(roi_fa.sum()),
        "shared_false_alarms": int((bc_fa & roi_fa).sum()),
        "bc_only_false_alarms": int((bc_fa & ~roi_fa).sum()),
        "roi_only_false_alarms": int((roi_fa & ~bc_fa).sum()),
        "fa_union": int((bc_fa | roi_fa).sum()),
        "target_samples": int(target.sum()),
        "shared_correct": int((bc_correct & roi_correct).sum()),
        "bc_only_correct": int((bc_correct & ~roi_correct).sum()),
        "roi_only_correct": int((roi_correct & ~bc_correct).sum()),
        "correct_union": int((bc_correct | roi_correct).sum()),
        "neither_correct": int((target & ~bc_correct & ~roi_correct).sum()),
    }


def build_summaries(
    fold_tables: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detection_rows: list[dict[str, object]] = []
    complementarity_rows: list[dict[str, object]] = []

    for fold, table in fold_tables.items():
        fold_label = f"{fold:02d}"
        for model in MODE_COLUMNS:
            detection_rows.append(detection_metrics(table, fold_label, model))
        for comparison in COMPARISONS:
            complementarity_rows.append(
                complementarity_metrics(table, fold_label, comparison)
            )

    combined = pd.concat(fold_tables.values(), ignore_index=True)
    for model in MODE_COLUMNS:
        detection_rows.append(detection_metrics(combined, "ALL", model))
    for comparison in COMPARISONS:
        complementarity_rows.append(
            complementarity_metrics(combined, "ALL", comparison)
        )

    return pd.DataFrame(detection_rows), pd.DataFrame(complementarity_rows)


def main() -> None:
    args = parse_args()
    folds = tuple(dict.fromkeys(args.folds))
    output_dir = resolve_output_dir(args.output_dir)
    ensure_writable_output(output_dir, args.overwrite)

    tables, alignment = build_joint_tables(folds, ("test",))
    fold_tables = {fold: tables[(fold, "test")] for fold in folds}

    for fold, table in fold_tables.items():
        table.to_csv(output_dir / f"fold_{fold:02d}_joint_test.csv", index=False)

    combined = pd.concat(fold_tables.values(), ignore_index=True)
    combined.to_csv(output_dir / "all_fold_joint_predictions.csv", index=False)

    detection, complementarity = build_summaries(fold_tables)
    detection.to_csv(output_dir / "detection_comparison.csv", index=False)
    complementarity.to_csv(
        output_dir / "complementarity_summary.csv", index=False
    )

    alignment = alignment.drop(columns=["split", "exact_scan_group_alignment"])
    alignment.to_csv(output_dir / "fold_alignment.csv", index=False)

    status = {
        "status": "ok",
        "bc_decision_source": "base_threshold_test_predictions.csv",
        "roi_decision_source": "refined_fixed_* columns from test_predictions.csv",
        "folds": list(folds),
        "rows": int(len(combined)),
        "exact_alignment": True,
        "test_threshold_retuning": False,
        "output_dir": display_path(output_dir),
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(detection[detection["fold"].eq("ALL")].to_string(index=False))
    print(f"rows={len(combined)}")
    print("exact_alignment=True")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
