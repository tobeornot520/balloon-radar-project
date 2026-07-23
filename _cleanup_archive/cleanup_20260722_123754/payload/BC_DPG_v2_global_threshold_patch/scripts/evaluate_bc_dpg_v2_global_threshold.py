#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    apply_threshold_and_metrics,
    json_safe,
    select_threshold_at_false_alarm_budget,
)


SCAN_PATTERN = re.compile(r"^(\d{8}_\d{6})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one global validation threshold across BC-DPG v2 folds, "
            "apply it unchanged to all test folds, and analyze scan-level "
            "background false alarms."
        )
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--experiment-template",
        default="bc_dpg_v2_tail_v4_fold{fold:02d}_seed42",
    )
    parser.add_argument(
        "--false-alarms-per-fold",
        type=int,
        default=2,
        help=(
            "Global validation false-alarm budget for the two-per-fold "
            "policy. Default: 2 per fold."
        ),
    )
    parser.add_argument(
        "--target-pfa",
        type=float,
        default=0.05,
        help="Validation Pfa cap for the pfa policy. Default: 0.05.",
    )
    parser.add_argument(
        "--primary-policy",
        choices=("two_per_fold", "pfa05"),
        default="two_per_fold",
        help=(
            "Policy used for scan-level analysis and the main console table."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/data_audit/bc_dpg_global_threshold"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def scan_group(sample_id: str, mat_path: str = "") -> str:
    for candidate in (str(sample_id), Path(str(mat_path)).stem):
        match = SCAN_PATTERN.match(candidate)
        if match:
            return match.group(1)
    return str(sample_id).split("_beam", 1)[0]


def load_fold(
    fold: int,
    experiment_template: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    experiment_name = experiment_template.format(fold=fold)
    table_dir = (
        PROJECT_ROOT
        / "results/experiments"
        / experiment_name
        / "tables"
    )

    val_path = table_dir / "val_predictions.csv"
    test_path = table_dir / "test_predictions.csv"
    summary_path = table_dir / "summary.json"

    for path in (val_path, test_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    val = pd.read_csv(val_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    summary = read_json(summary_path)

    required = {
        "sample_id",
        "target_present",
        "score",
        "raw_score",
        "localization_ok",
    }
    for split_name, frame in (("val", val), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Fold {fold} {split_name} missing columns: {sorted(missing)}"
            )

        frame["fold"] = int(fold)
        frame["split"] = split_name
        frame["row_id"] = [
            f"fold{fold:02d}:{split_name}:{index}"
            for index in range(len(frame))
        ]
        frame["scan_group"] = [
            scan_group(
                sample_id,
                mat_path,
            )
            for sample_id, mat_path in zip(
                frame["sample_id"],
                frame.get(
                    "mat_path",
                    pd.Series([""] * len(frame)),
                ),
            )
        ]

    return val, test, summary


def method_frame(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    result = frame.copy()
    if method == "raw":
        result["score"] = result["raw_score"].astype(float)
    elif method == "bc_v2":
        result["score"] = result["score"].astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")
    return result


def select_policy_threshold(
    frame: pd.DataFrame,
    max_false_alarms: int,
) -> tuple[float, pd.DataFrame]:
    threshold, curve = select_threshold_at_false_alarm_budget(
        frame,
        int(max_false_alarms),
    )
    return float(threshold), curve


def metric_row(
    *,
    method: str,
    policy: str,
    scope: str,
    fold: int | str,
    threshold: float,
    frame: pd.DataFrame,
    tolerance: DetectionTolerance,
) -> tuple[dict[str, Any], pd.DataFrame]:
    results, metrics = apply_threshold_and_metrics(
        frame,
        float(threshold),
        tolerance,
    )
    row = {
        "method": method,
        "policy": policy,
        "scope": scope,
        "fold": fold,
        "threshold": float(threshold),
        **metrics,
    }
    return row, results


def safe_reduction(raw_value: float, bc_value: float) -> float:
    if not np.isfinite(raw_value) or raw_value <= 0:
        return math.nan
    return (raw_value - bc_value) / raw_value


def build_scan_analysis(
    raw_results: pd.DataFrame,
    bc_results: pd.DataFrame,
    raw_threshold: float,
    bc_threshold: float,
) -> pd.DataFrame:
    raw_columns = [
        "row_id",
        "fold",
        "scan_group",
        "sample_id",
        "target_present",
        "score",
        "false_alarm",
    ]
    bc_columns = [
        "row_id",
        "score",
        "false_alarm",
    ]

    raw = raw_results[raw_columns].rename(
        columns={
            "score": "raw_score_global",
            "false_alarm": "raw_false_alarm_global",
        }
    )
    bc = bc_results[bc_columns].rename(
        columns={
            "score": "bc_score_global",
            "false_alarm": "bc_false_alarm_global",
        }
    )

    merged = raw.merge(
        bc,
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    background = merged.loc[
        merged["target_present"].astype(int) == 0
    ].copy()

    rows: list[dict[str, Any]] = []
    for (fold, group), part in background.groupby(
        ["fold", "scan_group"],
        sort=True,
    ):
        raw_count = int(
            part["raw_false_alarm_global"].astype(bool).sum()
        )
        bc_count = int(
            part["bc_false_alarm_global"].astype(bool).sum()
        )
        count = int(len(part))

        rows.append(
            {
                "fold": int(fold),
                "scan_group": group,
                "background_count": count,
                "raw_threshold": float(raw_threshold),
                "bc_threshold": float(bc_threshold),
                "raw_score_mean": float(
                    part["raw_score_global"].mean()
                ),
                "raw_score_max": float(
                    part["raw_score_global"].max()
                ),
                "bc_score_mean": float(
                    part["bc_score_global"].mean()
                ),
                "bc_score_max": float(
                    part["bc_score_global"].max()
                ),
                "raw_false_alarm_count": raw_count,
                "bc_false_alarm_count": bc_count,
                "raw_pfa": raw_count / count if count else math.nan,
                "bc_pfa": bc_count / count if count else math.nan,
                "false_alarm_count_reduction": raw_count - bc_count,
                "pfa_reduction_fraction": (
                    (raw_count - bc_count) / raw_count
                    if raw_count > 0
                    else math.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(
            [
                "raw_false_alarm_count",
                "raw_score_max",
                "background_count",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return result


def duplicate_report(frame: pd.DataFrame) -> dict[str, int]:
    duplicated = frame.duplicated(
        subset=["sample_id"],
        keep=False,
    )
    return {
        "row_count": int(len(frame)),
        "unique_sample_id_count": int(
            frame["sample_id"].nunique()
        ),
        "duplicate_row_count": int(duplicated.sum()),
    }


def main() -> None:
    args = parse_args()

    if args.false_alarms_per_fold < 0:
        raise ValueError("false-alarms-per-fold must be non-negative")
    if not 0.0 < args.target_pfa < 1.0:
        raise ValueError("target-pfa must be between 0 and 1")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    summaries: dict[int, dict[str, Any]] = {}
    tolerances: set[tuple[int, int]] = set()

    for fold in args.folds:
        val, test, summary = load_fold(
            fold,
            args.experiment_template,
        )
        val_frames.append(val)
        test_frames.append(test)
        summaries[fold] = summary

        config = summary.get("config", {})
        tolerances.add(
            (
                int(config.get("range_tolerance_gates", 2)),
                int(config.get("velocity_tolerance_bins", 3)),
            )
        )

    if len(tolerances) != 1:
        raise ValueError(
            f"Folds use inconsistent localization tolerances: {tolerances}"
        )

    range_tolerance, velocity_tolerance = next(iter(tolerances))
    tolerance = DetectionTolerance(
        range_gates=range_tolerance,
        velocity_bins=velocity_tolerance,
    )

    combined_val = pd.concat(
        val_frames,
        ignore_index=True,
    )
    combined_test = pd.concat(
        test_frames,
        ignore_index=True,
    )

    val_background_count = int(
        (combined_val["target_present"].astype(int) == 0).sum()
    )
    policy_budgets = {
        "two_per_fold": int(
            args.false_alarms_per_fold * len(args.folds)
        ),
        "pfa05": int(
            math.floor(args.target_pfa * val_background_count)
        ),
    }

    threshold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    threshold_curves: list[pd.DataFrame] = []
    result_frames: dict[tuple[str, str, str], pd.DataFrame] = {}

    for policy, max_false_alarms in policy_budgets.items():
        for method in ("raw", "bc_v2"):
            val_method = method_frame(combined_val, method)
            test_method = method_frame(combined_test, method)

            threshold, curve = select_policy_threshold(
                val_method,
                max_false_alarms,
            )
            curve = curve.copy()
            curve["method"] = method
            curve["policy"] = policy
            curve["max_false_alarms"] = max_false_alarms
            threshold_curves.append(curve)

            threshold_rows.append(
                {
                    "method": method,
                    "policy": policy,
                    "threshold": threshold,
                    "validation_background_count": (
                        val_background_count
                    ),
                    "max_false_alarms": max_false_alarms,
                    "target_pfa": (
                        max_false_alarms / val_background_count
                        if val_background_count
                        else math.nan
                    ),
                }
            )

            combined_row, combined_results = metric_row(
                method=method,
                policy=policy,
                scope="combined_test",
                fold="all",
                threshold=threshold,
                frame=test_method,
                tolerance=tolerance,
            )
            metric_rows.append(combined_row)
            result_frames[
                (method, policy, "combined_test")
            ] = combined_results

            for fold in args.folds:
                fold_frame = test_method.loc[
                    test_method["fold"] == fold
                ].copy()
                fold_row, fold_results = metric_row(
                    method=method,
                    policy=policy,
                    scope="per_fold_test",
                    fold=fold,
                    threshold=threshold,
                    frame=fold_frame,
                    tolerance=tolerance,
                )
                metric_rows.append(fold_row)
                result_frames[
                    (method, policy, f"fold_{fold:02d}")
                ] = fold_results

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(
        threshold_curves,
        ignore_index=True,
    )

    thresholds.to_csv(
        output_dir / "global_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "global_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_csv(
        output_dir / "global_validation_threshold_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = args.primary_policy
    raw_threshold = float(
        thresholds.loc[
            (thresholds["policy"] == primary)
            & (thresholds["method"] == "raw"),
            "threshold",
        ].iloc[0]
    )
    bc_threshold = float(
        thresholds.loc[
            (thresholds["policy"] == primary)
            & (thresholds["method"] == "bc_v2"),
            "threshold",
        ].iloc[0]
    )

    scan_analysis = build_scan_analysis(
        result_frames[("raw", primary, "combined_test")],
        result_frames[("bc_v2", primary, "combined_test")],
        raw_threshold,
        bc_threshold,
    )
    scan_analysis.to_csv(
        output_dir / "scan_group_false_alarm_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison_rows: list[dict[str, Any]] = []
    for policy in policy_budgets:
        for scope, fold in [
            ("combined_test", "all"),
            *[
                ("per_fold_test", fold_value)
                for fold_value in args.folds
            ],
        ]:
            raw_row = metrics.loc[
                (metrics["method"] == "raw")
                & (metrics["policy"] == policy)
                & (metrics["scope"] == scope)
                & (metrics["fold"].astype(str) == str(fold))
            ].iloc[0]
            bc_row = metrics.loc[
                (metrics["method"] == "bc_v2")
                & (metrics["policy"] == policy)
                & (metrics["scope"] == scope)
                & (metrics["fold"].astype(str) == str(fold))
            ].iloc[0]

            comparison_rows.append(
                {
                    "policy": policy,
                    "scope": scope,
                    "fold": fold,
                    "raw_threshold": float(raw_row["threshold"]),
                    "bc_threshold": float(bc_row["threshold"]),
                    "raw_joint_pd": float(raw_row["joint_pd"]),
                    "bc_joint_pd": float(bc_row["joint_pd"]),
                    "joint_pd_change": (
                        float(bc_row["joint_pd"])
                        - float(raw_row["joint_pd"])
                    ),
                    "raw_pfa": float(raw_row["pfa"]),
                    "bc_pfa": float(bc_row["pfa"]),
                    "pfa_absolute_change": (
                        float(bc_row["pfa"])
                        - float(raw_row["pfa"])
                    ),
                    "pfa_reduction_fraction": safe_reduction(
                        float(raw_row["pfa"]),
                        float(bc_row["pfa"]),
                    ),
                    "raw_roc_auc": float(raw_row["roc_auc"]),
                    "bc_roc_auc": float(bc_row["roc_auc"]),
                    "roc_auc_change": (
                        float(bc_row["roc_auc"])
                        - float(raw_row["roc_auc"])
                    ),
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output_dir / "global_threshold_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_primary = comparison.loc[
        (comparison["policy"] == primary)
        & (comparison["scope"] == "combined_test")
    ].iloc[0]

    summary = {
        "folds": list(args.folds),
        "experiment_template": args.experiment_template,
        "primary_policy": primary,
        "policy_budgets": policy_budgets,
        "localization_tolerance": {
            "range_gates": range_tolerance,
            "velocity_bins": velocity_tolerance,
        },
        "validation_duplicate_report": duplicate_report(
            combined_val
        ),
        "test_duplicate_report": duplicate_report(
            combined_test
        ),
        "primary_combined_test": {
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in combined_primary.to_dict().items()
        },
        "thresholds": threshold_rows,
        "output_files": {
            "thresholds": str(
                output_dir / "global_thresholds.csv"
            ),
            "metrics": str(
                output_dir / "global_threshold_metrics.csv"
            ),
            "comparison": str(
                output_dir / "global_threshold_comparison.csv"
            ),
            "scan_analysis": str(
                output_dir
                / "scan_group_false_alarm_analysis.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("GLOBAL THRESHOLD COMPARISON")
    print("=" * 88)
    display = comparison.loc[
        comparison["policy"] == primary,
        [
            "scope",
            "fold",
            "raw_threshold",
            "bc_threshold",
            "raw_joint_pd",
            "bc_joint_pd",
            "raw_pfa",
            "bc_pfa",
            "pfa_reduction_fraction",
            "raw_roc_auc",
            "bc_roc_auc",
        ],
    ]
    print(display.to_string(index=False))

    print("\n" + "=" * 88)
    print("HARDEST BACKGROUND SCAN GROUPS")
    print("=" * 88)
    if len(scan_analysis):
        print(
            scan_analysis[
                [
                    "fold",
                    "scan_group",
                    "background_count",
                    "raw_false_alarm_count",
                    "bc_false_alarm_count",
                    "raw_pfa",
                    "bc_pfa",
                    "pfa_reduction_fraction",
                ]
            ].head(15).to_string(index=False)
        )
    else:
        print("No background scan groups found.")

    print("\n" + "=" * 88)
    print(f"Saved outputs to: {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
