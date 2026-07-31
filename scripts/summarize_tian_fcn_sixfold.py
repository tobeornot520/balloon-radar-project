#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen-threshold Tian FCN results across folds"
    )
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--channels", nargs="+", choices=("H", "V", "HV"), required=True
    )
    parser.add_argument("--experiment-root", default="results/experiments")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def experiment_name(channel: str, fold: int, scope: str) -> str:
    return (
        f"tian_fcn_reproduction_v1_{channel.lower()}_"
        f"fold{fold:02d}_seed42_{scope}"
    )


def finite_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    if any(fold < 1 or fold > 6 for fold in args.folds):
        raise ValueError("folds must be between 1 and 6")
    experiment_root = resolve_path(args.experiment_root)
    output_dir = resolve_path(
        args.output_dir
        or f"results/reproduction/tian_fcn_v1/{args.scope}"
    )

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for channel in args.channels:
        for fold in args.folds:
            name = experiment_name(channel, fold, args.scope)
            summary_path = experiment_root / name / "tables" / "summary.json"
            if not summary_path.is_file():
                missing.append(str(summary_path))
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if payload.get("paper_metric_definition") != "tian_2024_set_euclidean_v2":
                raise RuntimeError(
                    "stale Tian metric definition; rerun only after validation "
                    f"localization is repaired: {summary_path}"
                )
            if payload.get("scope") != args.scope:
                raise RuntimeError(f"scope mismatch in {summary_path}")
            if int(payload.get("fold_id", -1)) != fold:
                raise RuntimeError(f"fold mismatch in {summary_path}")
            if str(payload.get("channel")) != channel:
                raise RuntimeError(f"channel mismatch in {summary_path}")

            if args.scope == "formal":
                if not payload.get("test_split_loaded"):
                    raise RuntimeError(f"formal run did not load test: {summary_path}")
                evaluation = payload["test"]
                project_metrics = evaluation[
                    "project_protocol_locked_threshold"
                ]
            else:
                if payload.get("test_split_loaded"):
                    raise RuntimeError(f"smoke unexpectedly loaded test: {summary_path}")
                evaluation = payload["validation"]
                project_metrics = evaluation[
                    "project_protocol_selected_threshold"
                ]
            paper_metrics = evaluation["paper_alignment_dynamic_pir"]
            rows.append(
                {
                    "scope": args.scope,
                    "channel": channel,
                    "fold": fold,
                    "validation_threshold": float(payload["validation_threshold"]),
                    "positive_count": int(project_metrics["positive_count"]),
                    "background_count": int(project_metrics["background_count"]),
                    "detected_positive_count": int(
                        project_metrics["detected_positive_count"]
                    ),
                    "correct_detection_count": int(
                        project_metrics["correct_detection_count"]
                    ),
                    "false_alarm_count": int(project_metrics["false_alarm_count"]),
                    "paper_pd": finite_or_nan(paper_metrics["paper_pd"]),
                    "paper_pf": finite_or_nan(paper_metrics["paper_pf"]),
                    "paper_d_min_euclidean_cells": finite_or_nan(
                        paper_metrics["paper_d_min_euclidean_cells"]
                    ),
                    "paper_d_5_euclidean_cells": finite_or_nan(
                        paper_metrics["paper_d_5_euclidean_cells"]
                    ),
                    "paper_d_avg_euclidean_cells": finite_or_nan(
                        paper_metrics["paper_d_avg_euclidean_cells"]
                    ),
                    "joint_pd": finite_or_nan(project_metrics["joint_pd"]),
                    "pfa": finite_or_nan(project_metrics["pfa"]),
                    "range_mae_gates": finite_or_nan(
                        project_metrics["range_mae_gates"]
                    ),
                    "velocity_mae_bins": finite_or_nan(
                        project_metrics["velocity_mae_bins"]
                    ),
                    "seconds_per_map": finite_or_nan(evaluation["seconds_per_map"]),
                }
            )

    if missing and args.require_all:
        raise FileNotFoundError("missing result summaries:\n" + "\n".join(missing))
    if not rows:
        raise RuntimeError("no completed Tian FCN runs were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["channel", "fold"])
    frame.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    aggregate: dict[str, Any] = {
        "scope": args.scope,
        "threshold_source": "fold-specific validation only",
        "test_threshold_retuning": False,
        "requested_folds": args.folds,
        "requested_channels": args.channels,
        "missing_summaries": missing,
        "channels": {},
    }
    metric_names = (
        "paper_pd",
        "paper_pf",
        "paper_d_min_euclidean_cells",
        "paper_d_5_euclidean_cells",
        "paper_d_avg_euclidean_cells",
        "joint_pd",
        "pfa",
        "range_mae_gates",
        "velocity_mae_bins",
        "seconds_per_map",
    )
    for channel, channel_frame in frame.groupby("channel", sort=True):
        channel_summary: dict[str, Any] = {
            "completed_folds": channel_frame["fold"].astype(int).tolist(),
            "fold_count": int(len(channel_frame)),
            "macro": {},
        }
        for metric in metric_names:
            values = channel_frame[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            channel_summary["macro"][metric] = {
                "mean": float(np.mean(finite)) if finite.size else math.nan,
                "std": (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan
                ),
                "min": float(np.min(finite)) if finite.size else math.nan,
                "max": float(np.max(finite)) if finite.size else math.nan,
            }

        positive_count = int(channel_frame["positive_count"].sum())
        background_count = int(channel_frame["background_count"].sum())
        detected_positive = int(channel_frame["detected_positive_count"].sum())
        correct = int(channel_frame["correct_detection_count"].sum())
        false_alarms = int(channel_frame["false_alarm_count"].sum())
        channel_summary["pooled_project"] = {
            "positive_count": positive_count,
            "background_count": background_count,
            "detected_positive_count": detected_positive,
            "correct_detection_count": correct,
            "false_alarm_count": false_alarms,
            "joint_pd": correct / positive_count if positive_count else math.nan,
            "pfa": false_alarms / background_count if background_count else math.nan,
        }
        worst_pd = channel_frame.loc[channel_frame["joint_pd"].idxmin()]
        worst_pfa = channel_frame.loc[channel_frame["pfa"].idxmax()]
        channel_summary["worst_fold"] = {
            "joint_pd_fold": int(worst_pd["fold"]),
            "joint_pd": float(worst_pd["joint_pd"]),
            "pfa_fold": int(worst_pfa["fold"]),
            "pfa": float(worst_pfa["pfa"]),
        }
        aggregate["channels"][channel] = channel_summary

    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(json_safe(aggregate), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(aggregate), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
