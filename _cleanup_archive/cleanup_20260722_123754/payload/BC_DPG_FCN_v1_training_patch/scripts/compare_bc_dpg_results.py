#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline DPG-FCN and BC-DPG-FCN results"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 4],
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "results/data_audit/bc_dpg_preflight/"
            "bc_dpg_fold14_comparison.csv"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def value(
    mapping: dict[str, Any],
    key: str,
) -> float:
    raw = mapping.get(key, math.nan)
    return float(raw) if raw is not None else math.nan


def main() -> None:
    args = parse_args()
    rows = []

    for fold in args.folds:
        fold_text = f"{fold:02d}"
        suffix = "_smoke" if args.smoke else ""

        baseline_path = (
            PROJECT_ROOT
            / "results/experiments"
            / f"dpg_fcn_v4_fold{fold_text}_seed42"
            / "tables/summary.json"
        )
        bc_path = (
            PROJECT_ROOT
            / "results/experiments"
            / (
                f"bc_dpg_fcn_v4_fold{fold_text}_seed42"
                f"{suffix}"
            )
            / "tables/summary.json"
        )

        baseline = read_json(baseline_path)
        bc = read_json(bc_path)

        baseline_metrics = baseline["test_metrics"]
        bc_metrics = bc["test_metrics"]
        raw_metrics = bc["raw_test_metrics"]
        fixed_metrics = bc["fixed_test_metrics"]
        raw_fixed_metrics = bc[
            "raw_fixed_test_metrics"
        ]

        row = {
            "fold": fold,
            "baseline_summary": str(baseline_path),
            "bc_summary": str(bc_path),
            "baseline_pd": value(
                baseline_metrics,
                "joint_pd",
            ),
            "baseline_pfa": value(
                baseline_metrics,
                "pfa",
            ),
            "baseline_auc": value(
                baseline_metrics,
                "roc_auc",
            ),
            "bc_pd": value(bc_metrics, "joint_pd"),
            "bc_pfa": value(bc_metrics, "pfa"),
            "bc_auc": value(bc_metrics, "roc_auc"),
            "raw_recomputed_pd": value(
                raw_metrics,
                "joint_pd",
            ),
            "raw_recomputed_pfa": value(
                raw_metrics,
                "pfa",
            ),
            "raw_recomputed_auc": value(
                raw_metrics,
                "roc_auc",
            ),
            "fixed_threshold": float(
                bc["fixed_threshold"]
            ),
            "fixed_bc_pd": value(
                fixed_metrics,
                "joint_pd",
            ),
            "fixed_bc_pfa": value(
                fixed_metrics,
                "pfa",
            ),
            "fixed_raw_pd": value(
                raw_fixed_metrics,
                "joint_pd",
            ),
            "fixed_raw_pfa": value(
                raw_fixed_metrics,
                "pfa",
            ),
            "bc_minus_baseline_pd": (
                value(bc_metrics, "joint_pd")
                - value(baseline_metrics, "joint_pd")
            ),
            "bc_minus_baseline_pfa": (
                value(bc_metrics, "pfa")
                - value(baseline_metrics, "pfa")
            ),
            "bc_minus_baseline_auc": (
                value(bc_metrics, "roc_auc")
                - value(baseline_metrics, "roc_auc")
            ),
            "fixed_pfa_reduction_fraction": (
                (
                    value(raw_fixed_metrics, "pfa")
                    - value(fixed_metrics, "pfa")
                )
                / value(raw_fixed_metrics, "pfa")
                if value(raw_fixed_metrics, "pfa") > 0
                else math.nan
            ),
            "argmax_preserved_test": bool(
                bc["argmax_preserved_test"]
            ),
            "best_epoch": int(bc["best_epoch"]),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "fold",
        "baseline_pd",
        "baseline_pfa",
        "baseline_auc",
        "bc_pd",
        "bc_pfa",
        "bc_auc",
        "fixed_raw_pfa",
        "fixed_bc_pfa",
        "fixed_pfa_reduction_fraction",
        "argmax_preserved_test",
    ]
    print(frame[display_columns].to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
