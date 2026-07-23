#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reduction(raw_pfa: float, calibrated_pfa: float) -> float:
    if raw_pfa <= 0:
        return math.nan
    return (raw_pfa - calibrated_pfa) / raw_pfa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[2, 4],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = []
    for fold in args.folds:
        fold_text = f"{fold:02d}"
        suffix = "_smoke" if args.smoke else ""
        path = (
            PROJECT_ROOT
            / "results/experiments"
            / (
                f"bc_dpg_v3_scan_target_v4_fold"
                f"{fold_text}_seed42{suffix}"
            )
            / "tables/summary.json"
        )
        summary = read_json(path)

        raw_base = summary[
            "raw_base_threshold_test_metrics"
        ]
        v3_base = summary[
            "base_threshold_test_metrics"
        ]
        raw_selected = summary["raw_test_metrics"]
        v3_selected = summary["test_metrics"]

        rows.append(
            {
                "fold": fold,
                "best_epoch": summary["best_epoch"],
                "base_threshold": summary["base_threshold"],
                "raw_base_pd": raw_base["joint_pd"],
                "v3_base_pd": v3_base["joint_pd"],
                "base_pd_change": (
                    v3_base["joint_pd"]
                    - raw_base["joint_pd"]
                ),
                "raw_base_pfa": raw_base["pfa"],
                "v3_base_pfa": v3_base["pfa"],
                "base_pfa_reduction": reduction(
                    raw_base["pfa"],
                    v3_base["pfa"],
                ),
                "raw_selected_pd": raw_selected["joint_pd"],
                "v3_selected_pd": v3_selected["joint_pd"],
                "selected_pd_change": (
                    v3_selected["joint_pd"]
                    - raw_selected["joint_pd"]
                ),
                "raw_selected_pfa": raw_selected["pfa"],
                "v3_selected_pfa": v3_selected["pfa"],
                "selected_pfa_reduction": reduction(
                    raw_selected["pfa"],
                    v3_selected["pfa"],
                ),
                "background_shift_mean": (
                    summary["shift_statistics_test"][
                        "background_mean"
                    ]
                ),
                "target_shift_mean": (
                    summary["shift_statistics_test"][
                        "target_mean"
                    ]
                ),
                "background_probability_mean": (
                    summary[
                        "background_probability_statistics_test"
                    ]["background_mean"]
                ),
                "target_probability_mean": (
                    summary[
                        "background_probability_statistics_test"
                    ]["target_mean"]
                ),
                "pd_floor_satisfied": (
                    summary["pd_floor_validation"]["satisfied"]
                ),
                "score_never_increased": (
                    summary["score_never_increased_test"]
                ),
            }
        )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    output = (
        PROJECT_ROOT
        / "results/data_audit/bc_dpg_preflight"
        / "bc_dpg_v3_fold24_comparison.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
