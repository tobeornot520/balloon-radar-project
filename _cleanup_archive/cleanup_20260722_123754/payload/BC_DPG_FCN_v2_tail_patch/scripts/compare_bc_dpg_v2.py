#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = []
    for fold in args.folds:
        fold_text = f"{fold:02d}"
        suffix = "_smoke" if args.smoke else ""

        v1 = read_json(
            PROJECT_ROOT
            / "results/experiments"
            / f"bc_dpg_fcn_v4_fold{fold_text}_seed42"
            / "tables/summary.json"
        )
        v2 = read_json(
            PROJECT_ROOT
            / "results/experiments"
            / (
                f"bc_dpg_v2_tail_v4_fold{fold_text}_seed42"
                f"{suffix}"
            )
            / "tables/summary.json"
        )

        base_threshold = float(v2["base_threshold"])
        raw = v2["raw_base_threshold_test_metrics"]
        v1_fixed = v1["test_metrics"]
        v2_fixed = v2["base_threshold_test_metrics"]

        rows.append(
            {
                "fold": fold,
                "base_threshold": base_threshold,
                "raw_pd": raw["joint_pd"],
                "raw_pfa": raw["pfa"],
                "v1_selected_pd": v1_fixed["joint_pd"],
                "v1_selected_pfa": v1_fixed["pfa"],
                "v2_base_pd": v2_fixed["joint_pd"],
                "v2_base_pfa": v2_fixed["pfa"],
                "v2_pfa_reduction_vs_raw": (
                    (raw["pfa"] - v2_fixed["pfa"])
                    / raw["pfa"]
                    if raw["pfa"] > 0
                    else math.nan
                ),
                "v2_score_never_increased": (
                    v2["score_never_increased_test"]
                ),
                "v2_shift_background_mean": (
                    v2["shift_statistics_test"][
                        "background_mean"
                    ]
                ),
                "v2_shift_positive_mean": (
                    v2["shift_statistics_test"][
                        "positive_mean"
                    ]
                ),
            }
        )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    output = (
        PROJECT_ROOT
        / "results/data_audit/bc_dpg_preflight"
        / "bc_dpg_v2_comparison.csv"
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
