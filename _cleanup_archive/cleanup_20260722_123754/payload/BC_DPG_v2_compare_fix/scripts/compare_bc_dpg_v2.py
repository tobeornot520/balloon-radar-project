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


def metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, math.nan)
    return float(value) if value is not None else math.nan


def reduction(raw_pfa: float, calibrated_pfa: float) -> float:
    if raw_pfa <= 0:
        return math.nan
    return (raw_pfa - calibrated_pfa) / raw_pfa


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BC-DPG-FCN v2 with its frozen raw DPG baseline. "
            "This version does not require BC-DPG v1 results."
        )
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
            "bc_dpg_v2_comparison.csv"
        ),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []

    for fold in args.folds:
        fold_text = f"{fold:02d}"
        suffix = "_smoke" if args.smoke else ""
        summary_path = (
            PROJECT_ROOT
            / "results/experiments"
            / (
                f"bc_dpg_v2_tail_v4_fold{fold_text}_seed42"
                f"{suffix}"
            )
            / "tables/summary.json"
        )
        summary = read_json(summary_path)

        raw_selected = summary["raw_test_metrics"]
        bc_selected = summary["test_metrics"]
        raw_base = summary["raw_base_threshold_test_metrics"]
        bc_base = summary["base_threshold_test_metrics"]

        raw_selected_pfa = metric(raw_selected, "pfa")
        bc_selected_pfa = metric(bc_selected, "pfa")
        raw_base_pfa = metric(raw_base, "pfa")
        bc_base_pfa = metric(bc_base, "pfa")

        rows.append(
            {
                "fold": fold,
                "best_epoch": int(summary["best_epoch"]),
                "base_threshold": float(summary["base_threshold"]),
                "selected_threshold": float(
                    summary["selected_threshold"]
                ),
                "raw_selected_pd": metric(
                    raw_selected,
                    "joint_pd",
                ),
                "raw_selected_pfa": raw_selected_pfa,
                "raw_selected_auc": metric(
                    raw_selected,
                    "roc_auc",
                ),
                "bc_selected_pd": metric(
                    bc_selected,
                    "joint_pd",
                ),
                "bc_selected_pfa": bc_selected_pfa,
                "bc_selected_auc": metric(
                    bc_selected,
                    "roc_auc",
                ),
                "selected_pfa_reduction": reduction(
                    raw_selected_pfa,
                    bc_selected_pfa,
                ),
                "raw_base_pd": metric(raw_base, "joint_pd"),
                "raw_base_pfa": raw_base_pfa,
                "bc_base_pd": metric(bc_base, "joint_pd"),
                "bc_base_pfa": bc_base_pfa,
                "base_pfa_reduction": reduction(
                    raw_base_pfa,
                    bc_base_pfa,
                ),
                "score_never_increased": bool(
                    summary["score_never_increased_test"]
                ),
                "shift_background_mean": float(
                    summary["shift_statistics_test"][
                        "background_mean"
                    ]
                ),
                "shift_positive_mean": float(
                    summary["shift_statistics_test"][
                        "positive_mean"
                    ]
                ),
                "summary_path": str(summary_path),
            }
        )

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
        "best_epoch",
        "base_threshold",
        "raw_selected_pd",
        "raw_selected_pfa",
        "bc_selected_pd",
        "bc_selected_pfa",
        "selected_pfa_reduction",
        "raw_base_pd",
        "raw_base_pfa",
        "bc_base_pd",
        "bc_base_pfa",
        "base_pfa_reduction",
        "score_never_increased",
    ]
    print(frame[display_columns].to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
