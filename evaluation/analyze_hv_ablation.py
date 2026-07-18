from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "final"
    / "hv_ablation.csv"
)

MODE_LABELS = {
    "H": "H-only",
    "V": "V-only",
    "HV": "H+V",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总H、V、HV三组Dual FCN消融结果"
    )
    parser.add_argument(
        "--prefix",
        default="hv_ablation_dual_",
        help="实验目录名前缀",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到实验汇总：{path}"
        )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def extract_record(
    channel: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = summary["test_metrics"]
    return {
        "Input": MODE_LABELS[channel],
        "Channel_Mode": channel,
        "Experiment": summary.get(
            "experiment_name", ""
        ),
        "Best_Epoch": summary.get(
            "best_epoch"
        ),
        "Range_MAE_gate": metrics[
            "mean_range_error"
        ],
        "Velocity_MAE_bin": metrics[
            "mean_velocity_error"
        ],
        "Strict_Hit_Rate_pct": (
            metrics["strict_hit_rate"] * 100.0
        ),
        "Relaxed_Hit_Rate_pct": (
            metrics["relaxed_hit_rate"] * 100.0
        ),
        "Application_Hit_Rate_pct": (
            metrics["application_hit_rate"] * 100.0
        ),
        "Zero_False_Peak_Count": metrics.get(
            "zero_false_peak_count"
        ),
        "Training_Seconds": summary.get(
            "training_seconds"
        ),
    }


def main() -> None:
    args = parse_arguments()
    records = []

    for channel in ["H", "V", "HV"]:
        summary_path = (
            PROJECT_ROOT
            / "results"
            / "experiments"
            / f"{args.prefix}{channel}"
            / "tables"
            / "summary.json"
        )
        summary = load_summary(summary_path)
        records.append(
            extract_record(channel, summary)
        )

    dataframe = pd.DataFrame(records)
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("H/V/HV消融结果")
    print("=" * 80)
    print(dataframe.to_string(index=False))
    print(f"\n结果文件：{args.output}")


if __name__ == "__main__":
    main()
