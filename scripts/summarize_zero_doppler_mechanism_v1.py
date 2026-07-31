#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize zero-Doppler mechanism runs")
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/zero_doppler_mechanism_v1",
    )
    return parser.parse_args()


def experiment_name(fold: int, mode: str, seed: int, smoke: bool) -> str:
    suffix = "_smoke" if smoke else ""
    return f"zero_doppler_v1_{mode}_fold{fold:02d}_seed{seed}{suffix}"


def load_detail(
    folds: list[int], modes: list[str], seed: int, smoke: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for mode in modes:
            name = experiment_name(fold, mode, seed, smoke)
            path = PROJECT_ROOT / "results/experiments" / name / "tables/summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing mechanism summary: {path}")
            with path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            for split in ("validation", "test"):
                metrics = summary[f"{split}_metrics"]
                rows.append(
                    {
                        "fold": fold,
                        "mode": mode,
                        "split": split,
                        "sample_count": metrics["sample_count"],
                        "positive_count": metrics["positive_count"],
                        "background_count": metrics["background_count"],
                        "false_alarm_count": metrics["false_alarm_count"],
                        "pfa": metrics["pfa"],
                        "joint_hit_count": metrics["correct_detection_count"],
                        "joint_pd": metrics["joint_pd"],
                        "roc_auc": metrics["roc_auc"],
                        "frozen_threshold": summary["frozen_threshold"],
                        "trainable_parameter_count": summary[
                            "trainable_parameter_count"
                        ],
                        "selected_epoch": summary.get("selected_epoch"),
                        "nonincrease_contract_applies": summary[
                            "nonincrease_contract_applies"
                        ],
                        "nonincrease_contract_satisfied": summary[
                            "nonincrease_contract_satisfied"
                        ],
                        "debug_per_class": summary["debug_per_class"],
                    }
                )
    return pd.DataFrame(rows)


def aggregate_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, split), group in detail.groupby(["mode", "split"], observed=True):
        background_count = int(group["background_count"].sum())
        positive_count = int(group["positive_count"].sum())
        false_alarms = int(group["false_alarm_count"].sum())
        joint_hits = int(group["joint_hit_count"].sum())
        rows.append(
            {
                "mode": mode,
                "split": split,
                "fold_count": int(group["fold"].nunique()),
                "background_count": background_count,
                "positive_count": positive_count,
                "false_alarm_count": false_alarms,
                "pooled_pfa": false_alarms / background_count,
                "worst_fold_pfa": float(group["pfa"].max()),
                "joint_hit_count": joint_hits,
                "pooled_joint_pd": joint_hits / positive_count,
                "worst_fold_joint_pd": float(group["joint_pd"].min()),
                "mean_auc": float(group["roc_auc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "mode"])


def make_report(detail: pd.DataFrame, aggregate: pd.DataFrame, smoke: bool) -> str:
    test_rows = aggregate.loc[aggregate["split"].eq("test")]
    return "\n".join(
        [
            "# Zero-Doppler Mechanism V1 Run Summary",
            "",
            "## Scope",
            "",
            (
                "Mechanical smoke only. Debug subsets verify the real checkpoint, "
                "training, threshold, localization, and output paths; they do not "
                "support mechanism selection."
                if smoke
                else "Development-only grouped runs; no outer fold is blind evidence."
            ),
            "",
            "All modes reuse each fold's frozen base-DPG validation threshold.",
            "No threshold is selected from a test scan.",
            "",
            "## Test-path mechanics",
            "",
            test_rows.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Fold detail",
            "",
            detail.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = load_detail(args.folds, args.modes, args.seed, args.smoke)
    aggregate = aggregate_detail(detail)
    suffix = "smoke" if args.smoke else "development"
    detail.to_csv(output_dir / f"detail_{suffix}.csv", index=False)
    aggregate.to_csv(output_dir / f"aggregate_{suffix}.csv", index=False)
    (output_dir / f"REPORT_{suffix}.md").write_text(
        make_report(detail, aggregate, args.smoke), encoding="utf-8"
    )
    status = {
        "status": "COMPLETE_MECHANICAL_SMOKE" if args.smoke else "COMPLETE_DEVELOPMENT_RUN",
        "folds": args.folds,
        "modes": args.modes,
        "seed": args.seed,
        "row_count": int(len(detail)),
        "claim_warning": (
            "debug subsets cannot support mechanism selection"
            if args.smoke
            else "consumed folds are development evidence only"
        ),
    }
    (output_dir / f"status_{suffix}.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
