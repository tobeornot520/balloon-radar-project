#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fixed-notch versus fixed-residual paired decisions"
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/zero_doppler_mechanism_v1",
    )
    return parser.parse_args()


def load_experiment(fold: int, mode: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = (
        PROJECT_ROOT
        / "results/experiments"
        / f"zero_doppler_v1_{mode}_fold{fold:02d}_seed{seed}"
        / "tables"
    )
    predictions = pd.read_csv(directory / "test_predictions.csv")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return predictions, summary


def paired_fold_audit(
    fixed: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    fold: int,
    selected_epoch: int,
) -> dict[str, Any]:
    required = {
        "sample_id",
        "target_present",
        "false_alarm",
        "correct_detection",
        "pred_range_index",
        "pred_velocity_index",
        "score",
    }
    for label, frame in (("fixed", fixed), ("residual", residual)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions are missing columns: {sorted(missing)}")
    paired = fixed.merge(
        residual,
        on="sample_id",
        suffixes=("_fixed", "_residual"),
        validate="one_to_one",
    )
    if len(paired) != len(fixed) or len(paired) != len(residual):
        raise ValueError("fixed and residual predictions do not cover identical samples")
    if not paired["target_present_fixed"].equals(paired["target_present_residual"]):
        raise ValueError("target labels differ between paired predictions")
    background = paired["target_present_fixed"].eq(0)
    target = ~background
    peak_changed = (
        paired["pred_range_index_fixed"].ne(paired["pred_range_index_residual"])
        | paired["pred_velocity_index_fixed"].ne(
            paired["pred_velocity_index_residual"]
        )
    )
    fixed_false_alarms = int((background & paired["false_alarm_fixed"]).sum())
    residual_false_alarms = int(
        (background & paired["false_alarm_residual"]).sum()
    )
    fixed_hits = int((target & paired["correct_detection_fixed"]).sum())
    residual_hits = int((target & paired["correct_detection_residual"]).sum())
    contract = bool(
        (paired["score_residual"] <= paired["score_fixed"] + 1e-7).all()
    )
    return {
        "fold": int(fold),
        "selected_epoch": int(selected_epoch),
        "background_count": int(background.sum()),
        "target_count": int(target.sum()),
        "fixed_false_alarm_count": fixed_false_alarms,
        "residual_false_alarm_count": residual_false_alarms,
        "background_removed": int(
            (
                background
                & paired["false_alarm_fixed"]
                & ~paired["false_alarm_residual"]
            ).sum()
        ),
        "background_added": int(
            (
                background
                & ~paired["false_alarm_fixed"]
                & paired["false_alarm_residual"]
            ).sum()
        ),
        "fixed_joint_hit_count": fixed_hits,
        "residual_joint_hit_count": residual_hits,
        "target_joint_lost": int(
            (
                target
                & paired["correct_detection_fixed"]
                & ~paired["correct_detection_residual"]
            ).sum()
        ),
        "target_joint_gained": int(
            (
                target
                & ~paired["correct_detection_fixed"]
                & paired["correct_detection_residual"]
            ).sum()
        ),
        "background_peak_changed": int((background & peak_changed).sum()),
        "target_peak_changed": int((target & peak_changed).sum()),
        "mean_score_delta": float(
            (paired["score_residual"] - paired["score_fixed"]).mean()
        ),
        "nonincrease_contract_satisfied": contract,
        "fold_gate_pass": bool(
            residual_false_alarms <= fixed_false_alarms
            and residual_hits >= fixed_hits
            and contract
        ),
    }


def aggregate_gate(detail: pd.DataFrame) -> dict[str, Any]:
    fixed_false_alarms = int(detail["fixed_false_alarm_count"].sum())
    residual_false_alarms = int(detail["residual_false_alarm_count"].sum())
    fixed_hits = int(detail["fixed_joint_hit_count"].sum())
    residual_hits = int(detail["residual_joint_hit_count"].sum())
    pooled_strict_improvement = residual_false_alarms < fixed_false_alarms
    return {
        "folds": detail["fold"].astype(int).tolist(),
        "fixed_false_alarm_count": fixed_false_alarms,
        "residual_false_alarm_count": residual_false_alarms,
        "false_alarm_delta": residual_false_alarms - fixed_false_alarms,
        "fixed_joint_hit_count": fixed_hits,
        "residual_joint_hit_count": residual_hits,
        "joint_hit_delta": residual_hits - fixed_hits,
        "background_removed": int(detail["background_removed"].sum()),
        "background_added": int(detail["background_added"].sum()),
        "target_joint_lost": int(detail["target_joint_lost"].sum()),
        "target_joint_gained": int(detail["target_joint_gained"].sum()),
        "background_peak_changed": int(detail["background_peak_changed"].sum()),
        "target_peak_changed": int(detail["target_peak_changed"].sum()),
        "all_fold_gates_pass": bool(detail["fold_gate_pass"].all()),
        "pooled_false_alarms_strictly_improve": pooled_strict_improvement,
        "gate_pass": bool(detail["fold_gate_pass"].all() and pooled_strict_improvement),
        "claim_warning": "consumed development folds; not blind evidence",
    }


def make_report(detail: pd.DataFrame, aggregate: dict[str, Any], label: str) -> str:
    return "\n".join(
        [
            "# Fixed-notch residual V2 paired audit",
            "",
            f"Output label: `{label}`.",
            "",
            "All comparisons use paired test samples from consumed development folds.",
            "Passing this gate is not external blind evidence.",
            "",
            "## Aggregate decision",
            "",
            "```json",
            json.dumps(aggregate, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Fold detail",
            "",
            detail.to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    if not args.folds or any(fold not in range(1, 7) for fold in args.folds):
        raise ValueError("folds must be in 1-6")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.label):
        raise ValueError("label contains unsafe characters")
    rows = []
    for fold in args.folds:
        fixed, _ = load_experiment(fold, "fixed_notch", args.seed)
        residual, residual_summary = load_experiment(
            fold, "fixed_residual", args.seed
        )
        rows.append(
            paired_fold_audit(
                fixed,
                residual,
                fold=fold,
                selected_epoch=int(residual_summary["selected_epoch"]),
            )
        )
    detail = pd.DataFrame(rows)
    aggregate = aggregate_gate(detail)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / f"paired_{args.label}.csv", index=False)
    (output_dir / f"gate_{args.label}.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"PAIRED_{args.label}.md").write_text(
        make_report(detail, aggregate, args.label), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
