#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a zero-Doppler candidate veto on frozen out-of-fold DPG outputs"
    )
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/zero_doppler_candidate_veto_v1",
    )
    parser.add_argument("--maximum-radius", type=int, default=12)
    parser.add_argument("--center-index", type=int, default=64)
    parser.add_argument("--maximum-joint-pd-drop", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_oof_predictions(experiment_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in range(1, 7):
        table_dir = (
            experiment_root
            / f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed42"
            / "tables"
        )
        prediction_path = table_dir / "base_threshold_test_predictions.csv"
        summary_path = table_dir / "summary.json"
        if not prediction_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing frozen Fold {fold} evidence")
        frame = pd.read_csv(prediction_path, encoding="utf-8-sig")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        frame = frame.copy()
        frame["fold"] = fold
        frame["frozen_threshold"] = float(summary["base_threshold"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    required = {
        "sample_id",
        "scan_group",
        "target_present",
        "raw_score",
        "pred_velocity_index",
        "localization_ok",
        "frozen_threshold",
    }
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"prediction tables missing columns: {sorted(missing)}")
    if combined["sample_id"].duplicated().any():
        raise ValueError("out-of-fold predictions contain duplicate sample IDs")
    combined["raw_detected"] = combined["raw_score"] >= combined["frozen_threshold"]
    combined["raw_joint_hit"] = (
        combined["target_present"].eq(1)
        & combined["raw_detected"]
        & combined["localization_ok"].astype(bool)
    )
    return combined


def evaluate_radius(frame: pd.DataFrame, radius: int, center_index: int) -> dict[str, Any]:
    if radius < -1:
        raise ValueError("radius must be -1 for baseline or nonnegative")
    vetoed = (
        pd.Series(False, index=frame.index)
        if radius == -1
        else frame["pred_velocity_index"].sub(center_index).abs().le(radius)
    )
    post_detected = frame["raw_detected"] & ~vetoed
    background = frame["target_present"].eq(0)
    target = frame["target_present"].eq(1)
    post_joint = target & post_detected & frame["localization_ok"].astype(bool)
    background_count = int(background.sum())
    target_count = int(target.sum())
    false_alarms = int((background & post_detected).sum())
    joint_hits = int(post_joint.sum())
    return {
        "candidate_veto_half_width_bins": radius,
        "background_count": background_count,
        "target_count": target_count,
        "false_alarm_count": false_alarms,
        "pooled_pfa": false_alarms / background_count,
        "score_detection_count": int((target & post_detected).sum()),
        "joint_hit_count": joint_hits,
        "joint_pd": joint_hits / target_count,
        "removed_false_alarm_count": int(
            (background & frame["raw_detected"] & vetoed).sum()
        ),
        "lost_joint_hit_count": int((frame["raw_joint_hit"] & vetoed).sum()),
        "vetoed_background_candidate_count": int((background & vetoed).sum()),
        "vetoed_target_candidate_count": int((target & vetoed).sum()),
    }


def group_metrics(frame: pd.DataFrame, radius: int, center_index: int) -> pd.DataFrame:
    background = frame.loc[frame["target_present"].eq(0)].copy()
    background["vetoed"] = (
        False
        if radius == -1
        else background["pred_velocity_index"].sub(center_index).abs().le(radius)
    )
    background["post_detected"] = background["raw_detected"] & ~background["vetoed"]
    rows = []
    for scan_group, group in background.groupby("scan_group", observed=True):
        false_alarms = int(group["post_detected"].sum())
        rows.append(
            {
                "candidate_veto_half_width_bins": radius,
                "scan_group": str(scan_group),
                "sample_count": int(len(group)),
                "false_alarm_count": false_alarms,
                "pfa": false_alarms / len(group),
                "removed_false_alarm_count": int(
                    (group["raw_detected"] & group["vetoed"]).sum()
                ),
                "vetoed_candidate_count": int(group["vetoed"].sum()),
            }
        )
    return pd.DataFrame(rows)


def make_report(summary: dict[str, Any], selected: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Zero-Doppler Candidate-veto Audit V1",
            "",
            "## Evidence boundary",
            "",
            "This is a development-only audit of frozen out-of-fold candidate tables.",
            "It is not a trained notch model, does not select a secondary heatmap peak,",
            "and does not create a new blind test.",
            "",
            "## Main result",
            "",
            f"- Samples: {summary['sample_count']}",
            f"- Baseline false alarms: {summary['baseline_false_alarm_count']}",
            f"- Baseline joint hits: {summary['baseline_joint_hit_count']}",
            f"- Largest radius without observed joint-hit loss: {summary['largest_no_loss_radius_bins']} bins",
            f"- Selected radius under the 1-point joint-Pd constraint: {summary['constraint_selected_radius_bins']} bins",
            "",
            selected.to_markdown(index=False, floatfmt=".4f"),
            "",
            "The sharp target-loss increase beyond the selected boundary requires",
            "target-protected dense supervision or learned clutter-aware suppression.",
            "A fixed veto must not be frozen as the final detector from these data.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    if args.maximum_radius < 0:
        raise ValueError("maximum-radius must be nonnegative")
    if not 0.0 <= args.maximum_joint_pd_drop <= 1.0:
        raise ValueError("maximum-joint-pd-drop must be in [0, 1]")
    experiment_root = resolve_path(args.experiment_root)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    frame = load_oof_predictions(experiment_root)
    radii = [-1, *range(args.maximum_radius + 1)]
    aggregate = pd.DataFrame(
        [evaluate_radius(frame, radius, args.center_index) for radius in radii]
    )
    groups = pd.concat(
        [group_metrics(frame, radius, args.center_index) for radius in radii],
        ignore_index=True,
    )
    worst_group = groups.groupby("candidate_veto_half_width_bins", observed=True)[
        "pfa"
    ].max()
    aggregate["worst_background_scan_pfa"] = aggregate[
        "candidate_veto_half_width_bins"
    ].map(worst_group)
    baseline = aggregate.loc[
        aggregate["candidate_veto_half_width_bins"].eq(-1)
    ].iloc[0]
    aggregate["joint_pd_drop"] = float(baseline["joint_pd"]) - aggregate["joint_pd"]
    candidates = aggregate.loc[
        aggregate["candidate_veto_half_width_bins"].ge(0)
        & aggregate["joint_pd_drop"].le(args.maximum_joint_pd_drop + 1e-12)
    ]
    selected = candidates.sort_values(
        ["worst_background_scan_pfa", "pooled_pfa", "candidate_veto_half_width_bins"]
    ).iloc[0]
    no_loss = aggregate.loc[
        aggregate["candidate_veto_half_width_bins"].ge(0)
        & aggregate["lost_joint_hit_count"].eq(0)
    ]
    largest_no_loss = int(no_loss["candidate_veto_half_width_bins"].max())

    aggregate.to_csv(output_dir / "radius_tradeoff.csv", index=False)
    groups.to_csv(output_dir / "background_scan_tradeoff.csv", index=False)
    selected_rows = aggregate.loc[
        aggregate["candidate_veto_half_width_bins"].isin(
            [-1, largest_no_loss, int(selected["candidate_veto_half_width_bins"])]
        ),
        [
            "candidate_veto_half_width_bins",
            "false_alarm_count",
            "pooled_pfa",
            "worst_background_scan_pfa",
            "joint_hit_count",
            "joint_pd",
            "lost_joint_hit_count",
        ],
    ]
    summary = {
        "status": "COMPLETE_DEVELOPMENT_ONLY_DIAGNOSTIC",
        "sample_count": int(len(frame)),
        "target_count": int(frame["target_present"].sum()),
        "background_count": int(frame["target_present"].eq(0).sum()),
        "baseline_false_alarm_count": int(baseline["false_alarm_count"]),
        "baseline_joint_hit_count": int(baseline["joint_hit_count"]),
        "largest_no_loss_radius_bins": largest_no_loss,
        "constraint_selected_radius_bins": int(selected["candidate_veto_half_width_bins"]),
        "maximum_joint_pd_drop": float(args.maximum_joint_pd_drop),
        "selected_metrics": selected.to_dict(),
        "claim_limits": [
            "candidate veto is not full heatmap notching",
            "all six outer folds are consumed diagnostic evidence",
            "no new blind-test claim",
            "target and background dates are confounded",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / "REPORT.md").write_text(
        make_report(summary, selected_rows), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
