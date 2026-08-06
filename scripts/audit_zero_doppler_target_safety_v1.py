#!/usr/bin/env python3
"""Audit paired target behavior for fixed-notch versus residual predictions."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_zero_doppler_false_alarm_library_v1 import (
    assert_sanitized,
    bool_series,
    load_config as load_prediction_contract,
    load_predictions,
    resolve_path,
    sha256_file,
    verify_hash,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/zero_doppler_target_safety_audit_v1.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/data_audit/zero_doppler_target_safety_audit_v1"
)
TARGET_COLUMNS = {
    "localization_ok",
    "correct_detection",
    "range_error_gates",
    "velocity_error_bins",
    "true_range_index",
    "true_velocity_index",
}
LOCAL_ONLY_COLUMNS = {"sample_id", "source_file"}
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit target-side safety of the frozen zero-Doppler residual."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_audit_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing target safety config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "audit_id",
        "prediction_contract",
        "prediction_contract_sha256",
        "expected_counts",
        "score_comparison_atol",
        "large_peak_shift_threshold_bins",
        "claim_boundaries",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"target safety config missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    if float(config["score_comparison_atol"]) < 0:
        raise ValueError("score_comparison_atol must be nonnegative")
    if int(config["large_peak_shift_threshold_bins"]) <= 0:
        raise ValueError("large peak-shift threshold must be positive")
    if any(config["claim_boundaries"].values()):
        raise ValueError("all target safety claim-boundary flags must remain false")
    return config


def transition(before: pd.Series, after: pd.Series, label: str) -> pd.Series:
    before = before.astype(bool)
    after = after.astype(bool)
    output = pd.Series(f"retained_not_{label}", index=before.index, dtype="string")
    output.loc[before & after] = f"retained_{label}"
    output.loc[before & ~after] = f"lost_{label}"
    output.loc[~before & after] = f"gained_{label}"
    return output


def pair_target_predictions(
    fixed: pd.DataFrame, residual: pd.DataFrame, fold: int
) -> pd.DataFrame:
    for role, frame in (("fixed", fixed), ("residual", residual)):
        missing = TARGET_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{role} fold {fold} missing target columns: {sorted(missing)}")
    identity = ["sample_id", "source_file", "target_present"]
    fixed_identity = fixed[identity].sort_values("sample_id").reset_index(drop=True)
    residual_identity = residual[identity].sort_values("sample_id").reset_index(drop=True)
    if not fixed_identity.equals(residual_identity):
        raise ValueError(f"fold {fold} fixed/residual sample identities differ")

    fixed_target = fixed[fixed["target_present"].eq(1)].copy()
    residual_target = residual[residual["target_present"].eq(1)].copy()
    keep = [
        "sample_id",
        "source_file",
        "score",
        "raw_score",
        "pred_range_index",
        "pred_velocity_index",
        "true_range_index",
        "true_velocity_index",
        "range_error_gates",
        "velocity_error_bins",
        "detected",
        "localization_ok",
        "correct_detection",
    ]
    paired = fixed_target[keep].merge(
        residual_target[keep],
        on=["sample_id", "source_file"],
        how="inner",
        validate="one_to_one",
        suffixes=("_fixed", "_residual"),
    )
    if len(paired) != len(fixed_target):
        raise ValueError(f"fold {fold} target pairing lost rows")
    if not paired["true_range_index_fixed"].equals(
        paired["true_range_index_residual"]
    ) or not paired["true_velocity_index_fixed"].equals(
        paired["true_velocity_index_residual"]
    ):
        raise ValueError(f"fold {fold} fixed/residual target labels differ")
    paired.insert(0, "fold", fold)
    for column in (
        "detected_fixed",
        "detected_residual",
        "localization_ok_fixed",
        "localization_ok_residual",
        "correct_detection_fixed",
        "correct_detection_residual",
    ):
        paired[column] = bool_series(paired[column], f"target_safety.{column}")
    paired["score_delta_residual_minus_fixed"] = (
        paired["score_residual"] - paired["score_fixed"]
    )
    paired["abs_range_peak_shift_bins"] = (
        paired["pred_range_index_residual"] - paired["pred_range_index_fixed"]
    ).abs()
    paired["abs_velocity_peak_shift_bins"] = (
        paired["pred_velocity_index_residual"]
        - paired["pred_velocity_index_fixed"]
    ).abs()
    paired["range_error_delta_bins"] = (
        paired["range_error_gates_residual"] - paired["range_error_gates_fixed"]
    )
    paired["velocity_error_delta_bins"] = (
        paired["velocity_error_bins_residual"]
        - paired["velocity_error_bins_fixed"]
    )
    paired["detection_transition"] = transition(
        paired["detected_fixed"], paired["detected_residual"], "detected"
    )
    paired["localization_transition"] = transition(
        paired["localization_ok_fixed"], paired["localization_ok_residual"], "localized"
    )
    paired["joint_transition"] = transition(
        paired["correct_detection_fixed"],
        paired["correct_detection_residual"],
        "joint_success",
    )
    return paired


def comparison_counts(cases: pd.DataFrame, config: dict[str, Any]) -> dict[str, int]:
    tolerance = float(config["score_comparison_atol"])
    score_delta = cases["score_delta_residual_minus_fixed"]
    peak_shift = cases[
        ["abs_range_peak_shift_bins", "abs_velocity_peak_shift_bins"]
    ].max(axis=1)
    large_threshold = int(config["large_peak_shift_threshold_bins"])
    range_delta = cases["range_error_delta_bins"]
    velocity_delta = cases["velocity_error_delta_bins"]
    return {
        "target_samples": int(len(cases)),
        "target_source_identifiers": int(cases["source_file"].nunique()),
        "fixed_detected": int(cases["detected_fixed"].sum()),
        "residual_detected": int(cases["detected_residual"].sum()),
        "detection_lost": int(cases["detection_transition"].eq("lost_detected").sum()),
        "detection_gained": int(cases["detection_transition"].eq("gained_detected").sum()),
        "fixed_localization_ok": int(cases["localization_ok_fixed"].sum()),
        "residual_localization_ok": int(cases["localization_ok_residual"].sum()),
        "localization_lost": int(
            cases["localization_transition"].eq("lost_localized").sum()
        ),
        "localization_gained": int(
            cases["localization_transition"].eq("gained_localized").sum()
        ),
        "fixed_joint_success": int(cases["correct_detection_fixed"].sum()),
        "residual_joint_success": int(cases["correct_detection_residual"].sum()),
        "joint_success_lost": int(
            cases["joint_transition"].eq("lost_joint_success").sum()
        ),
        "joint_success_gained": int(
            cases["joint_transition"].eq("gained_joint_success").sum()
        ),
        "peak_changed": int(peak_shift.gt(0).sum()),
        "large_peak_shift_over_10_bins": int(peak_shift.gt(large_threshold).sum()),
        "score_decreased": int(score_delta.lt(-tolerance).sum()),
        "score_equal": int(score_delta.abs().le(tolerance).sum()),
        "score_increased": int(score_delta.gt(tolerance).sum()),
        "range_error_worsened": int(range_delta.gt(0).sum()),
        "range_error_improved": int(range_delta.lt(0).sum()),
        "velocity_error_worsened": int(velocity_delta.gt(0).sum()),
        "velocity_error_improved": int(velocity_delta.lt(0).sum()),
    }


def fold_summary(cases: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tolerance = float(config["score_comparison_atol"])
    for fold, group in cases.groupby("fold", sort=True):
        score_delta = group["score_delta_residual_minus_fixed"]
        peak_changed = (
            group[["abs_range_peak_shift_bins", "abs_velocity_peak_shift_bins"]]
            .max(axis=1)
            .gt(0)
        )
        rows.append(
            {
                "fold": int(fold),
                "target_samples": int(len(group)),
                "fixed_detected": int(group["detected_fixed"].sum()),
                "residual_detected": int(group["detected_residual"].sum()),
                "detection_lost": int(
                    group["detection_transition"].eq("lost_detected").sum()
                ),
                "detection_gained": int(
                    group["detection_transition"].eq("gained_detected").sum()
                ),
                "fixed_localization_ok": int(group["localization_ok_fixed"].sum()),
                "residual_localization_ok": int(
                    group["localization_ok_residual"].sum()
                ),
                "fixed_joint_success": int(group["correct_detection_fixed"].sum()),
                "residual_joint_success": int(
                    group["correct_detection_residual"].sum()
                ),
                "joint_success_lost": int(
                    group["joint_transition"].eq("lost_joint_success").sum()
                ),
                "joint_success_gained": int(
                    group["joint_transition"].eq("gained_joint_success").sum()
                ),
                "peak_changed": int(peak_changed.sum()),
                "score_decreased": int(score_delta.lt(-tolerance).sum()),
                "score_equal": int(score_delta.abs().le(tolerance).sum()),
                "score_increased": int(score_delta.gt(tolerance).sum()),
                "score_delta_min": float(score_delta.min()),
                "score_delta_mean": float(score_delta.mean()),
                "score_delta_median": float(score_delta.median()),
                "max_abs_range_peak_shift_bins": int(
                    group["abs_range_peak_shift_bins"].max()
                ),
                "max_abs_velocity_peak_shift_bins": int(
                    group["abs_velocity_peak_shift_bins"].max()
                ),
                "range_error_worsened": int(group["range_error_delta_bins"].gt(0).sum()),
                "range_error_improved": int(group["range_error_delta_bins"].lt(0).sum()),
                "velocity_error_worsened": int(
                    group["velocity_error_delta_bins"].gt(0).sum()
                ),
                "velocity_error_improved": int(
                    group["velocity_error_delta_bins"].lt(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def peak_shift_histogram(cases: pd.DataFrame) -> pd.DataFrame:
    max_shift = cases[
        ["abs_range_peak_shift_bins", "abs_velocity_peak_shift_bins"]
    ].max(axis=1)
    bands = pd.cut(
        max_shift,
        bins=[-1, 0, 1, 10, np.inf],
        labels=["unchanged", "one_bin", "two_to_ten_bins", "over_ten_bins"],
    )
    frame = cases.assign(shift_band=bands)
    summary = (
        frame.groupby("shift_band", observed=False)
        .agg(
            target_count=("fold", "size"),
            fixed_detected=("detected_fixed", "sum"),
            residual_detected=("detected_residual", "sum"),
            fixed_joint_success=("correct_detection_fixed", "sum"),
            residual_joint_success=("correct_detection_residual", "sum"),
        )
        .reset_index()
    )
    summary["detection_lost"] = [
        int(
            frame.loc[frame["shift_band"].eq(band), "detection_transition"]
            .eq("lost_detected")
            .sum()
        )
        for band in summary["shift_band"]
    ]
    return summary


def score_quantiles(cases: pd.DataFrame) -> pd.DataFrame:
    delta = cases["score_delta_residual_minus_fixed"]
    return pd.DataFrame(
        {
            "quantile": QUANTILES,
            "score_delta_residual_minus_fixed": [
                float(delta.quantile(value)) for value in QUANTILES
            ],
        }
    )


def make_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return f"""# Zero-Doppler Target Safety Audit V1

Status: `COMPLETE_AS_DEVELOPMENT_AUDIT`.

The audit pairs {counts['target_samples']} frozen target predictions. Joint
success remains {counts['fixed_joint_success']} to
{counts['residual_joint_success']}, but raw detection changes from
{counts['fixed_detected']} to {counts['residual_detected']}: one already
mislocalized target loses its detected state. Six target peaks move, including
two with a maximum axis shift greater than 10 bins.

This is post-hoc consumed development evidence. It does not retune a threshold,
establish deployment safety, identify a physical mechanism, or replace a new
same-condition locked evaluation.

`target_case_library_local.csv` contains identifiers and remains local. Only
the fold summary, shift histogram, score quantiles, summary and this README are
shareable aggregates.
"""


def audit_target_safety(
    *, config_path: Path, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    output_dir = resolve_path(output_dir)
    config = load_audit_config(config_path)
    prediction_contract_path = resolve_path(Path(config["prediction_contract"]))
    verify_hash(
        prediction_contract_path,
        config["prediction_contract_sha256"],
        "prediction contract",
    )
    prediction_contract = load_prediction_contract(prediction_contract_path)
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_folds: list[pd.DataFrame] = []
    verified_inputs: dict[str, Any] = {
        "audit_config_sha256": sha256_file(config_path),
        "prediction_contract_sha256": sha256_file(prediction_contract_path),
        "fixed_predictions": {},
        "residual_predictions": {},
    }
    for fold in prediction_contract["folds"]:
        key = str(fold)
        fixed_path = resolve_path(
            Path(prediction_contract["fixed_prediction_template"].format(fold=fold))
        )
        residual_path = resolve_path(
            Path(prediction_contract["residual_prediction_template"].format(fold=fold))
        )
        verified_inputs["fixed_predictions"][key] = verify_hash(
            fixed_path,
            prediction_contract["input_sha256"]["fixed_predictions"][key],
            f"fixed fold {fold}",
        )
        verified_inputs["residual_predictions"][key] = verify_hash(
            residual_path,
            prediction_contract["input_sha256"]["residual_predictions"][key],
            f"residual fold {fold}",
        )
        paired_folds.append(
            pair_target_predictions(
                load_predictions(fixed_path, fold, "fixed"),
                load_predictions(residual_path, fold, "residual"),
                fold,
            )
        )
    cases = pd.concat(paired_folds, ignore_index=True)
    counts = comparison_counts(cases, config)
    expected = {key: int(value) for key, value in config["expected_counts"].items()}
    if counts != expected:
        differences = {
            key: {"expected": expected.get(key), "actual": counts.get(key)}
            for key in sorted(set(expected) | set(counts))
            if expected.get(key) != counts.get(key)
        }
        raise ValueError(f"frozen target safety count mismatch: {differences}")
    folds = fold_summary(cases, config)
    shifts = peak_shift_histogram(cases)
    quantiles = score_quantiles(cases)
    for role, frame in (
        ("fold target safety summary", folds),
        ("peak shift histogram", shifts),
        ("score delta quantiles", quantiles),
    ):
        assert_sanitized(frame, role)
        forbidden = LOCAL_ONLY_COLUMNS & set(frame.columns)
        if forbidden:
            raise ValueError(f"{role} exposes local-only fields: {sorted(forbidden)}")

    lost = cases[cases["detection_transition"].eq("lost_detected")]
    summary = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "COMPLETE_AS_DEVELOPMENT_AUDIT",
        "counts": counts,
        "score_delta": {
            "minimum": float(cases["score_delta_residual_minus_fixed"].min()),
            "mean": float(cases["score_delta_residual_minus_fixed"].mean()),
            "maximum": float(cases["score_delta_residual_minus_fixed"].max()),
        },
        "detection_loss_context": {
            "count": int(len(lost)),
            "fixed_joint_success_count": int(lost["correct_detection_fixed"].sum()),
            "residual_joint_success_count": int(
                lost["correct_detection_residual"].sum()
            ),
            "sample_identifiers_published": False,
        },
        "input_sha256": verified_inputs,
        "claim_boundaries": config["claim_boundaries"],
        "sharing_boundary": {
            "local_only_file": "target_case_library_local.csv",
            "shareable_aggregate_files": [
                "fold_target_safety_summary.csv",
                "peak_shift_histogram.csv",
                "score_delta_quantiles.csv",
                "summary.json",
                "README.md",
            ],
            "forbidden_row_level_fields": sorted(LOCAL_ONLY_COLUMNS),
        },
    }
    cases.to_csv(
        output_dir / "target_case_library_local.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(
        output_dir / "fold_target_safety_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shifts.to_csv(
        output_dir / "peak_shift_histogram.csv", index=False, encoding="utf-8-sig"
    )
    quantiles.to_csv(
        output_dir / "score_delta_quantiles.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(make_readme(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = audit_target_safety(
        config_path=args.config, output_dir=args.output_dir, overwrite=args.overwrite
    )
    counts = summary["counts"]
    print("Zero-Doppler target safety audit: PASS")
    print(f"status={summary['status']}")
    print(
        "joint_success="
        f"{counts['fixed_joint_success']}->{counts['residual_joint_success']}"
    )
    print(f"raw_detected={counts['fixed_detected']}->{counts['residual_detected']}")
    print(f"peak_changed={counts['peak_changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
