#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "balloon-radar-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_TABLE = (
    PROJECT_ROOT
    / "results"
    / "final_evidence"
    / "bc_dpg_v3_final"
    / "source_evidence"
    / "ablation_detail.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_evidence"
    / "bc_dpg_localization"
)
RANGE_RESOLUTION_M = 30.0
VELOCITY_RESOLUTION_MPS = 0.183
RANGE_TOLERANCE_GATES = 2
VELOCITY_TOLERANCE_BINS = 3
EXPECTED_FOLDS = tuple(range(1, 7))
EXPECTED_TARGETS = 318
EXPECTED_BACKGROUNDS = 830
EXPECTED_FALSE_ALARMS = 56
EXPECTED_CORRECT_DETECTIONS = 289

PREDICTION_COLUMNS = {
    "sample_id",
    "scan_group",
    "target_present",
    "pred_range_index",
    "pred_velocity_index",
    "true_range_index",
    "true_velocity_index",
    "range_error_gates",
    "velocity_error_bins",
    "localization_ok",
    "split",
    "score",
    "detected",
    "false_alarm",
    "correct_detection",
}
COORDINATE_COLUMNS = (
    "sample_id",
    "target_present",
    "pred_range_index",
    "pred_velocity_index",
    "true_range_index",
    "true_velocity_index",
    "range_error_gates",
    "velocity_error_bins",
    "localization_ok",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic aggregate localization evidence from the frozen "
            "six-fold BC-DPG base-threshold prediction tables."
        )
    )
    parser.add_argument(
        "--selection-table",
        type=Path,
        default=DEFAULT_SELECTION_TABLE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def coerce_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized).difference(allowed))
    if unexpected:
        raise ValueError(f"{name} contains invalid booleans: {unexpected}")
    return normalized.isin({"true", "1"})


def load_frozen_selection(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"fold", "mode", "experiment_name", "base_threshold"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Selection table is missing columns: {missing}")
    selected = frame.loc[
        frame["mode"].eq("full"),
        ["fold", "experiment_name", "base_threshold"],
    ].copy()
    selected["fold"] = selected["fold"].astype(int)
    selected = selected.sort_values("fold").reset_index(drop=True)
    if tuple(selected["fold"]) != EXPECTED_FOLDS:
        raise ValueError("Frozen full-model selection must contain folds 1-6 exactly")
    if selected["experiment_name"].duplicated().any():
        raise ValueError("Frozen full-model experiment names must be unique")
    return selected


def validate_predictions(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing = sorted(PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction table is empty")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Prediction sample_id values must be unique within a fold")
    if not frame["split"].astype(str).eq("test").all():
        raise ValueError("Frozen prediction table must contain only the test split")

    result = frame.copy()
    labels = result["target_present"].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("target_present must contain only 0 and 1")
    result["target_present"] = labels
    for column in ("localization_ok", "detected", "false_alarm", "correct_detection"):
        result[column] = coerce_bool(result[column], column)

    target = result["target_present"].eq(1)
    detected = result["score"].to_numpy(dtype=np.float64) >= float(threshold)
    false_alarm = ~target.to_numpy() & detected

    range_error = result.loc[target, "range_error_gates"].to_numpy(dtype=np.float64)
    velocity_error = result.loc[target, "velocity_error_bins"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(range_error).all() or not np.isfinite(velocity_error).all():
        raise ValueError("Target localization errors must be finite")
    if (range_error < 0).any() or (velocity_error < 0).any():
        raise ValueError("Target localization errors must be nonnegative")

    expected_range_error = np.abs(
        result.loc[target, "pred_range_index"].to_numpy(dtype=np.int64)
        - result.loc[target, "true_range_index"].to_numpy(dtype=np.int64)
    )
    expected_velocity_error = np.abs(
        result.loc[target, "pred_velocity_index"].to_numpy(dtype=np.int64)
        - result.loc[target, "true_velocity_index"].to_numpy(dtype=np.int64)
    )
    if not np.array_equal(range_error, expected_range_error):
        raise ValueError("range_error_gates does not match prediction indices")
    if not np.array_equal(velocity_error, expected_velocity_error):
        raise ValueError("velocity_error_bins does not match prediction indices")

    localization_ok = np.zeros(len(result), dtype=bool)
    localization_ok[target.to_numpy()] = (
        (range_error <= RANGE_TOLERANCE_GATES)
        & (velocity_error <= VELOCITY_TOLERANCE_BINS)
    )
    correct = target.to_numpy() & detected & localization_ok
    checks = {
        "detected": detected,
        "false_alarm": false_alarm,
        "localization_ok": localization_ok,
        "correct_detection": correct,
    }
    for column, expected in checks.items():
        if not np.array_equal(result[column].to_numpy(dtype=bool), expected):
            raise ValueError(f"{column} does not match the frozen evaluation rule")
    return result


def validate_coordinate_alignment(
    calibrated: pd.DataFrame,
    raw: pd.DataFrame,
) -> None:
    calibrated_coordinates = calibrated.loc[:, COORDINATE_COLUMNS].sort_values(
        "sample_id"
    ).reset_index(drop=True)
    raw_coordinates = raw.loc[:, COORDINATE_COLUMNS].sort_values(
        "sample_id"
    ).reset_index(drop=True)
    if calibrated_coordinates.shape != raw_coordinates.shape:
        raise ValueError("Calibrated and raw coordinate tables have different shapes")
    if not calibrated_coordinates["sample_id"].astype(str).equals(
        raw_coordinates["sample_id"].astype(str)
    ):
        raise ValueError("Calibrated and raw coordinate sample IDs do not align")
    for column in COORDINATE_COLUMNS[1:]:
        left = calibrated_coordinates[column].to_numpy()
        right = raw_coordinates[column].to_numpy()
        if pd.api.types.is_bool_dtype(calibrated_coordinates[column]):
            matches = np.array_equal(left.astype(bool), right.astype(bool))
        else:
            matches = np.allclose(
                left.astype(np.float64),
                right.astype(np.float64),
                equal_nan=True,
                rtol=0.0,
                atol=0.0,
            )
        if not matches:
            raise ValueError(
                f"Calibrated and raw coordinates differ in {column}"
            )


def load_manifest_targets(fold: int) -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "results"
        / "data_audit"
        / "dataset_v4_multifold"
        / f"fold_{fold:02d}_manifest.csv"
    )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id",
        "source_file",
        "new_split",
        "target_present",
        "distance_m",
        "velocity_mps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fold manifest is missing columns: {missing}")
    targets = frame.loc[
        frame["new_split"].eq("test") & frame["target_present"].eq(1),
        ["sample_id", "source_file", "distance_m", "velocity_mps"],
    ].copy()
    if targets["sample_id"].duplicated().any():
        raise ValueError(f"Fold {fold} target manifest sample IDs are not unique")
    return targets.rename(columns={"source_file": "manifest_scan_group"})


def load_frozen_predictions(
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    target_frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    all_sample_ids: list[str] = []
    total_backgrounds = 0
    total_false_alarms = 0

    for row in selection.itertuples(index=False):
        fold = int(row.fold)
        experiment_name = str(row.experiment_name)
        threshold = float(row.base_threshold)
        prediction_path = (
            PROJECT_ROOT
            / "results"
            / "experiments"
            / experiment_name
            / "tables"
            / "base_threshold_test_predictions.csv"
        )
        raw_prediction_path = prediction_path.with_name(
            "raw_base_threshold_test_predictions.csv"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen prediction table: {display_path(prediction_path)}"
            )
        if not raw_prediction_path.is_file():
            raise FileNotFoundError(
                "Missing frozen raw prediction table: "
                f"{display_path(raw_prediction_path)}"
            )
        predictions = validate_predictions(
            pd.read_csv(prediction_path, encoding="utf-8-sig"),
            threshold,
        )
        raw_predictions = validate_predictions(
            pd.read_csv(raw_prediction_path, encoding="utf-8-sig"),
            threshold,
        )
        validate_coordinate_alignment(predictions, raw_predictions)
        predictions["fold"] = fold
        predictions["experiment_name"] = experiment_name
        predictions["frozen_threshold"] = threshold

        backgrounds = predictions["target_present"].eq(0)
        total_backgrounds += int(backgrounds.sum())
        total_false_alarms += int(predictions.loc[backgrounds, "false_alarm"].sum())

        targets = predictions.loc[predictions["target_present"].eq(1)].copy()
        manifest_targets = load_manifest_targets(fold)
        merged = targets.merge(
            manifest_targets,
            on="sample_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"Fold {fold} target predictions do not align to manifest")
        if not merged["scan_group"].astype(str).eq(
            merged["manifest_scan_group"].astype(str)
        ).all():
            raise ValueError(f"Fold {fold} scan_group does not match manifest")
        merged = merged.drop(columns=["_merge", "manifest_scan_group", "mat_path"])
        target_frames.append(merged)
        all_sample_ids.extend(merged["sample_id"].astype(str).tolist())
        sources.append(
            {
                "fold": fold,
                "experiment_name": experiment_name,
                "frozen_threshold": threshold,
                "path": display_path(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
                "raw_path": display_path(raw_prediction_path),
                "raw_size_bytes": raw_prediction_path.stat().st_size,
                "raw_sha256": sha256_file(raw_prediction_path),
                "coordinate_match_with_raw": True,
                "rows": len(predictions),
                "target_rows": len(merged),
                "background_rows": int(backgrounds.sum()),
            }
        )

    targets = pd.concat(target_frames, ignore_index=True)
    if len(all_sample_ids) != len(set(all_sample_ids)):
        raise ValueError("Target sample IDs overlap across folds")
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError(f"Expected {EXPECTED_TARGETS} targets, got {len(targets)}")
    if total_backgrounds != EXPECTED_BACKGROUNDS:
        raise ValueError(
            f"Expected {EXPECTED_BACKGROUNDS} backgrounds, got {total_backgrounds}"
        )
    if total_false_alarms != EXPECTED_FALSE_ALARMS:
        raise ValueError(
            f"Expected {EXPECTED_FALSE_ALARMS} false alarms, got {total_false_alarms}"
        )
    correct = int(targets["correct_detection"].sum())
    if correct != EXPECTED_CORRECT_DETECTIONS:
        raise ValueError(
            f"Expected {EXPECTED_CORRECT_DETECTIONS} correct detections, got {correct}"
        )
    return targets, sources


def build_pooled_summary(targets: pd.DataFrame) -> pd.DataFrame:
    target_count = len(targets)
    detected = int(targets["detected"].sum())
    localized = int(targets["localization_ok"].sum())
    correct = int(targets["correct_detection"].sum())
    detected_not_localized = int(
        (targets["detected"] & ~targets["localization_ok"]).sum()
    )
    localized_not_detected = int(
        (~targets["detected"] & targets["localization_ok"]).sum()
    )
    neither = int((~targets["detected"] & ~targets["localization_ok"]).sum())
    return pd.DataFrame(
        [
            {
                "model": "BC-DPG-FCN v3 complete-scan",
                "evaluation_scope": "six-fold internal frozen test",
                "target_samples": target_count,
                "score_detected": detected,
                "score_pd": safe_divide(detected, target_count),
                "localization_ok_regardless_score": localized,
                "localization_ok_rate": safe_divide(localized, target_count),
                "correct_detection": correct,
                "joint_pd": safe_divide(correct, target_count),
                "localization_ok_among_detected": safe_divide(correct, detected),
                "detected_but_not_localized": detected_not_localized,
                "localized_but_not_detected": localized_not_detected,
                "neither_detected_nor_localized": neither,
                "range_tolerance_gates": RANGE_TOLERANCE_GATES,
                "velocity_tolerance_bins": VELOCITY_TOLERANCE_BINS,
                "range_resolution_m_per_gate": RANGE_RESOLUTION_M,
                "velocity_resolution_mps_per_bin": VELOCITY_RESOLUTION_MPS,
            }
        ]
    )


def summarize_errors(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mae": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mae": float(array.mean()),
        f"{prefix}_median": float(np.quantile(array, 0.50)),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
        f"{prefix}_p95": float(np.quantile(array, 0.95)),
        f"{prefix}_p99": float(np.quantile(array, 0.99)),
        f"{prefix}_max": float(array.max()),
    }


def summarize_subset(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    record: dict[str, Any] = {"scope": scope, "samples": len(frame)}
    record.update(summarize_errors(frame["range_error_gates"], "range_gates"))
    record.update(summarize_errors(frame["velocity_error_bins"], "velocity_bins"))
    record["range_mae_grid_equivalent_m"] = (
        record["range_gates_mae"] * RANGE_RESOLUTION_M
    )
    record["velocity_mae_grid_equivalent_mps"] = (
        record["velocity_bins_mae"] * VELOCITY_RESOLUTION_MPS
    )
    return record


def build_error_distribution(targets: pd.DataFrame) -> pd.DataFrame:
    scopes = (
        ("all_targets", pd.Series(True, index=targets.index)),
        ("score_detected_targets", targets["detected"]),
        ("joint_success_targets", targets["correct_detection"]),
    )
    return pd.DataFrame(
        [summarize_subset(targets.loc[mask], scope) for scope, mask in scopes]
    )


def build_fold_summary(targets: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold, group in targets.groupby("fold", sort=True):
        detected = int(group["detected"].sum())
        localized = int(group["localization_ok"].sum())
        correct = int(group["correct_detection"].sum())
        record: dict[str, Any] = {
            "fold": int(fold),
            "frozen_threshold": float(group["frozen_threshold"].iloc[0]),
            "target_samples": len(group),
            "score_detected": detected,
            "score_pd": safe_divide(detected, len(group)),
            "localization_ok_regardless_score": localized,
            "localization_ok_rate": safe_divide(localized, len(group)),
            "correct_detection": correct,
            "joint_pd": safe_divide(correct, len(group)),
            "localization_ok_among_detected": safe_divide(correct, detected),
        }
        record.update(summarize_errors(group["range_error_gates"], "range_gates"))
        record.update(
            summarize_errors(group["velocity_error_bins"], "velocity_bins")
        )
        detected_group = group.loc[group["detected"]]
        record["detected_range_mae_gates"] = float(
            detected_group["range_error_gates"].mean()
        )
        record["detected_velocity_mae_bins"] = float(
            detected_group["velocity_error_bins"].mean()
        )
        records.append(record)
    return pd.DataFrame(records)


def distance_stratum(distance_m: pd.Series) -> pd.Series:
    conditions = [distance_m <= 2040, distance_m <= 2130]
    labels = ["1950-2040 m", "2070-2130 m"]
    return pd.Series(
        np.select(conditions, labels, default="2160-2400 m"),
        index=distance_m.index,
        dtype="object",
    )


def velocity_stratum(velocity_mps: pd.Series) -> pd.Series:
    conditions = [
        velocity_mps <= -4.0,
        velocity_mps < 0.0,
        velocity_mps < 4.0,
    ]
    labels = [
        "negative_fast_le_-4_mps",
        "negative_slow_-4_to_0_mps",
        "positive_slow_0_to_4_mps",
    ]
    return pd.Series(
        np.select(conditions, labels, default="positive_fast_ge_4_mps"),
        index=velocity_mps.index,
        dtype="object",
    )


def build_stratified_summary(
    targets: pd.DataFrame,
    stratum_name: str,
    strata: pd.Series,
) -> pd.DataFrame:
    working = targets.copy()
    working[stratum_name] = strata
    records: list[dict[str, Any]] = []
    for value, group in working.groupby(stratum_name, sort=False):
        detected = int(group["detected"].sum())
        localized = int(group["localization_ok"].sum())
        correct = int(group["correct_detection"].sum())
        record: dict[str, Any] = {
            stratum_name: value,
            "target_samples": len(group),
            "score_detected": detected,
            "score_pd": safe_divide(detected, len(group)),
            "localization_ok_regardless_score": localized,
            "localization_ok_rate": safe_divide(localized, len(group)),
            "correct_detection": correct,
            "joint_pd": safe_divide(correct, len(group)),
            "range_mae_gates": float(group["range_error_gates"].mean()),
            "range_median_gates": float(group["range_error_gates"].median()),
            "range_p90_gates": float(group["range_error_gates"].quantile(0.90)),
            "velocity_mae_bins": float(group["velocity_error_bins"].mean()),
            "velocity_median_bins": float(group["velocity_error_bins"].median()),
            "velocity_p90_bins": float(
                group["velocity_error_bins"].quantile(0.90)
            ),
        }
        records.append(record)
    result = pd.DataFrame(records)
    if int(result["target_samples"].sum()) != len(targets):
        raise ValueError(f"{stratum_name} strata do not cover all target samples")
    orders = {
        "distance_stratum": [
            "1950-2040 m",
            "2070-2130 m",
            "2160-2400 m",
        ],
        "velocity_stratum": [
            "negative_fast_le_-4_mps",
            "negative_slow_-4_to_0_mps",
            "positive_slow_0_to_4_mps",
            "positive_fast_ge_4_mps",
        ],
    }
    if stratum_name in orders:
        rank = {value: index for index, value in enumerate(orders[stratum_name])}
        result = result.sort_values(
            stratum_name,
            key=lambda values: values.map(rank),
        ).reset_index(drop=True)
    return result


def plot_error_cdf(targets: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    panels = (
        (
            axes[0],
            targets["range_error_gates"].to_numpy(dtype=np.float64),
            RANGE_TOLERANCE_GATES,
            "Range error (gates)",
        ),
        (
            axes[1],
            targets["velocity_error_bins"].to_numpy(dtype=np.float64),
            VELOCITY_TOLERANCE_BINS,
            "Velocity error (bins)",
        ),
    )
    for axis, values, tolerance, label in panels:
        ordered = np.sort(values)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.step(ordered, cumulative, where="post", color="#1f6f8b", linewidth=2)
        axis.axvline(
            tolerance,
            color="#b33a3a",
            linestyle="--",
            linewidth=1.5,
            label=f"Tolerance = {tolerance}",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("Empirical cumulative fraction")
        axis.set_ylim(0, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right", frameon=False)
    figure.suptitle("BC-DPG frozen six-fold target localization errors")
    figure.savefig(
        output_dir / "fig1_localization_error_cdf.png",
        dpi=180,
        metadata={"Software": "balloon_radar_project"},
    )
    figure.savefig(
        output_dir / "fig1_localization_error_cdf.pdf",
        metadata={"Creator": "balloon_radar_project", "CreationDate": None},
    )
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)]
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "|" + "|".join(["---"] * len(selected.columns)) + "|"
    rows = []
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_report(
    pooled: pd.DataFrame,
    errors: pd.DataFrame,
    folds: pd.DataFrame,
    distance: pd.DataFrame,
    velocity: pd.DataFrame,
) -> str:
    summary = pooled.iloc[0]
    all_errors = errors.loc[errors["scope"].eq("all_targets")].iloc[0]
    detected_errors = errors.loc[
        errors["scope"].eq("score_detected_targets")
    ].iloc[0]
    return "\n".join(
        [
            "# BC-DPG frozen localization evidence",
            "",
            "## Scope",
            "",
            "This report aggregates the frozen base-threshold test predictions from "
            "the six complete-scan BC-DPG folds. It performs no training, inference, "
            "threshold selection, or test-set retuning. BC-DPG changes only candidate "
            "scores, so raw DPG and calibrated BC-DPG share the same predicted "
            "range-velocity coordinates.",
            "",
            "## Main result",
            "",
            f"Among {int(summary['target_samples'])} target samples, "
            f"{int(summary['score_detected'])} pass the frozen score thresholds, "
            f"{int(summary['localization_ok_regardless_score'])} satisfy the "
            "localization tolerance regardless of score, and "
            f"{int(summary['correct_detection'])} satisfy both conditions. The "
            f"pooled score Pd is {summary['score_pd']:.4f}, localization-ok rate is "
            f"{summary['localization_ok_rate']:.4f}, and joint Pd is "
            f"{summary['joint_pd']:.4f}. Among score-detected targets, "
            f"{summary['localization_ok_among_detected']:.4f} meet the localization "
            "tolerance.",
            "",
            f"The decomposition is {int(summary['correct_detection'])} detected and "
            f"localized, {int(summary['detected_but_not_localized'])} detected but "
            f"outside tolerance, {int(summary['localized_but_not_detected'])} within "
            "localization tolerance but below threshold, and "
            f"{int(summary['neither_detected_nor_localized'])} satisfying neither.",
            "",
            "## Error distribution",
            "",
            f"Across all targets, range error has MAE "
            f"{all_errors['range_gates_mae']:.3f} gates, median "
            f"{all_errors['range_gates_median']:.3f}, P90 "
            f"{all_errors['range_gates_p90']:.3f}, and maximum "
            f"{all_errors['range_gates_max']:.0f}. Velocity error has MAE "
            f"{all_errors['velocity_bins_mae']:.3f} bins, median "
            f"{all_errors['velocity_bins_median']:.3f}, P90 "
            f"{all_errors['velocity_bins_p90']:.3f}, and maximum "
            f"{all_errors['velocity_bins_max']:.0f}. The gap between P90 and the "
            "maximum shows a small catastrophic-error tail that MAE alone would "
            "obscure.",
            "",
            f"Conditioned on passing the frozen score threshold, range MAE is "
            f"{detected_errors['range_gates_mae']:.3f} gates and velocity MAE is "
            f"{detected_errors['velocity_bins_mae']:.3f} bins. Conditional metrics "
            "must be reported together with the unconditional and joint rates.",
            "",
            "The grid-equivalent conversions use 30 m per range gate and 0.183 m/s "
            "per velocity bin. They describe discrete-grid offsets, not continuous "
            "physical measurement error relative to unquantized ground truth.",
            "",
            "![Localization error CDF](figures/fig1_localization_error_cdf.png)",
            "",
            "## Fold results",
            "",
            markdown_table(
                folds,
                [
                    "fold",
                    "target_samples",
                    "score_pd",
                    "localization_ok_rate",
                    "joint_pd",
                    "range_gates_mae",
                    "velocity_bins_mae",
                ],
            ),
            "",
            "## Descriptive strata",
            "",
            "Distance and velocity strata are fixed descriptive slices. They were not "
            "used to choose a model, threshold, or claim.",
            "",
            markdown_table(
                distance,
                [
                    "distance_stratum",
                    "target_samples",
                    "score_pd",
                    "localization_ok_rate",
                    "joint_pd",
                ],
            ),
            "",
            markdown_table(
                velocity,
                [
                    "velocity_stratum",
                    "target_samples",
                    "score_pd",
                    "localization_ok_rate",
                    "joint_pd",
                ],
            ),
            "",
            "## Claim boundaries",
            "",
            "- This is an internal six-fold frozen-test aggregation, not external "
            "blind validation.",
            "- Complete-scan BC-DPG remains an offline upper bound because its "
            "context may include later samples.",
            "- Class and acquisition date are confounded in the current data.",
            "- SNR stratification is unavailable because the frozen manifests do not "
            "contain an SNR field.",
            "- Physical localization accuracy requires verified radar calibration and "
            "unquantized ground truth; current evidence is grid-index accuracy.",
            "",
        ]
    )


def generated_file_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "localization_manifest.json",
            "SHA256SUMS.txt",
        }:
            continue
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in paths
    ]
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Use --overwrite to replace it."
        )


def write_evidence(
    staging_dir: Path,
    selection_path: Path,
    sources: list[dict[str, Any]],
    targets: pd.DataFrame,
) -> None:
    tables_dir = staging_dir / "tables"
    figures_dir = staging_dir / "figures"
    tables_dir.mkdir()
    figures_dir.mkdir()

    pooled = build_pooled_summary(targets)
    errors = build_error_distribution(targets)
    folds = build_fold_summary(targets)
    distance = build_stratified_summary(
        targets,
        "distance_stratum",
        distance_stratum(targets["distance_m"]),
    )
    velocity = build_stratified_summary(
        targets,
        "velocity_stratum",
        velocity_stratum(targets["velocity_mps"]),
    )
    source_table = pd.DataFrame(sources)

    outputs = {
        "table_01_pooled_localization.csv": pooled,
        "table_02_fold_localization.csv": folds,
        "table_03_error_distribution.csv": errors,
        "table_04_distance_strata.csv": distance,
        "table_05_velocity_strata.csv": velocity,
        "table_06_source_files.csv": source_table,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables_dir / name, index=False, encoding="utf-8")
    plot_error_cdf(targets, figures_dir)
    (staging_dir / "LOCALIZATION_EVIDENCE_REPORT.md").write_text(
        build_report(pooled, errors, folds, distance, velocity),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "evidence_role": "frozen six-fold aggregate localization analysis",
        "training_performed": False,
        "inference_performed": False,
        "test_threshold_retuning": False,
        "sample_level_predictions_included": False,
        "coordinates_match_raw_dpg": True,
        "model": "BC-DPG-FCN v3 complete-scan offline upper bound",
        "folds": list(EXPECTED_FOLDS),
        "target_samples": len(targets),
        "score_detected": int(targets["detected"].sum()),
        "localization_ok_regardless_score": int(targets["localization_ok"].sum()),
        "correct_detections": int(targets["correct_detection"].sum()),
        "range_tolerance_gates": RANGE_TOLERANCE_GATES,
        "velocity_tolerance_bins": VELOCITY_TOLERANCE_BINS,
        "range_resolution_m_per_gate": RANGE_RESOLUTION_M,
        "velocity_resolution_mps_per_bin": VELOCITY_RESOLUTION_MPS,
        "selection_table": {
            "path": display_path(selection_path),
            "size_bytes": selection_path.stat().st_size,
            "sha256": sha256_file(selection_path),
        },
        "implementation": {
            "path": display_path(Path(__file__)),
            "size_bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__)),
        },
        "sources": sources,
        "generated_files": generated_file_records(staging_dir),
    }
    (staging_dir / "localization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(staging_dir)


def build_output(selection_path: Path, output_dir: Path, overwrite: bool) -> None:
    ensure_output_available(output_dir, overwrite)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen selection table: {display_path(selection_path)}"
        )
    selection = load_frozen_selection(selection_path)
    targets, sources = load_frozen_predictions(selection)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    staging_dir = staging_parent / output_dir.name
    staging_dir.mkdir()
    try:
        write_evidence(staging_dir, selection_path, sources, targets)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> int:
    args = parse_args()
    selection_path = resolve_path(args.selection_table)
    output_dir = resolve_path(args.output_dir)
    try:
        build_output(selection_path, output_dir, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("BC-DPG localization evidence: PASS")
    print(f"output_dir={display_path(output_dir)}")
    print("target_samples=318")
    print("correct_detections=289")
    print("training_performed=False")
    print("inference_performed=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
