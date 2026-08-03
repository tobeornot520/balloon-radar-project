#!/usr/bin/env python3
"""Run preregistered batch-grouped interpretable baselines on LAT-MRICD-1.0."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lat_mricd_grouped_baseline_v1.json"
CATEGORY_NAMES = {1: "UAV", 2: "bird", 3: "weather"}
MODEL_NAMES = {
    1: "Mavic 2",
    2: "Phantom 4",
    3: "Air 3S",
    4: "M30T",
    5: "racing drone",
    6: "self-built UAV",
    7: "pigeon",
    8: "goose",
    9: "weather clutter",
    10: "unspecified UAV",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed interpretable classifiers with batch-grouped held-out folds "
            "on the public LAT-MRICD dataset."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "dataset_root",
        "output_dir",
        "random_state",
        "n_splits",
        "splitter",
        "split_manifest",
        "split_summary",
        "group_key",
        "tasks",
        "models",
        "acceptance_contract",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config missing fields: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    if config["group_key"] != ["representation", "band_code", "batch_code"]:
        raise ValueError("group_key must preserve representation/band/batch isolation")
    if config["splitter"] != "metadata_only_balanced_milp_v1":
        raise ValueError("splitter must be the frozen metadata-only MILP assignment")
    if int(config["n_splits"]) < 3:
        raise ValueError("n_splits must be at least three")
    if not config["tasks"] or not config["models"]:
        raise ValueError("config must define tasks and models")
    contract = config["acceptance_contract"]
    if contract.get("random_row_split_allowed") is not False:
        raise ValueError("random row splitting must remain forbidden")
    if contract.get("hyperparameter_search_allowed") is not False:
        raise ValueError("hyperparameter search must remain disabled")
    return config


def single_public_matrix(path: Path) -> np.ndarray:
    payload = loadmat(path)
    matrices = [
        np.asarray(value)
        for key, value in payload.items()
        if not key.startswith("__")
        and np.asarray(value).ndim == 2
        and np.issubdtype(np.asarray(value).dtype, np.number)
    ]
    if len(matrices) != 1:
        raise ValueError(f"{path}: expected one public numeric matrix")
    return matrices[0]


def reconstruct_narrow_iq(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != 1028:
        raise ValueError("narrow matrix must have shape (records, 1028)")
    values = matrix[:, 4:]
    return values[:, 0::2] + 1j * values[:, 1::2]


def _row_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    variance = np.mean(centered**2, axis=1)
    std = np.sqrt(np.maximum(variance, 0.0))
    skew = np.mean(centered**3, axis=1) / np.maximum(std**3, EPS)
    kurtosis = np.mean(centered**4, axis=1) / np.maximum(variance**2, EPS)
    return mean, std, skew, kurtosis


def _lag_correlation(values: np.ndarray, lag: int) -> np.ndarray:
    left = values[:, lag:]
    right = values[:, :-lag]
    numerator = np.sum(left * np.conjugate(right), axis=1)
    denominator = np.sqrt(
        np.sum(np.abs(left) ** 2, axis=1)
        * np.sum(np.abs(right) ** 2, axis=1)
    )
    return np.abs(numerator) / np.maximum(denominator, EPS)


def _energy_quantile_positions(power: np.ndarray, quantile: float) -> np.ndarray:
    cumulative = np.cumsum(power, axis=1)
    positions = np.argmax(cumulative >= quantile, axis=1)
    return positions.astype(np.float64) / max(power.shape[1] - 1, 1)


def extract_hrrp_features(amplitude: np.ndarray) -> pd.DataFrame:
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.shape[1] < 4:
        raise ValueError("HRRP amplitude must be a two-dimensional sequence")
    if np.any(amplitude < 0) or not np.isfinite(amplitude).all():
        raise ValueError("HRRP amplitude must be finite and non-negative")

    rms = np.sqrt(np.mean(amplitude**2, axis=1))
    normalized = amplitude / np.maximum(rms[:, None], EPS)
    power = amplitude**2
    power /= np.maximum(power.sum(axis=1, keepdims=True), EPS)
    coordinate = np.linspace(0.0, 1.0, amplitude.shape[1], dtype=np.float64)
    centroid = np.sum(power * coordinate[None, :], axis=1)
    spread = np.sqrt(
        np.sum(power * (coordinate[None, :] - centroid[:, None]) ** 2, axis=1)
    )
    entropy = -np.sum(power * np.log(np.maximum(power, EPS)), axis=1)
    entropy /= math.log(amplitude.shape[1])
    mean, std, skew, kurtosis = _row_moments(normalized)
    peak_index = np.argmax(power, axis=1)
    q10 = _energy_quantile_positions(power, 0.10)
    q50 = _energy_quantile_positions(power, 0.50)
    q90 = _energy_quantile_positions(power, 0.90)

    return pd.DataFrame(
        {
            "energy_centroid_bin_fraction": centroid,
            "energy_spread_bin_fraction": spread,
            "energy_entropy_normalized": entropy,
            "peak_bin_fraction": peak_index / max(amplitude.shape[1] - 1, 1),
            "peak_power_fraction": np.max(power, axis=1),
            "crest_factor": np.max(normalized, axis=1),
            "amplitude_mean_rms_normalized": mean,
            "amplitude_std_rms_normalized": std,
            "amplitude_skewness": skew,
            "amplitude_kurtosis": kurtosis,
            "energy_q10_bin_fraction": q10,
            "energy_q50_bin_fraction": q50,
            "energy_q90_bin_fraction": q90,
            "energy_q90_q10_width_fraction": q90 - q10,
            "roughness_mean_abs_difference": np.mean(
                np.abs(np.diff(normalized, axis=1)), axis=1
            ),
            "autocorrelation_magnitude_lag1": _lag_correlation(normalized, 1),
            "autocorrelation_magnitude_lag4": _lag_correlation(normalized, 4),
            "autocorrelation_magnitude_lag16": _lag_correlation(normalized, 16),
        }
    )


def extract_narrow_features(iq: np.ndarray) -> pd.DataFrame:
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[1] < 64:
        raise ValueError("narrow IQ must be a two-dimensional slow-time sequence")
    if not np.isfinite(iq).all():
        raise ValueError("narrow IQ must be finite")

    rms = np.sqrt(np.mean(np.abs(iq) ** 2, axis=1))
    normalized = iq / np.maximum(rms[:, None], EPS)
    envelope = np.abs(normalized)
    env_mean, env_std, env_skew, env_kurtosis = _row_moments(envelope)

    increments = normalized[:, 1:] * np.conjugate(normalized[:, :-1])
    increment_unit = increments / np.maximum(np.abs(increments), EPS)
    mean_increment = np.mean(increment_unit, axis=1)

    window = np.hanning(iq.shape[1])[None, :]
    spectrum = np.fft.fftshift(np.fft.fft(normalized * window, axis=1), axes=1)
    spectral_power = np.abs(spectrum) ** 2
    spectral_power /= np.maximum(spectral_power.sum(axis=1, keepdims=True), EPS)
    frequency = np.fft.fftshift(np.fft.fftfreq(iq.shape[1], d=1.0))
    centroid = np.sum(spectral_power * frequency[None, :], axis=1)
    spread = np.sqrt(
        np.sum(
            spectral_power * (frequency[None, :] - centroid[:, None]) ** 2,
            axis=1,
        )
    )
    entropy = -np.sum(
        spectral_power * np.log(np.maximum(spectral_power, EPS)), axis=1
    ) / math.log(iq.shape[1])
    dominant_index = np.argmax(spectral_power, axis=1)
    dominant_frequency = frequency[dominant_index]
    positive = spectral_power[:, frequency > 0].sum(axis=1)
    negative = spectral_power[:, frequency < 0].sum(axis=1)
    zero_half_width = 2.0 / iq.shape[1]
    zero_fraction = spectral_power[:, np.abs(frequency) <= zero_half_width].sum(axis=1)
    spectral_flatness = np.exp(
        np.mean(np.log(np.maximum(spectral_power, EPS)), axis=1)
    ) / np.maximum(np.mean(spectral_power, axis=1), EPS)
    spectral_kurtosis = np.sum(
        spectral_power * (frequency[None, :] - centroid[:, None]) ** 4,
        axis=1,
    ) / np.maximum(spread**4, EPS)

    nfft = 2 * iq.shape[1]
    autocorrelation = np.fft.ifft(
        np.abs(np.fft.fft(normalized, n=nfft, axis=1)) ** 2,
        axis=1,
    )[:, :65]
    autocorrelation = np.abs(autocorrelation)
    autocorrelation /= np.maximum(autocorrelation[:, :1], EPS)
    periodic_region = autocorrelation[:, 2:65]
    periodic_index = np.argmax(periodic_region, axis=1)

    return pd.DataFrame(
        {
            "envelope_mean_rms_normalized": env_mean,
            "envelope_std_rms_normalized": env_std,
            "envelope_coefficient_of_variation": env_std / np.maximum(env_mean, EPS),
            "envelope_skewness": env_skew,
            "envelope_kurtosis": env_kurtosis,
            "envelope_crest_factor": np.max(envelope, axis=1),
            "phase_increment_mean_cycles_per_sample": np.angle(mean_increment)
            / (2.0 * np.pi),
            "phase_increment_resultant_length": np.abs(mean_increment),
            "autocorrelation_magnitude_lag1": _lag_correlation(normalized, 1),
            "autocorrelation_magnitude_lag2": _lag_correlation(normalized, 2),
            "autocorrelation_magnitude_lag4": _lag_correlation(normalized, 4),
            "autocorrelation_magnitude_lag8": _lag_correlation(normalized, 8),
            "autocorrelation_magnitude_lag16": _lag_correlation(normalized, 16),
            "autocorrelation_magnitude_lag32": _lag_correlation(normalized, 32),
            "periodicity_candidate_autocorrelation": np.max(periodic_region, axis=1),
            "periodicity_candidate_lag_fraction": (periodic_index + 2)
            / iq.shape[1],
            "spectrum_centroid_cycles_per_sample": centroid,
            "spectrum_abs_centroid_cycles_per_sample": np.sum(
                spectral_power * np.abs(frequency)[None, :], axis=1
            ),
            "spectrum_spread_cycles_per_sample": spread,
            "spectrum_entropy_normalized": entropy,
            "spectrum_dominant_frequency_cycles_per_sample": dominant_frequency,
            "spectrum_dominant_abs_frequency_cycles_per_sample": np.abs(
                dominant_frequency
            ),
            "spectrum_peak_power_fraction": np.max(spectral_power, axis=1),
            "spectrum_zero_band_power_fraction": zero_fraction,
            "spectrum_positive_negative_log10_ratio": np.log10(
                (positive + EPS) / (negative + EPS)
            ),
            "spectrum_flatness": spectral_flatness,
            "spectrum_kurtosis": spectral_kurtosis,
        }
    )


def build_grouped_fold_assignments(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Balance batch-class cells using metadata only, then freeze whole batches."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if labels.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("labels and groups must be aligned one-dimensional arrays")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    _ = random_state  # The metadata-only optimizer is deterministic after canonicalization.
    class_codes = np.sort(np.unique(labels))
    unique_groups = np.sort(np.unique(groups))
    if len(unique_groups) < n_splits:
        raise ValueError("fewer batch groups than held-out folds")

    group_class_counts = np.zeros(
        (len(unique_groups), len(class_codes)), dtype=np.float64
    )
    for group_index, group in enumerate(unique_groups):
        selected = groups == group
        for class_index, code in enumerate(class_codes):
            group_class_counts[group_index, class_index] = np.sum(
                selected & (labels == code)
            )
    group_class_presence = group_class_counts > 0
    class_batch_counts = group_class_presence.sum(axis=0)
    if np.any(class_batch_counts < n_splits):
        raise ValueError("every class must occur in at least one batch per fold")

    group_count = len(unique_groups)
    class_count = len(class_codes)
    assignment_variable_count = group_count * n_splits
    class_deviation_count = n_splits * class_count
    class_positive_offset = assignment_variable_count
    class_negative_offset = class_positive_offset + class_deviation_count
    total_positive_offset = class_negative_offset + class_deviation_count
    total_negative_offset = total_positive_offset + n_splits
    variable_count = total_negative_offset + n_splits

    objective = np.zeros(variable_count, dtype=np.float64)
    class_row_targets = group_class_counts.sum(axis=0) / n_splits
    for fold_index in range(n_splits):
        for class_index in range(class_count):
            deviation_index = fold_index * class_count + class_index
            weight = 1.0 / max(class_row_targets[class_index], 1.0)
            objective[class_positive_offset + deviation_index] = weight
            objective[class_negative_offset + deviation_index] = weight
        total_weight = 0.25 / max(len(labels) / n_splits, 1.0)
        objective[total_positive_offset + fold_index] = total_weight
        objective[total_negative_offset + fold_index] = total_weight

    constraint_count = (
        group_count
        + n_splits * class_count
        + n_splits
        + n_splits * class_count
        + n_splits
    )
    coefficients = lil_matrix((constraint_count, variable_count), dtype=np.float64)
    lower = np.empty(constraint_count, dtype=np.float64)
    upper = np.empty(constraint_count, dtype=np.float64)
    row = 0

    for group_index in range(group_count):
        for fold_index in range(n_splits):
            coefficients[row, group_index * n_splits + fold_index] = 1.0
        lower[row] = upper[row] = 1.0
        row += 1

    for fold_index in range(n_splits):
        for class_index in range(class_count):
            for group_index in range(group_count):
                if group_class_presence[group_index, class_index]:
                    coefficients[row, group_index * n_splits + fold_index] = 1.0
            target = class_batch_counts[class_index] / n_splits
            lower[row] = math.floor(target)
            upper[row] = math.ceil(target)
            row += 1

    for fold_index in range(n_splits):
        for group_index in range(group_count):
            coefficients[row, group_index * n_splits + fold_index] = 1.0
        target = group_count / n_splits
        lower[row] = math.floor(target)
        upper[row] = math.ceil(target)
        row += 1

    for fold_index in range(n_splits):
        for class_index in range(class_count):
            for group_index in range(group_count):
                count = group_class_counts[group_index, class_index]
                if count:
                    coefficients[row, group_index * n_splits + fold_index] = count
            deviation_index = fold_index * class_count + class_index
            coefficients[row, class_positive_offset + deviation_index] = -1.0
            coefficients[row, class_negative_offset + deviation_index] = 1.0
            lower[row] = upper[row] = class_row_targets[class_index]
            row += 1

    group_sizes = group_class_counts.sum(axis=1)
    total_row_target = len(labels) / n_splits
    for fold_index in range(n_splits):
        for group_index, count in enumerate(group_sizes):
            coefficients[row, group_index * n_splits + fold_index] = count
        coefficients[row, total_positive_offset + fold_index] = -1.0
        coefficients[row, total_negative_offset + fold_index] = 1.0
        lower[row] = upper[row] = total_row_target
        row += 1
    if row != constraint_count:
        raise AssertionError("MILP constraint matrix size mismatch")

    variable_upper = np.full(variable_count, np.inf, dtype=np.float64)
    variable_upper[:assignment_variable_count] = 1.0
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:assignment_variable_count] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), variable_upper),
        constraints=LinearConstraint(coefficients.tocsr(), lower, upper),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"metadata-only split optimization failed: {result.message}")
    assignment_matrix = result.x[:assignment_variable_count].reshape(
        group_count, n_splits
    )
    group_folds_zero_based = np.argmax(assignment_matrix, axis=1)
    if not np.all(
        assignment_matrix[np.arange(group_count), group_folds_zero_based] > 0.5
    ):
        raise RuntimeError("MILP returned a non-integral batch assignment")

    # Canonical fold numbering removes solver symmetry from the persisted manifest.
    signatures = [
        tuple(unique_groups[group_folds_zero_based == fold_index].tolist())
        for fold_index in range(n_splits)
    ]
    fold_order = sorted(range(n_splits), key=lambda index: signatures[index])
    canonical = {old: new + 1 for new, old in enumerate(fold_order)}
    group_to_fold = {
        int(group): canonical[int(fold)]
        for group, fold in zip(unique_groups, group_folds_zero_based, strict=True)
    }
    assignments = np.asarray([group_to_fold[int(group)] for group in groups])

    group_fold_counts = pd.DataFrame(
        {"group": groups, "fold": assignments}
    ).groupby("group", observed=True)["fold"].nunique()
    if not group_fold_counts.eq(1).all():
        raise AssertionError("a batch group was split across held-out folds")
    all_classes = set(class_codes.tolist())
    for fold in range(1, n_splits + 1):
        if set(labels[assignments == fold].tolist()) != all_classes:
            raise AssertionError(f"fold {fold} does not contain all classes")
    return assignments


def build_batch_split_manifest(
    *,
    task: dict[str, Any],
    metadata: np.ndarray,
    assignments: np.ndarray,
) -> pd.DataFrame:
    metadata = np.asarray(metadata, dtype=np.int64)
    if metadata.ndim != 2 or metadata.shape[1] != 4:
        raise ValueError("metadata must have four columns")
    labels = metadata[:, 1]
    models = metadata[:, 2]
    groups = metadata[:, 3]
    rows: list[dict[str, Any]] = []
    for group in np.sort(np.unique(groups)):
        selected = groups == group
        folds = np.unique(assignments[selected])
        if len(folds) != 1:
            raise AssertionError("batch group maps to multiple held-out folds")
        row: dict[str, Any] = {
            "task_id": task["task_id"],
            "representation": task["representation"],
            "band_code": int(task["band_code"]),
            "band": task["band"],
            "batch_code": int(group),
            "heldout_fold": int(folds[0]),
            "record_count": int(np.sum(selected)),
            "category_count": int(np.unique(labels[selected]).size),
            "model_count": int(np.unique(models[selected]).size),
        }
        for code in [int(value) for value in task["class_codes"]]:
            category = CATEGORY_NAMES[code].lower()
            count = int(np.sum(selected & (labels == code)))
            row[f"{category}_record_count"] = count
            row[f"{category}_present"] = count > 0
        rows.append(row)
    return pd.DataFrame(rows)


def load_frozen_split(
    *,
    config: dict[str, Any],
    config_path: Path,
    dataset_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = resolve_path(config["split_manifest"])
    summary_path = resolve_path(config["split_summary"])
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "frozen split is missing; run freeze_lat_mricd_grouped_split_v1.py first"
        )
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "task_id",
        "representation",
        "band_code",
        "batch_code",
        "heldout_fold",
        "record_count",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"frozen split manifest missing columns: {sorted(missing)}")
    if manifest.duplicated(["task_id", "batch_code"]).any():
        raise ValueError("frozen split has duplicate task/batch rows")
    expected_summary = {
        "status": "FROZEN_METADATA_ONLY_BATCH_SPLIT",
        "signal_columns_used_for_assignment": False,
        "group_key": config["group_key"],
        "n_splits": int(config["n_splits"]),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"unexpected frozen split {field}: {summary.get(field)!r}")
    if summary.get("config_sha256") != sha256_file(config_path):
        raise ValueError("frozen split config hash is stale")
    if summary.get("split_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("frozen split manifest hash is stale")
    source_lookup = {item["task_id"]: item for item in summary.get("source_files", [])}
    for task in config["tasks"]:
        path = dataset_root / task["relative_path"]
        source = source_lookup.get(task["task_id"])
        if source is None or source.get("sha256") != sha256_file(path):
            raise ValueError(f"frozen split source hash is stale for {task['task_id']}")
    return manifest, summary


def make_model(spec: dict[str, Any], *, random_state: int) -> Any:
    model_type = spec["type"]
    if model_type == "dummy":
        return DummyClassifier(strategy=spec["strategy"])
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=float(spec["C"]),
            class_weight=spec["class_weight"],
            max_iter=int(spec["max_iter"]),
            solver=spec["solver"],
            random_state=random_state,
        )
        return Pipeline(
            [("scaler", StandardScaler()), ("classifier", classifier)]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=spec["max_features"],
            class_weight=spec["class_weight"],
            n_jobs=int(spec["n_jobs"]),
            random_state=random_state,
        )
    raise ValueError(f"unsupported model type: {model_type}")


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_codes: list[int],
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=class_codes, average="macro")
        ),
        "multiclass_log_loss": float(
            log_loss(labels, probabilities, labels=class_codes)
        ),
    }
    recalls = recall_score(
        labels,
        predictions,
        labels=class_codes,
        average=None,
        zero_division=0,
    )
    for code, recall in zip(class_codes, recalls, strict=True):
        metrics[f"recall_{CATEGORY_NAMES[code].lower()}"] = float(recall)
    try:
        metrics["macro_ovr_roc_auc"] = float(
            roc_auc_score(
                labels,
                probabilities,
                labels=class_codes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        metrics["macro_ovr_roc_auc"] = math.nan
    return metrics


def batch_class_sample_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Equalize classes and, within each class, every batch-class cell."""
    frame = pd.DataFrame(
        {
            "category_code": np.asarray(labels, dtype=np.int64),
            "batch_code": np.asarray(groups, dtype=np.int64),
        }
    )
    if len(frame) == 0:
        raise ValueError("cannot weight an empty training split")
    cell_sizes = frame.groupby(
        ["category_code", "batch_code"], observed=True
    )["category_code"].transform("size")
    class_batch_counts = frame.groupby("category_code", observed=True)[
        "batch_code"
    ].transform("nunique")
    weights = 1.0 / (
        cell_sizes.to_numpy(dtype=np.float64)
        * class_batch_counts.to_numpy(dtype=np.float64)
    )
    weights /= weights.mean()
    return weights


def fit_with_sample_weights(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> Any:
    if isinstance(model, Pipeline):
        model.fit(
            features,
            labels,
            scaler__sample_weight=sample_weights,
            classifier__sample_weight=sample_weights,
        )
    else:
        model.fit(features, labels, sample_weight=sample_weights)
    return model


def batch_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correct"] = frame["category_code"].eq(frame["predicted_category_code"])
    rows: list[dict[str, Any]] = []
    for (task_id, model_id, fold, batch_code), group in frame.groupby(
        ["task_id", "model_id", "heldout_fold", "batch_code"], observed=True
    ):
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "heldout_fold": int(fold),
                "batch_code": int(batch_code),
                "category_code": "all",
                "category": "all",
                "record_count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "evaluation_unit": "batch",
            }
        )
    for (task_id, model_id, fold, batch_code, category_code), group in frame.groupby(
        [
            "task_id",
            "model_id",
            "heldout_fold",
            "batch_code",
            "category_code",
        ],
        observed=True,
    ):
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "heldout_fold": int(fold),
                "batch_code": int(batch_code),
                "category_code": int(category_code),
                "category": CATEGORY_NAMES[int(category_code)],
                "record_count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "evaluation_unit": "batch_class_cell",
            }
        )
    return pd.DataFrame(rows)


def batch_class_macro_metrics(cell_metrics: pd.DataFrame) -> dict[str, float]:
    if cell_metrics.empty:
        raise ValueError("batch-class metric table is empty")
    class_means = cell_metrics.groupby("category_code", observed=True)[
        "accuracy"
    ].mean()
    result = {
        "batch_class_macro_accuracy": float(class_means.mean()),
        "worst_batch_class_cell_accuracy": float(cell_metrics["accuracy"].min()),
        "batch_class_cell_accuracy_p10": float(cell_metrics["accuracy"].quantile(0.10)),
    }
    for code, value in class_means.items():
        result[f"batch_class_recall_{CATEGORY_NAMES[int(code)].lower()}"] = float(value)
    worst = cell_metrics.sort_values(
        ["accuracy", "record_count", "batch_code"], ascending=[True, True, True]
    ).iloc[0]
    result["worst_batch_class_cell_record_count"] = int(worst["record_count"])
    return result


def batch_class_distribution_table(batch_metrics: pd.DataFrame) -> pd.DataFrame:
    cells = batch_metrics.loc[
        batch_metrics["evaluation_unit"].eq("batch_class_cell")
    ]
    rows: list[dict[str, Any]] = []
    for (task_id, model_id, category_code, category), group in cells.groupby(
        ["task_id", "model_id", "category_code", "category"], observed=True
    ):
        worst = group.sort_values(
            ["accuracy", "record_count", "batch_code"],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "category_code": int(category_code),
                "category": category,
                "batch_class_cell_count": int(len(group)),
                "mean_accuracy": float(group["accuracy"].mean()),
                "minimum_accuracy": float(group["accuracy"].min()),
                "p10_accuracy": float(group["accuracy"].quantile(0.10)),
                "q25_accuracy": float(group["accuracy"].quantile(0.25)),
                "median_accuracy": float(group["accuracy"].median()),
                "q75_accuracy": float(group["accuracy"].quantile(0.75)),
                "worst_batch_code": int(worst["batch_code"]),
                "worst_cell_record_count": int(worst["record_count"]),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_intervals(
    batch_metrics: pd.DataFrame,
    *,
    replicates: int,
    random_state: int,
) -> pd.DataFrame:
    cells = batch_metrics.loc[
        batch_metrics["evaluation_unit"].eq("batch_class_cell")
    ].copy()
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    by_task_model = {
        (str(task), str(model)): group
        for (task, model), group in cells.groupby(
            ["task_id", "model_id"], observed=True
        )
    }

    for (task_id, model_id), group in by_task_model.items():
        batches = np.sort(group["batch_code"].unique())
        values: list[float] = []
        by_batch = {
            int(batch): frame
            for batch, frame in group.groupby("batch_code", observed=True)
        }
        for _ in range(replicates):
            selected = rng.choice(batches, size=len(batches), replace=True)
            sampled = pd.concat([by_batch[int(batch)] for batch in selected])
            if sampled["category_code"].nunique() < 3:
                continue
            values.append(batch_class_macro_metrics(sampled)["batch_class_macro_accuracy"])
        estimate = batch_class_macro_metrics(group)["batch_class_macro_accuracy"]
        rows.append(
            {
                "task_id": task_id,
                "comparison": model_id,
                "metric": "batch_class_macro_accuracy",
                "estimate": estimate,
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "requested_replicates": replicates,
                "valid_replicates": len(values),
                "resampling_unit": "batch_code",
            }
        )

    for task_id in sorted(cells["task_id"].unique()):
        logistic_key = (str(task_id), "logistic_batch_balanced")
        forest_key = (str(task_id), "random_forest_batch_balanced")
        if logistic_key not in by_task_model or forest_key not in by_task_model:
            continue
        logistic = by_task_model[logistic_key]
        forest = by_task_model[forest_key]
        batches = np.sort(logistic["batch_code"].unique())
        logistic_by_batch = {
            int(batch): frame
            for batch, frame in logistic.groupby("batch_code", observed=True)
        }
        forest_by_batch = {
            int(batch): frame
            for batch, frame in forest.groupby("batch_code", observed=True)
        }
        differences: list[float] = []
        for _ in range(replicates):
            selected = rng.choice(batches, size=len(batches), replace=True)
            logistic_sample = pd.concat(
                [logistic_by_batch[int(batch)] for batch in selected]
            )
            forest_sample = pd.concat(
                [forest_by_batch[int(batch)] for batch in selected]
            )
            if logistic_sample["category_code"].nunique() < 3:
                continue
            differences.append(
                batch_class_macro_metrics(forest_sample)[
                    "batch_class_macro_accuracy"
                ]
                - batch_class_macro_metrics(logistic_sample)[
                    "batch_class_macro_accuracy"
                ]
            )
        estimate = (
            batch_class_macro_metrics(forest)["batch_class_macro_accuracy"]
            - batch_class_macro_metrics(logistic)["batch_class_macro_accuracy"]
        )
        rows.append(
            {
                "task_id": task_id,
                "comparison": "random_forest_minus_logistic",
                "metric": "paired_batch_class_macro_accuracy_difference",
                "estimate": estimate,
                "ci_lower_95": float(np.quantile(differences, 0.025)),
                "ci_upper_95": float(np.quantile(differences, 0.975)),
                "requested_replicates": replicates,
                "valid_replicates": len(differences),
                "resampling_unit": "paired_batch_code",
            }
        )
    return pd.DataFrame(rows)


def subtype_pressure_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correct"] = frame["category_code"].eq(
        frame["predicted_category_code"]
    )
    rows: list[dict[str, Any]] = []
    for (task_id, model_id, model_code, model_name), group in frame.groupby(
        ["task_id", "model_id", "model_code", "model"], observed=True
    ):
        batch_accuracy = group.groupby("batch_code", observed=True)["correct"].mean()
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "target_model_code": int(model_code),
                "target_model": model_name,
                "category_code": int(group["category_code"].iloc[0]),
                "category": group["category"].iloc[0],
                "record_count": int(len(group)),
                "batch_count": int(group["batch_code"].nunique()),
                "row_accuracy": float(group["correct"].mean()),
                "batch_macro_accuracy": float(batch_accuracy.mean()),
                "worst_batch_accuracy": float(batch_accuracy.min()),
                "unseen_model_generalization_evidence": False,
            }
        )
    return pd.DataFrame(rows)


def model_importance_rows(
    model: Any,
    *,
    feature_names: list[str],
    task_id: str,
    model_id: str,
    fold: int,
) -> list[dict[str, Any]]:
    if isinstance(model, Pipeline):
        classifier = model.named_steps["classifier"]
        values = np.mean(np.abs(classifier.coef_), axis=0)
        importance_type = "mean_abs_standardized_multiclass_coefficient"
    elif isinstance(model, RandomForestClassifier):
        values = model.feature_importances_
        importance_type = "mean_decrease_impurity"
    else:
        return []
    return [
        {
            "task_id": task_id,
            "model_id": model_id,
            "heldout_fold": fold,
            "feature": feature,
            "importance": float(value),
            "importance_type": importance_type,
        }
        for feature, value in zip(feature_names, values, strict=True)
    ]


def _format_number(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else f"{float(value):.4f}"
    return str(value)


def make_report(summary: dict[str, Any], model_summary: pd.DataFrame) -> str:
    rows = []
    for record in model_summary.to_dict(orient="records"):
        rows.append(
            "| {task_id} | {model_id} | {fold_macro_balanced_accuracy} | "
            "{worst_fold_balanced_accuracy} | {batch_macro_accuracy} | "
            "{batch_accuracy_p10} | {worst_batch_accuracy} | "
            "{batch_class_macro_accuracy} |".format(
                **{key: _format_number(value) for key, value in record.items()}
            )
        )
    table = "\n".join(rows)
    return f"""# LAT-MRICD Grouped Interpretable Baseline V1

Status: `{summary['status']}`  
Implementation commit: `{summary['implementation_commit']}`  
Grouping: `(representation, band_code, batch_code)`  
Held-out folds: `{summary['n_splits']}`

## Primary grouped results

| Task | Model | Fold-macro balanced accuracy | Worst-fold balanced accuracy | Batch-macro accuracy | Batch accuracy P10 | Worst-batch accuracy | Batch-class macro accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

## Experimental contract

- The metadata-only batch manifest was frozen before any signal feature was extracted.
- Every physical row appears in exactly one held-out fold.
- No batch code appears in both training and held-out data within a fold.
- All held-out folds contain UAV, bird and weather records.
- Hyperparameters are fixed in `configs/lat_mricd_grouped_baseline_v1.json`; no search or
  held-out-driven model selection is performed.
- The dummy, balanced logistic and balanced random-forest results are all retained. A larger
  number does not authorize choosing a model on an external locked test.
- Training weights give each class equal total weight and each batch-class cell equal weight
  within its class. The primary batch-class metric applies the same hierarchy at evaluation.
- Metrics include sample, fold, batch and batch-class views because record counts are highly
  imbalanced across batches and classes.

## Feature scope

The HRRP branch uses per-record normalized amplitude geometry, entropy, quantile width,
roughness and autocorrelation. The Narrow branch uses scale/global-phase-invariant envelope,
phase-increment, autocorrelation and normalized Doppler-spectrum summaries. Frequencies are
reported only in cycles/sample.

## Claim boundary

This is an internal grouped public-data baseline, not an external blind test. It evaluates new
batch codes for already represented submodels; it is not unseen-model generalization. Batch
semantics are not independently verified, so batch isolation is a conservative proxy for
acquisition grouping.
The dataset contains no H/V pair, no balloon label and no verified PRF or continuous timestamp.
These results cannot establish physical micro-Doppler in Hz, polarimetric performance, causal
deployment, Tian reproduction or balloon-payload recognition.
"""


def run_baseline(
    *,
    config_path: Path,
    dataset_root: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    config = load_config(config_path)
    dataset_root = resolve_path(dataset_root or config["dataset_root"])
    output_dir = resolve_path(output_dir or config["output_dir"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if output_dir in {PROJECT_ROOT, dataset_root}:
        raise ValueError("output directory must be separate from project and raw data")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(config["random_state"])
    n_splits = int(config["n_splits"])
    frozen_split, frozen_split_summary = load_frozen_split(
        config=config,
        config_path=config_path,
        dataset_root=dataset_root,
    )
    coverage_rows: list[dict[str, Any]] = []
    feature_schema_rows: list[dict[str, Any]] = []
    feature_summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    for task in config["tasks"]:
        task_id = str(task["task_id"])
        representation = str(task["representation"])
        path = dataset_root / task["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"task matrix not found: {path}")
        matrix = single_public_matrix(path)
        expected_columns = 1028 if representation == "Narrow" else 504
        if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
            raise ValueError(
                f"{task_id}: expected (*, {expected_columns}), got {matrix.shape}"
            )
        metadata = np.rint(matrix[:, :4]).astype(np.int64)
        if not np.allclose(matrix[:, :4], metadata):
            raise ValueError(f"{task_id}: metadata columns must be integer codes")
        band_codes = set(metadata[:, 0].tolist())
        if band_codes != {int(task["band_code"])}:
            raise ValueError(f"{task_id}: unexpected band codes {band_codes}")
        labels = metadata[:, 1]
        models = metadata[:, 2]
        groups = metadata[:, 3]
        class_codes = [int(code) for code in task["class_codes"]]
        if set(np.unique(labels).tolist()) != set(class_codes):
            raise ValueError(f"{task_id}: class coverage does not match config")

        if representation == "Narrow":
            features = extract_narrow_features(reconstruct_narrow_iq(matrix))
        elif representation == "HRRP":
            features = extract_hrrp_features(matrix[:, 4:])
        else:
            raise ValueError(f"unsupported representation: {representation}")
        if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{task_id}: extracted features contain NaN or Inf")

        task_split = frozen_split.loc[frozen_split["task_id"].eq(task_id)].copy()
        expected_groups = set(np.unique(groups).tolist())
        frozen_groups = set(task_split["batch_code"].astype(int).tolist())
        if frozen_groups != expected_groups:
            raise ValueError(f"{task_id}: frozen batch set does not match source data")
        group_to_fold = task_split.set_index("batch_code")["heldout_fold"].to_dict()
        assignments = np.asarray(
            [int(group_to_fold[int(group)]) for group in groups], dtype=np.int64
        )
        for fold in range(1, n_splits + 1):
            train_groups = set(groups[assignments != fold].tolist())
            heldout_groups = set(groups[assignments == fold].tolist())
            if train_groups & heldout_groups:
                raise AssertionError("frozen split leaks a batch across partitions")
            if set(labels[assignments == fold].tolist()) != set(class_codes):
                raise ValueError(f"{task_id}: fold {fold} does not cover all classes")
        split_frame = pd.DataFrame(
            {
                "task_id": task_id,
                "source_row_index": np.arange(len(labels)),
                "representation": representation,
                "band_code": int(task["band_code"]),
                "band": task["band"],
                "category_code": labels,
                "category": [CATEGORY_NAMES[int(code)] for code in labels],
                "model_code": models,
                "model": [MODEL_NAMES[int(code)] for code in models],
                "batch_code": groups,
                "heldout_fold": assignments,
            }
        )
        for fold in range(1, n_splits + 1):
            selected = split_frame.loc[split_frame["heldout_fold"].eq(fold)]
            row: dict[str, Any] = {
                "task_id": task_id,
                "heldout_fold": fold,
                "record_count": int(len(selected)),
                "batch_count": int(selected["batch_code"].nunique()),
            }
            for code in class_codes:
                category = CATEGORY_NAMES[code].lower()
                subset = selected.loc[selected["category_code"].eq(code)]
                row[f"{category}_record_count"] = int(len(subset))
                row[f"{category}_batch_count"] = int(subset["batch_code"].nunique())
            coverage_rows.append(row)

        for name in features.columns:
            feature_schema_rows.append(
                {
                    "task_id": task_id,
                    "representation": representation,
                    "feature": name,
                    "physical_frequency_unit": "cycles/sample"
                    if "cycles_per_sample" in name
                    else "not_applicable",
                    "per_record_normalized": True,
                }
            )
        summary_frame = pd.concat(
            [split_frame[["category_code", "category"]], features], axis=1
        )
        feature_summary = (
            summary_frame.groupby(["category_code", "category"], observed=True)[
                list(features.columns)
            ]
            .agg(["mean", "std", "median"])
            .stack(level=0, future_stack=True)
            .reset_index()
            .rename(columns={"level_2": "feature"})
        )
        feature_summary.insert(0, "task_id", task_id)
        feature_summary_frames.append(feature_summary)

        feature_values = features.to_numpy(dtype=np.float64)
        feature_names = list(features.columns)
        for model_spec in config["models"]:
            model_id = str(model_spec["model_id"])
            for fold in range(1, n_splits + 1):
                heldout = assignments == fold
                train = ~heldout
                if set(groups[train].tolist()) & set(groups[heldout].tolist()):
                    raise AssertionError("batch leakage detected before model fitting")
                model = make_model(model_spec, random_state=random_state + fold)
                train_weights = batch_class_sample_weights(
                    labels[train], groups[train]
                )
                fit_with_sample_weights(
                    model,
                    feature_values[train],
                    labels[train],
                    train_weights,
                )
                predicted = model.predict(feature_values[heldout]).astype(np.int64)
                raw_probabilities = model.predict_proba(feature_values[heldout])
                probability_lookup = {
                    int(code): raw_probabilities[:, index]
                    for index, code in enumerate(model.classes_)
                }
                probabilities = np.column_stack(
                    [probability_lookup[code] for code in class_codes]
                )
                metrics = classification_metrics(
                    labels[heldout],
                    predicted,
                    probabilities,
                    class_codes=class_codes,
                )
                fold_metric_rows.append(
                    {
                        "task_id": task_id,
                        "model_id": model_id,
                        "heldout_fold": fold,
                        "record_count": int(np.sum(heldout)),
                        "batch_count": int(np.unique(groups[heldout]).size),
                        **metrics,
                    }
                )
                selected = split_frame.loc[heldout].copy()
                selected["model_id"] = model_id
                selected["predicted_category_code"] = predicted
                selected["predicted_category"] = [
                    CATEGORY_NAMES[int(code)] for code in predicted
                ]
                for index, code in enumerate(class_codes):
                    selected[f"probability_{CATEGORY_NAMES[code].lower()}"] = probabilities[
                        :, index
                    ]
                prediction_frames.append(selected)
                importance_rows.extend(
                    model_importance_rows(
                        model,
                        feature_names=feature_names,
                        task_id=task_id,
                        model_id=model_id,
                        fold=fold,
                    )
                )
        source_files.append(
            {
                "task_id": task_id,
                "relative_path": task["relative_path"],
                "sha256": sha256_file(path),
                "record_count": int(len(labels)),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    batch_metrics = batch_metric_rows(predictions)
    batch_class_distribution = batch_class_distribution_table(batch_metrics)
    bootstrap_intervals = cluster_bootstrap_intervals(
        batch_metrics,
        replicates=int(config["bootstrap_replicates"]),
        random_state=random_state,
    )
    subtype_pressure = subtype_pressure_table(predictions)
    importance = pd.DataFrame(importance_rows)
    feature_schema = pd.DataFrame(feature_schema_rows)
    feature_summary = pd.concat(feature_summary_frames, ignore_index=True)

    model_summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for (task_id, model_id), group in predictions.groupby(
        ["task_id", "model_id"], observed=True
    ):
        class_codes = [
            int(code)
            for code in next(
                task["class_codes"] for task in config["tasks"] if task["task_id"] == task_id
            )
        ]
        labels = group["category_code"].to_numpy(dtype=np.int64)
        predicted = group["predicted_category_code"].to_numpy(dtype=np.int64)
        probabilities = group[
            [f"probability_{CATEGORY_NAMES[code].lower()}" for code in class_codes]
        ].to_numpy(dtype=np.float64)
        pooled = classification_metrics(
            labels, predicted, probabilities, class_codes=class_codes
        )
        selected_folds = fold_metrics.loc[
            fold_metrics["task_id"].eq(task_id)
            & fold_metrics["model_id"].eq(model_id)
        ]
        selected_batches = batch_metrics.loc[
            batch_metrics["task_id"].eq(task_id)
            & batch_metrics["model_id"].eq(model_id)
            & batch_metrics["evaluation_unit"].eq("batch")
        ]
        selected_batch_cells = batch_metrics.loc[
            batch_metrics["task_id"].eq(task_id)
            & batch_metrics["model_id"].eq(model_id)
            & batch_metrics["evaluation_unit"].eq("batch_class_cell")
        ]
        grouped_metrics = batch_class_macro_metrics(selected_batch_cells)
        model_summary_rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "record_count": int(len(group)),
                "batch_count": int(group["batch_code"].nunique()),
                "pooled_accuracy": pooled["accuracy"],
                "pooled_balanced_accuracy": pooled["balanced_accuracy"],
                "pooled_macro_f1": pooled["macro_f1"],
                "pooled_multiclass_log_loss": pooled["multiclass_log_loss"],
                "pooled_macro_ovr_roc_auc": pooled["macro_ovr_roc_auc"],
                "fold_macro_balanced_accuracy": float(
                    selected_folds["balanced_accuracy"].mean()
                ),
                "worst_fold_balanced_accuracy": float(
                    selected_folds["balanced_accuracy"].min()
                ),
                "batch_macro_accuracy": float(selected_batches["accuracy"].mean()),
                "batch_accuracy_p10": float(
                    selected_batches["accuracy"].quantile(0.10)
                ),
                "worst_batch_accuracy": float(selected_batches["accuracy"].min()),
                **grouped_metrics,
            }
        )
        matrix = confusion_matrix(labels, predicted, labels=class_codes)
        for true_index, true_code in enumerate(class_codes):
            for predicted_index, predicted_code in enumerate(class_codes):
                confusion_rows.append(
                    {
                        "task_id": task_id,
                        "model_id": model_id,
                        "true_category_code": true_code,
                        "true_category": CATEGORY_NAMES[true_code],
                        "predicted_category_code": predicted_code,
                        "predicted_category": CATEGORY_NAMES[predicted_code],
                        "confusion_type": "row_count",
                        "value": float(matrix[true_index, predicted_index]),
                    }
                )
        weighted = group.copy()
        cell_sizes = weighted.groupby(
            ["category_code", "batch_code"], observed=True
        )["category_code"].transform("size")
        class_batch_counts = weighted.groupby("category_code", observed=True)[
            "batch_code"
        ].transform("nunique")
        weighted["batch_class_weight"] = 1.0 / (
            cell_sizes.to_numpy(dtype=np.float64)
            * class_batch_counts.to_numpy(dtype=np.float64)
        )
        for true_code in class_codes:
            for predicted_code in class_codes:
                value = weighted.loc[
                    weighted["category_code"].eq(true_code)
                    & weighted["predicted_category_code"].eq(predicted_code),
                    "batch_class_weight",
                ].sum()
                confusion_rows.append(
                    {
                        "task_id": task_id,
                        "model_id": model_id,
                        "true_category_code": true_code,
                        "true_category": CATEGORY_NAMES[true_code],
                        "predicted_category_code": predicted_code,
                        "predicted_category": CATEGORY_NAMES[predicted_code],
                        "confusion_type": "batch_class_row_normalized",
                        "value": float(value),
                    }
                )
    model_summary = pd.DataFrame(model_summary_rows)
    confusion = pd.DataFrame(confusion_rows)
    importance_summary = (
        importance.groupby(
            ["task_id", "model_id", "feature", "importance_type"], observed=True
        )["importance"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["task_id", "model_id", "mean"], ascending=[True, True, False])
    )
    claim_boundaries = pd.DataFrame(
        [
            {
                "claim": "LAT-MRICD X-band batch-code-held-out category baseline",
                "allowed": True,
                "reason": "frozen metadata-only batch split and out-of-fold predictions",
            },
            {
                "claim": "independent session or external blind generalization",
                "allowed": False,
                "reason": "batch semantics are not independently verified",
            },
            {
                "claim": "unseen-model generalization",
                "allowed": False,
                "reason": (
                    "submodels are represented across development folds and many "
                    "have few batches"
                ),
            },
            {
                "claim": "physical micro-Doppler in Hz or velocity",
                "allowed": False,
                "reason": "PRF and continuous timing are not verified",
            },
            {
                "claim": "polarimetric or balloon-payload recognition",
                "allowed": False,
                "reason": "the public dataset has no H/V pair or balloon/payload labels",
            },
            {
                "claim": "Tian 2024 exact reproduction",
                "allowed": False,
                "reason": "input, label, preprocessing and output alignment remain unavailable",
            },
        ]
    )

    summary = {
        "status": "COMPLETE_GROUPED_PUBLIC_DATA_BASELINE",
        "experiment_id": config["experiment_id"],
        "dataset": config.get("dataset", "LAT-MRICD-1.0"),
        "implementation_commit": current_commit(),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "n_splits": n_splits,
        "random_state": random_state,
        "group_key": config["group_key"],
        "splitter": config["splitter"],
        "frozen_split_manifest_sha256": frozen_split_summary[
            "split_manifest_sha256"
        ],
        "task_count": len(config["tasks"]),
        "model_count": len(config["models"]),
        "source_files": source_files,
        "group_leakage_detected": False,
        "all_rows_heldout_once": True,
        "all_folds_cover_all_classes": True,
        "hyperparameter_search_performed": False,
        "raw_data_in_output": False,
        "model_checkpoints_saved": False,
        "training_weighting": "class_equal_then_batch_class_cell_equal",
        "primary_group_metric": "batch_class_macro_accuracy",
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "physical_frequency_hz_reported": False,
        "claim_scope": "internal batch-grouped public-data baseline",
    }

    csv_outputs = {
        "split_manifest.csv": frozen_split,
        "fold_coverage.csv": pd.DataFrame(coverage_rows),
        "feature_definitions.csv": feature_schema,
        "feature_summary_by_category.csv": feature_summary,
        "oof_predictions.csv": predictions,
        "fold_metrics.csv": fold_metrics,
        "batch_class_metrics.csv": batch_metrics,
        "batch_class_distribution.csv": batch_class_distribution,
        "aggregate_metrics.csv": model_summary,
        "confusion_matrices.csv": confusion,
        "feature_importance_by_fold.csv": importance,
        "feature_importance.csv": importance_summary,
        "cluster_bootstrap_intervals.csv": bootstrap_intervals,
        "subtype_pressure.csv": subtype_pressure,
        "claim_boundaries.csv": claim_boundaries,
    }
    for name, frame in csv_outputs.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        make_report(summary, model_summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_baseline(
        config_path=args.config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
