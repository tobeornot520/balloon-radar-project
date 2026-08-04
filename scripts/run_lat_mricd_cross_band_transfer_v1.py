#!/usr/bin/env python3
"""Run the preregistered LAT-MRICD band-held-out transfer study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lat_mricd_grouped_baseline_v1 import (  # noqa: E402
    extract_hrrp_features,
    extract_narrow_features,
    fit_with_sample_weights,
    make_model,
    reconstruct_narrow_iq,
    single_public_matrix,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/lat_mricd_cross_band_transfer_v1.json"
GROUPED_BASELINE_CONFIG = PROJECT_ROOT / "configs/lat_mricd_grouped_baseline_v1.json"
FORMAL_OUTPUT_RELATIVE_PATH = "results/experiments/lat_mricd_cross_band_transfer_v1"
FORMAL_OUTPUT_DIR = (PROJECT_ROOT / FORMAL_OUTPUT_RELATIVE_PATH).resolve()
CLASS_CODES = (1, 3)
CLASS_NAMES = {1: "UAV", 3: "weather"}
PROBABILITY_COLUMNS = {1: "probability_uav", 3: "probability_weather"}

AGGREGATE_SOURCE_WHITELIST: dict[str, dict[str, Any]] = {
    "HRRP/X波段/data_hrrp_X.mat": {
        "source_id": "hrrp_x",
        "expected_sha256": "276029f334e24abf9e54860d3f58e99557ffd47398c602e49d4139f3a25ae267",
        "representation": "HRRP",
        "band_code": 2,
        "band": "X",
        "record_count": 3648,
        "column_count": 504,
    },
    "HRRP/Ku波段/data_hrrp_Ku.mat": {
        "source_id": "hrrp_ku",
        "expected_sha256": "985d679510b272e90180a77cdfa64c354909a45bc6b20cb00d176b27fbe44ed4",
        "representation": "HRRP",
        "band_code": 3,
        "band": "Ku",
        "record_count": 3471,
        "column_count": 504,
    },
    "Narrow/S波段/data_narrow_S.mat": {
        "source_id": "narrow_s",
        "expected_sha256": "e584b3ed8c0117265cfebe61bd289aa91c1d917aaa504e7cb21118615b5db328",
        "representation": "Narrow",
        "band_code": 1,
        "band": "S",
        "record_count": 4038,
        "column_count": 1028,
    },
    "Narrow/X波段/data_narrow_X.mat": {
        "source_id": "narrow_x",
        "expected_sha256": "da8ea23032929a60de67fad7a46b7616f068885efc02042a22b747da9f6e24c4",
        "representation": "Narrow",
        "band_code": 2,
        "band": "X",
        "record_count": 8715,
        "column_count": 1028,
    },
    "Narrow/Ku波段/data_narrow_Ku.mat": {
        "source_id": "narrow_ku",
        "expected_sha256": "92246be376fe93b00bb6bb56d64882916297f88e8ac20839daced3c9a2c8d926",
        "representation": "Narrow",
        "band_code": 3,
        "band": "Ku",
        "record_count": 3319,
        "column_count": 1028,
    },
}

EXPECTED_TRANSFERS = {
    "narrow_x_to_s_shared_binary": (
        "Narrow",
        ("narrow_x",),
        "narrow_s",
        "S",
        "locked_primary",
    ),
    "narrow_x_to_ku_shared_binary": (
        "Narrow",
        ("narrow_x",),
        "narrow_ku",
        "Ku",
        "locked_primary",
    ),
    "narrow_s_to_ku_shared_binary": (
        "Narrow",
        ("narrow_s",),
        "narrow_ku",
        "Ku",
        "secondary",
    ),
    "narrow_ku_to_s_shared_binary": (
        "Narrow",
        ("narrow_ku",),
        "narrow_s",
        "S",
        "secondary",
    ),
    "narrow_x_ku_to_s_shared_binary": (
        "Narrow",
        ("narrow_x", "narrow_ku"),
        "narrow_s",
        "S",
        "secondary",
    ),
    "narrow_x_s_to_ku_shared_binary": (
        "Narrow",
        ("narrow_x", "narrow_s"),
        "narrow_ku",
        "Ku",
        "secondary",
    ),
    "hrrp_x_to_ku_binary": (
        "HRRP",
        ("hrrp_x",),
        "hrrp_ku",
        "Ku",
        "exploratory",
    ),
}

OUTPUT_CSV_FILES = (
    "transfer_coverage.csv",
    "raw_batch_overlap_audit.csv",
    "training_weight_audit.csv",
    "aggregate_metrics.csv",
    "target_batch_class_metrics.csv",
    "bootstrap_intervals.csv",
    "confusion_matrices.csv",
    "feature_definitions.csv",
    "feature_importance.csv",
    "disjoint_sensitivity.csv",
    "claim_boundaries.csv",
)
OUTPUT_JSON_FILES = (
    "model_fit_manifest.json",
    "gate_decision.json",
    "summary.json",
)
OUTPUT_FILES = (*OUTPUT_CSV_FILES, *OUTPUT_JSON_FILES, "REPORT.md")
FROZEN_MODEL_FIT_COUNT = 27
FROZEN_BOOTSTRAP_INTERVAL_COUNT = 36


def expected_model_fit_count(config: dict[str, Any]) -> int:
    reportable = sum(
        transfer["raw_batch_disjoint_sensitivity"]["status"] == "REPORTABLE"
        for transfer in config["transfers"]
    )
    return (len(config["transfers"]) + reportable) * len(config["models"])


def expected_bootstrap_interval_count(config: dict[str, Any]) -> int:
    reportable = sum(
        transfer["raw_batch_disjoint_sensitivity"]["status"] == "REPORTABLE"
        for transfer in config["transfers"]
    )
    return (len(config["transfers"]) + reportable) * (
        len(config["models"]) + 1
    )


@dataclass(frozen=True)
class PreparedBand:
    source_id: str
    relative_path: str
    representation: str
    band_code: int
    band: str
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    full_batch_codes: frozenset[int]
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed source-only classifiers on held-out LAT-MRICD bands."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
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


def _git_file_sha256(commit: str, path: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"formal-run file is outside the project: {path}") from exc
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"formal-run commit does not contain {relative_path}")
    return hashlib.sha256(result.stdout).hexdigest()


def validate_pre_result_repository_state(
    config_path: Path, config: dict[str, Any]
) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("formal run requires a clean Git worktree")
    commit = current_commit()
    feature_path = PROJECT_ROOT / config["feature_contract"]["implementation"]
    required_paths = (
        config_path,
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests/test_lat_mricd_cross_band_transfer.py",
        feature_path,
    )
    for path in required_paths:
        if _git_file_sha256(commit, path) != sha256_file(path):
            raise ValueError(f"formal-run commit binding is stale for {path.name}")
    frozen_feature_commit = config["feature_contract"]["frozen_baseline_commit"]
    if _git_file_sha256(frozen_feature_commit, feature_path) != config[
        "feature_contract"
    ]["implementation_sha256"]:
        raise ValueError("frozen feature commit/hash binding is invalid")
    return commit


def _require_false(mapping: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if mapping.get(field) is not False:
            raise ValueError(f"{field} must remain false")


def validate_aggregate_sources(config: dict[str, Any]) -> None:
    sources = config.get("aggregate_sources", [])
    if len(sources) != len(AGGREGATE_SOURCE_WHITELIST):
        raise ValueError("exactly five aggregate MAT sources are required")
    source_ids = [str(source.get("source_id", "")) for source in sources]
    relative_paths = [str(source.get("relative_path", "")) for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("aggregate source_id values must be unique")
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("aggregate source paths must be unique")
    if set(relative_paths) != set(AGGREGATE_SOURCE_WHITELIST):
        raise ValueError("aggregate sources must match the five-file whitelist exactly")

    for source in sources:
        expected_fields = {
            "source_id",
            "relative_path",
            "expected_sha256",
            "representation",
            "band_code",
            "band",
            "expected_record_count",
            "expected_column_count",
            "expected_analysis_coverage",
        }
        if set(source) != expected_fields:
            raise ValueError(
                f"aggregate source schema mismatch: {source.get('source_id', '<unknown>')}"
            )
        relative_path = str(source["relative_path"])
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe aggregate source path: {relative_path}")
        expected = AGGREGATE_SOURCE_WHITELIST[relative_path]
        observed = {
            "source_id": source.get("source_id"),
            "expected_sha256": source.get("expected_sha256"),
            "representation": source.get("representation"),
            "band_code": int(source.get("band_code", -1)),
            "band": source.get("band"),
            "record_count": int(source.get("expected_record_count", -1)),
            "column_count": int(source.get("expected_column_count", -1)),
        }
        if observed != expected:
            raise ValueError(f"aggregate source contract mismatch: {relative_path}")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("expected_sha256", ""))
        ):
            raise ValueError(f"invalid source sha256: {relative_path}")


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "status",
        "dataset_root",
        "output_dir",
        "random_state",
        "aggregate_sources",
        "class_contract",
        "group_key",
        "feature_contract",
        "transfers",
        "models",
        "training_contract",
        "metrics",
        "bootstrap_replicates",
        "bootstrap_contract",
        "raw_batch_overlap_audit",
        "output_contract",
        "acceptance_contract",
        "stopping_rule",
        "claim_contract",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config missing fields: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    if config["experiment_id"] != "lat_mricd_cross_band_transfer_v1":
        raise ValueError("unexpected experiment_id")
    if config["status"] != "PREREGISTERED_NOT_RUN":
        raise ValueError("cross-band protocol must be preregistered and unrun")
    if config["output_dir"] != FORMAL_OUTPUT_RELATIVE_PATH:
        raise ValueError("formal output directory must remain fixed")

    validate_aggregate_sources(config)
    classes = config["class_contract"]
    if classes.get("class_codes") != [1, 3]:
        raise ValueError("cross-band task must remain UAV/weather binary")
    if classes.get("bird_rows_allowed") is not False:
        raise ValueError("bird rows are outside the cross-band class intersection")
    if classes.get("narrow_shared_uav_model_codes") != [1, 2, 3]:
        raise ValueError("Narrow transfers must use shared UAV models 1/2/3")
    if config["group_key"] != ["representation", "band_code", "batch_code"]:
        raise ValueError("group_key must remain band-qualified")

    feature_contract = config["feature_contract"]
    if feature_contract.get("feature_set_id") != "lat_mricd_grouped_baseline_v1_exact":
        raise ValueError("cross-band features must reuse the frozen grouped baseline")
    _require_false(
        feature_contract,
        (
            "feature_selection_allowed",
            "pca_allowed",
            "band_code_as_feature_allowed",
            "category_model_or_batch_metadata_as_feature_allowed",
            "target_band_statistics_for_feature_processing_allowed",
        ),
    )
    feature_implementation = PROJECT_ROOT / str(feature_contract.get("implementation", ""))
    if feature_implementation.resolve() != (
        PROJECT_ROOT / "scripts/run_lat_mricd_grouped_baseline_v1.py"
    ).resolve():
        raise ValueError("unexpected frozen feature implementation")
    if sha256_file(feature_implementation) != feature_contract.get(
        "implementation_sha256"
    ):
        raise ValueError("frozen feature implementation hash mismatch")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(feature_contract.get("frozen_baseline_commit", ""))
    ):
        raise ValueError("invalid frozen baseline commit")
    feature_schemas = feature_contract.get("feature_schemas")
    if not isinstance(feature_schemas, dict) or set(feature_schemas) != {
        "Narrow",
        "HRRP",
    }:
        raise ValueError("the Narrow and HRRP feature schemas must both be frozen")
    for representation, schema in feature_schemas.items():
        names = schema.get("feature_names") if isinstance(schema, dict) else None
        if (
            not isinstance(names, list)
            or not names
            or len(names) != len(set(names))
            or schema.get("feature_count") != len(names)
        ):
            raise ValueError(f"invalid frozen {representation} feature schema")

    source_by_id = {source["source_id"]: source for source in config["aggregate_sources"]}
    transfers = {transfer["transfer_id"]: transfer for transfer in config["transfers"]}
    if set(transfers) != set(EXPECTED_TRANSFERS):
        raise ValueError("transfer directions differ from the frozen seven-direction design")
    for transfer_id, expected in EXPECTED_TRANSFERS.items():
        transfer = transfers[transfer_id]
        representation, source_ids, target_id, target_band, role = expected
        observed = (
            transfer.get("representation"),
            tuple(transfer.get("source_ids", [])),
            transfer.get("target_source_id"),
            transfer.get("target_band"),
            transfer.get("role"),
        )
        if observed != expected:
            raise ValueError(f"transfer contract mismatch: {transfer_id}")
        if target_band == "X":
            raise ValueError("X-band targets are forbidden after baseline inspection")
        if target_id in source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError(f"source/target overlap in {transfer_id}")
        referenced = [*source_ids, target_id]
        if any(source_id not in source_by_id for source_id in referenced):
            raise ValueError(f"unknown aggregate source in {transfer_id}")
        if any(source_by_id[source_id]["representation"] != representation for source_id in referenced):
            raise ValueError(f"representation mismatch in {transfer_id}")
        expected_bands = [source_by_id[source_id]["band"] for source_id in source_ids]
        if transfer.get("source_bands") != expected_bands:
            raise ValueError(f"source_bands mismatch in {transfer_id}")
        expected_uav_codes = [1, 2, 3] if representation == "Narrow" else [1, 2, 3, 4, 5, 6, 10]
        if transfer.get("uav_model_codes") != expected_uav_codes:
            raise ValueError(f"UAV model filter mismatch in {transfer_id}")

    grouped_models = json.loads(
        GROUPED_BASELINE_CONFIG.read_text(encoding="utf-8")
    )["models"]
    if config["models"] != grouped_models:
        raise ValueError("model specifications must exactly match grouped baseline V1")

    training = config["training_contract"]
    if training.get("fit_scope") != "source_bands_only":
        raise ValueError("fit_scope must be source_bands_only")
    _require_false(
        training,
        (
            "target_rows_used_for_fit",
            "target_labels_used_for_fit_threshold_calibration_or_model_selection",
            "target_statistics_used_for_scaling_or_calibration",
            "threshold_tuning_allowed",
            "hyperparameter_search_allowed",
            "test_driven_model_selection_allowed",
        ),
    )
    if training.get("sample_weight_hierarchy") != [
        "category_code",
        "source_band_code",
        "source_band_batch_class_cell",
    ]:
        raise ValueError("unexpected multi-source weighting hierarchy")
    expected_training_values = {
        "scaler_fit_scope": "source_bands_only",
        "probability_calibration": "none",
        "prediction_decision": "argmax_over_model_classes_in_ascending_class_code_order",
        "argmax_tie_break": "lowest_class_code",
        "dummy_prior_fit_scope": "source_bands_only",
        "dummy_prior_uses_same_sample_weights_as_learned_models": True,
        "all_fixed_models_retained": True,
        "single_sealed_run_before_result_inspection": True,
    }
    for field, expected_value in expected_training_values.items():
        if training.get(field) != expected_value:
            raise ValueError(f"unexpected training contract {field}")

    if int(config["bootstrap_replicates"]) != 2000:
        raise ValueError("formal target-batch bootstrap must use 2000 replicates")
    bootstrap = config["bootstrap_contract"]
    if bootstrap.get("paired_comparison") != "logistic_batch_balanced_minus_dummy_prior":
        raise ValueError("paired bootstrap comparison must remain LR minus dummy")
    if bootstrap.get("paired_resamples_must_use_identical_target_batches") is not True:
        raise ValueError("paired bootstrap draws must be identical")
    if (
        bootstrap.get("confidence_interval_conditioning")
        != "conditional_on_each_fixed_source_fit"
    ):
        raise ValueError("bootstrap intervals must condition on each fixed source fit")
    if bootstrap.get("resample_complete_target_raw_batch_code_clusters") is not True:
        raise ValueError("bootstrap must resample complete target batch-code clusters")
    expected_bootstrap = {
        "interval_method": "percentile",
        "percentile_quantile_method": "linear",
        "analysis_scopes": [
            "band_qualified_primary",
            "raw_batch_code_disjoint_sensitivity",
        ],
        "raw_batch_code_disjoint_sensitivity_included_only_when_reportable": True,
        "draw_count_per_replicate": "number_of_unique_target_batch_codes_in_transfer_scope",
        "sample_with_replacement": True,
        "duplicate_batch_draws_count_with_multiplicity": True,
        "seed_derivation": "uint32_from_first_8_hex_sha256_of_random_state_pipe_analysis_scope_pipe_transfer_id",
    }
    for field, expected_value in expected_bootstrap.items():
        if bootstrap.get(field) != expected_value:
            raise ValueError(f"unexpected bootstrap contract {field}")

    acceptance = config["acceptance_contract"]
    if acceptance.get("aggregate_file_count") != 5:
        raise ValueError("the data contract requires exactly five aggregate files")
    if acceptance.get("class_codes_must_equal") != [1, 3]:
        raise ValueError("acceptance class codes must remain binary")
    if acceptance.get("target_x_allowed") is not False:
        raise ValueError("target X must remain forbidden")
    for field in (
        "formal_run_requires_clean_git_worktree",
        "formal_run_records_current_pre_result_commit",
        "pre_result_commit_must_contain_config_runner_tests_and_frozen_feature_implementation",
        "consumption_record_must_be_absent_before_formal_run",
        "record_before_target_load",
        "persist_on_failure",
    ):
        if acceptance.get(field) is not True:
            raise ValueError(f"formal-run acceptance field must remain true: {field}")
    if (
        acceptance.get("formal_run_consumption_record")
        != "results/final_evidence/lat_mricd_cross_band_transfer_v1.run_consumed.json"
    ):
        raise ValueError("unexpected formal-run consumption record")
    if acceptance.get("formal_output_overwrite_allowed") is not False:
        raise ValueError("formal output overwrite must remain forbidden")
    if (
        acceptance.get(
            "minimum_batch_count_per_source_band_and_class_for_reportable_disjoint_sensitivity"
        )
        != 3
    ):
        raise ValueError("disjoint sensitivity requires three batches per band and class")
    _require_false(
        acceptance,
        (
            "detail_files_allowed",
            "aggregate_and_detail_files_loaded_together_allowed",
            "physical_frequency_hz_reporting_allowed",
            "same_event_cross_band_fusion_allowed",
            "paired_cross_band_feature_comparison_allowed",
            "unseen_model_generalization_claim_allowed",
            "independent_session_or_external_generalization_claim_allowed",
        ),
    )
    output_contract = config["output_contract"]
    if output_contract.get("file_count") != len(OUTPUT_FILES) or output_contract.get(
        "files"
    ) != list(OUTPUT_FILES):
        raise ValueError("output contract must list the exact 15 aggregate-only files")
    _require_false(
        output_contract,
        (
            "sample_level_predictions_allowed",
            "oof_predictions_allowed",
            "raw_data_allowed",
            "per_sample_weights_allowed",
            "model_checkpoints_allowed",
            "raw_or_absolute_paths_allowed",
        ),
    )
    stopping = config["stopping_rule"]
    if stopping.get("uses_unrounded_values") is not True:
        raise ValueError("the stopping gate must use unrounded values")
    if (
        stopping.get("same_target_reuse_for_future_confirmatory_comparison_allowed")
        is not False
    ):
        raise ValueError("the same targets cannot be reused for confirmatory comparison")
    if expected_model_fit_count(config) != FROZEN_MODEL_FIT_COUNT:
        raise ValueError("the frozen design must produce exactly 27 source fits")
    if (
        expected_bootstrap_interval_count(config)
        != FROZEN_BOOTSTRAP_INTERVAL_COUNT
    ):
        raise ValueError("the frozen design must produce exactly 36 bootstrap intervals")

    frozen = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if config != frozen:
        raise ValueError("config differs from the frozen cross-band preregistration")
    return config


def _metadata_from_matrix(source: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    expected_shape = (
        int(source["expected_record_count"]),
        int(source["expected_column_count"]),
    )
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{source['source_id']}: expected matrix {expected_shape}, got {matrix.shape}"
        )
    metadata = np.rint(matrix[:, :4]).astype(np.int64)
    if not np.allclose(matrix[:, :4], metadata):
        raise ValueError(f"{source['source_id']}: metadata columns must be integer codes")
    if set(metadata[:, 0].tolist()) != {int(source["band_code"])}:
        raise ValueError(f"{source['source_id']}: unexpected band code")
    if not set(metadata[:, 1].tolist()) <= {1, 2, 3}:
        raise ValueError(f"{source['source_id']}: unexpected category code")
    return metadata


def prepare_aggregate_source(
    source: dict[str, Any],
    matrix: np.ndarray,
    class_contract: dict[str, Any],
    *,
    source_sha256: str = "synthetic",
) -> PreparedBand:
    metadata = _metadata_from_matrix(source, matrix)
    representation = str(source["representation"])
    category = metadata[:, 1]
    model = metadata[:, 2]
    full_batch_codes = frozenset(int(value) for value in metadata[:, 3])

    if representation == "Narrow":
        uav_codes = set(int(value) for value in class_contract["narrow_shared_uav_model_codes"])
        weather_codes = set(int(value) for value in class_contract["narrow_weather_model_codes"])
    elif representation == "HRRP":
        key = "hrrp_x_uav_model_codes" if source["band"] == "X" else "hrrp_ku_uav_model_codes"
        uav_codes = set(int(value) for value in class_contract[key])
        weather_codes = set(int(value) for value in class_contract["hrrp_weather_model_codes"])
    else:
        raise ValueError(f"unsupported representation: {representation}")
    selected = ((category == 1) & np.isin(model, list(uav_codes))) | (
        (category == 3) & np.isin(model, list(weather_codes))
    )
    selected_matrix = np.asarray(matrix)[selected]
    selected_metadata = metadata[selected]
    if set(selected_metadata[:, 1].tolist()) != set(CLASS_CODES):
        raise ValueError(f"{source['source_id']}: binary analysis subset lacks a class")

    if representation == "Narrow":
        features = extract_narrow_features(reconstruct_narrow_iq(selected_matrix))
    else:
        features = extract_hrrp_features(selected_matrix[:, 4:])
    feature_values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError(f"{source['source_id']}: extracted features contain NaN or Inf")

    frame = pd.DataFrame(
        {
            "source_row_index": np.flatnonzero(selected),
            "source_id": source["source_id"],
            "representation": representation,
            "band_code": selected_metadata[:, 0],
            "band": source["band"],
            "category_code": selected_metadata[:, 1],
            "model_code": selected_metadata[:, 2],
            "batch_code": selected_metadata[:, 3],
        }
    )
    frame = pd.concat([frame, features.reset_index(drop=True)], axis=1)
    return PreparedBand(
        source_id=str(source["source_id"]),
        relative_path=str(source["relative_path"]),
        representation=representation,
        band_code=int(source["band_code"]),
        band=str(source["band"]),
        frame=frame,
        feature_names=tuple(features.columns),
        full_batch_codes=full_batch_codes,
        sha256=source_sha256,
    )


def load_aggregate_sources(
    config: dict[str, Any],
    dataset_root: Path,
    *,
    matrix_loader: Callable[[Path], np.ndarray] = single_public_matrix,
) -> dict[str, PreparedBand]:
    validate_aggregate_sources(config)
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    loaded_paths: set[Path] = set()
    prepared: dict[str, PreparedBand] = {}
    for source in config["aggregate_sources"]:
        path = (dataset_root / source["relative_path"]).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"aggregate source escapes dataset root: {path}") from exc
        if path in loaded_paths:
            raise ValueError(f"aggregate matrix would be loaded twice: {path}")
        loaded_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"aggregate matrix not found: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != source["expected_sha256"]:
            raise ValueError(f"aggregate source hash mismatch: {source['source_id']}")
        item = prepare_aggregate_source(
            source,
            matrix_loader(path),
            config["class_contract"],
            source_sha256=actual_sha256,
        )
        expected_coverage = source["expected_analysis_coverage"]
        observed_coverage = {
            "record_count": int(len(item.frame)),
            "unique_batch_count": int(item.frame["batch_code"].nunique()),
            "classes": {
                str(code): {
                    "record_count": int(
                        item.frame["category_code"].eq(code).sum()
                    ),
                    "batch_count": int(
                        item.frame.loc[
                            item.frame["category_code"].eq(code), "batch_code"
                        ].nunique()
                    ),
                }
                for code in CLASS_CODES
            },
        }
        if observed_coverage != expected_coverage:
            raise ValueError(
                f"{source['source_id']}: analysis coverage differs from the frozen contract"
            )
        expected_features = tuple(
            config["feature_contract"]["feature_schemas"][item.representation][
                "feature_names"
            ]
        )
        if item.feature_names != expected_features:
            raise ValueError(
                f"{source['source_id']}: extracted feature schema differs from frozen V1"
            )
        prepared[source["source_id"]] = item
    return prepared


def assemble_transfer_frames(
    transfer: dict[str, Any],
    sources: dict[str, PreparedBand],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    source_items = [sources[source_id] for source_id in transfer["source_ids"]]
    target_item = sources[transfer["target_source_id"]]
    if target_item.source_id in {item.source_id for item in source_items}:
        raise ValueError("target source cannot appear in source training data")
    if any(item.representation != transfer["representation"] for item in [*source_items, target_item]):
        raise ValueError("transfer mixes representations")
    feature_names = source_items[0].feature_names
    if any(item.feature_names != feature_names for item in [*source_items[1:], target_item]):
        raise ValueError("source and target feature schemas differ")

    allowed_uav_models = set(int(value) for value in transfer["uav_model_codes"])

    def select(item: PreparedBand) -> pd.DataFrame:
        frame = item.frame
        selected = frame["category_code"].eq(3) | (
            frame["category_code"].eq(1)
            & frame["model_code"].isin(allowed_uav_models)
        )
        return frame.loc[selected].copy()

    source_frame = pd.concat([select(item) for item in source_items], ignore_index=True)
    target_frame = select(target_item).reset_index(drop=True)
    if set(source_frame["category_code"].unique()) != set(CLASS_CODES):
        raise ValueError("source training data lacks a binary class")
    if set(target_frame["category_code"].unique()) != set(CLASS_CODES):
        raise ValueError("target evaluation data lacks a binary class")
    for band, frame in source_frame.groupby("band", observed=True):
        if set(frame["category_code"].unique()) != set(CLASS_CODES):
            raise ValueError(f"source band {band} lacks a binary class")
    return source_frame, target_frame, feature_names


def cross_band_training_weights(
    labels: np.ndarray,
    source_band_codes: np.ndarray,
    batch_codes: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "category_code": np.asarray(labels, dtype=np.int64),
            "source_band_code": np.asarray(source_band_codes, dtype=np.int64),
            "batch_code": np.asarray(batch_codes, dtype=np.int64),
        }
    )
    if frame.empty or any(len(frame[column]) != len(frame) for column in frame):
        raise ValueError("training metadata must be nonempty and aligned")
    if set(frame["category_code"].unique()) != set(CLASS_CODES):
        raise ValueError("training weights require UAV and weather rows")
    cell_keys = ["category_code", "source_band_code", "batch_code"]
    cell_sizes = frame.groupby(cell_keys, observed=True)["category_code"].transform("size")
    batch_counts = frame.groupby(
        ["category_code", "source_band_code"], observed=True
    )["batch_code"].transform("nunique")
    band_counts = frame.groupby("category_code", observed=True)[
        "source_band_code"
    ].transform("nunique")
    weights = 1.0 / (
        cell_sizes.to_numpy(dtype=np.float64)
        * batch_counts.to_numpy(dtype=np.float64)
        * band_counts.to_numpy(dtype=np.float64)
    )
    weights /= weights.mean()
    return weights


def training_weight_audit_rows(
    transfer: dict[str, Any], source_frame: pd.DataFrame, weights: np.ndarray
) -> pd.DataFrame:
    frame = source_frame[["band", "band_code", "category_code", "batch_code"]].copy()
    frame["sample_weight"] = weights
    rows: list[dict[str, Any]] = []
    for (band, band_code, category), group in frame.groupby(
        ["band", "band_code", "category_code"], observed=True
    ):
        cell_totals = group.groupby("batch_code", observed=True)["sample_weight"].sum()
        rows.append(
            {
                "transfer_id": transfer["transfer_id"],
                "source_band": band,
                "source_band_code": int(band_code),
                "category_code": int(category),
                "category": CLASS_NAMES[int(category)],
                "record_count": int(len(group)),
                "batch_count": int(group["batch_code"].nunique()),
                "total_weight": float(group["sample_weight"].sum()),
                "minimum_batch_cell_weight": float(cell_totals.min()),
                "maximum_batch_cell_weight": float(cell_totals.max()),
            }
        )
    return pd.DataFrame(rows)


def fit_source_only_model(
    model: Any,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_weights: np.ndarray,
    target_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    source_weights = np.asarray(source_weights, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("source and target features must be matrices")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target feature widths differ")
    if len(source_features) != len(source_labels) or len(source_labels) != len(source_weights):
        raise ValueError("source fit arrays are not aligned")
    if set(source_labels.tolist()) != set(CLASS_CODES):
        raise ValueError("source fit must contain both frozen classes")
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise ValueError("source and target features must be finite")
    if (
        not np.isfinite(source_weights).all()
        or np.any(source_weights <= 0)
        or not np.isclose(source_weights.mean(), 1.0)
    ):
        raise ValueError("source sample weights must be positive, finite and mean-one")
    fit_with_sample_weights(model, source_features, source_labels, source_weights)
    predicted = np.asarray(model.predict(target_features), dtype=np.int64)
    raw_probabilities = np.asarray(model.predict_proba(target_features), dtype=np.float64)
    class_lookup = {int(code): index for index, code in enumerate(model.classes_)}
    if set(class_lookup) != set(CLASS_CODES):
        raise ValueError("fitted model does not expose both frozen classes")
    probabilities = np.column_stack(
        [raw_probabilities[:, class_lookup[code]] for code in CLASS_CODES]
    )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-9
    ):
        raise ValueError("model returned invalid target probabilities")
    argmax_predictions = np.asarray(CLASS_CODES, dtype=np.int64)[
        np.argmax(probabilities, axis=1)
    ]
    if not np.array_equal(predicted, argmax_predictions):
        raise ValueError(
            "model predictions violate ascending-class argmax and lowest-code tie break"
        )
    return predicted, probabilities


def binary_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), 2):
        raise ValueError("binary metrics require a two-column probability matrix")
    if not set(labels) <= set(CLASS_CODES) or not set(predictions) <= set(CLASS_CODES):
        raise ValueError("binary metrics received an unknown category")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("binary probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("binary probabilities must sum to one")
    try:
        roc_auc = float(roc_auc_score(labels == 1, probabilities[:, 0]))
    except ValueError:
        roc_auc = math.nan
    return {
        "pooled_accuracy": float(accuracy_score(labels, predictions)),
        "pooled_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "pooled_macro_f1": float(
            f1_score(labels, predictions, labels=list(CLASS_CODES), average="macro")
        ),
        "binary_log_loss": float(
            log_loss(labels, probabilities, labels=list(CLASS_CODES))
        ),
        "roc_auc": roc_auc,
        "recall_uav": float(np.mean(predictions[labels == 1] == 1)),
        "recall_weather": float(np.mean(predictions[labels == 3] == 3)),
    }


def source_fit_feature_importance_rows(
    model: Any,
    *,
    transfer: dict[str, Any],
    model_id: str,
    analysis_scope: str,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        values = np.mean(np.abs(np.asarray(classifier.coef_)), axis=0)
        importance_type = "mean_abs_standardized_multiclass_coefficient"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        importance_type = "mean_decrease_impurity"
    else:
        return []
    if len(values) != len(feature_names) or not np.isfinite(values).all():
        raise ValueError(f"{model_id}: invalid source-fit feature importance")
    return [
        {
            "transfer_id": transfer["transfer_id"],
            "role": transfer["role"],
            "analysis_scope": analysis_scope,
            "source_bands": "+".join(transfer["source_bands"]),
            "target_band": transfer["target_band"],
            "model_id": model_id,
            "feature": feature,
            "importance": float(value),
            "importance_type": importance_type,
            "fit_scope": "source_bands_only",
        }
        for feature, value in zip(feature_names, values, strict=True)
    ]


def source_fit_manifest_row(
    *,
    transfer: dict[str, Any],
    model_spec: dict[str, Any],
    analysis_scope: str,
    random_state: int,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    source_cells = source_frame[
        ["band_code", "category_code", "batch_code"]
    ].drop_duplicates()
    return {
        "transfer_id": transfer["transfer_id"],
        "role": transfer["role"],
        "analysis_scope": analysis_scope,
        "model_id": model_spec["model_id"],
        "source_ids": list(transfer["source_ids"]),
        "source_bands": list(transfer["source_bands"]),
        "target_source_id": transfer["target_source_id"],
        "target_band": transfer["target_band"],
        "random_state": int(random_state),
        "feature_count": len(feature_names),
        "source_record_count": int(len(source_frame)),
        "source_batch_class_cell_count": int(len(source_cells)),
        "target_record_count_evaluated": int(len(target_frame)),
        "fit_scope": "source_bands_only",
        "scaler_fit_scope": "source_bands_only",
        "target_rows_used_for_fit": False,
        "target_labels_used_for_fit_threshold_calibration_or_model_selection": False,
        "target_statistics_used_for_scaling_or_calibration": False,
        "sample_weights_fit_scope": "source_bands_only",
        "probability_calibration": "none",
        "threshold_tuning_performed": False,
        "hyperparameter_search_performed": False,
        "test_driven_model_selection_performed": False,
    }


def evaluate_models(
    transfer: dict[str, Any],
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    model_specs: list[dict[str, Any]],
    *,
    random_state: int,
    analysis_scope: str,
    importance_rows: list[dict[str, Any]] | None = None,
    fit_rows: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_features = source_frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    target_features = target_frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    source_labels = source_frame["category_code"].to_numpy(dtype=np.int64)
    source_weights = cross_band_training_weights(
        source_labels,
        source_frame["band_code"].to_numpy(dtype=np.int64),
        source_frame["batch_code"].to_numpy(dtype=np.int64),
    )
    weight_audit = training_weight_audit_rows(transfer, source_frame, source_weights)
    weight_audit.insert(1, "analysis_scope", analysis_scope)
    prediction_frames: list[pd.DataFrame] = []
    for model_index, model_spec in enumerate(model_specs):
        fit_random_state = random_state + model_index
        model = make_model(model_spec, random_state=fit_random_state)
        predicted, probabilities = fit_source_only_model(
            model,
            source_features,
            source_labels,
            source_weights,
            target_features,
        )
        frame = target_frame[
            [
                "source_id",
                "representation",
                "band_code",
                "band",
                "category_code",
                "model_code",
                "batch_code",
            ]
        ].copy()
        frame.insert(0, "transfer_id", transfer["transfer_id"])
        frame.insert(1, "role", transfer["role"])
        frame.insert(2, "analysis_scope", analysis_scope)
        frame.insert(3, "source_bands", "+".join(transfer["source_bands"]))
        frame["model_id"] = model_spec["model_id"]
        frame["predicted_category_code"] = predicted
        frame["probability_uav"] = probabilities[:, 0]
        frame["probability_weather"] = probabilities[:, 1]
        prediction_frames.append(frame)
        if importance_rows is not None:
            importance_rows.extend(
                source_fit_feature_importance_rows(
                    model,
                    transfer=transfer,
                    model_id=str(model_spec["model_id"]),
                    analysis_scope=analysis_scope,
                    feature_names=feature_names,
                )
            )
        if fit_rows is not None:
            fit_rows.append(
                source_fit_manifest_row(
                    transfer=transfer,
                    model_spec=model_spec,
                    analysis_scope=analysis_scope,
                    random_state=fit_random_state,
                    source_frame=source_frame,
                    target_frame=target_frame,
                    feature_names=feature_names,
                )
            )
    return pd.concat(prediction_frames, ignore_index=True), weight_audit


def target_batch_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correct"] = frame["category_code"].eq(
        frame["predicted_category_code"]
    )
    group_prefix = [
        "transfer_id",
        "role",
        "analysis_scope",
        "source_bands",
        "model_id",
        "representation",
        "band_code",
        "band",
        "batch_code",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_prefix, observed=True, sort=True):
        row = dict(zip(group_prefix, keys, strict=True))
        rows.append(
            {
                **row,
                "category_code": "all",
                "category": "all",
                "record_count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "evaluation_unit": "target_batch",
            }
        )
    for keys, group in frame.groupby(
        [*group_prefix, "category_code"], observed=True, sort=True
    ):
        row = dict(zip([*group_prefix, "category_code"], keys, strict=True))
        code = int(row["category_code"])
        rows.append(
            {
                **row,
                "category": CLASS_NAMES[code],
                "record_count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "evaluation_unit": "target_batch_class_cell",
            }
        )
    return pd.DataFrame(rows)


def _target_batch_class_summary(cells: pd.DataFrame) -> dict[str, float | int]:
    if cells.empty or set(cells["category_code"].astype(int)) != set(CLASS_CODES):
        raise ValueError("target batch-class metrics lack a frozen class")
    class_means = cells.groupby("category_code", observed=True)["accuracy"].mean()
    worst = cells.sort_values(
        ["accuracy", "record_count", "batch_code"], ascending=[True, True, True]
    ).iloc[0]
    return {
        "target_batch_class_macro_accuracy": float(class_means.mean()),
        "target_batch_class_recall_uav": float(class_means.loc[1]),
        "target_batch_class_recall_weather": float(class_means.loc[3]),
        "target_batch_class_cell_accuracy_p10": float(
            cells["accuracy"].quantile(0.10)
        ),
        "worst_target_batch_class_cell_accuracy": float(cells["accuracy"].min()),
        "worst_target_batch_class_cell_record_count": int(worst["record_count"]),
    }


def aggregate_transfer_metrics(
    predictions: pd.DataFrame, batch_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "transfer_id",
        "role",
        "analysis_scope",
        "source_bands",
        "model_id",
        "representation",
        "band_code",
        "band",
    ]
    for values, group in predictions.groupby(keys, observed=True, sort=True):
        identity = dict(zip(keys, values, strict=True))
        labels = group["category_code"].to_numpy(dtype=np.int64)
        predicted = group["predicted_category_code"].to_numpy(dtype=np.int64)
        probabilities = group[["probability_uav", "probability_weather"]].to_numpy(
            dtype=np.float64
        )
        selected = batch_metrics.loc[
            batch_metrics["transfer_id"].eq(identity["transfer_id"])
            & batch_metrics["analysis_scope"].eq(identity["analysis_scope"])
            & batch_metrics["model_id"].eq(identity["model_id"])
        ]
        batches = selected.loc[selected["evaluation_unit"].eq("target_batch")]
        cells = selected.loc[
            selected["evaluation_unit"].eq("target_batch_class_cell")
        ].copy()
        cells["category_code"] = cells["category_code"].astype(int)
        rows.append(
            {
                **identity,
                "target_record_count": int(len(group)),
                "target_batch_count": int(group["batch_code"].nunique()),
                **binary_classification_metrics(labels, predicted, probabilities),
                "target_batch_macro_accuracy": float(batches["accuracy"].mean()),
                "target_batch_accuracy_p10": float(batches["accuracy"].quantile(0.10)),
                "worst_target_batch_accuracy": float(batches["accuracy"].min()),
                **_target_batch_class_summary(cells),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["transfer_id", "role", "analysis_scope", "model_id", "band"]
    for values, group in predictions.groupby(keys, observed=True, sort=True):
        identity = dict(zip(keys, values, strict=True))
        for true_code in CLASS_CODES:
            true_rows = group.loc[group["category_code"].eq(true_code)]
            class_batches = true_rows["batch_code"].nunique()
            cell_sizes = true_rows.groupby("batch_code", observed=True)[
                "batch_code"
            ].transform("size")
            weights = 1.0 / (class_batches * cell_sizes.to_numpy(dtype=np.float64))
            for predicted_code in CLASS_CODES:
                selected = true_rows["predicted_category_code"].eq(predicted_code)
                rows.append(
                    {
                        **identity,
                        "true_category_code": true_code,
                        "true_category": CLASS_NAMES[true_code],
                        "predicted_category_code": predicted_code,
                        "predicted_category": CLASS_NAMES[predicted_code],
                        "confusion_type": "row_count",
                        "value": float(selected.sum()),
                    }
                )
                rows.append(
                    {
                        **identity,
                        "true_category_code": true_code,
                        "true_category": CLASS_NAMES[true_code],
                        "predicted_category_code": predicted_code,
                        "predicted_category": CLASS_NAMES[predicted_code],
                        "confusion_type": "target_batch_class_row_normalized",
                        "value": float(weights[selected.to_numpy()].sum()),
                    }
                )
    return pd.DataFrame(rows)


def _cell_accuracy_matrix(
    predictions: pd.DataFrame, batch_codes: np.ndarray
) -> np.ndarray:
    frame = predictions.copy()
    frame["correct"] = frame["category_code"].eq(
        frame["predicted_category_code"]
    )
    cells = frame.groupby(["batch_code", "category_code"], observed=True)[
        "correct"
    ].mean()
    matrix = np.full((len(batch_codes), len(CLASS_CODES)), np.nan, dtype=np.float64)
    batch_lookup = {int(code): index for index, code in enumerate(batch_codes)}
    class_lookup = {code: index for index, code in enumerate(CLASS_CODES)}
    for (batch, category), accuracy in cells.items():
        matrix[batch_lookup[int(batch)], class_lookup[int(category)]] = float(accuracy)
    return matrix


def _bootstrap_values(matrix: np.ndarray, draws: np.ndarray) -> np.ndarray:
    sampled = matrix[draws]
    valid_counts = np.sum(np.isfinite(sampled), axis=1)
    class_sums = np.nansum(sampled, axis=1)
    class_means = np.divide(
        class_sums,
        valid_counts,
        out=np.full_like(class_sums, np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    valid = np.isfinite(class_means).all(axis=1)
    values = np.full(len(draws), np.nan, dtype=np.float64)
    values[valid] = class_means[valid].mean(axis=1)
    return values


def bootstrap_seed(random_state: int, analysis_scope: str, transfer_id: str) -> int:
    material = f"{int(random_state)}|{analysis_scope}|{transfer_id}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def target_batch_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    random_state: int,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    if replicates <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap contract")
    rows: list[dict[str, Any]] = []
    for (transfer_id, analysis_scope), transfer_rows in predictions.groupby(
        ["transfer_id", "analysis_scope"], observed=True, sort=True
    ):
        derived_seed = bootstrap_seed(
            random_state, str(analysis_scope), str(transfer_id)
        )
        rng = np.random.default_rng(derived_seed)
        batch_codes = np.sort(transfer_rows["batch_code"].unique())
        draws = rng.integers(0, len(batch_codes), size=(replicates, len(batch_codes)))
        model_values: dict[str, np.ndarray] = {}
        model_matrices: dict[str, np.ndarray] = {}
        identity = transfer_rows.iloc[0]
        for model_id, model_rows in transfer_rows.groupby(
            "model_id", observed=True, sort=True
        ):
            matrix = _cell_accuracy_matrix(model_rows, batch_codes)
            values = _bootstrap_values(matrix, draws)
            model_values[str(model_id)] = values
            model_matrices[str(model_id)] = matrix
            valid = values[np.isfinite(values)]
            class_means = np.nanmean(matrix, axis=0)
            estimate = float(np.mean(class_means))
            alpha = (1.0 - confidence_level) / 2.0
            rows.append(
                {
                    "transfer_id": transfer_id,
                    "analysis_scope": analysis_scope,
                    "role": identity["role"],
                    "representation": identity["representation"],
                    "target_band": identity["band"],
                    "comparison": model_id,
                    "metric": "target_batch_class_macro_accuracy",
                    "estimate": estimate,
                    "ci_lower_95": float(
                        np.quantile(valid, alpha, method="linear")
                    ),
                    "ci_upper_95": float(
                        np.quantile(valid, 1.0 - alpha, method="linear")
                    ),
                    "requested_replicates": replicates,
                    "valid_replicates": int(len(valid)),
                    "discarded_replicates": int(replicates - len(valid)),
                    "bootstrap_seed": derived_seed,
                    "draw_count_per_replicate": len(batch_codes),
                    "resampling_unit": "representation_target_band_code_batch_code",
                    "duplicate_batch_draws_count_with_multiplicity": True,
                    "percentile_quantile_method": "linear",
                    "inference_scope": "conditional_on_each_fixed_source_fit",
                    "conditioning": "conditional_on_each_fixed_source_fit",
                    "identical_paired_draws": False,
                }
            )

        logistic = model_values.get("logistic_batch_balanced")
        dummy = model_values.get("dummy_prior")
        if logistic is None or dummy is None:
            raise ValueError(f"{transfer_id}: paired LR/dummy models are missing")
        paired = logistic - dummy
        valid = paired[np.isfinite(paired)]
        logistic_estimate = float(
            np.nanmean(model_matrices["logistic_batch_balanced"], axis=0).mean()
        )
        dummy_estimate = float(
            np.nanmean(model_matrices["dummy_prior"], axis=0).mean()
        )
        alpha = (1.0 - confidence_level) / 2.0
        rows.append(
            {
                "transfer_id": transfer_id,
                "analysis_scope": analysis_scope,
                "role": identity["role"],
                "representation": identity["representation"],
                "target_band": identity["band"],
                "comparison": "logistic_batch_balanced_minus_dummy_prior",
                "metric": "paired_target_batch_class_macro_accuracy_difference",
                "estimate": logistic_estimate - dummy_estimate,
                "ci_lower_95": float(np.quantile(valid, alpha, method="linear")),
                "ci_upper_95": float(
                    np.quantile(valid, 1.0 - alpha, method="linear")
                ),
                "requested_replicates": replicates,
                "valid_replicates": int(len(valid)),
                "discarded_replicates": int(replicates - len(valid)),
                "bootstrap_seed": derived_seed,
                "draw_count_per_replicate": len(batch_codes),
                "resampling_unit": "representation_target_band_code_batch_code",
                "duplicate_batch_draws_count_with_multiplicity": True,
                "percentile_quantile_method": "linear",
                "inference_scope": "conditional_on_each_fixed_source_fit",
                "conditioning": "conditional_on_each_fixed_source_fit",
                "identical_paired_draws": True,
            }
        )
    return pd.DataFrame(rows)


def raw_batch_overlap_row(
    transfer: dict[str, Any],
    source_items: list[PreparedBand],
    target_item: PreparedBand,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    overlap_contract: dict[str, Any] | None = None,
    validate_expected: bool = True,
) -> dict[str, Any]:
    full_source = set().union(*(set(item.full_batch_codes) for item in source_items))
    full_target = set(target_item.full_batch_codes)
    analysis_source = set(source_frame["batch_code"].astype(int))
    analysis_target = set(target_frame["batch_code"].astype(int))
    full_overlap = sorted(full_source & full_target)
    analysis_overlap = sorted(analysis_source & analysis_target)
    expected_analysis = int(
        transfer["raw_batch_disjoint_sensitivity"]["expected_overlap_code_count"]
    )
    if validate_expected and len(analysis_overlap) != expected_analysis:
        raise ValueError(
            f"{transfer['transfer_id']}: analysis-subset raw batch overlap changed "
            f"from {expected_analysis} to {len(analysis_overlap)}"
        )
    if validate_expected and overlap_contract is not None:
        if transfer["representation"] == "Narrow":
            full_expected = overlap_contract[
                "expected_narrow_full_release_pairwise_overlap_counts"
            ]
            analysis_expected = overlap_contract[
                "expected_narrow_shared_model_analysis_pairwise_overlap_counts"
            ]
        else:
            full_expected = overlap_contract["expected_hrrp_pairwise_overlap_counts"]
            analysis_expected = full_expected

        def expected_pair(mapping: dict[str, Any], left: str, right: str) -> int:
            for key in (f"{left}_{right}", f"{right}_{left}"):
                if key in mapping:
                    return int(mapping[key])
            raise ValueError(f"missing raw overlap contract for {left}/{right}")

        for item in source_items:
            source_analysis = set(
                source_frame.loc[
                    source_frame["source_id"].eq(item.source_id), "batch_code"
                ].astype(int)
            )
            observed_full = len(set(item.full_batch_codes) & full_target)
            observed_analysis = len(source_analysis & analysis_target)
            expected_full = expected_pair(full_expected, item.band, target_item.band)
            expected_filtered = expected_pair(
                analysis_expected, item.band, target_item.band
            )
            if observed_full != expected_full:
                raise ValueError(
                    f"{transfer['transfer_id']}: {item.band}/{target_item.band} "
                    f"full-release overlap changed from {expected_full} to {observed_full}"
                )
            if observed_analysis != expected_filtered:
                raise ValueError(
                    f"{transfer['transfer_id']}: {item.band}/{target_item.band} "
                    "analysis-subset overlap changed from "
                    f"{expected_filtered} to {observed_analysis}"
                )
    return {
        "transfer_id": transfer["transfer_id"],
        "role": transfer["role"],
        "representation": transfer["representation"],
        "source_bands": "+".join(transfer["source_bands"]),
        "target_band": transfer["target_band"],
        "full_release_source_batch_count": len(full_source),
        "full_release_target_batch_count": len(full_target),
        "full_release_overlap_code_count": len(full_overlap),
        "analysis_source_batch_count": len(analysis_source),
        "analysis_target_batch_count": len(analysis_target),
        "analysis_subset_overlap_code_count": len(analysis_overlap),
        "expected_analysis_subset_overlap_code_count": expected_analysis,
        "full_release_overlap_codes": ";".join(str(value) for value in full_overlap),
        "analysis_subset_overlap_codes": ";".join(
            str(value) for value in analysis_overlap
        ),
        "global_raw_batch_semantics_verified": False,
        "primary_group_key_band_qualified": True,
    }


def apply_raw_batch_disjoint_sensitivity(
    transfer: dict[str, Any],
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    minimum_batches_per_source_band_class: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_batches = set(target_frame["batch_code"].astype(int))
    filtered = source_frame.loc[
        ~source_frame["batch_code"].astype(int).isin(target_batches)
    ].copy()
    counts = (
        filtered[["band", "band_code", "category_code", "batch_code"]]
        .drop_duplicates()
        .groupby(["band", "band_code", "category_code"], observed=True)
        .size()
    )
    required_cells = {
        (band, int(band_code), category)
        for band, band_code in zip(
            transfer["source_bands"],
            [
                int(source_frame.loc[source_frame["band"].eq(band), "band_code"].iloc[0])
                for band in transfer["source_bands"]
            ],
            strict=True,
        )
        for category in CLASS_CODES
    }
    cell_counts = {
        (str(band), int(band_code), int(category)): int(count)
        for (band, band_code, category), count in counts.items()
    }
    minimum_observed = min(cell_counts.get(cell, 0) for cell in required_cells)
    computed_status = (
        "REPORTABLE"
        if minimum_observed >= minimum_batches_per_source_band_class
        else "NOT_IDENTIFIABLE"
    )
    declared_status = transfer["raw_batch_disjoint_sensitivity"]["status"]
    if declared_status != computed_status:
        raise ValueError(
            f"{transfer['transfer_id']}: declared disjoint status {declared_status} "
            f"does not match computed {computed_status}"
        )
    source_counts = {
        f"{band}_{CLASS_NAMES[category].lower()}_batch_count": cell_counts.get(
            (band, band_code, category), 0
        )
        for band, band_code, category in sorted(required_cells)
    }
    row = {
        "transfer_id": transfer["transfer_id"],
        "role": transfer["role"],
        "source_bands": "+".join(transfer["source_bands"]),
        "target_band": transfer["target_band"],
        "declared_status": declared_status,
        "computed_status": computed_status,
        "minimum_required_batches_per_source_band_class": minimum_batches_per_source_band_class,
        "minimum_observed_batches_per_source_band_class": minimum_observed,
        "source_record_count_before": int(len(source_frame)),
        "source_record_count_after": int(len(filtered)),
        "target_record_count_before": int(len(target_frame)),
        "target_record_count_after": int(len(target_frame)),
        "source_rows_removed": int(len(source_frame) - len(filtered)),
        "target_rows_removed": 0,
        "reason": transfer["raw_batch_disjoint_sensitivity"].get("reason", ""),
        **source_counts,
    }
    return filtered, row


def gate_decision_table(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    stopping_rule: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for transfer_id in stopping_rule["applies_to_transfer_ids"]:
        metric = aggregate.loc[
            aggregate["transfer_id"].eq(transfer_id)
            & aggregate["model_id"].eq(stopping_rule["model_id"])
            & aggregate["analysis_scope"].eq("band_qualified_primary")
        ]
        if len(metric) != 1:
            raise ValueError(f"missing stopping metric for {transfer_id}")
        metric_row = metric.iloc[0]
        interval = bootstrap.loc[
            bootstrap["transfer_id"].eq(transfer_id)
            & bootstrap["comparison"].eq(
                "logistic_batch_balanced_minus_dummy_prior"
            )
            & bootstrap["analysis_scope"].eq("band_qualified_primary")
        ]
        if len(interval) != 1:
            raise ValueError(f"missing paired stopping interval for {transfer_id}")
        ci_lower = float(interval.iloc[0]["ci_lower_95"])
        conditions = (
            (
                "target_batch_class_macro_accuracy",
                float(
                    stopping_rule[
                        "target_batch_class_macro_accuracy_strictly_greater_than"
                    ]
                ),
                float(metric_row["target_batch_class_macro_accuracy"]),
            ),
            (
                "target_batch_class_recall_uav",
                float(
                    stopping_rule[
                        "each_target_class_batch_recall_strictly_greater_than"
                    ]
                ),
                float(metric_row["target_batch_class_recall_uav"]),
            ),
            (
                "target_batch_class_recall_weather",
                float(
                    stopping_rule[
                        "each_target_class_batch_recall_strictly_greater_than"
                    ]
                ),
                float(metric_row["target_batch_class_recall_weather"]),
            ),
            (
                "paired_logistic_minus_dummy_ci_lower_95",
                float(
                    stopping_rule[
                        "paired_logistic_minus_dummy_ci_lower_strictly_greater_than"
                    ]
                ),
                ci_lower,
            ),
        )
        rows.extend(
            {
                "transfer_id": transfer_id,
                "metric": metric_name,
                "operator": ">",
                "threshold": threshold,
                "observed_value": observed,
                "passed": observed > threshold,
            }
            for metric_name, threshold, observed in conditions
        )
    return pd.DataFrame(rows)


def gate_decision_payload(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    stopping_rule: dict[str, Any],
) -> dict[str, Any]:
    conditions = gate_decision_table(aggregate, bootstrap, stopping_rule).to_dict(
        orient="records"
    )
    all_pass = bool(conditions) and all(bool(row["passed"]) for row in conditions)
    return {
        "status": "PASS_ENGINEERING_ONLY" if all_pass else "FAIL_STOP",
        "model_id": stopping_rule["model_id"],
        "uses_unrounded_values": True,
        "all_locked_primary_targets_pass": all_pass,
        "targets_consumed": True,
        "target_bands_consumed": list(
            stopping_rule["target_bands_consumed_by_this_run"]
        ),
        "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
        "conditions": conditions,
    }


def claim_boundaries_table(config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "claim": config["claim_contract"]["allowed_claim"],
            "allowed": True,
            "reason": "fixed source-only fit and released-band held-out evaluation",
        }
    ]
    rows.extend(
        {
            "claim": claim,
            "allowed": False,
            "reason": "outside the frozen dataset-internal band-held-out contract",
        }
        for claim in config["claim_contract"]["forbidden_claims"]
    )
    return pd.DataFrame(rows)


def transfer_coverage_table(sources: dict[str, PreparedBand]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_id, source in sorted(sources.items()):
        for category, group in source.frame.groupby(
            "category_code", observed=True, sort=True
        ):
            rows.append(
                {
                    "source_id": source_id,
                    "representation": source.representation,
                    "band_code": source.band_code,
                    "band": source.band,
                    "category_code": int(category),
                    "category": CLASS_NAMES[int(category)],
                    "record_count": int(len(group)),
                    "analysis_batch_count": int(group["batch_code"].nunique()),
                    "analysis_total_record_count": int(len(source.frame)),
                    "analysis_unique_batch_count": int(
                        source.frame["batch_code"].nunique()
                    ),
                    "full_release_batch_count": len(source.full_batch_codes),
                    "sha256": source.sha256,
                }
            )
    return pd.DataFrame(rows)


def feature_definitions_table(sources: dict[str, PreparedBand]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    schemas: dict[str, tuple[str, ...]] = {}
    for source in sources.values():
        existing = schemas.setdefault(source.representation, source.feature_names)
        if existing != source.feature_names:
            raise ValueError(
                f"{source.representation}: aggregate sources expose different features"
            )
    for representation, feature_names in sorted(schemas.items()):
        for feature in feature_names:
            rows.append(
                {
                    "feature_set_id": "lat_mricd_grouped_baseline_v1_exact",
                    "representation": representation,
                    "feature": feature,
                    "physical_frequency_unit": (
                        "cycles/sample"
                        if "cycles_per_sample" in feature
                        else "not_applicable"
                    ),
                    "per_record_normalized": True,
                    "metadata_as_feature": False,
                }
            )
    return pd.DataFrame(rows)


def disjoint_sensitivity_table(
    status_rows: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [
        column
        for column in metrics.columns
        if column
        not in {
            "transfer_id",
            "role",
            "analysis_scope",
            "source_bands",
            "representation",
            "band_code",
            "band",
        }
    ]
    for status in status_rows.to_dict(orient="records"):
        selected = metrics.loc[metrics["transfer_id"].eq(status["transfer_id"])]
        if status["computed_status"] == "REPORTABLE":
            if selected.empty:
                raise ValueError(
                    f"{status['transfer_id']}: reportable sensitivity lacks metrics"
                )
            for metric in selected.to_dict(orient="records"):
                rows.append(
                    {
                        **status,
                        "analysis_scope": "raw_batch_code_disjoint_sensitivity",
                        **{column: metric[column] for column in metric_columns},
                    }
                )
        else:
            if not selected.empty:
                raise ValueError(
                    f"{status['transfer_id']}: NOT_IDENTIFIABLE sensitivity has metrics"
                )
            rows.append(
                {
                    **status,
                    "analysis_scope": "raw_batch_code_disjoint_sensitivity",
                    **{column: math.nan for column in metric_columns},
                }
            )
    return pd.DataFrame(rows)


def _stable_sort(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    preferred = [
        "transfer_id",
        "analysis_scope",
        "model_id",
        "source_id",
        "band_code",
        "band",
        "batch_code",
        "category_code",
        "comparison",
        "claim",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True) if columns else frame.reset_index(drop=True)


def make_report(summary: dict[str, Any], aggregate: pd.DataFrame) -> str:
    primary = aggregate.loc[aggregate["role"].eq("locked_primary")]
    rows = "\n".join(
        "| {transfer_id} | {model_id} | {target_batch_class_macro_accuracy:.4f} | "
        "{target_batch_class_recall_uav:.4f} | {target_batch_class_recall_weather:.4f} |".format(
            **record
        )
        for record in primary.to_dict(orient="records")
    )
    return f"""# LAT-MRICD Cross-Band Transfer V1

Status: `{summary['status']}`
Class contract: UAV versus weather
Fit scope: released source bands only
Target bands: S and Ku

## Locked primary results

| Transfer | Model | Target batch-class macro accuracy | UAV batch recall | Weather batch recall |
|---|---|---:|---:|---:|
{rows}

## Evaluation contract

- The StandardScaler, model, weighted source prior and fixed argmax decision are fit from source
  rows only. Calibration and threshold tuning are disabled.
- Target rows are used once for final aggregate evaluation. Passing the stopping gate does not
  authorize reusing either target for a new confirmatory model comparison.
- Bootstrap intervals resample complete target raw batch codes and are conditional on the fixed
  source fit. Logistic-minus-dummy intervals use identical target-batch draws.
- Raw-code-disjoint sensitivity removes overlapping batch codes from source rows only; target
  rows remain unchanged. It is not primary evidence.

## Claim boundary

The only permitted interpretation is dataset-internal band-held-out UAV/weather performance
using fixed interpretable features. The result does not establish physical-frequency
invariance, physical micro-Doppler, same-event fusion, unseen-model or independent-scene
generalization, H/V polarimetry, balloon recognition, causal deployment, or Tian reproduction.
"""


def resolve_consumption_record_path(config: dict[str, Any]) -> Path:
    relative_path = config["acceptance_contract"]["formal_run_consumption_record"]
    path = resolve_path(relative_path)
    frozen = (
        PROJECT_ROOT
        / "results/final_evidence/lat_mricd_cross_band_transfer_v1.run_consumed.json"
    ).resolve()
    if path != frozen:
        raise ValueError("formal-run consumption record path differs from frozen config")
    return path


def resolve_formal_output_path(config: dict[str, Any]) -> Path:
    relative_path = config["output_dir"]
    if relative_path != FORMAL_OUTPUT_RELATIVE_PATH:
        raise ValueError("formal output directory differs from frozen config")
    path = resolve_path(relative_path)
    if path != FORMAL_OUTPUT_DIR:
        raise ValueError("formal output directory differs from frozen config")
    return path


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_formal_output_location(output_dir: Path, dataset_root: Path) -> None:
    output_dir = output_dir.resolve()
    dataset_root = dataset_root.resolve()
    git_path = (PROJECT_ROOT / ".git").resolve()
    if output_dir == PROJECT_ROOT or output_dir in PROJECT_ROOT.parents:
        raise ValueError("formal output directory cannot be the project root or its ancestor")
    if _paths_overlap(output_dir, dataset_root):
        raise ValueError("formal output directory must not overlap the raw dataset root")
    if _paths_overlap(output_dir, git_path):
        raise ValueError("formal output directory must not overlap Git metadata")


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_consumption_record(
    path: Path, payload: dict[str, Any], *, exclusive: bool
) -> None:
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent_directory(path)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent_directory(path)
    finally:
        if os.path.lexists(temporary_path):
            temporary_path.unlink()


def _consumption_record_payload(
    *,
    status: str,
    config: dict[str, Any],
    pre_result_commit: str,
    config_sha256: str,
    implementation_sha256: str,
    summary_sha256: str | None,
    failure_type: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "experiment_id": config["experiment_id"],
        "pre_result_commit": pre_result_commit,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "target_bands_consumed": list(
            config["stopping_rule"]["target_bands_consumed_by_this_run"]
        ),
        "summary_sha256": summary_sha256,
        "sealed_run_consumption_enforced": True,
        "formal_output_overwrite_allowed": False,
        "record_created_before_target_load": True,
        "persists_on_failure": True,
    }
    if failure_type is not None:
        payload["failure_type"] = failure_type
    return payload


def _sealed_run_entrypoint(
    execute: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def entrypoint(
        *,
        config_path: Path = DEFAULT_CONFIG,
        dataset_root: Path | None = None,
    ) -> dict[str, Any]:
        resolved_config_path = resolve_path(config_path)
        config = load_config(resolved_config_path)
        resolved_dataset_root = resolve_path(dataset_root or config["dataset_root"])
        resolved_output_dir = resolve_formal_output_path(config)
        validate_formal_output_location(resolved_output_dir, resolved_dataset_root)
        if os.path.lexists(resolved_output_dir):
            raise FileExistsError(
                f"formal output directory must not already exist: {resolved_output_dir}"
            )

        pre_result_commit = validate_pre_result_repository_state(
            resolved_config_path, config
        )
        consumption_path = resolve_consumption_record_path(config)
        if os.path.lexists(consumption_path):
            raise FileExistsError("formal-run consumption record already exists")
        config_sha256 = sha256_file(resolved_config_path)
        implementation_sha256 = sha256_file(Path(__file__).resolve())
        reserved = _consumption_record_payload(
            status="RESERVED",
            config=config,
            pre_result_commit=pre_result_commit,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            summary_sha256=None,
        )
        _write_consumption_record(consumption_path, reserved, exclusive=True)

        try:
            summary = execute(
                config_path=resolved_config_path,
                config=config,
                dataset_root=resolved_dataset_root,
                output_dir=resolved_output_dir,
                implementation_commit=pre_result_commit,
            )
            summary_sha256 = sha256_file(resolved_output_dir / "summary.json")
            completed = _consumption_record_payload(
                status="COMPLETED_AND_TARGETS_CONSUMED",
                config=config,
                pre_result_commit=pre_result_commit,
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha256,
                summary_sha256=summary_sha256,
            )
            _write_consumption_record(consumption_path, completed, exclusive=False)
            return summary
        except BaseException as exc:
            summary_path = resolved_output_dir / "summary.json"
            failed = _consumption_record_payload(
                status="FAILED_OR_INTERRUPTED_RUN_CONSUMED",
                config=config,
                pre_result_commit=pre_result_commit,
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha256,
                summary_sha256=(
                    sha256_file(summary_path) if summary_path.is_file() else None
                ),
                failure_type=type(exc).__name__,
            )
            try:
                _write_consumption_record(consumption_path, failed, exclusive=False)
            except BaseException:
                # The exclusive RESERVED marker still prevents another formal run.
                pass
            raise

    return entrypoint


@_sealed_run_entrypoint
def run_cross_band_transfer(
    *,
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    sources = load_aggregate_sources(config, dataset_root)
    prediction_frames: list[pd.DataFrame] = []
    weight_audits: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, Any]] = []
    disjoint_rows: list[dict[str, Any]] = []
    disjoint_prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    minimum_batches = int(
        config["acceptance_contract"][
            "minimum_batch_count_per_source_band_and_class_for_reportable_disjoint_sensitivity"
        ]
    )

    for transfer_index, transfer in enumerate(config["transfers"]):
        source_frame, target_frame, feature_names = assemble_transfer_frames(
            transfer, sources
        )
        source_items = [sources[source_id] for source_id in transfer["source_ids"]]
        target_item = sources[transfer["target_source_id"]]
        overlap_rows.append(
            raw_batch_overlap_row(
                transfer,
                source_items,
                target_item,
                source_frame,
                target_frame,
                overlap_contract=config["raw_batch_overlap_audit"],
            )
        )
        fit_seed = int(config["random_state"]) + transfer_index * 100
        predictions, weight_audit = evaluate_models(
            transfer,
            source_frame,
            target_frame,
            feature_names,
            config["models"],
            random_state=fit_seed,
            analysis_scope="band_qualified_primary",
            importance_rows=importance_rows,
            fit_rows=fit_rows,
        )
        prediction_frames.append(predictions)
        weight_audits.append(weight_audit)

        filtered_source, disjoint_row = apply_raw_batch_disjoint_sensitivity(
            transfer,
            source_frame,
            target_frame,
            minimum_batches_per_source_band_class=minimum_batches,
        )
        disjoint_rows.append(disjoint_row)
        if disjoint_row["computed_status"] == "REPORTABLE":
            sensitivity_predictions, sensitivity_weights = evaluate_models(
                transfer,
                filtered_source,
                target_frame,
                feature_names,
                config["models"],
                random_state=fit_seed,
                analysis_scope="raw_batch_code_disjoint_sensitivity",
                importance_rows=importance_rows,
                fit_rows=fit_rows,
            )
            disjoint_prediction_frames.append(sensitivity_predictions)
            weight_audits.append(sensitivity_weights)

    expected_fits = expected_model_fit_count(config)
    if len(fit_rows) != expected_fits:
        raise AssertionError(
            f"model fit manifest has {len(fit_rows)} fits; expected {expected_fits}"
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    target_metrics = target_batch_metric_rows(predictions)
    aggregate = aggregate_transfer_metrics(predictions, target_metrics)
    confusion = confusion_matrix_rows(predictions)
    if disjoint_prediction_frames:
        disjoint_predictions = pd.concat(disjoint_prediction_frames, ignore_index=True)
        disjoint_batch_metrics = target_batch_metric_rows(disjoint_predictions)
        disjoint_metrics = aggregate_transfer_metrics(
            disjoint_predictions, disjoint_batch_metrics
        )
    else:
        disjoint_predictions = pd.DataFrame(columns=predictions.columns)
        disjoint_metrics = pd.DataFrame(columns=aggregate.columns)

    bootstrap_predictions = pd.concat(
        [predictions, disjoint_predictions], ignore_index=True
    )
    bootstrap = target_batch_bootstrap_intervals(
        bootstrap_predictions,
        replicates=int(config["bootstrap_replicates"]),
        random_state=int(config["random_state"]),
        confidence_level=float(config["bootstrap_contract"]["confidence_level"]),
    )
    expected_intervals = expected_bootstrap_interval_count(config)
    if len(bootstrap) != expected_intervals:
        raise AssertionError(
            f"bootstrap table has {len(bootstrap)} rows; expected {expected_intervals}"
        )
    if bootstrap["valid_replicates"].min() < int(
        config["bootstrap_contract"]["minimum_valid_replicates"]
    ):
        raise ValueError("too few valid target-batch bootstrap replicates")

    status_frame = pd.DataFrame(disjoint_rows)
    gate = gate_decision_payload(aggregate, bootstrap, config["stopping_rule"])
    claims = claim_boundaries_table(config)
    tables = {
        "transfer_coverage.csv": transfer_coverage_table(sources),
        "raw_batch_overlap_audit.csv": pd.DataFrame(overlap_rows),
        "training_weight_audit.csv": pd.concat(weight_audits, ignore_index=True),
        "aggregate_metrics.csv": aggregate,
        "target_batch_class_metrics.csv": target_metrics,
        "bootstrap_intervals.csv": bootstrap,
        "confusion_matrices.csv": confusion,
        "feature_definitions.csv": feature_definitions_table(sources),
        "feature_importance.csv": pd.DataFrame(importance_rows),
        "disjoint_sensitivity.csv": disjoint_sensitivity_table(
            status_frame, disjoint_metrics
        ),
        "claim_boundaries.csv": claims,
    }
    if tuple(tables) != OUTPUT_CSV_FILES:
        raise AssertionError("cross-band output table contract is incomplete")
    if any("prediction" in name or "oof" in name for name in OUTPUT_FILES):
        raise AssertionError("sample-level predictions must never be persisted")

    implementation_sha256 = sha256_file(Path(__file__).resolve())
    config_sha256 = sha256_file(config_path)
    feature_implementation_sha256 = sha256_file(
        PROJECT_ROOT / config["feature_contract"]["implementation"]
    )
    source_files = [
        {
            "source_id": source.source_id,
            "sha256": source.sha256,
            "analysis_record_count": int(len(source.frame)),
            "full_release_batch_count": len(source.full_batch_codes),
        }
        for source in sorted(sources.values(), key=lambda item: item.source_id)
    ]
    model_fit_manifest = {
        "status": "COMPLETE_SOURCE_ONLY_MODEL_FIT_MANIFEST",
        "experiment_id": config["experiment_id"],
        "implementation_commit": implementation_commit,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "feature_implementation_sha256": feature_implementation_sha256,
        "fit_scope": "source_bands_only",
        "scaler_fit_scope": "source_bands_only",
        "probability_calibration": "none",
        "decision_rule": config["training_contract"]["prediction_decision"],
        "argmax_tie_break": "lowest_class_code",
        "dummy_prior_fit_scope": "source_bands_only",
        "dummy_prior_uses_same_sample_weights_as_learned_models": True,
        "target_rows_used_for_fit": False,
        "target_labels_used_for_fit_threshold_calibration_or_model_selection": False,
        "target_statistics_used_for_scaling_or_calibration": False,
        "threshold_tuning_performed": False,
        "hyperparameter_search_performed": False,
        "test_driven_model_selection_performed": False,
        "single_sealed_run": True,
        "sealed_run_consumption_enforced": True,
        "formal_output_overwrite_allowed": False,
        "sample_weight_hierarchy": config["training_contract"][
            "sample_weight_hierarchy"
        ],
        "source_files": source_files,
        "models": config["models"],
        "model_ids": [model["model_id"] for model in config["models"]],
        "transfer_ids": [
            transfer["transfer_id"] for transfer in config["transfers"]
        ],
        "analysis_scopes": [
            "band_qualified_primary",
            "raw_batch_code_disjoint_sensitivity",
        ],
        "fit_count": len(fit_rows),
        "transfers": fit_rows,
    }
    summary = {
        "status": "COMPLETE_PREREGISTERED_CROSS_BAND_TRANSFER",
        "experiment_id": config["experiment_id"],
        "dataset": config.get("dataset", "LAT-MRICD-1.0"),
        "implementation_commit": implementation_commit,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "feature_implementation_sha256": feature_implementation_sha256,
        "random_state": int(config["random_state"]),
        "transfer_count": len(config["transfers"]),
        "model_count": len(config["models"]),
        "model_fit_count": len(fit_rows),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "source_files": source_files,
        "target_rows_used_for_fit": False,
        "target_labels_used_for_fit_threshold_calibration_or_model_selection": False,
        "target_statistics_used_for_scaling_or_calibration": False,
        "probability_calibration_performed": False,
        "threshold_tuning_performed": False,
        "decision_rule": "ascending_class_argmax_with_lowest_code_tie_break",
        "dummy_prior_fit_scope": "weighted_source_rows_only",
        "hyperparameter_search_performed": False,
        "test_driven_model_selection_performed": False,
        "single_sealed_run": True,
        "sealed_run_consumption_enforced": True,
        "formal_output_overwrite_allowed": False,
        "sample_predictions_saved": False,
        "raw_data_in_output": False,
        "model_checkpoints_saved": False,
        "physical_frequency_hz_reported": False,
        "primary_gate_passed": gate["all_locked_primary_targets_pass"],
        "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
        "same_target_confirmatory_reuse_allowed": False,
        "bootstrap_inference_scope": "conditional_on_each_fixed_source_fit",
        "claim_scope": "dataset-internal released-band-held-out UAV/weather transfer",
        "output_files": list(OUTPUT_FILES),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        _stable_sort(frame).to_csv(
            output_dir / name,
            index=False,
            encoding="utf-8-sig",
            float_format="%.17g",
        )
    model_fit_manifest["training_weight_audit_sha256"] = sha256_file(
        output_dir / "training_weight_audit.csv"
    )
    (output_dir / "model_fit_manifest.json").write_text(
        json.dumps(
            model_fit_manifest, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "gate_decision.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        make_report(summary, aggregate), encoding="utf-8"
    )
    summary["output_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in OUTPUT_FILES
        if name != "summary.json"
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observed_outputs = {path.name for path in output_dir.iterdir()}
    if observed_outputs != set(OUTPUT_FILES):
        raise AssertionError(
            f"unexpected cross-band outputs: {sorted(observed_outputs ^ set(OUTPUT_FILES))}"
        )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_cross_band_transfer(
        config_path=args.config,
        dataset_root=args.dataset_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
