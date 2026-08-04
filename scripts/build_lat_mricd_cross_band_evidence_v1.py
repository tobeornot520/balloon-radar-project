#!/usr/bin/env python3
"""Validate and freeze sanitized LAT-MRICD cross-band aggregate evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "results/experiments/lat_mricd_cross_band_transfer_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/final_evidence/lat_mricd_cross_band_transfer_v1"
)
CONFIG_PATH = PROJECT_ROOT / "configs/lat_mricd_cross_band_transfer_v1.json"
IMPLEMENTATION_PATH = (
    PROJECT_ROOT / "scripts/run_lat_mricd_cross_band_transfer_v1.py"
)

SOURCE_ROOT_FILES = (
    "REPORT.md",
    "summary.json",
    "model_fit_manifest.json",
    "gate_decision.json",
)
SOURCE_TABLE_FILES = (
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
SOURCE_FILES = SOURCE_ROOT_FILES + SOURCE_TABLE_FILES
PUBLISHED_TABLE_FILES = SOURCE_TABLE_FILES
PUBLISHED_FILES = SOURCE_ROOT_FILES + tuple(
    f"tables/{name}" for name in PUBLISHED_TABLE_FILES
)
EXCLUDED_ARTIFACT_POLICIES = {
    "sample_level_predictions": "not produced or published",
    "raw_data": "not produced or published",
    "raw_or_absolute_paths": "forbidden in every published field",
    "per_sample_weights": "not produced or published",
    "model_checkpoints": "not produced or published",
}

EXPECTED_STATUS = "COMPLETE_PREREGISTERED_CROSS_BAND_TRANSFER"
EXPECTED_MODEL_IDS = {
    "dummy_prior",
    "logistic_batch_balanced",
    "random_forest_batch_balanced",
}
PRIMARY_MODEL_ID = "logistic_batch_balanced"
PRIMARY_ANALYSIS_SCOPE = "band_qualified_primary"
DISJOINT_ANALYSIS_SCOPE = "raw_batch_code_disjoint_sensitivity"
PAIR_COMPARISON = "logistic_batch_balanced_minus_dummy_prior"
PAIR_METRIC = "paired_target_batch_class_macro_accuracy_difference"
PRIMARY_METRIC = "target_batch_class_macro_accuracy"
CLASS_CODES = {1, 3}

EXPECTED_AGGREGATE_SOURCES = {
    "hrrp_x": (
        "HRRP/X\u6ce2\u6bb5/data_hrrp_X.mat",
        "276029f334e24abf9e54860d3f58e99557ffd47398c602e49d4139f3a25ae267",
        "HRRP",
        2,
        "X",
    ),
    "hrrp_ku": (
        "HRRP/Ku\u6ce2\u6bb5/data_hrrp_Ku.mat",
        "985d679510b272e90180a77cdfa64c354909a45bc6b20cb00d176b27fbe44ed4",
        "HRRP",
        3,
        "Ku",
    ),
    "narrow_s": (
        "Narrow/S\u6ce2\u6bb5/data_narrow_S.mat",
        "e584b3ed8c0117265cfebe61bd289aa91c1d917aaa504e7cb21118615b5db328",
        "Narrow",
        1,
        "S",
    ),
    "narrow_x": (
        "Narrow/X\u6ce2\u6bb5/data_narrow_X.mat",
        "da8ea23032929a60de67fad7a46b7616f068885efc02042a22b747da9f6e24c4",
        "Narrow",
        2,
        "X",
    ),
    "narrow_ku": (
        "Narrow/Ku\u6ce2\u6bb5/data_narrow_Ku.mat",
        "92246be376fe93b00bb6bb56d64882916297f88e8ac20839daced3c9a2c8d926",
        "Narrow",
        3,
        "Ku",
    ),
}

FORBIDDEN_FIELD_NAMES = {
    "absolute_path",
    "checkpoint",
    "checkpoint_path",
    "dataset_root",
    "file_path",
    "normalized_weight",
    "oof_prediction",
    "oof_predictions",
    "output_dir",
    "raw_path",
    "record_id",
    "relative_path",
    "row_index",
    "sample_id",
    "sample_index",
    "sample_weight",
    "source_dir",
    "source_row_index",
    "target_prediction",
    "target_predictions",
    "training_weight",
    "training_weights",
    "weight",
    "weights",
}
SENSITIVE_TEXT_MARKERS = (
    "/home/",
    "/Users/",
    "C:\\Users\\",
    "tobeornot8259748",
)
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=,:])(?:/[A-Za-z0-9_.-]+){2,}(?:$|[\s\"',;])"
)
RELATIVE_RAW_ARTIFACT_PATTERN = re.compile(
    r"(?:^|[\s\"'=,:])(?:[^/\\\s\"']+[/\\])*[^/\\\s\"']+\."
    r"(?:mat|npy|bin|ckpt|pt|pth)(?:$|[\s\"',;}\]])",
    re.IGNORECASE,
)
PACKED_SENSITIVE_PATTERN = re.compile(
    r"(?:^|[\"'{\[,]\s*)\"?(?:prediction|target_prediction|sample_id|"
    r"row_index|sample_weight)\"?\s*:",
    re.IGNORECASE,
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


CommitValidator = Callable[[str, Mapping[Path, str]], bool]
ConsumptionRecordProvider = Callable[[Path], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the sealed cross-band run and publish aggregate-only evidence "
            "to the frozen destination."
        )
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_seed(random_state: int, analysis_scope: str, transfer_id: str) -> int:
    material = f"{int(random_state)}|{analysis_scope}|{transfer_id}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_commit_bindings(commit: str, bindings: Mapping[Path, str]) -> bool:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        return False
    for path, expected_sha256 in bindings.items():
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return False
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0 or sha256_bytes(result.stdout) != expected_sha256:
            return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.columns.duplicated().any():
        raise ValueError(f"{path.name} contains duplicate columns")
    return frame


def _load_consumption_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("formal-run consumption record must not be a symlink")
    return _load_json(path)


def _require_exact_columns(
    frame: pd.DataFrame, expected: set[str], name: str
) -> None:
    actual = set(map(str, frame.columns))
    if actual != expected:
        raise ValueError(
            f"{name} column schema changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(map(str, mapping))
    if actual != expected:
        raise ValueError(
            f"{name} field schema changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_equal(mapping: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    for field, value in expected.items():
        if mapping.get(field) != value:
            raise ValueError(
                f"unexpected {name} {field}: {mapping.get(field)!r}; expected {value!r}"
            )


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{field} must be boolean, got {value!r}")


def _same_float(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
        left_value, right_value, rel_tol=1e-12, abs_tol=atol
    )


def _ensure_finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf in required numeric fields")


def _walk_json(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    nodes: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            nodes.append((child, item))
            nodes.extend(_walk_json(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            nodes.append((child, item))
            nodes.extend(_walk_json(item, child))
    return nodes


def _contains_forbidden_field_name(field: str) -> bool:
    tokens = {
        token.strip().lower()
        for token in re.split(r"[.\[\]]+", field)
        if token.strip()
    }
    return bool(tokens & FORBIDDEN_FIELD_NAMES)


def _contains_sensitive_text(value: str) -> bool:
    if any(marker in value for marker in SENSITIVE_TEXT_MARKERS):
        return True
    if re.search(r"[A-Za-z]:\\", value):
        return True
    return bool(
        ABSOLUTE_PATH_PATTERN.search(value)
        or RELATIVE_RAW_ARTIFACT_PATTERN.search(value)
        or PACKED_SENSITIVE_PATTERN.search(value)
    )


def audit_json_payload(payload: dict[str, Any], *, name: str) -> None:
    for field, value in _walk_json(payload):
        if _contains_forbidden_field_name(field):
            raise ValueError(f"{name} contains forbidden field {field!r}")
        if isinstance(value, str) and _contains_sensitive_text(value):
            raise ValueError(f"{name} contains an absolute or sensitive path in {field!r}")


def audit_table_payload(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = {
        column
        for column in frame.columns
        if _contains_forbidden_field_name(str(column))
    }
    if forbidden:
        raise ValueError(f"{name} contains forbidden fields: {sorted(forbidden)}")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        for value in frame[column].dropna().astype(str):
            if _contains_sensitive_text(value):
                raise ValueError(
                    f"{name} contains an absolute or sensitive path in column {column!r}"
                )


def audit_text_payload(text: str, *, name: str) -> None:
    if _contains_sensitive_text(text):
        raise ValueError(f"{name} contains an absolute or sensitive path")


def validate_config_contract(config: dict[str, Any]) -> None:
    _require_equal(
        config,
        {
            "schema_version": 1,
            "experiment_id": "lat_mricd_cross_band_transfer_v1",
            "status": "PREREGISTERED_NOT_RUN",
            "group_key": ["representation", "band_code", "batch_code"],
            "bootstrap_replicates": 2000,
        },
        "config",
    )
    sources = config.get("aggregate_sources")
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("config must bind exactly five aggregate sources")
    source_ids: set[str] = set()
    for source in sources:
        source_id = str(source.get("source_id", ""))
        digest = str(source.get("expected_sha256", ""))
        if not source_id or source_id in source_ids or not HEX_SHA256.fullmatch(digest):
            raise ValueError("aggregate source ids and hashes must be unique and valid")
        source_ids.add(source_id)
        if source_id not in EXPECTED_AGGREGATE_SOURCES:
            raise ValueError(f"unexpected aggregate source {source_id!r}")
        expected = EXPECTED_AGGREGATE_SOURCES[source_id]
        observed = (
            source.get("relative_path"),
            digest,
            source.get("representation"),
            int(source.get("band_code", -1)),
            source.get("band"),
        )
        if observed != expected:
            raise ValueError(f"{source_id}: aggregate source contract changed")
        coverage = source.get("expected_analysis_coverage")
        if not isinstance(coverage, dict):
            raise ValueError(f"{source_id}: expected analysis coverage is missing")
        classes = coverage.get("classes")
        if not isinstance(classes, dict) or set(classes) != {"1", "3"}:
            raise ValueError(f"{source_id}: expected class coverage is incomplete")
        class_records = 0
        for class_code in ("1", "3"):
            class_coverage = classes[class_code]
            if not isinstance(class_coverage, dict):
                raise ValueError(f"{source_id}: invalid class coverage")
            record_count = int(class_coverage.get("record_count", 0))
            batch_count = int(class_coverage.get("batch_count", 0))
            if record_count <= 0 or batch_count <= 0:
                raise ValueError(f"{source_id}: class coverage must be positive")
            if batch_count > int(coverage.get("unique_batch_count", 0)):
                raise ValueError(f"{source_id}: class batch coverage is inconsistent")
            class_records += record_count
        total_records = int(coverage.get("record_count", 0))
        unique_batches = int(coverage.get("unique_batch_count", 0))
        if (
            total_records != class_records
            or total_records > int(source["expected_record_count"])
            or unique_batches <= 0
        ):
            raise ValueError(f"{source_id}: expected analysis totals are inconsistent")
    if source_ids != set(EXPECTED_AGGREGATE_SOURCES):
        raise ValueError("aggregate source whitelist is incomplete")

    if config.get("class_contract", {}).get("class_codes") != [1, 3]:
        raise ValueError("cross-band evidence must retain the frozen binary classes")
    training = config.get("training_contract", {})
    _require_equal(
        training,
        {
            "fit_scope": "source_bands_only",
            "target_rows_used_for_fit": False,
            "target_labels_used_for_fit_threshold_calibration_or_model_selection": False,
            "target_statistics_used_for_scaling_or_calibration": False,
            "scaler_fit_scope": "source_bands_only",
            "probability_calibration": "none",
            "dummy_prior_fit_scope": "source_bands_only",
            "dummy_prior_uses_same_sample_weights_as_learned_models": True,
            "threshold_tuning_allowed": False,
            "hyperparameter_search_allowed": False,
            "test_driven_model_selection_allowed": False,
            "single_sealed_run_before_result_inspection": True,
        },
        "training contract",
    )
    feature = config.get("feature_contract", {})
    _require_equal(
        feature,
        {
            "feature_selection_allowed": False,
            "pca_allowed": False,
            "band_code_as_feature_allowed": False,
            "category_model_or_batch_metadata_as_feature_allowed": False,
            "target_band_statistics_for_feature_processing_allowed": False,
        },
        "feature contract",
    )
    _require_equal(
        feature,
        {
            "feature_set_id": "lat_mricd_grouped_baseline_v1_exact",
            "implementation": "scripts/run_lat_mricd_grouped_baseline_v1.py",
        },
        "feature contract",
    )
    if not HEX_SHA256.fullmatch(str(feature.get("implementation_sha256", ""))):
        raise ValueError("feature implementation SHA256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(feature.get("frozen_baseline_commit", ""))):
        raise ValueError("frozen feature implementation commit is invalid")
    schemas = feature.get("feature_schemas")
    if not isinstance(schemas, dict) or set(schemas) != {"Narrow", "HRRP"}:
        raise ValueError("Narrow and HRRP feature schemas must both be frozen")
    for representation, schema in schemas.items():
        names = schema.get("feature_names") if isinstance(schema, dict) else None
        if (
            not isinstance(names, list)
            or not names
            or len(names) != len(set(map(str, names)))
            or int(schema.get("feature_count", -1)) != len(names)
        ):
            raise ValueError(f"invalid frozen {representation} feature schema")
    bootstrap = config.get("bootstrap_contract", {})
    _require_equal(
        bootstrap,
        {
            "interval_method": "percentile",
            "confidence_level": 0.95,
            "percentile_quantile_method": "linear",
            "missing_class_replicate": "discard_and_report",
            "analysis_scopes": [PRIMARY_ANALYSIS_SCOPE, DISJOINT_ANALYSIS_SCOPE],
            "raw_batch_code_disjoint_sensitivity_included_only_when_reportable": True,
            "resampling_unit": [
                "representation",
                "target_band_code",
                "batch_code",
            ],
            "draw_count_per_replicate": "number_of_unique_target_batch_codes_in_transfer_scope",
            "sample_with_replacement": True,
            "duplicate_batch_draws_count_with_multiplicity": True,
            "seed_derivation": (
                "uint32_from_first_8_hex_sha256_of_random_state_pipe_"
                "analysis_scope_pipe_transfer_id"
            ),
            "class_macro_hierarchy_after_resampling": (
                "row_accuracy_within_drawn_batch_class_then_equal_drawn_batches_"
                "within_class_then_equal_classes"
            ),
            "required_interval_estimands": [
                "target_batch_class_macro_accuracy_for_each_fixed_model",
                "logistic_batch_balanced_minus_dummy_prior_paired_target_batch_"
                "class_macro_accuracy",
            ],
            "paired_comparison": PAIR_COMPARISON,
            "paired_resamples_must_use_identical_target_batches": True,
            "resample_complete_target_raw_batch_code_clusters": True,
            "confidence_interval_conditioning": "conditional_on_each_fixed_source_fit",
            "minimum_valid_replicates": 1900,
        },
        "bootstrap contract",
    )
    acceptance = config.get("acceptance_contract", {})
    _require_equal(
        acceptance,
        {
            "protocol_frozen_before_target_band_evaluation": True,
            "target_band_performance_inspected_before_preregistration": False,
            "formal_run_requires_clean_git_worktree": True,
            "formal_run_records_current_pre_result_commit": True,
            "pre_result_commit_must_contain_config_runner_tests_and_frozen_"
            "feature_implementation": True,
            "formal_run_consumption_record": (
                "results/final_evidence/"
                "lat_mricd_cross_band_transfer_v1.run_consumed.json"
            ),
            "consumption_record_must_be_absent_before_formal_run": True,
            "formal_output_overwrite_allowed": False,
            "record_before_target_load": True,
            "persist_on_failure": True,
            "aggregate_file_count": 5,
            "aggregate_files_only": True,
            "detail_files_allowed": False,
            "aggregate_and_detail_files_loaded_together_allowed": False,
            "target_x_allowed": False,
            "formal_target_bands": ["S", "Ku"],
            "minimum_batch_count_per_source_band_and_class_for_reportable_disjoint_sensitivity": 3,
        },
        "acceptance contract",
    )
    output = config.get("output_contract", {})
    _require_equal(
        output,
        {
            "file_count": len(SOURCE_FILES),
            "files": list(SOURCE_TABLE_FILES)
            + ["model_fit_manifest.json", "gate_decision.json", "summary.json", "REPORT.md"],
            "sample_level_predictions_allowed": False,
            "oof_predictions_allowed": False,
            "raw_data_allowed": False,
            "per_sample_weights_allowed": False,
            "model_checkpoints_allowed": False,
            "raw_or_absolute_paths_allowed": False,
        },
        "output contract",
    )
    stopping = config.get("stopping_rule", {})
    _require_equal(
        stopping,
        {
            "model_id": PRIMARY_MODEL_ID,
            "all_locked_primary_targets_must_pass": True,
            "uses_unrounded_values": True,
            "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
            "reserved_independent_holdout_exists": False,
            "hrrp_exploratory_result_can_open_deep_model_gate": False,
            "target_bands_consumed_by_this_run": ["S", "Ku"],
        },
        "stopping rule",
    )


def _configured_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(source["source_id"]): source for source in config["aggregate_sources"]
    }


def _configured_transfers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    transfers = {
        str(transfer["transfer_id"]): transfer for transfer in config["transfers"]
    }
    if len(transfers) != len(config["transfers"]):
        raise ValueError("config transfer ids must be unique")
    return transfers


def _configured_models(config: dict[str, Any]) -> set[str]:
    models = {str(model["model_id"]) for model in config["models"]}
    if models != EXPECTED_MODEL_IDS:
        raise ValueError("config model ids do not match the frozen three-model set")
    return models


def _expected_scope_pairs(config: dict[str, Any]) -> set[tuple[str, str]]:
    transfers = _configured_transfers(config)
    reportable = {
        transfer_id
        for transfer_id, transfer in transfers.items()
        if transfer["raw_batch_disjoint_sensitivity"]["status"] == "REPORTABLE"
    }
    return {
        (transfer_id, PRIMARY_ANALYSIS_SCOPE) for transfer_id in transfers
    } | {
        (transfer_id, DISJOINT_ANALYSIS_SCOPE) for transfer_id in reportable
    }


def validate_source_inventory(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source experiment directory not found: {source_dir}")
    entries = list(source_dir.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("source experiment must not contain symlinks")
    if any(entry.is_dir() for entry in entries):
        raise ValueError("source experiment must contain only the frozen root files")
    actual = {entry.name for entry in entries if entry.is_file()}
    expected = set(SOURCE_FILES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"source experiment file set mismatch; missing={missing}, extra={extra}"
        )


def validate_summary(
    summary: dict[str, Any],
    *,
    source_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    implementation_path: Path,
    commit_validator: CommitValidator,
) -> None:
    _require_exact_keys(
        summary,
        {
            "status",
            "experiment_id",
            "dataset",
            "implementation_commit",
            "config_sha256",
            "implementation_sha256",
            "feature_implementation_sha256",
            "random_state",
            "transfer_count",
            "model_count",
            "model_fit_count",
            "bootstrap_replicates",
            "source_files",
            "target_rows_used_for_fit",
            "target_labels_used_for_fit_threshold_calibration_or_model_selection",
            "target_statistics_used_for_scaling_or_calibration",
            "probability_calibration_performed",
            "threshold_tuning_performed",
            "decision_rule",
            "dummy_prior_fit_scope",
            "hyperparameter_search_performed",
            "test_driven_model_selection_performed",
            "single_sealed_run",
            "sealed_run_consumption_enforced",
            "formal_output_overwrite_allowed",
            "sample_predictions_saved",
            "raw_data_in_output",
            "model_checkpoints_saved",
            "physical_frequency_hz_reported",
            "primary_gate_passed",
            "same_target_reuse_for_future_confirmatory_comparison_allowed",
            "same_target_confirmatory_reuse_allowed",
            "bootstrap_inference_scope",
            "claim_scope",
            "output_files",
            "output_sha256",
        },
        "summary.json",
    )
    _require_equal(
        summary,
        {
            "status": EXPECTED_STATUS,
            "experiment_id": config["experiment_id"],
            "dataset": config["dataset"],
            "random_state": int(config["random_state"]),
            "transfer_count": len(config["transfers"]),
            "model_count": len(config["models"]),
            "model_fit_count": len(_expected_scope_pairs(config))
            * len(config["models"]),
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
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
            "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
            "same_target_confirmatory_reuse_allowed": False,
            "bootstrap_inference_scope": "conditional_on_each_fixed_source_fit",
            "claim_scope": "dataset-internal released-band-held-out UAV/weather transfer",
        },
        "source summary",
    )
    config_sha256 = sha256_file(config_path)
    implementation_sha256 = sha256_file(implementation_path)
    feature_path = (
        PROJECT_ROOT / str(config["feature_contract"]["implementation"])
    ).resolve()
    feature_sha256 = sha256_file(feature_path)
    if summary.get("config_sha256") != config_sha256:
        raise ValueError("source summary config hash is stale")
    if summary.get("implementation_sha256") != implementation_sha256:
        raise ValueError("source summary implementation hash is stale")
    if (
        feature_sha256 != config["feature_contract"]["implementation_sha256"]
        or summary.get("feature_implementation_sha256") != feature_sha256
    ):
        raise ValueError("source summary feature implementation hash is stale")
    commit = str(summary.get("implementation_commit", ""))
    if not HEX_COMMIT.fullmatch(commit):
        raise ValueError("source summary implementation commit is invalid")
    if not commit_validator(
        commit,
        {
            config_path: config_sha256,
            implementation_path: implementation_sha256,
            feature_path: feature_sha256,
        },
    ):
        raise ValueError("config, runner and features are not bound to the source commit")
    if not commit_validator(
        str(config["feature_contract"]["frozen_baseline_commit"]),
        {feature_path: feature_sha256},
    ):
        raise ValueError("feature implementation is not bound to its frozen commit")
    output_files = summary.get("output_files")
    if not isinstance(output_files, list) or list(map(str, output_files)) != list(
        config["output_contract"]["files"]
    ):
        raise ValueError("source summary output file inventory is not the frozen 15-file set")
    output_sha256 = summary.get("output_sha256")
    hashed_files = set(SOURCE_FILES) - {"summary.json"}
    if not isinstance(output_sha256, dict) or set(output_sha256) != hashed_files:
        raise ValueError("source summary output hashes are incomplete")
    for name in hashed_files:
        digest = str(output_sha256[name])
        if not HEX_SHA256.fullmatch(digest) or digest != sha256_file(source_dir / name):
            raise ValueError(f"source summary output hash mismatch: {name}")

    configured_sources = _configured_sources(config)
    source_files = summary.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != 5:
        raise ValueError("source summary must bind exactly five aggregate hashes")
    actual_sources: dict[str, dict[str, Any]] = {}
    for record in source_files:
        if not isinstance(record, dict):
            raise ValueError("source summary source_files entries must be objects")
        _require_exact_keys(
            record,
            {
                "source_id",
                "sha256",
                "analysis_record_count",
                "full_release_batch_count",
            },
            "source summary source file",
        )
        source_id = str(record.get("source_id", ""))
        if source_id in actual_sources:
            raise ValueError("source summary contains a duplicate source_id")
        if "relative_path" in record or "absolute_path" in record:
            raise ValueError("source summary must not publish source paths")
        actual_sources[source_id] = record
    if set(actual_sources) != set(configured_sources):
        raise ValueError("source summary source ids do not match the frozen config")
    for source_id, source in configured_sources.items():
        record = actual_sources[source_id]
        if record.get("sha256") != source["expected_sha256"]:
            raise ValueError(f"{source_id}: source summary SHA256 mismatch")
        coverage = source["expected_analysis_coverage"]
        count = int(record.get("analysis_record_count", 0))
        full_batches = int(record.get("full_release_batch_count", 0))
        if count != int(coverage["record_count"]):
            raise ValueError(f"{source_id}: invalid analysis record count")
        if full_batches < int(coverage["unique_batch_count"]):
            raise ValueError(f"{source_id}: invalid full-release batch count")
    audit_json_payload(summary, name="summary.json")


def validate_consumption_record(
    record: dict[str, Any],
    *,
    summary: dict[str, Any],
    summary_path: Path,
    config: dict[str, Any],
) -> None:
    if not isinstance(record, dict):
        raise ValueError("formal-run consumption record must be a JSON object")
    expected = {
        "schema_version": 1,
        "status": "COMPLETED_AND_TARGETS_CONSUMED",
        "experiment_id": config["experiment_id"],
        "pre_result_commit": summary["implementation_commit"],
        "config_sha256": summary["config_sha256"],
        "implementation_sha256": summary["implementation_sha256"],
        "target_bands_consumed": config["stopping_rule"][
            "target_bands_consumed_by_this_run"
        ],
        "summary_sha256": sha256_file(summary_path),
        "sealed_run_consumption_enforced": True,
        "formal_output_overwrite_allowed": False,
        "record_created_before_target_load": True,
        "persists_on_failure": True,
    }
    if set(record) != set(expected):
        raise ValueError("formal-run consumption record schema changed")
    _require_equal(record, expected, "formal-run consumption record")
    audit_json_payload(record, name="formal-run consumption record")


def _formal_consumption_record_path(config: dict[str, Any]) -> Path:
    relative = Path(
        str(config["acceptance_contract"]["formal_run_consumption_record"])
    )
    if relative.is_absolute():
        raise ValueError("formal-run consumption record path must remain relative")
    resolved = (PROJECT_ROOT / relative).resolve()
    expected = (
        PROJECT_ROOT
        / "results/final_evidence/lat_mricd_cross_band_transfer_v1.run_consumed.json"
    ).resolve()
    if resolved != expected:
        raise ValueError("formal-run consumption record path changed")
    return resolved


def validate_model_fit_manifest(
    manifest: dict[str, Any],
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    weight_audit: pd.DataFrame,
    weight_audit_sha256: str,
) -> None:
    training = config["training_contract"]
    _require_exact_keys(
        manifest,
        {
            "status",
            "experiment_id",
            "implementation_commit",
            "config_sha256",
            "implementation_sha256",
            "feature_implementation_sha256",
            "fit_scope",
            "scaler_fit_scope",
            "probability_calibration",
            "decision_rule",
            "argmax_tie_break",
            "dummy_prior_fit_scope",
            "dummy_prior_uses_same_sample_weights_as_learned_models",
            "target_rows_used_for_fit",
            "target_labels_used_for_fit_threshold_calibration_or_model_selection",
            "target_statistics_used_for_scaling_or_calibration",
            "threshold_tuning_performed",
            "hyperparameter_search_performed",
            "test_driven_model_selection_performed",
            "single_sealed_run",
            "sealed_run_consumption_enforced",
            "formal_output_overwrite_allowed",
            "sample_weight_hierarchy",
            "source_files",
            "models",
            "model_ids",
            "transfer_ids",
            "analysis_scopes",
            "fit_count",
            "transfers",
            "training_weight_audit_sha256",
        },
        "model_fit_manifest.json",
    )
    _require_equal(
        manifest,
        {
            "status": "COMPLETE_SOURCE_ONLY_MODEL_FIT_MANIFEST",
            "experiment_id": config["experiment_id"],
            "implementation_commit": summary["implementation_commit"],
            "config_sha256": summary["config_sha256"],
            "implementation_sha256": summary["implementation_sha256"],
            "feature_implementation_sha256": summary[
                "feature_implementation_sha256"
            ],
            "fit_scope": "source_bands_only",
            "scaler_fit_scope": "source_bands_only",
            "probability_calibration": "none",
            "decision_rule": training["prediction_decision"],
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
            "sample_weight_hierarchy": training["sample_weight_hierarchy"],
            "training_weight_audit_sha256": weight_audit_sha256,
        },
        "model fit manifest",
    )
    if set(map(str, manifest.get("model_ids", []))) != EXPECTED_MODEL_IDS:
        raise ValueError("model fit manifest model ids are incomplete")
    if set(map(str, manifest.get("transfer_ids", []))) != set(
        _configured_transfers(config)
    ):
        raise ValueError("model fit manifest transfer ids are incomplete")
    scopes = set(map(str, manifest.get("analysis_scopes", [])))
    if scopes != {PRIMARY_ANALYSIS_SCOPE, DISJOINT_ANALYSIS_SCOPE}:
        raise ValueError("model fit manifest analysis scopes are incomplete")
    source_files = manifest.get("source_files")
    if source_files != summary.get("source_files"):
        raise ValueError("model fit manifest source hashes differ from summary")
    if manifest.get("models") != config["models"]:
        raise ValueError("model fit manifest model specifications changed")

    fits = manifest.get("transfers")
    expected_scope_pairs = _expected_scope_pairs(config)
    expected_fit_keys = {
        (transfer_id, scope, model_id)
        for transfer_id, scope in expected_scope_pairs
        for model_id in EXPECTED_MODEL_IDS
    }
    if (
        not isinstance(fits, list)
        or int(manifest.get("fit_count", -1)) != len(expected_fit_keys)
        or int(summary.get("model_fit_count", -1)) != len(expected_fit_keys)
    ):
        raise ValueError("model fit manifest fit count changed")
    fit_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fit in fits:
        if not isinstance(fit, dict):
            raise ValueError("model fit records must be objects")
        _require_exact_keys(
            fit,
            {
                "transfer_id",
                "role",
                "analysis_scope",
                "model_id",
                "source_ids",
                "source_bands",
                "target_source_id",
                "target_band",
                "random_state",
                "feature_count",
                "source_record_count",
                "source_batch_class_cell_count",
                "target_record_count_evaluated",
                "fit_scope",
                "scaler_fit_scope",
                "target_rows_used_for_fit",
                "target_labels_used_for_fit_threshold_calibration_or_model_selection",
                "target_statistics_used_for_scaling_or_calibration",
                "sample_weights_fit_scope",
                "probability_calibration",
                "threshold_tuning_performed",
                "hyperparameter_search_performed",
                "test_driven_model_selection_performed",
            },
            "model fit record",
        )
        key = (
            str(fit.get("transfer_id", "")),
            str(fit.get("analysis_scope", "")),
            str(fit.get("model_id", "")),
        )
        if key in fit_lookup:
            raise ValueError(f"duplicate model fit record: {key}")
        fit_lookup[key] = fit
    if set(fit_lookup) != expected_fit_keys:
        raise ValueError("model fit transfer/scope/model coverage is incomplete")

    transfers = _configured_transfers(config)
    sources = _configured_sources(config)
    model_indexes = {
        str(model["model_id"]): index for index, model in enumerate(config["models"])
    }
    transfer_indexes = {
        str(transfer["transfer_id"]): index
        for index, transfer in enumerate(config["transfers"])
    }
    weight_totals = weight_audit.groupby(
        ["transfer_id", "analysis_scope"], observed=True
    ).agg(source_record_count=("record_count", "sum"), cell_count=("batch_count", "sum"))
    source_only_flags = {
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
    for (transfer_id, scope, model_id), fit in fit_lookup.items():
        transfer = transfers[transfer_id]
        target = sources[str(transfer["target_source_id"])]
        _require_equal(fit, source_only_flags, f"model fit {transfer_id}/{scope}/{model_id}")
        expected_seed = (
            int(config["random_state"])
            + transfer_indexes[transfer_id] * 100
            + model_indexes[model_id]
        )
        expected_identity = {
            "role": transfer["role"],
            "source_ids": transfer["source_ids"],
            "source_bands": transfer["source_bands"],
            "target_source_id": transfer["target_source_id"],
            "target_band": transfer["target_band"],
            "random_state": expected_seed,
            "feature_count": int(
                config["feature_contract"]["feature_schemas"][
                    transfer["representation"]
                ]["feature_count"]
            ),
            "source_record_count": int(
                weight_totals.loc[(transfer_id, scope), "source_record_count"]
            ),
            "source_batch_class_cell_count": int(
                weight_totals.loc[(transfer_id, scope), "cell_count"]
            ),
            "target_record_count_evaluated": int(
                target["expected_analysis_coverage"]["record_count"]
            ),
        }
        _require_equal(
            fit,
            expected_identity,
            f"model fit {transfer_id}/{scope}/{model_id}",
        )
    audit_json_payload(manifest, name="model_fit_manifest.json")


def validate_transfer_coverage(
    frame: pd.DataFrame, *, config: dict[str, Any], summary: dict[str, Any]
) -> None:
    required = {
        "source_id",
        "representation",
        "band_code",
        "band",
        "category_code",
        "category",
        "record_count",
        "analysis_batch_count",
        "analysis_total_record_count",
        "analysis_unique_batch_count",
        "full_release_batch_count",
        "sha256",
    }
    _require_exact_columns(frame, required, "transfer_coverage.csv")
    if frame.empty or frame.duplicated(["source_id", "category_code"]).any():
        raise ValueError("transfer coverage must contain one row per source/class")
    configured = _configured_sources(config)
    if set(frame["source_id"].astype(str)) != set(configured):
        raise ValueError("transfer coverage source ids are incomplete")
    if set(pd.to_numeric(frame["category_code"]).astype(int)) != CLASS_CODES:
        raise ValueError("transfer coverage must contain only UAV and weather")
    if len(frame) != len(configured) * len(CLASS_CODES):
        raise ValueError("transfer coverage source/class coverage is incomplete")
    _ensure_finite(
        frame,
        [
            "band_code",
            "category_code",
            "record_count",
            "analysis_batch_count",
            "analysis_total_record_count",
            "analysis_unique_batch_count",
            "full_release_batch_count",
        ],
        "transfer_coverage.csv",
    )
    summary_sources = {
        str(record["source_id"]): record for record in summary["source_files"]
    }
    for record in frame.to_dict(orient="records"):
        source = configured[str(record["source_id"])]
        coverage = source["expected_analysis_coverage"]
        class_code = int(record["category_code"])
        class_coverage = coverage["classes"][str(class_code)]
        if record["sha256"] != source["expected_sha256"]:
            raise ValueError(f"{record['source_id']}: transfer coverage hash mismatch")
        if str(record["representation"]) != source["representation"]:
            raise ValueError(f"{record['source_id']}: representation mismatch")
        if int(record["band_code"]) != int(source["band_code"]):
            raise ValueError(f"{record['source_id']}: band code mismatch")
        if str(record["band"]) != source["band"]:
            raise ValueError(f"{record['source_id']}: band mismatch")
        if str(record["category"]) != config["class_contract"]["class_names"][
            str(class_code)
        ]:
            raise ValueError(f"{record['source_id']}: category name mismatch")
        expected_counts = {
            "record_count": int(class_coverage["record_count"]),
            "analysis_batch_count": int(class_coverage["batch_count"]),
            "analysis_total_record_count": int(coverage["record_count"]),
            "analysis_unique_batch_count": int(coverage["unique_batch_count"]),
            "full_release_batch_count": int(
                summary_sources[str(record["source_id"])]["full_release_batch_count"]
            ),
        }
        for field, expected in expected_counts.items():
            if int(record[field]) != expected:
                raise ValueError(
                    f"{record['source_id']}/{class_code}: {field} changed"
                )
    audit_table_payload(frame, name="transfer_coverage.csv")


def validate_training_weight_audit(
    frame: pd.DataFrame, *, config: dict[str, Any]
) -> None:
    required = {
        "transfer_id",
        "analysis_scope",
        "source_band",
        "source_band_code",
        "category_code",
        "category",
        "record_count",
        "batch_count",
        "total_weight",
        "minimum_batch_cell_weight",
        "maximum_batch_cell_weight",
    }
    _require_exact_columns(frame, required, "training_weight_audit.csv")
    if frame.empty or frame.duplicated(
        ["transfer_id", "analysis_scope", "source_band_code", "category_code"]
    ).any():
        raise ValueError("training weight audit contains duplicate or missing cells")
    numeric = [
        "source_band_code",
        "category_code",
        "record_count",
        "batch_count",
        "total_weight",
        "minimum_batch_cell_weight",
        "maximum_batch_cell_weight",
    ]
    _ensure_finite(frame, numeric, "training_weight_audit.csv")
    positive = frame[[
        "record_count",
        "batch_count",
        "total_weight",
        "minimum_batch_cell_weight",
        "maximum_batch_cell_weight",
    ]].to_numpy(float)
    if np.any(positive <= 0):
        raise ValueError("training weight audit values must be positive")
    if not np.allclose(
        frame["minimum_batch_cell_weight"].to_numpy(float),
        frame["maximum_batch_cell_weight"].to_numpy(float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("batch-class cell total weights are not equal")

    transfers = _configured_transfers(config)
    reportable = {
        transfer_id
        for transfer_id, transfer in transfers.items()
        if transfer["raw_batch_disjoint_sensitivity"]["status"] == "REPORTABLE"
    }
    expected_pairs = {
        (transfer_id, PRIMARY_ANALYSIS_SCOPE) for transfer_id in transfers
    } | {(transfer_id, DISJOINT_ANALYSIS_SCOPE) for transfer_id in reportable}
    actual_pairs = set(
        zip(frame["transfer_id"].astype(str), frame["analysis_scope"].astype(str), strict=True)
    )
    if actual_pairs != expected_pairs:
        raise ValueError("training weight audit transfer/scope coverage is incomplete")

    for (transfer_id, scope), group in frame.groupby(
        ["transfer_id", "analysis_scope"], observed=True
    ):
        transfer = transfers[str(transfer_id)]
        if set(group["source_band"].astype(str)) != set(transfer["source_bands"]):
            raise ValueError(f"{transfer_id}/{scope}: source-band coverage mismatch")
        if set(pd.to_numeric(group["category_code"]).astype(int)) != CLASS_CODES:
            raise ValueError(f"{transfer_id}/{scope}: class coverage mismatch")
        if len(group) != len(transfer["source_bands"]) * len(CLASS_CODES):
            raise ValueError(f"{transfer_id}/{scope}: band/class cells are incomplete")
        source_by_band = dict(
            zip(transfer["source_bands"], transfer["source_ids"], strict=True)
        )
        for record in group.to_dict(orient="records"):
            class_code = int(record["category_code"])
            source = _configured_sources(config)[
                str(source_by_band[str(record["source_band"])])
            ]
            class_coverage = source["expected_analysis_coverage"]["classes"][
                str(class_code)
            ]
            if int(record["source_band_code"]) != int(source["band_code"]):
                raise ValueError(f"{transfer_id}/{scope}: source band code changed")
            if str(record["category"]) != config["class_contract"]["class_names"][
                str(class_code)
            ]:
                raise ValueError(f"{transfer_id}/{scope}: category name changed")
            if scope == PRIMARY_ANALYSIS_SCOPE:
                if int(record["record_count"]) != int(class_coverage["record_count"]):
                    raise ValueError(
                        f"{transfer_id}/{scope}: source record coverage changed"
                    )
                if int(record["batch_count"]) != int(class_coverage["batch_count"]):
                    raise ValueError(
                        f"{transfer_id}/{scope}: source batch coverage changed"
                    )
            elif (
                int(record["record_count"]) > int(class_coverage["record_count"])
                or int(record["batch_count"]) > int(class_coverage["batch_count"])
            ):
                raise ValueError(
                    f"{transfer_id}/{scope}: disjoint source coverage grew"
                )
        class_totals = group.groupby("category_code", observed=True)["total_weight"].sum()
        if not np.allclose(class_totals, class_totals.iloc[0], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{transfer_id}/{scope}: class weights are not equal")
        for _, class_rows in group.groupby("category_code", observed=True):
            totals = class_rows["total_weight"].to_numpy(float)
            if not np.allclose(totals, totals[0], rtol=1e-10, atol=1e-10):
                raise ValueError(
                    f"{transfer_id}/{scope}: source-band weights are not equal within class"
                )
    # The aggregate table is safe; only per-sample weight fields remain forbidden.
    forbidden_columns = {"sample_weight", "source_row_index", "sample_id"}
    if forbidden_columns & set(frame.columns):
        raise ValueError("training weight audit contains per-sample fields")
    audit_table_payload(frame, name="training_weight_audit.csv")


def _validate_metric_bounds(frame: pd.DataFrame, *, name: str) -> None:
    bounded = [
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_macro_f1",
        "roc_auc",
        "recall_uav",
        "recall_weather",
        "target_batch_macro_accuracy",
        "target_batch_accuracy_p10",
        "worst_target_batch_accuracy",
        "target_batch_class_macro_accuracy",
        "target_batch_class_recall_uav",
        "target_batch_class_recall_weather",
        "target_batch_class_cell_accuracy_p10",
        "worst_target_batch_class_cell_accuracy",
    ]
    present = [column for column in bounded if column in frame]
    _ensure_finite(frame, present, name)
    if present:
        values = frame[present].to_numpy(float)
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError(f"{name} contains a metric outside [0, 1]")
    if "binary_log_loss" in frame:
        _ensure_finite(frame, ["binary_log_loss"], name)
        if np.any(frame["binary_log_loss"].to_numpy(float) < 0):
            raise ValueError(f"{name} contains a negative log loss")


def validate_aggregate_metrics(
    frame: pd.DataFrame, *, config: dict[str, Any], name: str = "aggregate_metrics.csv"
) -> None:
    required = {
        "transfer_id",
        "role",
        "analysis_scope",
        "source_bands",
        "model_id",
        "representation",
        "band_code",
        "band",
        "target_record_count",
        "target_batch_count",
        "recall_uav",
        "recall_weather",
        *set(config["metrics"]["required_metrics"]),
    }
    _require_exact_columns(frame, required, name)
    if frame.empty or frame.duplicated(
        ["transfer_id", "analysis_scope", "model_id"]
    ).any():
        raise ValueError(f"{name} contains duplicate or missing model rows")
    transfers = _configured_transfers(config)
    sources = _configured_sources(config)
    models = _configured_models(config)
    expected = {
        (transfer_id, PRIMARY_ANALYSIS_SCOPE, model_id)
        for transfer_id in transfers
        for model_id in models
    }
    if name == "aggregate_metrics.csv":
        actual = set(
            zip(
                frame["transfer_id"].astype(str),
                frame["analysis_scope"].astype(str),
                frame["model_id"].astype(str),
                strict=True,
            )
        )
        if actual != expected:
            raise ValueError("aggregate primary transfer/model coverage is incomplete")
    for record in frame.to_dict(orient="records"):
        transfer_id = str(record["transfer_id"])
        if transfer_id not in transfers:
            raise ValueError(f"{name} contains an unknown transfer")
        transfer = transfers[transfer_id]
        if str(record["role"]) != transfer["role"]:
            raise ValueError(f"{transfer_id}: role mismatch")
        if str(record["representation"]) != transfer["representation"]:
            raise ValueError(f"{transfer_id}: representation mismatch")
        if str(record["source_bands"]) != "+".join(transfer["source_bands"]):
            raise ValueError(f"{transfer_id}: source bands mismatch")
        if str(record["band"]) != transfer["target_band"]:
            raise ValueError(f"{transfer_id}: target band mismatch")
        target = sources[str(transfer["target_source_id"])]
        if int(record["band_code"]) != int(target["band_code"]):
            raise ValueError(f"{transfer_id}: target band code mismatch")
        if str(record["model_id"]) not in models:
            raise ValueError(f"{transfer_id}: unknown model id")
        coverage = target["expected_analysis_coverage"]
        if int(record["target_record_count"]) != int(coverage["record_count"]):
            raise ValueError(f"{transfer_id}: target record coverage changed")
        if int(record["target_batch_count"]) != int(coverage["unique_batch_count"]):
            raise ValueError(f"{transfer_id}: target batch coverage changed")
    _validate_metric_bounds(frame, name=name)
    audit_table_payload(frame, name=name)


def validate_target_batch_metrics(
    frame: pd.DataFrame, *, aggregate: pd.DataFrame
) -> None:
    required = {
        "transfer_id",
        "role",
        "analysis_scope",
        "source_bands",
        "model_id",
        "representation",
        "band_code",
        "band",
        "batch_code",
        "category_code",
        "category",
        "record_count",
        "accuracy",
        "evaluation_unit",
    }
    _require_exact_columns(frame, required, "target_batch_class_metrics.csv")
    if frame.empty:
        raise ValueError("target batch metrics must not be empty")
    if set(frame["analysis_scope"].astype(str)) != {PRIMARY_ANALYSIS_SCOPE}:
        raise ValueError("target batch metrics contain an unexpected analysis scope")
    units = set(frame["evaluation_unit"].astype(str))
    if units != {"target_batch", "target_batch_class_cell"}:
        raise ValueError("target batch metrics evaluation units are incomplete")
    _ensure_finite(frame, ["record_count", "accuracy"], "target_batch_class_metrics.csv")
    if np.any(frame["record_count"].to_numpy(float) <= 0) or np.any(
        (frame["accuracy"].to_numpy(float) < -1e-12)
        | (frame["accuracy"].to_numpy(float) > 1.0 + 1e-12)
    ):
        raise ValueError("target batch metrics contain invalid counts or accuracies")
    duplicate_key = [
        "transfer_id",
        "analysis_scope",
        "model_id",
        "batch_code",
        "category_code",
        "evaluation_unit",
    ]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("target batch metrics contain duplicate cells")
    aggregate_keys = set(
        zip(
            aggregate["transfer_id"].astype(str),
            aggregate["analysis_scope"].astype(str),
            aggregate["model_id"].astype(str),
            strict=True,
        )
    )
    table_keys = set(
        zip(
            frame["transfer_id"].astype(str),
            frame["analysis_scope"].astype(str),
            frame["model_id"].astype(str),
            strict=True,
        )
    )
    if table_keys != aggregate_keys:
        raise ValueError("target batch metrics contain unknown or missing model rows")

    for record in aggregate.to_dict(orient="records"):
        selected = frame.loc[
            frame["transfer_id"].eq(record["transfer_id"])
            & frame["analysis_scope"].eq(record["analysis_scope"])
            & frame["model_id"].eq(record["model_id"])
        ]
        batches = selected.loc[selected["evaluation_unit"].eq("target_batch")]
        cells = selected.loc[
            selected["evaluation_unit"].eq("target_batch_class_cell")
        ].copy()
        if batches.empty or cells.empty:
            raise ValueError("target batch metrics do not cover every aggregate row")
        cells["category_code"] = pd.to_numeric(cells["category_code"]).astype(int)
        if set(cells["category_code"]) != CLASS_CODES:
            raise ValueError("target batch-class cells lack a frozen class")
        class_means = cells.groupby("category_code", observed=True)["accuracy"].mean()
        worst = cells.sort_values(
            ["accuracy", "record_count", "batch_code"],
            ascending=[True, True, True],
        ).iloc[0]
        recomputed = {
            "target_batch_class_macro_accuracy": float(class_means.mean()),
            "target_batch_class_recall_uav": float(class_means.loc[1]),
            "target_batch_class_recall_weather": float(class_means.loc[3]),
            "target_batch_class_cell_accuracy_p10": float(cells["accuracy"].quantile(0.10)),
            "worst_target_batch_class_cell_accuracy": float(cells["accuracy"].min()),
            "worst_target_batch_class_cell_record_count": int(
                worst["record_count"]
            ),
            "target_batch_macro_accuracy": float(batches["accuracy"].mean()),
            "target_batch_accuracy_p10": float(batches["accuracy"].quantile(0.10)),
            "worst_target_batch_accuracy": float(batches["accuracy"].min()),
        }
        for field, value in recomputed.items():
            if not _same_float(record[field], value):
                raise ValueError(
                    f"{record['transfer_id']}/{record['model_id']}: {field} does not "
                    "match target batch cells"
                )
        if int(record["target_record_count"]) != int(batches["record_count"].sum()):
            raise ValueError("aggregate target record count does not match target batches")
        if int(record["target_record_count"]) != int(cells["record_count"].sum()):
            raise ValueError("aggregate target record count does not match class cells")
        if int(record["target_batch_count"]) != int(batches["batch_code"].nunique()):
            raise ValueError("aggregate target batch count does not match target batches")
    audit_table_payload(frame, name="target_batch_class_metrics.csv")


def validate_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    aggregate: pd.DataFrame,
    disjoint: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    required = {
        "transfer_id",
        "role",
        "analysis_scope",
        "representation",
        "target_band",
        "comparison",
        "metric",
        "estimate",
        "ci_lower_95",
        "ci_upper_95",
        "requested_replicates",
        "valid_replicates",
        "discarded_replicates",
        "bootstrap_seed",
        "draw_count_per_replicate",
        "resampling_unit",
        "duplicate_batch_draws_count_with_multiplicity",
        "percentile_quantile_method",
        "inference_scope",
        "conditioning",
        "identical_paired_draws",
    }
    _require_exact_columns(frame, required, "bootstrap_intervals.csv")
    key_columns = ["transfer_id", "analysis_scope", "comparison"]
    if frame.empty or frame.duplicated(key_columns).any():
        raise ValueError("bootstrap intervals contain duplicate or missing comparisons")
    transfers = _configured_transfers(config)
    expected = {
        (transfer_id, scope, comparison)
        for transfer_id, scope in _expected_scope_pairs(config)
        for comparison in (*sorted(EXPECTED_MODEL_IDS), PAIR_COMPARISON)
    }
    actual = set(
        zip(
            frame["transfer_id"].astype(str),
            frame["analysis_scope"].astype(str),
            frame["comparison"].astype(str),
            strict=True,
        )
    )
    if actual != expected:
        raise ValueError("bootstrap transfer/scope/comparison coverage is incomplete")
    if len(frame) != 36:
        raise ValueError("bootstrap interval table must contain exactly 36 rows")
    numeric = [
        "estimate",
        "ci_lower_95",
        "ci_upper_95",
        "requested_replicates",
        "valid_replicates",
        "discarded_replicates",
        "bootstrap_seed",
        "draw_count_per_replicate",
    ]
    _ensure_finite(frame, numeric, "bootstrap_intervals.csv")
    requested = int(config["bootstrap_replicates"])
    minimum_valid = int(config["bootstrap_contract"]["minimum_valid_replicates"])
    if not frame["requested_replicates"].astype(int).eq(requested).all():
        raise ValueError("bootstrap requested replicate count changed")
    valid = frame["valid_replicates"].astype(int)
    discarded = frame["discarded_replicates"].astype(int)
    if (valid < minimum_valid).any() or (valid > requested).any():
        raise ValueError("bootstrap valid replicate count violates the frozen contract")
    if (discarded < 0).any() or not (valid + discarded).eq(requested).all():
        raise ValueError("bootstrap discarded replicate accounting is inconsistent")
    if not frame["conditioning"].eq("conditional_on_each_fixed_source_fit").all():
        raise ValueError("bootstrap conditioning scope changed")
    if not frame["inference_scope"].eq(
        "conditional_on_each_fixed_source_fit"
    ).all():
        raise ValueError("bootstrap inference scope changed")
    if not frame["resampling_unit"].eq(
        "representation_target_band_code_batch_code"
    ).all():
        raise ValueError("bootstrap resampling unit changed")
    if not frame["percentile_quantile_method"].eq("linear").all():
        raise ValueError("bootstrap percentile quantile method changed")
    multiplicity = frame["duplicate_batch_draws_count_with_multiplicity"].map(
        lambda value: _as_bool(value, field="duplicate_batch_draws_count_with_multiplicity")
    )
    if not multiplicity.all():
        raise ValueError("bootstrap duplicate draws must count with multiplicity")
    if (frame["ci_lower_95"] > frame["ci_upper_95"]).any():
        raise ValueError("bootstrap interval bounds are reversed")

    disjoint_models = disjoint.loc[disjoint["model_id"].notna()].copy()
    metric_rows = pd.concat(
        [
            aggregate,
            disjoint_models[
                [
                    "transfer_id",
                    "analysis_scope",
                    "model_id",
                    PRIMARY_METRIC,
                    "target_batch_count",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    if metric_rows.duplicated(["transfer_id", "analysis_scope", "model_id"]).any():
        raise ValueError("bootstrap metric lookup contains duplicate scope rows")
    metric_lookup = metric_rows.set_index(
        ["transfer_id", "analysis_scope", "model_id"]
    )
    for record in frame.to_dict(orient="records"):
        transfer_id = str(record["transfer_id"])
        scope = str(record["analysis_scope"])
        comparison = str(record["comparison"])
        transfer = transfers[transfer_id]
        if str(record["role"]) != transfer["role"]:
            raise ValueError(f"{transfer_id}/{scope}: bootstrap role changed")
        if str(record["representation"]) != transfer["representation"]:
            raise ValueError(f"{transfer_id}/{scope}: bootstrap representation changed")
        if str(record["target_band"]) != transfer["target_band"]:
            raise ValueError(f"{transfer_id}/{scope}: bootstrap target band changed")
        expected_seed = bootstrap_seed(int(config["random_state"]), scope, transfer_id)
        if int(record["bootstrap_seed"]) != expected_seed:
            raise ValueError(f"{transfer_id}/{scope}: bootstrap seed changed")
        expected_draw_count = int(
            metric_lookup.loc[
                (transfer_id, scope, PRIMARY_MODEL_ID), "target_batch_count"
            ]
        )
        if int(record["draw_count_per_replicate"]) != expected_draw_count:
            raise ValueError(f"{transfer_id}/{scope}: bootstrap draw count changed")
        is_pair = comparison == PAIR_COMPARISON
        if _as_bool(record["identical_paired_draws"], field="identical_paired_draws") != is_pair:
            raise ValueError("bootstrap paired-draw flag is inconsistent")
        if is_pair:
            if record["metric"] != PAIR_METRIC:
                raise ValueError("paired bootstrap metric changed")
            expected_estimate = float(
                metric_lookup.loc[
                    (transfer_id, scope, PRIMARY_MODEL_ID), PRIMARY_METRIC
                ]
            ) - float(
                metric_lookup.loc[
                    (transfer_id, scope, "dummy_prior"), PRIMARY_METRIC
                ]
            )
            if float(record["ci_lower_95"]) < -1.0 or float(
                record["ci_upper_95"]
            ) > 1.0:
                raise ValueError("paired bootstrap interval is outside [-1, 1]")
        else:
            if record["metric"] != PRIMARY_METRIC:
                raise ValueError("model bootstrap metric changed")
            expected_estimate = float(
                metric_lookup.loc[(transfer_id, scope, comparison), PRIMARY_METRIC]
            )
            if float(record["ci_lower_95"]) < 0.0 or float(
                record["ci_upper_95"]
            ) > 1.0:
                raise ValueError("model bootstrap interval is outside [0, 1]")
        if not _same_float(record["estimate"], expected_estimate):
            raise ValueError(
                f"{transfer_id}/{scope}/{comparison}: bootstrap estimate mismatch"
            )
    audit_table_payload(frame, name="bootstrap_intervals.csv")


def validate_raw_batch_overlap(
    frame: pd.DataFrame, *, config: dict[str, Any]
) -> None:
    required = {
        "transfer_id",
        "role",
        "representation",
        "source_bands",
        "target_band",
        "full_release_source_batch_count",
        "full_release_target_batch_count",
        "full_release_overlap_code_count",
        "analysis_source_batch_count",
        "analysis_target_batch_count",
        "analysis_subset_overlap_code_count",
        "expected_analysis_subset_overlap_code_count",
        "full_release_overlap_codes",
        "analysis_subset_overlap_codes",
        "global_raw_batch_semantics_verified",
        "primary_group_key_band_qualified",
    }
    _require_exact_columns(frame, required, "raw_batch_overlap_audit.csv")
    transfers = _configured_transfers(config)
    if frame.empty or frame["transfer_id"].duplicated().any() or set(
        frame["transfer_id"].astype(str)
    ) != set(transfers):
        raise ValueError("raw batch overlap audit transfer coverage is incomplete")
    count_columns = [column for column in required if column.endswith("_count")]
    _ensure_finite(frame, count_columns, "raw_batch_overlap_audit.csv")
    if np.any(frame[count_columns].to_numpy(float) < 0):
        raise ValueError("raw batch overlap counts must be nonnegative")
    for record in frame.to_dict(orient="records"):
        transfer = transfers[str(record["transfer_id"])]
        if str(record["role"]) != transfer["role"]:
            raise ValueError(f"{record['transfer_id']}: overlap role changed")
        if str(record["representation"]) != transfer["representation"]:
            raise ValueError(f"{record['transfer_id']}: overlap representation changed")
        if str(record["source_bands"]) != "+".join(transfer["source_bands"]):
            raise ValueError(f"{record['transfer_id']}: overlap source bands changed")
        if str(record["target_band"]) != transfer["target_band"]:
            raise ValueError(f"{record['transfer_id']}: overlap target band changed")
        expected_overlap = int(
            transfer["raw_batch_disjoint_sensitivity"]["expected_overlap_code_count"]
        )
        if int(record["analysis_subset_overlap_code_count"]) != int(
            expected_overlap
        ) or int(record["expected_analysis_subset_overlap_code_count"]) != int(
            expected_overlap
        ):
            raise ValueError(f"{record['transfer_id']}: analysis overlap count changed")
        if _as_bool(
            record["global_raw_batch_semantics_verified"],
            field="global_raw_batch_semantics_verified",
        ):
            raise ValueError("raw batch semantics must remain unverified")
        if not _as_bool(
            record["primary_group_key_band_qualified"],
            field="primary_group_key_band_qualified",
        ):
            raise ValueError("primary grouping must remain band-qualified")
    audit_table_payload(frame, name="raw_batch_overlap_audit.csv")


def validate_disjoint_sensitivity(
    frame: pd.DataFrame, *, config: dict[str, Any]
) -> None:
    transfers = _configured_transfers(config)
    cell_count_columns = {
        f"{band}_{config['class_contract']['class_names'][str(category)].lower()}_batch_count"
        for transfer in transfers.values()
        for band in transfer["source_bands"]
        for category in sorted(CLASS_CODES)
    }
    required = {
        "transfer_id",
        "role",
        "source_bands",
        "target_band",
        "declared_status",
        "computed_status",
        "minimum_required_batches_per_source_band_class",
        "minimum_observed_batches_per_source_band_class",
        "source_record_count_before",
        "source_record_count_after",
        "target_record_count_before",
        "target_record_count_after",
        "source_rows_removed",
        "target_rows_removed",
        "reason",
        "model_id",
        "analysis_scope",
        "target_record_count",
        "target_batch_count",
        "recall_uav",
        "recall_weather",
        *set(config["metrics"]["required_metrics"]),
        *cell_count_columns,
    }
    _require_exact_columns(frame, required, "disjoint_sensitivity.csv")
    if frame.empty or set(frame["transfer_id"].astype(str)) != set(transfers):
        raise ValueError("disjoint sensitivity transfer coverage is incomplete")
    metric_columns = [
        "target_record_count",
        "target_batch_count",
        "recall_uav",
        "recall_weather",
        *config["metrics"]["required_metrics"],
    ]
    sources = _configured_sources(config)
    minimum = int(
        config["acceptance_contract"][
            "minimum_batch_count_per_source_band_and_class_for_reportable_disjoint_sensitivity"
        ]
    )
    for transfer_id, rows in frame.groupby("transfer_id", observed=True):
        transfer = transfers[str(transfer_id)]
        declared = str(transfer["raw_batch_disjoint_sensitivity"]["status"])
        if len(rows) != (len(EXPECTED_MODEL_IDS) if declared == "REPORTABLE" else 1):
            raise ValueError(f"{transfer_id}: unexpected disjoint row count")
        if set(rows["analysis_scope"].astype(str)) != {DISJOINT_ANALYSIS_SCOPE}:
            raise ValueError(f"{transfer_id}: disjoint analysis scope changed")
        if set(rows["role"].astype(str)) != {str(transfer["role"])}:
            raise ValueError(f"{transfer_id}: disjoint role changed")
        if set(rows["source_bands"].astype(str)) != {
            "+".join(transfer["source_bands"])
        }:
            raise ValueError(f"{transfer_id}: disjoint source bands changed")
        if set(rows["target_band"].astype(str)) != {str(transfer["target_band"])}:
            raise ValueError(f"{transfer_id}: disjoint target band changed")
        if set(rows["declared_status"].astype(str)) != {declared} or set(
            rows["computed_status"].astype(str)
        ) != {declared}:
            raise ValueError(f"{transfer_id}: disjoint status differs from config")
        expected_reason = str(
            transfer["raw_batch_disjoint_sensitivity"].get("reason", "")
        )
        if set(rows["reason"].fillna("").astype(str)) != {expected_reason}:
            raise ValueError(f"{transfer_id}: disjoint reason differs from config")
        if not rows["target_rows_removed"].fillna(0).astype(int).eq(0).all():
            raise ValueError(f"{transfer_id}: disjoint sensitivity removed target rows")
        if not rows["target_record_count_before"].astype(int).equals(
            rows["target_record_count_after"].astype(int)
        ):
            raise ValueError(f"{transfer_id}: disjoint target population changed")
        expected_source_before = sum(
            int(sources[str(source_id)]["expected_analysis_coverage"]["record_count"])
            for source_id in transfer["source_ids"]
        )
        target = sources[str(transfer["target_source_id"])]
        expected_target_count = int(target["expected_analysis_coverage"]["record_count"])
        expected_target_batches = int(
            target["expected_analysis_coverage"]["unique_batch_count"]
        )
        if not rows["source_record_count_before"].astype(int).eq(
            expected_source_before
        ).all():
            raise ValueError(f"{transfer_id}: disjoint source population changed")
        if not rows["target_record_count_before"].astype(int).eq(
            expected_target_count
        ).all():
            raise ValueError(f"{transfer_id}: disjoint target coverage changed")
        source_after = rows["source_record_count_after"].astype(int)
        source_removed = rows["source_rows_removed"].astype(int)
        if (source_after < 0).any() or (source_removed < 0).any() or not (
            source_after + source_removed
        ).eq(expected_source_before).all():
            raise ValueError(f"{transfer_id}: disjoint source removal accounting changed")
        observed = int(rows["minimum_observed_batches_per_source_band_class"].iloc[0])
        if not rows["minimum_observed_batches_per_source_band_class"].astype(int).eq(
            observed
        ).all():
            raise ValueError(f"{transfer_id}: inconsistent disjoint support rows")
        transfer_cell_count_columns = [
            f"{band}_{config['class_contract']['class_names'][str(category)].lower()}_batch_count"
            for band in transfer["source_bands"]
            for category in sorted(CLASS_CODES)
        ]
        cell_counts = rows[transfer_cell_count_columns].astype(int)
        if (cell_counts < 0).any(axis=None) or int(cell_counts.iloc[0].min()) != observed:
            raise ValueError(f"{transfer_id}: disjoint batch-cell support changed")
        if not cell_counts.eq(cell_counts.iloc[0], axis="columns").all(axis=None):
            raise ValueError(f"{transfer_id}: inconsistent disjoint batch-cell rows")
        if not rows["minimum_required_batches_per_source_band_class"].astype(int).eq(
            minimum
        ).all():
            raise ValueError(f"{transfer_id}: disjoint minimum batch gate changed")
        if declared == "REPORTABLE":
            if observed < minimum:
                raise ValueError(f"{transfer_id}: reportable disjoint support is too small")
            model_rows = rows.loc[rows["model_id"].notna()]
            if set(model_rows["model_id"].astype(str)) != EXPECTED_MODEL_IDS:
                raise ValueError(f"{transfer_id}: reportable disjoint models are incomplete")
            if model_rows["model_id"].duplicated().any():
                raise ValueError(f"{transfer_id}: duplicate reportable disjoint model")
            _ensure_finite(model_rows, metric_columns, "disjoint_sensitivity.csv")
            if not model_rows["target_record_count"].astype(int).eq(
                expected_target_count
            ).all() or not model_rows["target_batch_count"].astype(int).eq(
                expected_target_batches
            ).all():
                raise ValueError(f"{transfer_id}: disjoint metric target coverage changed")
            _validate_metric_bounds(model_rows, name="disjoint_sensitivity.csv")
        else:
            if observed >= minimum:
                raise ValueError(f"{transfer_id}: NOT_IDENTIFIABLE status is inconsistent")
            if rows["model_id"].notna().any():
                raise ValueError(f"{transfer_id}: non-identifiable sensitivity has model metrics")
            if rows[metric_columns].notna().any(axis=None):
                raise ValueError(f"{transfer_id}: non-identifiable sensitivity imputed metrics")
    audit_table_payload(frame, name="disjoint_sensitivity.csv")


def validate_gate_decision(
    gate: dict[str, Any],
    *,
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    stopping = config["stopping_rule"]
    primary_transfers = list(map(str, stopping["applies_to_transfer_ids"]))
    _require_exact_keys(
        gate,
        {
            "status",
            "model_id",
            "uses_unrounded_values",
            "all_locked_primary_targets_pass",
            "targets_consumed",
            "target_bands_consumed",
            "same_target_reuse_for_future_confirmatory_comparison_allowed",
            "conditions",
        },
        "gate_decision.json",
    )
    _require_equal(
        gate,
        {
            "model_id": PRIMARY_MODEL_ID,
            "uses_unrounded_values": True,
            "targets_consumed": True,
            "target_bands_consumed": stopping["target_bands_consumed_by_this_run"],
            "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
        },
        "gate decision",
    )
    conditions = gate.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("gate decision conditions must be a list")
    aggregate_lookup = aggregate.set_index(["transfer_id", "model_id"])
    paired_lookup = bootstrap.loc[
        bootstrap["comparison"].eq(PAIR_COMPARISON)
        & bootstrap["analysis_scope"].eq(PRIMARY_ANALYSIS_SCOPE)
    ].set_index("transfer_id")
    thresholds = {
        PRIMARY_METRIC: float(
            stopping["target_batch_class_macro_accuracy_strictly_greater_than"]
        ),
        "target_batch_class_recall_uav": float(
            stopping["each_target_class_batch_recall_strictly_greater_than"]
        ),
        "target_batch_class_recall_weather": float(
            stopping["each_target_class_batch_recall_strictly_greater_than"]
        ),
        "paired_logistic_minus_dummy_ci_lower_95": float(
            stopping["paired_logistic_minus_dummy_ci_lower_strictly_greater_than"]
        ),
    }
    expected_values: dict[tuple[str, str], float] = {}
    for transfer_id in primary_transfers:
        row = aggregate_lookup.loc[(transfer_id, PRIMARY_MODEL_ID)]
        expected_values[(transfer_id, PRIMARY_METRIC)] = float(row[PRIMARY_METRIC])
        expected_values[(transfer_id, "target_batch_class_recall_uav")] = float(
            row["target_batch_class_recall_uav"]
        )
        expected_values[(transfer_id, "target_batch_class_recall_weather")] = float(
            row["target_batch_class_recall_weather"]
        )
        expected_values[(transfer_id, "paired_logistic_minus_dummy_ci_lower_95")] = float(
            paired_lookup.loc[transfer_id, "ci_lower_95"]
        )
    actual_keys: set[tuple[str, str]] = set()
    pass_values: list[bool] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("gate condition entries must be objects")
        _require_exact_keys(
            condition,
            {
                "transfer_id",
                "metric",
                "operator",
                "threshold",
                "observed_value",
                "passed",
            },
            "gate decision condition",
        )
        key = (str(condition.get("transfer_id", "")), str(condition.get("metric", "")))
        if key in actual_keys or key not in expected_values:
            raise ValueError(f"unexpected or duplicate gate condition {key}")
        actual_keys.add(key)
        if condition.get("operator") != ">":
            raise ValueError("gate conditions must use a strict greater-than operator")
        if not _same_float(condition.get("threshold"), thresholds[key[1]]):
            raise ValueError(f"{key}: gate threshold changed")
        if not _same_float(condition.get("observed_value"), expected_values[key]):
            raise ValueError(f"{key}: gate raw value does not match aggregate/bootstrap")
        expected_pass = expected_values[key] > thresholds[key[1]]
        actual_pass = _as_bool(condition.get("passed"), field=f"{key}.passed")
        if actual_pass != expected_pass:
            raise ValueError(f"{key}: gate pass flag does not match its raw value")
        pass_values.append(actual_pass)
    if actual_keys != set(expected_values):
        raise ValueError("gate decision does not contain all eight frozen conditions")
    all_pass = all(pass_values)
    if _as_bool(
        gate.get("all_locked_primary_targets_pass"),
        field="all_locked_primary_targets_pass",
    ) != all_pass:
        raise ValueError("global gate flag does not match condition conjunction")
    expected_status = "PASS_ENGINEERING_ONLY" if all_pass else "FAIL_STOP"
    if gate.get("status") != expected_status:
        raise ValueError("gate status does not match the frozen stop/continue rule")
    if bool(summary.get("primary_gate_passed")) != all_pass:
        raise ValueError("source summary gate flag differs from gate decision")
    audit_json_payload(gate, name="gate_decision.json")


def validate_ancillary_tables(
    *,
    confusion: pd.DataFrame,
    feature_definitions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    claims: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    _require_exact_columns(
        confusion,
        {
            "transfer_id",
            "role",
            "analysis_scope",
            "model_id",
            "band",
            "true_category_code",
            "true_category",
            "predicted_category_code",
            "predicted_category",
            "confusion_type",
            "value",
        },
        "confusion_matrices.csv",
    )
    if confusion.empty or set(confusion["confusion_type"].astype(str)) != {
        "row_count",
        "target_batch_class_row_normalized",
    }:
        raise ValueError("confusion matrix types are incomplete")
    _ensure_finite(confusion, ["value"], "confusion_matrices.csv")
    if np.any(confusion["value"].to_numpy(float) < 0):
        raise ValueError("confusion matrices contain negative values")
    confusion_key = [
        "transfer_id",
        "analysis_scope",
        "model_id",
        "true_category_code",
        "predicted_category_code",
        "confusion_type",
    ]
    if confusion.duplicated(confusion_key).any():
        raise ValueError("confusion matrices contain duplicate cells")
    expected_confusion = {
        (
            transfer_id,
            PRIMARY_ANALYSIS_SCOPE,
            model_id,
            true_code,
            predicted_code,
            confusion_type,
        )
        for transfer_id in _configured_transfers(config)
        for model_id in EXPECTED_MODEL_IDS
        for true_code in CLASS_CODES
        for predicted_code in CLASS_CODES
        for confusion_type in ("row_count", "target_batch_class_row_normalized")
    }
    actual_confusion = {
        (
            str(record["transfer_id"]),
            str(record["analysis_scope"]),
            str(record["model_id"]),
            int(record["true_category_code"]),
            int(record["predicted_category_code"]),
            str(record["confusion_type"]),
        )
        for record in confusion.to_dict(orient="records")
    }
    if actual_confusion != expected_confusion:
        raise ValueError("confusion matrix transfer/model coverage is incomplete")
    transfers = _configured_transfers(config)
    sources = _configured_sources(config)
    for (transfer_id, model_id, true_code, confusion_type), rows in confusion.groupby(
        ["transfer_id", "model_id", "true_category_code", "confusion_type"],
        observed=True,
    ):
        transfer = transfers[str(transfer_id)]
        if set(rows["role"].astype(str)) != {str(transfer["role"])} or set(
            rows["band"].astype(str)
        ) != {str(transfer["target_band"])}:
            raise ValueError(f"{transfer_id}/{model_id}: confusion identity changed")
        expected_name = config["class_contract"]["class_names"][str(int(true_code))]
        if set(rows["true_category"].astype(str)) != {expected_name}:
            raise ValueError(f"{transfer_id}/{model_id}: confusion class name changed")
        value_sum = float(rows["value"].sum())
        if confusion_type == "target_batch_class_row_normalized":
            if not _same_float(value_sum, 1.0):
                raise ValueError(f"{transfer_id}/{model_id}: normalized confusion row changed")
        else:
            target = sources[str(transfer["target_source_id"])]
            expected_count = int(
                target["expected_analysis_coverage"]["classes"][str(int(true_code))][
                    "record_count"
                ]
            )
            if not _same_float(value_sum, expected_count):
                raise ValueError(f"{transfer_id}/{model_id}: confusion row count changed")
    for record in confusion.to_dict(orient="records"):
        predicted_code = int(record["predicted_category_code"])
        expected_name = config["class_contract"]["class_names"][
            str(predicted_code)
        ]
        if str(record["predicted_category"]) != expected_name:
            raise ValueError("confusion predicted class name changed")

    _require_exact_columns(
        feature_definitions,
        {
            "representation",
            "feature",
            "feature_set_id",
            "physical_frequency_unit",
            "per_record_normalized",
            "metadata_as_feature",
        },
        "feature_definitions.csv",
    )
    if feature_definitions.empty or feature_definitions.duplicated(
        ["representation", "feature"]
    ).any():
        raise ValueError("feature definitions are empty or duplicated")
    if set(feature_definitions["feature_set_id"].astype(str)) != {
        config["feature_contract"]["feature_set_id"]
    }:
        raise ValueError("feature definition set id changed")
    expected_features = {
        (representation, str(feature))
        for representation, schema in config["feature_contract"][
            "feature_schemas"
        ].items()
        for feature in schema["feature_names"]
    }
    actual_features = set(
        zip(
            feature_definitions["representation"].astype(str),
            feature_definitions["feature"].astype(str),
            strict=True,
        )
    )
    if actual_features != expected_features:
        raise ValueError("feature definition schema changed")
    for record in feature_definitions.to_dict(orient="records"):
        expected_unit = (
            "cycles/sample"
            if "cycles_per_sample" in str(record["feature"])
            else "not_applicable"
        )
        if str(record["physical_frequency_unit"]) != expected_unit:
            raise ValueError("feature physical-frequency unit changed")
        if not _as_bool(record["per_record_normalized"], field="per_record_normalized"):
            raise ValueError("features must remain per-record normalized")
        if _as_bool(record["metadata_as_feature"], field="metadata_as_feature"):
            raise ValueError("metadata must not be used as a feature")

    _require_exact_columns(
        feature_importance,
        {
            "transfer_id",
            "role",
            "analysis_scope",
            "source_bands",
            "target_band",
            "model_id",
            "feature",
            "importance",
            "importance_type",
            "fit_scope",
        },
        "feature_importance.csv",
    )
    if feature_importance.empty:
        raise ValueError("feature importance table must not be empty")
    _ensure_finite(feature_importance, ["importance"], "feature_importance.csv")
    if (feature_importance["importance"].to_numpy(float) < 0).any():
        raise ValueError("feature importance contains a negative value")
    if set(feature_importance["analysis_scope"].astype(str)) - {
        PRIMARY_ANALYSIS_SCOPE,
        DISJOINT_ANALYSIS_SCOPE,
    }:
        raise ValueError("feature importance contains an unexpected analysis scope")
    if set(feature_importance["model_id"].astype(str)) - {
        "logistic_batch_balanced",
        "random_forest_batch_balanced",
    }:
        raise ValueError("feature importance contains an unsupported model")
    learned_models = {
        "logistic_batch_balanced": "mean_abs_standardized_multiclass_coefficient",
        "random_forest_batch_balanced": "mean_decrease_impurity",
    }
    expected_importance = {
        (transfer_id, scope, model_id, str(feature))
        for transfer_id, scope in _expected_scope_pairs(config)
        for model_id in learned_models
        for feature in config["feature_contract"]["feature_schemas"][
            transfers[transfer_id]["representation"]
        ]["feature_names"]
    }
    actual_importance: set[tuple[str, str, str, str]] = set()
    for record in feature_importance.to_dict(orient="records"):
        key = (
            str(record["transfer_id"]),
            str(record["analysis_scope"]),
            str(record["model_id"]),
            str(record["feature"]),
        )
        if key in actual_importance:
            raise ValueError(f"duplicate feature importance row: {key}")
        actual_importance.add(key)
        transfer = transfers[key[0]]
        _require_equal(
            record,
            {
                "role": transfer["role"],
                "source_bands": "+".join(transfer["source_bands"]),
                "target_band": transfer["target_band"],
                "importance_type": learned_models[key[2]],
                "fit_scope": "source_bands_only",
            },
            f"feature importance {key}",
        )
    if actual_importance != expected_importance:
        raise ValueError("feature importance transfer/scope/feature coverage is incomplete")

    _require_exact_columns(
        claims, {"claim", "allowed", "reason"}, "claim_boundaries.csv"
    )
    if claims.empty or claims["claim"].duplicated().any():
        raise ValueError("claim boundaries are empty or duplicated")
    allowed = claims["allowed"].map(lambda value: _as_bool(value, field="allowed"))
    if int(allowed.sum()) != 1 or len(claims) != 1 + len(
        config["claim_contract"]["forbidden_claims"]
    ):
        raise ValueError("claim boundary allow/deny coverage changed")
    expected_claims = {
        str(config["claim_contract"]["allowed_claim"]): True,
        **{
            str(claim): False
            for claim in config["claim_contract"]["forbidden_claims"]
        },
    }
    actual_claims = {
        str(record["claim"]): _as_bool(record["allowed"], field="allowed")
        for record in claims.to_dict(orient="records")
    }
    if actual_claims != expected_claims:
        raise ValueError("claim boundary text or status changed")
    expected_reasons = {
        str(config["claim_contract"]["allowed_claim"]): (
            "fixed source-only fit and released-band held-out evaluation"
        ),
        **{
            str(claim): "outside the frozen dataset-internal band-held-out contract"
            for claim in config["claim_contract"]["forbidden_claims"]
        },
    }
    actual_reasons = {
        str(record["claim"]): str(record["reason"])
        for record in claims.to_dict(orient="records")
    }
    if actual_reasons != expected_reasons:
        raise ValueError("claim boundary reasons changed")

    for name, frame in (
        ("confusion_matrices.csv", confusion),
        ("feature_definitions.csv", feature_definitions),
        ("feature_importance.csv", feature_importance),
        ("claim_boundaries.csv", claims),
    ):
        audit_table_payload(frame, name=name)


def validate_source_run(
    source_dir: Path,
    *,
    config_path: Path = CONFIG_PATH,
    implementation_path: Path = IMPLEMENTATION_PATH,
    commit_validator: CommitValidator = validate_commit_bindings,
    consumption_record_provider: ConsumptionRecordProvider = _load_consumption_record,
) -> dict[str, Any]:
    source_dir = resolve_path(source_dir)
    config_path = resolve_path(config_path)
    implementation_path = resolve_path(implementation_path)
    validate_source_inventory(source_dir)
    config = _load_json(config_path)
    validate_config_contract(config)
    _configured_models(config)
    _configured_transfers(config)

    summary = _load_json(source_dir / "summary.json")
    model_fit_manifest = _load_json(source_dir / "model_fit_manifest.json")
    gate_decision = _load_json(source_dir / "gate_decision.json")
    for name, payload in (
        ("summary.json", summary),
        ("model_fit_manifest.json", model_fit_manifest),
        ("gate_decision.json", gate_decision),
    ):
        audit_json_payload(payload, name=name)
    validate_summary(
        summary,
        source_dir=source_dir,
        config=config,
        config_path=config_path,
        implementation_path=implementation_path,
        commit_validator=commit_validator,
    )
    consumption_record = consumption_record_provider(
        _formal_consumption_record_path(config)
    )
    validate_consumption_record(
        consumption_record,
        summary=summary,
        summary_path=source_dir / "summary.json",
        config=config,
    )

    tables = {
        name: _load_csv(source_dir / name) for name in SOURCE_TABLE_FILES
    }
    for name, frame in tables.items():
        audit_table_payload(frame, name=name)
    validate_transfer_coverage(
        tables["transfer_coverage.csv"], config=config, summary=summary
    )
    weight_audit = tables["training_weight_audit.csv"]
    validate_training_weight_audit(weight_audit, config=config)
    validate_model_fit_manifest(
        model_fit_manifest,
        summary=summary,
        config=config,
        weight_audit=weight_audit,
        weight_audit_sha256=sha256_file(source_dir / "training_weight_audit.csv"),
    )
    aggregate = tables["aggregate_metrics.csv"]
    validate_aggregate_metrics(aggregate, config=config)
    validate_target_batch_metrics(
        tables["target_batch_class_metrics.csv"], aggregate=aggregate
    )
    disjoint = tables["disjoint_sensitivity.csv"]
    validate_disjoint_sensitivity(disjoint, config=config)
    bootstrap = tables["bootstrap_intervals.csv"]
    validate_bootstrap_intervals(
        bootstrap,
        aggregate=aggregate,
        disjoint=disjoint,
        config=config,
    )
    validate_raw_batch_overlap(tables["raw_batch_overlap_audit.csv"], config=config)
    validate_gate_decision(
        gate_decision,
        aggregate=aggregate,
        bootstrap=bootstrap,
        summary=summary,
        config=config,
    )
    validate_ancillary_tables(
        confusion=tables["confusion_matrices.csv"],
        feature_definitions=tables["feature_definitions.csv"],
        feature_importance=tables["feature_importance.csv"],
        claims=tables["claim_boundaries.csv"],
        config=config,
    )
    report = (source_dir / "REPORT.md").read_text(encoding="utf-8")
    audit_text_payload(report, name="REPORT.md")
    if f"Status: `{EXPECTED_STATUS}`" not in report:
        raise ValueError("REPORT.md status does not match the formal source summary")

    return {
        "config": config,
        "summary": summary,
        "model_fit_manifest": model_fit_manifest,
        "gate_decision": gate_decision,
        "consumption_record": consumption_record,
        "tables": tables,
    }


def audit_publication(output_dir: Path) -> None:
    expected = set(PUBLISHED_FILES) | {"evidence_manifest.json"}
    actual: set[str] = set()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("publication must not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(output_dir).as_posix())
    if actual != expected:
        raise ValueError(
            "publication file set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    for name in SOURCE_ROOT_FILES:
        path = output_dir / name
        if path.suffix == ".json":
            audit_json_payload(_load_json(path), name=name)
        else:
            audit_text_payload(path.read_text(encoding="utf-8"), name=name)
    for name in PUBLISHED_TABLE_FILES:
        audit_table_payload(_load_csv(output_dir / "tables" / name), name=name)
    audit_json_payload(
        _load_json(output_dir / "evidence_manifest.json"),
        name="evidence_manifest.json",
    )


def _published_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(PUBLISHED_FILES):
        path = output_dir / relative
        records.append(
            {
                "file": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def build_evidence(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = CONFIG_PATH,
    implementation_path: Path = IMPLEMENTATION_PATH,
    commit_validator: CommitValidator = validate_commit_bindings,
    consumption_record_provider: ConsumptionRecordProvider = _load_consumption_record,
    builder_commit_provider: Callable[[], str] = current_commit,
) -> dict[str, Any]:
    source_dir = resolve_path(source_dir)
    output_dir = resolve_path(output_dir)
    config_path = resolve_path(config_path)
    implementation_path = resolve_path(implementation_path)
    frozen_output_dir = DEFAULT_OUTPUT_DIR.resolve()
    if output_dir == PROJECT_ROOT or source_dir == output_dir:
        raise ValueError("evidence output must be separate from project and source roots")
    if source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("evidence output and source experiment must not be nested")
    if output_dir != frozen_output_dir and (
        PROJECT_ROOT in output_dir.parents or output_dir in PROJECT_ROOT.parents
    ):
        raise ValueError(
            "non-default evidence output must be outside the project tree and its ancestors"
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"evidence destination already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(
            f"evidence destination parent does not exist: {output_dir.parent}"
        )

    validated = validate_source_run(
        source_dir,
        config_path=config_path,
        implementation_path=implementation_path,
        commit_validator=commit_validator,
        consumption_record_provider=consumption_record_provider,
    )
    summary = validated["summary"]
    consumption_record = validated["consumption_record"]
    builder_commit = builder_commit_provider()
    if not HEX_COMMIT.fullmatch(builder_commit):
        raise ValueError("evidence builder commit is invalid")
    builder_path = Path(__file__).resolve()
    builder_sha256 = sha256_file(builder_path)
    if not commit_validator(builder_commit, {builder_path: builder_sha256}):
        raise ValueError("evidence builder is not bound to its recorded commit")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        tables_dir = staging / "tables"
        tables_dir.mkdir(parents=True)
        for name in SOURCE_ROOT_FILES:
            shutil.copyfile(source_dir / name, staging / name)
        for name in PUBLISHED_TABLE_FILES:
            shutil.copyfile(source_dir / name, tables_dir / name)

        records = _published_records(staging)
        source_hashes = [
            {
                "source_id": record["source_id"],
                "sha256": record["sha256"],
                "analysis_record_count": int(record["analysis_record_count"]),
            }
            for record in sorted(
                summary["source_files"], key=lambda item: str(item["source_id"])
            )
        ]
        manifest = {
            "status": "FROZEN_SANITIZED_CROSS_BAND_AGGREGATE_EVIDENCE",
            "evidence_id": "lat_mricd_cross_band_transfer_v1",
            "source_experiment": summary["experiment_id"],
            "source_implementation_commit": summary["implementation_commit"],
            "evidence_builder_commit": builder_commit,
            "config_sha256": summary["config_sha256"],
            "source_implementation_sha256": summary["implementation_sha256"],
            "evidence_builder_sha256": builder_sha256,
            "source_summary_sha256": sha256_file(source_dir / "summary.json"),
            "source_model_fit_manifest_sha256": sha256_file(
                source_dir / "model_fit_manifest.json"
            ),
            "source_gate_decision_sha256": sha256_file(
                source_dir / "gate_decision.json"
            ),
            "source_consumption_record_payload_sha256": sha256_bytes(
                (
                    json.dumps(
                        consumption_record,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            ),
            "source_files": source_hashes,
            "source_file_count": len(SOURCE_FILES),
            "source_only_contract_validated": True,
            "sealed_run_consumption_validated": True,
            "primary_gate_recomputed_from_unrounded_values": True,
            "bootstrap_minimum_valid_replicates_validated": True,
            "sample_level_predictions_included": False,
            "raw_data_included": False,
            "raw_or_absolute_paths_included": False,
            "per_sample_weights_included": False,
            "model_checkpoints_included": False,
            "excluded_artifact_policies": EXCLUDED_ARTIFACT_POLICIES,
            "published_file_count_excluding_manifest": len(records),
            "files": records,
            "claim_scope": summary["claim_scope"],
        }
        (staging / "evidence_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_publication(staging)

        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"evidence destination appeared during build: {output_dir}"
            )
        staging.rename(output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parse_args()
    manifest = build_evidence()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
