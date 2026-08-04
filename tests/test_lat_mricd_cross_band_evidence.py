from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.build_lat_mricd_cross_band_evidence_v1 import (
    DISJOINT_ANALYSIS_SCOPE,
    EXPECTED_MODEL_IDS,
    EXPECTED_STATUS,
    PAIR_COMPARISON,
    PAIR_METRIC,
    PRIMARY_ANALYSIS_SCOPE,
    PRIMARY_METRIC,
    PRIMARY_MODEL_ID,
    PUBLISHED_FILES,
    SOURCE_FILES,
    bootstrap_seed,
    build_evidence,
    parse_args,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = PROJECT_ROOT / "configs/lat_mricd_cross_band_transfer_v1.json"
FAKE_COMMIT = "a" * 40
BUILDER_COMMIT = "b" * 40


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )


def model_recall_values(model_id: str) -> tuple[float, float, float]:
    if model_id == "dummy_prior":
        uav, weather, log_loss = 1.0, 0.0, 0.6931471805599453
    elif model_id == PRIMARY_MODEL_ID:
        uav, weather, log_loss = 0.70, 0.70, 0.55
    else:
        uav, weather, log_loss = 0.65, 0.65, 0.60
    return uav, weather, log_loss


def target_tables_and_metrics(
    identity: dict[str, Any],
    model_id: str,
    target_source: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    coverage = target_source["expected_analysis_coverage"]
    batch_count = int(coverage["unique_batch_count"])
    recalls = dict(
        zip((1, 3), model_recall_values(model_id)[:2], strict=True)
    )
    cell_rows: list[dict[str, Any]] = []
    cells_by_batch: dict[int, list[dict[str, Any]]] = {
        batch_code: [] for batch_code in range(1, batch_count + 1)
    }
    for category_code in (1, 3):
        class_coverage = coverage["classes"][str(category_code)]
        class_batch_count = int(class_coverage["batch_count"])
        class_record_count = int(class_coverage["record_count"])
        if category_code == 1:
            batch_codes = list(range(1, class_batch_count + 1))
        else:
            batch_codes = list(
                range(batch_count - class_batch_count + 1, batch_count + 1)
            )
        quotient, remainder = divmod(class_record_count, class_batch_count)
        for index, batch_code in enumerate(batch_codes):
            record_count = quotient + (index < remainder)
            row = {
                **identity,
                "analysis_scope": PRIMARY_ANALYSIS_SCOPE,
                "model_id": model_id,
                "batch_code": batch_code,
                "category_code": category_code,
                "category": "UAV" if category_code == 1 else "weather",
                "record_count": int(record_count),
                "accuracy": recalls[category_code],
                "evaluation_unit": "target_batch_class_cell",
            }
            cell_rows.append(row)
            cells_by_batch[batch_code].append(row)

    batch_rows: list[dict[str, Any]] = []
    for batch_code, cells in cells_by_batch.items():
        record_count = sum(int(cell["record_count"]) for cell in cells)
        accuracy = sum(
            int(cell["record_count"]) * float(cell["accuracy"]) for cell in cells
        ) / record_count
        batch_rows.append(
            {
                **identity,
                "analysis_scope": PRIMARY_ANALYSIS_SCOPE,
                "model_id": model_id,
                "batch_code": batch_code,
                "category_code": "all",
                "category": "all",
                "record_count": record_count,
                "accuracy": accuracy,
                "evaluation_unit": "target_batch",
            }
        )

    batches = pd.DataFrame(batch_rows)
    cells = pd.DataFrame(cell_rows)
    class_means = cells.groupby("category_code", observed=True)["accuracy"].mean()
    worst = cells.sort_values(
        ["accuracy", "record_count", "batch_code"],
        ascending=[True, True, True],
    ).iloc[0]
    uav, weather, log_loss = model_recall_values(model_id)
    total_records = int(coverage["record_count"])
    pooled_accuracy = sum(
        int(coverage["classes"][str(code)]["record_count"]) * recalls[code]
        for code in (1, 3)
    ) / total_records
    values: dict[str, float | int] = {
        "target_record_count": total_records,
        "target_batch_count": batch_count,
        "pooled_accuracy": pooled_accuracy,
        "pooled_balanced_accuracy": (uav + weather) / 2,
        "pooled_macro_f1": (uav + weather) / 2,
        "binary_log_loss": log_loss,
        "roc_auc": 0.5 if model_id == "dummy_prior" else (uav + weather) / 2,
        "recall_uav": uav,
        "recall_weather": weather,
        "target_batch_macro_accuracy": float(batches["accuracy"].mean()),
        "target_batch_accuracy_p10": float(batches["accuracy"].quantile(0.10)),
        "worst_target_batch_accuracy": float(batches["accuracy"].min()),
        "target_batch_class_macro_accuracy": float(class_means.mean()),
        "target_batch_class_recall_uav": float(class_means.loc[1]),
        "target_batch_class_recall_weather": float(class_means.loc[3]),
        "target_batch_class_cell_accuracy_p10": float(
            cells["accuracy"].quantile(0.10)
        ),
        "worst_target_batch_class_cell_accuracy": float(cells["accuracy"].min()),
        "worst_target_batch_class_cell_record_count": int(worst["record_count"]),
    }
    return [*batch_rows, *cell_rows], values


def transfer_identity(
    transfer: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    target = sources[str(transfer["target_source_id"])]
    return {
        "transfer_id": transfer["transfer_id"],
        "role": transfer["role"],
        "source_bands": "+".join(transfer["source_bands"]),
        "representation": transfer["representation"],
        "band_code": int(target["band_code"]),
        "band": transfer["target_band"],
    }


def make_source_fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    implementation_path = tmp_path / "runner.py"
    implementation_path.write_text("# synthetic runner binding\n", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    sources = {
        str(source["source_id"]): source for source in config["aggregate_sources"]
    }
    transfers = {
        str(transfer["transfer_id"]): transfer for transfer in config["transfers"]
    }
    models = sorted(EXPECTED_MODEL_IDS)
    reportable = {
        transfer_id
        for transfer_id, transfer in transfers.items()
        if transfer["raw_batch_disjoint_sensitivity"]["status"] == "REPORTABLE"
    }

    coverage_rows: list[dict[str, Any]] = []
    full_release_batches: dict[str, int] = {}
    for source_id, source in sorted(sources.items()):
        coverage = source["expected_analysis_coverage"]
        full_release_batches[source_id] = int(coverage["unique_batch_count"]) + 1
        for category_code, category in ((1, "UAV"), (3, "weather")):
            class_coverage = coverage["classes"][str(category_code)]
            coverage_rows.append(
                {
                    "source_id": source_id,
                    "representation": source["representation"],
                    "band_code": source["band_code"],
                    "band": source["band"],
                    "category_code": category_code,
                    "category": category,
                    "record_count": int(class_coverage["record_count"]),
                    "analysis_batch_count": int(class_coverage["batch_count"]),
                    "analysis_total_record_count": int(coverage["record_count"]),
                    "analysis_unique_batch_count": int(
                        coverage["unique_batch_count"]
                    ),
                    "full_release_batch_count": full_release_batches[source_id],
                    "sha256": source["expected_sha256"],
                }
            )
    write_csv(source_dir / "transfer_coverage.csv", coverage_rows)

    weight_rows: list[dict[str, Any]] = []
    for transfer_id, transfer in transfers.items():
        scopes = [PRIMARY_ANALYSIS_SCOPE]
        if transfer_id in reportable:
            scopes.append(DISJOINT_ANALYSIS_SCOPE)
        for scope in scopes:
            band_count = len(transfer["source_ids"])
            if scope == PRIMARY_ANALYSIS_SCOPE:
                scope_record_total = sum(
                    int(sources[source_id]["expected_analysis_coverage"]["record_count"])
                    for source_id in transfer["source_ids"]
                )
            else:
                scope_record_total = 6 * len(transfer["source_ids"]) * 2
            for source_id in transfer["source_ids"]:
                source = sources[source_id]
                for category_code, category in ((1, "UAV"), (3, "weather")):
                    coverage = source["expected_analysis_coverage"]["classes"][
                        str(category_code)
                    ]
                    if scope == PRIMARY_ANALYSIS_SCOPE:
                        record_count = int(coverage["record_count"])
                        batch_count = int(coverage["batch_count"])
                    else:
                        record_count = 6
                        batch_count = 3
                    total = scope_record_total / (2.0 * band_count)
                    weight_rows.append(
                        {
                            "transfer_id": transfer_id,
                            "analysis_scope": scope,
                            "source_band": source["band"],
                            "source_band_code": source["band_code"],
                            "category_code": category_code,
                            "category": category,
                            "record_count": record_count,
                            "batch_count": batch_count,
                            "total_weight": total,
                            "minimum_batch_cell_weight": total / batch_count,
                            "maximum_batch_cell_weight": total / batch_count,
                        }
                    )
    write_csv(source_dir / "training_weight_audit.csv", weight_rows)

    aggregate_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    metric_lookup: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for transfer_id, transfer in transfers.items():
        identity = transfer_identity(transfer, sources)
        target_source = sources[str(transfer["target_source_id"])]
        for model_id in models:
            target_rows, values = target_tables_and_metrics(
                identity, model_id, target_source
            )
            metric_lookup[(transfer_id, PRIMARY_ANALYSIS_SCOPE, model_id)] = values
            aggregate_rows.append(
                {
                    **identity,
                    "analysis_scope": PRIMARY_ANALYSIS_SCOPE,
                    "model_id": model_id,
                    **values,
                }
            )
            batch_rows.extend(target_rows)
            recalls = {
                1: float(values["target_batch_class_recall_uav"]),
                3: float(values["target_batch_class_recall_weather"]),
            }
            for true_code in (1, 3):
                recall = recalls[true_code]
                true_name = "UAV" if true_code == 1 else "weather"
                class_record_count = int(
                    target_source["expected_analysis_coverage"]["classes"][
                        str(true_code)
                    ]["record_count"]
                )
                for predicted_code in (1, 3):
                    predicted_name = "UAV" if predicted_code == 1 else "weather"
                    correct = true_code == predicted_code
                    for confusion_type, value in (
                        (
                            "row_count",
                            class_record_count
                            * (recall if correct else 1 - recall),
                        ),
                        (
                            "target_batch_class_row_normalized",
                            recall if correct else 1 - recall,
                        ),
                    ):
                        confusion_rows.append(
                            {
                                "transfer_id": transfer_id,
                                "role": transfer["role"],
                                "analysis_scope": PRIMARY_ANALYSIS_SCOPE,
                                "model_id": model_id,
                                "band": transfer["target_band"],
                                "true_category_code": true_code,
                                "true_category": true_name,
                                "predicted_category_code": predicted_code,
                                "predicted_category": predicted_name,
                                "confusion_type": confusion_type,
                                "value": value,
                            }
                        )
    write_csv(source_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(source_dir / "target_batch_class_metrics.csv", batch_rows)
    write_csv(source_dir / "confusion_matrices.csv", confusion_rows)

    for transfer_id in reportable:
        for model_id in models:
            metric_lookup[(transfer_id, DISJOINT_ANALYSIS_SCOPE, model_id)] = dict(
                metric_lookup[(transfer_id, PRIMARY_ANALYSIS_SCOPE, model_id)]
            )

    bootstrap_rows: list[dict[str, Any]] = []
    for transfer_id, transfer in transfers.items():
        identity = transfer_identity(transfer, sources)
        scopes = [PRIMARY_ANALYSIS_SCOPE]
        if transfer_id in reportable:
            scopes.append(DISJOINT_ANALYSIS_SCOPE)
        for scope in scopes:
            derived_seed = bootstrap_seed(config["random_state"], scope, transfer_id)
            draw_count = int(
                sources[str(transfer["target_source_id"])][
                    "expected_analysis_coverage"
                ]["unique_batch_count"]
            )
            for comparison in [*models, PAIR_COMPARISON]:
                paired = comparison == PAIR_COMPARISON
                if paired:
                    estimate = float(
                        metric_lookup[(transfer_id, scope, PRIMARY_MODEL_ID)][
                            PRIMARY_METRIC
                        ]
                    ) - float(
                        metric_lookup[(transfer_id, scope, "dummy_prior")][
                            PRIMARY_METRIC
                        ]
                    )
                    lower, upper, metric = (
                        estimate - 0.10,
                        estimate + 0.10,
                        PAIR_METRIC,
                    )
                else:
                    estimate = float(
                        metric_lookup[(transfer_id, scope, comparison)][PRIMARY_METRIC]
                    )
                    lower, upper, metric = (
                        max(0.0, estimate - 0.10),
                        min(1.0, estimate + 0.10),
                        PRIMARY_METRIC,
                    )
                bootstrap_rows.append(
                    {
                        "transfer_id": transfer_id,
                        "role": identity["role"],
                        "analysis_scope": scope,
                        "representation": identity["representation"],
                        "target_band": identity["band"],
                        "comparison": comparison,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower_95": lower,
                        "ci_upper_95": upper,
                        "requested_replicates": 2000,
                        "valid_replicates": 2000,
                        "discarded_replicates": 0,
                        "bootstrap_seed": derived_seed,
                        "draw_count_per_replicate": draw_count,
                        "resampling_unit": "representation_target_band_code_batch_code",
                        "duplicate_batch_draws_count_with_multiplicity": True,
                        "percentile_quantile_method": "linear",
                        "inference_scope": "conditional_on_each_fixed_source_fit",
                        "conditioning": "conditional_on_each_fixed_source_fit",
                        "identical_paired_draws": paired,
                    }
                )
    write_csv(source_dir / "bootstrap_intervals.csv", bootstrap_rows)

    overlap_rows: list[dict[str, Any]] = []
    disjoint_rows: list[dict[str, Any]] = []
    for transfer_id, transfer in transfers.items():
        identity = transfer_identity(transfer, sources)
        overlap = int(
            transfer["raw_batch_disjoint_sensitivity"]["expected_overlap_code_count"]
        )
        overlap_rows.append(
            {
                "transfer_id": transfer_id,
                "role": transfer["role"],
                "representation": transfer["representation"],
                "source_bands": identity["source_bands"],
                "target_band": transfer["target_band"],
                "full_release_source_batch_count": sum(
                    full_release_batches[source_id]
                    for source_id in transfer["source_ids"]
                ),
                "full_release_target_batch_count": full_release_batches[
                    str(transfer["target_source_id"])
                ],
                "full_release_overlap_code_count": overlap,
                "analysis_source_batch_count": sum(
                    int(
                        sources[source_id]["expected_analysis_coverage"][
                            "unique_batch_count"
                        ]
                    )
                    for source_id in transfer["source_ids"]
                ),
                "analysis_target_batch_count": int(
                    sources[str(transfer["target_source_id"])][
                        "expected_analysis_coverage"
                    ]["unique_batch_count"]
                ),
                "analysis_subset_overlap_code_count": overlap,
                "expected_analysis_subset_overlap_code_count": overlap,
                "full_release_overlap_codes": "",
                "analysis_subset_overlap_codes": "",
                "global_raw_batch_semantics_verified": False,
                "primary_group_key_band_qualified": True,
            }
        )
        status = transfer["raw_batch_disjoint_sensitivity"]["status"]
        source_before = sum(
            int(sources[source_id]["expected_analysis_coverage"]["record_count"])
            for source_id in transfer["source_ids"]
        )
        target_coverage = sources[str(transfer["target_source_id"])][
            "expected_analysis_coverage"
        ]
        if status == "REPORTABLE":
            source_after = sum(
                int(row["record_count"])
                for row in weight_rows
                if row["transfer_id"] == transfer_id
                and row["analysis_scope"] == DISJOINT_ANALYSIS_SCOPE
            )
            minimum_observed = 3
        else:
            source_after = max(0, source_before - 20)
            minimum_observed = 2
        cell_counts = {
            f"{band}_{category.lower()}_batch_count": minimum_observed
            for band in transfer["source_bands"]
            for category in ("UAV", "weather")
        }
        common = {
            "transfer_id": transfer_id,
            "role": transfer["role"],
            "source_bands": identity["source_bands"],
            "target_band": transfer["target_band"],
            "declared_status": status,
            "computed_status": status,
            "minimum_required_batches_per_source_band_class": 3,
            "minimum_observed_batches_per_source_band_class": minimum_observed,
            "source_record_count_before": source_before,
            "source_record_count_after": source_after,
            "target_record_count_before": int(target_coverage["record_count"]),
            "target_record_count_after": int(target_coverage["record_count"]),
            "source_rows_removed": source_before - source_after,
            "target_rows_removed": 0,
            "reason": transfer["raw_batch_disjoint_sensitivity"].get("reason", ""),
            **cell_counts,
        }
        if status == "REPORTABLE":
            for model_id in models:
                disjoint_rows.append(
                    {
                        **common,
                        "model_id": model_id,
                        "analysis_scope": DISJOINT_ANALYSIS_SCOPE,
                        **metric_lookup[
                            (transfer_id, DISJOINT_ANALYSIS_SCOPE, model_id)
                        ],
                    }
                )
        else:
            disjoint_rows.append(
                {
                    **common,
                    "model_id": None,
                    "analysis_scope": DISJOINT_ANALYSIS_SCOPE,
                    "target_record_count": None,
                    "target_batch_count": None,
                    **{field: None for field in config["metrics"]["required_metrics"]},
                }
            )
    write_csv(source_dir / "raw_batch_overlap_audit.csv", overlap_rows)
    write_csv(source_dir / "disjoint_sensitivity.csv", disjoint_rows)

    feature_definition_rows = [
            {
                "representation": representation,
                "feature": feature,
                "feature_set_id": config["feature_contract"]["feature_set_id"],
                "physical_frequency_unit": (
                    "cycles/sample"
                    if "cycles_per_sample" in feature
                    else "not_applicable"
                ),
                "per_record_normalized": True,
                "metadata_as_feature": False,
            }
            for representation, schema in config["feature_contract"][
                "feature_schemas"
            ].items()
            for feature in schema["feature_names"]
        ]
    write_csv(source_dir / "feature_definitions.csv", feature_definition_rows)

    importance_rows: list[dict[str, Any]] = []
    importance_types = {
        "logistic_batch_balanced": "mean_abs_standardized_multiclass_coefficient",
        "random_forest_batch_balanced": "mean_decrease_impurity",
    }
    for transfer_id, transfer in transfers.items():
        scopes = [PRIMARY_ANALYSIS_SCOPE]
        if transfer_id in reportable:
            scopes.append(DISJOINT_ANALYSIS_SCOPE)
        features = config["feature_contract"]["feature_schemas"][
            transfer["representation"]
        ]["feature_names"]
        for scope in scopes:
            for model_id, importance_type in importance_types.items():
                for feature_index, feature in enumerate(features, 1):
                    importance_rows.append(
                        {
                            "transfer_id": transfer_id,
                            "role": transfer["role"],
                            "analysis_scope": scope,
                            "source_bands": "+".join(transfer["source_bands"]),
                            "target_band": transfer["target_band"],
                            "model_id": model_id,
                            "feature": feature,
                            "importance": feature_index / (10.0 * len(features)),
                            "importance_type": importance_type,
                            "fit_scope": "source_bands_only",
                        }
                    )
    write_csv(source_dir / "feature_importance.csv", importance_rows)
    claims = [
        {
            "claim": config["claim_contract"]["allowed_claim"],
            "allowed": True,
            "reason": "fixed source-only fit and released-band held-out evaluation",
        }
    ]
    claims.extend(
        {
            "claim": claim,
            "allowed": False,
            "reason": "outside the frozen dataset-internal band-held-out contract",
        }
        for claim in config["claim_contract"]["forbidden_claims"]
    )
    write_csv(source_dir / "claim_boundaries.csv", claims)

    primary_transfers = list(config["stopping_rule"]["applies_to_transfer_ids"])
    paired_lookup = {
        row["transfer_id"]: row
        for row in bootstrap_rows
        if row["comparison"] == PAIR_COMPARISON
        and row["analysis_scope"] == PRIMARY_ANALYSIS_SCOPE
    }
    conditions: list[dict[str, Any]] = []
    thresholds = {
        PRIMARY_METRIC: 0.60,
        "target_batch_class_recall_uav": 0.50,
        "target_batch_class_recall_weather": 0.50,
        "paired_logistic_minus_dummy_ci_lower_95": 0.0,
    }
    for transfer_id in primary_transfers:
        row = metric_lookup[
            (transfer_id, PRIMARY_ANALYSIS_SCOPE, PRIMARY_MODEL_ID)
        ]
        values = {
            PRIMARY_METRIC: row[PRIMARY_METRIC],
            "target_batch_class_recall_uav": row["target_batch_class_recall_uav"],
            "target_batch_class_recall_weather": row[
                "target_batch_class_recall_weather"
            ],
            "paired_logistic_minus_dummy_ci_lower_95": paired_lookup[transfer_id][
                "ci_lower_95"
            ],
        }
        for metric, value in values.items():
            conditions.append(
                {
                    "transfer_id": transfer_id,
                    "metric": metric,
                    "operator": ">",
                    "threshold": thresholds[metric],
                    "observed_value": value,
                    "passed": value > thresholds[metric],
                }
            )
    gate = {
        "status": "PASS_ENGINEERING_ONLY",
        "model_id": PRIMARY_MODEL_ID,
        "uses_unrounded_values": True,
        "all_locked_primary_targets_pass": True,
        "targets_consumed": True,
        "target_bands_consumed": config["stopping_rule"][
            "target_bands_consumed_by_this_run"
        ],
        "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
        "conditions": conditions,
    }
    write_json(source_dir / "gate_decision.json", gate)

    source_files = [
        {
            "source_id": source_id,
            "sha256": source["expected_sha256"],
            "analysis_record_count": int(
                source["expected_analysis_coverage"]["record_count"]
            ),
            "full_release_batch_count": full_release_batches[source_id],
        }
        for source_id, source in sorted(sources.items())
    ]
    config_sha256 = sha256_file(config_path)
    implementation_sha256 = sha256_file(implementation_path)
    feature_implementation_sha256 = str(
        config["feature_contract"]["implementation_sha256"]
    )
    weight_frame = pd.DataFrame(weight_rows)
    fit_rows: list[dict[str, Any]] = []
    for transfer_index, transfer in enumerate(config["transfers"]):
        transfer_id = str(transfer["transfer_id"])
        scopes = [PRIMARY_ANALYSIS_SCOPE]
        if transfer_id in reportable:
            scopes.append(DISJOINT_ANALYSIS_SCOPE)
        target_coverage = sources[str(transfer["target_source_id"])][
            "expected_analysis_coverage"
        ]
        for scope in scopes:
            selected_weights = weight_frame.loc[
                weight_frame["transfer_id"].eq(transfer_id)
                & weight_frame["analysis_scope"].eq(scope)
            ]
            for model_index, model in enumerate(config["models"]):
                fit_rows.append(
                    {
                        "transfer_id": transfer_id,
                        "role": transfer["role"],
                        "analysis_scope": scope,
                        "model_id": model["model_id"],
                        "source_ids": transfer["source_ids"],
                        "source_bands": transfer["source_bands"],
                        "target_source_id": transfer["target_source_id"],
                        "target_band": transfer["target_band"],
                        "random_state": int(config["random_state"])
                        + transfer_index * 100
                        + model_index,
                        "feature_count": int(
                            config["feature_contract"]["feature_schemas"][
                                transfer["representation"]
                            ]["feature_count"]
                        ),
                        "source_record_count": int(
                            selected_weights["record_count"].sum()
                        ),
                        "source_batch_class_cell_count": int(
                            selected_weights["batch_count"].sum()
                        ),
                        "target_record_count_evaluated": int(
                            target_coverage["record_count"]
                        ),
                        "fit_scope": "source_bands_only",
                        "scaler_fit_scope": "source_bands_only",
                        "target_rows_used_for_fit": False,
                        "target_labels_used_for_fit_threshold_calibration_or_"
                        "model_selection": False,
                        "target_statistics_used_for_scaling_or_calibration": False,
                        "sample_weights_fit_scope": "source_bands_only",
                        "probability_calibration": "none",
                        "threshold_tuning_performed": False,
                        "hyperparameter_search_performed": False,
                        "test_driven_model_selection_performed": False,
                    }
                )
    model_fit_manifest = {
        "status": "COMPLETE_SOURCE_ONLY_MODEL_FIT_MANIFEST",
        "experiment_id": config["experiment_id"],
        "implementation_commit": FAKE_COMMIT,
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
        "sample_weight_hierarchy": config["training_contract"]["sample_weight_hierarchy"],
        "training_weight_audit_sha256": sha256_file(
            source_dir / "training_weight_audit.csv"
        ),
        "models": config["models"],
        "model_ids": [model["model_id"] for model in config["models"]],
        "transfer_ids": [transfer["transfer_id"] for transfer in config["transfers"]],
        "analysis_scopes": [PRIMARY_ANALYSIS_SCOPE, DISJOINT_ANALYSIS_SCOPE],
        "fit_count": len(fit_rows),
        "transfers": fit_rows,
        "source_files": source_files,
    }
    write_json(source_dir / "model_fit_manifest.json", model_fit_manifest)

    summary = {
        "status": EXPECTED_STATUS,
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "implementation_commit": FAKE_COMMIT,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "feature_implementation_sha256": feature_implementation_sha256,
        "random_state": config["random_state"],
        "transfer_count": len(transfers),
        "model_count": len(models),
        "model_fit_count": len(fit_rows),
        "bootstrap_replicates": config["bootstrap_replicates"],
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
        "primary_gate_passed": True,
        "same_target_reuse_for_future_confirmatory_comparison_allowed": False,
        "same_target_confirmatory_reuse_allowed": False,
        "bootstrap_inference_scope": "conditional_on_each_fixed_source_fit",
        "claim_scope": "dataset-internal released-band-held-out UAV/weather transfer",
        "output_files": config["output_contract"]["files"],
    }
    (source_dir / "REPORT.md").write_text(
        f"# Synthetic Cross-Band Evidence\n\nStatus: `{EXPECTED_STATUS}`\n",
        encoding="utf-8",
    )
    summary["output_sha256"] = {
        name: sha256_file(source_dir / name)
        for name in SOURCE_FILES
        if name != "summary.json"
    }
    write_json(source_dir / "summary.json", summary)
    consumption_record_path = tmp_path / "run_consumed.json"
    write_json(
        consumption_record_path,
        {
            "schema_version": 1,
            "status": "COMPLETED_AND_TARGETS_CONSUMED",
            "experiment_id": config["experiment_id"],
            "pre_result_commit": FAKE_COMMIT,
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "target_bands_consumed": config["stopping_rule"][
                "target_bands_consumed_by_this_run"
            ],
            "summary_sha256": sha256_file(source_dir / "summary.json"),
            "sealed_run_consumption_enforced": True,
            "formal_output_overwrite_allowed": False,
            "record_created_before_target_load": True,
            "persists_on_failure": True,
        },
    )
    assert {path.name for path in source_dir.iterdir()} == set(SOURCE_FILES)
    return {
        "source_dir": source_dir,
        "config_path": config_path,
        "implementation_path": implementation_path,
        "consumption_record_path": consumption_record_path,
    }


def build_fixture(fixture: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return build_evidence(
        source_dir=fixture["source_dir"],
        output_dir=output_dir,
        config_path=fixture["config_path"],
        implementation_path=fixture["implementation_path"],
        commit_validator=lambda _commit, _bindings: True,
        consumption_record_provider=lambda _path: json.loads(
            fixture["consumption_record_path"].read_text(encoding="utf-8")
        ),
        builder_commit_provider=lambda: BUILDER_COMMIT,
    )


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def refresh_summary_output_hashes(source_dir: Path) -> None:
    summary_path = source_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_sha256"] = {
        name: sha256_file(source_dir / name)
        for name in SOURCE_FILES
        if name != "summary.json"
    }
    write_json(summary_path, summary)
    refresh_consumption_record(source_dir)


def refresh_consumption_record(source_dir: Path) -> None:
    path = source_dir.parent / "run_consumed.json"
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    record["summary_sha256"] = sha256_file(source_dir / "summary.json")
    write_json(path, record)


def test_builds_exact_sanitized_evidence_and_hash_manifest(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    output = tmp_path / "evidence"
    manifest = build_fixture(fixture, output)

    expected = set(PUBLISHED_FILES) | {"evidence_manifest.json"}
    assert set(tree_bytes(output)) == expected
    assert manifest["source_file_count"] == 15
    assert manifest["sample_level_predictions_included"] is False
    assert manifest["per_sample_weights_included"] is False
    assert manifest["sealed_run_consumption_validated"] is True
    assert len(manifest["source_consumption_record_payload_sha256"]) == 64
    assert (output / "tables/training_weight_audit.csv").is_file()
    bootstrap = pd.read_csv(
        output / "tables/bootstrap_intervals.csv", encoding="utf-8-sig"
    )
    assert len(bootstrap) == 36
    assert set(bootstrap["analysis_scope"]) == {
        PRIMARY_ANALYSIS_SCOPE,
        DISJOINT_ANALYSIS_SCOPE,
    }
    fit_manifest = json.loads(
        (output / "model_fit_manifest.json").read_text(encoding="utf-8")
    )
    assert fit_manifest["fit_count"] == 27
    assert len(fit_manifest["transfers"]) == 27
    assert not any(
        "relative_path" in path.read_text(encoding="utf-8-sig")
        for path in output.rglob("*.json")
    )
    for record in manifest["files"]:
        path = output / record["file"]
        assert path.stat().st_size == record["size_bytes"]
        assert sha256_file(path) == record["sha256"]


def test_rejects_existing_destination_without_deleting_it(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "user-owned.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination already exists"):
        build_fixture(fixture, output)

    assert marker.read_text(encoding="utf-8") == "preserve\n"


def test_rejects_nonfrozen_project_internal_destination(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the project tree"):
        build_fixture(fixture, PROJECT_ROOT / "docs/unsafe-evidence-output")


def test_cli_rejects_path_and_overwrite_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for argument in ("--output-dir", "--source-dir", "--overwrite"):
        monkeypatch.setattr("sys.argv", ["builder", argument, "ignored"])
        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.parametrize(
    "extra_name",
    ["target_predictions.csv", "raw_target.npy", "model_checkpoint.ckpt"],
)
def test_rejects_extra_prediction_raw_or_checkpoint_files(
    tmp_path: Path, extra_name: str
) -> None:
    fixture = make_source_fixture(tmp_path)
    (fixture["source_dir"] / extra_name).write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        build_fixture(fixture, tmp_path / "evidence")


def test_rejects_missing_source_file(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    (fixture["source_dir"] / "feature_importance.csv").unlink()
    with pytest.raises(ValueError, match="file set mismatch"):
        build_fixture(fixture, tmp_path / "evidence")


def test_rejects_hash_and_implementation_tampering(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    summary_path = fixture["source_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_files"][0]["sha256"] = "0" * 64
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="source summary SHA256 mismatch"):
        build_fixture(fixture, tmp_path / "evidence-hash")

    fixture = make_source_fixture(tmp_path / "fresh")
    fixture["implementation_path"].write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation hash is stale"):
        build_fixture(fixture, tmp_path / "evidence-script")


@pytest.mark.parametrize("field", ["relative_path", "absolute_path", "source_row_index"])
def test_rejects_sensitive_fields(tmp_path: Path, field: str) -> None:
    fixture = make_source_fixture(tmp_path)
    if field == "relative_path":
        summary_path = fixture["source_dir"] / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source_files"][0][field] = "Narrow/S/data.mat"
        write_json(summary_path, summary)
    else:
        table_path = fixture["source_dir"] / "aggregate_metrics.csv"
        table = pd.read_csv(table_path, encoding="utf-8-sig")
        table[field] = "/home/user/private/file.mat" if field == "absolute_path" else 1
        table.to_csv(table_path, index=False, encoding="utf-8-sig")
        refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="forbidden|must not publish source paths"):
        build_fixture(fixture, tmp_path / "evidence")


def test_rejects_per_sample_training_weights(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "training_weight_audit.csv"
    weights = pd.read_csv(path, encoding="utf-8-sig")
    weights["sample_weight"] = 1.0
    weights.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="per-sample fields|forbidden"):
        build_fixture(fixture, tmp_path / "evidence-sample-weight")


def test_rejects_nested_forbidden_json_fields_and_unbound_builder_commit(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["safe_container"] = {"relative_path": {"redacted": True}}
    write_json(path, summary)
    with pytest.raises(ValueError, match="forbidden field"):
        build_fixture(fixture, tmp_path / "evidence-nested-field")

    fixture = make_source_fixture(tmp_path / "fresh-builder-binding")
    with pytest.raises(ValueError, match="builder is not bound"):
        build_evidence(
            source_dir=fixture["source_dir"],
            output_dir=tmp_path / "evidence-builder-binding",
            config_path=fixture["config_path"],
            implementation_path=fixture["implementation_path"],
            commit_validator=lambda commit, _bindings: commit != BUILDER_COMMIT,
            consumption_record_provider=lambda _path: json.loads(
                fixture["consumption_record_path"].read_text(encoding="utf-8")
            ),
            builder_commit_provider=lambda: BUILDER_COMMIT,
        )


def test_rejects_relative_raw_path_hidden_behind_json_alias(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["provenance"] = {
        "path": "Narrow/S波段/data_narrow_S.mat",
    }
    write_json(path, summary)
    refresh_consumption_record(fixture["source_dir"])

    with pytest.raises(ValueError, match="field schema changed|sensitive path"):
        build_fixture(fixture, tmp_path / "evidence-json-alias")


def test_rejects_packed_prediction_payload_in_extra_csv_column(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "aggregate_metrics.csv"
    aggregate = pd.read_csv(path, encoding="utf-8-sig")
    aggregate["payload"] = '[{"prediction":1}]'
    aggregate.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])

    with pytest.raises(ValueError, match="column schema changed|sensitive path"):
        build_fixture(fixture, tmp_path / "evidence-csv-payload")


def test_rejects_relative_raw_path_in_report(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "REPORT.md"
    report = path.read_text(encoding="utf-8")
    path.write_text(
        report + "\nRaw source: Narrow/S波段/data_narrow_S.mat\n",
        encoding="utf-8",
    )
    refresh_summary_output_hashes(fixture["source_dir"])

    with pytest.raises(ValueError, match="REPORT.md contains.*sensitive path"):
        build_fixture(fixture, tmp_path / "evidence-report-relative-path")


def test_rejects_changed_disjoint_and_claim_reasons(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "disjoint_sensitivity.csv"
    disjoint = pd.read_csv(path, encoding="utf-8-sig")
    disjoint.loc[0, "reason"] = "modified but otherwise harmless text"
    disjoint.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="disjoint reason differs from config"):
        build_fixture(fixture, tmp_path / "evidence-disjoint-reason")

    fixture = make_source_fixture(tmp_path / "fresh-claim-reason")
    path = fixture["source_dir"] / "claim_boundaries.csv"
    claims = pd.read_csv(path, encoding="utf-8-sig")
    claims.loc[0, "reason"] = "modified but otherwise harmless text"
    claims.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="claim boundary reasons changed"):
        build_fixture(fixture, tmp_path / "evidence-claim-reason")


def test_rejects_gate_report_summary_and_bootstrap_tampering(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    gate_path = fixture["source_dir"] / "gate_decision.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["conditions"][0]["observed_value"] = 0.999
    write_json(gate_path, gate)
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="gate raw value"):
        build_fixture(fixture, tmp_path / "evidence-gate")

    fixture = make_source_fixture(tmp_path / "fresh-report")
    (fixture["source_dir"] / "REPORT.md").write_text(
        "# Wrong status\n", encoding="utf-8"
    )
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="REPORT.md status"):
        build_fixture(fixture, tmp_path / "evidence-report")

    fixture = make_source_fixture(tmp_path / "fresh-summary")
    summary_path = fixture["source_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["primary_gate_passed"] = False
    write_json(summary_path, summary)
    refresh_consumption_record(fixture["source_dir"])
    with pytest.raises(ValueError, match="summary gate flag"):
        build_fixture(fixture, tmp_path / "evidence-summary")

    fixture = make_source_fixture(tmp_path / "fresh-bootstrap")
    bootstrap_path = fixture["source_dir"] / "bootstrap_intervals.csv"
    bootstrap = pd.read_csv(bootstrap_path, encoding="utf-8-sig")
    bootstrap.loc[0, "valid_replicates"] = 1899
    bootstrap.to_csv(bootstrap_path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="valid replicate count"):
        build_fixture(fixture, tmp_path / "evidence-bootstrap")


def test_rejects_bootstrap_scope_seed_and_disjoint_estimate_tampering(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "bootstrap_intervals.csv"
    bootstrap = pd.read_csv(path, encoding="utf-8-sig")
    selected = bootstrap["analysis_scope"].eq(DISJOINT_ANALYSIS_SCOPE) & bootstrap[
        "comparison"
    ].eq(PRIMARY_MODEL_ID)
    bootstrap.loc[selected, "estimate"] += 0.01
    bootstrap.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="bootstrap estimate mismatch"):
        build_fixture(fixture, tmp_path / "evidence-disjoint-estimate")

    fixture = make_source_fixture(tmp_path / "fresh-seed")
    path = fixture["source_dir"] / "bootstrap_intervals.csv"
    bootstrap = pd.read_csv(path, encoding="utf-8-sig")
    bootstrap.loc[0, "bootstrap_seed"] += 1
    bootstrap.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="bootstrap seed changed"):
        build_fixture(fixture, tmp_path / "evidence-seed")

    fixture = make_source_fixture(tmp_path / "fresh-scope")
    path = fixture["source_dir"] / "bootstrap_intervals.csv"
    bootstrap = pd.read_csv(path, encoding="utf-8-sig").iloc[1:]
    bootstrap.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="scope/comparison coverage"):
        build_fixture(fixture, tmp_path / "evidence-scope")


def test_rejects_fit_manifest_feature_and_output_hash_tampering(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "model_fit_manifest.json"
    fit_manifest = json.loads(path.read_text(encoding="utf-8"))
    fit_manifest["transfers"][0]["target_record_count_evaluated"] += 1
    write_json(path, fit_manifest)
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="target_record_count_evaluated"):
        build_fixture(fixture, tmp_path / "evidence-fit")

    fixture = make_source_fixture(tmp_path / "fresh-feature")
    path = fixture["source_dir"] / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["feature_implementation_sha256"] = "0" * 64
    write_json(path, summary)
    with pytest.raises(ValueError, match="feature implementation hash is stale"):
        build_fixture(fixture, tmp_path / "evidence-feature")

    fixture = make_source_fixture(tmp_path / "fresh-output-hash")
    path = fixture["source_dir"] / "claim_boundaries.csv"
    path.write_text(path.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="output hash mismatch"):
        build_fixture(fixture, tmp_path / "evidence-output-hash")


def test_rejects_incomplete_or_mismatched_consumption_record(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["consumption_record_path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "RESERVED"
    write_json(path, record)
    with pytest.raises(ValueError, match="formal-run consumption record status"):
        build_fixture(fixture, tmp_path / "evidence-reserved")

    fixture = make_source_fixture(tmp_path / "fresh-consumption")
    path = fixture["consumption_record_path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["summary_sha256"] = "0" * 64
    write_json(path, record)
    with pytest.raises(ValueError, match="formal-run consumption record summary_sha256"):
        build_fixture(fixture, tmp_path / "evidence-consumption-hash")


def test_rejects_coverage_and_disjoint_source_accounting_tampering(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    path = fixture["source_dir"] / "transfer_coverage.csv"
    coverage = pd.read_csv(path, encoding="utf-8-sig")
    coverage.loc[0, "record_count"] += 1
    coverage.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="record_count changed"):
        build_fixture(fixture, tmp_path / "evidence-coverage")

    fixture = make_source_fixture(tmp_path / "fresh-disjoint")
    path = fixture["source_dir"] / "disjoint_sensitivity.csv"
    disjoint = pd.read_csv(path, encoding="utf-8-sig")
    disjoint.loc[0, "source_rows_removed"] += 1
    disjoint.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_summary_output_hashes(fixture["source_dir"])
    with pytest.raises(ValueError, match="source removal accounting"):
        build_fixture(fixture, tmp_path / "evidence-disjoint")


def test_build_is_byte_deterministic(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    output_a = tmp_path / "evidence-a"
    output_b = tmp_path / "evidence-b"
    build_fixture(fixture, output_a)
    build_fixture(fixture, output_b)
    assert tree_bytes(output_a) == tree_bytes(output_b)
