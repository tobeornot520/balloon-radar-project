from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score

import scripts.run_lat_mricd_cross_band_transfer_v1 as transfer_runner


def _frozen_config() -> dict[str, object]:
    return json.loads(
        transfer_runner.DEFAULT_CONFIG.read_text(encoding="utf-8")
    )


def _write_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_frozen_config_is_accepted_and_contract_mutations_are_rejected(
    tmp_path: Path,
) -> None:
    config = transfer_runner.load_config(transfer_runner.DEFAULT_CONFIG)
    assert config["status"] == "PREREGISTERED_NOT_RUN"
    assert transfer_runner.expected_model_fit_count(config) == 27
    assert transfer_runner.FROZEN_MODEL_FIT_COUNT == 27
    assert transfer_runner.expected_bootstrap_interval_count(config) == 36
    assert transfer_runner.FROZEN_BOOTSTRAP_INTERVAL_COUNT == 36

    mutations = [
        ("training_contract", "target_rows_used_for_fit", True),
        (
            "training_contract",
            "target_labels_used_for_fit_threshold_calibration_or_model_selection",
            True,
        ),
        (
            "bootstrap_contract",
            "confidence_interval_conditioning",
            "unconditional",
        ),
        ("acceptance_contract", "target_x_allowed", True),
        ("acceptance_contract", "formal_output_overwrite_allowed", True),
        ("acceptance_contract", "record_before_target_load", False),
    ]
    for index, (section, field, value) in enumerate(mutations):
        mutated = copy.deepcopy(config)
        mutated[section][field] = value
        with pytest.raises(ValueError):
            transfer_runner.load_config(
                _write_config(tmp_path / f"mutated-{index}.json", mutated)
            )


def test_aggregate_source_whitelist_rejects_hash_path_and_duplicate_mutations() -> None:
    config = _frozen_config()
    transfer_runner.validate_aggregate_sources(config)

    bad_hash = copy.deepcopy(config)
    bad_hash["aggregate_sources"][0]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract mismatch"):
        transfer_runner.validate_aggregate_sources(bad_hash)

    detail_path = copy.deepcopy(config)
    detail_path["aggregate_sources"][0]["relative_path"] = (
        "HRRP/X波段/data_hrrp_X_detail.mat"
    )
    with pytest.raises(ValueError, match="five-file whitelist"):
        transfer_runner.validate_aggregate_sources(detail_path)

    duplicate = copy.deepcopy(config)
    duplicate["aggregate_sources"][1]["relative_path"] = duplicate[
        "aggregate_sources"
    ][0]["relative_path"]
    with pytest.raises(ValueError, match="paths must be unique"):
        transfer_runner.validate_aggregate_sources(duplicate)


def test_cross_band_weights_equalize_class_band_and_batch_cells() -> None:
    labels = np.asarray([1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3])
    bands = np.asarray([1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2])
    batches = np.asarray([10, 11, 11, 20, 20, 30, 30, 40, 41, 41, 41, 41])

    weights = transfer_runner.cross_band_training_weights(labels, bands, batches)
    np.testing.assert_allclose(weights.mean(), 1.0)
    frame = np.rec.fromarrays(
        [labels, bands, batches, weights],
        names=["label", "band", "batch", "weight"],
    )

    class_totals = [frame.weight[frame.label == code].sum() for code in (1, 3)]
    np.testing.assert_allclose(class_totals, class_totals[0])
    for code in (1, 3):
        band_totals = [
            frame.weight[(frame.label == code) & (frame.band == band)].sum()
            for band in np.unique(frame.band[frame.label == code])
        ]
        np.testing.assert_allclose(band_totals, band_totals[0])
        for band in np.unique(frame.band[frame.label == code]):
            selected = (frame.label == code) & (frame.band == band)
            cell_totals = [
                frame.weight[selected & (frame.batch == batch)].sum()
                for batch in np.unique(frame.batch[selected])
            ]
            np.testing.assert_allclose(cell_totals, cell_totals[0])


def test_binary_metrics_use_uav_probability_as_the_positive_column() -> None:
    labels = np.asarray([1, 1, 3, 3])
    predictions = np.asarray([1, 3, 3, 1])
    probabilities = np.asarray(
        [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]]
    )

    observed = transfer_runner.binary_classification_metrics(
        labels, predictions, probabilities
    )

    assert observed["roc_auc"] == pytest.approx(
        roc_auc_score(labels == 1, probabilities[:, 0])
    )
    assert observed["binary_log_loss"] == pytest.approx(
        log_loss(labels, probabilities, labels=[1, 3])
    )


def test_narrow_preparation_keeps_only_shared_uav_models_and_weather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = np.asarray(
        [
            [2, 1, 1, 10],
            [2, 1, 2, 11],
            [2, 1, 3, 12],
            [2, 1, 4, 13],
            [2, 2, 7, 14],
            [2, 3, 9, 15],
        ],
        dtype=np.float64,
    )
    matrix = np.zeros((len(metadata), 1028), dtype=np.float64)
    matrix[:, :4] = metadata
    source = {
        "source_id": "synthetic_x",
        "relative_path": "synthetic.mat",
        "expected_sha256": "0" * 64,
        "representation": "Narrow",
        "band_code": 2,
        "band": "X",
        "expected_record_count": len(matrix),
        "expected_column_count": matrix.shape[1],
    }
    class_contract = _frozen_config()["class_contract"]
    monkeypatch.setattr(
        transfer_runner,
        "reconstruct_narrow_iq",
        lambda selected: np.zeros((len(selected), 8), dtype=np.complex128),
    )
    monkeypatch.setattr(
        transfer_runner,
        "extract_narrow_features",
        lambda iq: pd.DataFrame({"feature": np.arange(len(iq), dtype=float)}),
    )

    prepared = transfer_runner.prepare_aggregate_source(
        source, matrix, class_contract
    )

    assert prepared.frame["source_row_index"].tolist() == [0, 1, 2, 5]
    assert set(prepared.frame["category_code"]) == {1, 3}
    assert set(prepared.frame.loc[prepared.frame["category_code"].eq(1), "model_code"]) == {
        1,
        2,
        3,
    }


class _RecordingEstimator:
    def __init__(self) -> None:
        self.classes_ = np.asarray([1, 3])

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> "_RecordingEstimator":
        self.fit_features = np.array(features, copy=True)
        self.fit_labels = np.array(labels, copy=True)
        self.fit_weights = np.array(sample_weight, copy=True)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.prediction_features = np.array(features, copy=True)
        return np.full((len(features), 2), 0.5)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.ones(len(features), dtype=np.int64)


def test_source_only_fit_never_passes_target_rows_to_estimator() -> None:
    source_features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    source_labels = np.asarray([1, 1, 3, 3])
    source_weights = np.ones(4)
    target_features = np.asarray([[1.0e12], [-1.0e12]])
    estimator = _RecordingEstimator()

    predicted, probabilities = transfer_runner.fit_source_only_model(
        estimator,
        source_features,
        source_labels,
        source_weights,
        target_features,
    )

    np.testing.assert_array_equal(estimator.fit_features, source_features)
    np.testing.assert_array_equal(estimator.fit_labels, source_labels)
    np.testing.assert_array_equal(estimator.fit_weights, source_weights)
    np.testing.assert_array_equal(estimator.prediction_features, target_features)
    np.testing.assert_array_equal(predicted, [1, 1])
    np.testing.assert_allclose(probabilities, 0.5)


def test_source_only_scaler_and_classifier_ignore_target_mutation() -> None:
    model_spec = copy.deepcopy(_frozen_config()["models"][1])
    source_features = np.asarray(
        [[-2.0, 1.0], [-1.0, 2.0], [1.0, 2.0], [2.0, 1.0]]
    )
    source_labels = np.asarray([1, 1, 3, 3])
    source_weights = np.ones(4)
    models = [
        transfer_runner.make_model(model_spec, random_state=7),
        transfer_runner.make_model(model_spec, random_state=7),
    ]

    transfer_runner.fit_source_only_model(
        models[0],
        source_features,
        source_labels,
        source_weights,
        np.asarray([[10.0, 10.0], [-10.0, -10.0]]),
    )
    transfer_runner.fit_source_only_model(
        models[1],
        source_features,
        source_labels,
        source_weights,
        np.asarray([[1.0e9, -1.0e9], [-1.0e9, 1.0e9]]),
    )

    np.testing.assert_allclose(
        models[0].named_steps["scaler"].mean_,
        models[1].named_steps["scaler"].mean_,
    )
    np.testing.assert_allclose(
        models[0].named_steps["classifier"].coef_,
        models[1].named_steps["classifier"].coef_,
    )


def test_weighted_dummy_prior_uses_only_weighted_source_prevalence() -> None:
    model = DummyClassifier(strategy="prior")
    source_features = np.arange(3, dtype=float).reshape(-1, 1)
    source_labels = np.asarray([1, 1, 3])
    source_weights = np.asarray([0.375, 0.375, 2.25])

    predicted, probabilities = transfer_runner.fit_source_only_model(
        model,
        source_features,
        source_labels,
        source_weights,
        np.asarray([[999.0], [-999.0]]),
    )

    np.testing.assert_allclose(probabilities, [[0.25, 0.75], [0.25, 0.75]])
    np.testing.assert_array_equal(predicted, [3, 3])


def _prepared_band(
    source_id: str,
    band: str,
    band_code: int,
    full_batches: set[int],
    analysis_batches: list[int],
) -> transfer_runner.PreparedBand:
    categories = np.tile([1, 3], len(analysis_batches))
    batches = np.repeat(analysis_batches, 2)
    feature = np.where(categories == 1, -1.0, 1.0) + batches * 1.0e-4
    frame = pd.DataFrame(
        {
            "source_row_index": np.arange(len(categories)),
            "source_id": source_id,
            "representation": "Narrow",
            "band_code": band_code,
            "band": band,
            "category_code": categories,
            "model_code": np.where(categories == 1, 1, 9),
            "batch_code": batches,
            "feature_a": feature,
            "feature_b": feature**2,
        }
    )
    return transfer_runner.PreparedBand(
        source_id=source_id,
        relative_path="synthetic.mat",
        representation="Narrow",
        band_code=band_code,
        band=band,
        frame=frame,
        feature_names=("feature_a", "feature_b"),
        full_batch_codes=frozenset(full_batches),
        sha256=(str(band_code) * 64)[:64],
    )


def test_raw_batch_overlap_separates_full_release_and_analysis_subset() -> None:
    source = _prepared_band("narrow_x", "X", 2, set(range(1, 178)), list(range(1, 177)))
    target = _prepared_band("narrow_s", "S", 1, set(range(1, 178)), list(range(1, 177)))
    transfer = {
        "transfer_id": "narrow_x_to_s_shared_binary",
        "role": "locked_primary",
        "representation": "Narrow",
        "source_bands": ["X"],
        "target_band": "S",
        "raw_batch_disjoint_sensitivity": {"expected_overlap_code_count": 176},
    }
    overlap_contract = {
        "expected_narrow_full_release_pairwise_overlap_counts": {"S_X": 177},
        "expected_narrow_shared_model_analysis_pairwise_overlap_counts": {
            "S_X": 176
        },
    }

    row = transfer_runner.raw_batch_overlap_row(
        transfer,
        [source],
        target,
        source.frame,
        target.frame,
        overlap_contract=overlap_contract,
    )

    assert row["full_release_overlap_code_count"] == 177
    assert row["analysis_subset_overlap_code_count"] == 176


def test_disjoint_sensitivity_deletes_source_only_and_checks_every_band_class() -> None:
    x = _prepared_band("narrow_x", "X", 2, {1, 2, 3, 4}, [1, 2, 3, 4])
    ku = _prepared_band("narrow_ku", "Ku", 3, {1, 2, 3, 4}, [1, 2, 3, 4])
    source = pd.concat([x.frame, ku.frame], ignore_index=True)
    target = _prepared_band("narrow_s", "S", 1, {1, 99}, [1, 99]).frame
    original_target = target.copy(deep=True)
    transfer = {
        "transfer_id": "synthetic",
        "role": "secondary",
        "source_bands": ["X", "Ku"],
        "target_band": "S",
        "raw_batch_disjoint_sensitivity": {"status": "REPORTABLE"},
    }

    filtered, row = transfer_runner.apply_raw_batch_disjoint_sensitivity(
        transfer,
        source,
        target,
        minimum_batches_per_source_band_class=3,
    )

    assert 1 not in set(filtered["batch_code"])
    pd.testing.assert_frame_equal(target, original_target)
    assert row["target_rows_removed"] == 0
    assert row["minimum_observed_batches_per_source_band_class"] == 3
    assert row["computed_status"] == "REPORTABLE"
    for band in ("X", "Ku"):
        assert row[f"{band}_uav_batch_count"] == 3
        assert row[f"{band}_weather_batch_count"] == 3


def _bootstrap_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in ("dummy_prior", "logistic_batch_balanced"):
        for batch_code in (10, 11, 12, 13):
            for category_code in (1, 3):
                if model_id == "logistic_batch_balanced":
                    predicted = category_code if batch_code != 13 else 4 - category_code
                else:
                    predicted = 1
                rows.append(
                    {
                        "transfer_id": "synthetic",
                        "role": "locked_primary",
                        "analysis_scope": "band_qualified_primary",
                        "model_id": model_id,
                        "representation": "Narrow",
                        "band": "S",
                        "batch_code": batch_code,
                        "category_code": category_code,
                        "predicted_category_code": predicted,
                    }
                )
    return pd.DataFrame(rows)


def test_target_batch_bootstrap_is_deterministic_and_uses_identical_paired_draws() -> None:
    predictions = _bootstrap_predictions()
    observed = transfer_runner.target_batch_bootstrap_intervals(
        predictions, replicates=250, random_state=17
    )
    repeated = transfer_runner.target_batch_bootstrap_intervals(
        predictions, replicates=250, random_state=17
    )
    pd.testing.assert_frame_equal(observed, repeated)

    batch_codes = np.asarray([10, 11, 12, 13])
    seed = transfer_runner.bootstrap_seed(
        17, "band_qualified_primary", "synthetic"
    )
    assert seed == int(
        hashlib.sha256(b"17|band_qualified_primary|synthetic").hexdigest()[:8],
        16,
    )
    draws = np.random.default_rng(seed).integers(
        0, len(batch_codes), size=(250, len(batch_codes))
    )
    logistic = predictions.loc[
        predictions["model_id"].eq("logistic_batch_balanced")
    ]
    dummy = predictions.loc[predictions["model_id"].eq("dummy_prior")]
    expected = transfer_runner._bootstrap_values(
        transfer_runner._cell_accuracy_matrix(logistic, batch_codes), draws
    ) - transfer_runner._bootstrap_values(
        transfer_runner._cell_accuracy_matrix(dummy, batch_codes), draws
    )
    valid = expected[np.isfinite(expected)]
    paired = observed.loc[
        observed["comparison"].eq(
            "logistic_batch_balanced_minus_dummy_prior"
        )
    ].iloc[0]
    assert paired["ci_lower_95"] == pytest.approx(np.quantile(valid, 0.025))
    assert paired["ci_upper_95"] == pytest.approx(np.quantile(valid, 0.975))
    assert bool(paired["identical_paired_draws"])
    assert paired["inference_scope"] == "conditional_on_each_fixed_source_fit"


def test_bootstrap_discards_missing_class_replicates_and_reports_counts() -> None:
    predictions = _bootstrap_predictions()
    predictions = predictions.loc[
        predictions["category_code"].eq(1)
        | predictions["batch_code"].eq(10)
    ].reset_index(drop=True)

    intervals = transfer_runner.target_batch_bootstrap_intervals(
        predictions, replicates=400, random_state=29
    )

    assert intervals["requested_replicates"].eq(400).all()
    assert intervals["valid_replicates"].between(1, 399).all()
    assert (
        intervals["valid_replicates"] + intervals["discarded_replicates"]
    ).eq(400).all()
    assert intervals["valid_replicates"].nunique() == 1


def test_formal_repository_check_rejects_a_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transfer_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="?? uncommitted-file\n"),
    )
    with pytest.raises(ValueError, match="clean Git worktree"):
        transfer_runner.validate_pre_result_repository_state(
            transfer_runner.DEFAULT_CONFIG, _frozen_config()
        )


def test_formal_entrypoint_exposes_no_overwrite_or_validation_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(inspect.signature(transfer_runner.run_cross_band_transfer).parameters) == {
        "config_path",
        "dataset_root",
    }
    for forbidden_option in ("--overwrite", "--output-dir"):
        monkeypatch.setattr(sys, "argv", ["cross-band-runner", forbidden_option])
        with pytest.raises(SystemExit):
            transfer_runner.parse_args()


def test_direct_and_module_cli_help_are_available_without_bypass_options() -> None:
    runner_path = (
        transfer_runner.PROJECT_ROOT
        / "scripts/run_lat_mricd_cross_band_transfer_v1.py"
    )
    commands = (
        [sys.executable, str(runner_path), "--help"],
        [
            sys.executable,
            "-m",
            "scripts.run_lat_mricd_cross_band_transfer_v1",
            "--help",
        ],
    )

    for command in commands:
        completed = subprocess.run(
            command,
            cwd=transfer_runner.PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--config" in completed.stdout
        assert "--dataset-root" in completed.stdout
        assert "--output-dir" not in completed.stdout
        assert "--overwrite" not in completed.stdout


def test_formal_output_path_is_frozen_and_separate_from_protected_roots(
    tmp_path: Path,
) -> None:
    config = _frozen_config()
    output_dir = transfer_runner.resolve_formal_output_path(config)
    transfer_runner.validate_formal_output_location(
        output_dir, transfer_runner.resolve_path(config["dataset_root"])
    )

    mutated = copy.deepcopy(config)
    mutated["output_dir"] = str(tmp_path / "alternate-output")
    with pytest.raises(ValueError, match="differs from frozen config"):
        transfer_runner.resolve_formal_output_path(mutated)


def test_formal_output_location_rejects_ancestors_and_descendants(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    for output_dir in (dataset_root, dataset_root / "nested", tmp_path):
        with pytest.raises(ValueError, match="raw dataset root"):
            transfer_runner.validate_formal_output_location(
                output_dir, dataset_root
            )

    git_path = transfer_runner.PROJECT_ROOT / ".git"
    for output_dir in (git_path, git_path / "nested"):
        with pytest.raises(ValueError, match="Git metadata"):
            transfer_runner.validate_formal_output_location(
                output_dir, dataset_root
            )

    for output_dir in (
        transfer_runner.PROJECT_ROOT,
        transfer_runner.PROJECT_ROOT.parent,
    ):
        with pytest.raises(ValueError, match="project root or its ancestor"):
            transfer_runner.validate_formal_output_location(
                output_dir, dataset_root
            )


def test_formal_run_rejects_an_existing_output_directory_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()
    monkeypatch.setattr(
        transfer_runner, "load_config", lambda path: _synthetic_run_config()
    )
    monkeypatch.setattr(
        transfer_runner,
        "resolve_formal_output_path",
        lambda config: output_dir,
    )
    monkeypatch.setattr(
        transfer_runner,
        "resolve_consumption_record_path",
        lambda config: pytest.fail("consumption path must not be resolved"),
    )
    with pytest.raises(FileExistsError, match="must not already exist"):
        transfer_runner.run_cross_band_transfer(
            dataset_root=tmp_path / "unused-data",
        )


def test_clean_repository_validation_precedes_consumption_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _synthetic_run_config()
    calls: list[str] = []

    def load_test_config(path: Path) -> dict[str, object]:
        calls.append("load_config")
        return config

    def resolve_test_output(observed_config: dict[str, object]) -> Path:
        calls.append("resolve_formal_output_path")
        return tmp_path / "output"

    def validate_repository(
        config_path: Path, observed_config: dict[str, object]
    ) -> str:
        calls.append("validate_pre_result_repository_state")
        return "a" * 40

    def resolve_consumption(observed_config: dict[str, object]) -> Path:
        calls.append("resolve_consumption_record_path")
        return tmp_path / "consumed.json"

    def write_consumption(
        path: Path, payload: dict[str, object], *, exclusive: bool
    ) -> None:
        calls.append(f"write_{payload['status']}")

    def reject_source_load(
        observed_config: dict[str, object], dataset_root: Path
    ) -> dict[str, transfer_runner.PreparedBand]:
        calls.append("load_aggregate_sources")
        raise RuntimeError("synthetic stop after reservation")

    monkeypatch.setattr(transfer_runner, "load_config", load_test_config)
    monkeypatch.setattr(
        transfer_runner, "resolve_formal_output_path", resolve_test_output
    )
    monkeypatch.setattr(
        transfer_runner,
        "validate_pre_result_repository_state",
        validate_repository,
    )
    monkeypatch.setattr(
        transfer_runner, "resolve_consumption_record_path", resolve_consumption
    )
    monkeypatch.setattr(
        transfer_runner, "_write_consumption_record", write_consumption
    )
    monkeypatch.setattr(
        transfer_runner, "load_aggregate_sources", reject_source_load
    )

    with pytest.raises(RuntimeError, match="synthetic stop"):
        transfer_runner.run_cross_band_transfer(
            dataset_root=tmp_path / "dataset",
        )

    assert calls == [
        "load_config",
        "resolve_formal_output_path",
        "validate_pre_result_repository_state",
        "resolve_consumption_record_path",
        "write_RESERVED",
        "load_aggregate_sources",
        "write_FAILED_OR_INTERRUPTED_RUN_CONSUMED",
    ]


def test_parent_directory_fsync_always_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    expected_flags = transfer_runner.os.O_RDONLY | getattr(
        transfer_runner.os, "O_DIRECTORY", 0
    )

    def open_directory(path: Path, flags: int) -> int:
        calls.append(("open", (path, flags)))
        return 73

    def fail_fsync(descriptor: int) -> None:
        calls.append(("fsync", descriptor))
        raise OSError("synthetic directory fsync failure")

    monkeypatch.setattr(transfer_runner.os, "open", open_directory)
    monkeypatch.setattr(transfer_runner.os, "fsync", fail_fsync)
    monkeypatch.setattr(
        transfer_runner.os,
        "close",
        lambda descriptor: calls.append(("close", descriptor)),
    )

    record_path = tmp_path / "record.json"
    with pytest.raises(OSError, match="directory fsync failure"):
        transfer_runner._fsync_parent_directory(record_path)
    assert calls == [
        ("open", (record_path.parent, expected_flags)),
        ("fsync", 73),
        ("close", 73),
    ]


def test_consumption_record_syncs_parent_after_create_and_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_path = tmp_path / "record.json"
    synced: list[Path] = []
    monkeypatch.setattr(
        transfer_runner,
        "_fsync_parent_directory",
        lambda path: synced.append(path),
    )

    transfer_runner._write_consumption_record(
        record_path, {"status": "RESERVED"}, exclusive=True
    )
    transfer_runner._write_consumption_record(
        record_path, {"status": "COMPLETED"}, exclusive=False
    )

    assert synced == [record_path, record_path]
    assert json.loads(record_path.read_text(encoding="utf-8")) == {
        "status": "COMPLETED"
    }


def test_output_allowlist_and_claim_boundaries_are_explicit() -> None:
    expected = {
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
        "model_fit_manifest.json",
        "gate_decision.json",
        "summary.json",
        "REPORT.md",
    }
    assert set(transfer_runner.OUTPUT_FILES) == expected
    assert not any(
        token in name.lower()
        for name in transfer_runner.OUTPUT_FILES
        for token in ("prediction", "oof", "checkpoint")
    )

    claims = transfer_runner.claim_boundaries_table(_frozen_config())
    forbidden = claims.loc[~claims["allowed"].astype(bool), "claim"].str.lower().str.cat(
        sep=" "
    )
    for marker in (
        "physical",
        "same-event",
        "unseen",
        "h/v",
        "balloon",
        "causal",
        "tian",
    ):
        assert marker in forbidden


def _synthetic_run_config() -> dict[str, object]:
    config = copy.deepcopy(_frozen_config())
    config["transfers"] = [
        {
            "transfer_id": "narrow_x_to_s_shared_binary",
            "representation": "Narrow",
            "source_ids": ["narrow_x"],
            "source_bands": ["X"],
            "target_source_id": "narrow_s",
            "target_band": "S",
            "role": "locked_primary",
            "uav_model_codes": [1, 2, 3],
            "raw_batch_disjoint_sensitivity": {
                "expected_overlap_code_count": 0,
                "status": "REPORTABLE",
            },
        },
        {
            "transfer_id": "narrow_x_to_ku_shared_binary",
            "representation": "Narrow",
            "source_ids": ["narrow_x"],
            "source_bands": ["X"],
            "target_source_id": "narrow_ku",
            "target_band": "Ku",
            "role": "locked_primary",
            "uav_model_codes": [1, 2, 3],
            "raw_batch_disjoint_sensitivity": {
                "expected_overlap_code_count": 0,
                "status": "REPORTABLE",
            },
        },
    ]
    config["models"][2]["n_estimators"] = 5
    config["bootstrap_replicates"] = 40
    config["bootstrap_contract"]["minimum_valid_replicates"] = 1
    config["raw_batch_overlap_audit"][
        "expected_narrow_full_release_pairwise_overlap_counts"
    ] = {"S_X": 0, "X_Ku": 0}
    config["raw_batch_overlap_audit"][
        "expected_narrow_shared_model_analysis_pairwise_overlap_counts"
    ] = {"S_X": 0, "X_Ku": 0}
    config["stopping_rule"]["applies_to_transfer_ids"] = [
        "narrow_x_to_s_shared_binary",
        "narrow_x_to_ku_shared_binary",
    ]
    return config


def test_synthetic_end_to_end_writes_only_deterministic_aggregate_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _synthetic_run_config()
    sources = {
        "narrow_x": _prepared_band("narrow_x", "X", 2, {1, 2, 3, 4}, [1, 2, 3, 4]),
        "narrow_s": _prepared_band(
            "narrow_s", "S", 1, {11, 12, 13, 14}, [11, 12, 13, 14]
        ),
        "narrow_ku": _prepared_band(
            "narrow_ku", "Ku", 3, {21, 22, 23, 24}, [21, 22, 23, 24]
        ),
    }
    monkeypatch.setattr(transfer_runner, "load_config", lambda path: config)
    consumption_records = [
        tmp_path / "run-a-consumed.json",
        tmp_path / "run-b-consumed.json",
    ]
    record_iterator = iter(consumption_records)
    active_records: list[Path] = []

    def resolve_test_consumption_record(observed_config: dict[str, object]) -> Path:
        path = next(record_iterator)
        active_records.append(path)
        return path

    def load_synthetic_sources(
        observed_config: dict[str, object], dataset_root: Path
    ) -> dict[str, transfer_runner.PreparedBand]:
        reserved = json.loads(active_records[-1].read_text(encoding="utf-8"))
        assert reserved["status"] == "RESERVED"
        assert reserved["summary_sha256"] is None
        return sources

    monkeypatch.setattr(
        transfer_runner,
        "resolve_consumption_record_path",
        resolve_test_consumption_record,
    )
    monkeypatch.setattr(
        transfer_runner,
        "validate_pre_result_repository_state",
        lambda config_path, observed_config: "a" * 40,
    )
    monkeypatch.setattr(
        transfer_runner,
        "load_aggregate_sources",
        load_synthetic_sources,
    )

    outputs = [tmp_path / "run-a", tmp_path / "run-b"]
    output_iterator = iter(outputs)
    monkeypatch.setattr(
        transfer_runner,
        "resolve_formal_output_path",
        lambda observed_config: next(output_iterator),
    )
    summaries = [
        transfer_runner.run_cross_band_transfer(
            dataset_root=tmp_path / "unused-data",
        )
        for _ in outputs
    ]

    for output in outputs:
        assert {path.name for path in output.iterdir()} == set(
            transfer_runner.OUTPUT_FILES
        )
        assert not (output / "target_predictions.csv").exists()
    assert summaries[0]["output_sha256"] == summaries[1]["output_sha256"]
    assert summaries[0]["sealed_run_consumption_enforced"] is True
    assert summaries[0]["formal_output_overwrite_allowed"] is False
    for name in transfer_runner.OUTPUT_FILES:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    for consumption_record, output in zip(
        consumption_records, outputs, strict=True
    ):
        consumed = json.loads(consumption_record.read_text(encoding="utf-8"))
        assert consumed["status"] == "COMPLETED_AND_TARGETS_CONSUMED"
        assert consumed["summary_sha256"] == transfer_runner.sha256_file(
            output / "summary.json"
        )
        assert consumed["target_bands_consumed"] == ["S", "Ku"]
        assert not any("path" in key for key in consumed)

    gate = json.loads((outputs[0] / "gate_decision.json").read_text())
    assert set(gate) == {
        "status",
        "model_id",
        "uses_unrounded_values",
        "all_locked_primary_targets_pass",
        "targets_consumed",
        "target_bands_consumed",
        "same_target_reuse_for_future_confirmatory_comparison_allowed",
        "conditions",
    }
    assert len(gate["conditions"]) == 8
    assert all(condition["operator"] == ">" for condition in gate["conditions"])

    manifest = json.loads(
        (outputs[0] / "model_fit_manifest.json").read_text()
    )
    assert manifest["fit_count"] == 12
    assert manifest["sealed_run_consumption_enforced"] is True
    assert manifest["formal_output_overwrite_allowed"] is False
    assert all(
        fit["fit_scope"] == "source_bands_only"
        and not fit["target_rows_used_for_fit"]
        and not fit[
            "target_labels_used_for_fit_threshold_calibration_or_model_selection"
        ]
        for fit in manifest["transfers"]
    )


def test_synthetic_run_rejects_too_few_valid_bootstrap_replicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _synthetic_run_config()
    config["bootstrap_contract"]["minimum_valid_replicates"] = 40
    sources = {
        "narrow_x": _prepared_band("narrow_x", "X", 2, {1, 2, 3, 4}, [1, 2, 3, 4]),
        "narrow_s": _prepared_band(
            "narrow_s", "S", 1, {11, 12, 13, 14}, [11, 12, 13, 14]
        ),
        "narrow_ku": _prepared_band(
            "narrow_ku", "Ku", 3, {21, 22, 23, 24}, [21, 22, 23, 24]
        ),
    }
    for source_id, only_weather_batch in (("narrow_s", 11), ("narrow_ku", 21)):
        frame = sources[source_id].frame
        sources[source_id] = transfer_runner.PreparedBand(
            **{
                **sources[source_id].__dict__,
                "frame": frame.loc[
                    frame["category_code"].eq(1)
                    | frame["batch_code"].eq(only_weather_batch)
                ].reset_index(drop=True),
            }
        )
    monkeypatch.setattr(transfer_runner, "load_config", lambda path: config)
    consumption_record = tmp_path / "failed-run-consumed.json"
    monkeypatch.setattr(
        transfer_runner,
        "resolve_consumption_record_path",
        lambda observed_config: consumption_record,
    )
    monkeypatch.setattr(
        transfer_runner,
        "validate_pre_result_repository_state",
        lambda config_path, observed_config: "a" * 40,
    )
    monkeypatch.setattr(
        transfer_runner,
        "load_aggregate_sources",
        lambda observed_config, dataset_root: sources,
    )
    monkeypatch.setattr(
        transfer_runner,
        "resolve_formal_output_path",
        lambda observed_config: tmp_path / "rejected-run",
    )

    with pytest.raises(ValueError, match="too few valid target-batch bootstrap"):
        transfer_runner.run_cross_band_transfer(
            dataset_root=tmp_path / "unused-data",
        )
    failed = json.loads(consumption_record.read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED_OR_INTERRUPTED_RUN_CONSUMED"
    assert failed["summary_sha256"] is None
    assert failed["failure_type"] == "ValueError"

    with pytest.raises(FileExistsError, match="consumption record already exists"):
        transfer_runner.run_cross_band_transfer(
            dataset_root=tmp_path / "unused-data",
        )
