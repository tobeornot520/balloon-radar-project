#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.classification_dataset_v2 import (  # noqa: E402
    DEFAULT_DATA_ROOT as DEFAULT_CLASSIFICATION_ROOT,
    ClassificationRadarDatasetV2,
)
from datasets.detection_dataset_v2 import DetectionGeometry, _load_iq_pair  # noqa: E402
from datasets.polarimetric_detection_dataset_v2 import _recover_data_path  # noqa: E402
from features.multidomain_radar_features import (  # noqa: E402
    FEATURE_DOMAINS,
    MULTIDOMAIN_FEATURE_NAMES,
    MultiDomainFeatureConfig,
    extract_multidomain_features,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results/data_audit/dataset_v4_multifold/fold_01_manifest.csv"
)
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "results/experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a label-independent multi-domain feature catalog"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--classification-root", default=str(DEFAULT_CLASSIFICATION_ROOT)
    )
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/multidomain_feature_catalog_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return only frozen catalog fields, in the fusion contract order."""
    ordered_names = tuple(
        name
        for names in MULTIDOMAIN_FEATURE_NAMES.values()
        for name in names
    )
    return [
        name
        for name in ordered_names
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])
    ]


def standardized_effect(positive: np.ndarray, background: np.ndarray) -> float:
    positive = positive[np.isfinite(positive)]
    background = background[np.isfinite(background)]
    if len(positive) < 2 or len(background) < 2:
        return math.nan
    denominator = len(positive) + len(background) - 2
    pooled_variance = (
        (len(positive) - 1) * float(np.var(positive, ddof=1))
        + (len(background) - 1) * float(np.var(background, ddof=1))
    ) / denominator
    if pooled_variance <= 1e-24:
        return 0.0
    return float((positive.mean() - background.mean()) / np.sqrt(pooled_variance))


def univariate_separability(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    labels = frame["target_present"].to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for feature in feature_columns(frame):
        values = frame[feature].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if len(np.unique(labels[finite])) != 2:
            continue
        auc = float(roc_auc_score(labels[finite], values[finite]))
        positive = values[(labels == 1) & finite]
        background = values[(labels == 0) & finite]
        rows.append(
            {
                "feature": feature,
                "auc_target_high": auc,
                "orientation_free_auc": max(auc, 1.0 - auc),
                "cohens_d_target_minus_background": standardized_effect(
                    positive, background
                ),
                "target_mean": float(positive.mean()),
                "background_mean": float(background.mean()),
                "evidence_warning": "target_background_date_confounded",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "orientation_free_auc", ascending=False
    )


def background_group_stress_table(
    frame: pd.DataFrame,
    group_column: str = "source_file",
) -> pd.DataFrame:
    """Measure whether a pooled feature direction survives each background scan."""
    from sklearn.metrics import roc_auc_score

    target = frame.loc[frame["target_present"].eq(1)]
    background = frame.loc[frame["target_present"].eq(0)]
    rows: list[dict[str, Any]] = []
    for feature in feature_columns(frame):
        target_values = target[feature].to_numpy(dtype=np.float64)
        pooled_background = background[feature].to_numpy(dtype=np.float64)
        pooled_values = np.concatenate([target_values, pooled_background])
        pooled_labels = np.concatenate(
            [np.ones(len(target_values)), np.zeros(len(pooled_background))]
        )
        finite = np.isfinite(pooled_values)
        if len(np.unique(pooled_labels[finite])) != 2:
            continue
        pooled_auc = float(roc_auc_score(pooled_labels[finite], pooled_values[finite]))
        direction = 1.0 if pooled_auc >= 0.5 else -1.0
        group_scores: list[tuple[str, float]] = []
        for group, group_frame in background.groupby(group_column, observed=True):
            group_values = group_frame[feature].to_numpy(dtype=np.float64)
            values = direction * np.concatenate([target_values, group_values])
            labels = np.concatenate(
                [np.ones(len(target_values)), np.zeros(len(group_values))]
            )
            valid = np.isfinite(values)
            if len(np.unique(labels[valid])) == 2:
                group_scores.append(
                    (str(group), float(roc_auc_score(labels[valid], values[valid])))
                )
        if not group_scores:
            continue
        worst_group, worst_auc = min(group_scores, key=lambda item: item[1])
        scores = np.asarray([score for _, score in group_scores], dtype=np.float64)
        rows.append(
            {
                "feature": feature,
                "pooled_direction": "target_high" if direction > 0 else "target_low",
                "pooled_oriented_auc": max(pooled_auc, 1.0 - pooled_auc),
                "background_group_count": len(group_scores),
                "worst_background_group_auc": worst_auc,
                "median_background_group_auc": float(np.median(scores)),
                "best_background_group_auc": float(np.max(scores)),
                "worst_background_group": worst_group,
                "evidence_warning": "target_background_date_confounded",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["worst_background_group_auc", "median_background_group_auc"],
        ascending=False,
    )


def group_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    if len(values) < 2 or len(np.unique(groups)) < 2:
        return math.nan
    grand = float(values.mean())
    total = float(np.sum((values - grand) ** 2))
    if total <= 1e-24:
        return 0.0
    between = 0.0
    for group in np.unique(groups):
        group_values = values[groups == group]
        between += len(group_values) * float((group_values.mean() - grand) ** 2)
    return float(np.clip(between / total, 0.0, 1.0))


def group_dependency_table(
    frame: pd.DataFrame,
    group_column: str,
    label: str,
) -> pd.DataFrame:
    groups = frame[group_column].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for feature in feature_columns(frame):
        values = frame[feature].to_numpy(dtype=np.float64)
        group_means = frame.groupby(group_column, observed=True)[feature].mean()
        rows.append(
            {
                "feature": feature,
                "group_eta_squared": group_eta_squared(values, groups),
                "group_mean_min": float(group_means.min()),
                "group_mean_max": float(group_means.max()),
                "group_mean_range": float(group_means.max() - group_means.min()),
                "group_role": label,
            }
        )
    return pd.DataFrame(rows).sort_values("group_eta_squared", ascending=False)


def build_detection_catalog(
    manifest_path: Path,
    candidates: pd.DataFrame,
    config: MultiDomainFeatureConfig,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {
        "new_split",
        "class_name",
        "target_present",
        "sample_id",
        "source_file",
        "beam_layer",
        "azimuth_deg",
        "distance_m",
        "velocity_mps",
        "mat_path",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Fold 1 manifest must contain one row per physical sample")
    candidate_lookup = candidates.set_index("sample_id")
    if not candidate_lookup.index.is_unique:
        raise ValueError("frozen candidates contain duplicate sample IDs")
    missing_candidates = set(manifest["sample_id"].astype(str)) - set(candidate_lookup.index)
    if missing_candidates:
        raise ValueError(f"missing frozen candidates: {sorted(missing_candidates)[:5]}")
    geometry = DetectionGeometry()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for position, record in manifest.iterrows():
        sample_id = str(record["sample_id"])
        try:
            mat_path = _recover_data_path(str(record["mat_path"]))
            h, v = _load_iq_pair(mat_path)
            candidate = candidate_lookup.loc[sample_id]
            features = extract_multidomain_features(
                h,
                v,
                config,
                anchor_velocity_index=int(candidate["pred_velocity_index"]),
                anchor_range_index=int(candidate["pred_range_index"]),
            )
            metadata: dict[str, Any] = {
                "sample_id": sample_id,
                "split": str(record["new_split"]),
                "class_name": str(record["class_name"]),
                "target_present": int(record["target_present"]),
                "source_file": str(record["source_file"]),
                "beam_layer": int(record["beam_layer"]),
                "azimuth_deg": float(record["azimuth_deg"]),
                "distance_m": float(record["distance_m"]),
                "velocity_mps": float(record["velocity_mps"]),
                "candidate_fold": int(candidate["fold"]),
                "candidate_raw_score": float(candidate["raw_score"]),
            }
            if metadata["target_present"]:
                true_range = geometry.range_to_index(metadata["distance_m"])
                true_velocity = geometry.velocity_to_index(metadata["velocity_mps"])
                metadata["anchor_range_error_gates"] = abs(
                    int(features["rd_anchor_range_index"]) - true_range
                )
                metadata["anchor_velocity_error_bins"] = abs(
                    int(features["rd_anchor_velocity_index"]) - true_velocity
                )
            else:
                metadata["anchor_range_error_gates"] = math.nan
                metadata["anchor_velocity_error_bins"] = math.nan
            rows.append({**metadata, **features})
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": repr(exc)})
        if (position + 1) % 200 == 0 or position + 1 == len(manifest):
            print(f"detection: {position + 1}/{len(manifest)}", flush=True)
    return pd.DataFrame(rows), errors


def load_frozen_dpg_candidates(candidate_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in range(1, 7):
        path = (
            candidate_root
            / f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed42"
            / "tables/base_threshold_test_predictions.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"frozen DPG candidate table missing: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {
            "sample_id",
            "raw_score",
            "pred_range_index",
            "pred_velocity_index",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"candidate table missing columns: {sorted(missing)}")
        frame = frame.loc[:, sorted(required)].copy()
        frame["fold"] = fold
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("frozen out-of-fold candidates contain duplicate sample IDs")
    return combined


def build_classification_catalog(
    data_root: Path,
    config: MultiDomainFeatureConfig,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for split in ("train", "test"):
        dataset = ClassificationRadarDatasetV2(
            data_root=data_root,
            split=split,
            input_mode="ri4",
        )
        for index in range(len(dataset)):
            record = dataset.records[index]
            sample_id = f"{record['mat_path'].stem}__{record['sample_index']:03d}"
            try:
                item = dataset[index]
                array = item["input"].numpy()
                h = array[0].astype(np.float32) + 1j * array[1].astype(np.float32)
                v = array[2].astype(np.float32) + 1j * array[3].astype(np.float32)
                features = extract_multidomain_features(h, v, config)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "class_name": str(item["class_name"]),
                        "target_present": 1,
                        "source_file": record["mat_path"].stem,
                        "sample_index": int(record["sample_index"]),
                        **features,
                    }
                )
            except Exception as exc:
                errors.append({"sample_id": sample_id, "error": repr(exc)})
        print(f"classification {split}: {len(dataset)}", flush=True)
    return pd.DataFrame(rows), errors


def build_feature_schema(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in feature_columns(frame):
        domain = feature.split("_", 1)[0]
        rows.append(
            {
                "feature": feature,
                "domain": domain,
                "domain_description": FEATURE_DOMAINS[domain],
                "uses_class_label": False,
                "physical_units_status": (
                    "normalized_frequency"
                    if feature.startswith("tf_")
                    else "relative_or_index" if feature.startswith("polar_") else "native_or_index"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_report(summary: dict[str, Any]) -> str:
    top_auc = summary["detection"]["top_confounded_univariate_features"]
    top_stress = summary["detection"]["top_background_group_stress_features"]
    top_group = summary["detection"]["top_background_group_sensitive_features"]
    top_file = summary["classification_uav_only"]["top_file_dependent_features"]
    lines = [
        "# Multi-domain Feature Catalog V1",
        "",
        "## Scope",
        "",
        f"- Detection samples: {summary['detection']['sample_count']}",
        f"- Detection scan groups: {summary['detection']['source_group_count']}",
        f"- Long-window UAV samples: {summary['classification_uav_only']['sample_count']}",
        f"- Long-window UAV files: {summary['classification_uav_only']['source_file_count']}",
        f"- Extracted feature count: {summary['feature_count']}",
        "",
        "Detection-local time, polarimetric and time-frequency features use the",
        "frozen out-of-fold DPG candidate. Scene features and UAV-only long-window",
        "features use the strongest combined H/V power cell. Labels and truth",
        "coordinates are never used to construct features.",
        "",
        "## Current interpretation boundaries",
        "",
        "- Detection target/background labels are acquisition-date confounded.",
        "- Classification records contain UAV only and cannot support class accuracy.",
        "- Polarimetric values are relative until H/V calibration is verified.",
        "- Time-frequency frequencies are normalized until PRF and timing are supplied.",
        "",
        "## Top confounded target/background features",
        "",
        pd.DataFrame(top_auc).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Features surviving all six background-group stress checks",
        "",
        pd.DataFrame(top_stress).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top background scan-group-sensitive features",
        "",
        pd.DataFrame(top_group).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top UAV file-dependent features",
        "",
        pd.DataFrame(top_file).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    return "\n".join(lines)


def records(frame: pd.DataFrame, columns: list[str], count: int = 10) -> list[dict[str, Any]]:
    return frame.loc[:, columns].head(count).to_dict(orient="records")


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    classification_root = resolve_path(args.classification_root)
    candidate_root = resolve_path(args.candidate_root)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    config = MultiDomainFeatureConfig()
    candidates = load_frozen_dpg_candidates(candidate_root)
    detection, detection_errors = build_detection_catalog(
        manifest_path, candidates, config
    )
    classification, classification_errors = build_classification_catalog(
        classification_root, config
    )
    if detection.empty or classification.empty:
        raise RuntimeError("feature catalog cannot be empty")
    detection.to_csv(output_dir / "detection_sample_features.csv", index=False)
    classification.to_csv(
        output_dir / "classification_uav_sample_features.csv", index=False
    )
    pd.DataFrame(detection_errors + classification_errors).to_csv(
        output_dir / "errors.csv", index=False
    )

    separability = univariate_separability(detection)
    separability.to_csv(output_dir / "detection_univariate_separability.csv", index=False)
    background_stress = background_group_stress_table(detection)
    background_stress.to_csv(
        output_dir / "detection_background_group_stress.csv", index=False
    )
    detection.groupby(
        ["source_file", "target_present"], observed=True
    )[feature_columns(detection)].agg(["count", "mean", "std", "median"]).to_csv(
        output_dir / "detection_scan_profiles.csv"
    )
    background = detection.loc[detection["target_present"].eq(0)].copy()
    background_dependency = group_dependency_table(
        background, "source_file", "background_scan"
    )
    background_dependency.to_csv(
        output_dir / "background_scan_dependency.csv", index=False
    )
    classification_dependency = group_dependency_table(
        classification, "source_file", "uav_capture_file"
    )
    classification_dependency.to_csv(
        output_dir / "classification_file_dependency.csv", index=False
    )
    classification.groupby("source_file", observed=True)[
        feature_columns(classification)
    ].agg(["count", "mean", "std", "median"]).to_csv(
        output_dir / "classification_file_profiles.csv"
    )
    schema = build_feature_schema(detection)
    schema.to_csv(output_dir / "feature_schema.csv", index=False)

    zero_profiles = (
        background.groupby("source_file", observed=True)
        .agg(
            sample_count=("sample_id", "size"),
            zero_peak_fraction=("rd_peak_at_zero_band", "mean"),
            edge_peak_fraction=("rd_peak_at_edge_band", "mean"),
            zero_energy_mean=("rd_zero_doppler_energy_fraction", "mean"),
            edge_energy_mean=("rd_edge_doppler_energy_fraction", "mean"),
            spectral_width_mean=("rd_anchor_spectral_width_bins", "mean"),
            spectral_entropy_mean=("rd_anchor_entropy", "mean"),
        )
        .reset_index()
    )
    zero_profiles.to_csv(output_dir / "background_zero_doppler_profiles.csv", index=False)

    top_auc = records(
        separability,
        ["feature", "orientation_free_auc", "cohens_d_target_minus_background"],
    )
    top_group = records(
        background_dependency,
        ["feature", "group_eta_squared", "group_mean_range"],
    )
    top_stress = records(
        background_stress,
        [
            "feature",
            "pooled_direction",
            "pooled_oriented_auc",
            "worst_background_group_auc",
            "median_background_group_auc",
            "worst_background_group",
        ],
    )
    top_file = records(
        classification_dependency,
        ["feature", "group_eta_squared", "group_mean_range"],
    )
    target_detection = detection.loc[detection["target_present"].eq(1)]
    candidate_joint_localized = (
        target_detection["anchor_range_error_gates"].le(2)
        & target_detection["anchor_velocity_error_bins"].le(3)
    )
    summary = {
        "status": "COMPLETE_EXPLORATORY_CATALOG",
        "evidence_role": "current_data_feature_mining_not_model_selection",
        "manifest_path": str(manifest_path),
        "feature_count": int(len(schema)),
        "feature_domains": dict(FEATURE_DOMAINS),
        "anchor_policy": {
            "detection_local_features": "frozen_out_of_fold_DPG_candidate",
            "classification_local_features": "strongest_combined_HV_power_RD_cell",
            "scene_features": "strongest_combined_HV_power_RD_cell",
            "truth_coordinates_used": False,
        },
        "error_count": len(detection_errors) + len(classification_errors),
        "detection": {
            "sample_count": int(len(detection)),
            "target_count": int(detection["target_present"].sum()),
            "background_count": int(detection["target_present"].eq(0).sum()),
            "source_group_count": int(detection["source_file"].nunique()),
            "background_source_group_count": int(background["source_file"].nunique()),
            "long_window_available_count": int(detection["tf_long_window_available"].sum()),
            "candidate_joint_localization_count": int(candidate_joint_localized.sum()),
            "candidate_joint_localization_target_count": int(len(target_detection)),
            "candidate_joint_localization_rate": float(candidate_joint_localized.mean()),
            "top_confounded_univariate_features": top_auc,
            "top_background_group_stress_features": top_stress,
            "top_background_group_sensitive_features": top_group,
            "zero_doppler_profiles": zero_profiles.to_dict(orient="records"),
            "claim_warning": "target/background acquisition dates are fully confounded",
        },
        "classification_uav_only": {
            "sample_count": int(len(classification)),
            "source_file_count": int(classification["source_file"].nunique()),
            "class_names": sorted(classification["class_name"].unique().tolist()),
            "long_window_available_count": int(
                classification["tf_long_window_available"].sum()
            ),
            "top_file_dependent_features": top_file,
            "claim_warning": "UAV only; no classification performance can be estimated",
        },
        "blocked_domains": {
            "physical_micro_doppler": ["PRF", "continuous event timing"],
            "trajectory": ["verified timestamps", "hardware order", "track ID"],
            "absolute_polarimetry": ["H/V amplitude and phase calibration"],
            "balloon_payload_recognition": ["balloon and payload class samples"],
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / "REPORT.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
