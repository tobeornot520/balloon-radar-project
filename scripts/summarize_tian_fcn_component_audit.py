#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Fold 1 Tian FCN PIR/MDP failure mechanisms"
    )
    parser.add_argument("--diagnostic-rows", type=Path, required=True)
    parser.add_argument("--raw-peak-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def classify_failure_mode(row: pd.Series) -> str:
    if bool(row["responsible_cell_selected"]):
        return "responsible_cell_selected"
    if bool(row["responsible_cell_in_selected_component"]):
        return "selected_component_mdp_chose_other_cell"
    return "responsible_cell_in_discarded_component"


def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, observed=True, dropna=False)
        .agg(
            sample_count=("sample_id", "size"),
            responsible_in_selected_component_count=(
                "responsible_cell_in_selected_component",
                "sum",
            ),
            responsible_cell_selected_count=("responsible_cell_selected", "sum"),
            responsible_in_selected_component_rate=(
                "responsible_cell_in_selected_component",
                "mean",
            ),
            responsible_cell_selected_rate=("responsible_cell_selected", "mean"),
            H_label_db_mean=("H_label_db", "mean"),
            H_label_db_min=("H_label_db", "min"),
            H_label_db_max=("H_label_db", "max"),
            decoded_error_mean=("nearest_decoded_euclidean_error", "mean"),
            decoded_error_max=("nearest_decoded_euclidean_error", "max"),
        )
        .reset_index()
    )


def main() -> int:
    args = parse_args()
    diagnostic_path = resolve_path(args.diagnostic_rows)
    raw_path = resolve_path(args.raw_peak_rows)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(diagnostic_path)
    required = {
        "sample_id",
        "source_file",
        "velocity_mps",
        "target_present",
        "responsible_cell_in_selected_component",
        "responsible_cell_selected",
        "pir_component_count",
        "selected_component_size",
        "selected_component_grid_y_span",
        "selected_component_grid_x_span",
        "selected_offset_norm",
        "responsible_offset_norm",
        "nearest_decoded_euclidean_error",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"diagnostic rows missing columns: {sorted(missing)}")
    targets = rows.loc[rows["target_present"].eq(1)].copy()
    if targets.empty:
        raise ValueError("diagnostic rows contain no target samples")

    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    raw_required = {"sample_id", "H_label_db", "H_local_abs_velocity_offset"}
    raw_missing = raw_required - set(raw.columns)
    if raw_missing:
        raise ValueError(f"raw peak rows missing columns: {sorted(raw_missing)}")
    raw_subset = raw[list(raw_required)].drop_duplicates("sample_id", keep=False)
    targets = targets.merge(raw_subset, on="sample_id", how="left", validate="one_to_one")
    if targets[list(raw_required - {"sample_id"})].isna().any().any():
        missing_ids = targets.loc[targets["H_label_db"].isna(), "sample_id"].tolist()
        raise ValueError(f"raw H evidence missing for samples: {missing_ids}")

    targets["failure_mode"] = targets.apply(classify_failure_mode, axis=1)
    targets["H_label_strength_band"] = pd.cut(
        targets["H_label_db"],
        bins=[-np.inf, 20.0, 25.0, 30.0, np.inf],
        labels=["<=20 dB", "(20,25] dB", "(25,30] dB", ">30 dB"],
    )

    failure_summary = aggregate(targets, ["failure_mode"])
    failure_summary["sample_fraction"] = (
        failure_summary["sample_count"] / len(targets)
    )
    source_summary = aggregate(targets, ["source_file", "velocity_mps"])
    strength_summary = aggregate(targets, ["H_label_strength_band"])

    mode_counts = targets["failure_mode"].value_counts().to_dict()
    summary = {
        "status": "PASS",
        "evidence_role": "fold01_validation_component_mechanism_diagnostic_only",
        "test_split_loaded": False,
        "target_count": int(len(targets)),
        "raw_H_evidence_match_count": int(targets["H_label_db"].notna().sum()),
        "raw_H_local_velocity_exact_label_count": int(
            targets["H_local_abs_velocity_offset"].eq(0).sum()
        ),
        "raw_H_local_velocity_within_one_bin_count": int(
            targets["H_local_abs_velocity_offset"].le(1).sum()
        ),
        "pir_component_count_unique": sorted(
            int(value) for value in targets["pir_component_count"].unique()
        ),
        "selected_component_shapes": sorted(
            {
                f"{int(row.selected_component_grid_y_span)}x"
                f"{int(row.selected_component_grid_x_span)}"
                for row in targets.itertuples()
            }
        ),
        "failure_mode_counts": {key: int(value) for key, value in mode_counts.items()},
        "mean_H_label_db_by_failure_mode": {
            str(row.failure_mode): float(row.H_label_db_mean)
            for row in failure_summary.itertuples()
        },
        "interpretation": (
            "two_stage_failure: component competition plus within-component "
            "minimum-offset selection; raw H strength is associated but source and "
            "velocity are confounded"
        ),
        "inputs": {
            "diagnostic_rows": str(diagnostic_path),
            "raw_peak_rows": str(raw_path),
        },
    }

    targets.to_csv(output_dir / "sample_failure_modes.csv", index=False)
    failure_summary.to_csv(output_dir / "failure_mode_summary.csv", index=False)
    source_summary.to_csv(output_dir / "source_velocity_summary.csv", index=False)
    strength_summary.to_csv(output_dir / "raw_H_strength_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
