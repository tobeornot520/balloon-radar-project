#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "data_audit" / "roi_bc_dpg_joint_tables_v1"
)
FOLDS = tuple(range(1, 7))
SPLITS = ("val", "test")

ROI_EXPERIMENTS = {
    "roi_base": "power2_baseline",
    "roi_power": "power2_roi_power_control",
    "roi_ri4": "power2_roi_ri4",
}

BC_REQUIRED_COLUMNS = {
    "sample_id",
    "scan_group",
    "target_present",
    "mat_path",
    "raw_score",
    "score",
    "p_background",
    "suppression",
    "detected",
    "false_alarm",
    "correct_detection",
}

ROI_REQUIRED_COLUMNS = {
    "sample_id",
    "source_file",
    "target_present",
    "mat_path",
    "raw_score",
    "refined_score",
    "suppression",
    "refined_fixed_detected",
    "refined_fixed_false_alarm",
    "refined_fixed_correct_detection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic, sample-aligned ROI and BC-DPG tables."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generated files in an existing output directory to be replaced.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def prediction_paths(fold: int, split: str) -> dict[str, Path]:
    if fold not in FOLDS:
        raise ValueError(f"Unsupported fold: {fold}")
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    fold_tag = f"fold{fold:02d}"
    experiments = ROOT / "results" / "experiments"
    paths = {
        "bc": (
            experiments
            / f"bc_dpg_v3_scan_target_v4_{fold_tag}_seed42"
            / "tables"
            / f"base_threshold_{split}_predictions.csv"
        )
    }
    for name, mode in ROI_EXPERIMENTS.items():
        paths[name] = (
            experiments
            / f"roi_polar_stage4_v1_{mode}_v4_{fold_tag}_seed42_formal"
            / "tables"
            / f"{split}_predictions.csv"
        )
    return paths


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def read_prediction_table(
    path: Path,
    table_name: str,
    required: Iterable[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{table_name} not found: {path}")

    frame = pd.read_csv(path)
    require_columns(frame, required, table_name)
    if frame.empty:
        raise ValueError(f"{table_name} is empty: {path}")

    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
    if frame["sample_id"].eq("").any():
        raise ValueError(f"{table_name} contains an empty sample_id")

    duplicates = frame.loc[
        frame["sample_id"].duplicated(keep=False), "sample_id"
    ].unique()
    if len(duplicates):
        raise ValueError(
            f"{table_name} contains duplicate sample_id values: "
            f"{duplicates[:10].tolist()}"
        )
    return frame


def align_to_reference(
    reference_ids: pd.Series,
    frame: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    reference_set = set(reference_ids)
    actual_set = set(frame["sample_id"])
    if reference_set != actual_set:
        missing = sorted(reference_set - actual_set)[:10]
        extra = sorted(actual_set - reference_set)[:10]
        raise ValueError(
            f"{table_name} sample_id mismatch: missing={missing}, extra={extra}"
        )

    indexed = frame.set_index("sample_id")
    if not indexed.index.is_unique:
        raise ValueError(f"{table_name} sample_id index is not unique")
    return indexed.loc[reference_ids].reset_index()


def boolean_series(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "nan": False,
        "": False,
    }
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{name} contains unsupported boolean values: {unknown[:10]}")
    return normalized.map(mapping).astype(bool)


def assert_equal(
    reference: pd.Series,
    candidate: pd.Series,
    description: str,
) -> None:
    left = reference.reset_index(drop=True)
    right = candidate.reset_index(drop=True)
    if left.equals(right):
        return

    mismatch = left.ne(right) & ~(left.isna() & right.isna())
    indices = mismatch[mismatch].index.tolist()[:10]
    raise ValueError(f"{description} mismatch at rows {indices}")


def build_fold_joint(fold: int, split: str) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = prediction_paths(fold, split)
    bc = read_prediction_table(paths["bc"], "bc", BC_REQUIRED_COLUMNS)
    reference_ids = bc["sample_id"].reset_index(drop=True)

    roi_tables: dict[str, pd.DataFrame] = {}
    for name in ROI_EXPERIMENTS:
        table = read_prediction_table(paths[name], name, ROI_REQUIRED_COLUMNS)
        roi_tables[name] = align_to_reference(reference_ids, table, name)

    bc = bc.reset_index(drop=True)
    bc_labels = pd.to_numeric(bc["target_present"], errors="raise").astype(int)
    bc_paths = bc["mat_path"].astype(str).str.strip()
    bc_groups = bc["scan_group"].astype(str).str.strip()

    for name, table in roi_tables.items():
        assert_equal(
            bc_labels,
            pd.to_numeric(table["target_present"], errors="raise").astype(int),
            f"Fold {fold:02d} {split} {name} target_present",
        )
        assert_equal(
            bc_paths,
            table["mat_path"].astype(str).str.strip(),
            f"Fold {fold:02d} {split} {name} mat_path",
        )
        assert_equal(
            bc_groups,
            table["source_file"].astype(str).str.strip(),
            f"Fold {fold:02d} {split} {name} scan group",
        )

    joint = pd.DataFrame(
        {
            "fold": fold,
            "split": split,
            "sample_id": reference_ids,
            "scan_group": bc_groups,
            "target_present": bc_labels,
            "mat_path": bc_paths,
            "bc_raw_score": pd.to_numeric(bc["raw_score"], errors="raise"),
            "bc_score": pd.to_numeric(bc["score"], errors="raise"),
            "bc_p_background": pd.to_numeric(
                bc["p_background"], errors="raise"
            ),
            "bc_suppression": pd.to_numeric(bc["suppression"], errors="raise"),
            "bc_detected": boolean_series(bc["detected"], "bc.detected"),
            "bc_false_alarm": boolean_series(
                bc["false_alarm"], "bc.false_alarm"
            ),
            "bc_correct": boolean_series(
                bc["correct_detection"], "bc.correct_detection"
            ),
        }
    )

    for prefix, table in roi_tables.items():
        joint[f"{prefix}_raw_score"] = pd.to_numeric(
            table["raw_score"], errors="raise"
        )
        joint[f"{prefix}_score"] = pd.to_numeric(
            table["refined_score"], errors="raise"
        )
        joint[f"{prefix}_suppression"] = pd.to_numeric(
            table["suppression"], errors="raise"
        )
        joint[f"{prefix}_detected"] = boolean_series(
            table["refined_fixed_detected"], f"{prefix}.detected"
        )
        joint[f"{prefix}_false_alarm"] = boolean_series(
            table["refined_fixed_false_alarm"], f"{prefix}.false_alarm"
        )
        joint[f"{prefix}_correct"] = boolean_series(
            table["refined_fixed_correct_detection"], f"{prefix}.correct"
        )

    joint["bc_score_drop"] = joint["bc_raw_score"] - joint["bc_score"]
    joint["roi_power_score_drop"] = (
        joint["roi_power_raw_score"] - joint["roi_power_score"]
    )
    joint["roi_ri4_score_drop"] = (
        joint["roi_ri4_raw_score"] - joint["roi_ri4_score"]
    )

    for prefix in ("roi_power", "roi_ri4"):
        short = "power" if prefix == "roi_power" else "ri4"
        joint[f"fa_bc_only_vs_{short}"] = (
            joint["bc_false_alarm"] & ~joint[f"{prefix}_false_alarm"]
        )
        joint[f"fa_{short}_only_vs_bc"] = (
            joint[f"{prefix}_false_alarm"] & ~joint["bc_false_alarm"]
        )
        joint[f"target_bc_only_vs_{short}"] = (
            joint["bc_correct"] & ~joint[f"{prefix}_correct"]
        )
        joint[f"target_{short}_only_vs_bc"] = (
            joint[f"{prefix}_correct"] & ~joint["bc_correct"]
        )

    alignment = {
        "fold": fold,
        "split": split,
        "rows": int(len(joint)),
        "unique_sample_id": int(joint["sample_id"].nunique()),
        "exact_sample_alignment": True,
        "exact_label_alignment": True,
        "exact_mat_path_alignment": True,
        "exact_scan_group_alignment": True,
        "bc_source": str(paths["bc"].relative_to(ROOT)),
    }
    return joint, alignment


def build_joint_tables(
    folds: Iterable[int],
    splits: Iterable[str],
) -> tuple[dict[tuple[int, str], pd.DataFrame], pd.DataFrame]:
    tables: dict[tuple[int, str], pd.DataFrame] = {}
    alignment_rows: list[dict[str, object]] = []
    for fold in folds:
        for split in splits:
            table, alignment = build_fold_joint(int(fold), str(split))
            tables[(int(fold), str(split))] = table
            alignment_rows.append(alignment)
    return tables, pd.DataFrame(alignment_rows)


def ensure_writable_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace generated files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    folds = tuple(dict.fromkeys(args.folds))
    splits = tuple(dict.fromkeys(args.splits))
    output_dir = resolve_output_dir(args.output_dir)
    ensure_writable_output(output_dir, args.overwrite)

    tables, alignment = build_joint_tables(folds, splits)
    all_frames: list[pd.DataFrame] = []
    for (fold, split), table in tables.items():
        table.to_csv(
            output_dir / f"fold_{fold:02d}_{split}_joint.csv",
            index=False,
        )
        all_frames.append(table)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_dir / "all_joint_predictions.csv", index=False)
    alignment.to_csv(output_dir / "alignment.csv", index=False)

    status = {
        "status": "ok",
        "folds": list(folds),
        "splits": list(splits),
        "rows": int(len(combined)),
        "rows_by_split": {
            split: int(combined["split"].eq(split).sum()) for split in splits
        },
        "exact_alignment": True,
        "test_threshold_retuning": False,
        "bc_decision_source": "base_threshold_{split}_predictions.csv",
        "roi_decision_source": "refined_fixed_* columns from {split}_predictions.csv",
        "output_dir": display_path(output_dir),
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"rows={len(combined)}")
    print(f"exact_alignment=True")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
