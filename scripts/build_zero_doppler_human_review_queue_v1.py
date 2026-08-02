#!/usr/bin/env python3
"""Build a privacy-preserving review queue for fixed-notch residual diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CATALOG = (
    PROJECT_ROOT
    / "results/data_audit/multidomain_feature_catalog_v1/detection_sample_features.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/zero_doppler_human_review_v1"
FEATURE_COLUMNS = (
    "rd_anchor_zero_doppler_fraction",
    "rd_anchor_peak_fraction",
    "rd_anchor_main_band_fraction",
    "rd_anchor_entropy",
    "polar_roi_zdr_iqr_db",
    "polar_roi_rho_mean",
    "polar_roi_phase_resultant",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a human-review queue from the paired fixed-notch and "
            "target-protected residual V2 predictions."
        )
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-velocity-index", type=int, default=64)
    parser.add_argument("--feature-catalog", type=Path, default=DEFAULT_FEATURE_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(fold: int, mode: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "results/experiments"
        / f"zero_doppler_v1_{mode}_fold{fold:02d}_seed{seed}"
        / "tables/test_predictions.csv"
    )


def load_prediction_pair(fold: int, seed: int) -> tuple[pd.DataFrame, dict[str, Path]]:
    fixed_path = prediction_path(fold, "fixed_notch", seed)
    residual_path = prediction_path(fold, "fixed_residual", seed)
    fixed = pd.read_csv(fixed_path)
    residual = pd.read_csv(residual_path)
    required = {
        "sample_id",
        "source_file",
        "target_present",
        "score",
        "raw_score",
        "pred_range_index",
        "pred_velocity_index",
        "detected",
        "false_alarm",
        "correct_detection",
    }
    for label, frame in (("fixed_notch", fixed), ("fixed_residual", residual)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} fold {fold} missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{label} fold {fold} has duplicate sample IDs")
    paired = fixed.merge(
        residual,
        on="sample_id",
        suffixes=("_fixed", "_residual"),
        validate="one_to_one",
    )
    if len(paired) != len(fixed) or len(paired) != len(residual):
        raise ValueError(f"fold {fold} fixed/residual sample coverage differs")
    if not paired["target_present_fixed"].equals(paired["target_present_residual"]):
        raise ValueError(f"fold {fold} target labels differ")
    if not paired["source_file_fixed"].equals(paired["source_file_residual"]):
        raise ValueError(f"fold {fold} source-file values differ")
    paired.insert(0, "fold", int(fold))
    return paired, {"fixed_notch": fixed_path, "fixed_residual": residual_path}


def load_feature_catalog(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path)
    if "sample_id" not in features:
        raise ValueError("feature catalog has no sample_id column")
    if features["sample_id"].duplicated().any():
        raise ValueError("feature catalog has duplicate sample IDs")
    available = [column for column in FEATURE_COLUMNS if column in features.columns]
    selected = features[["sample_id", *available]].copy()
    return selected.rename(columns={column: f"feature_{column}" for column in available})


def priority(row: pd.Series) -> str:
    if bool(row["residual_removed"]):
        return "P0_removed_by_residual"
    if int(row["zero_velocity_distance_bins"]) <= 7:
        return "P1_near_zero_doppler"
    return "P2_retained_false_alarm"


def prepare_review_queue(
    pairs: Iterable[pd.DataFrame],
    features: pd.DataFrame,
    zero_velocity_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.concat(list(pairs), ignore_index=True)
    background = paired[paired["target_present_fixed"].eq(0)].copy()
    background["fixed_notch_false_alarm"] = background["false_alarm_fixed"].astype(bool)
    background["residual_false_alarm"] = background["false_alarm_residual"].astype(bool)
    background["residual_removed"] = (
        background["fixed_notch_false_alarm"] & ~background["residual_false_alarm"]
    )
    background["residual_added"] = (
        ~background["fixed_notch_false_alarm"] & background["residual_false_alarm"]
    )
    background["score_delta_residual_minus_fixed"] = (
        background["score_residual"] - background["score_fixed"]
    )
    background["zero_velocity_distance_bins"] = (
        background["pred_velocity_index_fixed"].astype(int) - int(zero_velocity_index)
    ).abs()

    summary = (
        background.groupby(["fold", "source_file_fixed"], as_index=False)
        .agg(
            background_sample_count=("sample_id", "size"),
            fixed_notch_false_alarms=("fixed_notch_false_alarm", "sum"),
            residual_false_alarms=("residual_false_alarm", "sum"),
            residual_removed=("residual_removed", "sum"),
            residual_added=("residual_added", "sum"),
            median_fixed_velocity_distance_bins=("zero_velocity_distance_bins", "median"),
        )
        .rename(columns={"source_file_fixed": "source_file"})
        .sort_values(["fixed_notch_false_alarms", "fold"], ascending=[False, True])
    )

    queue = background[background["fixed_notch_false_alarm"]].copy()
    queue = queue.merge(features, on="sample_id", how="left", validate="one_to_one")
    columns = [
        "fold",
        "sample_id",
        "source_file_fixed",
        "score_fixed",
        "raw_score_fixed",
        "score_residual",
        "score_delta_residual_minus_fixed",
        "pred_range_index_fixed",
        "pred_velocity_index_fixed",
        "pred_range_index_residual",
        "pred_velocity_index_residual",
        "zero_velocity_distance_bins",
        "fixed_notch_false_alarm",
        "residual_false_alarm",
        "residual_removed",
        "residual_added",
    ]
    columns.extend(column for column in queue.columns if column.startswith("feature_"))
    queue = queue[columns].rename(columns={"source_file_fixed": "source_file"})
    queue["review_priority"] = queue.apply(priority, axis=1)
    queue["review_status"] = "pending"
    queue["visible_pattern"] = "unreviewed"
    queue["physical_class"] = "unknown"
    queue["evidence_source"] = "prediction_and_relative_features_only"
    queue["review_note"] = ""
    queue = queue.sort_values(
        ["review_priority", "zero_velocity_distance_bins", "score_fixed"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return queue, summary


def make_readme(queue: pd.DataFrame, summary: pd.DataFrame) -> str:
    removed = int(queue["residual_removed"].sum())
    return f"""# 零多普勒虚警人工复核队列 V1

本目录把 fixed soft notch 的背景误警整理为人工复核队列。它用于帮助研究负责人描述
可见模式和未知项，不用于凭空给样本贴“建筑”“鸟类”等物理标签。

## 当前范围

- fixed-notch 背景误警：{len(queue)} 个；
- 被 target-protected residual V2 移除：{removed} 个；
- 输入是六个已经被开发过程使用的外层折，不是盲测；
- `zero_velocity_distance_bins` 是预测速度 bin 到第 64 bin 的距离，不是未经确认 PRF 下的
  物理速度；
- `feature_*` 字段是相对特征；绝对极化解释仍需要 H/V 标定。

## 你需要做什么

逐行查看 RD 图或对应原始文件后，仅在证据允许时填写：

| 字段 | 可填内容 |
|---|---|
| `review_status` | `reviewed`、`needs_more_context`、`unavailable` |
| `visible_pattern` | `near_zero_doppler_peak`、`multiple_peaks`、`broad_structure`、`edge_peak`、`no_clear_pattern` |
| `physical_class` | 默认 `unknown`；只有独立场景记录支持时才填具体类别 |
| `evidence_source` | 默认 `prediction_and_relative_features_only`；填写具体物理类别时改为 `independent_scene_record` |
| `review_note` | 描述你实际看到了什么，不写推测性因果结论 |

不要根据文件名、日期、分数或 H/V 数值单独推断地物类别。完成后保存为新的版本文件，
不要覆盖 `review_queue.csv`。之后运行 `audit_zero_doppler_human_review_v1.py` 检查复核结果。

## 文件

- `review_queue.csv`：逐样本待复核队列；
- `background_group_summary.csv`：每折/背景扫描组汇总；
- `manifest.json`：输入文件哈希和生成参数。
"""


def build_queue(
    *,
    folds: Iterable[int],
    seed: int,
    zero_velocity_index: int,
    feature_catalog: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    folds = list(folds)
    if not folds or any(fold not in range(1, 7) for fold in folds):
        raise ValueError("folds must be integers from 1 to 6")
    feature_catalog = resolve_path(feature_catalog)
    output_dir = resolve_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[pd.DataFrame] = []
    source_paths: dict[str, str] = {"feature_catalog": str(feature_catalog)}
    for fold in folds:
        pair, paths = load_prediction_pair(fold, seed)
        pairs.append(pair)
        for mode, path in paths.items():
            source_paths[f"fold_{fold:02d}_{mode}"] = str(path)
    features = load_feature_catalog(feature_catalog)
    queue, summary = prepare_review_queue(pairs, features, zero_velocity_index)
    queue.to_csv(output_dir / "review_queue.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(
        output_dir / "background_group_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "README.md").write_text(make_readme(queue, summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "role": "human_review_queue_not_model_evidence",
        "folds": folds,
        "seed": int(seed),
        "zero_velocity_index": int(zero_velocity_index),
        "queue_count": int(len(queue)),
        "residual_removed_in_queue": int(queue["residual_removed"].sum()),
        "source_files": {
            label: {"path": path, "sha256": sha256_file(Path(path))}
            for label, path in source_paths.items()
        },
        "claim_boundary": (
            "relative-feature and paired-development review aid; no physical "
            "background labels, blind-test claim, or calibrated polarimetric "
            "claim is established"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_queue(
        folds=args.folds,
        seed=args.seed,
        zero_velocity_index=args.zero_velocity_index,
        feature_catalog=args.feature_catalog,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("Zero-Doppler human review queue: PASS")
    print(f"queue_count={manifest['queue_count']}")
    print(f"residual_removed_in_queue={manifest['residual_removed_in_queue']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
