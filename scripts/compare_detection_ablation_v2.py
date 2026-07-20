#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENTS = {
    "H": "detection_h_baseline_v2",
    "V": "detection_v_baseline_v2",
    "HV": "detection_hv_baseline_v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较H、V、HV完整检测基线，并输出样本级错误分析。"
    )
    parser.add_argument(
        "--experiments-root",
        default="results/experiments",
        help="实验目录根路径。",
    )
    parser.add_argument(
        "--output",
        default="results/experiments/detection_ablation_analysis_v2",
        help="分析结果输出目录。",
    )
    return parser.parse_args()


def load_experiment(
    experiments_root: Path,
    display_name: str,
    folder_name: str,
) -> tuple[dict, pd.DataFrame]:
    table_dir = experiments_root / folder_name / "tables"
    summary_path = table_dir / "summary.json"
    predictions_path = table_dir / "test_predictions.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少文件：{summary_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"缺少文件：{predictions_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(predictions_path, encoding="utf-8-sig")

    required = {
        "sample_id",
        "target_present",
        "score",
        "pred_range_index",
        "pred_velocity_index",
        "true_range_index",
        "true_velocity_index",
        "range_error_gates",
        "velocity_error_bins",
        "localization_ok",
        "detected",
        "false_alarm",
        "correct_detection",
        "mat_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{display_name} 的 test_predictions.csv 缺少字段：{missing}"
        )

    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[
            frame["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(
            f"{display_name} 出现重复 sample_id，示例：{duplicates[:5]}"
        )

    return summary, frame


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def normalize_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        "sample_id",
        "target_present",
        "score",
        "pred_range_index",
        "pred_velocity_index",
        "true_range_index",
        "true_velocity_index",
        "range_error_gates",
        "velocity_error_bins",
        "localization_ok",
        "detected",
        "false_alarm",
        "correct_detection",
        "beam_layer",
        "azimuth_deg",
        "distance_m",
        "velocity_mps",
        "mat_path",
    ]
    keep = [column for column in keep if column in frame.columns]
    result = frame[keep].copy()

    for column in [
        "localization_ok",
        "detected",
        "false_alarm",
        "correct_detection",
    ]:
        if column in result.columns:
            result[column] = bool_series(result[column])

    shared = {
        "sample_id",
        "target_present",
        "true_range_index",
        "true_velocity_index",
        "beam_layer",
        "azimuth_deg",
        "distance_m",
        "velocity_mps",
        "mat_path",
    }
    rename = {
        column: f"{prefix}_{column}"
        for column in result.columns
        if column not in shared
    }
    return result.rename(columns=rename)


def merge_predictions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for name in ("H", "V", "HV"):
        current = normalize_frame(frames[name], name)
        if merged is None:
            merged = current
            continue

        shared_candidates = [
            "sample_id",
            "target_present",
            "true_range_index",
            "true_velocity_index",
            "beam_layer",
            "azimuth_deg",
            "distance_m",
            "velocity_mps",
            "mat_path",
        ]
        join_columns = [
            column
            for column in shared_candidates
            if column in merged.columns and column in current.columns
        ]
        if "sample_id" not in join_columns:
            raise RuntimeError("无法按 sample_id 合并预测结果")

        merged = merged.merge(
            current,
            on=join_columns,
            how="outer",
            validate="one_to_one",
        )

    assert merged is not None
    if merged.isna().all(axis=1).any():
        raise RuntimeError("合并后出现完全空行")
    return merged.sort_values(
        ["target_present", "sample_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def experiment_summary_rows(summaries: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for name in ("H", "V", "HV"):
        summary = summaries[name]
        test = summary["test_metrics"]
        rows.append(
            {
                "input": name,
                "best_epoch": int(summary["best_epoch"]),
                "threshold": float(summary["validation_threshold"]),
                "test_joint_pd": float(test["joint_pd"]),
                "test_correct_count": int(test["correct_detection_count"]),
                "test_positive_count": int(test["positive_count"]),
                "test_pfa": float(test["pfa"]),
                "test_false_alarm_count": int(test["false_alarm_count"]),
                "test_background_count": int(test["background_count"]),
                "test_score_recall": float(test["score_recall"]),
                "test_auc": float(test["roc_auc"]),
                "range_mae_gates": float(test["all_positive_range_mae_gates"]),
                "velocity_mae_bins": float(
                    test["all_positive_velocity_mae_bins"]
                ),
                "binary_precision": float(test["binary_precision"]),
                "binary_f1": float(test["binary_f1"]),
            }
        )
    return rows


def classify_positive_failure(row: pd.Series, prefix: str) -> str:
    if bool(row[f"{prefix}_correct_detection"]):
        return "correct"
    if not bool(row[f"{prefix}_detected"]):
        return "score_miss"
    if not bool(row[f"{prefix}_localization_ok"]):
        return "localization_failure"
    return "other"


def save_failure_tables(merged: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    backgrounds = merged[merged["target_present"] == 0].copy()
    positives = merged[merged["target_present"] == 1].copy()

    for name in ("H", "V", "HV"):
        positives[f"{name}_failure_type"] = positives.apply(
            classify_positive_failure,
            axis=1,
            prefix=name,
        )

    false_alarm_mask = (
        backgrounds["H_false_alarm"]
        | backgrounds["V_false_alarm"]
        | backgrounds["HV_false_alarm"]
    )
    false_alarms = backgrounds.loc[false_alarm_mask].copy()
    false_alarms["false_alarm_models"] = false_alarms.apply(
        lambda row: ",".join(
            name
            for name in ("H", "V", "HV")
            if bool(row[f"{name}_false_alarm"])
        ),
        axis=1,
    )
    false_alarms.to_csv(
        output_dir / "false_alarm_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    positive_failures = positives.loc[
        ~(
            positives["H_correct_detection"]
            & positives["V_correct_detection"]
            & positives["HV_correct_detection"]
        )
    ].copy()
    positive_failures.to_csv(
        output_dir / "positive_failure_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hv_rescued_vs_h = positives.loc[
        positives["HV_correct_detection"]
        & ~positives["H_correct_detection"]
    ].copy()
    hv_rescued_vs_h.to_csv(
        output_dir / "hv_rescued_vs_h.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hv_regressed_vs_h = positives.loc[
        ~positives["HV_correct_detection"]
        & positives["H_correct_detection"]
    ].copy()
    hv_regressed_vs_h.to_csv(
        output_dir / "hv_regressed_vs_h.csv",
        index=False,
        encoding="utf-8-sig",
    )

    common_hard = positives.loc[
        ~positives["H_correct_detection"]
        & ~positives["V_correct_detection"]
        & ~positives["HV_correct_detection"]
    ].copy()
    common_hard.to_csv(
        output_dir / "common_hard_positive_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_only_rows = []
    for name in ("H", "V", "HV"):
        others = [other for other in ("H", "V", "HV") if other != name]
        mask = positives[f"{name}_correct_detection"]
        for other in others:
            mask &= ~positives[f"{other}_correct_detection"]
        subset = positives.loc[mask].copy()
        subset["only_correct_model"] = name
        best_only_rows.append(subset)

    if best_only_rows:
        pd.concat(best_only_rows, ignore_index=True).to_csv(
            output_dir / "model_unique_successes.csv",
            index=False,
            encoding="utf-8-sig",
        )

    failure_rows: list[dict] = []
    for name in ("H", "V", "HV"):
        counts = positives[f"{name}_failure_type"].value_counts()
        failure_rows.append(
            {
                "input": name,
                "correct": int(counts.get("correct", 0)),
                "score_miss": int(counts.get("score_miss", 0)),
                "localization_failure": int(
                    counts.get("localization_failure", 0)
                ),
                "other": int(counts.get("other", 0)),
                "false_alarms": int(backgrounds[f"{name}_false_alarm"].sum()),
            }
        )

    failure_counts = pd.DataFrame(failure_rows)
    failure_counts.to_csv(
        output_dir / "failure_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return failure_counts


def plot_metrics(summary: pd.DataFrame, figure_dir: Path) -> None:
    x = np.arange(len(summary))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, summary["test_joint_pd"], width, label="Joint Pd")
    ax.bar(x, summary["test_auc"], width, label="AUC")
    ax.bar(x + width, summary["test_pfa"], width, label="Pfa")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["input"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Detection ablation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "detection_metrics.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        summary["range_mae_gates"],
        width,
        label="Range MAE (gates)",
    )
    ax.bar(
        x + width / 2,
        summary["velocity_mae_bins"],
        width,
        label="Velocity MAE (bins)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["input"])
    ax.set_ylabel("Absolute error")
    ax.set_title("Localization error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "localization_mae.png", dpi=200)
    plt.close(fig)


def plot_failures(failure_counts: pd.DataFrame, figure_dir: Path) -> None:
    x = np.arange(len(failure_counts))
    bottom = np.zeros(len(failure_counts), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in [
        ("correct", "Correct"),
        ("score_miss", "Score miss"),
        ("localization_failure", "Localization failure"),
        ("other", "Other"),
    ]:
        values = failure_counts[column].to_numpy(float)
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(failure_counts["input"])
    ax.set_ylabel("Positive sample count")
    ax.set_title("Positive outcome breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "positive_failure_breakdown.png", dpi=200)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    summary: pd.DataFrame,
    merged: pd.DataFrame,
    failure_counts: pd.DataFrame,
) -> None:
    positives = merged[merged["target_present"] == 1]
    backgrounds = merged[merged["target_present"] == 0]

    h_correct = positives["H_correct_detection"]
    v_correct = positives["V_correct_detection"]
    hv_correct = positives["HV_correct_detection"]

    lines = [
        "# H/V/HV 检测消融样本级分析",
        "",
        "## 总体结果",
        "",
        summary.to_markdown(index=False),
        "",
        "## 样本级结论",
        "",
        f"- 测试集正样本：{len(positives)}",
        f"- 测试集背景：{len(backgrounds)}",
        f"- HV 相对 H 救回：{int((hv_correct & ~h_correct).sum())} 个正样本",
        f"- HV 相对 H 退化：{int((~hv_correct & h_correct).sum())} 个正样本",
        f"- HV 相对 V 救回：{int((hv_correct & ~v_correct).sum())} 个正样本",
        f"- 三种输入均失败：{int((~h_correct & ~v_correct & ~hv_correct).sum())} 个正样本",
        f"- 三种输入均虚警：{int((backgrounds['H_false_alarm'] & backgrounds['V_false_alarm'] & backgrounds['HV_false_alarm']).sum())} 个背景样本",
        "",
        "## 输出文件说明",
        "",
        "- `summary_comparison.csv`：三组总体指标。",
        "- `sample_level_comparison.csv`：按 sample_id 合并后的全部预测。",
        "- `failure_counts.csv`：分数漏检、定位失败、正确检测和虚警数量。",
        "- `false_alarm_samples.csv`：至少被一种输入误报的背景。",
        "- `positive_failure_samples.csv`：至少一种输入未正确检测的正样本。",
        "- `hv_rescued_vs_h.csv`：HV正确、H失败的样本。",
        "- `hv_regressed_vs_h.csv`：H正确、HV失败的样本。",
        "- `common_hard_positive_samples.csv`：H、V、HV均失败的正样本。",
        "- `model_unique_successes.csv`：仅某一种输入正确的样本。",
    ]
    (output_dir / "README_分析结论.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}

    for display_name, folder_name in EXPERIMENTS.items():
        summary, frame = load_experiment(
            experiments_root,
            display_name,
            folder_name,
        )
        summaries[display_name] = summary
        frames[display_name] = frame

    summary_frame = pd.DataFrame(experiment_summary_rows(summaries))
    summary_frame.to_csv(
        output_dir / "summary_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged = merge_predictions(frames)
    merged.to_csv(
        output_dir / "sample_level_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failure_counts = save_failure_tables(merged, output_dir)
    plot_metrics(summary_frame, figure_dir)
    plot_failures(failure_counts, figure_dir)
    write_readme(output_dir, summary_frame, merged, failure_counts)

    positives = merged[merged["target_present"] == 1]
    backgrounds = merged[merged["target_present"] == 0]

    print("=" * 78)
    print("H/V/HV 样本级消融分析完成")
    print(summary_frame[
        [
            "input",
            "test_correct_count",
            "test_joint_pd",
            "test_false_alarm_count",
            "test_pfa",
            "test_auc",
            "range_mae_gates",
            "velocity_mae_bins",
        ]
    ].to_string(index=False))
    print("-" * 78)
    print(
        "HV相对H救回正样本：",
        int(
            (
                positives["HV_correct_detection"]
                & ~positives["H_correct_detection"]
            ).sum()
        ),
    )
    print(
        "HV相对H退化正样本：",
        int(
            (
                ~positives["HV_correct_detection"]
                & positives["H_correct_detection"]
            ).sum()
        ),
    )
    print(
        "三种输入均失败正样本：",
        int(
            (
                ~positives["H_correct_detection"]
                & ~positives["V_correct_detection"]
                & ~positives["HV_correct_detection"]
            ).sum()
        ),
    )
    print(
        "至少一种输入发生虚警的背景：",
        int(
            (
                backgrounds["H_false_alarm"]
                | backgrounds["V_false_alarm"]
                | backgrounds["HV_false_alarm"]
            ).sum()
        ),
    )
    print(f"结果目录：{output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
