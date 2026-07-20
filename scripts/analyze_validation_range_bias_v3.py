#!/usr/bin/env python3
"""Validation-only signed range bias analysis for H/V/HV detection baselines.

Rules:
- Candidate range offsets are selected ONLY from validation predictions.
- Test predictions are never used to select an offset.
- The script writes both the validation selection and a locked one-shot test diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

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

ALIASES: Dict[str, tuple[str, ...]] = {
    "target_present": ("target_present", "has_target", "present", "label_present"),
    "score": ("score", "max_score", "detection_score", "pred_score"),
    "pred_range": ("pred_range_index", "pred_range", "range_pred_index"),
    "pred_velocity": ("pred_velocity_index", "pred_velocity", "velocity_pred_index"),
    "true_range": ("true_range_index", "range_index", "target_range_index"),
    "true_velocity": ("true_velocity_index", "velocity_index", "target_velocity_index"),
    "sample_id": ("sample_id", "id", "name"),
}


def find_column(df: pd.DataFrame, logical: str, required: bool = True) -> Optional[str]:
    normalized = {str(c).lstrip("\ufeff"): c for c in df.columns}
    for alias in ALIASES[logical]:
        if alias in normalized:
            return normalized[alias]
    if required:
        raise KeyError(f"缺少字段 {logical}，现有字段：{list(df.columns)}")
    return None


def normalize_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()
    for key in ("target_present", "score", "pred_range", "pred_velocity", "true_range", "true_velocity"):
        col = find_column(df, key)
        out[key] = pd.to_numeric(df[col], errors="coerce")
    sid = find_column(df, "sample_id", required=False)
    out["sample_id"] = df[sid].astype(str) if sid else np.arange(len(df)).astype(str)
    return out


def read_summary(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_find_number(obj: Any, keys: Iterable[str]) -> Optional[float]:
    wanted = set(keys)
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if key in wanted and isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            result = deep_find_number(value, wanted)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = deep_find_number(value, wanted)
            if result is not None:
                return result
    return None


def threshold_from_summary(summary: Mapping[str, Any]) -> float:
    value = deep_find_number(summary, ("threshold", "best_threshold", "selected_threshold", "val_threshold"))
    if value is None:
        raise KeyError("summary.json 中找不到阈值字段。")
    return value


def metrics(df: pd.DataFrame, threshold: float, offset: int, range_tol: int, velocity_tol: int) -> Dict[str, float]:
    positive = df["target_present"] == 1
    background = ~positive
    detected = df["score"] >= threshold
    corrected_range = df["pred_range"] + int(offset)
    range_error = (corrected_range - df["true_range"]).abs()
    velocity_error = (df["pred_velocity"] - df["true_velocity"]).abs()
    localization_ok = (range_error <= range_tol) & (velocity_error <= velocity_tol)
    correct = positive & detected & localization_ok
    false_alarm = background & detected
    pos_n = int(positive.sum())
    bg_n = int(background.sum())
    detected_pos = positive & detected
    return {
        "offset_gates": int(offset),
        "positive_count": pos_n,
        "background_count": bg_n,
        "score_detected_positive_count": int(detected_pos.sum()),
        "correct_count": int(correct.sum()),
        "joint_pd": float(correct.sum() / pos_n) if pos_n else math.nan,
        "false_alarm_count": int(false_alarm.sum()),
        "pfa": float(false_alarm.sum() / bg_n) if bg_n else math.nan,
        "range_mae_all_positive": float(range_error[positive].mean()) if pos_n else math.nan,
        "range_mae_score_detected": float(range_error[detected_pos].mean()) if detected_pos.any() else math.nan,
        "velocity_mae_all_positive": float(velocity_error[positive].mean()) if pos_n else math.nan,
    }


def signed_stats(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    positive = df["target_present"] == 1
    detected_positive = positive & (df["score"] >= threshold)
    signed = df.loc[positive, "pred_range"] - df.loc[positive, "true_range"]
    signed_detected = df.loc[detected_positive, "pred_range"] - df.loc[detected_positive, "true_range"]
    def safe(series: pd.Series, fn: str) -> float:
        return float(getattr(series, fn)()) if len(series) else math.nan
    return {
        "positive_count": int(positive.sum()),
        "score_detected_positive_count": int(detected_positive.sum()),
        "signed_mean_all_positive": safe(signed, "mean"),
        "signed_median_all_positive": safe(signed, "median"),
        "signed_mean_score_detected": safe(signed_detected, "mean"),
        "signed_median_score_detected": safe(signed_detected, "median"),
        "near_minus_3_count": int(((signed_detected >= -3.5) & (signed_detected <= -2.5)).sum()),
        "negative_count": int((signed_detected < 0).sum()),
        "zero_count": int((signed_detected == 0).sum()),
        "positive_count_signed": int((signed_detected > 0).sum()),
    }


def select_offset(sweep: pd.DataFrame) -> pd.Series:
    ranked = sweep.copy()
    ranked["abs_offset"] = ranked["offset_gates"].abs()
    ranked = ranked.sort_values(
        ["joint_pd", "range_mae_score_detected", "abs_offset", "offset_gates"],
        ascending=[False, True, True, True],
    )
    return ranked.iloc[0]


def plot_histogram(df: pd.DataFrame, threshold: float, title: str, output: Path) -> None:
    mask = (df["target_present"] == 1) & (df["score"] >= threshold)
    signed = (df.loc[mask, "pred_range"] - df.loc[mask, "true_range"]).dropna().to_numpy()
    plt.figure(figsize=(8, 5))
    if signed.size:
        lo = int(np.floor(signed.min())) - 1
        hi = int(np.ceil(signed.max())) + 1
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        plt.hist(signed, bins=bins, edgecolor="black")
        plt.axvline(0, linestyle="--", linewidth=1.2)
        plt.axvline(float(np.mean(signed)), linestyle=":", linewidth=1.4, label=f"mean={np.mean(signed):.3f}")
        plt.legend()
    plt.xlabel("Signed range error = predicted - true (gates)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments/detection_validation_bias_v3"))
    parser.add_argument("--min-offset", type=int, default=-5)
    parser.add_argument("--max-offset", type=int, default=5)
    parser.add_argument("--range-tol", type=int, default=2)
    parser.add_argument("--velocity-tol", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)

    signed_rows = []
    locked_rows = []
    all_sweeps = []
    locked_json: Dict[str, Any] = {
        "selection_rule": "validation_only",
        "range_tolerance_gates": args.range_tol,
        "velocity_tolerance_bins": args.velocity_tol,
        "models": {},
    }

    for mode, exp_name in EXPERIMENTS.items():
        exp_dir = args.experiments_root / exp_name
        tables = exp_dir / "tables"
        val_path = tables / "val_predictions.csv"
        test_path = tables / "test_predictions.csv"
        summary_path = tables / "summary.json"
        for path in (val_path, test_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(f"缺少文件：{path}")

        summary = read_summary(summary_path)
        threshold = threshold_from_summary(summary)
        val_df = normalize_predictions(val_path)
        test_df = normalize_predictions(test_path)

        stats = signed_stats(val_df, threshold)
        stats.update({"input": mode, "threshold": threshold})
        signed_rows.append(stats)

        mode_sweep = []
        for offset in range(args.min_offset, args.max_offset + 1):
            row = metrics(val_df, threshold, offset, args.range_tol, args.velocity_tol)
            row.update({"input": mode, "split": "val"})
            mode_sweep.append(row)
        sweep_df = pd.DataFrame(mode_sweep)
        selected = select_offset(sweep_df)
        locked_offset = int(selected["offset_gates"])

        val_before = metrics(val_df, threshold, 0, args.range_tol, args.velocity_tol)
        val_after = metrics(val_df, threshold, locked_offset, args.range_tol, args.velocity_tol)
        test_before = metrics(test_df, threshold, 0, args.range_tol, args.velocity_tol)
        test_after = metrics(test_df, threshold, locked_offset, args.range_tol, args.velocity_tol)

        locked_rows.extend([
            {"input": mode, "split": "val", "stage": "before", **val_before},
            {"input": mode, "split": "val", "stage": "validation_locked", **val_after},
            {"input": mode, "split": "test", "stage": "before", **test_before},
            {"input": mode, "split": "test", "stage": "validation_locked", **test_after},
        ])
        all_sweeps.extend(mode_sweep)
        locked_json["models"][mode] = {
            "experiment": exp_name,
            "threshold": threshold,
            "locked_range_offset_gates": locked_offset,
            "selected_from": "validation predictions only",
            "validation_before": val_before,
            "validation_after": val_after,
            "test_before_diagnostic": test_before,
            "test_after_locked_diagnostic": test_after,
        }
        plot_histogram(
            val_df,
            threshold,
            f"{mode} validation signed range error (score-detected positives)",
            figures / f"{mode}_validation_signed_range_error.png",
        )

    signed_df = pd.DataFrame(signed_rows)
    sweep_all = pd.DataFrame(all_sweeps)
    locked_df = pd.DataFrame(locked_rows)
    signed_df.to_csv(args.output_dir / "validation_signed_bias_summary.csv", index=False, encoding="utf-8-sig")
    sweep_all.to_csv(args.output_dir / "validation_offset_sweep.csv", index=False, encoding="utf-8-sig")
    locked_df.to_csv(args.output_dir / "locked_offset_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "locked_validation_offsets.json").write_text(
        json.dumps(locked_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 验证集有符号距离偏差分析",
        "",
        "所有候选距离偏移均只由验证集选择；测试集只接受已锁定规则的一次性诊断，不参与选择。",
        "",
        "## 验证集偏差统计",
        "",
        "```text",
        signed_df.to_string(index=False),
        "```",
        "",
        "## 验证集锁定偏移前后",
        "",
        "```text",
        locked_df.to_string(index=False),
        "```",
        "",
        "注意：只有当验证集呈现稳定且可解释的偏差时，才考虑把校准作为正式后处理；不得根据测试结果回头修改偏移。",
    ]
    (args.output_dir / "README_验证集偏差结论.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 78)
    print("验证集距离偏差分析完成")
    print(signed_df.to_string(index=False))
    print("-" * 78)
    for mode in EXPERIMENTS:
        offset = locked_json["models"][mode]["locked_range_offset_gates"]
        val_before = locked_json["models"][mode]["validation_before"]["joint_pd"]
        val_after = locked_json["models"][mode]["validation_after"]["joint_pd"]
        print(f"{mode}: 验证集锁定偏移={offset:+d}门, joint Pd {val_before:.4f} -> {val_after:.4f}")
    print(f"结果目录：{args.output_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
