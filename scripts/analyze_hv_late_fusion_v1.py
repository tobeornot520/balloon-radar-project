#!/usr/bin/env python3
"""H/V validation-locked decision-level late-fusion diagnostic.

Purpose
-------
Use existing H-only and V-only prediction CSV files. Candidate fusion rules,
thresholds, and any weights are selected ONLY on the validation set. The
selected rule is then applied once to the test set for diagnosis.

This is a low-cost diagnostic baseline, not the final trainable dual-branch
network. It must not be tuned again using test results.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


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


def normalize_predictions(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()
    sid = find_column(df, "sample_id", required=False)
    out["sample_id"] = df[sid].astype(str) if sid else np.arange(len(df)).astype(str)
    for key in ("target_present", "score", "pred_range", "pred_velocity", "true_range", "true_velocity"):
        col = find_column(df, key)
        out[f"{prefix}_{key}"] = pd.to_numeric(df[col], errors="coerce")
    if out["sample_id"].duplicated().any():
        dup = out.loc[out["sample_id"].duplicated(), "sample_id"].tolist()[:10]
        raise ValueError(f"{path} 存在重复sample_id，例如：{dup}")
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
            result = deep_find_number(value, keys)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = deep_find_number(value, keys)
            if result is not None:
                return result
    return None


def threshold_from_summary(summary: Mapping[str, Any]) -> float:
    value = deep_find_number(
        summary,
        ("threshold", "best_threshold", "selected_threshold", "val_threshold"),
    )
    if value is None:
        raise KeyError("summary.json 中找不到阈值字段。")
    return float(value)


def merge_modes(h: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    merged = h.merge(v, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(h) or len(merged) != len(v):
        raise ValueError(f"H/V样本未完全对齐：H={len(h)}, V={len(v)}, merge={len(merged)}")
    checks = [
        ("target_present", 0.0),
        ("true_range", 0.0),
        ("true_velocity", 0.0),
    ]
    for field, tol in checks:
        left = merged[f"H_{field}"].to_numpy(float)
        right = merged[f"V_{field}"].to_numpy(float)
        equal = np.isclose(left, right, atol=tol, equal_nan=True)
        if not np.all(equal):
            bad = merged.loc[~equal, "sample_id"].tolist()[:10]
            raise ValueError(f"H/V的{field}不一致，例如：{bad}")
    merged["target_present"] = merged["H_target_present"]
    merged["true_range"] = merged["H_true_range"]
    merged["true_velocity"] = merged["H_true_velocity"]
    return merged


def signed_margin(score: np.ndarray, threshold: float) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    upper = max(1.0 - threshold, 1e-8)
    lower = max(threshold, 1e-8)
    return np.where(
        score >= threshold,
        (score - threshold) / upper,
        (score - threshold) / lower,
    )


def choose_coordinates(
    df: pd.DataFrame,
    rule: str,
    alpha: float,
    h_threshold: float,
    v_threshold: float,
    agreement_range: int,
    agreement_velocity: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hs = df["H_score"].to_numpy(float)
    vs = df["V_score"].to_numpy(float)
    hr = df["H_pred_range"].to_numpy(float)
    vr = df["V_pred_range"].to_numpy(float)
    hv = df["H_pred_velocity"].to_numpy(float)
    vv = df["V_pred_velocity"].to_numpy(float)

    hm = signed_margin(hs, h_threshold)
    vm = signed_margin(vs, v_threshold)

    if rule == "higher_raw_score":
        choose_h = hs >= vs
        pr = np.where(choose_h, hr, vr)
        pv = np.where(choose_h, hv, vv)
        source = np.where(choose_h, "H", "V")
    elif rule == "higher_normalized_margin":
        choose_h = hm >= vm
        pr = np.where(choose_h, hr, vr)
        pv = np.where(choose_h, hv, vv)
        source = np.where(choose_h, "H", "V")
    elif rule == "agreement_mean_else_margin":
        agree = (
            np.abs(hr - vr) <= agreement_range
        ) & (
            np.abs(hv - vv) <= agreement_velocity
        )
        choose_h = hm >= vm
        margin_r = np.where(choose_h, hr, vr)
        margin_v = np.where(choose_h, hv, vv)
        pr = np.where(agree, np.rint((hr + vr) / 2.0), margin_r)
        pv = np.where(agree, np.rint((hv + vv) / 2.0), margin_v)
        source = np.where(agree, "mean", np.where(choose_h, "H", "V"))
    elif rule == "alpha_weighted_coordinates":
        pr = np.rint(alpha * hr + (1.0 - alpha) * vr)
        pv = np.rint(alpha * hv + (1.0 - alpha) * vv)
        source = np.full(len(df), f"weighted_{alpha:.2f}", dtype=object)
    else:
        raise ValueError(f"未知坐标规则：{rule}")
    return pr, pv, source


def make_fused_score(
    df: pd.DataFrame,
    score_rule: str,
    alpha: float,
    h_threshold: float,
    v_threshold: float,
) -> np.ndarray:
    hs = df["H_score"].to_numpy(float)
    vs = df["V_score"].to_numpy(float)
    if score_rule == "raw_weighted":
        return alpha * hs + (1.0 - alpha) * vs
    if score_rule == "normalized_margin_weighted":
        hm = signed_margin(hs, h_threshold)
        vm = signed_margin(vs, v_threshold)
        # Map approximately [-1, 1] to [0, 1] for convenient thresholding.
        return 0.5 * (alpha * hm + (1.0 - alpha) * vm + 1.0)
    if score_rule == "max_raw":
        return np.maximum(hs, vs)
    if score_rule == "max_normalized_margin":
        return 0.5 * (np.maximum(
            signed_margin(hs, h_threshold),
            signed_margin(vs, v_threshold),
        ) + 1.0)
    raise ValueError(f"未知分数规则：{score_rule}")


def metric_dict(
    df: pd.DataFrame,
    score: np.ndarray,
    pred_range: np.ndarray,
    pred_velocity: np.ndarray,
    threshold: float,
    range_tol: int,
    velocity_tol: int,
) -> Dict[str, float]:
    target = df["target_present"].to_numpy(float) == 1
    background = ~target
    detected = score >= threshold
    range_error = np.abs(pred_range - df["true_range"].to_numpy(float))
    velocity_error = np.abs(pred_velocity - df["true_velocity"].to_numpy(float))
    location_ok = (range_error <= range_tol) & (velocity_error <= velocity_tol)
    correct = target & detected & location_ok
    false_alarm = background & detected
    detected_pos = target & detected

    auc = math.nan
    if roc_auc_score is not None and len(np.unique(target.astype(int))) == 2:
        try:
            auc = float(roc_auc_score(target.astype(int), score))
        except Exception:
            auc = math.nan

    return {
        "threshold": float(threshold),
        "positive_count": int(target.sum()),
        "background_count": int(background.sum()),
        "score_detected_positive_count": int(detected_pos.sum()),
        "score_detection_pd": float(detected_pos.sum() / target.sum()) if target.any() else math.nan,
        "correct_count": int(correct.sum()),
        "joint_pd": float(correct.sum() / target.sum()) if target.any() else math.nan,
        "false_alarm_count": int(false_alarm.sum()),
        "pfa": float(false_alarm.sum() / background.sum()) if background.any() else math.nan,
        "auc": auc,
        "range_mae_all_positive": float(np.mean(range_error[target])) if target.any() else math.nan,
        "range_mae_score_detected": float(np.mean(range_error[detected_pos])) if detected_pos.any() else math.nan,
        "velocity_mae_all_positive": float(np.mean(velocity_error[target])) if target.any() else math.nan,
        "velocity_mae_score_detected": float(np.mean(velocity_error[detected_pos])) if detected_pos.any() else math.nan,
    }


def select_threshold(
    df: pd.DataFrame,
    score: np.ndarray,
    pred_range: np.ndarray,
    pred_velocity: np.ndarray,
    pfa_cap: float,
    range_tol: int,
    velocity_tol: int,
) -> Dict[str, float]:
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        raise ValueError("融合分数全部无效。")
    thresholds = np.unique(finite)
    thresholds = np.r_[np.nextafter(thresholds.max(), np.inf), thresholds[::-1]]
    rows = []
    for threshold in thresholds:
        row = metric_dict(
            df, score, pred_range, pred_velocity,
            float(threshold), range_tol, velocity_tol,
        )
        if row["pfa"] <= pfa_cap + 1e-12:
            rows.append(row)
    if not rows:
        # Always feasible at a threshold above max(score), but keep a safe fallback.
        row = metric_dict(
            df, score, pred_range, pred_velocity,
            float(np.nextafter(finite.max(), np.inf)), range_tol, velocity_tol,
        )
        rows.append(row)

    ranked = pd.DataFrame(rows).sort_values(
        [
            "joint_pd",
            "score_detection_pd",
            "pfa",
            "range_mae_score_detected",
            "velocity_mae_score_detected",
            "threshold",
        ],
        ascending=[False, False, True, True, True, False],
        na_position="last",
    )
    return ranked.iloc[0].to_dict()


def baseline_metrics(
    df: pd.DataFrame,
    mode: str,
    threshold: float,
    range_tol: int,
    velocity_tol: int,
) -> Dict[str, float]:
    return metric_dict(
        df,
        df[f"{mode}_score"].to_numpy(float),
        df[f"{mode}_pred_range"].to_numpy(float),
        df[f"{mode}_pred_velocity"].to_numpy(float),
        threshold,
        range_tol,
        velocity_tol,
    )


def candidate_specs(alpha_step: float) -> list[dict[str, Any]]:
    alphas = np.arange(0.0, 1.0 + alpha_step / 2.0, alpha_step)
    specs: list[dict[str, Any]] = []
    for alpha in alphas:
        for score_rule in ("raw_weighted", "normalized_margin_weighted"):
            for coordinate_rule in (
                "higher_normalized_margin",
                "agreement_mean_else_margin",
                "alpha_weighted_coordinates",
            ):
                specs.append({
                    "score_rule": score_rule,
                    "coordinate_rule": coordinate_rule,
                    "alpha": float(round(alpha, 10)),
                })
    for score_rule in ("max_raw", "max_normalized_margin"):
        for coordinate_rule in (
            "higher_raw_score",
            "higher_normalized_margin",
            "agreement_mean_else_margin",
        ):
            specs.append({
                "score_rule": score_rule,
                "coordinate_rule": coordinate_rule,
                "alpha": 0.5,
            })
    return specs


def build_candidate(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    h_threshold: float,
    v_threshold: float,
    agreement_range: int,
    agreement_velocity: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = float(spec["alpha"])
    score = make_fused_score(
        df, str(spec["score_rule"]), alpha, h_threshold, v_threshold,
    )
    pr, pv, source = choose_coordinates(
        df,
        str(spec["coordinate_rule"]),
        alpha,
        h_threshold,
        v_threshold,
        agreement_range,
        agreement_velocity,
    )
    return score, pr, pv, source


def prediction_table(
    df: pd.DataFrame,
    score: np.ndarray,
    pred_range: np.ndarray,
    pred_velocity: np.ndarray,
    source: np.ndarray,
    threshold: float,
    range_tol: int,
    velocity_tol: int,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "sample_id": df["sample_id"].astype(str),
        "target_present": df["target_present"].astype(int),
        "true_range_index": df["true_range"],
        "true_velocity_index": df["true_velocity"],
        "H_score": df["H_score"],
        "V_score": df["V_score"],
        "H_pred_range_index": df["H_pred_range"],
        "V_pred_range_index": df["V_pred_range"],
        "H_pred_velocity_index": df["H_pred_velocity"],
        "V_pred_velocity_index": df["V_pred_velocity"],
        "fusion_score": score,
        "fusion_threshold": threshold,
        "fusion_detected": score >= threshold,
        "fusion_pred_range_index": pred_range,
        "fusion_pred_velocity_index": pred_velocity,
        "coordinate_source": source,
    })
    out["range_error_gates"] = np.abs(
        out["fusion_pred_range_index"] - out["true_range_index"]
    )
    out["velocity_error_bins"] = np.abs(
        out["fusion_pred_velocity_index"] - out["true_velocity_index"]
    )
    out["localization_ok"] = (
        (out["range_error_gates"] <= range_tol)
        & (out["velocity_error_bins"] <= velocity_tol)
    )
    out["joint_correct"] = (
        (out["target_present"] == 1)
        & out["fusion_detected"]
        & out["localization_ok"]
    )
    out["false_alarm"] = (
        (out["target_present"] == 0)
        & out["fusion_detected"]
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments/hv_late_fusion_diagnostic_v1"))
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--range-tol", type=int, default=2)
    parser.add_argument("--velocity-tol", type=int, default=3)
    parser.add_argument("--agreement-range", type=int, default=2)
    parser.add_argument("--agreement-velocity", type=int, default=3)
    parser.add_argument(
        "--pfa-cap",
        type=float,
        default=None,
        help="Validation Pfa cap. Default: use the existing HV validation Pfa.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds: Dict[str, float] = {}
    split_data: Dict[str, Dict[str, pd.DataFrame]] = {"val": {}, "test": {}}
    for mode, exp_name in EXPERIMENTS.items():
        tables = args.experiments_root / exp_name / "tables"
        summary_path = tables / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"缺少文件：{summary_path}")
        thresholds[mode] = threshold_from_summary(read_summary(summary_path))
        for split in ("val", "test"):
            path = tables / f"{split}_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(f"缺少文件：{path}")
            split_data[split][mode] = normalize_predictions(path, mode)

    merged: Dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        merged_hv = merge_modes(split_data[split]["H"], split_data[split]["V"])
        hv_extra = split_data[split]["HV"][[
            "sample_id",
            "HV_score",
            "HV_pred_range",
            "HV_pred_velocity",
        ]]
        merged[split] = merged_hv.merge(
            hv_extra, on="sample_id", how="inner", validate="one_to_one",
        )
        if len(merged[split]) != len(merged_hv):
            raise ValueError(f"{split} 中HV样本未完全对齐。")

    hv_val_baseline = baseline_metrics(
        merged["val"], "HV", thresholds["HV"], args.range_tol, args.velocity_tol,
    )
    pfa_cap = float(args.pfa_cap) if args.pfa_cap is not None else float(hv_val_baseline["pfa"])

    candidates = []
    for index, spec in enumerate(candidate_specs(args.alpha_step), start=1):
        score, pr, pv, source = build_candidate(
            merged["val"],
            spec,
            thresholds["H"],
            thresholds["V"],
            args.agreement_range,
            args.agreement_velocity,
        )
        best = select_threshold(
            merged["val"], score, pr, pv, pfa_cap,
            args.range_tol, args.velocity_tol,
        )
        row = {
            "candidate_id": index,
            **dict(spec),
            "validation_pfa_cap": pfa_cap,
            **best,
        }
        candidates.append(row)

    candidate_df = pd.DataFrame(candidates)
    candidate_df["alpha_distance_from_half"] = np.abs(candidate_df["alpha"] - 0.5)
    selected = candidate_df.sort_values(
        [
            "joint_pd",
            "score_detection_pd",
            "pfa",
            "range_mae_score_detected",
            "velocity_mae_score_detected",
            "alpha_distance_from_half",
            "candidate_id",
        ],
        ascending=[False, False, True, True, True, True, True],
        na_position="last",
    ).iloc[0]

    selected_spec = {
        "score_rule": str(selected["score_rule"]),
        "coordinate_rule": str(selected["coordinate_rule"]),
        "alpha": float(selected["alpha"]),
    }
    locked_threshold = float(selected["threshold"])

    selected_rows = []
    prediction_paths = {}
    for split in ("val", "test"):
        score, pr, pv, source = build_candidate(
            merged[split],
            selected_spec,
            thresholds["H"],
            thresholds["V"],
            args.agreement_range,
            args.agreement_velocity,
        )
        metrics = metric_dict(
            merged[split], score, pr, pv, locked_threshold,
            args.range_tol, args.velocity_tol,
        )
        selected_rows.append({
            "method": "H_V_validation_locked_late_fusion",
            "split": split,
            **selected_spec,
            **metrics,
        })
        pred_df = prediction_table(
            merged[split], score, pr, pv, source, locked_threshold,
            args.range_tol, args.velocity_tol,
        )
        pred_path = args.output_dir / f"{split}_fused_predictions.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        prediction_paths[split] = str(pred_path)

    comparison_rows = []
    for split in ("val", "test"):
        for mode in ("H", "V", "HV"):
            row = baseline_metrics(
                merged[split], mode, thresholds[mode],
                args.range_tol, args.velocity_tol,
            )
            comparison_rows.append({
                "method": mode,
                "split": split,
                "score_rule": "existing_baseline",
                "coordinate_rule": mode,
                "alpha": math.nan,
                **row,
            })
    comparison_rows.extend(selected_rows)

    candidate_df.drop(columns=["alpha_distance_from_half"]).to_csv(
        args.output_dir / "validation_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(
        args.output_dir / "selected_fusion_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        args.output_dir / "baseline_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_json = {
        "selection_source": "validation only",
        "test_used_for_selection": False,
        "pfa_cap_source": (
            "command line" if args.pfa_cap is not None
            else "existing HV validation Pfa"
        ),
        "validation_pfa_cap": pfa_cap,
        "range_tolerance_gates": args.range_tol,
        "velocity_tolerance_bins": args.velocity_tol,
        "agreement_range_gates": args.agreement_range,
        "agreement_velocity_bins": args.agreement_velocity,
        "base_thresholds": thresholds,
        "selected_candidate": {
            "candidate_id": int(selected["candidate_id"]),
            **selected_spec,
            "locked_threshold": locked_threshold,
            "validation_metrics": {
                key: (
                    int(selected[key])
                    if key in {
                        "positive_count", "background_count",
                        "score_detected_positive_count",
                        "correct_count", "false_alarm_count",
                    }
                    else float(selected[key])
                )
                for key in (
                    "positive_count", "background_count",
                    "score_detected_positive_count",
                    "score_detection_pd", "correct_count", "joint_pd",
                    "false_alarm_count", "pfa", "auc",
                    "range_mae_all_positive", "range_mae_score_detected",
                    "velocity_mae_all_positive", "velocity_mae_score_detected",
                )
            },
        },
        "prediction_files": prediction_paths,
    }
    (args.output_dir / "selected_fusion_rule.json").write_text(
        json.dumps(selected_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    val_row = selected_df[selected_df["split"] == "val"].iloc[0]
    test_row = selected_df[selected_df["split"] == "test"].iloc[0]
    hv_val = comparison_df[
        (comparison_df["method"] == "HV") & (comparison_df["split"] == "val")
    ].iloc[0]
    hv_test = comparison_df[
        (comparison_df["method"] == "HV") & (comparison_df["split"] == "test")
    ].iloc[0]

    conclusion = f"""# H/V验证集锁定晚期融合诊断

## 实验纪律

- 候选融合规则、alpha和融合阈值仅使用验证集选择。
- 验证集Pfa上限：{pfa_cap:.6f}，来源：{'命令行' if args.pfa_cap is not None else '现有HV验证集Pfa'}。
- 测试集仅进行一次锁定规则诊断，不参与任何选择。
- 本结果是决策级低成本诊断，不替代后续可训练的H/V独立编码双分支模型。

## 锁定规则

- score_rule：{selected_spec['score_rule']}
- coordinate_rule：{selected_spec['coordinate_rule']}
- alpha：{selected_spec['alpha']:.3f}
- locked_threshold：{locked_threshold:.9f}

## 验证集

- 晚期融合：joint Pd={val_row['joint_pd']:.6f}，Pfa={val_row['pfa']:.6f}，AUC={val_row['auc']:.6f}
- 现有HV：joint Pd={hv_val['joint_pd']:.6f}，Pfa={hv_val['pfa']:.6f}，AUC={hv_val['auc']:.6f}

## 测试集一次性诊断

- 晚期融合：joint Pd={test_row['joint_pd']:.6f}，Pfa={test_row['pfa']:.6f}，AUC={test_row['auc']:.6f}
- 现有HV：joint Pd={hv_test['joint_pd']:.6f}，Pfa={hv_test['pfa']:.6f}，AUC={hv_test['auc']:.6f}

## 解释规则

1. 若验证集与测试集均优于现有HV，说明H/V互补可以被简单晚期融合利用，下一步应建立可训练的独立编码双分支。
2. 若验证集提升、测试集下降，说明小验证集上存在选择过拟合，不能把该规则作为正式改进。
3. 若验证集也不优于HV，则决策级融合不足，应直接比较可训练双分支与RI4输入。
4. 无论结果如何，都不得根据测试结果重新修改alpha、阈值或坐标规则。
"""
    (args.output_dir / "README_晚期融合结论.md").write_text(
        conclusion, encoding="utf-8",
    )

    print("=" * 78)
    print("H/V验证集锁定晚期融合诊断完成")
    print(
        f"锁定规则：score={selected_spec['score_rule']}, "
        f"coord={selected_spec['coordinate_rule']}, "
        f"alpha={selected_spec['alpha']:.3f}, "
        f"threshold={locked_threshold:.9f}"
    )
    print(
        f"验证集：joint Pd={val_row['joint_pd']:.4f}, "
        f"Pfa={val_row['pfa']:.4f}, AUC={val_row['auc']:.4f}"
    )
    print(
        f"测试集：joint Pd={test_row['joint_pd']:.4f}, "
        f"Pfa={test_row['pfa']:.4f}, AUC={test_row['auc']:.4f}"
    )
    print(f"结果目录：{args.output_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
