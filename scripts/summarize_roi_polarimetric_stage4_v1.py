#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "power2_baseline",
    "power2_roi_power_control",
    "power2_roi_ri4",
    "power2_roi_polar6_gated",
    "power2_roi_ri4_polar6_gated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, required=True)
    parser.add_argument("--scope", choices=["smoke", "formal"], required=True)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def experiment_name(mode: str, fold: int, scope: str) -> str:
    return f"roi_polar_stage4_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"


def safe(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def threshold_transfer(val: pd.DataFrame, test: pd.DataFrame, column: str) -> dict[str, float]:
    val_bg = val.loc[val["target_present"] == 0, column].to_numpy(dtype=float)
    test_bg = test.loc[test["target_present"] == 0, column].to_numpy(dtype=float)
    if val_bg.size == 0 or test_bg.size == 0:
        return {
            "validation_background_q99": float("nan"),
            "test_background_exceed_q99_fraction": float("nan"),
            "background_median_shift": float("nan"),
            "background_q95_shift": float("nan"),
        }
    q99 = float(np.quantile(val_bg, 0.99))
    return {
        "validation_background_q99": q99,
        "test_background_exceed_q99_fraction": float((test_bg >= q99).mean()),
        "background_median_shift": float(np.median(test_bg) - np.median(val_bg)),
        "background_q95_shift": float(np.quantile(test_bg, 0.95) - np.quantile(val_bg, 0.95)),
    }


def main() -> None:
    args = parse_args()
    output = PROJECT_ROOT / "results/data_audit/roi_polarimetric_stage4_v1"
    output.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    predictions: dict[tuple[int, str], pd.DataFrame] = {}

    for fold in args.folds:
        for mode in args.modes:
            name = experiment_name(mode, fold, args.scope)
            root = PROJECT_ROOT / "results/experiments" / name
            summary_path = root / "tables/summary.json"
            val_path = root / "tables/val_predictions.csv"
            test_path = root / "tables/test_predictions.csv"
            if not summary_path.is_file() or not val_path.is_file() or not test_path.is_file():
                missing.append(str(root.relative_to(PROJECT_ROOT)))
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            raw = summary["raw_test_fixed_threshold_metrics"]
            refined = summary["refined_test_fixed_threshold_metrics"]
            selected = summary["refined_test_selected_threshold_metrics"]
            detail_rows.append({
                "scope": args.scope,
                "fold": fold,
                "mode": mode,
                "experiment_name": name,
                "parameter_count": int(summary["parameter_count"]),
                "best_epoch": int(summary["best_epoch"]),
                "base_threshold": safe(summary["base_power2_threshold"]),
                "selected_threshold": safe(summary["selected_refined_validation_threshold"]),
                "raw_false_alarms": int(raw["false_alarm_count"]),
                "refined_false_alarms": int(refined["false_alarm_count"]),
                "refined_fixed_pfa": safe(refined["pfa"]),
                "refined_fixed_score_pd": safe(refined["score_detection_pd"]),
                "refined_fixed_joint_pd": safe(refined["joint_pd"]),
                "refined_auc": safe(refined["roc_auc"]),
                "refined_pauc_5pct": safe(refined["partial_auc_5pct_fpr"]),
                "refined_tpr_1pct": safe(refined["tpr_at_1pct_fpr"]),
                "refined_tpr_5pct": safe(refined["tpr_at_5pct_fpr"]),
                "range_mae": safe(refined["all_positive_range_mae_gates"]),
                "velocity_mae": safe(refined["all_positive_velocity_mae_bins"]),
                "selected_test_pfa": safe(selected["pfa"]),
                "selected_test_joint_pd": safe(selected["joint_pd"]),
                "delta_false_alarms_vs_raw": int(refined["false_alarm_count"]) - int(raw["false_alarm_count"]),
                "delta_joint_pd_vs_raw": safe(refined["joint_pd"]) - safe(raw["joint_pd"]),
            })
            val = pd.read_csv(val_path)
            test = pd.read_csv(test_path)
            predictions[(fold, mode)] = test
            transfer = threshold_transfer(val, test, "refined_score")
            transfer_rows.append({"scope": args.scope, "fold": fold, "mode": mode, **transfer})

    detail = pd.DataFrame(detail_rows)
    transfer = pd.DataFrame(transfer_rows)
    aggregate = pd.DataFrame()
    if not detail.empty:
        aggregate = detail.groupby("mode", as_index=False).agg(
            folds=("fold", "nunique"),
            total_raw_false_alarms=("raw_false_alarms", "sum"),
            total_refined_false_alarms=("refined_false_alarms", "sum"),
            mean_fixed_pfa=("refined_fixed_pfa", "mean"),
            mean_fixed_score_pd=("refined_fixed_score_pd", "mean"),
            mean_fixed_joint_pd=("refined_fixed_joint_pd", "mean"),
            mean_auc=("refined_auc", "mean"),
            mean_pauc_5pct=("refined_pauc_5pct", "mean"),
            mean_tpr_1pct=("refined_tpr_1pct", "mean"),
            mean_tpr_5pct=("refined_tpr_5pct", "mean"),
            mean_range_mae=("range_mae", "mean"),
            mean_velocity_mae=("velocity_mae", "mean"),
            mean_selected_test_pfa=("selected_test_pfa", "mean"),
            mean_selected_test_joint_pd=("selected_test_joint_pd", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        baseline = aggregate.loc[aggregate["mode"] == "power2_baseline"]
        if not baseline.empty:
            base = baseline.iloc[0]
            aggregate["delta_false_alarms_vs_power2"] = (
                aggregate["total_refined_false_alarms"] - int(base["total_refined_false_alarms"])
            )
            aggregate["delta_joint_pd_vs_power2"] = (
                aggregate["mean_fixed_joint_pd"] - float(base["mean_fixed_joint_pd"])
            )
            aggregate["delta_pauc_vs_power2"] = (
                aggregate["mean_pauc_5pct"] - float(base["mean_pauc_5pct"])
            )

    for fold in args.folds:
        baseline = predictions.get((fold, "power2_baseline"))
        if baseline is None:
            continue
        base = baseline.set_index("sample_id")
        for mode in args.modes:
            if mode == "power2_baseline" or (fold, mode) not in predictions:
                continue
            candidate = predictions[(fold, mode)].set_index("sample_id")
            common = base.index.intersection(candidate.index)
            b = base.loc[common]
            c = candidate.loc[common]
            base_fa = b["raw_fixed_false_alarm"].astype(bool)
            ref_fa = c["refined_fixed_false_alarm"].astype(bool)
            base_cd = b["raw_fixed_correct_detection"].astype(bool)
            ref_cd = c["refined_fixed_correct_detection"].astype(bool)
            pair_rows.append({
                "scope": args.scope,
                "fold": fold,
                "mode": mode,
                "samples": len(common),
                "background_false_alarms_removed": int((base_fa & ~ref_fa).sum()),
                "background_false_alarms_added": int((~base_fa & ref_fa).sum()),
                "target_correct_detections_preserved": int((base_cd & ref_cd).sum()),
                "target_correct_detections_regressed": int((base_cd & ~ref_cd).sum()),
                "target_correct_detections_rescued": int((~base_cd & ref_cd).sum()),
            })
    pairwise = pd.DataFrame(pair_rows)

    suffix = args.scope
    detail.to_csv(output / f"stage4_detail_{suffix}.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(output / f"stage4_aggregate_{suffix}.csv", index=False, encoding="utf-8-sig")
    transfer.to_csv(output / f"stage4_threshold_transfer_{suffix}.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output / f"stage4_rescue_regression_{suffix}.csv", index=False, encoding="utf-8-sig")

    status = {
        "scope": args.scope,
        "requested": len(args.folds) * len(args.modes),
        "found": len(detail_rows),
        "missing": len(missing),
        "missing_paths": missing,
        "primary_evaluation": "unchanged Power2 deployment threshold",
        "sample_independent": True,
        "scan_context": False,
        "power2_location_frozen": True,
        "test_set_model_selection_forbidden": True,
    }
    (output / f"latest_run_status_{suffix}.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Stage 4候选区域极化精修汇总（{suffix}）",
        "",
        "主评价使用各折冻结Power2 checkpoint中的原始部署阈值；候选距离和速度位置始终沿用Power2。",
        "",
        aggregate.to_markdown(index=False, floatfmt=".4f") if not aggregate.empty else "暂无结果。",
        "",
        f"found={len(detail_rows)} missing={len(missing)}",
        "",
        "> smoke只验证接口，不用于表征选择。正式结果也仍属于当前预实验数据的内部评估。",
    ]
    (output / f"README_stage4_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("=" * 90)
    print(f"Stage 4 summary ({suffix})")
    print(f"found   : {len(detail_rows)}")
    print(f"missing : {len(missing)}")
    print(f"output  : {output}")
    print("=" * 90)
    if missing and args.require_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
