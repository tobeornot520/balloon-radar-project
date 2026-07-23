#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("power2", "ri4", "polar6_gated", "ri8_gated")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", nargs="+", type=int, required=True)
    p.add_argument("--modes", nargs="+", choices=MODES, required=True)
    p.add_argument("--scope", choices=["smoke", "formal"], required=True)
    p.add_argument("--require-all", action="store_true")
    return p.parse_args()


def safe(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def main() -> None:
    args = parse_args()
    out = PROJECT_ROOT / "results/data_audit/polarimetric_representation_benchmark_v2"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for fold in args.folds:
        for mode in args.modes:
            name = f"polar_repr_v2_{mode}_v4_fold{fold:02d}_seed42_{args.scope}"
            path = PROJECT_ROOT / "results/experiments" / name / "tables/summary.json"
            if not path.is_file():
                missing.append(str(path.relative_to(PROJECT_ROOT)))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            test = data["test_metrics"]
            val = data["validation_metrics"]
            rows.append({
                "scope": args.scope,
                "fold": fold,
                "mode": mode,
                "experiment_name": name,
                "parameter_count": data["parameter_count"],
                "best_epoch": data["best_epoch"],
                "threshold": data["validation_threshold"],
                "val_pd": safe(val.get("joint_pd")),
                "val_pfa": safe(val.get("pfa")),
                "val_auc": safe(val.get("roc_auc")),
                "test_positive_count": int(test.get("positive_count", 0)),
                "test_background_count": int(test.get("background_count", 0)),
                "test_false_alarm_count": int(test.get("false_alarm_count", 0)),
                "test_pfa": safe(test.get("pfa")),
                "test_pd": safe(test.get("joint_pd")),
                "test_auc": safe(test.get("roc_auc")),
                "test_range_mae": safe(test.get("all_positive_range_mae_gates")),
                "test_velocity_mae": safe(test.get("all_positive_velocity_mae_bins")),
            })
    detail = pd.DataFrame(rows)
    detail_path = out / f"representation_detail_{args.scope}.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    if detail.empty:
        aggregate = pd.DataFrame()
    else:
        aggregate = detail.groupby("mode", as_index=False).agg(
            folds=("fold", "nunique"),
            total_positive=("test_positive_count", "sum"),
            total_background=("test_background_count", "sum"),
            total_false_alarms=("test_false_alarm_count", "sum"),
            mean_pfa=("test_pfa", "mean"),
            mean_pd=("test_pd", "mean"),
            mean_auc=("test_auc", "mean"),
            mean_range_mae=("test_range_mae", "mean"),
            mean_velocity_mae=("test_velocity_mae", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        if "power2" in set(aggregate["mode"]):
            base = aggregate.loc[aggregate["mode"] == "power2"].iloc[0]
            aggregate["delta_false_alarms_vs_power2"] = aggregate["total_false_alarms"] - int(base["total_false_alarms"])
            aggregate["delta_mean_pd_vs_power2"] = aggregate["mean_pd"] - float(base["mean_pd"])
            aggregate["delta_mean_auc_vs_power2"] = aggregate["mean_auc"] - float(base["mean_auc"])
    aggregate_path = out / f"representation_aggregate_{args.scope}.csv"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    status = {
        "scope": args.scope,
        "requested": len(args.folds) * len(args.modes),
        "found": len(rows),
        "missing": len(missing),
        "missing_paths": missing,
        "selection_rule": "No test-set model selection. This table is descriptive representation comparison.",
    }
    (out / "latest_run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# 功率门控显式极化表征基准（{args.scope}）", "",
        "所有模式均使用固定8通道输入网络；未使用扫描组上下文。", "",
        "> Power2、RI4、Polar6-gated、RI8-gated仅比较输入表征，不代表已完成物理极化标定。", "",
        aggregate.to_markdown(index=False, floatfmt=".4f") if not aggregate.empty else "暂无结果。",
        "", f"found={len(rows)} missing={len(missing)}",
    ]
    (out / f"README_representation_benchmark_{args.scope}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("=" * 82)
    print(f"representation summary ({args.scope})")
    print(f"found   : {len(rows)}")
    print(f"missing : {len(missing)}")
    print(f"detail  : {detail_path}")
    print(f"aggregate: {aggregate_path}")
    print("=" * 82)
    if missing and args.require_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
