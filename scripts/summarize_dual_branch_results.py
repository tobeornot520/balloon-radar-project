#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def nested(data, *keys):
    value = data
    for key in keys:
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总DPG-FCN多随机种子与现有H/V/HV基线")
    parser.add_argument("--prefix", default="dpg_fcn_v1")
    parser.add_argument("--output", default="results/final/dpg_fcn_v1_summary.csv")
    args = parser.parse_args()

    rows = []
    baseline_names = ["detection_h_baseline_v2", "detection_v_baseline_v2", "detection_hv_baseline_v2"]
    for name in baseline_names:
        path = PROJECT_ROOT / "results" / "experiments" / name / "tables" / "summary.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            m = data["test_metrics"]
            rows.append({
                "method": name, "seed": data.get("config", {}).get("seed"),
                "correct_count": m["correct_detection_count"], "joint_pd": m["joint_pd"],
                "false_alarm_count": m["false_alarm_count"], "pfa": m["pfa"], "auc": m["roc_auc"],
                "range_mae_detected": m.get("detected_positive_range_mae_gates"),
                "velocity_mae_detected": m.get("detected_positive_velocity_mae_bins"),
                "best_epoch": data.get("best_epoch"), "parameter_count": data.get("parameter_count"),
            })

    experiments_root = PROJECT_ROOT / "results" / "experiments"
    for path in sorted(experiments_root.glob(f"{args.prefix}_seed*/tables/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data["test_metrics"]
        rows.append({
            "method": data["experiment_name"], "seed": data.get("config", {}).get("seed"),
            "correct_count": m["correct_detection_count"], "joint_pd": m["joint_pd"],
            "false_alarm_count": m["false_alarm_count"], "pfa": m["pfa"], "auc": m["roc_auc"],
            "range_mae_detected": m.get("detected_positive_range_mae_gates"),
            "velocity_mae_detected": m.get("detected_positive_velocity_mae_bins"),
            "best_epoch": data.get("best_epoch"), "parameter_count": data.get("parameter_count"),
        })
    if not rows:
        raise FileNotFoundError("未找到可汇总的summary.json")

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    dpg = frame[frame["method"].str.startswith(args.prefix)]
    aggregate = {}
    if len(dpg):
        for column in ("correct_count", "joint_pd", "false_alarm_count", "pfa", "auc", "range_mae_detected", "velocity_mae_detected"):
            aggregate[column] = {"mean": float(dpg[column].mean()), "std": float(dpg[column].std(ddof=0))}
    aggregate_path = output.with_name(output.stem + "_aggregate.json")
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\n汇总CSV：{output}")
    print(f"多种子均值与标准差：{aggregate_path}")


if __name__ == "__main__":
    main()
