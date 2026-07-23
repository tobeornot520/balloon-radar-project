#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("power2_baseline", "power2_roi_power_control", "power2_roi_ri4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, required=True)
    parser.add_argument("--scope", choices=["smoke", "formal"], required=True)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def experiment_name(mode: str, fold: int, scope: str) -> str:
    return f"roi_polar_stage4_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"


def bootstrap(values: pd.Series, iterations: int = 20000) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    samples = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    args = parse_args()
    output = PROJECT_ROOT / "results/data_audit/roi_stage4_selected_sixfold_v1"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing: list[str] = []
    predictions: dict[tuple[int, str], pd.DataFrame] = {}

    for fold in args.folds:
        for mode in args.modes:
            root = PROJECT_ROOT / "results/experiments" / experiment_name(mode, fold, args.scope) / "tables"
            summary_path = root / "summary.json"
            val_path = root / "val_predictions.csv"
            test_path = root / "test_predictions.csv"
            if not summary_path.is_file() or not val_path.is_file() or not test_path.is_file():
                missing.append(str(root.relative_to(PROJECT_ROOT)))
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            refined = summary["refined_test_fixed_threshold_metrics"]
            selected = summary["refined_test_selected_threshold_metrics"]
            val = pd.read_csv(val_path)
            test = pd.read_csv(test_path)
            predictions[(fold, mode)] = test
            val_bg = val.loc[val["target_present"] == 0, "refined_score"].to_numpy(dtype=float)
            test_bg = test.loc[test["target_present"] == 0, "refined_score"].to_numpy(dtype=float)
            q99 = float(np.quantile(val_bg, 0.99))
            rows.append({
                "fold": fold,
                "mode": mode,
                "parameter_count": int(summary["parameter_count"]),
                "best_epoch": int(summary["best_epoch"]),
                "base_threshold": float(summary["base_power2_threshold"]),
                "selected_threshold": float(summary["selected_refined_validation_threshold"]),
                "false_alarms": int(refined["false_alarm_count"]),
                "fixed_pfa": float(refined["pfa"]),
                "fixed_score_pd": float(refined["score_detection_pd"]),
                "fixed_joint_pd": float(refined["joint_pd"]),
                "auc": float(refined["roc_auc"]),
                "pauc_5pct": float(refined["partial_auc_5pct_fpr"]),
                "tpr_1pct": float(refined["tpr_at_1pct_fpr"]),
                "tpr_5pct": float(refined["tpr_at_5pct_fpr"]),
                "range_mae": float(refined["range_mae_gates"]),
                "velocity_mae": float(refined["velocity_mae_bins"]),
                "selected_test_pfa": float(selected["pfa"]),
                "selected_test_joint_pd": float(selected["joint_pd"]),
                "validation_background_q99": q99,
                "test_background_exceed_q99_fraction": float((test_bg >= q99).mean()),
                "background_median_shift": float(np.median(test_bg) - np.median(val_bg)),
                "background_q95_shift": float(np.quantile(test_bg, 0.95) - np.quantile(val_bg, 0.95)),
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(output / f"detail_{args.scope}.csv", index=False, encoding="utf-8-sig")
    if detail.empty:
        raise SystemExit(2)

    metrics = [
        "false_alarms", "fixed_pfa", "fixed_score_pd", "fixed_joint_pd",
        "auc", "pauc_5pct", "tpr_1pct", "tpr_5pct", "range_mae",
        "velocity_mae", "test_background_exceed_q99_fraction",
        "background_median_shift", "background_q95_shift",
    ]
    aggregate = detail.groupby("mode", as_index=False).agg(
        folds=("fold", "nunique"),
        **{f"mean_{metric}": (metric, "mean") for metric in metrics},
        total_false_alarms=("false_alarms", "sum"),
        parameter_count=("parameter_count", "first"),
    )
    baseline = aggregate.loc[aggregate["mode"] == "power2_baseline"].iloc[0]
    aggregate["delta_total_false_alarms_vs_power2"] = (
        aggregate["total_false_alarms"] - int(baseline["total_false_alarms"])
    )
    aggregate["delta_mean_joint_pd_vs_power2"] = (
        aggregate["mean_fixed_joint_pd"] - float(baseline["mean_fixed_joint_pd"])
    )
    aggregate["delta_mean_pauc_vs_power2"] = (
        aggregate["mean_pauc_5pct"] - float(baseline["mean_pauc_5pct"])
    )
    aggregate.to_csv(output / f"aggregate_{args.scope}.csv", index=False, encoding="utf-8-sig")

    paired_rows: list[dict] = []
    for fold in args.folds:
        baseline_df = predictions.get((fold, "power2_baseline"))
        if baseline_df is None:
            continue
        baseline_df = baseline_df.set_index("sample_id")
        for mode in args.modes:
            if mode == "power2_baseline" or (fold, mode) not in predictions:
                continue
            candidate = predictions[(fold, mode)].set_index("sample_id")
            common = baseline_df.index.intersection(candidate.index)
            base = baseline_df.loc[common]
            ref = candidate.loc[common]
            base_fa = base["raw_fixed_false_alarm"].astype(bool)
            ref_fa = ref["refined_fixed_false_alarm"].astype(bool)
            base_cd = base["raw_fixed_correct_detection"].astype(bool)
            ref_cd = ref["refined_fixed_correct_detection"].astype(bool)
            paired_rows.append({
                "fold": fold,
                "mode": mode,
                "false_alarms_removed": int((base_fa & ~ref_fa).sum()),
                "false_alarms_added": int((~base_fa & ref_fa).sum()),
                "target_correct_preserved": int((base_cd & ref_cd).sum()),
                "target_correct_regressed": int((base_cd & ~ref_cd).sum()),
                "target_correct_rescued": int((~base_cd & ref_cd).sum()),
            })
    pd.DataFrame(paired_rows).to_csv(
        output / f"paired_effects_{args.scope}.csv", index=False, encoding="utf-8-sig"
    )

    ci_rows: list[dict] = []
    for mode in args.modes:
        subset = detail.loc[detail["mode"] == mode]
        for metric in [
            "fixed_pfa", "fixed_joint_pd", "auc", "pauc_5pct",
            "tpr_5pct", "test_background_exceed_q99_fraction",
        ]:
            mean, low, high = bootstrap(subset[metric])
            ci_rows.append({
                "mode": mode, "metric": metric, "folds": len(subset),
                "mean": mean, "bootstrap_95_low": low, "bootstrap_95_high": high,
            })
    pd.DataFrame(ci_rows).to_csv(
        output / f"fold_bootstrap_ci_{args.scope}.csv", index=False, encoding="utf-8-sig"
    )

    consistency_rows: list[dict] = []
    baseline_detail = detail.loc[detail["mode"] == "power2_baseline"].set_index("fold")
    for mode in args.modes:
        subset = detail.loc[detail["mode"] == mode].set_index("fold")
        common = subset.index.intersection(baseline_detail.index)
        consistency_rows.append({
            "mode": mode,
            "folds": len(common),
            "folds_false_alarms_not_worse": int(
                (subset.loc[common, "false_alarms"] <= baseline_detail.loc[common, "false_alarms"]).sum()
            ),
            "folds_joint_pd_not_worse": int(
                (subset.loc[common, "fixed_joint_pd"] >= baseline_detail.loc[common, "fixed_joint_pd"] - 1e-12).sum()
            ),
            "folds_pauc_improved": int(
                (subset.loc[common, "pauc_5pct"] > baseline_detail.loc[common, "pauc_5pct"]).sum()
            ),
            "max_joint_pd_drop": float(
                (baseline_detail.loc[common, "fixed_joint_pd"] - subset.loc[common, "fixed_joint_pd"]).max()
            ),
        })
    pd.DataFrame(consistency_rows).to_csv(
        output / f"fold_consistency_{args.scope}.csv", index=False, encoding="utf-8-sig"
    )

    status = {
        "scope": args.scope,
        "requested": len(args.folds) * len(args.modes),
        "found": len(detail),
        "missing": missing,
        "selection_frozen_before_sixfold": True,
        "test_set_model_selection_forbidden": True,
    }
    (output / f"status_{args.scope}.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = (
        "# Stage 4 selected six-fold summary\n\n"
        + aggregate.to_markdown(index=False, floatfmt=".4f")
        + f"\n\nfound={len(detail)} missing={len(missing)}\n"
    )
    (output / f"README_{args.scope}.md").write_text(readme, encoding="utf-8")
    print(f"found={len(detail)} missing={len(missing)} output={output}")
    if missing and args.require_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
