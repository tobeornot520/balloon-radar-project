#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_MODES = ("power2", "ri4", "polar6", "ri8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit validation-to-test score shift for v1 polarimetric benchmarks")
    p.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    p.add_argument("--modes", nargs="+", choices=V1_MODES, default=list(V1_MODES))
    p.add_argument("--scope", choices=["formal", "smoke"], default="formal")
    p.add_argument("--require-all", action="store_true")
    return p.parse_args()


def quantiles(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {k: math.nan for k in ("min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max", "mean", "std")}
    return {
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if negatives.size == 0 or positives.size == 0:
        return math.nan
    threshold = float(np.quantile(negatives, 1.0 - max_fpr, method="higher"))
    return float(np.mean(positives >= threshold))


def safe_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float | None = None) -> float:
    if np.unique(labels).size < 2:
        return math.nan
    try:
        return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    except ValueError:
        return math.nan


def load_manifest(fold: int) -> pd.DataFrame:
    path = PROJECT_ROOT / f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
    frame = pd.read_csv(path)
    keep = [c for c in ("sample_id", "new_split", "source_file", "class_name", "beam_layer", "azimuth_deg") if c in frame.columns]
    return frame[keep].copy()


def main() -> None:
    args = parse_args()
    out = PROJECT_ROOT / "results/data_audit/polarimetric_score_shift_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    distribution_rows: list[dict] = []
    transfer_rows: list[dict] = []
    source_rows: list[dict] = []
    missing: list[str] = []

    for fold in args.folds:
        manifest = load_manifest(fold)
        for mode in args.modes:
            name = f"polar_repr_v1_{mode}_v4_fold{fold:02d}_seed42_{args.scope}"
            root = PROJECT_ROOT / "results/experiments" / name / "tables"
            summary_path = root / "summary.json"
            val_path = root / "val_predictions.csv"
            test_path = root / "test_predictions.csv"
            required = [summary_path, val_path, test_path]
            absent = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.is_file()]
            if absent:
                missing.extend(absent)
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            threshold = float(summary["validation_threshold"])
            split_frames = {}
            for split, path in (("val", val_path), ("test", test_path)):
                frame = pd.read_csv(path)
                if not {"sample_id", "target_present", "score"}.issubset(frame.columns):
                    raise ValueError(f"Unexpected prediction columns in {path}")
                frame = frame.merge(
                    manifest.loc[manifest["new_split"] == split].drop(columns=["new_split"]),
                    on="sample_id",
                    how="left",
                    validate="one_to_one",
                )
                split_frames[split] = frame
                for target_present, class_label in ((0, "background"), (1, "target")):
                    values = frame.loc[frame["target_present"] == target_present, "score"].to_numpy(float)
                    row = {
                        "fold": fold,
                        "mode": mode,
                        "split": split,
                        "class_label": class_label,
                        "count": int(values.size),
                        "threshold": threshold,
                        "fraction_above_threshold": float(np.mean(values >= threshold)) if values.size else math.nan,
                    }
                    row.update(quantiles(values))
                    distribution_rows.append(row)

                labels = frame["target_present"].to_numpy(int)
                scores = frame["score"].to_numpy(float)
                transfer_rows.append({
                    "fold": fold,
                    "mode": mode,
                    "split": split,
                    "threshold": threshold,
                    "auc": safe_auc(labels, scores),
                    "partial_auc_fpr_0_05": safe_auc(labels, scores, 0.05),
                    "partial_auc_fpr_0_10": safe_auc(labels, scores, 0.10),
                    "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores, 0.01),
                    "tpr_at_5pct_fpr": tpr_at_fpr(labels, scores, 0.05),
                    "background_exceedance_at_val_threshold": float(np.mean(scores[labels == 0] >= threshold)),
                    "target_exceedance_at_val_threshold": float(np.mean(scores[labels == 1] >= threshold)),
                })

            val_bg = split_frames["val"].loc[split_frames["val"]["target_present"] == 0, "score"].to_numpy(float)
            test_bg = split_frames["test"].loc[split_frames["test"]["target_present"] == 0, "score"].to_numpy(float)
            transfer_rows[-1]["val_background_q99"] = float(np.quantile(val_bg, 0.99))
            transfer_rows[-1]["test_background_q99"] = float(np.quantile(test_bg, 0.99))
            transfer_rows[-1]["test_to_val_background_median_ratio"] = float(
                np.median(test_bg) / max(np.median(val_bg), 1e-12)
            )

            for split, frame in split_frames.items():
                if "source_file" not in frame.columns:
                    continue
                for (source_file, target_present), group in frame.groupby(["source_file", "target_present"], dropna=False):
                    values = group["score"].to_numpy(float)
                    source_rows.append({
                        "fold": fold,
                        "mode": mode,
                        "split": split,
                        "source_file": source_file,
                        "class_label": "target" if int(target_present) else "background",
                        "count": int(len(group)),
                        "score_mean": float(np.mean(values)),
                        "score_q50": float(np.quantile(values, 0.50)),
                        "score_q95": float(np.quantile(values, 0.95)),
                        "score_max": float(np.max(values)),
                        "fraction_above_threshold": float(np.mean(values >= threshold)),
                    })

    distributions = pd.DataFrame(distribution_rows)
    transfer = pd.DataFrame(transfer_rows)
    sources = pd.DataFrame(source_rows)
    distributions.to_csv(out / "score_distribution_quantiles.csv", index=False, encoding="utf-8-sig")
    transfer.to_csv(out / "threshold_transfer_and_low_fpr_metrics.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(out / "source_file_score_stratification.csv", index=False, encoding="utf-8-sig")

    test_rows = transfer.loc[transfer["split"] == "test"].copy() if not transfer.empty else transfer
    aggregate = test_rows.groupby("mode", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_auc=("auc", "mean"),
        mean_partial_auc_fpr_0_05=("partial_auc_fpr_0_05", "mean"),
        mean_tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "mean"),
        mean_tpr_at_5pct_fpr=("tpr_at_5pct_fpr", "mean"),
        mean_background_exceedance=("background_exceedance_at_val_threshold", "mean"),
        mean_target_exceedance=("target_exceedance_at_val_threshold", "mean"),
    ) if not test_rows.empty else pd.DataFrame()
    aggregate.to_csv(out / "score_shift_aggregate.csv", index=False, encoding="utf-8-sig")
    status = {
        "status": "PASS" if not missing else "INCOMPLETE",
        "requested": len(args.folds) * len(args.modes),
        "found": len(transfer_rows) // 2,
        "missing": len(missing),
        "missing_paths": missing,
        "note": "Low-FPR test metrics are descriptive diagnostics and are not used for model selection.",
    }
    (out / "latest_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 极化表征验证—测试分数迁移诊断", "",
        "该审计复用既有预测，不重新训练。测试集低FPR指标只用于诊断，不用于选择模型。", "",
        aggregate.to_markdown(index=False, floatfmt=".4f") if not aggregate.empty else "暂无完整结果。", "",
        f"found={status['found']} missing={status['missing']}",
    ]
    (out / "README_score_shift_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("=" * 82)
    print("polarimetric score-shift audit")
    print(f"found   : {status['found']}")
    print(f"missing : {status['missing']}")
    print(f"output  : {out}")
    print("=" * 82)
    if missing and args.require_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
