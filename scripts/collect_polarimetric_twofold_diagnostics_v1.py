#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集 Fold 1 / Fold 4 四种极化表征的完整诊断上下文，并自动打包验收ZIP。

核心原则：
1. 只读现有正式结果，不重新训练；
2. 不复制大模型权重和原始数据；
3. 自动发现逐样本预测CSV；
4. 若能识别标签与分数字段，则直接计算低FPR与阈值迁移指标；
5. 若逐样本预测不存在，则收集真实源码、checkpoint键名和CSV字段，
   为下一步精确生成预测导出器提供足够上下文。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    roc_auc_score = None
    roc_curve = None

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--modes", nargs="+",
        default=["power2", "ri4", "polar6_gated", "ri8_gated"]
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-copy-size-mb", type=float, default=None)
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def load_config(root: Path, explicit: Optional[Path]) -> dict[str, Any]:
    path = explicit or root / "configs" / "polarimetric_twofold_diagnostics_v1.json"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_project_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    required = ["results", "scripts", "training", "configs", "models"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"项目根目录不完整，缺少：{missing}；当前路径：{root}")
    return root


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        if columns:
            for col in columns:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[columns + [c for c in df.columns if c not in columns]]
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_safe(path: Path) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding), None
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    return None, "无法识别CSV编码"


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def infer_split_from_name(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "validation" in name or re.search(r"(^|[_-])val([_-]|$)", name):
        return "validation"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return None


def normalize_binary_target(series: pd.Series) -> Optional[np.ndarray]:
    if series.empty:
        return None
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).to_numpy()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        values = numeric.fillna(0).to_numpy()
        uniques = set(np.unique(values).tolist())
        if uniques.issubset({0, 1}):
            return values.astype(int)
    mapping = {
        "background": 0, "bg": 0, "negative": 0, "false": 0, "no": 0,
        "target": 1, "uav": 1, "positive": 1, "true": 1, "yes": 1,
    }
    mapped = series.astype(str).str.strip().str.lower().map(mapping)
    if mapped.notna().mean() > 0.95:
        return mapped.fillna(0).astype(int).to_numpy()
    return None


def calculate_tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    if roc_curve is None or len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, score)
    valid = np.where(fpr <= target_fpr + 1e-12)[0]
    return float(np.max(tpr[valid])) if valid.size else 0.0


def calculate_partial_auc(y_true: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    if roc_auc_score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, score, max_fpr=max_fpr))
    except Exception:
        return float("nan")


def discover_experiment_dirs(
    root: Path, folds: list[int], modes: list[str], template: str
) -> list[dict[str, Any]]:
    result_root = root / "results" / "experiments"
    rows = []
    for fold in folds:
        for mode in modes:
            expected_name = template.format(mode=mode, fold=fold)
            expected = result_root / expected_name
            candidates = [expected]
            if not expected.exists():
                pattern = f"*{mode}*fold{fold:02d}*formal*"
                candidates.extend(sorted(result_root.glob(pattern)))
            found = next((p for p in candidates if p.exists() and p.is_dir()), None)
            rows.append({
                "fold": fold,
                "mode": mode,
                "expected_name": expected_name,
                "found": bool(found),
                "experiment_dir": str(found) if found else "",
            })
    return rows


def file_category(path: Path) -> str:
    name = path.name.lower()
    if "prediction" in name or "score" in name or "inference" in name:
        return "prediction_candidate"
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return "config_or_metrics"
    if path.suffix.lower() in {".md", ".txt", ".log"}:
        return "text"
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    return "other"


def inventory_experiment_files(root: Path, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for exp in experiments:
        if not exp["found"]:
            continue
        exp_dir = Path(exp["experiment_dir"])
        for path in sorted(p for p in exp_dir.rglob("*") if p.is_file()):
            try:
                stat = path.stat()
                rows.append({
                    "fold": exp["fold"],
                    "mode": exp["mode"],
                    "experiment_name": exp["expected_name"],
                    "relative_path": safe_rel(path, root),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "category": file_category(path),
                })
            except OSError as exc:
                rows.append({
                    "fold": exp["fold"], "mode": exp["mode"],
                    "experiment_name": exp["expected_name"],
                    "relative_path": safe_rel(path, root),
                    "filename": path.name, "extension": path.suffix.lower(),
                    "size_bytes": -1, "category": "stat_error",
                    "error": str(exc),
                })
    return rows


def inspect_csvs(
    root: Path,
    file_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    schemas = []
    candidates = []
    usable_tables = []

    target_candidates = config["target_column_candidates"]
    score_candidates = config["score_column_candidates"]
    split_candidates = config["split_column_candidates"]
    group_candidates = config["group_column_candidates"]

    for row in file_rows:
        if row.get("extension") != ".csv" or row.get("size_bytes", -1) < 0:
            continue
        path = root / row["relative_path"]
        df, error = read_csv_safe(path)
        if df is None:
            schemas.append({
                **{k: row[k] for k in ("fold", "mode", "experiment_name", "relative_path")},
                "rows": "", "columns": "", "column_names": "", "read_error": error,
            })
            continue

        target_col = first_existing(df.columns, target_candidates)
        score_col = first_existing(df.columns, score_candidates)
        split_col = first_existing(df.columns, split_candidates)
        group_col = first_existing(df.columns, group_candidates)
        inferred_split = infer_split_from_name(path)

        schema = {
            **{k: row[k] for k in ("fold", "mode", "experiment_name", "relative_path")},
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": "|".join(map(str, df.columns)),
            "target_column": target_col or "",
            "score_column": score_col or "",
            "split_column": split_col or "",
            "group_column": group_col or "",
            "inferred_split": inferred_split or "",
            "read_error": "",
        }
        schemas.append(schema)

        keyword_hit = any(
            keyword in path.name.lower()
            for keyword in config["prediction_filename_keywords"]
        )
        is_candidate = bool(target_col and score_col) or keyword_hit
        if is_candidate:
            candidates.append({
                **schema,
                "usable_for_metrics": bool(target_col and score_col),
            })
        if target_col and score_col:
            usable_tables.append({
                "fold": row["fold"],
                "mode": row["mode"],
                "experiment_name": row["experiment_name"],
                "path": path,
                "df": df,
                "target_col": target_col,
                "score_col": score_col,
                "split_col": split_col,
                "group_col": group_col,
                "inferred_split": inferred_split,
            })
    return schemas, candidates, usable_tables


def split_usable_tables(usable_tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in usable_tables:
        df = item["df"]
        split_col = item["split_col"]
        if split_col:
            for split_value, part in df.groupby(split_col, dropna=False):
                split_name = str(split_value).strip().lower()
                normalized.append({**item, "df": part.copy(), "split": split_name})
        else:
            normalized.append({
                **item,
                "split": item["inferred_split"] or "unknown",
            })
    return normalized


def compute_metrics(
    tables: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quantile_rows = []
    low_fpr_rows = []
    transfer_rows = []
    group_rows = []

    prepared: dict[tuple[int, str, str], dict[str, Any]] = {}

    for item in split_usable_tables(tables):
        df = item["df"].copy()
        y_true = normalize_binary_target(df[item["target_col"]])
        score = pd.to_numeric(df[item["score_col"]], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(score)
        if y_true is None:
            continue
        valid &= np.isfinite(y_true)
        y_true = y_true[valid]
        score = score[valid]
        if len(score) == 0:
            continue

        key = (int(item["fold"]), str(item["mode"]), str(item["split"]).lower())
        prepared[key] = {
            "y_true": y_true,
            "score": score,
            "df": df.loc[valid].reset_index(drop=True),
            "group_col": item["group_col"],
            "path": item["path"],
        }

        for cls, cls_name in ((0, "background"), (1, "target")):
            values = score[y_true == cls]
            if values.size == 0:
                continue
            row = {
                "fold": item["fold"], "mode": item["mode"],
                "split": item["split"], "class": cls_name,
                "count": int(values.size), "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
                row[f"q{int(q*100):02d}"] = float(np.quantile(values, q))
            quantile_rows.append(row)

        auc = float("nan")
        if roc_auc_score is not None and len(np.unique(y_true)) == 2:
            try:
                auc = float(roc_auc_score(y_true, score))
            except Exception:
                pass
        low_fpr_rows.append({
            "fold": item["fold"], "mode": item["mode"], "split": item["split"],
            "n": int(len(y_true)),
            "positive": int((y_true == 1).sum()),
            "background": int((y_true == 0).sum()),
            "roc_auc": auc,
            "partial_auc_at_5pct_fpr": calculate_partial_auc(y_true, score, 0.05),
            "tpr_at_1pct_fpr": calculate_tpr_at_fpr(y_true, score, 0.01),
            "tpr_at_5pct_fpr": calculate_tpr_at_fpr(y_true, score, 0.05),
        })

        group_col = item["group_col"]
        if group_col and group_col in df.columns:
            temp = df.loc[valid, [group_col]].copy()
            temp["_y"] = y_true
            temp["_score"] = score
            for group, g in temp.groupby(group_col, dropna=False):
                for cls, cls_name in ((0, "background"), (1, "target")):
                    values = g.loc[g["_y"] == cls, "_score"].to_numpy(dtype=float)
                    if values.size:
                        group_rows.append({
                            "fold": item["fold"], "mode": item["mode"],
                            "split": item["split"], "group_column": group_col,
                            "group": str(group), "class": cls_name,
                            "count": int(values.size),
                            "mean_score": float(np.mean(values)),
                            "max_score": float(np.max(values)),
                            "q95_score": float(np.quantile(values, 0.95)),
                            "q99_score": float(np.quantile(values, 0.99)),
                        })

    for fold in sorted({k[0] for k in prepared}):
        for mode in sorted({k[1] for k in prepared if k[0] == fold}):
            val = prepared.get((fold, mode, "validation")) or prepared.get((fold, mode, "val"))
            test = prepared.get((fold, mode, "test"))
            if not val or not test:
                continue
            val_bg = val["score"][val["y_true"] == 0]
            test_bg = test["score"][test["y_true"] == 0]
            if val_bg.size == 0 or test_bg.size == 0:
                continue

            # 以验证背景99%分位数近似 1% FPR 工作点；
            # 若正式预测表中已有 threshold 列，优先读取其唯一值。
            threshold = float(np.quantile(val_bg, 0.99))
            for source in (val, test):
                df = source["df"]
                for candidate in ("threshold", "decision_threshold", "selected_threshold"):
                    if candidate in df.columns:
                        values = pd.to_numeric(df[candidate], errors="coerce").dropna().unique()
                        if len(values) == 1:
                            threshold = float(values[0])
                            break

            test_y = test["y_true"]
            test_score = test["score"]
            transfer_rows.append({
                "fold": fold,
                "mode": mode,
                "threshold": threshold,
                "val_background_count": int(val_bg.size),
                "test_background_count": int(test_bg.size),
                "val_background_exceed_fraction": float(np.mean(val_bg >= threshold)),
                "test_background_exceed_fraction": float(np.mean(test_bg >= threshold)),
                "test_target_detect_fraction": float(
                    np.mean(test_score[test_y == 1] >= threshold)
                ) if np.any(test_y == 1) else float("nan"),
                "val_background_q50": float(np.quantile(val_bg, 0.50)),
                "val_background_q95": float(np.quantile(val_bg, 0.95)),
                "val_background_q99": float(np.quantile(val_bg, 0.99)),
                "test_background_q50": float(np.quantile(test_bg, 0.50)),
                "test_background_q95": float(np.quantile(test_bg, 0.95)),
                "test_background_q99": float(np.quantile(test_bg, 0.99)),
                "background_median_shift": float(
                    np.quantile(test_bg, 0.50) - np.quantile(val_bg, 0.50)
                ),
                "background_q95_shift": float(
                    np.quantile(test_bg, 0.95) - np.quantile(val_bg, 0.95)
                ),
            })

    return quantile_rows, low_fpr_rows, transfer_rows, group_rows


def inspect_checkpoint(path: Path, root: Path, fold: int, mode: str) -> dict[str, Any]:
    row = {
        "fold": fold, "mode": mode, "path": safe_rel(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "top_level_keys": "", "epoch": "", "threshold": "",
        "load_error": "",
    }
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            row["top_level_keys"] = "|".join(map(str, payload.keys()))
            for key in ("epoch", "best_epoch"):
                if key in payload:
                    row["epoch"] = payload[key]
                    break
            for key in ("threshold", "best_threshold", "decision_threshold"):
                if key in payload:
                    value = payload[key]
                    try:
                        row["threshold"] = float(value)
                    except Exception:
                        row["threshold"] = str(value)
                    break
        else:
            row["top_level_keys"] = type(payload).__name__
        del payload
    except Exception as exc:
        row["load_error"] = f"{type(exc).__name__}: {exc}"
    return row


def collect_checkpoints(root: Path, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for exp in experiments:
        if not exp["found"]:
            continue
        exp_dir = Path(exp["experiment_dir"])
        candidates = []
        for pattern in ("**/best.pt", "**/best.pth", "**/*.ckpt"):
            candidates.extend(exp_dir.glob(pattern))
        seen = set()
        for path in sorted(candidates):
            if path.resolve() in seen or not path.is_file():
                continue
            seen.add(path.resolve())
            rows.append(inspect_checkpoint(path, root, exp["fold"], exp["mode"]))
    return rows


def copy_small_outputs(
    root: Path,
    output_dir: Path,
    file_rows: list[dict[str, Any]],
    max_mb: float,
    protected_exts: set[str],
) -> list[dict[str, Any]]:
    copied = []
    limit = int(max_mb * 1024 * 1024)
    allowed = {".csv", ".json", ".yaml", ".yml", ".md", ".txt", ".log"}
    for row in file_rows:
        ext = row.get("extension", "")
        if ext in protected_exts or ext not in allowed:
            continue
        size = int(row.get("size_bytes", -1))
        if size < 0 or size > limit:
            continue
        src = root / row["relative_path"]
        dst = output_dir / "collected_outputs" / row["experiment_name"] / src.relative_to(
            Path(row["relative_path"]).parents[len(Path(row["relative_path"]).parts)-1]
            if False else Path(src).parents[0]
        )
        # 保留实验目录内部相对路径，避免同名文件冲突
        exp_dir = root / "results" / "experiments" / row["experiment_name"]
        try:
            internal = src.relative_to(exp_dir)
        except Exception:
            internal = Path(src.name)
        dst = output_dir / "collected_outputs" / row["experiment_name"] / internal
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({
            "source": safe_rel(src, root),
            "destination": safe_rel(dst, root),
            "size_bytes": size,
        })
    return copied


def copy_source_context(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    exact = [
        "training/train_polarimetric_representation_fcn_v2.py",
        "scripts/run_polarimetric_representation_benchmark_v2.py",
        "scripts/summarize_polarimetric_representation_benchmark_v2.py",
        "scripts/audit_polarimetric_score_shift_v1.py",
        "scripts/audit_polarimetric_channels_v1.py",
        "scripts/test_polarimetric_gated_pipeline_v2.py",
        "datasets/polarimetric_detection_dataset_v2.py",
        "features/polarimetric_gated_rd.py",
        "configs/polarimetric_representation_benchmark_v2.yaml",
        "results/data_audit/polarimetric_representation_benchmark_v2/representation_detail_formal.csv",
        "results/data_audit/polarimetric_representation_benchmark_v2/representation_aggregate_formal.csv",
        "results/data_audit/polarimetric_representation_benchmark_v2/latest_run_plan.json",
        "results/data_audit/polarimetric_representation_benchmark_v2/latest_run_status.json",
        "results/data_audit/polarimetric_representation_benchmark_v2/README_representation_benchmark_formal.md",
    ]
    rows = []
    for rel in exact:
        src = root / rel
        if not src.exists() or not src.is_file():
            rows.append({"path": rel, "found": False, "sha256": "", "size_bytes": ""})
            continue
        dst = output_dir / "source_context" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append({
            "path": rel, "found": True,
            "sha256": sha256_file(src), "size_bytes": src.stat().st_size,
        })
    return rows


def build_readme(
    output_dir: Path,
    experiments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    low_fpr_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    zip_path: Path,
) -> None:
    found_exp = sum(bool(x["found"]) for x in experiments)
    usable = sum(bool(x.get("usable_for_metrics")) for x in candidates)
    status = "COMPLETE" if low_fpr_rows and transfer_rows else "NEED_PREDICTION_EXPORT"
    text = f"""# Polarimetric Twofold Diagnostics Collection V1

生成时间：{utc_now()}

## 状态

- experiments requested：{len(experiments)}
- experiments found：{found_exp}
- prediction candidate tables：{len(candidates)}
- usable prediction tables：{usable}
- low-FPR rows：{len(low_fpr_rows)}
- threshold-transfer rows：{len(transfer_rows)}
- low-FPR status：{status}
- acceptance ZIP：{zip_path.name}

## 说明

本次工具只读取已有 Fold 1 / Fold 4 正式实验，不重新训练、不修改权重。
若状态为 `COMPLETE`，可直接使用本包中的低FPR、分数分位数、阈值迁移和组级统计。
若状态为 `NEED_PREDICTION_EXPORT`，说明现有实验结果未保存可识别的逐样本标签与分数表；
本包已同时收集真实训练入口、Dataset、特征构造、配置、checkpoint键名和全部CSV字段，
可据此生成与当前工程零猜测兼容的预测导出补丁。
"""
    output_dir.joinpath("README_diagnostics_collection.md").write_text(text, encoding="utf-8")


def package_output(root: Path, output_dir: Path, zip_name: str, terminal_log: Path) -> Path:
    zip_path = root / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
            arc = Path(output_dir.name) / path.relative_to(output_dir)
            zf.write(path, arcname=str(arc))
        if terminal_log.exists():
            zf.write(terminal_log, arcname=terminal_log.name)
        config = root / "configs" / "polarimetric_twofold_diagnostics_v1.json"
        if config.exists():
            zf.write(config, arcname=str(Path("configs") / config.name))
    return zip_path


def main() -> int:
    args = parse_args()
    root = ensure_project_root(args.project_root)
    config = load_config(root, args.config)
    folds = args.folds
    modes = args.modes
    max_mb = args.max_copy_size_mb or float(config["max_copy_size_mb"])
    protected_exts = set(config["protected_extensions"])

    output_dir = root / config["audit_output"]
    if output_dir.exists():
        backup = output_dir.with_name(
            output_dir.name + "_previous_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        output_dir.rename(backup)
    output_dir.mkdir(parents=True)

    print("=" * 78)
    print("Polarimetric Twofold Diagnostics Collection V1")
    print(f"project root : {root}")
    print(f"folds        : {folds}")
    print(f"modes        : {modes}")
    print("=" * 78)

    experiments = discover_experiment_dirs(
        root, folds, modes, config["experiment_name_template"]
    )
    write_csv(output_dir / "experiment_status.csv", experiments)

    file_rows = inventory_experiment_files(root, experiments)
    write_csv(output_dir / "experiment_file_inventory.csv", file_rows)

    schemas, prediction_candidates, usable_tables = inspect_csvs(root, file_rows, config)
    write_csv(output_dir / "csv_schema_inventory.csv", schemas)
    write_csv(output_dir / "prediction_candidate_inventory.csv", prediction_candidates)

    checkpoints = collect_checkpoints(root, experiments)
    write_csv(output_dir / "checkpoint_inventory.csv", checkpoints)

    copied = copy_small_outputs(
        root, output_dir, file_rows, max_mb=max_mb, protected_exts=protected_exts
    )
    write_csv(output_dir / "copied_output_inventory.csv", copied)

    source_context = copy_source_context(root, output_dir)
    write_csv(output_dir / "source_context_inventory.csv", source_context)

    quantiles, low_fpr, transfers, groups = compute_metrics(usable_tables, config)
    write_csv(output_dir / "score_distribution_quantiles.csv", quantiles)
    write_csv(output_dir / "low_fpr_metrics.csv", low_fpr)
    write_csv(output_dir / "threshold_transfer_metrics.csv", transfers)
    write_csv(output_dir / "group_score_stratification.csv", groups)

    missing = []
    for exp in experiments:
        if not exp["found"]:
            missing.append({
                "type": "experiment_directory",
                "fold": exp["fold"], "mode": exp["mode"],
                "requirement": exp["expected_name"],
                "reason": "正式实验目录未找到",
            })
    usable_count = sum(bool(x.get("usable_for_metrics")) for x in prediction_candidates)
    if usable_count == 0:
        missing.append({
            "type": "prediction_table",
            "fold": "", "mode": "",
            "requirement": "validation/test逐样本标签+分数CSV",
            "reason": "现有CSV中未同时识别到target与score字段",
        })
    if not transfers:
        missing.append({
            "type": "threshold_transfer",
            "fold": "", "mode": "",
            "requirement": "同一fold/mode的validation与test逐样本分数",
            "reason": "无法配对验证集和测试集预测表",
        })
    write_csv(output_dir / "missing_requirements.csv", missing)

    status = {
        "generated_at": utc_now(),
        "project_root": str(root),
        "folds": folds,
        "modes": modes,
        "experiments_requested": len(experiments),
        "experiments_found": sum(bool(x["found"]) for x in experiments),
        "files_inventoried": len(file_rows),
        "csv_files_inspected": len(schemas),
        "prediction_candidate_tables": len(prediction_candidates),
        "usable_prediction_tables": usable_count,
        "low_fpr_rows": len(low_fpr),
        "threshold_transfer_rows": len(transfers),
        "group_rows": len(groups),
        "low_fpr_status": "COMPLETE" if low_fpr and transfers else "NEED_PREDICTION_EXPORT",
        "errors": [],
    }
    (output_dir / "collection_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    terminal_log = root / "polarimetric_twofold_diagnostics_terminal_v1.log"
    zip_path = root / config["acceptance_zip"]
    build_readme(
        output_dir, experiments, prediction_candidates, low_fpr,
        transfers, missing, zip_path
    )

    if not args.no_package:
        zip_path = package_output(
            root, output_dir, config["acceptance_zip"], terminal_log
        )

    print("-" * 78)
    print(f"experiments requested : {len(experiments)}")
    print(f"experiments found     : {status['experiments_found']}")
    print(f"prediction candidates : {len(prediction_candidates)}")
    print(f"usable predictions    : {usable_count}")
    print(f"low-FPR status        : {status['low_fpr_status']}")
    print(f"audit output          : {output_dir}")
    if not args.no_package:
        print(f"acceptance zip        : {zip_path}")
    print("-" * 78)
    if status["experiments_found"] != len(experiments):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("=" * 78, file=sys.stderr)
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        print("=" * 78, file=sys.stderr)
        raise
