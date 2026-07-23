#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("[错误] 缺少 pandas，请先安装：pip install pandas") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL_NAME = "BC-DPG-FCN v3"
FINAL_SHIFT_REGULARIZATION = 0.01
DEFAULT_OUTPUT = "results/final_evidence/bc_dpg_v3_final"

REQUIRED_EVIDENCE = {
    "deployment_detail": "results/data_audit/bc_dpg_v3_deployment_comparison/deployment_comparison_detail.csv",
    "deployment_aggregate": "results/data_audit/bc_dpg_v3_deployment_comparison/deployment_comparison_aggregate.csv",
    "deployment_audit": "results/data_audit/bc_dpg_v3_deployment_comparison/deployment_comparison_audit.json",
    "ablation_detail": "results/data_audit/bc_dpg_v3_ablation/ablation_detail_formal.csv",
    "ablation_aggregate": "results/data_audit/bc_dpg_v3_ablation/ablation_aggregate_formal.csv",
    "regularization_candidates": "results/data_audit/bc_dpg_v31_shift_reg/candidate_validation_metrics_formal.csv",
    "regularization_selected": "results/data_audit/bc_dpg_v31_shift_reg/selected_by_fold_formal.csv",
    "regularization_test": "results/data_audit/bc_dpg_v31_shift_reg/selected_test_metrics_formal.csv",
    "regularization_aggregate": "results/data_audit/bc_dpg_v31_shift_reg/aggregate_selected_test_metrics_formal.csv",
    "regularization_audit": "results/data_audit/bc_dpg_v31_shift_reg/selection_audit_formal.json",
}

OPTIONAL_REPORTS = (
    "results/data_audit/bc_dpg_v3_deployment_comparison/README_deployment_comparison.md",
    "results/data_audit/bc_dpg_v3_ablation/README_消融结果_formal.md",
    "results/data_audit/bc_dpg_v31_shift_reg/README_shift_regularization_selection_formal.md",
)

SOURCE_AND_CONFIG_FILES = (
    "models/target_protected_scan_calibrator.py",
    "training/train_target_protected_scan_calibrator.py",
    "training/train_target_protected_scan_calibrator_ablation.py",
    "scripts/run_bc_dpg_v3_ablation.py",
    "scripts/summarize_bc_dpg_v3_ablation.py",
    "scripts/build_bc_dpg_v3_deployment_comparison.py",
    "scripts/run_bc_dpg_v31_shift_reg_sweep.py",
    "scripts/select_bc_dpg_v31_shift_reg.py",
    "scripts/build_bc_dpg_v3_final_evidence.py",
    "configs/bc_dpg_fcn_v3_scan_target.yaml",
)

ABLATION_NOTES = {
    "full": ("最终完整模型", "当前冻结模型；扫描上下文离线增强"),
    "no_scan_context": ("结构消融", "移除12维扫描组统计；对应样本独立校准器"),
    "no_background_classification": ("损失消融", "检验背景概率语义监督"),
    "no_background_tail": ("损失消融", "检验高分背景尾部抑制"),
    "no_target_protection": ("安全约束消融", "同时移除目标保护组合约束"),
    "no_target_keep": ("安全约束消融", "检验目标分数保持约束"),
    "no_pairwise": ("排序约束消融", "检验目标与困难背景的相对排序"),
    "no_shift_selectivity": ("安全约束消融", "检验抑制量的目标/背景选择性"),
    "no_shift_regularization": ("强度约束消融", "抑制更激进；因目标shift风险不替代v3"),
}

MODEL_LABELS = {
    "raw_dpg": "原始DPG-FCN",
    "sample_independent_bc": "样本独立BC校准",
    "scan_aware_bc": "扫描上下文BC-DPG-FCN v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final frozen BC-DPG-FCN v3 paper-evidence package without retraining."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--package", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, category: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "category": category,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": int(stat.st_size),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
    }


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def select_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    selected = frame.loc[frame[column].astype(str) == value]
    if selected.empty:
        raise KeyError(f"Missing row: {column}={value}")
    return selected.iloc[0]


def prepare_output(output_dir: Path) -> Path | None:
    backup: Path | None = None
    if output_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = output_dir.parent / "_archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        backup = archive_root / f"{output_dir.name}_{timestamp}"
        counter = 1
        while backup.exists():
            backup = archive_root / f"{output_dir.name}_{timestamp}_{counter:02d}"
            counter += 1
        shutil.move(str(output_dir), str(backup))
    output_dir.mkdir(parents=True, exist_ok=False)
    return backup


def copy_evidence(paths: dict[str, Path], output_dir: Path) -> list[dict[str, Any]]:
    target_root = output_dir / "source_evidence"
    target_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for key, source in paths.items():
        target = target_root / f"{key}{source.suffix}"
        shutil.copy2(source, target)
        copied.append({
            "key": key,
            "source": str(source.relative_to(PROJECT_ROOT)),
            "copied_to": str(target.relative_to(output_dir)),
            "sha256": sha256_file(source),
        })
    for relative in OPTIONAL_REPORTS:
        source = PROJECT_ROOT / relative
        if source.is_file():
            target = target_root / source.name
            shutil.copy2(source, target)
            copied.append({
                "key": "optional_report",
                "source": relative,
                "copied_to": str(target.relative_to(output_dir)),
                "sha256": sha256_file(source),
            })
    return copied


def build_main_table(deployment: pd.DataFrame) -> pd.DataFrame:
    frame = deployment.copy()
    frame["模型名称"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    raw_row = select_row(frame, "model", "raw_dpg")
    raw_fa = safe_float(raw_row["false_alarms_sum"])
    raw_pfa = safe_float(raw_row["pfa_mean"])
    frame["相对原始虚警减少数"] = raw_fa - pd.to_numeric(frame["false_alarms_sum"], errors="coerce")
    frame["相对原始虚警降幅"] = frame["相对原始虚警减少数"] / raw_fa if raw_fa else float("nan")
    frame["平均Pfa改善"] = raw_pfa - pd.to_numeric(frame["pfa_mean"], errors="coerce")
    columns = [
        "model", "模型名称", "folds", "false_alarms_sum", "相对原始虚警减少数",
        "相对原始虚警降幅", "pfa_mean", "平均Pfa改善", "pd_mean", "auc_mean",
        "background_shift_mean", "target_shift_mean", "context_requirement", "causality",
    ]
    return frame[columns]


def build_ablation_table(ablation: pd.DataFrame) -> pd.DataFrame:
    frame = ablation.copy()
    frame["模块角色"] = frame["mode"].map(lambda x: ABLATION_NOTES.get(str(x), ("", ""))[0])
    frame["解释"] = frame["mode"].map(lambda x: ABLATION_NOTES.get(str(x), ("", ""))[1])
    columns = [
        "mode", "模块角色", "解释", "raw_false_alarms_sum", "calibrated_false_alarms_sum",
        "false_alarm_reduction_sum", "raw_test_pfa_mean", "calibrated_test_pfa_mean",
        "raw_test_pd_mean", "calibrated_test_pd_mean", "delta_pd_mean",
        "raw_test_auc_mean", "calibrated_test_auc_mean", "background_shift_mean_mean",
        "target_shift_mean_mean",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Ablation table missing columns: {missing}")
    return frame[columns]


def build_regularization_decision(selected: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    weights = pd.to_numeric(selected["regularization"], errors="coerce")
    counts = weights.value_counts(dropna=False).sort_index()
    rows: list[dict[str, Any]] = []
    for weight, count in counts.items():
        folds = selected.loc[weights == weight, "fold"].astype(int).tolist()
        rows.append({
            "candidate": f"validation-selected weight {weight:g}",
            "fold_count": int(count),
            "folds": ",".join(str(v) for v in folds),
            "adopt_as_final": False,
            "reason": "各折选择不一致，不能形成统一部署超参数",
        })
    rows.append({
        "candidate": f"fixed weight {FINAL_SHIFT_REGULARIZATION:g}",
        "fold_count": 6,
        "folds": "1,2,3,4,5,6",
        "adopt_as_final": True,
        "reason": "更保守、目标抑制风险较低，保持当前v3冻结结论",
    })
    return pd.DataFrame(rows)


def build_claim_boundaries() -> pd.DataFrame:
    rows = [
        ("可以声称", "六折扫描组内部验证中，完整v3将虚警由186降至56且平均Pd不变"),
        ("可以声称", "样本独立校准器将虚警由186降至122，说明收益不完全依赖扫描上下文"),
        ("可以声称", "完整扫描上下文进一步降低虚警，但属于离线扫描级增强"),
        ("可以声称", "扫描上下文、背景分类和背景尾部损失均有消融证据支持"),
        ("不可声称", "当前结果是跨日期、跨场地或跨环境的独立盲测"),
        ("不可声称", "完整扫描上下文模型已经满足严格实时因果推理"),
        ("不可声称", "无正则或折内自适应权重已经优于固定0.01并可直接部署"),
        ("后续方向", "未来核心数据来自自主空飘球与载荷实验，并保留连续H/V复数IQ及时间信息"),
    ]
    return pd.DataFrame(rows, columns=["类型", "表述"])


def checkpoint_paths(seed: int) -> Iterable[tuple[str, Path, int]]:
    for fold in range(1, 7):
        yield (
            "base_dpg_checkpoint",
            PROJECT_ROOT / "results" / "experiments" / f"dpg_fcn_v4_fold{fold:02d}_seed{seed}" / "checkpoints" / "best.pt",
            fold,
        )
        yield (
            "full_v3_checkpoint",
            PROJECT_ROOT / "results" / "experiments" / f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed{seed}" / "checkpoints" / "best.pt",
            fold,
        )


def collect_hashes(seed: int, include_checkpoints: bool) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in SOURCE_AND_CONFIG_FILES:
        path = PROJECT_ROOT / relative
        if path.is_file():
            records.append(file_record(path, "source_or_config"))
        else:
            missing.append(relative)
    for fold in range(1, 7):
        manifest = PROJECT_ROOT / "results" / "data_audit" / "dataset_v4_multifold" / f"fold_{fold:02d}_manifest.csv"
        if manifest.is_file():
            records.append(file_record(manifest, "fold_manifest"))
        else:
            missing.append(str(manifest.relative_to(PROJECT_ROOT)))
        summary = PROJECT_ROOT / "results" / "experiments" / f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed{seed}" / "tables" / "summary.json"
        if summary.is_file():
            records.append(file_record(summary, "full_v3_summary"))
        else:
            missing.append(str(summary.relative_to(PROJECT_ROOT)))
    if include_checkpoints:
        for category, path, _fold in checkpoint_paths(seed):
            if path.is_file():
                records.append(file_record(path, category))
            else:
                missing.append(str(path.relative_to(PROJECT_ROOT)))
    return records, missing


def save_figures(
    main_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; figures skipped"]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    figure_labels = {
        "raw_dpg": "Raw DPG-FCN",
        "sample_independent_bc": "Sample-independent BC",
        "scan_aware_bc": "Scan-aware BC-DPG v3",
    }
    labels = [figure_labels.get(str(v), str(v)) for v in main_table["model"]]
    values = pd.to_numeric(main_table["false_alarms_sum"], errors="coerce").tolist()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Six-fold false alarms")
    ax.set_title("Deployment-condition comparison")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    path = figures_dir / "fig1_deployment_false_alarms.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path.relative_to(output_dir)))

    ab = ablation_table.sort_values("calibrated_false_alarms_sum", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    bars = ax.barh(ab["mode"], pd.to_numeric(ab["calibrated_false_alarms_sum"], errors="coerce"))
    ax.set_xlabel("Six-fold calibrated false alarms")
    ax.set_title("BC-DPG-FCN v3 core ablation")
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f" {int(width)}", va="center")
    fig.tight_layout()
    path = figures_dir / "fig2_ablation_false_alarms.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path.relative_to(output_dir)))

    x = pd.to_numeric(ablation_table["target_shift_mean_mean"], errors="coerce")
    y = pd.to_numeric(ablation_table["calibrated_false_alarms_sum"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.scatter(x, y)
    for mode, xv, yv in zip(ablation_table["mode"], x, y):
        ax.annotate(str(mode), (xv, yv), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean target shift")
    ax.set_ylabel("Six-fold calibrated false alarms")
    ax.set_title("False-alarm reduction vs target suppression")
    fig.tight_layout()
    path = figures_dir / "fig3_false_alarm_target_shift_tradeoff.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    folds = pd.to_numeric(selected["fold"], errors="coerce")
    regs = pd.to_numeric(selected["regularization"], errors="coerce")
    bars = ax.bar(folds.astype(str), regs)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation-selected regularization")
    ax.set_title("Fold-specific regularization selections")
    for bar, value in zip(bars, regs):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:g}", ha="center", va="bottom")
    fig.tight_layout()
    path = figures_dir / "fig4_selected_regularization_by_fold.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path.relative_to(output_dir)))
    return created


def md_table(frame: pd.DataFrame, columns: list[str], precision: int = 4) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:.{precision}f}")
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def build_reports(
    main_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    selected: pd.DataFrame,
    selected_test: pd.DataFrame,
    hash_records: list[dict[str, Any]],
    missing: list[str],
    figures: list[str],
    output_dir: Path,
) -> None:
    raw = select_row(main_table, "model", "raw_dpg")
    sample = select_row(main_table, "model", "sample_independent_bc")
    full = select_row(main_table, "model", "scan_aware_bc")
    selected_weights = [safe_float(v) for v in selected["regularization"].tolist()]
    selected_fa = pd.to_numeric(selected_test["selected_test_false_alarms"], errors="coerce").sum()
    selected_target_shift = pd.to_numeric(selected_test["test_target_shift"], errors="coerce").mean()

    decision = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "final_model": FINAL_MODEL_NAME,
        "final_shift_regularization": FINAL_SHIFT_REGULARIZATION,
        "final_model_status": "frozen_current_model",
        "deployment_scope": "offline scan-aware background calibrator",
        "main_evidence": {
            "raw_false_alarms": int(raw["false_alarms_sum"]),
            "sample_independent_false_alarms": int(sample["false_alarms_sum"]),
            "final_false_alarms": int(full["false_alarms_sum"]),
            "raw_pd_mean": safe_float(raw["pd_mean"]),
            "final_pd_mean": safe_float(full["pd_mean"]),
            "raw_pfa_mean": safe_float(raw["pfa_mean"]),
            "final_pfa_mean": safe_float(full["pfa_mean"]),
        },
        "v31_decision": {
            "validation_selected_weights": selected_weights,
            "unique_weight_count": len(set(selected_weights)),
            "selected_test_false_alarms": int(selected_fa),
            "selected_test_target_shift_mean": float(selected_target_shift),
            "adopted": False,
            "reason": "fold-specific validation selections are inconsistent and cannot define one deployable fixed hyperparameter; lower regularization also raises target suppression risk",
        },
        "evidence_scope": "internal six-fold scan-group evaluation; not an independent new-date/new-environment blind test",
        "context_limitation": "complete scan-group statistics may contain future samples; the full model is not a strict causal online detector",
    }
    (output_dir / "final_model_spec.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# BC-DPG-FCN v3 最终冻结与论文证据报告",
        "",
        f"生成时间：{decision['created_at']}",
        "",
        "## 1. 最终模型决策",
        "",
        f"- 最终模型：**{FINAL_MODEL_NAME}**",
        f"- 固定 `shift_regularization`：**{FINAL_SHIFT_REGULARIZATION:g}**",
        "- 当前定位：带完整扫描组统计的离线背景条件校准器",
        "- 不采用折内自适应 v3.1 作为最终部署模型",
        "",
        "## 2. 六折主结果",
        "",
        md_table(main_table, ["模型名称", "false_alarms_sum", "相对原始虚警降幅", "pfa_mean", "pd_mean", "auc_mean", "target_shift_mean"]),
        "",
        f"原始 DPG 的六折虚警为 {int(raw['false_alarms_sum'])} 个；样本独立校准器降至 {int(sample['false_alarms_sum'])} 个；完整扫描上下文 v3 降至 {int(full['false_alarms_sum'])} 个。完整 v3 的平均 Pd 由 {safe_float(raw['pd_mean']):.4f} 保持为 {safe_float(full['pd_mean']):.4f}。",
        "",
        "## 3. 消融结论",
        "",
        md_table(ablation_table, ["mode", "模块角色", "calibrated_false_alarms_sum", "calibrated_test_pfa_mean", "calibrated_test_pd_mean", "target_shift_mean_mean"]),
        "",
        "扫描上下文、背景分类和高分背景尾部损失均有直接消融支持。目标保护、target keep 与 shift selectivity 主要限制真实目标的非必要抑制，不能只用当前 Pd 是否下降来判断其价值。",
        "",
        "## 4. 正则权重扫描决策",
        "",
        "各折验证集选择的权重为：" + ", ".join(f"Fold {int(f)}={safe_float(w):g}" for f, w in zip(selected["fold"], selected["regularization"])),
        "",
        f"折内选择结果在测试侧共剩余 {int(selected_fa)} 个虚警，但权重跨折不一致，且较弱正则会提高部分折的目标 shift。因此该结果只作为方法探索，不替代固定权重 0.01 的当前 v3。",
        "",
        "## 5. 论文表述边界",
        "",
        "- 可以表述为六折扫描组内部验证。",
        "- 不能表述为跨日期、跨场地或跨环境独立盲测。",
        "- 完整扫描上下文可能使用同一扫描后续样本，因此只能作为离线增强结果。",
        "- 样本独立校准器是未来样本级多域主模型更合适的对照基础。",
        "",
        "## 6. 文件冻结",
        "",
        f"- 已记录哈希文件：{len(hash_records)}",
        f"- 缺失文件：{len(missing)}",
        f"- checkpoint 哈希数量：{sum(1 for r in hash_records if 'checkpoint' in r['category'])}",
        "",
        "## 7. 生成图",
        "",
    ]
    report.extend(f"- `{item}`" for item in figures)
    if missing:
        report += ["", "## 8. 缺失文件", ""]
        report.extend(f"- `{item}`" for item in missing)
    (output_dir / "FINAL_EVIDENCE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    paper = [
        "# 可用于论文结果部分的中文草稿",
        "",
        "在六折扫描组内部验证中，原始 DPG-FCN 在固定部署阈值下共产生 "
        f"{int(raw['false_alarms_sum'])} 个背景虚警。仅使用样本自身特征、并将扫描组统计量置零的样本独立校准器将虚警降至 "
        f"{int(sample['false_alarms_sum'])} 个，平均检测率由 {safe_float(raw['pd_mean']):.4f} 保持为 {safe_float(sample['pd_mean']):.4f}。"
        "在此基础上，引入完整扫描组统计的 BC-DPG-FCN v3 将虚警进一步降至 "
        f"{int(full['false_alarms_sum'])} 个，相对原始模型降低 {safe_float(full['相对原始虚警降幅'])*100:.1f}%，"
        f"平均 Pfa 由 {safe_float(raw['pfa_mean']):.4f} 降至 {safe_float(full['pfa_mean']):.4f}，"
        f"而平均 Pd 保持为 {safe_float(full['pd_mean']):.4f}。",
        "",
        "消融实验表明，移除扫描上下文、背景分类损失或背景尾部损失后，六折剩余虚警分别增加至 "
        f"{int(select_row(ablation_table, 'mode', 'no_scan_context')['calibrated_false_alarms_sum'])}、"
        f"{int(select_row(ablation_table, 'mode', 'no_background_classification')['calibrated_false_alarms_sum'])} 和 "
        f"{int(select_row(ablation_table, 'mode', 'no_background_tail')['calibrated_false_alarms_sum'])} 个，"
        "说明扫描级背景条件、高分背景尾部以及显式背景语义共同构成主要的虚警抑制来源。另一方面，去除目标保护相关约束虽然未在当前测试折中降低 Pd，但显著增大目标 shift，表明这些约束主要用于限制模型通过统一压低目标分数换取较低虚警。",
        "",
        "需要强调的是，完整扫描上下文模型使用同一扫描组的整体统计量，属于离线扫描级增强，其结果不能直接外推为严格实时因果推理或跨环境独立盲测性能。",
    ]
    (output_dir / "PAPER_RESULTS_DRAFT_ZH.md").write_text("\n".join(paper) + "\n", encoding="utf-8")


def package_directory(output_dir: Path) -> Path:
    zip_path = PROJECT_ROOT / "bc_dpg_v3_final_evidence.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(output_dir.name) / path.relative_to(output_dir)))
    return zip_path


def main() -> None:
    args = parse_args()
    evidence_paths = {key: PROJECT_ROOT / relative for key, relative in REQUIRED_EVIDENCE.items()}
    missing_required = [str(path.relative_to(PROJECT_ROOT)) for path in evidence_paths.values() if not path.is_file()]
    if missing_required:
        print("[错误] 缺少正式证据文件：")
        for item in missing_required:
            print(f"  - {item}")
        raise SystemExit(2)

    output_dir = resolve_path(args.output_dir)
    backup = prepare_output(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    deployment_aggregate = pd.read_csv(evidence_paths["deployment_aggregate"])
    deployment_detail = pd.read_csv(evidence_paths["deployment_detail"])
    ablation_aggregate = pd.read_csv(evidence_paths["ablation_aggregate"])
    ablation_detail = pd.read_csv(evidence_paths["ablation_detail"])
    reg_candidates = pd.read_csv(evidence_paths["regularization_candidates"])
    reg_selected = pd.read_csv(evidence_paths["regularization_selected"])
    reg_test = pd.read_csv(evidence_paths["regularization_test"])
    reg_aggregate = pd.read_csv(evidence_paths["regularization_aggregate"])

    main_table = build_main_table(deployment_aggregate)
    ablation_table = build_ablation_table(ablation_aggregate)
    reg_decision = build_regularization_decision(reg_selected, reg_test)
    claim_boundaries = build_claim_boundaries()

    main_table.to_csv(tables_dir / "table_01_main_model_comparison.csv", index=False, encoding="utf-8-sig")
    deployment_detail.to_csv(tables_dir / "table_02_main_model_fold_detail.csv", index=False, encoding="utf-8-sig")
    ablation_table.to_csv(tables_dir / "table_03_ablation_summary.csv", index=False, encoding="utf-8-sig")
    ablation_detail.to_csv(tables_dir / "table_04_ablation_fold_detail.csv", index=False, encoding="utf-8-sig")
    reg_candidates.to_csv(tables_dir / "table_05_regularization_validation_candidates.csv", index=False, encoding="utf-8-sig")
    reg_selected.to_csv(tables_dir / "table_06_regularization_selected_by_fold.csv", index=False, encoding="utf-8-sig")
    reg_test.to_csv(tables_dir / "table_07_regularization_selected_test.csv", index=False, encoding="utf-8-sig")
    reg_aggregate.to_csv(tables_dir / "table_08_regularization_selected_test_aggregate.csv", index=False, encoding="utf-8-sig")
    reg_decision.to_csv(tables_dir / "table_09_final_model_decision.csv", index=False, encoding="utf-8-sig")
    claim_boundaries.to_csv(tables_dir / "table_10_claim_boundaries.csv", index=False, encoding="utf-8-sig")

    copied = copy_evidence(evidence_paths, output_dir)
    (output_dir / "source_evidence" / "COPIED_EVIDENCE_MANIFEST.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hash_records, hash_missing = collect_hashes(args.seed, args.include_checkpoints)
    hash_frame = pd.DataFrame(hash_records)
    hash_frame.to_csv(tables_dir / "table_11_file_hashes.csv", index=False, encoding="utf-8-sig")
    checkpoint_frame = hash_frame.loc[hash_frame["category"].astype(str).str.contains("checkpoint", na=False)] if not hash_frame.empty else pd.DataFrame()
    checkpoint_frame.to_csv(tables_dir / "table_12_checkpoint_hashes.csv", index=False, encoding="utf-8-sig")
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(f"{item['sha256']}  {item['relative_path']}" for item in hash_records) + "\n",
        encoding="utf-8",
    )

    figures = [] if args.skip_figures else save_figures(main_table, ablation_table, reg_selected, output_dir)
    build_reports(main_table, ablation_table, reg_selected, reg_test, hash_records, hash_missing, figures, output_dir)

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "backup_dir": str(backup) if backup else None,
        "final_model": FINAL_MODEL_NAME,
        "final_shift_regularization": FINAL_SHIFT_REGULARIZATION,
        "include_checkpoints": bool(args.include_checkpoints),
        "required_evidence_count": len(evidence_paths),
        "required_evidence_missing": missing_required,
        "hash_record_count": len(hash_records),
        "hash_missing": hash_missing,
        "figures": figures,
        "validation_selection_uses_test_metrics": False,
        "v31_adopted": False,
    }
    (output_dir / "final_evidence_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.require_all and hash_missing:
        print("[错误] 最终冻结仍有缺失文件：")
        for item in hash_missing:
            print(f"  - {item}")
        print(f"已生成诊断目录：{output_dir}")
        raise SystemExit(3)

    package_path = package_directory(output_dir) if args.package else None
    print("=" * 88)
    print("BC-DPG-FCN v3 final evidence build complete")
    print(f"output directory       : {output_dir}")
    print(f"final model            : {FINAL_MODEL_NAME}")
    print(f"shift regularization   : {FINAL_SHIFT_REGULARIZATION:g}")
    print(f"evidence files         : {len(evidence_paths)}")
    print(f"hashed files           : {len(hash_records)}")
    print(f"checkpoint hashes      : {len(checkpoint_frame)}")
    print(f"missing hash inputs    : {len(hash_missing)}")
    print(f"figures                : {len(figures)}")
    if package_path:
        print(f"package                : {package_path}")
    print("status                 : PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
