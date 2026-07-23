#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装、校验并复用“极化门控两折正式诊断最终分析 V1”。

功能：
1. 安全解压分析包到 results/data_audit；
2. 校验要求的 11 个结果文件及关键 CSV 字段；
3. 计算 SHA256 清单；
4. 动态生成 Stage 3 冻结结论与下一阶段 ROI 极化精修接口说明；
5. 自动打包单一验收 ZIP。

不会训练模型，不会修改 checkpoint，不会覆盖 BC-DPG-FCN v3 正式结果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ANALYSIS_DIRNAME = "polarimetric_twofold_diagnostics_final_analysis_v1"
REQUIRED_FILES = [
    "README_极化门控两折正式诊断最终分析.md",
    "background_false_alarm_overlap.csv",
    "threshold_transfer_detail.csv",
    "fixed_threshold_detail.csv",
    "low_fpr_detail.csv",
    "fixed_threshold_macro.csv",
    "threshold_transfer_macro.csv",
    "target_scan_group_metrics.csv",
    "score_distribution_quantiles.csv",
    "low_fpr_macro.csv",
    "target_failure_breakdown.csv",
]
EXPECTED_COLUMNS = {
    "fixed_threshold_macro.csv": {
        "mode", "false_alarms", "mean_pfa", "mean_score_recall", "mean_joint_pd",
        "mean_auc", "mean_range_mae", "mean_velocity_mae"
    },
    "low_fpr_macro.csv": {"mode", "mean_auc", "mean_pauc5", "mean_tpr1", "mean_tpr5"},
    "threshold_transfer_macro.csv": {
        "mode", "mean_test_bg_exceed", "mean_test_target_detect",
        "mean_bg_median_shift", "mean_bg_q95_shift"
    },
    "fixed_threshold_detail.csv": {
        "fold", "mode", "best_epoch", "validation_threshold", "test_false_alarms",
        "test_pfa", "test_score_recall", "test_joint_pd", "test_auc",
        "range_mae_all_gates", "velocity_mae_all_bins"
    },
    "low_fpr_detail.csv": {
        "fold", "mode", "split", "roc_auc", "partial_auc_at_5pct_fpr",
        "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"
    },
}
MODES = ["power2", "ri4", "polar6_gated", "ri8_gated"]
DISPLAY = {
    "power2": "Power2",
    "ri4": "RI4",
    "polar6_gated": "Polar6-gated",
    "ri8_gated": "RI8-gated",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str) -> float:
    try:
        x = float(value)
        if not math.isfinite(x):
            raise ValueError
        return x
    except Exception as exc:
        raise ValueError(f"无法解析有限浮点数: {value!r}") from exc


def safe_extract_analysis(zip_path: Path, destination_parent: Path) -> Path:
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / ANALYSIS_DIRNAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if destination.exists():
        backup_root = destination_parent.parents[1] / "backups" / f"{ANALYSIS_DIRNAME}_{timestamp}"
        backup_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(backup_root))
        print(f"BACKUP   {destination} -> {backup_root}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        selected: Dict[str, zipfile.ZipInfo] = {}
        for info in members:
            if info.is_dir():
                continue
            normalized = Path(info.filename.replace("\\", "/"))
            if ".." in normalized.parts:
                raise RuntimeError(f"分析包包含不安全路径: {info.filename}")
            basename = normalized.name
            if basename in REQUIRED_FILES:
                selected[basename] = info

        missing = [name for name in REQUIRED_FILES if name not in selected]
        if missing:
            raise RuntimeError("分析包缺少要求文件: " + ", ".join(missing))

        destination.mkdir(parents=True, exist_ok=False)
        for basename, info in selected.items():
            out = destination / basename
            with zf.open(info, "r") as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"INSTALLED {out}")
    return destination


def verify(directory: Path) -> Dict[str, object]:
    report: Dict[str, object] = {
        "status": "PASS",
        "analysis_directory": str(directory),
        "required_count": len(REQUIRED_FILES),
        "found_count": 0,
        "missing": [],
        "column_errors": {},
        "mode_errors": {},
        "sha256": {},
    }
    missing = []
    for name in REQUIRED_FILES:
        path = directory / name
        if not path.is_file():
            missing.append(name)
        else:
            report["found_count"] = int(report["found_count"]) + 1
            report["sha256"][name] = sha256(path)
    report["missing"] = missing

    for name, expected in EXPECTED_COLUMNS.items():
        path = directory / name
        if not path.is_file():
            continue
        rows = read_csv(path)
        if not rows:
            report["column_errors"][name] = ["CSV为空"]
            continue
        actual = set(rows[0].keys())
        absent = sorted(expected - actual)
        if absent:
            report["column_errors"][name] = absent

    for name in ["fixed_threshold_macro.csv", "low_fpr_macro.csv", "threshold_transfer_macro.csv"]:
        path = directory / name
        if not path.is_file():
            continue
        rows = read_csv(path)
        actual_modes = {r.get("mode", "") for r in rows}
        absent_modes = sorted(set(MODES) - actual_modes)
        if absent_modes:
            report["mode_errors"][name] = absent_modes

    if missing or report["column_errors"] or report["mode_errors"]:
        report["status"] = "FAIL"
    return report


def rows_by_mode(path: Path) -> Dict[str, Dict[str, str]]:
    return {row["mode"]: row for row in read_csv(path)}


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def generate_frozen_conclusion(directory: Path, output: Path) -> None:
    fixed = rows_by_mode(directory / "fixed_threshold_macro.csv")
    low = rows_by_mode(directory / "low_fpr_macro.csv")
    transfer = rows_by_mode(directory / "threshold_transfer_macro.csv")

    best_pauc = max(MODES, key=lambda m: fnum(low[m]["mean_pauc5"]))
    lowest_pfa = min(MODES, key=lambda m: fnum(fixed[m]["mean_pfa"]))
    best_pd = max(MODES, key=lambda m: fnum(fixed[m]["mean_joint_pd"]))
    smallest_shift = min(MODES, key=lambda m: fnum(transfer[m]["mean_test_bg_exceed"]))

    lines = [
        "# Stage 3 显式极化门控表征冻结结论",
        "",
        "> 本文件由 `use_polarimetric_final_analysis_v1.py` 从正式分析CSV自动生成。",
        "",
        "## 1. 数据与证据范围",
        "",
        "- 使用 Fold 1 与 Fold 4 两个困难折。",
        "- 四种表征采用相同网络规模和样本独立推理，不使用扫描上下文。",
        "- 当前结论用于方法诊断和下一阶段设计，不代表跨日期、跨场地或独立外部泛化。",
        "",
        "## 2. 固定验证阈值结果",
        "",
        "| 表征 | 虚警数 | 平均Pfa | 分数检测率 | 联合Pd | AUC | 距离MAE | 速度MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        r = fixed[mode]
        lines.append(
            f"| {DISPLAY[mode]} | {int(round(fnum(r['false_alarms'])))} | "
            f"{fmt(fnum(r['mean_pfa']))} | {fmt(fnum(r['mean_score_recall']))} | "
            f"{fmt(fnum(r['mean_joint_pd']))} | {fmt(fnum(r['mean_auc']))} | "
            f"{fmt(fnum(r['mean_range_mae']), 2)} | {fmt(fnum(r['mean_velocity_mae']), 2)} |"
        )
    lines += [
        "",
        "## 3. 低FPR工作区间",
        "",
        "| 表征 | pAUC@5%FPR | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        r = low[mode]
        lines.append(
            f"| {DISPLAY[mode]} | {fmt(fnum(r['mean_pauc5']))} | "
            f"{fmt(fnum(r['mean_tpr1']))} | {fmt(fnum(r['mean_tpr5']))} |"
        )
    lines += [
        "",
        "## 4. 验证—测试背景分数迁移",
        "",
        "| 表征 | 测试背景超过验证背景Q99 | 背景中位数上移 | 背景Q95上移 |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        r = transfer[mode]
        lines.append(
            f"| {DISPLAY[mode]} | {fmt(fnum(r['mean_test_bg_exceed']))} | "
            f"{fmt(fnum(r['mean_bg_median_shift']))} | {fmt(fnum(r['mean_bg_q95_shift']))} |"
        )
    lines += [
        "",
        "## 5. 自动判读",
        "",
        f"- 低FPR排序能力最高：**{DISPLAY[best_pauc]}**。",
        f"- 固定阈值虚警最低：**{DISPLAY[lowest_pfa]}**。",
        f"- 联合检测率最高：**{DISPLAY[best_pd]}**。",
        f"- 验证—测试背景超阈比例最低：**{DISPLAY[smallest_shift]}**。",
        "- Power2继续作为主检测与定位表征；门控证明了低功率极化噪声抑制方向有效，但密集显式极化输入尚未形成稳定独立增益。",
        "- 不将 Polar6-gated 或 RI8-gated 直接替代 Power2，也不立即扩展为六折主模型。",
        "",
        "## 6. 下一阶段",
        "",
        "采用候选区域引导的极化精修：Power2负责候选峰和定位，只在候选峰局部ROI内提取RI4及门控极化特征，用于候选确认或重排序。",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def generate_next_stage_interface(output: Path) -> None:
    text = """# 候选区域引导极化精修：下一阶段固定接口

## 1. 研究目标

保留 Power2 的高检测率和定位能力，只在 Power2 候选峰周围提取局部极化特征，判断候选是真目标还是背景伪峰。极化分支不重新决定全图峰位置。

## 2. 第一轮对照

1. `power2_baseline`：原始 Power2。
2. `power2_roi_power_control`：只使用候选ROI功率统计，作为ROI机制控制组。
3. `power2_roi_ri4`：ROI内 H/V 实部和虚部。
4. `power2_roi_polar6_gated`：ROI内门控 relative_ZDR_like、local_rho_HV 与相对相位。
5. `power2_roi_ri4_polar6_gated`：复数信息与门控极化联合。

第一轮仅运行 Fold 1 与 Fold 4；保持相同 manifest、seed、Power2 checkpoint、阈值选择规则和测试评价规则。

## 3. 建议输入输出

候选输入至少包含：

```text
sample_id
fold
raw_power2_score
power2_pred_range_index
power2_pred_velocity_index
roi_tensor
roi_valid_mask
```

模型统一输出：

```python
{
    "raw_power2_score": ...,
    "refined_score": ...,
    "roi_quality": ...,
    "polarimetric_confidence": ...,
    "score_shift": ...,
    "pred_range_index": ...,   # 默认沿用Power2
    "pred_velocity_index": ... # 默认沿用Power2
}
```

## 4. 评价指标

除固定阈值 Pd、Pfa、AUC、距离和速度MAE外，必须报告：

```text
partial AUC@5% FPR
TPR@1% FPR
TPR@5% FPR
验证—测试背景分数漂移
Power2虚警救回/新增虚警
Power2目标救回/退化
```

## 5. 约束

- 样本独立推理，不使用完整扫描上下文。
- 不修改已冻结的 BC-DPG-FCN v3。
- 不覆盖现有 Power2、RI4、Polar6-gated、RI8-gated 正式结果。
- smoke只验证接口，不用于模型选择。
- 如果ROI极化在两个困难折均无稳定增益，应冻结当前预实验，把重点转向未来严格极化标定与连续时序采集。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def make_acceptance_zip(project_root: Path, analysis_dir: Path, generated: Iterable[Path], terminal_log: Path | None) -> Path:
    out = project_root / "polarimetric_final_analysis_usage_acceptance_v1.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in generated:
            if path.is_file():
                zf.write(path, arcname=path.relative_to(project_root))
        for name in [
            "fixed_threshold_macro.csv", "low_fpr_macro.csv", "threshold_transfer_macro.csv",
            "target_failure_breakdown.csv", "background_false_alarm_overlap.csv"
        ]:
            path = analysis_dir / name
            if path.is_file():
                zf.write(path, arcname=path.relative_to(project_root))
        if terminal_log and terminal_log.is_file():
            zf.write(terminal_log, arcname=terminal_log.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="安装、校验并复用极化门控两折最终分析包")
    parser.add_argument("--project-root", default=".", help="项目根目录，默认当前目录")
    parser.add_argument("--analysis-zip", default="polarimetric_twofold_diagnostics_final_analysis_v1.zip")
    parser.add_argument("--terminal-log", default="polarimetric_final_analysis_usage_terminal_v1.log")
    parser.add_argument("--verify-only", action="store_true", help="只校验已安装目录，不解压")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    analysis_parent = project_root / "results" / "data_audit"
    analysis_dir = analysis_parent / ANALYSIS_DIRNAME

    print("=" * 78)
    print("极化门控两折最终分析包：安装、校验与复用 V1")
    print(f"project_root : {project_root}")
    print(f"analysis_dir : {analysis_dir}")

    if not args.verify_only:
        zip_path = Path(args.analysis_zip).expanduser()
        if not zip_path.is_absolute():
            zip_path = project_root / zip_path
        zip_path = zip_path.resolve()
        if not zip_path.is_file():
            print(f"ERROR: 找不到分析包: {zip_path}", file=sys.stderr)
            return 2
        analysis_dir = safe_extract_analysis(zip_path, analysis_parent)
    elif not analysis_dir.is_dir():
        print(f"ERROR: 已安装目录不存在: {analysis_dir}", file=sys.stderr)
        return 2

    report = verify(analysis_dir)
    verification_json = analysis_dir / "USAGE_VERIFICATION.json"
    verification_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    docs_root = project_root / "docs" / "polarimetric_stage3"
    conclusion = docs_root / "STAGE3_FROZEN_CONCLUSION.md"
    next_interface = docs_root / "NEXT_STAGE_ROI_POLARIMETRIC_INTERFACE.md"
    generate_frozen_conclusion(analysis_dir, conclusion)
    generate_next_stage_interface(next_interface)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": report["status"],
        "analysis_dir": str(analysis_dir.relative_to(project_root)),
        "frozen_conclusion": str(conclusion.relative_to(project_root)),
        "next_stage_interface": str(next_interface.relative_to(project_root)),
        "scientific_use": "Stage 3结果冻结、论文表格引用、下一阶段ROI极化精修设计",
        "not_for": "直接训练、替代Power2、证明跨环境独立泛化",
    }
    usage_manifest = analysis_dir / "USAGE_MANIFEST.json"
    usage_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    terminal_log = project_root / args.terminal_log
    generated = [verification_json, usage_manifest, conclusion, next_interface]
    acceptance = make_acceptance_zip(project_root, analysis_dir, generated, terminal_log)

    print("-" * 78)
    print(f"required : {report['required_count']}")
    print(f"found    : {report['found_count']}")
    print(f"status   : {report['status']}")
    print(f"结论文件 : {conclusion}")
    print(f"接口文件 : {next_interface}")
    print(f"验收包   : {acceptance}")
    print("=" * 78)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
