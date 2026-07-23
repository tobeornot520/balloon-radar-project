#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balloon Radar Project 根目录安全清理工具 V1

默认只预览。--execute 会把候选文件移动到 _cleanup_archive/cleanup_<timestamp>/，
不会永久删除。--restore-latest 可恢复最近一次清理。
--permanent 仅在显式确认短语后永久删除候选，不建议日常使用。
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime
from typing import Iterable, Dict, List, Tuple, Set


SCRIPT_NAME = Path(__file__).name
ARCHIVE_ROOT_NAME = "_cleanup_archive"

# 这些目录是项目当前运行、数据、模型与正式结果的核心，永不自动清理。
PROTECTED_TOP_LEVEL = {
    ".git",
    ".github",
    ".vscode",
    ARCHIVE_ROOT_NAME,
    "backups",
    "baselines",
    "checkpoints",
    "configs",
    "data",
    "datasets",
    "dist",
    "docs",
    "evaluation",
    "features",
    "logs",
    "losses",
    "metrics",
    "models",
    "notebooks",
    "results",
    "scripts",
    "training",
}

# 根目录中明确保留的环境与工具文件。
PROTECTED_FILES = {
    SCRIPT_NAME,
    "environment.yml",
    "requirements-lock.txt",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "README.md",
    "test_read_mat.py",
    "collect_bc_dpg_v3_context.sh",
    "package_balloon_radar_project.sh",
}

# 已经集成进正式源码、可从根目录移走的历史补丁/诊断目录。
LEGACY_DIRECTORY_RULES: List[Tuple[str, str, str]] = [
    ("legacy_patch_dir", "BC_DPG_FCN_v1_*", "早期 BC-DPG-FCN v1 补丁目录，功能已集成"),
    ("legacy_patch_dir", "BC_DPG_FCN_v2_*", "早期 BC-DPG-FCN v2 补丁目录，功能已集成"),
    ("legacy_patch_dir", "BC_DPG_FCN_v3_scan_target_patch", "v3 扫描目标补丁目录，功能已集成"),
    ("legacy_audit_dir", "BC_DPG_overfitting_audit_*", "旧过拟合审计补丁目录"),
    ("legacy_patch_dir", "BC_DPG_v2_*", "旧 v2 比较/阈值补丁目录"),
    ("legacy_paper_dir", "BC_DPG_v3_paper_results_*", "旧论文结果生成补丁目录"),
    ("legacy_package_dir", "current_structure_reader_package", "一次性项目结构读取包"),
    ("legacy_package_dir", "detection_ablation_analysis_v2_package", "旧检测消融分析包"),
    ("legacy_package_dir", "detection_diagnostics_v3_package", "旧检测诊断包"),
    ("legacy_package_dir", "detection_group_split_v1_package", "旧分组划分包"),
    ("legacy_package_dir", "detection_visualization_hotfix_v3_1", "旧可视化热修复目录"),
    ("legacy_patch_dir", "dpg_fcn_v1_patch", "早期 DPG-FCN v1 补丁目录"),
    ("legacy_package_dir", "full_detection_baseline_v2_package", "旧完整检测基线包"),
    ("legacy_diagnostic_dir", "hv_late_fusion_diagnostic_v1", "旧 H/V 后融合诊断目录"),
]

# 根目录中的安装器、补丁包、验收包、过程日志和阶段 README。
# 正式源码与结果已保存在 features/datasets/models/training/scripts/configs/results 中。
ROOT_FILE_RULES: List[Tuple[str, str, str]] = [
    ("installer", "apply_bc_dpg_current_model_stage*.py", "已执行的当前模型 Stage 2 安装器"),
    ("installer", "apply_bc_dpg_v3_ablation_*.py", "已执行的 v3 消融安装器"),
    ("installer", "apply_bc_dpg_v3_final_freeze_*.py", "已执行的最终冻结安装器"),
    ("installer", "apply_polarimetric_representation_benchmark_*.py", "已执行的极化表征安装器"),
    ("installer", "apply_polarimetric_gated_representation_*.py", "已执行的极化门控安装器"),

    ("patch_zip", "BC_DPG_current_model_stage2_patch_*.zip", "已安装的 Stage 2 补丁压缩包"),
    ("patch_zip", "BC_DPG_explicit_polarimetric_representation_stage2_patch_*.zip", "已安装的极化 Stage 2 补丁压缩包"),
    ("patch_zip", "BC_DPG_polarimetric_gated_diagnostic_stage3_patch_*.zip", "已安装的极化 Stage 3 补丁压缩包"),
    ("patch_zip", "BC_DPG_v3_final_freeze_paper_evidence_patch_*.zip", "已安装的 v3 最终冻结补丁压缩包"),

    ("acceptance_zip", "bc_dpg_current_model_stage2_smoke_acceptance*.zip", "Stage 2 验收包，结果已进入 results"),
    ("acceptance_zip", "bc_dpg_v3_ablation_*acceptance*.zip", "v3 消融验收包，结果已进入 results"),
    ("acceptance_zip", "bc_dpg_v31_shift_reg_formal_acceptance*.zip", "正则扫描验收包，结果已进入 results"),
    ("evidence_zip", "bc_dpg_v3_final_evidence.zip", "根目录证据副本，正式文件保存在 results/final_evidence"),
    ("acceptance_zip", "bc_dpg_v3_final_freeze_acceptance*.zip", "最终冻结验收包"),
    ("acceptance_zip", "polarimetric_representation_stage2_smoke_acceptance*.zip", "极化 Stage 2 smoke 验收包"),
    ("acceptance_zip", "polarimetric_representation_twofold_formal_acceptance*.zip", "极化两折正式验收包"),
    ("acceptance_zip", "polarimetric_gated_stage3_smoke_acceptance*.zip", "极化 Stage 3 smoke 验收包"),

    ("terminal_log", "bc_dpg_current_model_stage2_smoke_terminal.log", "Stage 2 根目录过程日志"),
    ("terminal_log", "bc_dpg_v3_ablation_*terminal*.log", "v3 消融根目录过程日志"),
    ("terminal_log", "bc_dpg_v31_shift_reg_formal_terminal.log", "正则扫描根目录过程日志"),
    ("terminal_log", "bc_dpg_v3_final_evidence_terminal.log", "最终证据根目录过程日志"),
    ("terminal_log", "polarimetric_representation_stage2_smoke_terminal.log", "极化 Stage 2 根目录过程日志"),
    ("terminal_log", "polarimetric_representation_twofold_formal_terminal.log", "极化两折正式根目录过程日志"),
    ("terminal_log", "polarimetric_gated_stage3_smoke_terminal.log", "极化 Stage 3 根目录过程日志"),

    ("stage_readme", "README_显式极化表征基准Stage2*.md", "阶段安装说明，正式文档已在项目目录内"),
    ("stage_readme", "README_显式极化门控与分数迁移*.md", "阶段安装说明，正式文档已在项目目录内"),
    ("stage_readme", "README_BC_DPG_v3最终冻结与论文*.md", "阶段安装说明，正式报告已在 results/final_evidence"),
    ("stage_readme", "README_BC_DPG当前模型Stage2*.md", "阶段安装说明"),
]

CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
CACHE_FILE_PATTERNS = ("*.pyc", "*.pyo", "*~", ".DS_Store")


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def path_size(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                pass
        return total
    except OSError:
        return 0


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def assert_project_root(root: Path) -> None:
    required = ["configs", "data", "datasets", "models", "results"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise SystemExit(
            "安全检查失败：当前目录不像 balloon_radar_project 项目根目录。\n"
            f"缺少：{', '.join(missing)}\n"
            f"当前目录：{root}"
        )


def is_protected(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    if not rel.parts:
        return True
    if rel.parts[0] in PROTECTED_TOP_LEVEL:
        return True
    if len(rel.parts) == 1 and rel.name in PROTECTED_FILES:
        return True
    return False


def collect_rule_matches(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen: Set[Path] = set()

    for category, pattern, reason in LEGACY_DIRECTORY_RULES:
        for path in sorted(root.glob(pattern)):
            if not path.is_dir() or path.is_symlink():
                continue
            if is_protected(path, root):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(make_row(path, root, category, reason, "archive"))

    for category, pattern, reason in ROOT_FILE_RULES:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() and not path.is_symlink():
                continue
            if is_protected(path, root):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(make_row(path, root, category, reason, "archive"))

    return rows


def collect_caches(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen: Set[Path] = set()

    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)

        # 永不遍历归档、数据和结果目录，避免耗时和误删。
        skip_names = {
            ARCHIVE_ROOT_NAME,
            ".git",
            "data",
            "results",
            "checkpoints",
            "backups",
        }
        dirs[:] = [d for d in dirs if d not in skip_names]

        for d in list(dirs):
            if d in CACHE_DIR_NAMES:
                p = current_path / d
                if p.resolve() not in seen:
                    seen.add(p.resolve())
                    rows.append(make_row(
                        p, root, "cache_dir",
                        "Python/测试/Notebook 可再生缓存",
                        "delete_cache"
                    ))
                dirs.remove(d)

        for f in files:
            if any(fnmatch.fnmatch(f, pattern) for pattern in CACHE_FILE_PATTERNS):
                p = current_path / f
                if p.resolve() not in seen:
                    seen.add(p.resolve())
                    rows.append(make_row(
                        p, root, "cache_file",
                        "可再生临时文件",
                        "delete_cache"
                    ))
    return rows


def make_row(path: Path, root: Path, category: str, reason: str, action: str) -> Dict[str, object]:
    size = path_size(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "kind": "directory" if path.is_dir() else "file",
        "category": category,
        "recommended_action": action,
        "reason": reason,
        "size_bytes": size,
        "size_human": human_size(size),
        "sha256": sha256_file(path),
    }


def write_report(
    root: Path,
    rows: List[Dict[str, object]],
    mode: str,
    report_dir: Path,
    moved: List[Dict[str, object]] | None = None,
    errors: List[Dict[str, str]] | None = None,
) -> Tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "cleanup_manifest.csv"
    json_path = report_dir / "cleanup_manifest.json"
    md_path = report_dir / "CLEANUP_REPORT.md"

    fieldnames = [
        "relative_path", "kind", "category", "recommended_action",
        "reason", "size_bytes", "size_human", "sha256",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "tool_version": "1.0",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "mode": mode,
        "candidate_count": len(rows),
        "candidate_bytes": sum(int(r["size_bytes"]) for r in rows),
        "candidate_human": human_size(sum(int(r["size_bytes"]) for r in rows)),
        "moved": moved or [],
        "errors": errors or [],
        "protected_top_level": sorted(PROTECTED_TOP_LEVEL),
        "protected_files": sorted(PROTECTED_FILES),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    by_category: Dict[str, Dict[str, int]] = {}
    for row in rows:
        cat = str(row["category"])
        by_category.setdefault(cat, {"count": 0, "bytes": 0})
        by_category[cat]["count"] += 1
        by_category[cat]["bytes"] += int(row["size_bytes"])

    lines = [
        "# Balloon Radar Project 根目录清理报告",
        "",
        f"- 模式：`{mode}`",
        f"- 候选项目数：**{len(rows)}**",
        f"- 候选总大小：**{human_size(sum(int(r['size_bytes']) for r in rows))}**",
        f"- 项目根目录：`{root}`",
        "",
        "## 分类统计",
        "",
        "| 分类 | 数量 | 大小 |",
        "|---|---:|---:|",
    ]
    for cat, item in sorted(by_category.items()):
        lines.append(f"| {cat} | {item['count']} | {human_size(item['bytes'])} |")

    lines += [
        "",
        "## 永久保留范围",
        "",
        "以下顶层目录不会被自动移动或删除：",
        "",
        "```text",
        "\n".join(sorted(PROTECTED_TOP_LEVEL)),
        "```",
        "",
        "## 候选清单",
        "",
        "| 路径 | 类型 | 分类 | 建议 | 大小 | 原因 |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        reason = str(row["reason"]).replace("|", "/")
        lines.append(
            f"| `{row['relative_path']}` | {row['kind']} | {row['category']} | "
            f"{row['recommended_action']} | {row['size_human']} | {reason} |"
        )

    if moved:
        lines += ["", "## 已处理", "", f"共处理 **{len(moved)}** 项。"]
    if errors:
        lines += ["", "## 错误", ""]
        for err in errors:
            lines.append(f"- `{err.get('path','')}`：{err.get('error','')}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def move_candidates(
    root: Path,
    rows: List[Dict[str, object]],
    archive_dir: Path,
    clean_caches: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    moved: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    archive_payload = archive_dir / "payload"
    archive_payload.mkdir(parents=True, exist_ok=True)

    # 先移动普通候选，缓存最后删除。
    ordered = sorted(rows, key=lambda r: (r["recommended_action"] == "delete_cache", r["relative_path"]))

    for row in ordered:
        rel = Path(str(row["relative_path"]))
        src = root / rel
        action = str(row["recommended_action"])
        try:
            if not src.exists() and not src.is_symlink():
                continue

            if action == "delete_cache":
                if not clean_caches:
                    continue
                if src.is_dir() and not src.is_symlink():
                    shutil.rmtree(src)
                else:
                    src.unlink()
                moved.append({
                    "relative_path": rel.as_posix(),
                    "action": "cache_deleted",
                })
                continue

            dst = archive_payload / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(f"归档目标已存在：{dst}")
            shutil.move(str(src), str(dst))
            moved.append({
                "relative_path": rel.as_posix(),
                "action": "moved_to_archive",
                "archive_relative_path": dst.relative_to(archive_dir).as_posix(),
            })
        except Exception as exc:
            errors.append({"path": rel.as_posix(), "error": repr(exc)})

    return moved, errors


def restore_archive(root: Path, archive_dir: Path) -> None:
    manifest_path = archive_dir / "cleanup_execution.json"
    payload_dir = archive_dir / "payload"
    if not payload_dir.is_dir():
        raise SystemExit(f"归档中缺少 payload：{archive_dir}")

    restored = 0
    conflicts: List[str] = []
    # 先文件后目录，通过遍历 payload 的顶层条目即可保留结构。
    items = sorted(payload_dir.iterdir(), key=lambda p: p.name)
    for src in items:
        dst = root / src.name
        if dst.exists() or dst.is_symlink():
            conflicts.append(src.name)
            continue
        shutil.move(str(src), str(dst))
        restored += 1

    if conflicts:
        print("以下顶层路径已存在，未覆盖：")
        for name in conflicts:
            print(f"  - {name}")
    print(f"恢复完成：{restored} 个顶层条目")
    print("注意：缓存文件不会恢复，因为它们可重新生成。")


def find_latest_archive(root: Path) -> Path:
    archive_root = root / ARCHIVE_ROOT_NAME
    candidates = sorted(
        [p for p in archive_root.glob("cleanup_*") if p.is_dir()],
        key=lambda p: p.name
    )
    if not candidates:
        raise SystemExit("未找到可恢复的清理归档。")
    return candidates[-1]


def permanent_delete(
    root: Path,
    rows: List[Dict[str, object]],
    clean_caches: bool,
    confirm: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    if confirm != "DELETE_ROOT_ARTIFACTS":
        raise SystemExit(
            "永久删除需要附加：--confirm DELETE_ROOT_ARTIFACTS\n"
            "推荐使用 --execute，让文件先进入可恢复归档。"
        )
    processed: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    for row in rows:
        rel = Path(str(row["relative_path"]))
        src = root / rel
        action = str(row["recommended_action"])
        if action == "delete_cache" and not clean_caches:
            continue
        try:
            if not src.exists() and not src.is_symlink():
                continue
            if src.is_dir() and not src.is_symlink():
                shutil.rmtree(src)
            else:
                src.unlink()
            processed.append({"relative_path": rel.as_posix(), "action": "permanently_deleted"})
        except Exception as exc:
            errors.append({"path": rel.as_posix(), "error": repr(exc)})
    return processed, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="安全整理 balloon_radar_project 根目录中的历史补丁与过程文件。"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="仅生成候选清单（默认）")
    mode.add_argument("--execute", action="store_true", help="移动候选到可恢复归档")
    mode.add_argument("--restore-latest", action="store_true", help="恢复最近一次归档")
    mode.add_argument("--permanent", action="store_true", help="永久删除候选（不推荐）")
    parser.add_argument(
        "--clean-caches",
        action="store_true",
        help="同时删除可再生缓存（__pycache__、pyc等）"
    )
    parser.add_argument("--confirm", default="", help="永久删除确认短语")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    assert_project_root(root)

    if args.restore_latest:
        archive_dir = find_latest_archive(root)
        print(f"准备恢复：{archive_dir}")
        restore_archive(root, archive_dir)
        return 0

    rows = collect_rule_matches(root)
    if args.clean_caches:
        rows.extend(collect_caches(root))
    rows = sorted(rows, key=lambda r: str(r["relative_path"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_dir = root / ARCHIVE_ROOT_NAME / f"preview_{timestamp}"

    if not args.execute and not args.permanent:
        csv_path, json_path, md_path = write_report(
            root, rows, "preview", preview_dir
        )
        total = sum(int(r["size_bytes"]) for r in rows)
        print("=" * 88)
        print("BALLOON RADAR PROJECT CLEANUP PREVIEW")
        print("=" * 88)
        print(f"project root : {root}")
        print(f"candidates   : {len(rows)}")
        print(f"total size   : {human_size(total)}")
        print(f"report       : {md_path}")
        print()
        for row in rows:
            print(
                f"[{row['category']:<20}] {row['size_human']:>10}  "
                f"{row['relative_path']}"
            )
        print()
        print("预览完成，未移动或删除任何文件。")
        print(f"执行归档清理：python {SCRIPT_NAME} --execute --clean-caches")
        return 0

    if args.permanent:
        report_dir = root / ARCHIVE_ROOT_NAME / f"permanent_delete_report_{timestamp}"
        processed, errors = permanent_delete(
            root, rows, args.clean_caches, args.confirm
        )
        write_report(root, rows, "permanent", report_dir, processed, errors)
        print(f"永久删除完成：{len(processed)} 项，错误：{len(errors)} 项")
        print(f"报告：{report_dir}")
        return 1 if errors else 0

    archive_dir = root / ARCHIVE_ROOT_NAME / f"cleanup_{timestamp}"
    moved, errors = move_candidates(root, rows, archive_dir, args.clean_caches)
    csv_path, json_path, md_path = write_report(
        root, rows, "execute", archive_dir, moved, errors
    )
    execution = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "archive_dir": str(archive_dir),
        "moved_count": len(moved),
        "error_count": len(errors),
        "moved": moved,
        "errors": errors,
    }
    (archive_dir / "cleanup_execution.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=" * 88)
    print("BALLOON RADAR PROJECT CLEANUP EXECUTE")
    print("=" * 88)
    print(f"archive dir : {archive_dir}")
    print(f"processed   : {len(moved)}")
    print(f"errors      : {len(errors)}")
    print(f"report      : {md_path}")
    print()
    if errors:
        for err in errors:
            print(f"[ERROR] {err['path']}: {err['error']}")
        print("存在错误，请上传验收包核对。")
        return 1
    print("清理完成。历史补丁和根目录过程文件已移入可恢复归档。")
    print(f"恢复命令：python {SCRIPT_NAME} --restore-latest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
