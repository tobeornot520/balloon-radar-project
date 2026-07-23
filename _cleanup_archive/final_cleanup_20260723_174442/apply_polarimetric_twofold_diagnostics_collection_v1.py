#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装 Polarimetric Twofold Diagnostics Collection V1 到雷达项目根目录。

用法：
    python apply_polarimetric_twofold_diagnostics_collection_v1.py

脚本会自动识别包含 models、training、scripts、configs 的项目根目录，
并安装采集脚本、配置与说明文件；不会覆盖已有正式实验结果。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


def looks_like_project(path: Path) -> bool:
    required = ("models", "training", "scripts", "configs", "results")
    return all((path / name).exists() for name in required)


def detect_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.home() / "projects" / "balloon_radar_project",
        PACKAGE_DIR.parent,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if looks_like_project(candidate):
            return candidate
    raise RuntimeError(
        "未找到 balloon_radar_project 项目根目录。"
        "请把本补丁ZIP解压到项目根目录后重新运行。"
    )


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup = dst.with_suffix(dst.suffix + ".before_diagnostics_v1.bak")
        if not backup.exists():
            shutil.copy2(dst, backup)
    shutil.copy2(src, dst)


def main() -> int:
    root = detect_project_root()
    mapping = {
        PACKAGE_DIR / "payload" / "scripts" / "collect_polarimetric_twofold_diagnostics_v1.py":
            root / "scripts" / "collect_polarimetric_twofold_diagnostics_v1.py",
        PACKAGE_DIR / "payload" / "configs" / "polarimetric_twofold_diagnostics_v1.json":
            root / "configs" / "polarimetric_twofold_diagnostics_v1.json",
        PACKAGE_DIR / "payload" / "docs" / "README_极化门控两折诊断采集_V1.md":
            root / "docs" / "README_极化门控两折诊断采集_V1.md",
    }
    for src, dst in mapping.items():
        copy_file(src, dst)
        print(f"INSTALLED  {dst.relative_to(root)}")

    collector = root / "scripts" / "collect_polarimetric_twofold_diagnostics_v1.py"
    collector.chmod(collector.stat().st_mode | 0o111)

    print("=" * 78)
    print(f"项目根目录：{root}")
    print("安装完成。未修改已冻结 BC-DPG-FCN v3，也未覆盖两折正式实验结果。")
    print("下一步运行：")
    print("python scripts/collect_polarimetric_twofold_diagnostics_v1.py --folds 1 4")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
