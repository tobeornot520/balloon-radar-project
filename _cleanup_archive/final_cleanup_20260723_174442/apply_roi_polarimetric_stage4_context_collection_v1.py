#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
FILES = [
    "scripts/collect_roi_polarimetric_stage4_context_v1.py",
    "configs/roi_polarimetric_stage4_context_v1.json",
    "docs/README_候选区域极化精修Stage4上下文采集_V1.md",
]

def main() -> int:
    if not (ROOT / "scripts").is_dir() or not (ROOT / "configs").is_dir():
        print("错误：请把ZIP放在balloon_radar_project项目根目录后解压运行。", file=sys.stderr)
        return 1
    installed = 0
    for rel in FILES:
        src = HERE / rel
        dst = ROOT / rel
        if not src.is_file():
            print(f"缺少安装源文件: {src}", file=sys.stderr)
            return 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        print(f"INSTALLED  {rel}")
        installed += 1
    print(f"安装完成，共{installed}个文件。未修改模型、数据、checkpoint和正式结果。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
