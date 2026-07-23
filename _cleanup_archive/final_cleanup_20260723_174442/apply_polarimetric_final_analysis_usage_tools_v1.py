#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import shutil
from pathlib import Path

FILES = {
    "payload/scripts/use_polarimetric_final_analysis_v1.py": "scripts/use_polarimetric_final_analysis_v1.py",
    "payload/docs/README_极化门控最终分析包使用工具_V1.md": "docs/README_极化门控最终分析包使用工具_V1.md",
}

def main() -> int:
    here = Path(__file__).resolve().parent
    project = Path.cwd().resolve()
    for src_rel, dst_rel in FILES.items():
        src = here / src_rel
        dst = project / dst_rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"INSTALLED {dst_rel}")
    print("安装完成。不会修改模型、checkpoint或正式实验结果。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
