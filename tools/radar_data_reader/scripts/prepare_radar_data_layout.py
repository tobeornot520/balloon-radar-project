#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional


CLASS_ALIASES: Dict[str, tuple[str, ...]] = {
    "uav": ("无人机", "#U65e0#U4eba#U673a", "uav"),
    "balloon_line_array": ("气球+10米线阵", "气球＋10米线阵", "balloon_line_array"),
    "balloon_solar_panel": ("气球+太阳能板", "气球＋太阳能板", "balloon_solar_panel"),
    "balloon_box": ("气球+方盒子", "气球＋方盒子", "balloon_box"),
    "balloon_circuit_board": ("气球+电路板", "气球＋电路板", "balloon_circuit_board"),
}


def decode_hash_unicode(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 16))
    return re.sub(r"#U([0-9a-fA-F]{4,6})", repl, text)


def normalized_name(path: Path) -> str:
    return decode_hash_unicode(path.name).replace("＋", "+").strip().lower()


def find_dir(root: Path, suffix_parts: tuple[str, ...]) -> Optional[Path]:
    root = root.resolve()
    candidates = [p for p in root.rglob(suffix_parts[-1]) if p.is_dir()]
    for p in candidates:
        parts = p.parts
        if len(parts) >= len(suffix_parts) and tuple(parts[-len(suffix_parts):]) == suffix_parts:
            return p
    return None


def ensure_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(mode)


def link_all(files: Iterable[Path], dst_dir: Path, mode: str) -> int:
    count = 0
    for src in sorted(files):
        ensure_link_or_copy(src, dst_dir / src.name, mode)
        count += 1
    return count


def prepare_detection(src_root: Path, out_root: Path, split: str, mode: str) -> None:
    bg = find_dir(src_root, ("Background_IQ", "IQ_Data"))
    uav = find_dir(src_root, ("UAV_IQ", "IQ_Data"))
    labels = find_dir(src_root, ("UAV_IQ", "Labels"))
    if not bg or not uav or not labels:
        raise FileNotFoundError(
            f"{src_root} 中没有同时找到 Background_IQ/IQ_Data、UAV_IQ/IQ_Data、UAV_IQ/Labels"
        )
    base = out_root / "detection" / split
    n_bg = link_all(bg.glob("*.mat"), base / "background" / "iq", mode)
    n_uav = link_all(uav.glob("*.mat"), base / "uav" / "iq", mode)
    n_lab = link_all(labels.glob("*.txt"), base / "uav" / "labels", mode)
    print(f"[detection/{split}] background={n_bg}, uav_iq={n_uav}, labels={n_lab}")


def find_class_dir(src_root: Path, aliases: tuple[str, ...]) -> Optional[Path]:
    alias_norm = {decode_hash_unicode(a).replace("＋", "+").lower() for a in aliases}
    for p in src_root.rglob("*"):
        if p.is_dir() and normalized_name(p) in alias_norm:
            return p
    return None


def prepare_classification(src_root: Path, out_root: Path, split: str, mode: str) -> None:
    total = 0
    found = 0
    for class_name, aliases in CLASS_ALIASES.items():
        src = find_class_dir(src_root, aliases)
        if src is None:
            print(f"[classification/{split}] 未找到类别 {class_name}，暂时跳过")
            continue
        n = link_all(src.glob("*.mat"), out_root / "classification" / split / class_name, mode)
        print(f"[classification/{split}] {class_name}: {n} mat files")
        total += n
        found += 1
    if found == 0:
        raise FileNotFoundError(f"{src_root} 中没有识别到任何分类类别目录")
    print(f"[classification/{split}] total_mat_files={total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="把原始雷达数据映射成统一目录，不重复占用大文件空间。")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("symlink", "copy", "move"), default="symlink")
    parser.add_argument("--det-train-src", type=Path)
    parser.add_argument("--det-val-src", type=Path)
    parser.add_argument("--det-test-src", type=Path)
    parser.add_argument("--cls-train-src", type=Path)
    parser.add_argument("--cls-val-src", type=Path)
    parser.add_argument("--cls-test-src", type=Path)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        det_src = getattr(args, f"det_{split}_src")
        cls_src = getattr(args, f"cls_{split}_src")
        if det_src:
            prepare_detection(det_src, args.output_root, split, args.mode)
        if cls_src:
            prepare_classification(cls_src, args.output_root, split, args.mode)


if __name__ == "__main__":
    main()
