#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n执行：", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行DPG-FCN烟雾测试或多随机种子正式实验")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2026])
    parser.add_argument("--prefix", default="dpg_fcn_v1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    for seed in args.seeds:
        name = f"{args.prefix}_seed{seed}"
        command = [
            sys.executable, "training/train_dual_branch_gated.py",
            "--name", name, "--seed", str(seed),
            "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        ]
        if args.smoke:
            command += [
                "--epochs", "2", "--warmup-epochs", "1", "--partial-unfreeze-epochs", "1",
                "--debug-per-class", "8", "--early-stopping-patience", "3",
            ]
        if args.resume:
            command.append("--resume")
        run(command)

    if not args.smoke:
        run([sys.executable, "scripts/summarize_dual_branch_results.py", "--prefix", args.prefix])


if __name__ == "__main__":
    main()
