#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BC-DPG-FCN v3 on selected folds"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[2, 4],
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.smoke and args.formal:
        raise ValueError("Choose only one of --smoke or --formal")

    smoke = args.smoke or not args.formal
    epochs = 4 if smoke else 30
    debug_per_class = 12 if smoke else 0
    batch_size = args.batch_size or (16 if smoke else 64)

    for fold in args.folds:
        fold_text = f"{fold:02d}"
        suffix = "_smoke" if smoke else ""
        name = (
            f"bc_dpg_v3_scan_target_v4_fold{fold_text}_seed42"
            f"{suffix}"
        )

        command = [
            sys.executable,
            str(
                PROJECT_ROOT
                / "training"
                / "train_target_protected_scan_calibrator.py"
            ),
            "--name",
            name,
            "--manifest-path",
            str(
                PROJECT_ROOT
                / "results/data_audit/dataset_v4_multifold"
                / f"fold_{fold_text}_manifest.csv"
            ),
            "--base-checkpoint",
            str(
                PROJECT_ROOT
                / "results/experiments"
                / f"dpg_fcn_v4_fold{fold_text}_seed42"
                / "checkpoints/best.pt"
            ),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--debug-per-class",
            str(debug_per_class),
        ]

        if args.overwrite:
            command.append("--overwrite")
        if args.no_memory_cache:
            command.append("--no-memory-cache")
        if args.no_amp:
            command.append("--no-amp")

        print("\n" + "=" * 82)
        print("Running:")
        print(" ".join(command))
        print("=" * 82)
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
