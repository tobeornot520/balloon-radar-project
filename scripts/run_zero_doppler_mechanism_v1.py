#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "fixed_notch", "dense_negative", "clutter_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped zero-Doppler mechanism comparisons"
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--modes", nargs="+", default=["all"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-memory-cache", action="store_true")
    return parser.parse_args()


def expand_modes(values: list[str]) -> tuple[str, ...]:
    if values == ["all"]:
        return MODES
    invalid = sorted(set(values) - set(MODES))
    if invalid:
        raise ValueError(f"unknown modes: {invalid}")
    return tuple(dict.fromkeys(values))


def experiment_name(fold: int, mode: str, seed: int, smoke: bool) -> str:
    suffix = "_smoke" if smoke else ""
    return f"zero_doppler_v1_{mode}_fold{fold:02d}_seed{seed}{suffix}"


def is_complete(path: Path) -> bool:
    return all(
        item.is_file()
        for item in (
            path / "tables" / "summary.json",
            path / "tables" / "val_predictions.csv",
            path / "tables" / "test_predictions.csv",
        )
    )


def main() -> int:
    args = parse_args()
    if args.smoke and args.formal:
        raise ValueError("choose only one of --smoke and --formal")
    smoke = args.smoke or not args.formal
    modes = expand_modes(list(args.modes))
    if any(fold not in range(1, 7) for fold in args.folds):
        raise ValueError("folds must be in 1-6")
    epochs = 2 if smoke else 12
    debug_per_class = 8 if smoke else 0
    batch_size = args.batch_size or (8 if smoke else 16)
    audit_dir = PROJECT_ROOT / "results/data_audit/zero_doppler_mechanism_v1"
    audit_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folds": args.folds,
        "modes": modes,
        "smoke": smoke,
        "epochs": epochs,
        "debug_per_class": debug_per_class,
        "batch_size": batch_size,
        "seed": args.seed,
        "threshold_policy": "reuse_frozen_base_DPG_validation_threshold",
    }
    (audit_dir / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for fold in args.folds:
        for mode in modes:
            name = experiment_name(fold, mode, args.seed, smoke)
            output_dir = PROJECT_ROOT / "results/experiments" / name
            status: dict[str, Any] = {
                "fold": fold,
                "mode": mode,
                "name": name,
                "status": "pending",
            }
            statuses.append(status)
            if is_complete(output_dir) and not args.overwrite:
                status["status"] = "skipped_complete"
                print(f"[skip] {name}")
                continue
            if output_dir.exists() and not args.overwrite:
                raise FileExistsError(
                    f"incomplete experiment exists: {output_dir}; inspect or use --overwrite"
                )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "training/train_zero_doppler_mechanism.py"),
                "--name",
                name,
                "--mode",
                mode,
                "--manifest-path",
                str(
                    PROJECT_ROOT
                    / f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
                ),
                "--base-checkpoint",
                str(
                    PROJECT_ROOT
                    / f"results/experiments/dpg_fcn_v4_fold{fold:02d}_seed42/checkpoints/best.pt"
                ),
                "--epochs",
                str(epochs),
                "--debug-per-class",
                str(debug_per_class),
                "--batch-size",
                str(batch_size),
                "--seed",
                str(args.seed),
            ]
            if args.overwrite:
                command.append("--overwrite")
            if args.no_memory_cache:
                command.append("--no-memory-cache")
            status["command"] = command
            print(" ".join(command))
            if args.dry_run:
                status["status"] = "dry_run"
                continue
            try:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                status["status"] = "completed"
            except subprocess.CalledProcessError as exc:
                status["status"] = "failed"
                status["return_code"] = exc.returncode
                (audit_dir / "run_status.json").write_text(
                    json.dumps(statuses, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise

    (audit_dir / "run_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.dry_run:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/summarize_zero_doppler_mechanism_v1.py"),
            "--folds",
            *[str(fold) for fold in args.folds],
            "--modes",
            *modes,
            "--seed",
            str(args.seed),
        ]
        if smoke:
            command.append("--smoke")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
