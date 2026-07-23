#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "power2_baseline",
    "power2_roi_power_control",
    "power2_roi_ri4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected Stage-4 modes across scan-group folds.")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--smoke", action="store_true")
    scope.add_argument("--formal", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + "=" * 100)
    print("COMMAND:")
    print(" ".join(command))
    print("=" * 100, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def experiment_name(mode: str, fold: int, scope: str) -> str:
    return f"roi_polar_stage4_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"


def complete(mode: str, fold: int, scope: str) -> bool:
    tables = PROJECT_ROOT / "results/experiments" / experiment_name(mode, fold, scope) / "tables"
    return all((tables / name).is_file() for name in ("summary.json", "val_predictions.csv", "test_predictions.csv"))


def main() -> None:
    args = parse_args()
    scope = "formal" if args.formal else "smoke"
    epochs = args.epochs if args.epochs is not None else (100 if args.formal else 3)
    debug_per_class = 0 if args.formal else 8

    audit_dir = PROJECT_ROOT / "results/data_audit/roi_stage4_selected_sixfold_v1"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now().isoformat(),
        "scope": scope,
        "folds": args.folds,
        "modes": args.modes,
        "epochs": epochs,
        "selection_frozen_before_sixfold": True,
        "sample_independent": True,
        "scan_context": False,
        "power2_location_frozen": True,
    }
    (audit_dir / f"run_plan_{scope}.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for fold in args.folds:
        manifest = PROJECT_ROOT / f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
        base_checkpoint = PROJECT_ROOT / (
            f"results/experiments/polar_repr_v2_power2_v4_fold{fold:02d}_seed42_formal/"
            "checkpoints/best.pt"
        )
        if not manifest.is_file() or not base_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing manifest or base Power2 checkpoint for fold {fold}: "
                f"{manifest} / {base_checkpoint}"
            )

        cache_dir = PROJECT_ROOT / (
            f"results/data_audit/roi_polarimetric_stage4_v1/cache/fold_{fold:02d}_{scope}"
        )
        cache_command = [
            sys.executable,
            "scripts/build_roi_polarimetric_cache_v1.py",
            "--fold", str(fold),
            "--manifest-path", str(manifest),
            "--base-checkpoint", str(base_checkpoint),
            "--output-dir", str(cache_dir),
            "--batch-size", str(args.cache_batch_size),
            "--num-workers", str(args.num_workers),
            "--debug-per-class", str(debug_per_class),
            "--roi-velocity-radius", "5",
            "--roi-range-radius", "4",
        ]
        if args.rebuild_cache:
            cache_command.append("--overwrite")
        run(cache_command)

        for mode in args.modes:
            if complete(mode, fold, scope):
                print(f"[skip complete] {experiment_name(mode, fold, scope)}")
                continue
            output_dir = PROJECT_ROOT / "results/experiments" / experiment_name(mode, fold, scope)
            if output_dir.exists() and not args.overwrite_incomplete:
                raise FileExistsError(
                    f"Incomplete result exists: {output_dir}. Add --overwrite-incomplete to rerun."
                )
            train_command = [
                sys.executable,
                "training/train_roi_polarimetric_refiner_v1.py",
                "--name", experiment_name(mode, fold, scope),
                "--cache-dir", str(cache_dir),
                "--mode", mode,
                "--fold-id", str(fold),
                "--epochs", str(epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--debug-per-class", str(debug_per_class),
                "--max-val-pd-drop", "0.01",
                "--seed", "42",
            ]
            if output_dir.exists() and args.overwrite_incomplete:
                train_command.append("--overwrite")
            run(train_command)

    run([
        sys.executable,
        "scripts/summarize_roi_stage4_selected_sixfold_v1.py",
        "--folds", *[str(value) for value in args.folds],
        "--modes", *args.modes,
        "--scope", scope,
        "--require-all",
    ])


if __name__ == "__main__":
    main()
