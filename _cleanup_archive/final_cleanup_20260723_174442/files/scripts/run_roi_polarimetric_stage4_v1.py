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
    "power2_roi_polar6_gated",
    "power2_roi_ri4_polar6_gated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-4 ROI polarimetric refinement v1.")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--smoke", action="store_true")
    scope.add_argument("--formal", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--roi-velocity-radius", type=int, default=5)
    parser.add_argument("--roi-range-radius", type=int, default=4)
    parser.add_argument("--max-val-pd-drop", type=float, default=0.01)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + "=" * 108)
    print("COMMAND:")
    print(" ".join(command))
    print("=" * 108)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def experiment_name(mode: str, fold: int, scope: str) -> str:
    return f"roi_polar_stage4_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"


def complete(name: str) -> bool:
    root = PROJECT_ROOT / "results/experiments" / name
    return (root / "checkpoints/best.pt").is_file() and (root / "tables/summary.json").is_file()


def main() -> None:
    args = parse_args()
    if any(fold <= 0 for fold in args.folds):
        raise ValueError("folds must be positive")
    scope = "formal" if args.formal else "smoke"
    epochs = args.epochs if args.epochs is not None else (100 if args.formal else 3)
    debug_per_class = 0 if args.formal else 8
    audit_dir = PROJECT_ROOT / "results/data_audit/roi_polarimetric_stage4_v1"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now().isoformat(),
        "scope": scope,
        "folds": args.folds,
        "modes": args.modes,
        "epochs": epochs,
        "debug_per_class": debug_per_class,
        "roi_velocity_radius": args.roi_velocity_radius,
        "roi_range_radius": args.roi_range_radius,
        "max_val_pd_drop": args.max_val_pd_drop,
        "base_detector": "frozen Power2 formal checkpoint",
        "primary_threshold": "unchanged fold-specific Power2 deployment threshold",
        "sample_independent": True,
        "scan_context": False,
        "power2_location_frozen": True,
    }
    (audit_dir / f"latest_run_plan_{scope}.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for fold in args.folds:
        manifest = PROJECT_ROOT / f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
        base_checkpoint = PROJECT_ROOT / (
            f"results/experiments/polar_repr_v2_power2_v4_fold{fold:02d}_seed42_formal/"
            "checkpoints/best.pt"
        )
        if not manifest.is_file() or not base_checkpoint.is_file():
            raise FileNotFoundError(f"Missing manifest/base checkpoint: {manifest} / {base_checkpoint}")
        cache_dir = audit_dir / "cache" / f"fold_{fold:02d}_{scope}"
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
            "--roi-velocity-radius", str(args.roi_velocity_radius),
            "--roi-range-radius", str(args.roi_range_radius),
        ]
        if args.rebuild_cache:
            cache_command.append("--overwrite")
        run(cache_command)

        for mode in args.modes:
            name = experiment_name(mode, fold, scope)
            root = PROJECT_ROOT / "results/experiments" / name
            if complete(name):
                print(f"[skip complete] {name}")
                continue
            if root.exists() and not args.overwrite_incomplete:
                raise FileExistsError(
                    f"Incomplete result exists: {root}. Add --overwrite-incomplete to rerun."
                )
            command = [
                sys.executable,
                "training/train_roi_polarimetric_refiner_v1.py",
                "--name", name,
                "--cache-dir", str(cache_dir),
                "--mode", mode,
                "--fold-id", str(fold),
                "--epochs", str(epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--debug-per-class", str(debug_per_class),
                "--max-val-pd-drop", str(args.max_val_pd_drop),
                "--seed", "42",
            ]
            if root.exists() and args.overwrite_incomplete:
                command.append("--overwrite")
            run(command)

    run([
        sys.executable,
        "scripts/summarize_roi_polarimetric_stage4_v1.py",
        "--folds", *[str(value) for value in args.folds],
        "--modes", *args.modes,
        "--scope", scope,
        "--require-all",
    ])


if __name__ == "__main__":
    main()
