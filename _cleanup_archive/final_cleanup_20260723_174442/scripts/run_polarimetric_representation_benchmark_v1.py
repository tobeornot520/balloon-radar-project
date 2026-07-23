#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("power2", "ri4", "polar6", "ri8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--smoke", action="store_true")
    scope.add_argument("--formal", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overwrite-incomplete", action="store_true")
    return p.parse_args()


def complete(name: str) -> bool:
    root = PROJECT_ROOT / "results/experiments" / name
    return (root / "checkpoints/best.pt").is_file() and (root / "tables/summary.json").is_file()


def run(command: list[str]) -> None:
    print("\n" + "=" * 100)
    print("COMMAND:")
    print(" ".join(command))
    print("=" * 100)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if any(fold <= 0 for fold in args.folds):
        raise ValueError("folds must be positive")
    scope = "formal" if args.formal else "smoke"
    epochs = args.epochs if args.epochs is not None else (100 if args.formal else 3)
    debug_per_class = 0 if args.formal else 8
    out = PROJECT_ROOT / "results/data_audit/polarimetric_representation_benchmark_v1"
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now().isoformat(),
        "scope": scope,
        "folds": args.folds,
        "modes": args.modes,
        "epochs": epochs,
        "debug_per_class": debug_per_class,
        "scientific_role": "sample-independent representation comparison; no scan context",
    }
    (out / "latest_run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for fold in args.folds:
        manifest = PROJECT_ROOT / f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        for mode in args.modes:
            name = f"polar_repr_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"
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
                "training/train_polarimetric_representation_fcn.py",
                "--name", name,
                "--manifest-path", str(manifest),
                "--input-mode", mode,
                "--dataset-version", "V4",
                "--fold-id", str(fold),
                "--epochs", str(epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--seed", "42",
            ]
            if debug_per_class:
                command.extend(["--debug-per-class", str(debug_per_class)])
            if root.exists() and args.overwrite_incomplete:
                command.append("--overwrite")
            run(command)

    run([
        sys.executable,
        "scripts/summarize_polarimetric_representation_benchmark_v1.py",
        "--folds", *[str(v) for v in args.folds],
        "--modes", *args.modes,
        "--scope", scope,
        "--require-all",
    ])


if __name__ == "__main__":
    main()
