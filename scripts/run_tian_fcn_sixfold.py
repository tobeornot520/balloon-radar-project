#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Tian FCN transfer across grouped folds"
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--smoke", action="store_true")
    scope.add_argument("--formal", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--channels", nargs="+", choices=("H", "V", "HV"), default=["H"]
    )
    parser.add_argument("--classification-epochs", type=int, default=None)
    parser.add_argument("--regression-epochs", type=int, default=None)
    parser.add_argument("--joint-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-per-class", type=int, default=4)
    parser.add_argument("--max-val-false-alarms", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment-root", default="results/experiments")
    parser.add_argument("--summary-output-dir", default="")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def experiment_name(channel: str, fold: int, scope: str) -> str:
    return (
        f"tian_fcn_reproduction_v1_{channel.lower()}_"
        f"fold{fold:02d}_seed42_{scope}"
    )


def result_complete(
    path: Path,
    scope: str,
    expected_config: dict[str, object],
) -> bool:
    summary_path = path / "tables" / "summary.json"
    if not summary_path.is_file() or not (path / "checkpoints" / "best.pt").is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        return False
    if payload.get("scope") != scope:
        return False
    config = payload.get("config")
    if not isinstance(config, dict):
        return False
    if any(config.get(key) != value for key, value in expected_config.items()):
        return False
    test_split_loaded = payload.get("test_split_loaded")
    if scope == "formal":
        return test_split_loaded is True and (
            path / "tables" / "test_predictions.csv"
        ).is_file()
    return test_split_loaded is False


def run(command: list[str]) -> None:
    print("\nCOMMAND:\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    scope = "formal" if args.formal else "smoke"
    folds = args.folds if args.folds is not None else ([1, 2, 3, 4, 5, 6] if args.formal else [1])
    if not folds or any(fold < 1 or fold > 6 for fold in folds):
        raise ValueError("folds must be between 1 and 6")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    if len(set(args.channels)) != len(args.channels):
        raise ValueError("channels must not contain duplicates")
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_val_false_alarms < 0:
        raise ValueError("invalid batch, worker, or false-alarm count")
    if args.debug_per_class <= 0:
        raise ValueError("--debug-per-class must be positive for smoke")

    default_epochs = 20 if args.formal else 1
    classification_epochs = (
        default_epochs
        if args.classification_epochs is None
        else args.classification_epochs
    )
    regression_epochs = (
        default_epochs if args.regression_epochs is None else args.regression_epochs
    )
    joint_epochs = default_epochs if args.joint_epochs is None else args.joint_epochs
    if min(classification_epochs, regression_epochs, joint_epochs) <= 0:
        raise ValueError("all stage epoch counts must be positive")

    experiment_root = resolve_path(args.experiment_root)
    plan_dir = PROJECT_ROOT / "results" / "data_audit" / "tian_fcn_reproduction_v1"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now().isoformat(),
        "scope": scope,
        "folds": folds,
        "channels": args.channels,
        "stage_epochs": {
            "classification": classification_epochs,
            "regression": regression_epochs,
            "joint": joint_epochs,
        },
        "validation_threshold_only": True,
        "test_threshold_retuning": False,
        "test_loaded_only_after_model_selection": scope == "formal",
        "primary_reproduction_channel": "H",
    }
    (plan_dir / f"run_plan_{scope}.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for channel in args.channels:
        for fold in folds:
            manifest = PROJECT_ROOT / (
                f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
            )
            if not manifest.is_file():
                raise FileNotFoundError(f"missing fold manifest: {manifest}")
            name = experiment_name(channel, fold, scope)
            experiment_dir = experiment_root / name
            expected_config: dict[str, object] = {
                "scope": scope,
                "fold_id": fold,
                "channel": channel,
                "classification_epochs": classification_epochs,
                "regression_epochs": regression_epochs,
                "joint_epochs": joint_epochs,
                "batch_size": args.batch_size,
                "max_val_false_alarms": args.max_val_false_alarms,
                "debug_per_class": args.debug_per_class if scope == "smoke" else 0,
            }
            if result_complete(experiment_dir, scope, expected_config):
                print(f"[skip complete] {name}")
                continue
            if experiment_dir.exists() and not args.overwrite_incomplete:
                raise FileExistsError(
                    f"incomplete output exists: {experiment_dir}; "
                    "use --overwrite-incomplete explicitly"
                )
            command = [
                sys.executable,
                "training/train_tian_fcn.py",
                "--name",
                name,
                "--scope",
                scope,
                "--manifest-path",
                str(manifest),
                "--output-root",
                str(experiment_root),
                "--fold-id",
                str(fold),
                "--channel",
                channel,
                "--classification-epochs",
                str(classification_epochs),
                "--regression-epochs",
                str(regression_epochs),
                "--joint-epochs",
                str(joint_epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--max-val-false-alarms",
                str(args.max_val_false_alarms),
                "--device",
                args.device,
            ]
            if scope == "smoke":
                command.extend(("--debug-per-class", str(args.debug_per_class)))
            if args.overwrite_incomplete and experiment_dir.exists():
                command.append("--overwrite")
            if args.no_memory_cache:
                command.append("--no-memory-cache")
            if args.no_amp:
                command.append("--no-amp")
            run(command)

    summary_command = [
        sys.executable,
        "scripts/summarize_tian_fcn_sixfold.py",
        "--scope",
        scope,
        "--folds",
        *[str(fold) for fold in folds],
        "--channels",
        *args.channels,
        "--experiment-root",
        str(experiment_root),
        "--require-all",
    ]
    if args.summary_output_dir:
        summary_command.extend(("--output-dir", args.summary_output_dir))
    run(summary_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
