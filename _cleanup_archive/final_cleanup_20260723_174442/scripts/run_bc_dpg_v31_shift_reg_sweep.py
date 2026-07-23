#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGULARIZATIONS = (0.01, 0.005, 0.0025, 0.001, 0.0)
EXISTING_POLICIES = ("smart", "error", "skip", "overwrite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BC-DPG-FCN v3.1 shift-regularization candidates. "
            "Candidate selection is delegated to a validation-only selector."
        )
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--regularizations",
        nargs="+",
        type=float,
        default=list(DEFAULT_REGULARIZATIONS),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--existing-policy",
        choices=EXISTING_POLICIES,
        default="smart",
        help=(
            "smart: skip complete candidates, back up incomplete smoke runs, "
            "and reject incomplete formal runs; error: always stop; "
            "skip: always skip; overwrite: delete and rerun."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Alias for --existing-policy overwrite.",
    )
    parser.add_argument(
        "--no-reuse-full-v3",
        action="store_true",
        help="Retrain 0.01 instead of reusing completed full-v3 ablation results.",
    )
    return parser.parse_args()


def reg_slug(value: float) -> str:
    text = format(float(value), ".10g")
    return text.replace("-", "m").replace(".", "p")


def experiment_name(
    fold: int,
    regularization: float,
    seed: int,
    smoke: bool,
) -> str:
    suffix = "_smoke" if smoke else ""
    return (
        f"bc_dpg_v31_shiftreg_{reg_slug(regularization)}_"
        f"v4_fold{fold:02d}_seed{seed}{suffix}"
    )


def full_v3_name(fold: int, seed: int, smoke: bool) -> str:
    suffix = "_smoke" if smoke else ""
    return f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed{seed}{suffix}"


def completion_files(path: Path) -> tuple[Path, ...]:
    return (
        path / "tables" / "summary.json",
        path / "tables" / "test_predictions.csv",
        path / "checkpoints" / "best.pt",
    )


def inspect_existing(path: Path) -> dict[str, Any]:
    required = completion_files(path)
    missing = [str(item) for item in required if not item.is_file()]
    return {
        "exists": path.exists(),
        "complete": path.is_dir() and not missing,
        "missing": missing,
    }


def backup_incomplete(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / "results" / "experiments" / "_incomplete_backup"
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / f"{path.name}_{timestamp}"
    counter = 1
    while destination.exists():
        destination = backup_root / f"{path.name}_{timestamp}_{counter:02d}"
        counter += 1
    shutil.move(str(path), str(destination))
    return destination


def decide_action(
    path: Path,
    *,
    policy: str,
    smoke: bool,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    state = inspect_existing(path)
    if not state["exists"]:
        return "run", state
    if policy == "error":
        raise FileExistsError(f"Experiment already exists: {path}")
    if policy == "skip":
        return "skip", state
    if policy == "overwrite":
        if not dry_run:
            shutil.rmtree(path)
        return "overwrite_then_run", state

    # smart
    if state["complete"]:
        return "skip_complete", state
    if smoke:
        if dry_run:
            return "backup_incomplete_then_run", state
        destination = backup_incomplete(path)
        state["backup_path"] = str(destination)
        return "run_after_backup", state

    missing_text = "\n".join(f"  - {item}" for item in state["missing"])
    raise RuntimeError(
        "Incomplete formal experiment exists and was not changed:\n"
        f"  {path}\nMissing completion files:\n{missing_text}\n"
        "Inspect it first, then explicitly use --existing-policy overwrite if intended."
    )


def main() -> None:
    args = parse_args()
    if args.smoke and args.formal:
        raise ValueError("Choose only one of --smoke or --formal")
    smoke = bool(args.smoke or not args.formal)
    policy = "overwrite" if args.overwrite else args.existing_policy
    folds = tuple(dict.fromkeys(args.folds))
    regularizations = tuple(dict.fromkeys(float(v) for v in args.regularizations))
    if any(fold not in range(1, 7) for fold in folds):
        raise ValueError(f"Folds must be 1-6: {folds}")
    if any(value < 0 for value in regularizations):
        raise ValueError("Regularization weights must be non-negative")

    epochs = 4 if smoke else 30
    debug_per_class = 12 if smoke else 0
    batch_size = args.batch_size or (16 if smoke else 64)

    plan_dir = PROJECT_ROOT / "results" / "data_audit" / "bc_dpg_v31_shift_reg"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folds": list(folds),
        "regularizations": list(regularizations),
        "smoke": smoke,
        "epochs": epochs,
        "debug_per_class": debug_per_class,
        "batch_size": batch_size,
        "seed": args.seed,
        "existing_policy": policy,
        "reuse_full_v3": not args.no_reuse_full_v3,
        "selection_protocol": "validation_only_lexicographic",
    }
    (plan_dir / "latest_run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    statuses: list[dict[str, Any]] = []
    for fold in folds:
        fold_text = f"{fold:02d}"
        for regularization in regularizations:
            if (
                not args.no_reuse_full_v3
                and math.isclose(regularization, 0.01, rel_tol=0.0, abs_tol=1e-12)
            ):
                reuse_name = full_v3_name(fold, args.seed, smoke)
                reuse_dir = PROJECT_ROOT / "results" / "experiments" / reuse_name
                reuse_state = inspect_existing(reuse_dir)
                if reuse_state["complete"]:
                    print("\n" + "=" * 92)
                    print(
                        f"Fold={fold_text} Reg={regularization:g} Smoke={smoke} "
                        "Action=reuse_full_v3"
                    )
                    print(f"[复用] {reuse_dir}")
                    print("=" * 92)
                    statuses.append(
                        {
                            "fold": fold,
                            "regularization": regularization,
                            "experiment_name": reuse_name,
                            "experiment_dir": str(reuse_dir),
                            "action": "reuse_full_v3",
                            "run_status": "reused",
                            "existing_state": reuse_state,
                        }
                    )
                    continue
                print(
                    f"[提示] Fold {fold_text} 的 full-v3 结果不完整，"
                    "将训练独立的 0.01 候选。"
                )

            name = experiment_name(fold, regularization, args.seed, smoke)
            output_dir = PROJECT_ROOT / "results" / "experiments" / name
            action, state = decide_action(
                output_dir,
                policy=policy,
                smoke=smoke,
                dry_run=args.dry_run,
            )
            status: dict[str, Any] = {
                "fold": fold,
                "regularization": regularization,
                "experiment_name": name,
                "experiment_dir": str(output_dir),
                "action": action,
                "existing_state": state,
                "run_status": "pending",
            }
            statuses.append(status)

            print("\n" + "=" * 92)
            print(
                f"Fold={fold_text} Reg={regularization:g} Smoke={smoke} "
                f"ExistingAction={action}"
            )
            if action in {"skip", "skip_complete"}:
                print(f"[跳过] {output_dir}")
                status["run_status"] = "skipped"
                print("=" * 92)
                continue
            if "backup_path" in state:
                print(f"[恢复] 残缺 smoke 已备份到：{state['backup_path']}")

            command = [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "training"
                    / "train_target_protected_scan_calibrator_ablation.py"
                ),
                "--name",
                name,
                "--ablation-mode",
                "full",
                "--manifest-path",
                str(
                    PROJECT_ROOT
                    / "results"
                    / "data_audit"
                    / "dataset_v4_multifold"
                    / f"fold_{fold_text}_manifest.csv"
                ),
                "--base-checkpoint",
                str(
                    PROJECT_ROOT
                    / "results"
                    / "experiments"
                    / f"dpg_fcn_v4_fold{fold_text}_seed{args.seed}"
                    / "checkpoints"
                    / "best.pt"
                ),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--debug-per-class",
                str(debug_per_class),
                "--seed",
                str(args.seed),
                "--shift-regularization",
                format(regularization, ".10g"),
            ]
            if action == "overwrite_then_run":
                # Directory has already been removed by this runner.
                pass
            if args.no_memory_cache:
                command.append("--no-memory-cache")
            if args.no_amp:
                command.append("--no-amp")

            print(" ".join(command))
            print("=" * 92)
            status["command"] = command
            if args.dry_run:
                status["run_status"] = "dry_run"
                continue
            try:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                status["run_status"] = "completed"
            except subprocess.CalledProcessError as exc:
                status["run_status"] = "failed"
                status["return_code"] = exc.returncode
                (plan_dir / "latest_run_status.json").write_text(
                    json.dumps(statuses, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise

    (plan_dir / "latest_run_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.dry_run:
        selector = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "select_bc_dpg_v31_shift_reg.py"),
            "--folds",
            *[str(fold) for fold in folds],
            "--regularizations",
            *[format(value, ".10g") for value in regularizations],
            "--seed",
            str(args.seed),
            "--require-all",
        ]
        if smoke:
            selector.append("--smoke")
        subprocess.run(selector, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
