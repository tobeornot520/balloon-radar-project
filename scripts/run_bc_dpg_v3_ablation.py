#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_MODES = (
    "full",
    "no_scan_context",
    "no_background_classification",
    "no_background_tail",
    "no_target_protection",
)
ALL_MODES = (
    *CORE_MODES,
    "no_target_keep",
    "no_pairwise",
    "no_shift_selectivity",
    "no_shift_regularization",
)

EXISTING_POLICIES = (
    "smart",
    "error",
    "skip",
    "overwrite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-controlled BC-DPG-FCN v3 ablations. "
            "Defaults to smoke tests on hard folds 1 and 4."
        )
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 4],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["core"],
        help=(
            "Ablation names, or one keyword: core / all. "
            f"Core={','.join(CORE_MODES)}"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--existing-policy",
        choices=EXISTING_POLICIES,
        default="smart",
        help=(
            "How to handle an existing experiment directory. "
            "smart (default): skip complete results; back up and rerun "
            "incomplete smoke results; reject incomplete formal results. "
            "error: always stop; skip: always skip; overwrite: delete and rerun."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Backward-compatible alias for "
            "--existing-policy overwrite."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def expand_modes(values: list[str]) -> tuple[str, ...]:
    if values == ["core"]:
        return CORE_MODES
    if values == ["all"]:
        return ALL_MODES

    invalid = sorted(set(values) - set(ALL_MODES))
    if invalid:
        raise ValueError(
            f"Unknown modes: {invalid}. Valid modes: {ALL_MODES}"
        )
    return tuple(dict.fromkeys(values))


def experiment_name(
    fold: int,
    mode: str,
    seed: int,
    smoke: bool,
) -> str:
    suffix = "_smoke" if smoke else ""
    return (
        f"bc_dpg_v3_ablation_{mode}_"
        f"v4_fold{fold:02d}_seed{seed}{suffix}"
    )


def experiment_dir(name: str) -> Path:
    return PROJECT_ROOT / "results" / "experiments" / name


def completion_files(path: Path) -> tuple[Path, ...]:
    return (
        path / "tables" / "summary.json",
        path / "tables" / "test_predictions.csv",
        path / "checkpoints" / "best.pt",
    )


def inspect_existing(path: Path) -> dict[str, Any]:
    required = completion_files(path)
    found = [item for item in required if item.is_file()]
    missing = [item for item in required if not item.is_file()]
    return {
        "exists": path.exists(),
        "complete": path.is_dir() and not missing,
        "found": [str(item) for item in found],
        "missing": [str(item) for item in missing],
    }


def backup_incomplete(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / "_incomplete_backup"
    )
    backup_root.mkdir(parents=True, exist_ok=True)
    candidate = backup_root / f"{path.name}_{timestamp}"
    counter = 1
    while candidate.exists():
        candidate = backup_root / f"{path.name}_{timestamp}_{counter:02d}"
        counter += 1
    shutil.move(str(path), str(candidate))
    return candidate


def decide_existing_action(
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
        raise FileExistsError(
            f"Experiment already exists: {path}\n"
            "Use --existing-policy smart/skip/overwrite as appropriate."
        )

    if policy == "skip":
        return "skip", state

    if policy == "overwrite":
        return "overwrite", state

    # smart policy
    if state["complete"]:
        return "skip_complete", state

    if smoke:
        if dry_run:
            return "backup_incomplete_then_run", state
        backup_path = backup_incomplete(path)
        state["backup_path"] = str(backup_path)
        return "run_after_backup", state

    missing_text = "\n".join(f"  - {item}" for item in state["missing"])
    raise RuntimeError(
        "Incomplete formal experiment directory exists and was not changed:\n"
        f"  {path}\n"
        "Missing completion files:\n"
        f"{missing_text}\n"
        "Inspect it first, then explicitly use "
        "--existing-policy overwrite if rerunning is intended."
    )


def main() -> None:
    args = parse_args()
    if args.smoke and args.formal:
        raise ValueError("Choose only one of --smoke or --formal")

    if args.overwrite:
        if args.existing_policy not in {"smart", "overwrite"}:
            raise ValueError(
                "--overwrite conflicts with "
                f"--existing-policy {args.existing_policy}"
            )
        existing_policy = "overwrite"
    else:
        existing_policy = args.existing_policy

    smoke = args.smoke or not args.formal
    epochs = 4 if smoke else 30
    debug_per_class = 12 if smoke else 0
    batch_size = args.batch_size or (16 if smoke else 64)
    modes = expand_modes(list(args.modes))

    for fold in args.folds:
        if fold not in range(1, 7):
            raise ValueError(f"Fold must be 1-6, got {fold}")

    plan_dir = (
        PROJECT_ROOT
        / "results"
        / "data_audit"
        / "bc_dpg_v3_ablation"
    )
    plan_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[dict[str, Any]] = []
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folds": list(args.folds),
        "modes": list(modes),
        "smoke": smoke,
        "epochs": epochs,
        "debug_per_class": debug_per_class,
        "batch_size": batch_size,
        "seed": args.seed,
        "existing_policy": existing_policy,
        "dry_run": bool(args.dry_run),
    }
    (plan_dir / "latest_run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for fold in args.folds:
        fold_text = f"{fold:02d}"
        for mode in modes:
            name = experiment_name(
                fold,
                mode,
                args.seed,
                smoke,
            )
            output_path = experiment_dir(name)
            action, state = decide_existing_action(
                output_path,
                policy=existing_policy,
                smoke=smoke,
                dry_run=args.dry_run,
            )

            status: dict[str, Any] = {
                "fold": fold,
                "mode": mode,
                "name": name,
                "experiment_dir": str(output_path),
                "existing_action": action,
                "existing_state": state,
                "run_status": "pending",
            }
            statuses.append(status)

            print("\n" + "=" * 88)
            print(
                f"Fold={fold_text}  Mode={mode}  Smoke={smoke}  "
                f"ExistingAction={action}"
            )

            if action in {"skip", "skip_complete"}:
                label = (
                    "完整结果已存在，智能跳过"
                    if action == "skip_complete"
                    else "按策略跳过已有目录"
                )
                print(f"[跳过] {label}: {output_path}")
                status["run_status"] = "skipped"
                print("=" * 88)
                continue

            if action in {
                "backup_incomplete_then_run",
                "run_after_backup",
            }:
                backup_path = state.get("backup_path", "<dry-run backup>")
                print(
                    "[恢复] 检测到残缺 smoke 目录；"
                    f"已/将备份到：{backup_path}"
                )

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
                mode,
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
                    / f"dpg_fcn_v4_fold{fold_text}_seed42"
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
            ]
            if action == "overwrite":
                command.append("--overwrite")
            if args.no_memory_cache:
                command.append("--no-memory-cache")
            if args.no_amp:
                command.append("--no-amp")

            print(" ".join(command))
            print("=" * 88)
            status["command"] = command

            if args.dry_run:
                status["run_status"] = "dry_run"
                continue

            try:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=True,
                )
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
        summarize = [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "summarize_bc_dpg_v3_ablation.py"
            ),
            "--folds",
            *[str(fold) for fold in args.folds],
            "--modes",
            *modes,
            "--seed",
            str(args.seed),
        ]
        if smoke:
            summarize.append("--smoke")
        subprocess.run(
            summarize,
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
