#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
READINESS_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "detection_acquisition_order"
    / "audit_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one development-only BC-DPG past-only smoke test using the "
            "explicitly unverified beam/azimuth order. This is not a formal run."
        )
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--debug-per-class", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fold not in range(1, 7):
        raise ValueError("fold must be in 1-6")
    if args.history_window <= 0:
        raise ValueError("history-window must be positive")
    if args.epochs not in range(1, 5):
        raise ValueError("causal smoke is limited to 1-4 epochs")
    if args.debug_per_class not in range(1, 33):
        raise ValueError("causal smoke requires 1-32 samples per class and split")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")


def load_readiness_manifest(path: Path = READINESS_MANIFEST) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            "Run scripts/audit_detection_acquisition_order.py before causal smoke"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("verified_within_scan_order_available") is not False:
        raise ValueError("Unexpected acquisition-order readiness state")
    if payload.get("formal_causal_training_gate_open") is not False:
        raise ValueError("This smoke runner is only for the current closed-gate state")
    return payload


def experiment_name(args: argparse.Namespace) -> str:
    return (
        "bc_dpg_causal_dev_inferred_"
        f"w{args.history_window:02d}_v4_fold{args.fold:02d}_"
        f"seed{args.seed}_smoke"
    )


def build_command(args: argparse.Namespace) -> list[str]:
    fold_tag = f"fold{args.fold:02d}"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "train_target_protected_scan_calibrator.py"),
        "--name",
        experiment_name(args),
        "--manifest-path",
        str(
            PROJECT_ROOT
            / "results"
            / "data_audit"
            / "dataset_v4_multifold"
            / f"fold_{args.fold:02d}_manifest.csv"
        ),
        "--base-checkpoint",
        str(
            PROJECT_ROOT
            / "results"
            / "experiments"
            / f"dpg_fcn_v4_{fold_tag}_seed42"
            / "checkpoints"
            / "best.pt"
        ),
        "--scan-context-mode",
        "past_only",
        "--history-window",
        str(args.history_window),
        "--allow-inferred-order",
        "--validation-only",
        "--epochs",
        str(args.epochs),
        "--debug-per-class",
        str(args.debug_per_class),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def validate_output(args: argparse.Namespace) -> Path:
    summary_path = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / experiment_name(args)
        / "tables"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    context = summary.get("scan_context", {})
    if context.get("mode") != "past_only":
        raise ValueError("Smoke output did not use past_only context")
    if context.get("history_window") != args.history_window:
        raise ValueError("Smoke output history window mismatch")
    if context.get("order_verified_by_timestamp") is not False:
        raise ValueError("Smoke output incorrectly marks order as timestamp-verified")
    if context.get("evidence_role") != "development_only_inferred_order":
        raise ValueError("Smoke output evidence role mismatch")
    if summary.get("test_evaluation_performed") is not False:
        raise ValueError("Causal smoke must not evaluate the test split")
    if "test_metrics" in summary:
        raise ValueError("Causal smoke summary must not contain test metrics")
    if (summary_path.parent / "precomputed_test.csv").exists():
        raise ValueError("Causal smoke unexpectedly wrote test precomputed features")
    return summary_path


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        load_readiness_manifest()
        command = build_command(args)
        print("Development-only inferred-order smoke command:")
        print(" ".join(command))
        print(
            "validation_metrics_must_not_be_used_for_model_or_window_selection=True",
            flush=True,
        )
        if args.dry_run:
            return 0
        with tempfile.TemporaryDirectory(prefix="balloon-radar-mpl-") as mpl_cache:
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = mpl_cache
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                env=environment,
            )
        summary_path = validate_output(args)
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("BC-DPG causal development smoke: PASS")
    print(f"summary={summary_path.relative_to(PROJECT_ROOT)}")
    print("formal_causal_training_performed=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
