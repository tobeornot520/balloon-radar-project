#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.thesis_tian2024_dataset import ThesisTian2024Dataset  # noqa: E402
from evaluation.thesis_tian2024_postprocess import direct_max_detections  # noqa: E402
from models.thesis_tian2024_adapter import ThesisTian2024Adapter  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA256 must be 64 lowercase hexadecimal characters")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def compute_frozen_metrics(
    table: pd.DataFrame,
    threshold: float,
    range_tolerance: int,
    velocity_tolerance: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "target_present",
        "peak_score",
        "peak_grid_x",
        "peak_grid_y",
        "pred_range_index",
        "pred_velocity_index",
        "true_range_index",
        "true_velocity_index",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    table = table.copy()
    positive = table["target_present"] == 1
    background = table["target_present"] == 0
    if not positive.any() or not background.any():
        raise ValueError("held-out partition must contain targets and backgrounds")
    table["detected"] = table["peak_score"] > threshold
    table["range_error_gates"] = np.where(
        positive,
        (table["pred_range_index"] - table["true_range_index"]).abs(),
        np.nan,
    )
    table["velocity_error_bins"] = np.where(
        positive,
        (table["pred_velocity_index"] - table["true_velocity_index"]).abs(),
        np.nan,
    )
    table["responsible_grid_selected"] = positive & (
        table["peak_grid_x"] == table["true_range_index"] // 4
    ) & (table["peak_grid_y"] == table["true_velocity_index"] // 4)
    table["localization_ok"] = positive & (
        table["range_error_gates"] <= range_tolerance
    ) & (table["velocity_error_bins"] <= velocity_tolerance)
    table["joint_ok"] = table["detected"] & table["localization_ok"]
    target_count = int(positive.sum())
    background_count = int(background.sum())
    joint_count = int(table.loc[positive, "joint_ok"].sum())
    false_alarm_count = int(table.loc[background, "detected"].sum())
    metrics = {
        "frozen_threshold": float(threshold),
        "target_count": target_count,
        "background_count": background_count,
        "joint_success_count": joint_count,
        "false_alarm_count": false_alarm_count,
        "joint_pd": float(joint_count / target_count),
        "pfa": float(false_alarm_count / background_count),
        "responsible_grid_selection_rate": float(
            table.loc[positive, "responsible_grid_selected"].mean()
        ),
        "range_mae_gates": float(table.loc[positive, "range_error_gates"].mean()),
        "velocity_mae_bins": float(
            table.loc[positive, "velocity_error_bins"].mean()
        ),
        "unique_target_peak_grids": int(
            table.loc[positive, ["peak_grid_y", "peak_grid_x"]]
            .drop_duplicates()
            .shape[0]
        ),
    }
    return metrics, table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen thesis Tian2024 adapter on local held-out data"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.range_tolerance_gates < 0 or args.velocity_tolerance_bins < 0:
        raise ValueError("localization tolerances must be nonnegative")
    checkpoint_path = resolve_path(args.checkpoint)
    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = require_hash(checkpoint_path, args.checkpoint_sha256)
    manifest_hash = require_hash(manifest_path, args.manifest_sha256)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if config.get("normalization_scope") != "sample_channel":
        raise RuntimeError("frozen candidate must use sample_channel normalization")
    frozen_validation_threshold = float(
        payload.get("validation_metrics", {}).get("validation_threshold", math.nan)
    )
    if not math.isclose(args.threshold, frozen_validation_threshold, abs_tol=1e-12):
        raise RuntimeError(
            "requested threshold does not match the checkpoint validation threshold"
        )

    model = ThesisTian2024Adapter("sample_channel").to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    dataset = ThesisTian2024Dataset(manifest_path, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            output = model(inputs)
            detections = direct_max_detections(
                output.classification_logits.detach().float().cpu(),
                output.normalized_offsets.detach().float().cpu(),
                output.original_shape,
                threshold=None,
            )
            for index, detection in enumerate(detections):
                if detection is None:
                    raise RuntimeError("unthresholded direct-max decoding failed")
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "target_present": int(batch["target_present"][index]),
                        "peak_score": detection.score,
                        "peak_grid_x": detection.grid_x,
                        "peak_grid_y": detection.grid_y,
                        "pred_range_index": detection.range_index,
                        "pred_velocity_index": detection.velocity_index,
                        "true_range_index": int(batch["range_index"][index]),
                        "true_velocity_index": int(batch["velocity_index"][index]),
                    }
                )
    metrics, table = compute_frozen_metrics(
        pd.DataFrame(rows),
        args.threshold,
        args.range_tolerance_gates,
        args.velocity_tolerance_bins,
    )
    metrics["seconds_per_map"] = float((time.perf_counter() - started) / len(dataset))
    table.to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "status": "COMPLETE_SINGLE_FROZEN_LOCAL_HELDOUT_EVALUATION",
        "evidence_role": "historical_local_heldout_not_external_blind_test",
        "method": "thesis_tian2024_local_adaptation_v1",
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "normalization_scope": "sample_channel",
        "threshold_source": "frozen_validation",
        "test_retuning_performed": False,
        "metrics": metrics,
        "not_claimed": [
            "external_blind_generalization",
            "exact_numeric_thesis_reproduction",
            "balloon_payload_recognition",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
