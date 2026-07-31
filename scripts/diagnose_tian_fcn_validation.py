#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3
from evaluation.tian_fcn_postprocess import tian_pir_mdp
from models.tian_fcn import TianFastUAVFCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Tian FCN score and offset behavior without test data"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--channel", choices=("H", "V", "HV"), default="H")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--compare-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-margin", type=float, default=0.1)
    parser.add_argument("--absolute-threshold", type=float)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def describe(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def checkpoint_deltas(
    current: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    selected = (
        "shared_conv1.weight",
        "classification_head.weight",
        "regression_branch.conv2.weight",
        "regression_branch.conv4.weight",
        "regression_head.weight",
        "regression_head.bias",
    )
    result = {}
    for name in selected:
        difference = (current[name].float() - reference[name].float()).abs()
        result[name] = {
            "max_absolute_delta": float(difference.max()),
            "mean_absolute_delta": float(difference.mean()),
        }
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.probability_margin <= 1.0:
        raise ValueError("--probability-margin must be between zero and one")
    if args.absolute_threshold is not None and not 0.0 <= args.absolute_threshold <= 1.0:
        raise ValueError("--absolute-threshold must be between zero and one")
    if args.range_tolerance_gates < 0 or args.velocity_tolerance_bins < 0:
        raise ValueError("localization tolerances must be nonnegative")
    checkpoint_path = resolve_path(args.checkpoint)
    manifest_path = resolve_path(args.manifest_path)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TianFastUAVFCN(in_channels=2 if args.channel == "HV" else 1)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    dataset = DetectionRadarDatasetV3(
        manifest_path=manifest_path,
        split=args.split,
        channel_mode=args.channel,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    peak_cells: Counter[tuple[int, int]] = Counter()
    target_probability_maps: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["input"])
            probability = torch.sigmoid(output.classification_logits)
            detections = tian_pir_mdp(
                output.classification_logits,
                output.normalized_offsets,
                original_shape=output.original_shape,
                output_stride=model.output_stride,
                probability_margin=args.probability_margin,
                absolute_threshold=args.absolute_threshold,
            )
            flattened = probability.flatten(1)
            peak_score, flat_index = flattened.max(dim=1)
            peak_y = flat_index // probability.shape[-1]
            peak_x = flat_index % probability.shape[-1]
            for index in range(len(peak_score)):
                present = int(batch["target_present"][index])
                true_range = int(batch["range_index"][index])
                true_velocity = int(batch["velocity_index"][index])
                if present:
                    target_probability_maps.append(
                        probability[index, 0].detach().cpu().numpy().copy()
                    )
                    grid_x = true_range // model.output_stride[1]
                    grid_y = true_velocity // model.output_stride[0]
                    target_grid_score = float(probability[index, 0, grid_y, grid_x])
                    range_offset = float(
                        output.normalized_offsets[index, 0, grid_y, grid_x]
                    )
                    velocity_offset = float(
                        output.normalized_offsets[index, 1, grid_y, grid_x]
                    )
                    target_detections = detections[index]
                    threshold = (
                        target_detections[0].threshold
                        if target_detections
                        else math.nan
                    )
                    selected = min(
                        target_detections,
                        key=lambda item: math.hypot(
                            item.range_index - true_range,
                            item.velocity_index - true_velocity,
                        ),
                        default=None,
                    )
                    selected_cells = (
                        selected.component_cells if selected is not None else ()
                    )
                    responsible_selected = any(
                        item.grid_x == grid_x and item.grid_y == grid_y
                        for item in target_detections
                    )
                    responsible_in_selected_component = any(
                        (grid_y, grid_x) in item.component_cells
                        for item in target_detections
                    )
                    responsible_above_threshold = (
                        target_grid_score > threshold
                        if math.isfinite(threshold)
                        else False
                    )
                    responsible_decoded_range = round(
                        grid_x * model.output_stride[1]
                        + range_offset * model.output_stride[1]
                    )
                    responsible_decoded_velocity = round(
                        grid_y * model.output_stride[0]
                        + velocity_offset * model.output_stride[0]
                    )
                    oracle_range_error = abs(
                        responsible_decoded_range - true_range
                    )
                    oracle_velocity_error = abs(
                        responsible_decoded_velocity - true_velocity
                    )
                else:
                    grid_x = -1
                    grid_y = -1
                    target_grid_score = math.nan
                    range_offset = math.nan
                    velocity_offset = math.nan
                    target_detections = detections[index]
                    threshold = (
                        target_detections[0].threshold
                        if target_detections
                        else math.nan
                    )
                    selected = None
                    selected_cells = ()
                    responsible_selected = False
                    responsible_in_selected_component = False
                    responsible_above_threshold = False
                    oracle_range_error = math.nan
                    oracle_velocity_error = math.nan
                cell = (int(peak_y[index]), int(peak_x[index]))
                peak_cells[cell] += 1
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "source_file": batch["source_file"][index],
                        "beam_layer": int(batch["beam_layer"][index]),
                        "velocity_mps": float(batch["velocity_mps"][index]),
                        "target_present": present,
                        "peak_score": float(peak_score[index]),
                        "peak_grid_y": cell[0],
                        "peak_grid_x": cell[1],
                        "true_grid_y": grid_y,
                        "true_grid_x": grid_x,
                        "target_grid_score": target_grid_score,
                        "responsible_range_offset": range_offset,
                        "responsible_velocity_offset": velocity_offset,
                        "true_responsible_range_offset": (
                            true_range % model.output_stride[1]
                        ) / model.output_stride[1] if present else math.nan,
                        "true_responsible_velocity_offset": (
                            true_velocity % model.output_stride[0]
                        ) / model.output_stride[0] if present else math.nan,
                        "pir_threshold": threshold,
                        "final_detection_count": len(target_detections),
                        "pir_component_count": (
                            selected.component_count if selected is not None else 0
                        ),
                        "selected_component_size": (
                            selected.component_size if selected is not None else 0
                        ),
                        "selected_component_mean_score": (
                            selected.component_mean_score
                            if selected is not None
                            else math.nan
                        ),
                        "selected_component_max_score": (
                            selected.component_max_score
                            if selected is not None
                            else math.nan
                        ),
                        "selected_component_grid_y_span": (
                            max(row for row, _ in selected_cells)
                            - min(row for row, _ in selected_cells)
                            + 1
                            if selected_cells
                            else 0
                        ),
                        "selected_component_grid_y_min": (
                            min(row for row, _ in selected_cells)
                            if selected_cells
                            else -1
                        ),
                        "selected_component_grid_y_max": (
                            max(row for row, _ in selected_cells)
                            if selected_cells
                            else -1
                        ),
                        "pir_velocity_edge_component_count": (
                            sum(
                                min_y == 0
                                or max_y == probability.shape[-2] - 1
                                for min_y, max_y, _, _ in selected.component_bounds
                            )
                            if selected is not None
                            else 0
                        ),
                        "selected_component_touches_velocity_edge": (
                            any(row in {0, probability.shape[-2] - 1} for row, _ in selected_cells)
                            if selected_cells
                            else False
                        ),
                        "selected_component_grid_x_span": (
                            max(col for _, col in selected_cells)
                            - min(col for _, col in selected_cells)
                            + 1
                            if selected_cells
                            else 0
                        ),
                        "responsible_cell_above_threshold": responsible_above_threshold,
                        "responsible_cell_in_selected_component": (
                            responsible_in_selected_component
                        ),
                        "responsible_cell_selected": responsible_selected,
                        "responsible_oracle_range_error": oracle_range_error,
                        "responsible_oracle_velocity_error": oracle_velocity_error,
                        "responsible_oracle_localization_ok": (
                            oracle_range_error <= args.range_tolerance_gates
                            and oracle_velocity_error <= args.velocity_tolerance_bins
                            if present
                            else False
                        ),
                        "nearest_selected_grid_y": (
                            selected.grid_y if selected is not None else -1
                        ),
                        "nearest_selected_grid_x": (
                            selected.grid_x if selected is not None else -1
                        ),
                        "nearest_selected_grid_distance": (
                            math.hypot(selected.grid_x - grid_x, selected.grid_y - grid_y)
                            if selected is not None and present
                            else math.nan
                        ),
                        "selected_offset_norm": (
                            math.hypot(
                                float(
                                    output.normalized_offsets[
                                        index, 0, selected.grid_y, selected.grid_x
                                    ]
                                ),
                                float(
                                    output.normalized_offsets[
                                        index, 1, selected.grid_y, selected.grid_x
                                    ]
                                ),
                            )
                            if selected is not None
                            else math.nan
                        ),
                        "responsible_offset_norm": (
                            math.hypot(range_offset, velocity_offset)
                            if present
                            else math.nan
                        ),
                        "nearest_decoded_euclidean_error": (
                            math.hypot(
                                selected.range_index - true_range,
                                selected.velocity_index - true_velocity,
                            )
                            if selected is not None and present
                            else math.nan
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    background = frame.loc[frame["target_present"].eq(0)]
    target = frame.loc[frame["target_present"].eq(1)]
    target_probability_stack = np.stack(target_probability_maps)
    target_probability_template = target_probability_stack.mean(axis=0)
    target_probability_std_map = target_probability_stack.std(axis=0)
    centered_maps = target_probability_stack.reshape(len(target), -1)
    centered_maps = centered_maps - centered_maps.mean(axis=1, keepdims=True)
    centered_template = target_probability_template.reshape(-1)
    centered_template = centered_template - centered_template.mean()
    denominator = np.linalg.norm(centered_maps, axis=1) * np.linalg.norm(
        centered_template
    )
    template_correlations = np.divide(
        centered_maps @ centered_template,
        denominator,
        out=np.full(len(target), np.nan),
        where=denominator > 0,
    )
    mean_flat_indices = np.argsort(target_probability_template, axis=None)[::-1][:12]
    mean_peak_cells = [
        np.unravel_index(index, target_probability_template.shape)
        for index in mean_flat_indices
    ]
    summary: dict[str, Any] = {
        "status": "PASS",
        "evidence_role": "train_validation_diagnostic_only",
        "test_split_loaded": False,
        "split": args.split,
        "channel": args.channel,
        "sample_count": len(frame),
        "background_peak_score": describe(background["peak_score"].tolist()),
        "target_peak_score": describe(target["peak_score"].tolist()),
        "target_responsible_grid_score": describe(
            target["target_grid_score"].tolist()
        ),
        "target_probability_template_correlation": describe(
            template_correlations.tolist()
        ),
        "target_probability_cellwise_std": describe(
            target_probability_std_map.reshape(-1).tolist()
        ),
        "target_responsible_range_offset": describe(
            target["responsible_range_offset"].tolist()
        ),
        "target_responsible_velocity_offset": describe(
            target["responsible_velocity_offset"].tolist()
        ),
        "target_responsible_cell_above_threshold_rate": float(
            target["responsible_cell_above_threshold"].mean()
        ),
        "target_responsible_cell_selected_rate": float(
            target["responsible_cell_selected"].mean()
        ),
        "target_responsible_cell_in_selected_component_rate": float(
            target["responsible_cell_in_selected_component"].mean()
        ),
        "target_responsible_oracle_range_error": describe(
            target["responsible_oracle_range_error"].tolist()
        ),
        "target_responsible_oracle_velocity_error": describe(
            target["responsible_oracle_velocity_error"].tolist()
        ),
        "target_responsible_oracle_joint_rate": float(
            target["responsible_oracle_localization_ok"].mean()
        ),
        "target_final_detection_count": describe(
            target["final_detection_count"].tolist()
        ),
        "target_pir_component_count": describe(
            target["pir_component_count"].tolist()
        ),
        "target_selected_component_size": describe(
            target["selected_component_size"].tolist()
        ),
        "target_selected_component_mean_score": describe(
            target["selected_component_mean_score"].tolist()
        ),
        "target_selected_component_max_score": describe(
            target["selected_component_max_score"].tolist()
        ),
        "target_selected_component_grid_y_span": describe(
            target["selected_component_grid_y_span"].tolist()
        ),
        "target_pir_velocity_edge_component_count": describe(
            target["pir_velocity_edge_component_count"].tolist()
        ),
        "target_selected_component_touches_velocity_edge_rate": float(
            target["selected_component_touches_velocity_edge"].mean()
        ),
        "target_selected_component_grid_x_span": describe(
            target["selected_component_grid_x_span"].tolist()
        ),
        "target_selected_offset_norm": describe(
            target["selected_offset_norm"].tolist()
        ),
        "target_responsible_offset_norm": describe(
            target["responsible_offset_norm"].tolist()
        ),
        "target_nearest_selected_grid_distance": describe(
            target["nearest_selected_grid_distance"].tolist()
        ),
        "target_nearest_decoded_euclidean_error": describe(
            target["nearest_decoded_euclidean_error"].tolist()
        ),
        "top_peak_grid_cells": [
            {"grid_y": cell[0], "grid_x": cell[1], "count": count}
            for cell, count in peak_cells.most_common(12)
        ],
        "top_target_mean_probability_cells": [
            {
                "grid_y": int(cell[0]),
                "grid_x": int(cell[1]),
                "mean_probability": float(target_probability_template[cell]),
                "sample_std": float(target_probability_std_map[cell]),
            }
            for cell in mean_peak_cells
        ],
    }
    if args.compare_checkpoint is not None:
        reference_path = resolve_path(args.compare_checkpoint)
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=False
        )
        summary["checkpoint_parameter_deltas"] = checkpoint_deltas(
            payload["model_state_dict"], reference["model_state_dict"]
        )

    frame.to_csv(output_dir / "validation_diagnostic_rows.csv", index=False)
    pd.DataFrame(target_probability_template).to_csv(
        output_dir / "target_probability_mean_map.csv", index=False
    )
    pd.DataFrame(target_probability_std_map).to_csv(
        output_dir / "target_probability_std_map.csv", index=False
    )
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
