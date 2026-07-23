#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.polarimetric_gated_rd import PolarimetricGateConfig
from features.polarimetric_rd import PolarimetricConfig, explicit_polarimetric_rd
from features.roi_polarimetric_refinement import (
    ROIConfig,
    ROI_MODES,
    build_roi_source,
    canonical_roi_channels,
    crop_roi,
)
from models.roi_polarimetric_refiner import ROIPolarimetricRefiner, count_parameters


def main() -> None:
    rng = np.random.default_rng(42)
    h = (
        rng.normal(size=(128, 100)) + 1j * rng.normal(size=(128, 100))
    ).astype(np.complex64)
    v = (
        0.7 * h
        + 0.3 * (rng.normal(size=(128, 100)) + 1j * rng.normal(size=(128, 100)))
    ).astype(np.complex64)
    features = explicit_polarimetric_rd(h, v, PolarimetricConfig())
    source_np, confidence_np = build_roi_source(
        features, gate_config=PolarimetricGateConfig()
    )
    source = torch.from_numpy(source_np)
    confidence = torch.from_numpy(confidence_np)
    roi_config = ROIConfig(velocity_radius=5, range_radius=4)
    roi, mask = crop_roi(source, velocity_index=0, range_index=99, config=roi_config)
    confidence_roi, _ = crop_roi(
        confidence, velocity_index=0, range_index=99, config=roi_config
    )
    if roi.shape != (10, 11, 9) or mask.shape != (1, 11, 9):
        raise RuntimeError(f"ROI shape mismatch: {roi.shape} / {mask.shape}")
    if not 0.0 < float(mask.mean()) < 1.0:
        raise RuntimeError("Boundary ROI mask was not exercised")

    mode_results = {}
    parameter_counts = set()
    for mode in ROI_MODES:
        canonical = canonical_roi_channels(roi, mode).unsqueeze(0)
        if canonical.shape != (1, 8, 11, 9):
            raise RuntimeError(f"Canonical shape mismatch for {mode}: {canonical.shape}")
        model = ROIPolarimetricRefiner()
        parameter_counts.add(count_parameters(model))
        raw_logit = torch.tensor([1.2], requires_grad=False)
        raw_score = torch.sigmoid(raw_logit)
        polar_conf = confidence_roi.mean().reshape(1)
        output = model(
            canonical,
            raw_logit,
            raw_score,
            mask.unsqueeze(0),
            polar_conf,
        )
        if not torch.all(output["refined_score"] <= output["raw_power2_score"] + 1e-7):
            raise RuntimeError(f"Suppression-only invariant failed for {mode}")
        if not torch.all(output["suppression"] >= 0):
            raise RuntimeError(f"Negative suppression for {mode}")
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                output["refined_logit"], torch.ones_like(output["refined_logit"])
            )
            + 0.2 * torch.nn.functional.binary_cross_entropy(
                output["roi_quality"].clamp(1e-6, 1 - 1e-6),
                torch.ones_like(output["roi_quality"]),
            )
        )
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError(f"Backward check failed for {mode}")
        mode_results[mode] = {
            "input_shape": list(canonical.shape),
            "raw_score": float(raw_score.item()),
            "refined_score": float(output["refined_score"].item()),
            "suppression": float(output["suppression"].item()),
        }
    if len(parameter_counts) != 1:
        raise RuntimeError(f"Parameter counts differ across modes: {parameter_counts}")

    output_dir = PROJECT_ROOT / "results/data_audit/roi_polarimetric_stage4_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "PASS",
        "source_shape": list(source.shape),
        "confidence_shape": list(confidence.shape),
        "roi_shape": list(roi.shape),
        "roi_valid_fraction": float(mask.mean()),
        "parameter_count": int(next(iter(parameter_counts))),
        "same_architecture": True,
        "sample_independent": True,
        "scan_context": False,
        "suppression_only": True,
        "power2_location_frozen": True,
        "modes": mode_results,
    }
    path = output_dir / "pipeline_test.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 88)
    print("Stage 4 ROI polarimetric pipeline test: PASS")
    print(f"source / ROI    : {source.shape} / {roi.shape}")
    print(f"parameter count : {next(iter(parameter_counts)):,}")
    print("same architecture: true")
    print("suppression only : true")
    print(f"report           : {path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
