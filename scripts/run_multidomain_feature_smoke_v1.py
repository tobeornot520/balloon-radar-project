#!/usr/bin/env python3
"""Run a bounded, data-free smoke through the multi-domain feature path.

This command checks feature ordering, relative H/V behavior, normalized
time-frequency response, and validity-masked fusion.  It never reads project
data, trains a model, or computes Pd/Pfa/AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.multidomain_radar_features import (  # noqa: E402
    MULTIDOMAIN_FEATURE_NAMES,
    MultiDomainFeatureConfig,
    extract_multidomain_features,
    split_multidomain_features,
)
from features.synthetic_radar import SyntheticIQConfig, make_synthetic_hv_iq  # noqa: E402
from models.multidomain_feature_fusion import (  # noqa: E402
    DEFAULT_DOMAIN_DIMENSIONS,
    MultiDomainFeatureFusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path for the smoke summary; stdout is always emitted",
    )
    return parser.parse_args()


def _feature_row(mode: str) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    h, v = make_synthetic_hv_iq(SyntheticIQConfig(mode=mode))
    features = extract_multidomain_features(
        h,
        v,
        MultiDomainFeatureConfig(stft_nperseg=128),
    )
    return features, split_multidomain_features(features)


def _fusion_check(
    vectors: dict[str, np.ndarray],
) -> dict[str, Any]:
    torch.manual_seed(20260806)
    model = MultiDomainFeatureFusion().eval()
    inputs = {
        name: torch.from_numpy(values).reshape(1, -1)
        for name, values in vectors.items()
    }
    with torch.no_grad():
        complete = model(inputs)
        validity = torch.ones(1, len(model.domain_names))
        polar_index = model.domain_names.index("polar")
        validity[:, polar_index] = 0.0
        masked = model(inputs, validity)
    if tuple(complete.embedding.shape) != (1, model.embedding_dim):
        raise AssertionError("complete fusion embedding shape mismatch")
    if not torch.isfinite(complete.embedding).all():
        raise AssertionError("complete fusion embedding is non-finite")
    if not torch.all(masked.domain_weights[:, polar_index] == 0):
        raise AssertionError("invalid polar domain received a non-zero weight")
    return {
        "embedding_shape": list(complete.embedding.shape),
        "complete_weight_sum": float(complete.domain_weights.sum()),
        "masked_polar_weight": float(masked.domain_weights[:, polar_index].item()),
    }


def run_smoke() -> dict[str, Any]:
    expected_dimensions = {
        name: len(names) for name, names in MULTIDOMAIN_FEATURE_NAMES.items()
    }
    if expected_dimensions != DEFAULT_DOMAIN_DIMENSIONS:
        raise AssertionError(
            "feature-name contract and fusion dimensions disagree: "
            f"{expected_dimensions} != {DEFAULT_DOMAIN_DIMENSIONS}"
        )

    tone, tone_vectors = _feature_row("tone")
    sweep, sweep_vectors = _feature_row("sweep")
    for mode, vectors in (("tone", tone_vectors), ("sweep", sweep_vectors)):
        for domain, expected in DEFAULT_DOMAIN_DIMENSIONS.items():
            if vectors[domain].shape != (expected,):
                raise AssertionError(f"{mode}/{domain} vector shape mismatch")
            if not np.isfinite(vectors[domain]).all():
                raise AssertionError(f"{mode}/{domain} vector is non-finite")

    if tone["time_hv_coherence"] < 0.999:
        raise AssertionError("synthetic coherent H/V tone was not recovered")
    expected_ratio = 20.0 * np.log10(1.0 / 0.7)
    if not np.isclose(tone["time_hv_power_ratio_db"], expected_ratio, atol=0.02):
        raise AssertionError("relative H/V power ratio changed unexpectedly")
    if sweep["tf_dominant_span_normalized"] <= tone["tf_dominant_span_normalized"] + 0.1:
        raise AssertionError("normalized STFT did not expose the synthetic sweep")
    if tone["tf_long_window_available"] != 1 or sweep["tf_long_window_available"] != 1:
        raise AssertionError("long-window readiness gate changed unexpectedly")

    return {
        "status": "PASS",
        "scope": "data_free_interface_smoke",
        "model_training": False,
        "performance_metrics": False,
        "domain_dimensions": expected_dimensions,
        "tone": {
            "hv_coherence": float(tone["time_hv_coherence"]),
            "hv_power_ratio_db": float(tone["time_hv_power_ratio_db"]),
            "tf_dominant_span_normalized": float(
                tone["tf_dominant_span_normalized"]
            ),
        },
        "sweep": {
            "tf_dominant_span_normalized": float(
                sweep["tf_dominant_span_normalized"]
            ),
            "tf_frame_count": int(sweep["tf_frame_count"]),
        },
        "fusion": _fusion_check(sweep_vectors),
        "interpretation": (
            "Normalized-frequency interface behavior only; no PRF, physical "
            "micro-Doppler, labels, or generalization claim."
        ),
    }


def main() -> None:
    args = parse_args()
    summary = run_smoke()
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"summary: {output}")


if __name__ == "__main__":
    main()

