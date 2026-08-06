#!/usr/bin/env python3
"""Run the data-free multidomain interface preflight.

The preflight combines the frozen YAML contract audit with the deterministic
synthetic H/V feature-and-fusion smoke.  It never reads project data, trains a
model, or reports Pd/Pfa/AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_multidomain_feature_contract_v1 import (
    DEFAULT_CONTRACT,
    audit_contract,
    load_contract,
)
from scripts.run_multidomain_feature_smoke_v1 import run_smoke
from features.multidomain_radar_features import extract_multidomain_features  # noqa: E402
from features.synthetic_radar import SyntheticIQConfig, make_synthetic_hv_iq  # noqa: E402


def _invariance_checks() -> dict[str, float]:
    """Check transformations that should preserve relative H/V descriptors."""
    h, v = make_synthetic_hv_iq(SyntheticIQConfig(mode="tone"))
    base = extract_multidomain_features(h, v)

    common_phase = np.exp(1j * 0.73)
    rotated = extract_multidomain_features(h * common_phase, v * common_phase)
    common_scale = extract_multidomain_features(h * 2.5, v * 2.5)
    swapped = extract_multidomain_features(v, h)

    invariant_names = (
        "time_hv_coherence",
        "time_hv_phase_resultant",
        "polar_roi_zdr_median_db",
        "polar_roi_rho_mean",
        "polar_roi_phase_resultant",
        "polar_roi_stokes_s1_mean",
        "polar_roi_stokes_s2_mean",
        "polar_roi_stokes_s3_mean",
    )
    for name in invariant_names:
        if not np.isclose(base[name], rotated[name], atol=1e-5, rtol=1e-5):
            raise AssertionError(f"common phase rotation changed {name}")
        if not np.isclose(base[name], common_scale[name], atol=1e-5, rtol=1e-5):
            raise AssertionError(f"common amplitude scaling changed {name}")

    if not np.isclose(
        base["time_hv_power_ratio_db"],
        -swapped["time_hv_power_ratio_db"],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise AssertionError("H/V swap did not reverse relative power ratio")
    if not np.isclose(
        base["time_hv_coherence"], swapped["time_hv_coherence"], atol=1e-5
    ):
        raise AssertionError("H/V swap changed coherence")
    return {
        "common_phase_coherence": float(rotated["time_hv_coherence"]),
        "common_scale_zdr_like_db": float(common_scale["polar_roi_zdr_median_db"]),
        "hv_swap_power_ratio_db": float(swapped["time_hv_power_ratio_db"]),
    }


def run_preflight(contract_path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_summary = audit_contract(load_contract(contract_path))
    smoke_summary = run_smoke()
    invariance_summary = _invariance_checks()
    if contract_summary["status"] != "PASS":
        raise AssertionError("multidomain contract audit did not pass")
    if smoke_summary["status"] != "PASS":
        raise AssertionError("multidomain synthetic smoke did not pass")
    return {
        "status": "PASS",
        "scope": "data_free_multidomain_preflight",
        "model_training": False,
        "performance_metrics": False,
        "contract": contract_summary,
        "smoke": smoke_summary,
        "invariance": invariance_summary,
        "interpretation": (
            "Interface and claim-boundary preflight only; no real-data, physical-Hz, "
            "Pd/Pfa/AUC, or generalization claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(args.contract)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"summary: {output}")


if __name__ == "__main__":
    main()
