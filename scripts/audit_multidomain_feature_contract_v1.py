#!/usr/bin/env python3
"""Audit the executable multi-domain feature contract.

This is a data-free gate.  It checks that the YAML contract agrees with the
frozen feature names, fusion dimensions, and current claim boundaries.  It
does not load IQ data, train a model, or report Pd/Pfa/AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.multidomain_radar_features import (  # noqa: E402
    FEATURE_DOMAINS,
    MULTIDOMAIN_FEATURE_NAMES,
)
from models.multidomain_feature_fusion import DEFAULT_DOMAIN_DIMENSIONS  # noqa: E402


EXPECTED_FUSION_DOMAINS = ("quality", "time", "rd", "polar", "tf")
EXPECTED_FUSION_DIMENSIONS = {
    "quality": 3,
    "time": 11,
    "rd": 22,
    "polar": 8,
    "tf": 12,
}
EXPECTED_PHYSICAL_DOMAINS = (
    "time",
    "polarimetric",
    "range_doppler",
    "time_frequency",
    "trajectory",
    "wind_dynamics",
)
EXPECTED_ANALYSIS_RULES = {
    "do_not_train_detection_claims_from_date_confounded_target_background_labels",
    "use_source_file_as_independent_group",
    "report_pooled_and_worst_background_group_behavior",
    "current_classification_data_are_UAV_only",
    "current_heldout_detection_partition_is_consumed_historical_evidence",
    "fit_scalar_normalization_on_training_acquisition_groups_only",
}
EXPECTED_POLAR_BLOCKED_CLAIMS = {"absolute_ZDR", "absolute_PhiDP"}
EXPECTED_POLAR_UNLOCK_REQUIREMENTS = {
    "simultaneous_HV_capture",
    "amplitude_calibration",
    "phase_calibration",
}
EXPECTED_TF_UNLOCK_REQUIREMENTS = {
    "PRF",
    "continuous_order",
    "sample_duration",
    "target_aligned_events",
}


def resolve_default_contract(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the contract in either repository or sanitized-package layout."""
    candidates = (
        project_root / "configs/multidomain_feature_contract_v1.yaml",
        project_root / "assets/contracts/multidomain_feature_contract_v1.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_CONTRACT = resolve_default_contract()


class ContractAuditError(ValueError):
    """Raised when the checked contract has drifted from the implementation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractAuditError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{name} must be a mapping")
    return value


def load_contract(path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractAuditError(f"contract not found: {resolved}") from exc
    except yaml.YAMLError as exc:
        raise ContractAuditError(f"invalid YAML contract: {resolved}") from exc
    _require(isinstance(document, dict), "contract root must be a mapping")
    return document


def audit_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(document.get("contract"), "contract")
    domains = _mapping(document.get("domains"), "domains")
    rules = document.get("analysis_rules")
    fusion = _mapping(document.get("fusion_interface"), "fusion_interface")

    _require(
        contract.get("name") == "multidomain_radar_feature_catalog_v1",
        "contract.name does not identify multidomain_radar_feature_catalog_v1",
    )
    _require(
        contract.get("physical_frequency_units_available_for_all_current_data") is False,
        "physical-frequency availability must remain false",
    )
    _require(
        tuple(domains) == EXPECTED_PHYSICAL_DOMAINS,
        "physical domain keys/order changed: "
        f"expected {EXPECTED_PHYSICAL_DOMAINS}, got {tuple(domains)}",
    )
    _require(
        isinstance(rules, list) and EXPECTED_ANALYSIS_RULES.issubset(rules),
        "analysis_rules are incomplete",
    )

    configured_domains = fusion.get("domains")
    _require(
        tuple(configured_domains or ()) == EXPECTED_FUSION_DOMAINS,
        "fusion domains/order changed: "
        f"expected {EXPECTED_FUSION_DOMAINS}, got {configured_domains}",
    )
    dimensions = fusion.get("current_dimensions")
    _require(
        dimensions == [EXPECTED_FUSION_DIMENSIONS[name] for name in EXPECTED_FUSION_DOMAINS],
        "fusion current_dimensions changed: "
        f"expected {[EXPECTED_FUSION_DIMENSIONS[name] for name in EXPECTED_FUSION_DOMAINS]}, "
        f"got {dimensions}",
    )
    _require(dict(DEFAULT_DOMAIN_DIMENSIONS) == EXPECTED_FUSION_DIMENSIONS, "model fusion dimensions drifted")
    implementation_dimensions = {
        name: len(names) for name, names in MULTIDOMAIN_FEATURE_NAMES.items()
    }
    _require(implementation_dimensions == EXPECTED_FUSION_DIMENSIONS, "feature name dimensions drifted")
    _require(dict(FEATURE_DOMAINS) == {
        "quality": "input and anchor quality",
        "time": "anchor-range slow-time statistics",
        "rd": "range-Doppler energy and clutter morphology",
        "polar": "relative H/V polarimetric statistics",
        "tf": "normalized-frequency time-frequency descriptors",
    }, "feature domain catalog drifted")
    _require(sum(dimensions) == 56, "frozen multidomain feature total must be 56")
    _require(
        fusion.get("missing_domain_policy") == "explicit_validity_mask_and_zero_fusion_weight",
        "missing-domain policy changed",
    )
    _require(fusion.get("training_status") == "scaffold_only", "training status must remain scaffold_only")

    polarimetric = _mapping(domains["polarimetric"], "domains.polarimetric")
    time_frequency = _mapping(domains["time_frequency"], "domains.time_frequency")
    trajectory = _mapping(domains["trajectory"], "domains.trajectory")
    wind_dynamics = _mapping(domains["wind_dynamics"], "domains.wind_dynamics")
    _require(polarimetric.get("current_status") == "relative_only", "polarimetric status must remain relative_only")
    _require(
        EXPECTED_POLAR_BLOCKED_CLAIMS.issubset(
            set(polarimetric.get("blocked_absolute_claims", []))
        ),
        "absolute polarimetric claim gate removed",
    )
    _require(
        EXPECTED_POLAR_UNLOCK_REQUIREMENTS.issubset(
            set(polarimetric.get("unlock_requires", []))
        ),
        "polarimetric unlock requirements are incomplete",
    )
    _require(time_frequency.get("current_status") == "normalized_frequency_descriptors_available", "time-frequency status changed")
    _require("micro_Doppler_Hz" in time_frequency.get("blocked_physical_claims", []), "micro-Doppler Hz claim gate removed")
    _require("rotor_rate_Hz" in time_frequency.get("blocked_physical_claims", []), "rotor-rate Hz claim gate removed")
    _require(
        EXPECTED_TF_UNLOCK_REQUIREMENTS.issubset(
            set(time_frequency.get("unlock_requires", []))
        ),
        "time-frequency unlock requirements are incomplete",
    )
    _require(trajectory.get("current_status") == "blocked", "trajectory gate must remain blocked")
    _require(wind_dynamics.get("current_status") == "blocked", "wind-dynamics gate must remain blocked")

    return {
        "status": "PASS",
        "scope": "data_free_contract_audit",
        "contract_name": contract["name"],
        "physical_domains": list(EXPECTED_PHYSICAL_DOMAINS),
        "fusion_domains": list(EXPECTED_FUSION_DOMAINS),
        "domain_dimensions": EXPECTED_FUSION_DIMENSIONS,
        "total_features": 56,
        "training_status": fusion["training_status"],
        "physical_frequency_units_available": False,
        "polarimetric_status": domains["polarimetric"]["current_status"],
        "time_frequency_status": domains["time_frequency"]["current_status"],
        "blocked_domains": ["trajectory", "wind_dynamics"],
        "interpretation": "Contract and implementation agree; this is an interface/claim gate only.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_contract(load_contract(args.contract))
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"summary: {output}")


if __name__ == "__main__":
    main()
