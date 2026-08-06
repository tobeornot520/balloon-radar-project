from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.audit_multidomain_feature_contract_v1 import (
    ContractAuditError,
    audit_contract,
    load_contract,
)


@pytest.fixture()
def contract() -> dict:
    return load_contract()


def test_frozen_multidomain_contract_passes(contract: dict) -> None:
    summary = audit_contract(contract)

    assert summary["status"] == "PASS"
    assert summary["total_features"] == 56
    assert summary["domain_dimensions"] == {
        "quality": 3,
        "time": 11,
        "rd": 22,
        "polar": 8,
        "tf": 12,
    }
    assert summary["physical_frequency_units_available"] is False
    assert summary["training_status"] == "scaffold_only"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("fusion_interface", "current_dimensions"), [3, 11, 22, 8, 13], "current_dimensions"),
        (("fusion_interface", "missing_domain_policy"), "zero_fill", "missing-domain policy"),
        (("domains", "polarimetric", "current_status"), "absolute", "polarimetric status"),
        (("domains", "time_frequency", "blocked_physical_claims"), [], "micro-Doppler Hz"),
    ],
)
def test_contract_drift_is_rejected(
    contract: dict, path: tuple[str, ...], value: object, message: str
) -> None:
    changed = deepcopy(contract)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ContractAuditError, match=message):
        audit_contract(changed)
