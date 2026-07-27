from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_bc_dpg_causal_smoke import (
    build_command,
    experiment_name,
    load_readiness_manifest,
    validate_args,
)
from training.train_target_protected_scan_calibrator import (
    scan_context_metadata,
    validate_scan_context_args,
)


def smoke_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "fold": 1,
        "history_window": 4,
        "epochs": 2,
        "debug_per_class": 12,
        "batch_size": 16,
        "seed": 42,
        "overwrite": False,
        "dry_run": False,
    }
    values.update(overrides)
    return Namespace(**values)


def context_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "scan_context_mode": "complete_scan",
        "history_window": None,
        "allow_inferred_order": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_past_only_training_requires_explicit_inferred_order_acknowledgement() -> None:
    with pytest.raises(ValueError, match="--allow-inferred-order"):
        validate_scan_context_args(context_args(scan_context_mode="past_only"))

    args = context_args(
        scan_context_mode="past_only",
        history_window=4,
        allow_inferred_order=True,
    )
    validate_scan_context_args(args)
    assert scan_context_metadata(args) == {
        "mode": "past_only",
        "history_window": 4,
        "order_columns": ["beam_layer", "azimuth_deg", "sample_id"],
        "order_verified_by_timestamp": False,
        "evidence_role": "development_only_inferred_order",
    }


def test_complete_scan_rejects_history_window() -> None:
    with pytest.raises(ValueError, match="only valid for past_only"):
        validate_scan_context_args(context_args(history_window=4))


def test_smoke_runner_is_bounded_and_builds_development_command() -> None:
    args = smoke_args()
    validate_args(args)
    command = build_command(args)
    assert experiment_name(args).endswith("seed42_smoke")
    assert "--scan-context-mode" in command
    assert "past_only" in command
    assert "--allow-inferred-order" in command
    assert "--validation-only" in command
    with pytest.raises(ValueError, match="1-4 epochs"):
        validate_args(smoke_args(epochs=5))


def test_smoke_runner_requires_closed_gate_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "audit_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "verified_within_scan_order_available": False,
                "formal_causal_training_gate_open": False,
            }
        ),
        encoding="utf-8",
    )
    payload = load_readiness_manifest(manifest)
    assert payload["formal_causal_training_gate_open"] is False
