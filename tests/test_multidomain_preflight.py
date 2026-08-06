from __future__ import annotations

from scripts.run_multidomain_preflight_v1 import run_preflight


def test_data_free_multidomain_preflight_passes_without_metrics() -> None:
    summary = run_preflight()

    assert summary["status"] == "PASS"
    assert summary["model_training"] is False
    assert summary["performance_metrics"] is False
    assert summary["contract"]["total_features"] == 56
    assert summary["smoke"]["fusion"]["masked_polar_weight"] == 0.0
