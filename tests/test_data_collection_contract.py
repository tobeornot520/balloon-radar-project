from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_data_collection_manifest import (
    DEFAULT_CONTRACT,
    build_column_coverage,
    load_contract,
    validate_collection_frame,
    write_precheck_output,
)


def valid_manifest() -> pd.DataFrame:
    shared = {
        "payload_class": "",
        "motion_state": "",
        "order_source": "hardware_sequence",
        "order_verified": True,
        "clock_id": "clock-a",
        "clock_reset_counter": 0,
        "timestamp_resolution_ns": 1000,
        "dropped_frames_before": 0,
        "sample_duration_s": 0.1,
        "collection_date": "2026-07-28",
        "site_id": "site-a",
        "beam_layer": 1,
        "azimuth_deg": 10.0,
        "radar_config_id": "radar-v1",
        "calibration_id": "cal-v1",
        "h_channel_valid": True,
        "v_channel_valid": True,
        "weather_id": "weather-a",
    }
    rows = []
    specifications = [
        ("dev-bg", 0, "background", "development", "00:00:01Z"),
        ("dev-target", 1, "uav", "development", "00:00:02Z"),
        ("lock-bg", 0, "background", "locked_test", "00:00:03Z"),
        ("lock-target", 1, "uav", "locked_test", "00:00:04Z"),
    ]
    for number, (sample_id, target_present, target_class, partition, time_text) in enumerate(
        specifications
    ):
        target = target_present == 1
        rows.append(
            {
                **shared,
                "sample_id": sample_id,
                "iq_path": f"iq/{sample_id}.mat",
                "label_path": f"labels/{sample_id}.txt" if target else "",
                "target_present": target_present,
                "target_class": target_class,
                "background_type": "clear_sky" if not target else "",
                "acquisition_timestamp_utc": f"2026-07-28T{time_text}",
                "hardware_sequence": 100 + number,
                "scan_id": f"scan-{sample_id}",
                "scan_sequence": 0,
                "session_id": f"session-{sample_id}",
                "observation_id": f"observation-{sample_id}",
                "event_id": f"event-{sample_id}",
                "event_start_utc": "2026-07-28T00:00:00Z",
                "event_end_utc": "2026-07-28T00:01:00Z",
                "flight_id": f"flight-{sample_id}" if target else "",
                "platform_id": "uav-a" if target else "",
                "distance_m": 2100.0 if target else "",
                "velocity_mps": 3.0 if target else "",
                "snr_db": 8.0 if target else "",
                "evaluation_partition": partition,
                "outer_group_id": f"outer-{sample_id}",
            }
        )
    contract = load_contract(DEFAULT_CONTRACT)
    columns = [item["name"] for item in contract["columns"]]
    return pd.DataFrame(rows).loc[:, columns]


@pytest.mark.parametrize("profile", ["capture", "causal", "locked_evaluation"])
def test_complete_manifest_passes_each_profile(profile: str) -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    report, coverage = validate_collection_frame(valid_manifest(), contract, profile)
    assert report["status"] == "PASS"
    assert report["issue_count"] == 0
    assert coverage["present"].all()
    if profile == "locked_evaluation":
        assert report["formal_causal_training_gate_open"] is True
        assert report["locked_evaluation_gate_open"] is True


def test_missing_contract_column_fails_schema_gate() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest().drop(columns=["hardware_sequence"])
    report, _ = validate_collection_frame(frame, contract, "causal")
    assert report["status"] == "FAIL"
    assert report["gates"]["schema"] == "FAIL"
    assert report["gates"]["causal_order"] == "BLOCKED"
    assert "hardware_sequence" in report["missing_columns"]


def test_unverified_or_nonmonotonic_order_fails_causal_gate() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest()
    frame.loc[0, "order_verified"] = False
    frame.loc[1, "scan_id"] = frame.loc[0, "scan_id"]
    frame.loc[1, "session_id"] = frame.loc[0, "session_id"]
    frame.loc[1, "outer_group_id"] = frame.loc[0, "outer_group_id"]
    frame.loc[1, "hardware_sequence"] = frame.loc[0, "hardware_sequence"]
    report, _ = validate_collection_frame(frame, contract, "causal")
    codes = {item["code"] for item in report["issues"]}
    assert report["gates"]["causal_order"] == "FAIL"
    assert "ORDER_NOT_VERIFIED" in codes
    assert "NONMONOTONIC_HARDWARE_SEQUENCE" in codes


def test_capture_can_record_unverified_order_but_causal_rejects_it() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest()
    frame["order_source"] = "unknown"
    frame["order_verified"] = False
    capture, _ = validate_collection_frame(frame, contract, "capture")
    causal, _ = validate_collection_frame(frame, contract, "causal")
    assert capture["status"] == "PASS"
    assert causal["status"] == "FAIL"
    assert causal["gates"]["causal_order"] == "FAIL"


def test_condition_confounding_and_partition_leakage_fail_locked_gate() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest()
    frame.loc[2, "collection_date"] = "2026-07-29"
    frame.loc[2, "acquisition_timestamp_utc"] = "2026-07-29T00:00:03Z"
    frame.loc[2, "event_start_utc"] = "2026-07-29T00:00:00Z"
    frame.loc[2, "event_end_utc"] = "2026-07-29T00:01:00Z"
    frame.loc[3, "outer_group_id"] = frame.loc[1, "outer_group_id"]
    report, _ = validate_collection_frame(frame, contract, "locked_evaluation")
    codes = {item["code"] for item in report["issues"]}
    assert report["status"] == "FAIL"
    assert "CONDITION_CLASS_CONFOUNDING" in codes
    assert "PARTITION_GROUP_LEAKAGE" in codes


def test_loaded_balloon_requires_payload_and_motion_labels() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest()
    frame.loc[1, "target_class"] = "balloon_loaded"
    report, _ = validate_collection_frame(frame, contract, "capture")
    codes = {item["code"] for item in report["issues"]}
    assert "MISSING_BALLOON_PAYLOAD_CLASS" in codes
    assert "MISSING_BALLOON_MOTION_STATE" in codes


def test_absolute_path_and_non_utc_time_fail_schema() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest()
    frame.loc[0, "iq_path"] = "/private/data/sample.mat"
    frame.loc[0, "acquisition_timestamp_utc"] = "2026-07-28T08:00:01+08:00"
    report, _ = validate_collection_frame(frame, contract, "capture")
    invalid_columns = [
        item for item in report["issues"] if item["code"] == "INVALID_COLUMN_VALUE"
    ]
    assert report["gates"]["schema"] == "FAIL"
    assert len(invalid_columns) == 2


def test_template_header_matches_contract() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    template = pd.read_csv(
        DEFAULT_CONTRACT.with_name("data_collection_manifest_template_v1.csv")
    )
    assert template.empty
    assert template.columns.tolist() == [
        item["name"] for item in contract["columns"]
    ]


def test_precheck_output_omits_absolute_input_directory(tmp_path: Path) -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    frame = valid_manifest()
    manifest = tmp_path / "private_collection.csv"
    frame.to_csv(manifest, index=False)
    report, coverage = validate_collection_frame(
        frame,
        contract,
        "locked_evaluation",
    )
    output_dir = tmp_path / "report"
    write_precheck_output(
        output_dir,
        report,
        coverage,
        manifest_path=manifest,
        contract_path=DEFAULT_CONTRACT,
        overwrite=False,
    )
    payload = json.loads((output_dir / "preflight.json").read_text())
    assert payload["input"]["name"] == manifest.name
    assert str(tmp_path) not in (output_dir / "preflight.json").read_text()
    assert (output_dir / "PRECHECK_REPORT.md").is_file()
    assert (output_dir / "SHA256SUMS.txt").is_file()


def test_column_coverage_marks_legacy_fields_missing() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    legacy = pd.DataFrame({"sample_id": ["one"], "target_present": [1]})
    coverage = build_column_coverage(legacy, contract).set_index("column")
    assert bool(coverage.loc["sample_id", "present"])
    assert not bool(coverage.loc["hardware_sequence", "present"])
