from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.validate_environment_context_v1 import (
    DEFAULT_SCHEMA,
    DEFAULT_TEMPLATE,
    validate_frame,
    load_schema,
)


def empty_row() -> dict[str, str]:
    columns = pd.read_csv(DEFAULT_TEMPLATE, nrows=0).columns
    return {column: "" for column in columns}


def valid_row(index: int = 0) -> dict[str, str]:
    row = empty_row()
    row.update(
        {
            "experiment_id": "E1",
            "session_id": "session-01",
            "scan_id": f"scan-{index}",
            "event_id": f"event-{index}",
            "observation_id": f"obs-{index}",
            "site_id": "site-01",
            "radar_config_id": "radar-v1",
            "calibration_id": "cal-v1",
            "observation_timestamp_utc": f"2026-09-26T00:00:{index:02d}Z",
            "observation_quality": "valid",
            "weather_sensor_id": "weather-01",
            "weather_sensor_height_m": "2",
            "temperature_c": "24",
            "relative_humidity_pct": "55",
            "pressure_hpa": "1010",
            "wind_speed_mean_mps": "2",
            "wind_speed_gust_mps": "3",
            "wind_direction_deg": "90",
            "precip_rate_mm_h": "0",
            "visibility_m": "10000",
            "weather_quality": "valid",
            "wind_u_mps": "-2",
            "wind_v_mps": "0",
            "wind_gust_ratio": "1.5",
            "wind_relative_to_radar_deg": "90",
            "weather_observation_age_s": "0",
            "weather_missing_indicator": "0",
            "site_class": "open",
            "radar_height_m": "2",
            "radar_azimuth_deg": "0",
        }
    )
    return row


def test_template_matches_schema_and_empty_template_is_allowed() -> None:
    schema = load_schema(DEFAULT_SCHEMA)
    frame = pd.read_csv(DEFAULT_TEMPLATE)
    report = validate_frame(frame, schema, allow_empty=True)
    expected = list(schema["required_identifiers"])
    for section in ("weather_fields", "derived_weather_fields", "scene_static_fields", "scene_empirical_fields"):
        expected.extend(schema[section])
    assert list(frame.columns) == expected + [
        "background_interval_start_utc",
        "background_interval_end_utc",
        "background_fit_partition",
        "source_record_id",
        "notes",
    ]
    assert report["status"] == "EMPTY_TEMPLATE"


def test_valid_context_row_passes() -> None:
    schema = load_schema(DEFAULT_SCHEMA)
    report = validate_frame(pd.DataFrame([valid_row()]), schema)
    assert report["status"] == "PASS"
    assert report["row_count"] == 1


def test_invalid_timestamp_and_range_are_reported() -> None:
    schema = load_schema(DEFAULT_SCHEMA)
    row = valid_row()
    row["observation_timestamp_utc"] = "2026-09-26 00:00:00"
    row["relative_humidity_pct"] = "120"
    report = validate_frame(pd.DataFrame([row]), schema)
    codes = {issue["code"] for issue in report["issues"]}
    assert report["status"] == "FAIL"
    assert "INVALID_UTC_TIMESTAMP" in codes
    assert "VALUE_OUT_OF_RANGE" in codes


def test_empirical_scene_values_require_provenance() -> None:
    schema = load_schema(DEFAULT_SCHEMA)
    row = valid_row()
    row["background_power_q90"] = "4.2"
    report = validate_frame(pd.DataFrame([row]), schema)
    assert report["status"] == "FAIL"
    assert "EMPIRICAL_BACKGROUND_WITHOUT_PROVENANCE" in {issue["code"] for issue in report["issues"]}


def test_missing_weather_requires_missing_quality_and_indicator() -> None:
    schema = load_schema(DEFAULT_SCHEMA)
    row = valid_row()
    for column in (
        "temperature_c",
        "relative_humidity_pct",
        "pressure_hpa",
        "wind_speed_mean_mps",
        "wind_speed_gust_mps",
        "wind_direction_deg",
        "precip_rate_mm_h",
        "visibility_m",
    ):
        row[column] = ""
    row["weather_quality"] = "missing"
    row["weather_missing_indicator"] = "1"
    report = validate_frame(pd.DataFrame([row]), schema)
    assert report["status"] == "PASS"
