#!/usr/bin/env python3
"""Validate weather/site observations against the V1 environment contract."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "configs/environment_context_observation_schema_v1.json"
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs/environment_context_observation_template_v1.csv"
REQUIRED_SECTIONS = (
    "weather_fields",
    "derived_weather_fields",
    "scene_static_fields",
    "scene_empirical_fields",
)
WEATHER_CORE = (
    "temperature_c",
    "relative_humidity_pct",
    "pressure_hpa",
    "wind_speed_mean_mps",
    "wind_speed_gust_mps",
    "wind_direction_deg",
    "precip_rate_mm_h",
    "visibility_m",
)
EMPIRICAL_FIELDS = (
    "background_power_q50",
    "background_power_q90",
    "background_power_q99",
    "background_occupancy",
    "background_peak_stability",
    "background_zero_doppler_fraction",
    "background_risk_at_candidate",
    "background_update_age_s",
)
OPTIONAL_PROVENANCE = (
    "background_interval_start_utc",
    "background_interval_end_utc",
    "background_map_version",
    "background_fit_partition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_utc(value: Any) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.astimezone(timezone.utc)


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if schema.get("schema_version") != 1:
        raise ValueError("only environment schema_version=1 is supported")
    for section in REQUIRED_SECTIONS:
        if not isinstance(schema.get(section), dict):
            raise ValueError(f"missing schema section: {section}")
    required = schema.get("required_identifiers")
    if not isinstance(required, list) or len(required) != len(set(required)):
        raise ValueError("required_identifiers must be a unique list")
    return schema


def field_specs(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for section in REQUIRED_SECTIONS:
        output.update(schema[section])
    return output


def add_issue(issues: list[dict[str, Any]], code: str, message: str, rows: Any = None) -> None:
    examples = [] if rows is None else [int(value) + 2 for value in list(rows)[:10]]
    issues.append({"severity": "ERROR", "code": code, "message": message, "row_examples": examples})


def validate_frame(frame: pd.DataFrame, schema: dict[str, Any], *, allow_empty: bool = False) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    required = list(schema["required_identifiers"])
    specs = field_specs(schema)
    expected = required + list(specs)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        add_issue(issues, "MISSING_COLUMNS", f"missing required columns: {missing}")
        return {"status": "FAIL", "row_count": int(len(frame)), "issues": issues}
    if frame.empty:
        if not allow_empty:
            add_issue(issues, "EMPTY_OBSERVATION_TABLE", "observation table is empty")
        status = "EMPTY_TEMPLATE" if allow_empty else "FAIL"
        return {"status": status, "row_count": 0, "issues": issues}

    work = frame.fillna("").copy()
    for column in required:
        empty = ~nonempty(work[column])
        if empty.any():
            add_issue(issues, "MISSING_REQUIRED_VALUE", f"{column} contains empty values", work.index[empty])

    duplicate = work["observation_id"].duplicated(keep=False)
    if duplicate.any():
        add_issue(issues, "DUPLICATE_OBSERVATION_ID", "observation_id must be unique", work.index[duplicate])

    quality = set(schema["weather_fields"]["weather_quality"]["allowed"])
    invalid_quality = ~work["weather_quality"].astype(str).isin(quality)
    if invalid_quality.any():
        add_issue(issues, "INVALID_WEATHER_QUALITY", f"weather_quality must be one of {sorted(quality)}", work.index[invalid_quality])
    observation_quality = {"valid", "estimated", "missing", "fault"}
    invalid_observation_quality = ~work["observation_quality"].astype(str).isin(observation_quality)
    if invalid_observation_quality.any():
        add_issue(issues, "INVALID_OBSERVATION_QUALITY", "observation_quality has an unsupported value", work.index[invalid_observation_quality])

    parsed_time = work["observation_timestamp_utc"].map(parse_utc)
    invalid_time = parsed_time.isna()
    if invalid_time.any():
        add_issue(issues, "INVALID_UTC_TIMESTAMP", "observation_timestamp_utc must be UTC", work.index[invalid_time])
    work["_timestamp"] = parsed_time

    for column, spec in specs.items():
        values = work[column]
        present = nonempty(values)
        if not present.any():
            continue
        value_type = spec.get("type")
        if value_type == "number" or value_type == "integer":
            numeric = pd.to_numeric(values.loc[present], errors="coerce")
            invalid = numeric.isna() | ~np.isfinite(numeric)
            if value_type == "integer":
                invalid |= numeric.mod(1).ne(0)
            if invalid.any():
                add_issue(issues, "INVALID_NUMERIC_VALUE", f"{column} contains invalid numeric values", invalid.index[invalid])
                continue
            if "minimum" in spec:
                bad = numeric.lt(float(spec["minimum"]))
                if bad.any():
                    add_issue(issues, "VALUE_BELOW_MINIMUM", f"{column} is below its minimum", bad.index[bad])
            if "range" in spec:
                lower, upper = spec["range"]
                bad = numeric.lt(float(lower)) | numeric.gt(float(upper))
                if bad.any():
                    add_issue(issues, "VALUE_OUT_OF_RANGE", f"{column} is outside its allowed range", bad.index[bad])
            if "maximum_exclusive" in spec:
                bad = numeric.ge(float(spec["maximum_exclusive"]))
                if bad.any():
                    add_issue(issues, "VALUE_AT_EXCLUSIVE_MAXIMUM", f"{column} reaches its exclusive maximum", bad.index[bad])
            if "range_exclusive" in spec:
                lower, upper = spec["range_exclusive"]
                bad = numeric.le(float(lower)) | numeric.ge(float(upper))
                if bad.any():
                    add_issue(issues, "VALUE_OUT_OF_EXCLUSIVE_RANGE", f"{column} is outside its exclusive range", bad.index[bad])
        elif value_type == "enum":
            invalid = ~values.loc[present].astype(str).isin(spec.get("allowed", []))
            if invalid.any():
                add_issue(issues, "INVALID_ENUM_VALUE", f"{column} contains an unsupported enum", invalid.index[invalid])

    for session_id, group in work.groupby("session_id", sort=False):
        ordered = group.sort_values("_timestamp", na_position="last")
        timestamps = ordered["_timestamp"].dropna().map(datetime.timestamp).to_numpy(dtype=float)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            add_issue(issues, "NONMONOTONIC_SESSION_TIME", f"timestamps are not monotonic in session {session_id}", ordered.index)

    weather_values_present = work[list(WEATHER_CORE)].apply(nonempty).any(axis=1)
    valid_quality = work["weather_quality"].isin({"valid", "estimated"})
    missing_weather = valid_quality & ~weather_values_present
    if missing_weather.any():
        add_issue(issues, "VALID_WEATHER_WITHOUT_MEASUREMENT", "valid/estimated weather rows need at least one raw weather value", work.index[missing_weather])

    missing_indicator = pd.to_numeric(work["weather_missing_indicator"], errors="coerce")
    expected_missing = (~valid_quality).astype(int)
    mismatch = missing_indicator.notna() & missing_indicator.ne(expected_missing)
    if mismatch.any():
        add_issue(issues, "MISSING_INDICATOR_MISMATCH", "weather_missing_indicator disagrees with weather_quality", work.index[mismatch])

    empirical_present = work[list(EMPIRICAL_FIELDS)].apply(nonempty).any(axis=1)
    provenance_present = work[list(OPTIONAL_PROVENANCE)].apply(nonempty).any(axis=1)
    missing_provenance = empirical_present & ~provenance_present
    if missing_provenance.any():
        add_issue(issues, "EMPIRICAL_BACKGROUND_WITHOUT_PROVENANCE", "empirical scene fields require background interval/version/partition provenance", work.index[missing_provenance])

    status = "PASS" if not issues else "FAIL"
    return {
        "status": status,
        "row_count": int(len(work)),
        "session_count": int(work["session_id"].nunique()),
        "site_count": int(work["site_id"].nunique()),
        "weather_quality_counts": work["weather_quality"].value_counts(dropna=False).to_dict(),
        "observation_quality_counts": work["observation_quality"].value_counts(dropna=False).to_dict(),
        "issues": issues,
    }


def write_report(output_dir: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    output_dir = resolve_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    issue_rows = report.get("issues", [])
    pd.DataFrame(issue_rows, columns=["severity", "code", "message", "row_examples"]).to_csv(
        output_dir / "issues.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "README.md").write_text(
        "# Environment context validation\n\n"
        f"Status: `{report['status']}`\n\n"
        "This report checks schema, timestamps, quality flags and background provenance. "
        "It does not prove radar calibration, H/V coherence, causality or model performance.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    schema = load_schema(args.schema)
    observations = resolve_path(args.observations)
    frame = pd.read_csv(observations, encoding="utf-8-sig")
    report = validate_frame(frame, schema, allow_empty=args.allow_empty)
    write_report(args.output_dir, report, overwrite=args.overwrite)
    print(json.dumps({key: report[key] for key in report if key != "issues"}, ensure_ascii=False))
    return 0 if report["status"] in {"PASS", "EMPTY_TEMPLATE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
