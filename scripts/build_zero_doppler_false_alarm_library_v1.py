#!/usr/bin/env python3
"""Build a local case library and sanitized aggregate false-alarm evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_zero_doppler_human_review_v1 import validate_reviews


DEFAULT_CONFIG = PROJECT_ROOT / "configs/zero_doppler_false_alarm_library_v1.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/data_audit/zero_doppler_false_alarm_library_v1"
)
PREDICTION_COLUMNS = {
    "sample_id",
    "source_file",
    "target_present",
    "score",
    "raw_score",
    "pred_range_index",
    "pred_velocity_index",
    "detected",
    "false_alarm",
}
REVIEW_COLUMNS = [
    "review_priority",
    "review_status",
    "visible_pattern",
    "physical_class",
    "evidence_source",
    "review_note",
]
LOCAL_ONLY_COLUMNS = {"sample_id", "source_file", "review_note"}
TRANSITION_ORDER = [
    "removed_by_residual",
    "added_by_residual",
    "retained_false_alarm",
    "stable_non_false_alarm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen zero-Doppler false-alarm case library."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "library_id",
        "folds",
        "fixed_prediction_template",
        "residual_prediction_template",
        "review_queue",
        "reviewed_queue",
        "input_sha256",
        "expected_counts",
        "scan_alias",
        "claim_boundaries",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    if sorted(config["folds"]) != list(range(1, 7)):
        raise ValueError("V1 requires exactly folds 1 through 6")
    required_claims = {
        "model_training_performed",
        "threshold_retuning_performed",
        "blind_test_claim_allowed",
        "physical_taxonomy_established",
        "calibrated_polarimetry_claim_allowed",
    }
    if set(config["claim_boundaries"]) != required_claims:
        raise ValueError("claim_boundaries do not match the frozen V1 contract")
    if any(config["claim_boundaries"].values()):
        raise ValueError("all V1 claim-boundary flags must remain false")
    return config


def verify_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def bool_series(series: pd.Series, role: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"true", "false", "1", "0"})
    if invalid:
        raise ValueError(f"{role} contains invalid booleans: {invalid}")
    return normalized.isin({"true", "1"})


def load_predictions(path: Path, fold: int, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{variant} fold {fold} missing columns: {sorted(missing)}")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{variant} fold {fold} must have unique, nonempty sample IDs")
    normalized = frame.copy()
    normalized["sample_id"] = normalized["sample_id"].astype(str)
    normalized["source_file"] = normalized["source_file"].astype(str)
    normalized["target_present"] = pd.to_numeric(
        normalized["target_present"], errors="raise"
    ).astype(int)
    if not set(normalized["target_present"]).issubset({0, 1}):
        raise ValueError(f"{variant} fold {fold} has invalid target_present values")
    for column in ("detected", "false_alarm"):
        normalized[column] = bool_series(normalized[column], f"{variant}.{column}")
    expected_false_alarm = normalized["detected"] & normalized["target_present"].eq(0)
    if not normalized["false_alarm"].equals(expected_false_alarm):
        raise ValueError(f"{variant} fold {fold} has inconsistent false_alarm values")
    return normalized


def pair_background_predictions(
    fixed: pd.DataFrame, residual: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    identity = ["sample_id", "source_file", "target_present"]
    fixed_identity = fixed[identity].sort_values("sample_id").reset_index(drop=True)
    residual_identity = residual[identity].sort_values("sample_id").reset_index(drop=True)
    if not fixed_identity.equals(residual_identity):
        raise ValueError(f"fold {fold} fixed/residual sample identities differ")

    total = len(fixed)
    target_count = int(fixed["target_present"].eq(1).sum())
    fixed_bg = fixed[fixed["target_present"].eq(0)].copy()
    residual_bg = residual[residual["target_present"].eq(0)].copy()
    keep = [
        "sample_id",
        "source_file",
        "score",
        "raw_score",
        "pred_range_index",
        "pred_velocity_index",
        "false_alarm",
    ]
    paired = fixed_bg[keep].merge(
        residual_bg[keep],
        on=["sample_id", "source_file"],
        how="inner",
        validate="one_to_one",
        suffixes=("_fixed", "_residual"),
    )
    if len(paired) != len(fixed_bg):
        raise ValueError(f"fold {fold} background pairing lost rows")
    paired.insert(0, "fold", fold)
    paired["score_delta_residual_minus_fixed"] = (
        paired["score_residual"] - paired["score_fixed"]
    )
    fixed_fa = paired["false_alarm_fixed"].astype(bool)
    residual_fa = paired["false_alarm_residual"].astype(bool)
    paired["transition"] = "stable_non_false_alarm"
    paired.loc[fixed_fa & residual_fa, "transition"] = "retained_false_alarm"
    paired.loc[fixed_fa & ~residual_fa, "transition"] = "removed_by_residual"
    paired.loc[~fixed_fa & residual_fa, "transition"] = "added_by_residual"
    paired["reviewed"] = False
    for column in REVIEW_COLUMNS:
        paired[column] = ""
    counts = {
        "test_samples": total,
        "target_samples": target_count,
        "background_samples": len(paired),
    }
    return paired, counts


def validate_queue(case_library: pd.DataFrame, queue: pd.DataFrame) -> None:
    required = {
        "fold",
        "sample_id",
        "source_file",
        "fixed_notch_false_alarm",
        "residual_false_alarm",
        "residual_removed",
        "residual_added",
        "review_priority",
    }
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"review queue missing columns: {sorted(missing)}")
    queue_keys = set(zip(queue["fold"].astype(int), queue["sample_id"].astype(str)))
    fixed_fa = case_library[case_library["false_alarm_fixed"]]
    expected_keys = set(zip(fixed_fa["fold"], fixed_fa["sample_id"]))
    if queue_keys != expected_keys:
        raise ValueError("review queue is not exactly the fixed-notch false-alarm set")
    queue_flags = queue.copy()
    for column in (
        "fixed_notch_false_alarm",
        "residual_false_alarm",
        "residual_removed",
        "residual_added",
    ):
        queue_flags[column] = bool_series(queue_flags[column], f"queue.{column}")
    joined = fixed_fa.merge(
        queue_flags,
        on=["fold", "sample_id", "source_file"],
        how="inner",
        validate="one_to_one",
    )
    checks = {
        "fixed_notch_false_alarm": joined["false_alarm_fixed"],
        "residual_false_alarm": joined["false_alarm_residual"],
        "residual_removed": joined["transition"].eq("removed_by_residual"),
        "residual_added": joined["transition"].eq("added_by_residual"),
    }
    for queue_column, expected in checks.items():
        if not joined[queue_column].reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        ):
            raise ValueError(f"review queue has inconsistent {queue_column} values")


def apply_reviews(case_library: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    normalized = validate_reviews(reviewed)
    if not normalized["review_status"].eq("reviewed").all():
        raise ValueError("reviewed_queue must contain only completed reviewed rows")
    keys = ["fold", "sample_id"]
    lookup = case_library[keys + ["transition"]]
    checked = normalized.merge(lookup, on=keys, how="left", validate="one_to_one")
    if checked["transition"].isna().any():
        raise ValueError("reviewed_queue contains samples outside the case library")
    if not checked["transition"].eq("removed_by_residual").all():
        raise ValueError("V1 reviewed rows must all be removed_by_residual cases")
    if normalized["physical_class"].ne("unknown").any():
        raise ValueError("V1 has no independently established physical labels")

    output = case_library.copy().set_index(keys)
    indexed_reviews = normalized.set_index(keys)
    for column in REVIEW_COLUMNS:
        output.loc[indexed_reviews.index, column] = indexed_reviews[column]
    output.loc[indexed_reviews.index, "reviewed"] = True
    return output.reset_index()


def scan_alias(source_file: str, config: dict[str, Any]) -> str:
    alias_config = config["scan_alias"]
    if alias_config["algorithm"] != "sha256_prefix":
        raise ValueError("unsupported scan alias algorithm")
    length = int(alias_config["hex_characters"])
    if not 8 <= length <= 32:
        raise ValueError("scan alias prefix must contain 8 to 32 hex characters")
    token = f"{alias_config['namespace']}:{source_file}".encode("utf-8")
    return "scan_" + hashlib.sha256(token).hexdigest()[:length]


def build_summaries(
    cases: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_summary = (
        cases.groupby("fold", sort=True)
        .agg(
            background_samples=("sample_id", "size"),
            fixed_false_alarms=("false_alarm_fixed", "sum"),
            residual_false_alarms=("false_alarm_residual", "sum"),
            reviewed_cases=("reviewed", "sum"),
        )
        .reset_index()
    )
    transition_counts = (
        cases.groupby(["fold", "transition"]).size().unstack(fill_value=0)
    )
    for transition in TRANSITION_ORDER:
        fold_summary[transition] = fold_summary["fold"].map(
            transition_counts.get(transition, pd.Series(dtype=int))
        ).fillna(0).astype(int)
    fold_summary["fixed_pfa"] = (
        fold_summary["fixed_false_alarms"] / fold_summary["background_samples"]
    )
    fold_summary["residual_pfa"] = (
        fold_summary["residual_false_alarms"] / fold_summary["background_samples"]
    )

    scan_cases = cases.copy()
    scan_cases["scan_alias"] = scan_cases["source_file"].map(
        lambda value: scan_alias(str(value), config)
    )
    scan_summary = (
        scan_cases.groupby(["scan_alias", "fold"], sort=True)
        .agg(
            background_samples=("sample_id", "size"),
            fixed_false_alarms=("false_alarm_fixed", "sum"),
            residual_false_alarms=("false_alarm_residual", "sum"),
            reviewed_cases=("reviewed", "sum"),
        )
        .reset_index()
    )
    scan_transitions = (
        scan_cases.groupby(["scan_alias", "fold", "transition"])
        .size()
        .unstack(fill_value=0)
    )
    for transition in TRANSITION_ORDER:
        values = scan_transitions.get(transition, pd.Series(dtype=int))
        scan_summary[transition] = [
            int(values.get((row.scan_alias, row.fold), 0))
            for row in scan_summary.itertuples(index=False)
        ]
    scan_summary["fixed_pfa"] = (
        scan_summary["fixed_false_alarms"] / scan_summary["background_samples"]
    )
    scan_summary["residual_pfa"] = (
        scan_summary["residual_false_alarms"] / scan_summary["background_samples"]
    )

    reviewed = cases[cases["reviewed"]].copy()
    review_pattern_summary = (
        reviewed.groupby(
            [
                "transition",
                "review_status",
                "visible_pattern",
                "physical_class",
                "evidence_source",
            ],
            dropna=False,
        )
        .size()
        .rename("case_count")
        .reset_index()
        .sort_values(["transition", "visible_pattern"])
        .reset_index(drop=True)
    )
    return fold_summary, scan_summary, review_pattern_summary


def assert_expected_counts(
    cases: pd.DataFrame,
    totals: dict[str, int],
    scan_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, int]:
    actual = dict(totals)
    actual.update(
        {
            "background_scans": int(len(scan_summary)),
            "fixed_false_alarms": int(cases["false_alarm_fixed"].sum()),
            "residual_false_alarms": int(cases["false_alarm_residual"].sum()),
            "removed_by_residual": int(
                cases["transition"].eq("removed_by_residual").sum()
            ),
            "added_by_residual": int(
                cases["transition"].eq("added_by_residual").sum()
            ),
            "retained_false_alarms": int(
                cases["transition"].eq("retained_false_alarm").sum()
            ),
            "review_queue_rows": int(cases["false_alarm_fixed"].sum()),
            "reviewed_rows": int(cases["reviewed"].sum()),
            "named_physical_labels": int(
                (cases["reviewed"] & cases["physical_class"].ne("unknown")).sum()
            ),
        }
    )
    expected = {key: int(value) for key, value in config["expected_counts"].items()}
    if actual != expected:
        differences = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)
        }
        raise ValueError(f"frozen count mismatch: {differences}")
    return actual


def assert_sanitized(frame: pd.DataFrame, role: str) -> None:
    forbidden = LOCAL_ONLY_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"{role} exposes local-only columns: {sorted(forbidden)}")
    unix_home_marker = "/" + "home" + "/"
    for column in frame.columns:
        values = frame[column].fillna("").astype(str)
        has_unix_home = values.str.contains(unix_home_marker, regex=False).any()
        has_windows_users = values.str.contains(r"\\Users\\", regex=False).any()
        if has_unix_home or has_windows_users:
            raise ValueError(f"{role}.{column} contains a local path")


def make_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return f"""# Zero-Doppler False-Alarm Library V1

Status: `COMPLETE_AS_DEVELOPMENT_AUDIT`.

This build pairs the frozen fixed-notch and residual predictions for all
{counts['background_samples']} background test samples. It records
{counts['fixed_false_alarms']} fixed-notch false alarms and
{counts['residual_false_alarms']} residual false alarms: {counts['removed_by_residual']}
were removed, {counts['added_by_residual']} were added, and
{counts['retained_false_alarms']} were retained.

The local `case_library_local.csv` contains sample identifiers, source-file
identifiers and human notes. It must remain local. The fold, anonymous-scan and
review-pattern tables are aggregate-only and are suitable for a sanitized
evidence package.

## Claim boundary

- This is a post-hoc audit of already consumed development predictions.
- No model was trained and no threshold was retuned in this build.
- The results are not an external blind test or a deployment Pfa estimate.
- The 11 reviewed cases describe visible image patterns only. Their physical
  classes remain `unknown` because no independent scene record is available.
- Relative H/V features remain uncalibrated and do not establish polarimetry.

## Files

- `case_library_local.csv`: local-only row-level case registry;
- `fold_transition_summary.csv`: shareable six-fold aggregate;
- `scan_transition_summary.csv`: shareable stable anonymous-scan aggregate;
- `review_pattern_summary.csv`: shareable human visible-pattern aggregate;
- `summary.json`: input hashes, frozen counts and claim flags.
"""


def build_library(
    *, config_path: Path, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    output_dir = resolve_path(output_dir)
    config = load_config(config_path)
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[pd.DataFrame] = []
    totals = {"test_samples": 0, "target_samples": 0, "background_samples": 0}
    verified_inputs: dict[str, Any] = {
        "config_sha256": sha256_file(config_path),
        "fixed_predictions": {},
        "residual_predictions": {},
    }
    for fold in config["folds"]:
        fold_key = str(fold)
        fixed_path = resolve_path(
            Path(config["fixed_prediction_template"].format(fold=fold))
        )
        residual_path = resolve_path(
            Path(config["residual_prediction_template"].format(fold=fold))
        )
        verified_inputs["fixed_predictions"][fold_key] = verify_hash(
            fixed_path,
            config["input_sha256"]["fixed_predictions"][fold_key],
            f"fixed fold {fold}",
        )
        verified_inputs["residual_predictions"][fold_key] = verify_hash(
            residual_path,
            config["input_sha256"]["residual_predictions"][fold_key],
            f"residual fold {fold}",
        )
        paired, fold_totals = pair_background_predictions(
            load_predictions(fixed_path, fold, "fixed"),
            load_predictions(residual_path, fold, "residual"),
            fold,
        )
        cases.append(paired)
        for key, value in fold_totals.items():
            totals[key] += int(value)
    case_library = pd.concat(cases, ignore_index=True)

    queue_path = resolve_path(Path(config["review_queue"]))
    reviewed_path = resolve_path(Path(config["reviewed_queue"]))
    verified_inputs["review_queue"] = verify_hash(
        queue_path, config["input_sha256"]["review_queue"], "review queue"
    )
    verified_inputs["reviewed_queue"] = verify_hash(
        reviewed_path, config["input_sha256"]["reviewed_queue"], "reviewed queue"
    )
    queue = pd.read_csv(queue_path, encoding="utf-8-sig")
    reviewed = pd.read_csv(reviewed_path, encoding="utf-8-sig")
    validate_queue(case_library, queue)
    case_library = apply_reviews(case_library, reviewed)

    fold_summary, scan_summary, review_summary = build_summaries(case_library, config)
    for role, frame in (
        ("fold summary", fold_summary),
        ("scan summary", scan_summary),
        ("review summary", review_summary),
    ):
        assert_sanitized(frame, role)
    counts = assert_expected_counts(case_library, totals, scan_summary, config)
    summary = {
        "schema_version": 1,
        "library_id": config["library_id"],
        "status": "COMPLETE_AS_DEVELOPMENT_AUDIT",
        "counts": counts,
        "review_completion_fraction": counts["reviewed_rows"]
        / counts["review_queue_rows"],
        "input_sha256": verified_inputs,
        "claim_boundaries": config["claim_boundaries"],
        "sharing_boundary": {
            "local_only_file": "case_library_local.csv",
            "shareable_aggregate_files": [
                "fold_transition_summary.csv",
                "scan_transition_summary.csv",
                "review_pattern_summary.csv",
                "summary.json",
                "README.md",
            ],
            "forbidden_row_level_fields": sorted(LOCAL_ONLY_COLUMNS),
            "scan_alias_mapping_published": False,
        },
    }

    case_library.to_csv(
        output_dir / "case_library_local.csv", index=False, encoding="utf-8-sig"
    )
    fold_summary.to_csv(
        output_dir / "fold_transition_summary.csv", index=False, encoding="utf-8-sig"
    )
    scan_summary.to_csv(
        output_dir / "scan_transition_summary.csv", index=False, encoding="utf-8-sig"
    )
    review_summary.to_csv(
        output_dir / "review_pattern_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(make_readme(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = build_library(
        config_path=args.config, output_dir=args.output_dir, overwrite=args.overwrite
    )
    counts = summary["counts"]
    print("Zero-Doppler false-alarm library: PASS")
    print(f"status={summary['status']}")
    print(f"background_samples={counts['background_samples']}")
    print(
        "false_alarm_transition="
        f"{counts['fixed_false_alarms']}->{counts['residual_false_alarms']} "
        f"(removed={counts['removed_by_residual']}, added={counts['added_by_residual']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
