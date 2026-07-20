#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
)

OUTPUT_ROOT = (
    EXPERIMENT_ROOT
    / "dataset_v4_multifold_comparison"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPERIMENT_PATTERNS = {
    "H": (
        "detection_h_v4_"
        "fold{fold:02d}_seed42"
    ),
    "V": (
        "detection_v_v4_"
        "fold{fold:02d}_seed42"
    ),
    "HV": (
        "detection_hv_v4_"
        "fold{fold:02d}_seed42"
    ),
    "DPG": (
        "dpg_fcn_v4_"
        "fold{fold:02d}_seed42"
    ),
}


TARGET_FILES = (
    "val_predictions.csv",
    "test_predictions.csv",
    "val_branch_disagreement.csv",
    "test_branch_disagreement.csv",
)


CANDIDATE_PATTERNS = {
    "sample_id": (
        "sample_id",
        "id",
        "sample",
    ),
    "source_file": (
        "source_file",
        "source",
        "file",
        "filename",
        "path",
    ),
    "class_label": (
        "target_present",
        "label",
        "class",
        "is_positive",
        "ground_truth",
    ),
    "score": (
        "score",
        "max_score",
        "prediction_score",
        "confidence",
        "probability",
        "peak",
    ),
    "prediction": (
        "predicted_present",
        "prediction",
        "predicted_class",
        "is_detection",
        "detected",
    ),
    "threshold": (
        "threshold",
    ),
    "h_score": (
        "h_score",
        "score_h",
        "h_prediction",
        "h_confidence",
        "h_peak",
    ),
    "v_score": (
        "v_score",
        "score_v",
        "v_prediction",
        "v_confidence",
        "v_peak",
    ),
    "h_gate": (
        "h_gate",
        "gate_h",
        "h_weight",
        "alpha_h",
        "mean_h_gate",
    ),
    "v_gate": (
        "v_gate",
        "gate_v",
        "v_weight",
        "alpha_v",
        "mean_v_gate",
    ),
    "dominant_branch": (
        "dominant_branch",
        "dominant",
        "branch",
    ),
    "gate_margin": (
        "gate_margin",
        "margin",
    ),
}


def normalize(text: str) -> str:
    return text.strip().lower()


def find_candidates(
    columns: list[str],
) -> dict[str, list[str]]:
    normalized = {
        column: normalize(column)
        for column in columns
    }

    result: dict[str, list[str]] = {}

    for role, patterns in (
        CANDIDATE_PATTERNS.items()
    ):
        matches: list[str] = []

        for column, lowered in (
            normalized.items()
        ):
            if any(
                pattern == lowered
                or pattern in lowered
                for pattern in patterns
            ):
                matches.append(column)

        result[role] = matches

    return result


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


schema_rows: list[dict[str, Any]] = []

candidate_report: dict[str, Any] = {}


for fold in range(1, 7):
    for model, pattern in (
        EXPERIMENT_PATTERNS.items()
    ):
        experiment_name = pattern.format(
            fold=fold
        )

        table_root = (
            EXPERIMENT_ROOT
            / experiment_name
            / "tables"
        )

        for filename in TARGET_FILES:
            path = table_root / filename

            if not path.is_file():
                schema_rows.append(
                    {
                        "fold": fold,
                        "model": model,
                        "experiment_name": (
                            experiment_name
                        ),
                        "filename": filename,
                        "exists": False,
                        "row_count": 0,
                        "column_count": 0,
                        "columns": "",
                    }
                )
                continue

            frame = pd.read_csv(path)

            columns = [
                str(column)
                for column in frame.columns
            ]

            candidates = find_candidates(
                columns
            )

            schema_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "experiment_name": (
                        experiment_name
                    ),
                    "filename": filename,
                    "exists": True,
                    "row_count": len(frame),
                    "column_count": len(
                        columns
                    ),
                    "columns": "|".join(
                        columns
                    ),
                }
            )

            key = (
                f"fold{fold:02d}_"
                f"{model}_{filename}"
            )

            first_two_rows = []

            for row in (
                frame
                .head(2)
                .to_dict(orient="records")
            ):
                converted = {
                    str(column): json_safe(
                        value
                    )
                    for column, value in (
                        row.items()
                    )
                }

                first_two_rows.append(
                    converted
                )

            candidate_report[key] = {
                "path": str(
                    path.resolve()
                ),
                "row_count": len(frame),
                "columns": columns,
                "candidate_columns": (
                    candidates
                ),
                "first_two_rows": (
                    first_two_rows
                ),
            }


schema = pd.DataFrame(
    schema_rows
)


schema_output = (
    OUTPUT_ROOT
    / "prediction_gate_schema_summary.csv"
)

candidate_output = (
    OUTPUT_ROOT
    / "prediction_gate_candidate_columns.json"
)


schema.to_csv(
    schema_output,
    index=False,
    encoding="utf-8-sig",
)


candidate_output.write_text(
    json.dumps(
        candidate_report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 110)
print("FILE AVAILABILITY")
print("=" * 110)

availability = (
    schema
    .groupby(
        [
            "model",
            "filename",
        ]
    )["exists"]
    .agg(
        [
            "sum",
            "count",
        ]
    )
    .reset_index()
)

print(
    availability.to_string(
        index=False
    )
)


print()
print("=" * 110)
print("KEY FIELD CANDIDATES")
print("=" * 110)


for key, item in (
    candidate_report.items()
):
    if not (
        key.endswith(
            "test_predictions.csv"
        )
        or key.endswith(
            "test_branch_disagreement.csv"
        )
    ):
        continue

    print()
    print(key)
    print(
        "rows:",
        item["row_count"],
    )
    print(
        "columns:",
        item["columns"],
    )

    candidates = item[
        "candidate_columns"
    ]

    roles = (
        "sample_id",
        "source_file",
        "class_label",
        "score",
        "prediction",
        "h_score",
        "v_score",
        "h_gate",
        "v_gate",
        "dominant_branch",
        "gate_margin",
    )

    for role in roles:
        matches = candidates.get(
            role,
            [],
        )

        if matches:
            print(
                f"  {role}: "
                f"{matches}"
            )


print()
print("=" * 110)
print("OUTPUT FILES")
print("=" * 110)

print(schema_output.resolve())
print(candidate_output.resolve())

print()
print(
    "V4_PREDICTION_GATE_"
    "SCHEMA_AUDIT_OK"
)
