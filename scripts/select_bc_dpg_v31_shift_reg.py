#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGULARIZATIONS = (0.01, 0.005, 0.0025, 0.001, 0.0)


@dataclass(frozen=True)
class CandidateRef:
    fold: int
    regularization: float
    experiment_name: str
    experiment_dir: Path
    source_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select BC-DPG-FCN v3.1 shift regularization using validation "
            "metrics only, then report the chosen candidate on test data."
        )
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--regularizations",
        nargs="+",
        type=float,
        default=list(DEFAULT_REGULARIZATIONS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/bc_dpg_v31_shift_reg",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested candidate is missing.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def reg_slug(value: float) -> str:
    text = format(float(value), ".10g")
    return text.replace("-", "m").replace(".", "p")


def candidate_ref(
    fold: int,
    regularization: float,
    seed: int,
    smoke: bool,
) -> CandidateRef:
    suffix = "_smoke" if smoke else ""
    # Prefer the already completed full-v3 result at the original weight 0.01.
    # If it is unavailable (for example when --no-reuse-full-v3 was used),
    # fall back to the independently trained v3.1 candidate with weight 0.01.
    if math.isclose(regularization, 0.01, rel_tol=0.0, abs_tol=1e-12):
        full_name = f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed{seed}{suffix}"
        full_dir = PROJECT_ROOT / "results" / "experiments" / full_name
        if (full_dir / "tables" / "summary.json").is_file():
            name = full_name
            source_kind = "reused_full_v3"
        else:
            name = (
                f"bc_dpg_v31_shiftreg_{reg_slug(regularization)}_"
                f"v4_fold{fold:02d}_seed{seed}{suffix}"
            )
            source_kind = "v31_candidate"
    else:
        name = (
            f"bc_dpg_v31_shiftreg_{reg_slug(regularization)}_"
            f"v4_fold{fold:02d}_seed{seed}{suffix}"
        )
        source_kind = "v31_candidate"
    return CandidateRef(
        fold=fold,
        regularization=float(regularization),
        experiment_name=name,
        experiment_dir=PROJECT_ROOT / "results" / "experiments" / name,
        source_kind=source_kind,
    )


def nested_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def int_or_nan(value: Any) -> float | int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return float("nan")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def validation_record(ref: CandidateRef) -> dict[str, Any]:
    """Extract only validation-side fields used by candidate selection.

    Test metrics are deliberately not referenced in this function. The chosen
    candidate is fixed first; test fields are loaded in a separate pass.
    """
    summary_path = ref.experiment_dir / "tables" / "summary.json"
    payload = load_json(summary_path)
    metrics = nested_get(payload, "base_threshold_validation_metrics", default={}) or {}
    raw_metrics = nested_get(
        payload,
        "raw_base_threshold_validation_metrics",
        default={},
    ) or {}
    pd_floor = nested_get(payload, "pd_floor_validation", default={}) or {}
    shifts = nested_get(payload, "shift_statistics_validation", default={}) or {}

    record = {
        "fold": ref.fold,
        "regularization": ref.regularization,
        "experiment_name": ref.experiment_name,
        "experiment_dir": str(ref.experiment_dir),
        "source_kind": ref.source_kind,
        "best_epoch": int_or_nan(payload.get("best_epoch")),
        "base_threshold": float_or_nan(payload.get("base_threshold")),
        "val_false_alarms": int_or_nan(metrics.get("false_alarm_count")),
        "val_pfa": float_or_nan(metrics.get("pfa")),
        "val_pd": float_or_nan(metrics.get("joint_pd")),
        "raw_val_pd": float_or_nan(raw_metrics.get("joint_pd")),
        "val_auc": float_or_nan(metrics.get("roc_auc")),
        "val_background_shift": float_or_nan(shifts.get("background_mean")),
        "val_target_shift": float_or_nan(shifts.get("target_mean")),
        "pd_floor_satisfied": bool(pd_floor.get("satisfied", False)),
        "score_never_increased_validation": bool(
            payload.get("score_never_increased_validation", False)
        ),
        "summary_path": str(summary_path),
    }
    record["eligible"] = bool(
        record["pd_floor_satisfied"]
        and record["score_never_increased_validation"]
    )
    return record


def finite_or(value: Any, fallback: float) -> float:
    number = float_or_nan(value)
    return number if math.isfinite(number) else fallback


def selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic validation-only ranking, smaller tuple is better.

    1. Require the validation Pd floor and monotonic score constraint.
    2. Minimize false alarms at the frozen DPG threshold.
    3. Maximize validation Pd, then AUC.
    4. Minimize target suppression.
    5. Prefer stronger regularization on exact ties for deployment safety.
    """
    return (
        0.0 if bool(record["eligible"]) else 1.0,
        finite_or(record["val_false_alarms"], float("inf")),
        -finite_or(record["val_pd"], -float("inf")),
        -finite_or(record["val_auc"], -float("inf")),
        finite_or(record["val_target_shift"], float("inf")),
        -float(record["regularization"]),
    )


def selected_test_record(validation_choice: dict[str, Any]) -> dict[str, Any]:
    """Load test metrics only after a validation winner has been fixed."""
    summary_path = Path(str(validation_choice["summary_path"]))
    payload = load_json(summary_path)
    test_metrics = nested_get(payload, "base_threshold_test_metrics", default={}) or {}
    raw_test_metrics = nested_get(
        payload,
        "raw_base_threshold_test_metrics",
        default={},
    ) or {}
    shifts = nested_get(payload, "shift_statistics_test", default={}) or {}
    probabilities = nested_get(
        payload,
        "background_probability_statistics_test",
        default={},
    ) or {}

    raw_fa = int_or_nan(raw_test_metrics.get("false_alarm_count"))
    calibrated_fa = int_or_nan(test_metrics.get("false_alarm_count"))
    reduction = (
        int(raw_fa) - int(calibrated_fa)
        if isinstance(raw_fa, int) and isinstance(calibrated_fa, int)
        else float("nan")
    )
    return {
        "fold": int(validation_choice["fold"]),
        "selected_regularization": float(validation_choice["regularization"]),
        "experiment_name": validation_choice["experiment_name"],
        "source_kind": validation_choice["source_kind"],
        "base_threshold": validation_choice["base_threshold"],
        "raw_test_false_alarms": raw_fa,
        "selected_test_false_alarms": calibrated_fa,
        "false_alarm_reduction": reduction,
        "raw_test_pfa": float_or_nan(raw_test_metrics.get("pfa")),
        "selected_test_pfa": float_or_nan(test_metrics.get("pfa")),
        "raw_test_pd": float_or_nan(raw_test_metrics.get("joint_pd")),
        "selected_test_pd": float_or_nan(test_metrics.get("joint_pd")),
        "raw_test_auc": float_or_nan(raw_test_metrics.get("roc_auc")),
        "selected_test_auc": float_or_nan(test_metrics.get("roc_auc")),
        "test_background_shift": float_or_nan(shifts.get("background_mean")),
        "test_target_shift": float_or_nan(shifts.get("target_mean")),
        "test_p_background_background": float_or_nan(
            probabilities.get("background_mean")
        ),
        "test_p_background_target": float_or_nan(probabilities.get("target_mean")),
        "score_never_increased_test": bool(
            payload.get("score_never_increased_test", False)
        ),
    }


def aggregate_test(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric = [
        "raw_test_false_alarms",
        "selected_test_false_alarms",
        "false_alarm_reduction",
        "raw_test_pfa",
        "selected_test_pfa",
        "raw_test_pd",
        "selected_test_pd",
        "raw_test_auc",
        "selected_test_auc",
        "test_background_shift",
        "test_target_shift",
    ]
    rows: list[dict[str, Any]] = []
    for column in numeric:
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "metric": column,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "sum": float(series.sum()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    validation_frame: pd.DataFrame,
    selected_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    missing: list[str],
) -> str:
    lines = [
        "# BC-DPG-FCN v3.1 shift regularization validation selection",
        "",
        "## Protocol",
        "",
        "Each fold selects its shift-regularization weight using validation-side "
        "metrics only. The selector never ranks candidates using test Pd, Pfa, "
        "AUC, false alarms, or test shift statistics.",
        "",
        "Selection order:",
        "",
        "1. validation Pd floor and score-never-increased constraints;",
        "2. minimum validation false-alarm count at the frozen DPG threshold;",
        "3. maximum validation Pd;",
        "4. maximum validation AUC;",
        "5. minimum validation target shift;",
        "6. stronger regularization on an exact tie.",
        "",
        "Candidate training currently creates test files, but these fields are not "
        "read by the selection pass. For a future independent blind dataset, the "
        "same selected rule should be frozen before opening the blind test set.",
        "",
        "## Selected weight by fold",
        "",
    ]
    if selected_frame.empty:
        lines.append("No complete fold selection was available.")
    else:
        display_columns = [
            "fold",
            "regularization",
            "val_false_alarms",
            "val_pd",
            "val_auc",
            "val_target_shift",
            "eligible",
        ]
        lines.append(selected_frame[display_columns].to_markdown(index=False))

    lines += ["", "## Selected candidate test report", ""]
    if test_frame.empty:
        lines.append("No selected test results were available.")
    else:
        display_columns = [
            "fold",
            "selected_regularization",
            "raw_test_false_alarms",
            "selected_test_false_alarms",
            "raw_test_pd",
            "selected_test_pd",
            "test_target_shift",
        ]
        lines.append(test_frame[display_columns].to_markdown(index=False))
        lines += [
            "",
            f"Raw false alarms (sum): {int(test_frame['raw_test_false_alarms'].sum())}",
            f"Selected false alarms (sum): {int(test_frame['selected_test_false_alarms'].sum())}",
            f"Mean raw Pd: {test_frame['raw_test_pd'].mean():.6f}",
            f"Mean selected Pd: {test_frame['selected_test_pd'].mean():.6f}",
            f"Mean selected target shift: {test_frame['test_target_shift'].mean():.6f}",
        ]

    lines += ["", "## Candidate coverage", ""]
    lines.append(f"Candidate rows found: {len(validation_frame)}")
    lines.append(f"Folds selected: {len(selected_frame)}")
    lines.append(f"Missing candidates: {len(missing)}")
    if missing:
        lines += ["", "Missing summary files:", ""]
        lines.extend(f"- `{item}`" for item in missing)
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This remains development-stage internal cross-validation. It optimizes "
        "a hyperparameter without using candidate test metrics in the ranking, but "
        "it does not replace a frozen-model, new-date, new-environment blind test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for fold in args.folds:
        if fold not in range(1, 7):
            raise ValueError(f"Fold must be 1-6, got {fold}")
    regularizations = tuple(dict.fromkeys(float(v) for v in args.regularizations))
    if any(value < 0 for value in regularizations):
        raise ValueError("Regularization weights must be non-negative")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else "_formal"

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in args.folds:
        for regularization in regularizations:
            ref = candidate_ref(fold, regularization, args.seed, args.smoke)
            summary_path = ref.experiment_dir / "tables" / "summary.json"
            if not summary_path.is_file():
                missing.append(str(summary_path))
                continue
            records.append(validation_record(ref))

    validation_frame = pd.DataFrame(records)
    validation_path = output_dir / f"candidate_validation_metrics{suffix}.csv"
    validation_frame.to_csv(validation_path, index=False, encoding="utf-8-sig")

    selected_records: list[dict[str, Any]] = []
    for fold in args.folds:
        fold_records = [record for record in records if int(record["fold"]) == fold]
        if len(fold_records) != len(regularizations):
            continue
        chosen = min(fold_records, key=selection_key)
        chosen = dict(chosen)
        chosen["selection_key"] = list(selection_key(chosen))
        selected_records.append(chosen)

    selected_frame = pd.DataFrame(selected_records)
    selected_path = output_dir / f"selected_by_fold{suffix}.csv"
    selected_frame.to_csv(selected_path, index=False, encoding="utf-8-sig")

    test_records = [selected_test_record(record) for record in selected_records]
    test_frame = pd.DataFrame(test_records)
    test_path = output_dir / f"selected_test_metrics{suffix}.csv"
    test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")

    aggregate = aggregate_test(test_frame)
    aggregate_path = output_dir / f"aggregate_selected_test_metrics{suffix}.csv"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    audit = {
        "folds": list(args.folds),
        "regularizations": list(regularizations),
        "smoke": bool(args.smoke),
        "selection_uses_test_metrics": False,
        "selection_fields": [
            "pd_floor_satisfied",
            "score_never_increased_validation",
            "val_false_alarms",
            "val_pd",
            "val_auc",
            "val_target_shift",
            "regularization",
        ],
        "candidate_rows_found": len(records),
        "selected_folds": len(selected_records),
        "missing_summary_files": missing,
    }
    audit_path = output_dir / f"selection_audit{suffix}.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = output_dir / f"README_shift_regularization_selection{suffix}.md"
    report_path.write_text(
        build_report(validation_frame, selected_frame, test_frame, missing),
        encoding="utf-8",
    )

    print("=" * 82)
    print("BC-DPG-FCN v3.1 validation-only regularization selection complete")
    print(f"candidate table : {validation_path}")
    print(f"selected table  : {selected_path}")
    print(f"test report     : {test_path}")
    print(f"aggregate       : {aggregate_path}")
    print(f"audit           : {audit_path}")
    print(f"report          : {report_path}")
    print(f"candidate rows  : {len(records)}")
    print(f"selected folds  : {len(selected_records)}")
    print(f"missing         : {len(missing)}")
    print("=" * 82)

    if args.require_all and missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
