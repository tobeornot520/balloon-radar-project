#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deployment-assumption comparison for raw DPG, "
            "sample-independent BC-DPG, and scan-aware BC-DPG."
        )
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/bc_dpg_v3_deployment_comparison",
    )
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def metric(payload: dict[str, Any], block: str, key: str) -> float:
    value = payload.get(block, {}).get(key)
    return float(value)


def mode_name(mode: str, fold: int, seed: int) -> str:
    return f"bc_dpg_v3_ablation_{mode}_v4_fold{fold:02d}_seed{seed}"


def model_record(
    fold: int,
    label: str,
    payload: dict[str, Any],
    *,
    use_raw: bool,
    experiment_name: str,
) -> dict[str, Any]:
    metrics_block = (
        "raw_base_threshold_test_metrics"
        if use_raw
        else "base_threshold_test_metrics"
    )
    metrics = payload[metrics_block]
    shift = (
        {"background_mean": 0.0, "target_mean": 0.0}
        if use_raw
        else payload["shift_statistics_test"]
    )
    if label == "raw_dpg":
        context_requirement = "current sample only"
        causality = "causal/sample-independent"
    elif label == "sample_independent_bc":
        context_requirement = "current sample features; 12 scan statistics forced to zero"
        causality = "causal/sample-independent"
    else:
        context_requirement = "current sample plus complete scan-group statistics"
        causality = "offline scan-aware; may use future samples within the scan"

    return {
        "fold": fold,
        "model": label,
        "experiment_name": experiment_name,
        "context_requirement": context_requirement,
        "causality": causality,
        "false_alarms": int(metrics["false_alarm_count"]),
        "pfa": float(metrics["pfa"]),
        "pd": float(metrics["joint_pd"]),
        "auc": float(metrics["roc_auc"]),
        "background_shift": float(shift["background_mean"]),
        "target_shift": float(shift["target_mean"]),
    }


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in detail.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "folds": int(len(group)),
                "false_alarms_sum": int(group["false_alarms"].sum()),
                "pfa_mean": float(group["pfa"].mean()),
                "pd_mean": float(group["pd"].mean()),
                "auc_mean": float(group["auc"].mean()),
                "background_shift_mean": float(group["background_shift"].mean()),
                "target_shift_mean": float(group["target_shift"].mean()),
                "context_requirement": group["context_requirement"].iloc[0],
                "causality": group["causality"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def build_report(detail: pd.DataFrame, aggregate_frame: pd.DataFrame, missing: list[str]) -> str:
    lines = [
        "# BC-DPG-FCN v3 deployment-assumption comparison",
        "",
        "This report separates detection performance from deployment assumptions.",
        "",
        "## Compared systems",
        "",
        "- **raw_dpg**: the frozen DPG detector; one sample can be evaluated independently.",
        "- **sample_independent_bc**: the same calibrator architecture trained with all "
        "  12 scan-group features fixed to zero (`no_scan_context`).",
        "- **scan_aware_bc**: the complete v3 calibrator using statistics computed from "
        "  the complete scan group.",
        "",
        "The third system is currently an offline scan-level calibrator. Its result should "
        "not be presented as strictly causal real-time inference because complete-scan "
        "statistics can include samples occurring after the sample being classified.",
        "",
        "## Six-fold aggregate",
        "",
    ]
    if aggregate_frame.empty:
        lines.append("No complete results were available.")
    else:
        display = aggregate_frame[
            [
                "model",
                "false_alarms_sum",
                "pfa_mean",
                "pd_mean",
                "auc_mean",
                "target_shift_mean",
                "causality",
            ]
        ]
        lines.append(display.to_markdown(index=False))

    lines += [
        "",
        "## Correct interpretation",
        "",
        "The difference between `scan_aware_bc` and `sample_independent_bc` measures "
        "the benefit of scan-group context under the current dataset and split. It does "
        "not prove that an arbitrary future environment has the same group structure.",
        "",
        "For the later self-collected core dataset, the main model should remain "
        "sample-independent at test time. A separate causal history module may use only "
        "past observations from a continuously operating radar.",
        "",
        "## Coverage",
        "",
        f"Detail rows: {len(detail)}",
        f"Missing summaries: {len(missing)}",
    ]
    if missing:
        lines += ["", "Missing files:", ""]
        lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    rows: list[dict[str, Any]] = []

    for fold in args.folds:
        if fold not in range(1, 7):
            raise ValueError(f"Fold must be 1-6, got {fold}")
        full_name = mode_name("full", fold, args.seed)
        no_context_name = mode_name("no_scan_context", fold, args.seed)
        full_summary = (
            PROJECT_ROOT / "results" / "experiments" / full_name / "tables" / "summary.json"
        )
        no_context_summary = (
            PROJECT_ROOT
            / "results"
            / "experiments"
            / no_context_name
            / "tables"
            / "summary.json"
        )
        for path in (full_summary, no_context_summary):
            if not path.is_file():
                missing.append(str(path))
        if not full_summary.is_file() or not no_context_summary.is_file():
            continue

        full_payload = load_summary(full_summary)
        no_context_payload = load_summary(no_context_summary)
        rows.append(
            model_record(
                fold,
                "raw_dpg",
                full_payload,
                use_raw=True,
                experiment_name=full_name,
            )
        )
        rows.append(
            model_record(
                fold,
                "sample_independent_bc",
                no_context_payload,
                use_raw=False,
                experiment_name=no_context_name,
            )
        )
        rows.append(
            model_record(
                fold,
                "scan_aware_bc",
                full_payload,
                use_raw=False,
                experiment_name=full_name,
            )
        )

    detail = pd.DataFrame(rows)
    aggregate_frame = aggregate(detail) if not detail.empty else pd.DataFrame()
    detail_path = output_dir / "deployment_comparison_detail.csv"
    aggregate_path = output_dir / "deployment_comparison_aggregate.csv"
    report_path = output_dir / "README_deployment_comparison.md"
    audit_path = output_dir / "deployment_comparison_audit.json"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    aggregate_frame.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(detail, aggregate_frame, missing),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "folds": list(args.folds),
                "seed": args.seed,
                "detail_rows": len(detail),
                "missing_summary_files": missing,
                "test_metrics_used_for_reporting_only": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 82)
    print("BC-DPG-FCN v3 deployment comparison complete")
    print(f"detail    : {detail_path}")
    print(f"aggregate : {aggregate_path}")
    print(f"report    : {report_path}")
    print(f"audit     : {audit_path}")
    print(f"rows      : {len(detail)}")
    print(f"missing   : {len(missing)}")
    print("=" * 82)
    if args.require_all and missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
