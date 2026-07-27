#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "balloon_radar_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "final_roi_bc_dpg_joint_v2_base_threshold"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_evidence"
    / "roi_bc_dpg_joint_fixed_threshold"
)

EXPECTED_FOLDS = (1, 2, 3, 4, 5, 6)
EXPECTED_ROWS = 1148
EXPECTED_BC_SOURCE = "base_threshold_test_predictions.csv"
EXPECTED_ROI_SOURCE = "refined_fixed_* columns from test_predictions.csv"
EXPECTED_AUDIT_NAME = "final_roi_bc_dpg_joint_v2_base_threshold"

SOURCE_FILES = (
    "detection_comparison.csv",
    "complementarity_summary.csv",
    "fold_alignment.csv",
    "status.json",
    "all_fold_joint_predictions.csv",
)

MODEL_COLUMNS = {
    "bc_dpg_v3": ("bc_false_alarm", "bc_correct"),
    "roi_baseline": ("roi_base_false_alarm", "roi_base_correct"),
    "roi_power_control": ("roi_power_false_alarm", "roi_power_correct"),
    "roi_ri4": ("roi_ri4_false_alarm", "roi_ri4_correct"),
}
COMPARISON_COLUMNS = {
    "roi_power_control": ("roi_power_false_alarm", "roi_power_correct"),
    "roi_ri4": ("roi_ri4_false_alarm", "roi_ri4_correct"),
}
MODEL_LABELS = {
    "bc_dpg_v3": "BC-DPG-FCN v3",
    "roi_baseline": "Power2 baseline",
    "roi_power_control": "ROI power control",
    "roi_ri4": "ROI RI4",
}
MODEL_ROLES = {
    "bc_dpg_v3": "frozen current detector",
    "roi_baseline": "frozen candidate baseline",
    "roi_power_control": "ROI suppression control",
    "roi_ri4": "independent ROI suppression study",
}
MODEL_COLORS = {
    "bc_dpg_v3": "#1B4965",
    "roi_baseline": "#7A7F87",
    "roi_power_control": "#D9822B",
    "roi_ri4": "#238B7E",
}


@dataclass(frozen=True)
class AuditData:
    input_dir: Path
    status: dict[str, Any]
    detection: pd.DataFrame
    complementarity: pd.DataFrame
    alignment: pd.DataFrame
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic paper assets from the frozen fixed-threshold "
            "ROI/BC-DPG joint audit. No model training or threshold selection is run."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing nonempty output directory after a successful build.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def boolean_series(series: pd.Series, label: str) -> pd.Series:
    truthy = {"1", "true", "yes", "y", "t"}
    falsy = {"0", "false", "no", "n", "f", "", "none", "nan"}
    normalized = series.map(lambda value: str(value).strip().lower())
    unknown = sorted(set(normalized) - truthy - falsy)
    if unknown:
        raise ValueError(f"{label} contains unsupported boolean values: {unknown}")
    return normalized.isin(truthy)


def validate_status(status: dict[str, Any]) -> None:
    errors: list[str] = []
    if status.get("status") != "ok":
        errors.append("status must be 'ok'")
    if status.get("bc_decision_source") != EXPECTED_BC_SOURCE:
        errors.append(f"BC decision source must be {EXPECTED_BC_SOURCE!r}")
    if status.get("roi_decision_source") != EXPECTED_ROI_SOURCE:
        errors.append(f"ROI decision source must be {EXPECTED_ROI_SOURCE!r}")
    if status.get("folds") != list(EXPECTED_FOLDS):
        errors.append(f"folds must be {list(EXPECTED_FOLDS)}")
    if status.get("rows") != EXPECTED_ROWS:
        errors.append(f"rows must be {EXPECTED_ROWS}")
    if status.get("exact_alignment") is not True:
        errors.append("exact_alignment must be true")
    if status.get("test_threshold_retuning") is not False:
        errors.append("test_threshold_retuning must be false")
    if Path(str(status.get("output_dir", ""))).name != EXPECTED_AUDIT_NAME:
        errors.append(f"status output_dir must identify {EXPECTED_AUDIT_NAME!r}")
    if errors:
        raise ValueError("Invalid frozen audit status: " + "; ".join(errors))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def fold_label(value: Any) -> str:
    text = str(value).strip()
    if text.upper() == "ALL":
        return "ALL"
    return f"{int(float(text)):02d}"


def detection_metrics(frame: pd.DataFrame, fold: str, model: str) -> dict[str, Any]:
    false_alarm_column, correct_column = MODEL_COLUMNS[model]
    labels = pd.to_numeric(frame["target_present"], errors="raise").astype(int)
    background_count = int(labels.eq(0).sum())
    target_count = int(labels.eq(1).sum())
    false_alarms = int(frame[false_alarm_column].sum())
    correct = int(frame[correct_column].sum())
    return {
        "fold": fold,
        "model": model,
        "background_samples": background_count,
        "target_samples": target_count,
        "false_alarms": false_alarms,
        "pfa": false_alarms / background_count,
        "correct_detections": correct,
        "joint_pd": correct / target_count,
    }


def complementarity_metrics(
    frame: pd.DataFrame,
    fold: str,
    comparison: str,
) -> dict[str, Any]:
    roi_false_alarm_column, roi_correct_column = COMPARISON_COLUMNS[comparison]
    labels = pd.to_numeric(frame["target_present"], errors="raise").astype(int)
    background = labels.eq(0)
    target = labels.eq(1)
    bc_false_alarm = frame["bc_false_alarm"] & background
    roi_false_alarm = frame[roi_false_alarm_column] & background
    bc_correct = frame["bc_correct"] & target
    roi_correct = frame[roi_correct_column] & target
    return {
        "fold": fold,
        "comparison": comparison,
        "bc_false_alarms": int(bc_false_alarm.sum()),
        "roi_false_alarms": int(roi_false_alarm.sum()),
        "shared_false_alarms": int((bc_false_alarm & roi_false_alarm).sum()),
        "bc_only_false_alarms": int((bc_false_alarm & ~roi_false_alarm).sum()),
        "roi_only_false_alarms": int((roi_false_alarm & ~bc_false_alarm).sum()),
        "fa_union": int((bc_false_alarm | roi_false_alarm).sum()),
        "target_samples": int(target.sum()),
        "shared_correct": int((bc_correct & roi_correct).sum()),
        "bc_only_correct": int((bc_correct & ~roi_correct).sum()),
        "roi_only_correct": int((roi_correct & ~bc_correct).sum()),
        "correct_union": int((bc_correct | roi_correct).sum()),
        "neither_correct": int((target & ~bc_correct & ~roi_correct).sum()),
    }


def derive_summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detection_rows: list[dict[str, Any]] = []
    complementarity_rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        current = predictions.loc[predictions["fold"].eq(fold)]
        label = f"{fold:02d}"
        for model in MODEL_COLUMNS:
            detection_rows.append(detection_metrics(current, label, model))
        for comparison in COMPARISON_COLUMNS:
            complementarity_rows.append(
                complementarity_metrics(current, label, comparison)
            )
    for model in MODEL_COLUMNS:
        detection_rows.append(detection_metrics(predictions, "ALL", model))
    for comparison in COMPARISON_COLUMNS:
        complementarity_rows.append(
            complementarity_metrics(predictions, "ALL", comparison)
        )
    return pd.DataFrame(detection_rows), pd.DataFrame(complementarity_rows)


def assert_frames_match(
    reported: pd.DataFrame,
    derived: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    if reported.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate keys")
    if derived.duplicated(keys).any():
        raise ValueError(f"derived {label} contains duplicate keys")
    metrics = [column for column in derived.columns if column not in keys]
    require_columns(reported, [*keys, *metrics], label)
    merged = reported[[*keys, *metrics]].merge(
        derived,
        on=keys,
        how="outer",
        suffixes=("_reported", "_derived"),
        indicator=True,
        validate="one_to_one",
    )
    if len(merged) != len(derived) or not merged["_merge"].eq("both").all():
        raise ValueError(f"{label} keys do not match the prediction-derived summary")
    mismatches: list[str] = []
    for metric in metrics:
        left = pd.to_numeric(merged[f"{metric}_reported"], errors="coerce")
        right = pd.to_numeric(merged[f"{metric}_derived"], errors="coerce")
        equal = np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
        if not bool(np.all(equal)):
            mismatches.append(metric)
    if mismatches:
        raise ValueError(f"{label} disagrees with predictions in: {mismatches}")


def validate_alignment(alignment: pd.DataFrame, predictions: pd.DataFrame) -> None:
    columns = (
        "fold",
        "rows",
        "unique_sample_id",
        "exact_sample_alignment",
        "exact_label_alignment",
        "exact_mat_path_alignment",
        "bc_source",
    )
    require_columns(alignment, columns, "fold_alignment.csv")
    if alignment["fold"].astype(int).tolist() != list(EXPECTED_FOLDS):
        raise ValueError("fold_alignment.csv must contain folds 1 through 6 in order")
    for column in (
        "exact_sample_alignment",
        "exact_label_alignment",
        "exact_mat_path_alignment",
    ):
        if not boolean_series(alignment[column], column).all():
            raise ValueError(f"{column} must be true for every fold")
    rows = pd.to_numeric(alignment["rows"], errors="raise").astype(int)
    unique_ids = pd.to_numeric(
        alignment["unique_sample_id"], errors="raise"
    ).astype(int)
    observed = predictions.groupby("fold", sort=True).size().reindex(EXPECTED_FOLDS)
    if rows.tolist() != observed.tolist() or not rows.equals(unique_ids):
        raise ValueError("fold_alignment.csv row counts do not match predictions")
    if int(rows.sum()) != EXPECTED_ROWS:
        raise ValueError(f"fold_alignment.csv must total {EXPECTED_ROWS} rows")
    sources = alignment["bc_source"].map(lambda value: Path(str(value)).name)
    if not sources.eq(EXPECTED_BC_SOURCE).all():
        raise ValueError("fold_alignment.csv contains a non-frozen BC decision source")


def load_and_validate(input_dir: Path) -> AuditData:
    input_dir = resolve_path(input_dir)
    missing = [name for name in SOURCE_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen audit is missing required files: {missing}")

    status = json.loads((input_dir / "status.json").read_text(encoding="utf-8"))
    validate_status(status)
    detection = pd.read_csv(input_dir / "detection_comparison.csv", dtype={"fold": str})
    complementarity = pd.read_csv(
        input_dir / "complementarity_summary.csv", dtype={"fold": str}
    )
    alignment = pd.read_csv(input_dir / "fold_alignment.csv")
    predictions = pd.read_csv(input_dir / "all_fold_joint_predictions.csv")

    prediction_columns = {
        "fold",
        "split",
        "sample_id",
        "target_present",
        *(column for pair in MODEL_COLUMNS.values() for column in pair),
    }
    require_columns(predictions, prediction_columns, "all_fold_joint_predictions.csv")
    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(f"prediction table must contain {EXPECTED_ROWS} rows")
    predictions["fold"] = pd.to_numeric(predictions["fold"], errors="raise").astype(int)
    if sorted(predictions["fold"].unique().tolist()) != list(EXPECTED_FOLDS):
        raise ValueError("prediction table must contain exactly folds 1 through 6")
    if not predictions["split"].astype(str).str.lower().eq("test").all():
        raise ValueError("prediction table must contain test rows only")
    if predictions.duplicated(["fold", "sample_id"]).any():
        raise ValueError("prediction table contains duplicate fold/sample_id rows")
    labels = pd.to_numeric(predictions["target_present"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("target_present must contain only 0 and 1")
    predictions["target_present"] = labels
    for column in {column for pair in MODEL_COLUMNS.values() for column in pair}:
        predictions[column] = boolean_series(predictions[column], column)
    for false_alarm_column, correct_column in MODEL_COLUMNS.values():
        if (predictions[false_alarm_column] & labels.eq(1)).any():
            raise ValueError(f"{false_alarm_column} is true for a target row")
        if (predictions[correct_column] & labels.eq(0)).any():
            raise ValueError(f"{correct_column} is true for a background row")

    detection["fold"] = detection["fold"].map(fold_label)
    complementarity["fold"] = complementarity["fold"].map(fold_label)
    derived_detection, derived_complementarity = derive_summaries(predictions)
    assert_frames_match(
        detection,
        derived_detection,
        ["fold", "model"],
        "detection_comparison.csv",
    )
    assert_frames_match(
        complementarity,
        derived_complementarity,
        ["fold", "comparison"],
        "complementarity_summary.csv",
    )
    validate_alignment(alignment, predictions)

    return AuditData(
        input_dir=input_dir,
        status=status,
        detection=detection,
        complementarity=complementarity,
        alignment=alignment,
        predictions=predictions,
    )


def pooled_detection_table(detection: pd.DataFrame) -> pd.DataFrame:
    pooled = detection.loc[detection["fold"].eq("ALL")].copy()
    expected = list(MODEL_COLUMNS)
    if pooled["model"].tolist() != expected:
        pooled = pooled.set_index("model").loc[expected].reset_index()
    pooled.insert(1, "display_name", pooled["model"].map(MODEL_LABELS))
    pooled.insert(2, "evidence_role", pooled["model"].map(MODEL_ROLES))
    return pooled[
        [
            "model",
            "display_name",
            "evidence_role",
            "background_samples",
            "target_samples",
            "false_alarms",
            "pfa",
            "correct_detections",
            "joint_pd",
        ]
    ]


def fold_detection_table(detection: pd.DataFrame) -> pd.DataFrame:
    detail = detection.loc[~detection["fold"].eq("ALL")].copy()
    detail.insert(2, "display_name", detail["model"].map(MODEL_LABELS))
    return detail[
        [
            "fold",
            "model",
            "display_name",
            "background_samples",
            "target_samples",
            "false_alarms",
            "pfa",
            "correct_detections",
            "joint_pd",
        ]
    ]


def pooled_complementarity_table(complementarity: pd.DataFrame) -> pd.DataFrame:
    pooled = complementarity.loc[complementarity["fold"].eq("ALL")].copy()
    pooled.insert(
        2,
        "comparison_name",
        pooled["comparison"].map(
            lambda model: f"BC-DPG-FCN v3 vs {MODEL_LABELS[str(model)]}"
        ),
    )
    return pooled


def build_simple_combination_diagnostics(
    detection: pd.DataFrame,
    complementarity: pd.DataFrame,
) -> pd.DataFrame:
    pooled_detection = detection.loc[detection["fold"].eq("ALL")].set_index("model")
    selected = complementarity.loc[
        complementarity["fold"].eq("ALL")
        & complementarity["comparison"].eq("roi_ri4")
    ]
    if len(selected) != 1:
        raise ValueError("Expected one pooled BC-DPG vs ROI RI4 comparison")
    overlap = selected.iloc[0]
    target_samples = int(overlap["target_samples"])
    bc = pooled_detection.loc["bc_dpg_v3"]
    roi = pooled_detection.loc["roi_ri4"]
    rows = [
        {
            "rule": "BC-DPG-FCN v3 alone",
            "logical_definition": "frozen BC decision",
            "false_alarms": int(bc["false_alarms"]),
            "correct_detections": int(bc["correct_detections"]),
            "target_samples": target_samples,
            "missed_targets": target_samples - int(bc["correct_detections"]),
            "selection_status": "frozen current detector",
            "interpretation": "Primary detector retained by the current evidence.",
        },
        {
            "rule": "ROI RI4 alone",
            "logical_definition": "frozen ROI RI4 decision",
            "false_alarms": int(roi["false_alarms"]),
            "correct_detections": int(roi["correct_detections"]),
            "target_samples": target_samples,
            "missed_targets": target_samples - int(roi["correct_detections"]),
            "selection_status": "independent suppression study",
            "interpretation": "Reported as an independent ROI study, not a replacement.",
        },
        {
            "rule": "AND / intersection",
            "logical_definition": "BC detected AND ROI RI4 detected",
            "false_alarms": int(overlap["shared_false_alarms"]),
            "correct_detections": int(overlap["shared_correct"]),
            "target_samples": target_samples,
            "missed_targets": target_samples - int(overlap["shared_correct"]),
            "selection_status": "diagnostic only, not selected",
            "interpretation": (
                "Reduces false alarms but discards BC-only correct detections."
            ),
        },
        {
            "rule": "OR / union",
            "logical_definition": "BC detected OR ROI RI4 detected",
            "false_alarms": int(overlap["fa_union"]),
            "correct_detections": int(overlap["correct_union"]),
            "target_samples": target_samples,
            "missed_targets": target_samples - int(overlap["correct_union"]),
            "selection_status": "diagnostic only, not selected",
            "interpretation": (
                "Recovers ROI-only correct detections but admits the false-alarm union."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_claim_boundaries() -> pd.DataFrame:
    rows = [
        (
            "supported",
            "At the frozen thresholds, BC-DPG-FCN v3 has 56 false alarms and 289/318 correct detections.",
            "Six-fold sample-aligned fixed-threshold audit.",
        ),
        (
            "supported",
            "BC-DPG-FCN v3 and ROI RI4 have complementary target outcomes and partially overlapping false alarms.",
            "The pooled overlap is reported as counts, without selecting a new rule.",
        ),
        (
            "supported",
            "ROI refinement is an independent suppression study under frozen Power2 candidate locations.",
            "ROI results are not presented as a trained joint detector.",
        ),
        (
            "diagnostic_only",
            "AND and OR counts describe logical intersections and unions only.",
            "Neither rule was selected on training or validation data for this audit.",
        ),
        (
            "not_supported",
            "A naive AND, OR, or serial combination improves the deployed detector.",
            "Choosing such a rule from these test outcomes would be test-set selection.",
        ),
        (
            "not_supported",
            "The evidence represents a trained or learned ROI/BC-DPG joint model.",
            "No joint model was trained in this evidence build.",
        ),
        (
            "not_supported",
            "The results establish cross-site blind generalization or balloon-payload classification.",
            "The current scope is an internal six-fold H/V UAV detection and localization front end.",
        ),
        (
            "method_constraint",
            "Any future learned combination must be selected using training or validation data and tested once with frozen rules.",
            "Test thresholds and combination rules remain untuned in this audit.",
        ),
    ]
    return pd.DataFrame(rows, columns=["boundary", "statement", "basis"])


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for _, row in view.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.compression": 9,
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> list[str]:
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "balloon_radar_project"},
    )
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Creator": "balloon_radar_project",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(fig)
    return [f"figures/{png.name}", f"figures/{pdf.name}"]


def build_figures(
    pooled: pd.DataFrame,
    fold_detail: pd.DataFrame,
    complementarity: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    configure_plots()
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for _, row in pooled.iterrows():
        model = str(row["model"])
        ax.scatter(
            row["false_alarms"],
            row["correct_detections"],
            s=86,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offsets = {
            "bc_dpg_v3": (7, -14),
            "roi_baseline": (-98, 7),
            "roi_power_control": (-102, -17),
            "roi_ri4": (7, 7),
        }
        ax.annotate(
            str(row["display_name"]),
            (row["false_alarms"], row["correct_detections"]),
            xytext=offsets[model],
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("False alarms across six folds (lower is better)")
    ax.set_ylabel("Correct detections out of 318 (higher is better)")
    ax.set_title("Frozen fixed-threshold detection trade-off")
    ax.set_xlim(0, 330)
    ax.set_ylim(255, 300)
    ax.grid(axis="both", color="#D9DDE2", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    fig.tight_layout()
    created.extend(save_figure(fig, figures_dir, "fig1_pooled_detection_tradeoff"))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    models = list(MODEL_COLUMNS)
    x = np.arange(len(EXPECTED_FOLDS), dtype=float)
    width = 0.19
    for index, model in enumerate(models):
        values = (
            fold_detail.loc[fold_detail["model"].eq(model)]
            .sort_values("fold")["false_alarms"]
            .to_numpy()
        )
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set_xticks(x, [f"Fold {fold}" for fold in EXPECTED_FOLDS])
    ax.set_ylabel("False alarms")
    ax.set_title("Fold-level false alarms at frozen thresholds")
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncols=2, loc="upper right")
    fig.tight_layout()
    created.extend(save_figure(fig, figures_dir, "fig2_fold_false_alarms"))

    selected = complementarity.loc[
        complementarity["fold"].eq("ALL")
        & complementarity["comparison"].eq("roi_ri4")
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    panels = [
        (
            axes[0],
            [
                int(selected["shared_false_alarms"]),
                int(selected["bc_only_false_alarms"]),
                int(selected["roi_only_false_alarms"]),
            ],
            ["Shared", "BC only", "ROI RI4 only"],
            ["#586F7C", MODEL_COLORS["bc_dpg_v3"], MODEL_COLORS["roi_ri4"]],
            "False-alarm union",
            "Background outcomes (216 union)",
        ),
        (
            axes[1],
            [
                int(selected["shared_correct"]),
                int(selected["bc_only_correct"]),
                int(selected["roi_only_correct"]),
                int(selected["neither_correct"]),
            ],
            ["Shared", "BC only", "ROI RI4 only", "Neither"],
            [
                "#586F7C",
                MODEL_COLORS["bc_dpg_v3"],
                MODEL_COLORS["roi_ri4"],
                "#C8CDD3",
            ],
            "Target outcome partition",
            "Target outcomes (318 total)",
        ),
    ]
    for axis, values, labels, colors, xlabel, title in panels:
        left = 0
        for value, label, color in zip(values, labels, colors):
            axis.barh([0], [value], left=left, color=color, height=0.48, label=label)
            if value >= 12:
                axis.text(
                    left + value / 2,
                    0,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if color != "#C8CDD3" else "#252A30",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                axis.annotate(
                    str(value),
                    (left + value / 2, 0.24),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            left += value
        axis.set_yticks([])
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2)
    fig.suptitle("BC-DPG-FCN v3 and ROI RI4 complementarity", y=1.02, fontsize=12)
    fig.tight_layout()
    created.extend(save_figure(fig, figures_dir, "fig3_complementarity"))
    return created


def build_report(
    data: AuditData,
    pooled: pd.DataFrame,
    complementarity: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    bc = pooled.loc[pooled["model"].eq("bc_dpg_v3")].iloc[0]
    ri4 = complementarity.loc[complementarity["comparison"].eq("roi_ri4")].iloc[0]
    lines = [
        "# Fixed-threshold ROI and BC-DPG joint audit: paper evidence",
        "",
        "## Evidence status",
        "",
        "This package is generated only from the authoritative frozen six-fold joint audit. "
        "All 1,148 test rows are aligned exactly by fold, label, sample ID, and MAT path. "
        "BC decisions come from `base_threshold_test_predictions.csv`; ROI decisions come "
        "from the frozen `refined_fixed_*` columns. Test thresholds were not retuned.",
        "",
        f"Source: `{display_path(data.input_dir)}/`",
        "",
        "## Pooled detector results",
        "",
        markdown_table(
            pooled,
            [
                "display_name",
                "false_alarms",
                "pfa",
                "correct_detections",
                "joint_pd",
            ],
        ),
        "",
        f"BC-DPG-FCN v3 remains the strongest current detector with "
        f"{int(bc['false_alarms'])} false alarms and "
        f"{int(bc['correct_detections'])}/318 correct detections.",
        "",
        "## BC-DPG and ROI RI4 complementarity",
        "",
        f"The two methods share {int(ri4['shared_false_alarms'])} false alarms. "
        f"BC-DPG contributes {int(ri4['bc_only_false_alarms'])} BC-only false alarms, "
        f"while ROI RI4 contributes {int(ri4['roi_only_false_alarms'])} ROI-only false alarms. "
        f"For targets, they share {int(ri4['shared_correct'])} correct detections; "
        f"{int(ri4['bc_only_correct'])} are BC-only and "
        f"{int(ri4['roi_only_correct'])} are ROI-only.",
        "",
        "## Simple logical-combination diagnostics",
        "",
        markdown_table(
            diagnostics,
            [
                "rule",
                "false_alarms",
                "correct_detections",
                "missed_targets",
                "selection_status",
            ],
        ),
        "",
        "The OR/union diagnostic recovers five targets missed by BC-DPG but raises false "
        "alarms from 56 to 216. The AND/intersection diagnostic reduces false alarms to "
        "36 but loses 26 BC-only correct detections. These counts are descriptive test-set "
        "diagnostics, not candidate rules selected for deployment.",
        "",
        "## Interpretation and claim boundary",
        "",
        "ROI remains an independent suppression study rather than a trained joint model. "
        "The fixed-threshold audit does not support selecting a naive AND, OR, or serial "
        "combination. Any future learned combination must be selected using training or "
        "validation data and evaluated once with frozen rules.",
        "",
        "The evidence supports an internal six-fold H/V UAV detection and localization "
        "front end. It does not establish balloon-payload classification, cross-site blind "
        "generalization, or strict real-time causal scan adaptation.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/build_roi_bc_dpg_joint_paper_assets.py",
        "```",
        "",
        "The build validates source status, row alignment, and prediction-derived metrics "
        "before writing tables or figures. `evidence_manifest.json` records SHA256 hashes "
        "for all source and generated artifacts.",
    ]
    return "\n".join(lines) + "\n"


def source_records(input_dir: Path) -> list[dict[str, Any]]:
    records = [
        {
            "category": "frozen_audit_input",
            "path": display_path(input_dir / name),
            "size_bytes": (input_dir / name).stat().st_size,
            "sha256": sha256_file(input_dir / name),
        }
        for name in SOURCE_FILES
    ]
    build_script = Path(__file__).resolve()
    records.append(
        {
            "category": "build_script",
            "path": display_path(build_script),
            "size_bytes": build_script.stat().st_size,
            "sha256": sha256_file(build_script),
        }
    )
    return records


def generated_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(output_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "evidence_manifest.json"
    ]


def write_assets(data: AuditData, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pooled = pooled_detection_table(data.detection)
    fold_detail = fold_detection_table(data.detection)
    complementarity = pooled_complementarity_table(data.complementarity)
    diagnostics = build_simple_combination_diagnostics(
        data.detection, data.complementarity
    )
    claims = build_claim_boundaries()

    tables = {
        "table_01_pooled_detection.csv": pooled,
        "table_02_fold_detection.csv": fold_detail,
        "table_03_complementarity.csv": complementarity,
        "table_04_simple_combination_diagnostics.csv": diagnostics,
        "table_05_claim_boundaries.csv": claims,
    }
    for name, frame in tables.items():
        frame.to_csv(tables_dir / name, index=False, lineterminator="\n")

    build_figures(pooled, fold_detail, complementarity, figures_dir)
    report = build_report(data, pooled, complementarity, diagnostics)
    (output_dir / "JOINT_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "evidence_scope": "fixed-threshold six-fold ROI/BC-DPG joint audit",
        "deterministic_build": True,
        "test_threshold_retuning": False,
        "joint_model_trained": False,
        "combination_selected": False,
        "folds": list(EXPECTED_FOLDS),
        "aligned_rows": EXPECTED_ROWS,
        "sources": source_records(data.input_dir),
        "generated_files": generated_records(output_dir),
    }
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Use --overwrite to replace it."
        )


def validate_nonoverlapping_paths(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if (
        input_dir == output_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError("Input and output directories must not overlap")


def build_package(input_dir: Path, output_dir: Path, overwrite: bool = False) -> Path:
    input_dir = resolve_path(input_dir)
    output_dir = resolve_path(output_dir)
    validate_nonoverlapping_paths(input_dir, output_dir)
    data = load_and_validate(input_dir)
    ensure_output_available(output_dir, overwrite)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    try:
        write_assets(data, staging)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        output_dir = build_package(args.input_dir, args.output_dir, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("Joint paper-evidence build: PASS")
    print(f"output_dir={display_path(output_dir)}")
    print(f"aligned_rows={EXPECTED_ROWS}")
    print("test_threshold_retuning=False")
    print("combination_selected=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
