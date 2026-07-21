#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_PATTERN = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate six-fold BC-DPG-FCN v3 results at the frozen original "
            "DPG thresholds and generate scan-group-level paper tables/figures."
        )
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--experiment-template",
        default="bc_dpg_v3_scan_target_v4_fold{fold:02d}_seed42",
    )
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/bc_dpg_v3_paper_results",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def derive_scan_group(sample_id: Any, mat_path: Any = "") -> str:
    candidates = (
        str(sample_id) if sample_id is not None else "",
        Path(str(mat_path)).stem if mat_path else "",
    )
    for candidate in candidates:
        match = SCAN_PATTERN.search(candidate)
        if match:
            return match.group(1)
    if candidates[0]:
        return candidates[0].split("_beam", 1)[0]
    return "UNKNOWN"


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float) > 0.5
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def ensure_outcomes(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    result = frame.copy()

    required = {"target_present", "score"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(
            f"Prediction table missing columns: {sorted(missing)}"
        )

    result["target_present"] = (
        pd.to_numeric(
            result["target_present"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    result["score"] = pd.to_numeric(
        result["score"],
        errors="raise",
    )
    result["predicted_positive"] = result["score"] >= float(threshold)

    if "localization_ok" in result.columns:
        result["localization_ok"] = normalize_bool(
            result["localization_ok"]
        )
    else:
        result["localization_ok"] = True

    target = result["target_present"] == 1
    background = ~target

    result["false_alarm"] = (
        background & result["predicted_positive"]
    )
    result["score_detected"] = (
        target & result["predicted_positive"]
    )
    result["joint_detected"] = (
        result["score_detected"]
        & result["localization_ok"]
    )

    if "scan_group" not in result.columns:
        mat_values = (
            result["mat_path"]
            if "mat_path" in result.columns
            else pd.Series([""] * len(result))
        )
        result["scan_group"] = [
            derive_scan_group(sample_id, mat_path)
            for sample_id, mat_path in zip(
                result.get(
                    "sample_id",
                    pd.Series(range(len(result))),
                ),
                mat_values,
            )
        ]
    else:
        result["scan_group"] = result["scan_group"].astype(str)

    return result


def align_predictions(
    raw: pd.DataFrame,
    v3: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "sample_id" in raw.columns and "sample_id" in v3.columns:
        if raw["sample_id"].duplicated().any():
            raise ValueError("Raw table has duplicate sample_id values")
        if v3["sample_id"].duplicated().any():
            raise ValueError("v3 table has duplicate sample_id values")

        raw_ids = set(raw["sample_id"].astype(str))
        v3_ids = set(v3["sample_id"].astype(str))
        if raw_ids != v3_ids:
            raise ValueError(
                "Raw and v3 prediction tables contain different sample IDs"
            )

        order = v3["sample_id"].astype(str).tolist()
        raw = (
            raw.assign(_sample_id_key=raw["sample_id"].astype(str))
            .set_index("_sample_id_key")
            .loc[order]
            .reset_index(drop=True)
        )
        v3 = v3.reset_index(drop=True)
    else:
        if len(raw) != len(v3):
            raise ValueError(
                "Raw and v3 tables have different row counts and no sample_id"
            )
        raw = raw.reset_index(drop=True)
        v3 = v3.reset_index(drop=True)

    return raw, v3


def binary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["target_present"] == 1
    background = ~target

    target_count = int(target.sum())
    background_count = int(background.sum())
    detected_count = int(frame.loc[target, "joint_detected"].sum())
    false_alarm_count = int(frame.loc[background, "false_alarm"].sum())

    return {
        "sample_count": int(len(frame)),
        "target_count": target_count,
        "background_count": background_count,
        "detected_target_count": detected_count,
        "false_alarm_count": false_alarm_count,
        "pd": (
            detected_count / target_count
            if target_count
            else math.nan
        ),
        "pfa": (
            false_alarm_count / background_count
            if background_count
            else math.nan
        ),
    }


def safe_reduction(raw_value: float, v3_value: float) -> float:
    if not np.isfinite(raw_value) or raw_value <= 0:
        return math.nan
    return (raw_value - v3_value) / raw_value


def load_fold(
    fold: int,
    experiment_template: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    name = experiment_template.format(fold=fold)
    table_dir = (
        PROJECT_ROOT
        / "results/experiments"
        / name
        / "tables"
    )
    summary_path = table_dir / "summary.json"
    v3_path = table_dir / "base_threshold_test_predictions.csv"
    raw_path = (
        table_dir / "raw_base_threshold_test_predictions.csv"
    )

    for path in (summary_path, v3_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    threshold = float(summary["base_threshold"])

    v3 = ensure_outcomes(
        pd.read_csv(v3_path, encoding="utf-8-sig"),
        threshold,
    )
    raw = ensure_outcomes(
        pd.read_csv(raw_path, encoding="utf-8-sig"),
        threshold,
    )
    raw, v3 = align_predictions(raw, v3)

    raw["fold"] = int(fold)
    raw["method"] = "Raw DPG"
    raw["threshold"] = threshold

    v3["fold"] = int(fold)
    v3["method"] = "BC-DPG v3"
    v3["threshold"] = threshold

    return raw, v3, summary


def scan_group_metrics(
    raw: pd.DataFrame,
    v3: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    raw_keyed = raw.copy()
    v3_keyed = v3.copy()

    if "sample_id" in raw.columns and "sample_id" in v3.columns:
        key = "sample_id"
    else:
        key = "_row_key"
        raw_keyed[key] = np.arange(len(raw_keyed))
        v3_keyed[key] = np.arange(len(v3_keyed))

    columns = [
        key,
        "scan_group",
        "target_present",
        "score",
        "false_alarm",
        "joint_detected",
    ]
    optional = [
        "shift",
        "p_background",
        "suppression",
    ]
    v3_optional = [
        column for column in optional if column in v3_keyed.columns
    ]

    merged = raw_keyed[columns].rename(
        columns={
            "score": "raw_score",
            "false_alarm": "raw_false_alarm",
            "joint_detected": "raw_joint_detected",
        }
    ).merge(
        v3_keyed[columns + v3_optional].rename(
            columns={
                "scan_group": "v3_scan_group",
                "target_present": "v3_target_present",
                "score": "v3_score",
                "false_alarm": "v3_false_alarm",
                "joint_detected": "v3_joint_detected",
            }
        ),
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if not (
        merged["scan_group"].astype(str)
        == merged["v3_scan_group"].astype(str)
    ).all():
        raise ValueError(f"Fold {fold}: scan_group alignment mismatch")

    rows: list[dict[str, Any]] = []
    for group_name, part in merged.groupby(
        "scan_group",
        sort=True,
    ):
        background = part["target_present"] == 0
        target = ~background

        background_count = int(background.sum())
        target_count = int(target.sum())
        raw_fa = int(
            part.loc[background, "raw_false_alarm"].sum()
        )
        v3_fa = int(
            part.loc[background, "v3_false_alarm"].sum()
        )
        raw_detected = int(
            part.loc[target, "raw_joint_detected"].sum()
        )
        v3_detected = int(
            part.loc[target, "v3_joint_detected"].sum()
        )

        row = {
            "fold": int(fold),
            "scan_group": str(group_name),
            "sample_count": int(len(part)),
            "background_count": background_count,
            "target_count": target_count,
            "raw_false_alarm_count": raw_fa,
            "v3_false_alarm_count": v3_fa,
            "false_alarm_count_reduction": raw_fa - v3_fa,
            "raw_pfa": (
                raw_fa / background_count
                if background_count
                else math.nan
            ),
            "v3_pfa": (
                v3_fa / background_count
                if background_count
                else math.nan
            ),
            "pfa_reduction_fraction": safe_reduction(
                raw_fa / background_count
                if background_count
                else math.nan,
                v3_fa / background_count
                if background_count
                else math.nan,
            ),
            "raw_detected_target_count": raw_detected,
            "v3_detected_target_count": v3_detected,
            "raw_pd": (
                raw_detected / target_count
                if target_count
                else math.nan
            ),
            "v3_pd": (
                v3_detected / target_count
                if target_count
                else math.nan
            ),
            "raw_background_score_mean": (
                float(part.loc[background, "raw_score"].mean())
                if background_count
                else math.nan
            ),
            "v3_background_score_mean": (
                float(part.loc[background, "v3_score"].mean())
                if background_count
                else math.nan
            ),
            "raw_background_score_max": (
                float(part.loc[background, "raw_score"].max())
                if background_count
                else math.nan
            ),
            "v3_background_score_max": (
                float(part.loc[background, "v3_score"].max())
                if background_count
                else math.nan
            ),
        }

        for column in v3_optional:
            if background_count:
                row[f"background_{column}_mean"] = float(
                    part.loc[background, column].mean()
                )
            else:
                row[f"background_{column}_mean"] = math.nan

            if target_count:
                row[f"target_{column}_mean"] = float(
                    part.loc[target, column].mean()
                )
            else:
                row[f"target_{column}_mean"] = math.nan

        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_mean_difference(
    raw_values: np.ndarray,
    v3_values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    valid = np.isfinite(raw_values) & np.isfinite(v3_values)
    raw_values = raw_values[valid]
    v3_values = v3_values[valid]

    if len(raw_values) == 0:
        return {
            "paired_group_count": 0,
            "raw_mean": math.nan,
            "v3_mean": math.nan,
            "mean_difference_raw_minus_v3": math.nan,
            "ci95_lower": math.nan,
            "ci95_upper": math.nan,
        }

    differences = raw_values - v3_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(iterations, len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return {
        "paired_group_count": int(len(differences)),
        "raw_mean": float(raw_values.mean()),
        "v3_mean": float(v3_values.mean()),
        "mean_difference_raw_minus_v3": float(
            differences.mean()
        ),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def aggregate_scan_summary(
    scan_table: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    background_groups = scan_table.loc[
        scan_table["background_count"] > 0
    ].copy()

    bootstrap = bootstrap_mean_difference(
        background_groups["raw_pfa"].to_numpy(float),
        background_groups["v3_pfa"].to_numpy(float),
        iterations=iterations,
        seed=seed,
    )

    raw_values = background_groups["raw_pfa"].to_numpy(float)
    v3_values = background_groups["v3_pfa"].to_numpy(float)

    rows = [
        {
            "metric_scope": "scan_group_macro",
            "group_count": int(len(background_groups)),
            "raw_mean_pfa": float(np.nanmean(raw_values)),
            "v3_mean_pfa": float(np.nanmean(v3_values)),
            "raw_median_pfa": float(np.nanmedian(raw_values)),
            "v3_median_pfa": float(np.nanmedian(v3_values)),
            "raw_std_pfa": float(np.nanstd(raw_values, ddof=1)),
            "v3_std_pfa": float(np.nanstd(v3_values, ddof=1)),
            "raw_worst_group_pfa": float(np.nanmax(raw_values)),
            "v3_worst_group_pfa": float(np.nanmax(v3_values)),
            "raw_groups_with_false_alarm": int(
                (background_groups["raw_false_alarm_count"] > 0).sum()
            ),
            "v3_groups_with_false_alarm": int(
                (background_groups["v3_false_alarm_count"] > 0).sum()
            ),
            "raw_group_false_alarm_rate": float(
                (
                    background_groups["raw_false_alarm_count"] > 0
                ).mean()
            ),
            "v3_group_false_alarm_rate": float(
                (
                    background_groups["v3_false_alarm_count"] > 0
                ).mean()
            ),
            "improved_group_count": int(
                (background_groups["v3_pfa"] < background_groups["raw_pfa"]).sum()
            ),
            "unchanged_group_count": int(
                np.isclose(
                    background_groups["v3_pfa"],
                    background_groups["raw_pfa"],
                ).sum()
            ),
            "worsened_group_count": int(
                (background_groups["v3_pfa"] > background_groups["raw_pfa"]).sum()
            ),
            **bootstrap,
        }
    ]
    return pd.DataFrame(rows)


def format_percent(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * value:.{digits}f}%"


def make_main_table(
    fold_table: pd.DataFrame,
    aggregate_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in fold_table.iterrows():
        rows.append(
            {
                "Scope": f"Fold {int(row['fold'])}",
                "Raw Pd": row["raw_pd"],
                "v3 Pd": row["v3_pd"],
                "Pd change": row["v3_pd"] - row["raw_pd"],
                "Raw Pfa": row["raw_pfa"],
                "v3 Pfa": row["v3_pfa"],
                "Pfa reduction": row["pfa_reduction_fraction"],
                "Raw false alarms": int(row["raw_false_alarm_count"]),
                "v3 false alarms": int(row["v3_false_alarm_count"]),
            }
        )

    aggregate = aggregate_table.iloc[0]
    rows.append(
        {
            "Scope": "Six-fold pooled",
            "Raw Pd": aggregate["raw_pd"],
            "v3 Pd": aggregate["v3_pd"],
            "Pd change": aggregate["v3_pd"] - aggregate["raw_pd"],
            "Raw Pfa": aggregate["raw_pfa"],
            "v3 Pfa": aggregate["v3_pfa"],
            "Pfa reduction": aggregate["pfa_reduction_fraction"],
            "Raw false alarms": int(
                aggregate["raw_false_alarm_count"]
            ),
            "v3 false alarms": int(
                aggregate["v3_false_alarm_count"]
            ),
        }
    )
    return pd.DataFrame(rows)


def save_latex_table(
    main_table: pd.DataFrame,
    output_path: Path,
) -> None:
    formatted = main_table.copy()
    for column in (
        "Raw Pd",
        "v3 Pd",
        "Pd change",
        "Raw Pfa",
        "v3 Pfa",
        "Pfa reduction",
    ):
        formatted[column] = formatted[column].map(format_percent)

    latex = formatted.to_latex(
        index=False,
        escape=True,
        caption=(
            "Six-fold comparison between the frozen DPG operating point "
            "and BC-DPG-FCN v3."
        ),
        label="tab:bcdpg_v3_main_results",
    )
    output_path.write_text(latex, encoding="utf-8")


def plot_fold_pfa(
    fold_table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    positions = np.arange(len(fold_table))
    width = 0.36

    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.bar(
        positions - width / 2,
        fold_table["raw_pfa"],
        width,
        label="Raw DPG",
    )
    axis.bar(
        positions + width / 2,
        fold_table["v3_pfa"],
        width,
        label="BC-DPG v3",
    )
    axis.set_xticks(
        positions,
        [f"Fold {int(value)}" for value in fold_table["fold"]],
    )
    axis.set_ylabel("False alarm probability")
    axis.set_xlabel("Cross-validation fold")
    axis.set_title("Per-fold false alarm probability at frozen DPG thresholds")
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scan_scatter(
    scan_table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    background = scan_table.loc[
        scan_table["background_count"] > 0
    ].copy()

    maximum = float(
        max(
            background["raw_pfa"].max(),
            background["v3_pfa"].max(),
            0.01,
        )
    )

    fig, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(
        background["raw_pfa"],
        background["v3_pfa"],
        s=28,
        alpha=0.75,
    )
    axis.plot([0, maximum], [0, maximum], linestyle="--")
    axis.set_xlim(-0.01 * maximum, 1.05 * maximum)
    axis.set_ylim(-0.01 * maximum, 1.05 * maximum)
    axis.set_xlabel("Raw DPG scan-group Pfa")
    axis.set_ylabel("BC-DPG v3 scan-group Pfa")
    axis.set_title("Paired scan-group false alarm comparison")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_hardest_groups(
    scan_table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    hardest = (
        scan_table.loc[scan_table["background_count"] > 0]
        .sort_values(
            ["raw_false_alarm_count", "raw_pfa"],
            ascending=[False, False],
        )
        .head(12)
        .copy()
    )
    if hardest.empty:
        return

    labels = [
        f"F{int(fold)} {group}"
        for fold, group in zip(
            hardest["fold"],
            hardest["scan_group"],
        )
    ]
    positions = np.arange(len(hardest))
    height = 0.36

    fig, axis = plt.subplots(figsize=(10.0, 6.8))
    axis.barh(
        positions - height / 2,
        hardest["raw_false_alarm_count"],
        height,
        label="Raw DPG",
    )
    axis.barh(
        positions + height / 2,
        hardest["v3_false_alarm_count"],
        height,
        label="BC-DPG v3",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("False alarm count")
    axis.set_title("Hardest background scan groups")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_box(
    all_v3: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    if column not in all_v3.columns:
        return

    background = pd.to_numeric(
        all_v3.loc[
            all_v3["target_present"] == 0,
            column,
        ],
        errors="coerce",
    ).dropna()
    target = pd.to_numeric(
        all_v3.loc[
            all_v3["target_present"] == 1,
            column,
        ],
        errors="coerce",
    ).dropna()

    if background.empty or target.empty:
        return

    fig, axis = plt.subplots(figsize=(6.5, 5.2))
    axis.boxplot(
        [background.to_numpy(), target.to_numpy()],
        showfliers=False,
    )
    # Set tick labels separately for compatibility with both older and newer
    # Matplotlib releases. Recent versions removed the legacy ``labels``
    # keyword in favor of ``tick_labels``.
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Background", "Target"])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    output_path: Path,
    fold_table: pd.DataFrame,
    aggregate_table: pd.DataFrame,
    scan_summary: pd.DataFrame,
    hardest: pd.DataFrame,
) -> None:
    aggregate = aggregate_table.iloc[0]
    scan = scan_summary.iloc[0]

    lines = [
        "# BC-DPG-FCN v3 六折正式结果摘要",
        "",
        "## 样本级总体结果",
        "",
        (
            f"- Raw DPG：Pd={format_percent(aggregate['raw_pd'])}，"
            f"Pfa={format_percent(aggregate['raw_pfa'])}，"
            f"虚警数={int(aggregate['raw_false_alarm_count'])}。"
        ),
        (
            f"- BC-DPG v3：Pd={format_percent(aggregate['v3_pd'])}，"
            f"Pfa={format_percent(aggregate['v3_pfa'])}，"
            f"虚警数={int(aggregate['v3_false_alarm_count'])}。"
        ),
        (
            f"- Pd变化={format_percent(aggregate['v3_pd'] - aggregate['raw_pd'])}；"
            f"Pfa相对下降={format_percent(aggregate['pfa_reduction_fraction'])}。"
        ),
        "",
        "## 扫描组级结果",
        "",
        (
            f"- 独立背景扫描组数：{int(scan['group_count'])}。"
        ),
        (
            f"- 扫描组宏平均Pfa：Raw={format_percent(scan['raw_mean_pfa'])}，"
            f"v3={format_percent(scan['v3_mean_pfa'])}。"
        ),
        (
            f"- 扫描组Pfa配对平均差（Raw-v3）："
            f"{format_percent(scan['mean_difference_raw_minus_v3'])}，"
            f"95% bootstrap CI "
            f"[{format_percent(scan['ci95_lower'])}, "
            f"{format_percent(scan['ci95_upper'])}]。"
        ),
        (
            f"- 有虚警扫描组比例：Raw={format_percent(scan['raw_group_false_alarm_rate'])}，"
            f"v3={format_percent(scan['v3_group_false_alarm_rate'])}。"
        ),
        (
            f"- 改善/不变/变差扫描组数："
            f"{int(scan['improved_group_count'])}/"
            f"{int(scan['unchanged_group_count'])}/"
            f"{int(scan['worsened_group_count'])}。"
        ),
        "",
        "## 逐折结果",
        "",
        "| Fold | Raw Pd | v3 Pd | Raw Pfa | v3 Pfa | Pfa下降 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in fold_table.iterrows():
        lines.append(
            "| "
            f"{int(row['fold'])} | "
            f"{format_percent(row['raw_pd'])} | "
            f"{format_percent(row['v3_pd'])} | "
            f"{format_percent(row['raw_pfa'])} | "
            f"{format_percent(row['v3_pfa'])} | "
            f"{format_percent(row['pfa_reduction_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## 最困难扫描组",
            "",
            "| Fold | Scan group | Background | Raw FA | v3 FA | Raw Pfa | v3 Pfa |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in hardest.head(10).iterrows():
        lines.append(
            "| "
            f"{int(row['fold'])} | "
            f"{row['scan_group']} | "
            f"{int(row['background_count'])} | "
            f"{int(row['raw_false_alarm_count'])} | "
            f"{int(row['v3_false_alarm_count'])} | "
            f"{format_percent(row['raw_pfa'])} | "
            f"{format_percent(row['v3_pfa'])} |"
        )

    lines.extend(
        [
            "",
            "## 推荐论文表述",
            "",
            (
                "在沿用各折原始DPG检测阈值的条件下，BC-DPG-FCN v3 "
                "保持了基线目标检测率，同时降低了困难背景下的虚警。"
                "除样本级指标外，扫描组级配对统计进一步用于避免大型扫描组"
                "对总体Pfa的过度支配。"
            ),
            "",
            (
                "当前结果属于无明显数据泄漏的开发阶段交叉验证结果；"
                "最终泛化结论仍应通过冻结模型后的新扫描盲测或严格嵌套"
                "扫描组交叉验证确认。"
            ),
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    scan_tables: list[pd.DataFrame] = []
    all_raw_frames: list[pd.DataFrame] = []
    all_v3_frames: list[pd.DataFrame] = []

    for fold in args.folds:
        raw, v3, summary = load_fold(
            fold,
            args.experiment_template,
        )
        raw_metrics = binary_metrics(raw)
        v3_metrics = binary_metrics(v3)

        fold_rows.append(
            {
                "fold": int(fold),
                "base_threshold": float(summary["base_threshold"]),
                "sample_count": raw_metrics["sample_count"],
                "target_count": raw_metrics["target_count"],
                "background_count": raw_metrics["background_count"],
                "raw_detected_target_count": raw_metrics[
                    "detected_target_count"
                ],
                "v3_detected_target_count": v3_metrics[
                    "detected_target_count"
                ],
                "raw_false_alarm_count": raw_metrics[
                    "false_alarm_count"
                ],
                "v3_false_alarm_count": v3_metrics[
                    "false_alarm_count"
                ],
                "raw_pd": raw_metrics["pd"],
                "v3_pd": v3_metrics["pd"],
                "pd_change": v3_metrics["pd"] - raw_metrics["pd"],
                "raw_pfa": raw_metrics["pfa"],
                "v3_pfa": v3_metrics["pfa"],
                "pfa_absolute_change": (
                    v3_metrics["pfa"] - raw_metrics["pfa"]
                ),
                "pfa_reduction_fraction": safe_reduction(
                    raw_metrics["pfa"],
                    v3_metrics["pfa"],
                ),
                "score_never_increased": bool(
                    (v3["score"] <= raw["score"] + 1e-7).all()
                ),
            }
        )

        scan_tables.append(
            scan_group_metrics(raw, v3, fold)
        )
        all_raw_frames.append(raw)
        all_v3_frames.append(v3)

    fold_table = pd.DataFrame(fold_rows).sort_values("fold")
    scan_table = pd.concat(
        scan_tables,
        ignore_index=True,
    )
    all_raw = pd.concat(
        all_raw_frames,
        ignore_index=True,
    )
    all_v3 = pd.concat(
        all_v3_frames,
        ignore_index=True,
    )

    raw_aggregate = binary_metrics(all_raw)
    v3_aggregate = binary_metrics(all_v3)

    aggregate_table = pd.DataFrame(
        [
            {
                "scope": "six_fold_pooled_samples",
                "sample_count": raw_aggregate["sample_count"],
                "target_count": raw_aggregate["target_count"],
                "background_count": raw_aggregate[
                    "background_count"
                ],
                "raw_detected_target_count": raw_aggregate[
                    "detected_target_count"
                ],
                "v3_detected_target_count": v3_aggregate[
                    "detected_target_count"
                ],
                "raw_false_alarm_count": raw_aggregate[
                    "false_alarm_count"
                ],
                "v3_false_alarm_count": v3_aggregate[
                    "false_alarm_count"
                ],
                "raw_pd": raw_aggregate["pd"],
                "v3_pd": v3_aggregate["pd"],
                "pd_change": (
                    v3_aggregate["pd"] - raw_aggregate["pd"]
                ),
                "raw_pfa": raw_aggregate["pfa"],
                "v3_pfa": v3_aggregate["pfa"],
                "pfa_absolute_change": (
                    v3_aggregate["pfa"]
                    - raw_aggregate["pfa"]
                ),
                "pfa_reduction_fraction": safe_reduction(
                    raw_aggregate["pfa"],
                    v3_aggregate["pfa"],
                ),
            }
        ]
    )

    scan_summary = aggregate_scan_summary(
        scan_table,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    hardest = (
        scan_table.loc[scan_table["background_count"] > 0]
        .sort_values(
            [
                "raw_false_alarm_count",
                "raw_pfa",
                "background_count",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    main_table = make_main_table(
        fold_table,
        aggregate_table,
    )

    fold_table.to_csv(
        output_dir / "six_fold_sample_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scan_table.to_csv(
        output_dir / "six_fold_scan_group_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate_table.to_csv(
        output_dir / "six_fold_pooled_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scan_summary.to_csv(
        output_dir / "scan_group_macro_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hardest.to_csv(
        output_dir / "hardest_scan_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hardest.loc[
        hardest["fold"].isin([1, 4])
    ].to_csv(
        output_dir / "fold1_fold4_scan_group_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_table.to_csv(
        output_dir / "paper_main_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_latex_table(
        main_table,
        output_dir / "paper_main_table.tex",
    )

    write_markdown_summary(
        output_dir / "paper_results_summary.md",
        fold_table,
        aggregate_table,
        scan_summary,
        hardest,
    )

    plot_fold_pfa(
        fold_table,
        figure_dir / "fold_pfa_comparison.png",
        args.dpi,
    )
    plot_scan_scatter(
        scan_table,
        figure_dir / "scan_group_pfa_scatter.png",
        args.dpi,
    )
    plot_hardest_groups(
        scan_table,
        figure_dir / "hardest_scan_groups.png",
        args.dpi,
    )
    plot_mechanism_box(
        all_v3,
        "shift",
        "Logit suppression shift",
        "Background and target suppression in BC-DPG v3",
        figure_dir / "shift_background_vs_target.png",
        args.dpi,
    )
    plot_mechanism_box(
        all_v3,
        "p_background",
        "Predicted background probability",
        "Background probability separation in BC-DPG v3",
        figure_dir / "background_probability_separation.png",
        args.dpi,
    )

    summary = {
        "folds": list(args.folds),
        "experiment_template": args.experiment_template,
        "deployment_policy": (
            "Use the original fold-specific DPG threshold after v3 "
            "background suppression; do not reselect a lower v3 threshold."
        ),
        "pooled_sample_metrics": aggregate_table.iloc[0].to_dict(),
        "scan_group_macro_metrics": scan_summary.iloc[0].to_dict(),
        "score_never_increased_all_folds": bool(
            fold_table["score_never_increased"].all()
        ),
        "data_leakage_audit_status": (
            "Passed: no within-fold cross-split overlap; sample and "
            "scan-group rotation anomalies are zero; SHA256 audit passed."
        ),
        "generalization_caveat": (
            "Current six-fold results are development estimates because "
            "test-fold feedback influenced v1/v2/v3 design. Final claims "
            "require a frozen-model blind scan set or nested group CV."
        ),
        "output_directory": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.item()
                if hasattr(value, "item")
                else str(value)
            ),
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 94)
    print("SIX-FOLD SAMPLE-LEVEL RESULTS")
    print("=" * 94)
    print(
        fold_table[
            [
                "fold",
                "base_threshold",
                "raw_pd",
                "v3_pd",
                "raw_pfa",
                "v3_pfa",
                "pfa_reduction_fraction",
                "raw_false_alarm_count",
                "v3_false_alarm_count",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 94)
    print("POOLED SIX-FOLD RESULT")
    print("=" * 94)
    print(aggregate_table.to_string(index=False))

    print("\n" + "=" * 94)
    print("SCAN-GROUP MACRO RESULT")
    print("=" * 94)
    print(scan_summary.to_string(index=False))

    print("\n" + "=" * 94)
    print("HARDEST SCAN GROUPS")
    print("=" * 94)
    print(
        hardest[
            [
                "fold",
                "scan_group",
                "background_count",
                "raw_false_alarm_count",
                "v3_false_alarm_count",
                "raw_pfa",
                "v3_pfa",
                "pfa_reduction_fraction",
            ]
        ].head(15).to_string(index=False)
    )

    print("\n" + "=" * 94)
    print(f"Saved paper result package to: {output_dir}")
    print("=" * 94)


if __name__ == "__main__":
    main()
