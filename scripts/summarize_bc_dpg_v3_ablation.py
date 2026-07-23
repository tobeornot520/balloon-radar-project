#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = (
    "full",
    "no_scan_context",
    "no_background_classification",
    "no_background_tail",
    "no_target_protection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BC-DPG-FCN v3 ablation results."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=(
            "results/data_audit/"
            "bc_dpg_v3_ablation"
        ),
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = math.nan,
) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def experiment_name(
    fold: int,
    mode: str,
    seed: int,
    smoke: bool,
) -> str:
    suffix = "_smoke" if smoke else ""
    return (
        f"bc_dpg_v3_ablation_{mode}_"
        f"v4_fold{fold:02d}_seed{seed}{suffix}"
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for fold in args.folds:
        for mode in args.modes:
            name = experiment_name(
                fold,
                mode,
                args.seed,
                args.smoke,
            )
            summary_path = (
                PROJECT_ROOT
                / "results"
                / "experiments"
                / name
                / "tables"
                / "summary.json"
            )
            if not summary_path.is_file():
                missing.append(str(summary_path))
                continue

            data = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            calibrated = get_nested(
                data,
                "base_threshold_test_metrics",
                default={},
            )
            raw = get_nested(
                data,
                "raw_base_threshold_test_metrics",
                default={},
            )
            cal_fa = float(
                calibrated.get(
                    "false_alarm_count",
                    math.nan,
                )
            )
            raw_fa = float(
                raw.get(
                    "false_alarm_count",
                    math.nan,
                )
            )
            fa_reduction = raw_fa - cal_fa
            fa_reduction_rate = (
                fa_reduction / raw_fa
                if raw_fa > 0
                else math.nan
            )

            rows.append(
                {
                    "fold": fold,
                    "mode": mode,
                    "experiment_name": name,
                    "best_epoch": data.get(
                        "best_epoch",
                        math.nan,
                    ),
                    "base_threshold": data.get(
                        "base_threshold",
                        math.nan,
                    ),
                    "raw_false_alarms": raw_fa,
                    "calibrated_false_alarms": cal_fa,
                    "false_alarm_reduction": fa_reduction,
                    "false_alarm_reduction_rate": (
                        fa_reduction_rate
                    ),
                    "raw_test_pfa": raw.get(
                        "pfa",
                        math.nan,
                    ),
                    "calibrated_test_pfa": calibrated.get(
                        "pfa",
                        math.nan,
                    ),
                    "raw_test_pd": raw.get(
                        "joint_pd",
                        math.nan,
                    ),
                    "calibrated_test_pd": calibrated.get(
                        "joint_pd",
                        math.nan,
                    ),
                    "delta_pd": (
                        float(
                            calibrated.get(
                                "joint_pd",
                                math.nan,
                            )
                        )
                        - float(
                            raw.get(
                                "joint_pd",
                                math.nan,
                            )
                        )
                    ),
                    "raw_test_auc": raw.get(
                        "roc_auc",
                        math.nan,
                    ),
                    "calibrated_test_auc": calibrated.get(
                        "roc_auc",
                        math.nan,
                    ),
                    "background_shift_mean": get_nested(
                        data,
                        "shift_statistics_test",
                        "background_mean",
                    ),
                    "target_shift_mean": get_nested(
                        data,
                        "shift_statistics_test",
                        "target_mean",
                    ),
                    "pd_floor_satisfied": get_nested(
                        data,
                        "pd_floor_validation",
                        "satisfied",
                        default=False,
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    detail_path = output_dir / (
        "ablation_detail_smoke.csv"
        if args.smoke
        else "ablation_detail_formal.csv"
    )
    frame.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    if not frame.empty:
        numeric_columns = [
            "raw_false_alarms",
            "calibrated_false_alarms",
            "false_alarm_reduction",
            "false_alarm_reduction_rate",
            "raw_test_pfa",
            "calibrated_test_pfa",
            "raw_test_pd",
            "calibrated_test_pd",
            "delta_pd",
            "raw_test_auc",
            "calibrated_test_auc",
            "background_shift_mean",
            "target_shift_mean",
        ]
        grouped = frame.groupby(
            "mode",
            sort=False,
        )[numeric_columns].agg(["mean", "std", "sum"])
        grouped.columns = [
            f"{name}_{stat}"
            for name, stat in grouped.columns
        ]
        grouped = grouped.reset_index()
    else:
        grouped = pd.DataFrame()

    aggregate_path = output_dir / (
        "ablation_aggregate_smoke.csv"
        if args.smoke
        else "ablation_aggregate_formal.csv"
    )
    grouped.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    md_path = output_dir / (
        "README_消融结果_smoke.md"
        if args.smoke
        else "README_消融结果_formal.md"
    )
    lines = [
        "# BC-DPG-FCN v3 消融结果",
        "",
        "主比较口径：沿用各折原始 DPG 部署阈值，"
        "不使用测试集重新调阈值。",
        "",
    ]
    if frame.empty:
        lines.append("未发现可汇总的实验结果。")
    else:
        display_columns = [
            "fold",
            "mode",
            "raw_false_alarms",
            "calibrated_false_alarms",
            "false_alarm_reduction",
            "raw_test_pd",
            "calibrated_test_pd",
            "raw_test_pfa",
            "calibrated_test_pfa",
        ]
        lines.append(
            frame[display_columns].to_markdown(
                index=False,
                floatfmt=".4f",
            )
        )
        lines.extend(
            [
                "",
                "## 判读原则",
                "",
                "1. 首先检查 calibrated_test_pd 是否下降；",
                "2. 在目标检测率保持时比较虚警数和 Pfa；",
                "3. no_scan_context 仅将12维扫描组特征置零，"
                "网络参数量不变；",
                "4. no_target_protection 同时关闭 target keep、"
                "pairwise 和 shift selectivity 三项显式保护约束；",
                "5. smoke结果只用于检查接口，不作为论文结论。",
            ]
        )

    if missing:
        lines.extend(
            [
                "",
                "## 未找到的结果",
                "",
                *[f"- `{item}`" for item in missing],
            ]
        )
    md_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 82)
    print("BC-DPG-FCN v3 ablation summary complete")
    print(f"detail    : {detail_path}")
    print(f"aggregate : {aggregate_path}")
    print(f"report    : {md_path}")
    print(f"found     : {len(frame)}")
    print(f"missing   : {len(missing)}")
    print("=" * 82)


if __name__ == "__main__":
    main()
