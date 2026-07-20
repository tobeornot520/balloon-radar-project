#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

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


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return float("nan")

    return numerator / denominator


def load_all_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for fold in range(1, 7):
        for model, pattern in (
            EXPERIMENT_PATTERNS.items()
        ):
            experiment_name = pattern.format(
                fold=fold
            )

            summary_path = (
                EXPERIMENT_ROOT
                / experiment_name
                / "tables"
                / "summary.json"
            )

            if not summary_path.is_file():
                raise FileNotFoundError(
                    "找不到实验汇总文件："
                    f"{summary_path}"
                )

            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

            val = summary[
                "validation_metrics"
            ]

            test = summary[
                "test_metrics"
            ]

            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "experiment_name": (
                        experiment_name
                    ),
                    "best_epoch": summary[
                        "best_epoch"
                    ],
                    "best_stage": summary.get(
                        "best_stage"
                    ),
                    "threshold": summary[
                        "validation_threshold"
                    ],

                    "val_positive_count": val[
                        "positive_count"
                    ],
                    "val_correct_detection_count": (
                        val[
                            "correct_detection_count"
                        ]
                    ),
                    "val_background_count": val[
                        "background_count"
                    ],
                    "val_false_alarm_count": val[
                        "false_alarm_count"
                    ],
                    "val_pd": val[
                        "joint_pd"
                    ],
                    "val_pfa": val[
                        "pfa"
                    ],
                    "val_auc": val[
                        "roc_auc"
                    ],

                    "test_positive_count": test[
                        "positive_count"
                    ],
                    "test_correct_detection_count": (
                        test[
                            "correct_detection_count"
                        ]
                    ),
                    "test_background_count": test[
                        "background_count"
                    ],
                    "test_false_alarm_count": test[
                        "false_alarm_count"
                    ],
                    "test_pd": test[
                        "joint_pd"
                    ],
                    "test_pfa": test[
                        "pfa"
                    ],
                    "test_auc": test[
                        "roc_auc"
                    ],

                    "delta_pd": (
                        test["joint_pd"]
                        - val["joint_pd"]
                    ),
                    "delta_pfa": (
                        test["pfa"]
                        - val["pfa"]
                    ),

                    "range_mae": test[
                        "all_positive_"
                        "range_mae_gates"
                    ],
                    "velocity_mae": test[
                        "all_positive_"
                        "velocity_mae_bins"
                    ],
                }
            )

    result = pd.DataFrame(rows)

    expected_rows = 6 * 4

    if len(result) != expected_rows:
        raise RuntimeError(
            "实验结果数量异常："
            f"期望{expected_rows}行，"
            f"实际{len(result)}行"
        )

    return result


def make_robust_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model, part in detail.groupby(
        "model",
        sort=True,
    ):
        positive_total = int(
            part[
                "test_positive_count"
            ].sum()
        )

        correct_total = int(
            part[
                "test_correct_detection_count"
            ].sum()
        )

        background_total = int(
            part[
                "test_background_count"
            ].sum()
        )

        false_alarm_total = int(
            part[
                "test_false_alarm_count"
            ].sum()
        )

        positive_weights = part[
            "test_positive_count"
        ].astype(float)

        range_mae_weighted = safe_divide(
            float(
                (
                    part["range_mae"]
                    * positive_weights
                ).sum()
            ),
            float(
                positive_weights.sum()
            ),
        )

        velocity_mae_weighted = safe_divide(
            float(
                (
                    part["velocity_mae"]
                    * positive_weights
                ).sum()
            ),
            float(
                positive_weights.sum()
            ),
        )

        both_conditions = (
            (part["test_pd"] >= 0.80)
            & (part["test_pfa"] <= 0.05)
        )

        rows.append(
            {
                "model": model,

                "test_pd_macro_mean": (
                    part["test_pd"].mean()
                ),
                "test_pd_macro_std": (
                    part["test_pd"].std()
                ),
                "test_pd_median": (
                    part["test_pd"].median()
                ),
                "test_pd_q1": (
                    part["test_pd"].quantile(
                        0.25
                    )
                ),
                "test_pd_q3": (
                    part["test_pd"].quantile(
                        0.75
                    )
                ),
                "test_pd_micro": safe_divide(
                    correct_total,
                    positive_total,
                ),

                "test_pfa_macro_mean": (
                    part["test_pfa"].mean()
                ),
                "test_pfa_macro_std": (
                    part["test_pfa"].std()
                ),
                "test_pfa_median": (
                    part["test_pfa"].median()
                ),
                "test_pfa_q1": (
                    part["test_pfa"].quantile(
                        0.25
                    )
                ),
                "test_pfa_q3": (
                    part["test_pfa"].quantile(
                        0.75
                    )
                ),
                "test_pfa_micro": safe_divide(
                    false_alarm_total,
                    background_total,
                ),

                "test_auc_mean": (
                    part["test_auc"].mean()
                ),
                "test_auc_median": (
                    part["test_auc"].median()
                ),
                "test_auc_min": (
                    part["test_auc"].min()
                ),

                "range_mae_weighted": (
                    range_mae_weighted
                ),
                "velocity_mae_weighted": (
                    velocity_mae_weighted
                ),

                "delta_pd_mean": (
                    part["delta_pd"].mean()
                ),
                "delta_pfa_mean": (
                    part["delta_pfa"].mean()
                ),

                "folds_pd_ge_0_80": int(
                    (
                        part["test_pd"]
                        >= 0.80
                    ).sum()
                ),
                "folds_pfa_le_0_05": int(
                    (
                        part["test_pfa"]
                        <= 0.05
                    ).sum()
                ),
                "folds_both_conditions": int(
                    both_conditions.sum()
                ),

                "worst_fold_pd": (
                    part["test_pd"].min()
                ),
                "worst_fold_pfa": (
                    part["test_pfa"].max()
                ),

                "pooled_positive_count": (
                    positive_total
                ),
                "pooled_correct_count": (
                    correct_total
                ),
                "pooled_background_count": (
                    background_total
                ),
                "pooled_false_alarm_count": (
                    false_alarm_total
                ),
            }
        )

    return pd.DataFrame(rows)


def make_paired_tables(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = detail.pivot(
        index="fold",
        columns="model",
        values=[
            "test_pd",
            "test_pfa",
            "test_auc",
            "range_mae",
            "velocity_mae",
        ],
    )

    paired_rows: list[
        dict[str, object]
    ] = []

    for baseline in (
        "H",
        "V",
        "HV",
    ):
        for fold in range(1, 7):
            dpg_pd = wide.loc[
                fold,
                ("test_pd", "DPG"),
            ]
            baseline_pd = wide.loc[
                fold,
                ("test_pd", baseline),
            ]

            dpg_pfa = wide.loc[
                fold,
                ("test_pfa", "DPG"),
            ]
            baseline_pfa = wide.loc[
                fold,
                ("test_pfa", baseline),
            ]

            dpg_auc = wide.loc[
                fold,
                ("test_auc", "DPG"),
            ]
            baseline_auc = wide.loc[
                fold,
                ("test_auc", baseline),
            ]

            dpg_range_mae = wide.loc[
                fold,
                ("range_mae", "DPG"),
            ]
            baseline_range_mae = wide.loc[
                fold,
                ("range_mae", baseline),
            ]

            dpg_velocity_mae = wide.loc[
                fold,
                (
                    "velocity_mae",
                    "DPG",
                ),
            ]
            baseline_velocity_mae = wide.loc[
                fold,
                (
                    "velocity_mae",
                    baseline,
                ),
            ]

            paired_rows.append(
                {
                    "fold": fold,
                    "comparison": (
                        f"DPG-{baseline}"
                    ),

                    # 正数表示DPG的Pd更高。
                    "delta_test_pd": (
                        dpg_pd
                        - baseline_pd
                    ),

                    # 负数表示DPG的Pfa更低。
                    "delta_test_pfa": (
                        dpg_pfa
                        - baseline_pfa
                    ),

                    # 正数表示DPG的AUC更高。
                    "delta_test_auc": (
                        dpg_auc
                        - baseline_auc
                    ),

                    # 负数表示DPG距离误差更小。
                    "delta_range_mae": (
                        dpg_range_mae
                        - baseline_range_mae
                    ),

                    # 负数表示DPG速度误差更小。
                    "delta_velocity_mae": (
                        dpg_velocity_mae
                        - baseline_velocity_mae
                    ),
                }
            )

    paired = pd.DataFrame(
        paired_rows
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for comparison, part in paired.groupby(
        "comparison",
        sort=True,
    ):
        summary_rows.append(
            {
                "comparison": comparison,

                "pd_gain_mean": (
                    part[
                        "delta_test_pd"
                    ].mean()
                ),
                "pd_gain_median": (
                    part[
                        "delta_test_pd"
                    ].median()
                ),
                "pd_better_fold_count": int(
                    (
                        part[
                            "delta_test_pd"
                        ]
                        > 0
                    ).sum()
                ),
                "pd_equal_fold_count": int(
                    (
                        part[
                            "delta_test_pd"
                        ]
                        == 0
                    ).sum()
                ),

                "pfa_change_mean": (
                    part[
                        "delta_test_pfa"
                    ].mean()
                ),
                "pfa_change_median": (
                    part[
                        "delta_test_pfa"
                    ].median()
                ),
                "pfa_lower_fold_count": int(
                    (
                        part[
                            "delta_test_pfa"
                        ]
                        < 0
                    ).sum()
                ),
                "pfa_equal_fold_count": int(
                    (
                        part[
                            "delta_test_pfa"
                        ]
                        == 0
                    ).sum()
                ),

                "auc_gain_mean": (
                    part[
                        "delta_test_auc"
                    ].mean()
                ),
                "auc_better_fold_count": int(
                    (
                        part[
                            "delta_test_auc"
                        ]
                        > 0
                    ).sum()
                ),

                "range_mae_change_mean": (
                    part[
                        "delta_range_mae"
                    ].mean()
                ),
                "range_better_fold_count": int(
                    (
                        part[
                            "delta_range_mae"
                        ]
                        < 0
                    ).sum()
                ),

                "velocity_mae_change_mean": (
                    part[
                        "delta_velocity_mae"
                    ].mean()
                ),
                "velocity_better_fold_count": int(
                    (
                        part[
                            "delta_velocity_mae"
                        ]
                        < 0
                    ).sum()
                ),
            }
        )

    paired_summary = pd.DataFrame(
        summary_rows
    )

    return paired, paired_summary


def save_results(
    detail: pd.DataFrame,
    robust_summary: pd.DataFrame,
    paired: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    detail.to_csv(
        OUTPUT_ROOT
        / "all_fold_model_results_robust.csv",
        index=False,
        encoding="utf-8-sig",
    )

    robust_summary.to_csv(
        OUTPUT_ROOT
        / "robust_six_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        OUTPUT_ROOT
        / "paired_dpg_fold_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_summary.to_csv(
        OUTPUT_ROOT
        / "paired_dpg_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    detail = load_all_results()

    robust_summary = make_robust_summary(
        detail
    )

    paired, paired_summary = (
        make_paired_tables(detail)
    )

    save_results(
        detail=detail,
        robust_summary=robust_summary,
        paired=paired,
        paired_summary=paired_summary,
    )

    print("=" * 120)
    print("ROBUST SIX-FOLD SUMMARY")
    print("=" * 120)
    print(
        robust_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("PAIRED DPG SUMMARY")
    print("=" * 120)
    print(
        paired_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("OUTPUT FILES")
    print("=" * 120)

    output_files = [
        "all_fold_model_results_robust.csv",
        "robust_six_fold_summary.csv",
        "paired_dpg_fold_differences.csv",
        "paired_dpg_summary.csv",
    ]

    for filename in output_files:
        output_path = (
            OUTPUT_ROOT
            / filename
        )

        if not output_path.is_file():
            raise RuntimeError(
                "输出文件未生成："
                f"{output_path}"
            )

        print(output_path.resolve())

    print()
    print("V4_ROBUST_ANALYSIS_OK")


if __name__ == "__main__":
    main()
