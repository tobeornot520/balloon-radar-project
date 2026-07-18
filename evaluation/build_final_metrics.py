from pathlib import Path
import pandas as pd
import json


TABLE_DIR = Path("results/tables")

OUT_DIR = Path("results/final")

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)



def main():


    results = []


    # =====================================
    # 1. CA-CFAR
    # =====================================

    cfar = pd.read_csv(
        TABLE_DIR /
        "cfar_summary.csv"
    )


    # 第一行=全部样本

    cfar_all = cfar.iloc[0]


    results.append({

        "Model":
            "CA-CFAR",

        "Range_MAE_gate":
            cfar_all[
                "mean_absolute_range_error_gates"
            ],

        "Velocity_MAE_bin":
            cfar_all[
                "mean_absolute_velocity_error_bins"
            ],

        "Strict_Hit_Rate":
            cfar_all[
                "strict_detection_rate"
            ],

        "Relaxed_Hit_Rate":
            cfar_all[
                "relaxed_detection_rate"
            ],

        "Description":
            "Traditional baseline"

    })



    # =====================================
    # 2. Simple FCN
    # =====================================


    simple = load_json(

        TABLE_DIR /
        "simple_fcn_summary.json"

    )


    test = simple["test_metrics"]


    results.append({

        "Model":
            "Simple FCN",

        "Range_MAE_gate":
            test[
                "mean_range_error"
            ],

        "Velocity_MAE_bin":
            test[
                "mean_velocity_error"
            ],

        "Strict_Hit_Rate":
            test[
                "strict_hit_rate"
            ],

        "Relaxed_Hit_Rate":
            test[
                "relaxed_hit_rate"
            ],

        "Description":
            "Single channel FCN baseline"

    })



    # =====================================
    # 3. Dual FCN coarse
    # =====================================


    oof = pd.read_csv(

        TABLE_DIR /
        "cv_oof_predictions.csv"

    )


    results.append({

        "Model":
            "Dual FCN",

        "Range_MAE_gate":
            oof[
                "coarse_range_error"
            ].mean(),

        "Velocity_MAE_bin":
            oof[
                "coarse_velocity_error"
            ].mean(),

        "Strict_Hit_Rate":
            oof[
                "coarse_strict_hit"
            ].mean(),

        "Relaxed_Hit_Rate":
            oof[
                "coarse_relaxed_hit"
            ].mean(),

        "Description":
            "H/V dual-channel coarse detection"

    })



    # =====================================
    # 4. Dual FCN + Refiner
    # =====================================


    results.append({

        "Model":
            "Dual FCN + Refiner",

        "Range_MAE_gate":
            oof[
                "refined_range_error"
            ].mean(),

        "Velocity_MAE_bin":
            oof[
                "refined_velocity_error"
            ].mean(),

        "Strict_Hit_Rate":
            oof[
                "refined_strict_hit"
            ].mean(),

        "Relaxed_Hit_Rate":
            oof[
                "refined_relaxed_hit"
            ].mean(),

        "Description":
            "Proposed two-stage localization"

    })



    df = pd.DataFrame(results)



    # 百分比化

    df["Strict_Hit_Rate"] = (
        df["Strict_Hit_Rate"]*100
    )


    df["Relaxed_Hit_Rate"] = (
        df["Relaxed_Hit_Rate"]*100
    )


    output = (
        OUT_DIR /
        "model_comparison.csv"
    )


    df.to_csv(

        output,

        index=False,

        encoding="utf-8-sig"

    )


    print("\n========== Final Metrics ==========")

    print(df)


    print(
        "\n保存:",
        output
    )



if __name__ == "__main__":

    main()