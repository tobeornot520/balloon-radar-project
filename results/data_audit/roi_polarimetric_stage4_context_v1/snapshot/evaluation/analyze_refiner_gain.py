from pathlib import Path
import pandas as pd


INPUT = Path(
    "results/tables/cv_oof_predictions.csv"
)

OUTPUT = Path(
    "results/final/refiner_gain.csv"
)


def main():

    df = pd.read_csv(INPUT)


    results = []


    # ======================
    # Dual FCN
    # ======================

    coarse = {

        "Stage":
            "Dual FCN",

        "Range_MAE_gate":
            df[
                "coarse_range_error"
            ].mean(),

        "Velocity_MAE_bin":
            df[
                "coarse_velocity_error"
            ].mean(),

        "Strict_Hit_Rate":
            df[
                "coarse_strict_hit"
            ].mean(),

        "Relaxed_Hit_Rate":
            df[
                "coarse_relaxed_hit"
            ].mean()
    }


    results.append(coarse)



    # ======================
    # Refiner
    # ======================

    refined = {

        "Stage":
            "Dual FCN + Refiner",

        "Range_MAE_gate":
            df[
                "refined_range_error"
            ].mean(),

        "Velocity_MAE_bin":
            df[
                "refined_velocity_error"
            ].mean(),

        "Strict_Hit_Rate":
            df[
                "refined_strict_hit"
            ].mean(),

        "Relaxed_Hit_Rate":
            df[
                "refined_relaxed_hit"
            ].mean()

    }


    results.append(refined)



    result_df = pd.DataFrame(results)



    # ======================
    # 增益
    # ======================

    range_gain = (

        1 -
        result_df.loc[1,
        "Range_MAE_gate"]
        /
        result_df.loc[0,
        "Range_MAE_gate"]

    )


    strict_gain = (

        result_df.loc[1,
        "Strict_Hit_Rate"]

        -

        result_df.loc[0,
        "Strict_Hit_Rate"]

    )


    gain_row = pd.DataFrame([{

        "Stage":
            "Improvement",

        "Range_Error_Reduction":
            range_gain,

        "Strict_Hit_Increase":
            strict_gain

    }])


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    result_df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


    gain_row.to_csv(

        OUTPUT.parent /
        "refiner_gain_summary.csv",

        index=False,

        encoding="utf-8-sig"

    )


    print("\n===== Refiner Gain =====")

    print(result_df)

    print("\n提升:")

    print(gain_row)


if __name__ == "__main__":
    main()