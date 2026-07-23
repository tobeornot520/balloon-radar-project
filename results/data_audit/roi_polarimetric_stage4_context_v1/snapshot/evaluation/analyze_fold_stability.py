from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path(
    "results/tables/cv_fold_metrics.csv"
)


OUT_DIR = Path(
    "results/final"
)


FIG_DIR = OUT_DIR / "figures"


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)



def main():


    df = pd.read_csv(INPUT)


    # 保存整理结果

    columns = [

        "cv_fold",

        "refined_mean_range_error",

        "refined_mean_velocity_error",

        "refined_strict_hit_rate",

        "refined_relaxed_hit_rate"

    ]


    result = df[columns]


    result.to_csv(

        OUT_DIR /
        "fold_stability.csv",

        index=False,

        encoding="utf-8-sig"

    )


    print("\n===== Fold Stability =====")

    print(result)



    # ==========================
    # Strict Hit Rate
    # ==========================

    plt.figure(
        figsize=(7,4)
    )


    plt.plot(

        result["cv_fold"],

        result[
            "refined_strict_hit_rate"
        ]*100,

        marker="o"

    )


    plt.xlabel(
        "Fold"
    )


    plt.ylabel(
        "Strict Hit Rate (%)"
    )


    plt.title(
        "5-fold Cross Validation Stability"
    )


    plt.grid()


    plt.tight_layout()


    plt.savefig(

        FIG_DIR /
        "fold_stability.png",

        dpi=300

    )


    plt.close()



if __name__=="__main__":

    main()