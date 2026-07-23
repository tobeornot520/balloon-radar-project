from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 路径
# ==================================================

TABLE_DIR = Path("results/tables")

FINAL_DIR = Path("results/final")

FIG_DIR = FINAL_DIR / "figures"


FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ==================================================
# 工具函数
# ==================================================

def save_csv(df, name):

    path = FINAL_DIR / name

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print("保存:", path)



def load_json(path):

    if not path.exists():

        print("缺失:", path)

        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



# ==================================================
# 1. 模型总体比较
# ==================================================

def generate_model_comparison():


    records = []



    # ------------------
    # CA-CFAR
    # ------------------

    cfar_path = (
        TABLE_DIR /
        "cfar_summary.csv"
    )


    if cfar_path.exists():

        cfar = pd.read_csv(
            cfar_path
        )


        record = {

            "Model":
                "CA-CFAR",

            "Type":
                "Traditional"

        }


        # 自动读取存在字段

        for col in cfar.columns:

            if "hit" in col.lower():

                record[col] = cfar[col].iloc[0]


        records.append(record)



    # ------------------
    # Simple FCN
    # ------------------

    simple = load_json(

        TABLE_DIR /
        "simple_fcn_summary.json"

    )


    if simple:


        record = {

            "Model":
                "Simple FCN",

            "Type":
                "Baseline DL"

        }


        for key,value in simple.items():

            if isinstance(value,(int,float)):

                record[key] = value


        records.append(record)



    # ------------------
    # Dual FCN
    # ------------------

    cv = load_json(

        TABLE_DIR /
        "cv_summary.json"

    )


    if cv:


        record = {


            "Model":
                "Dual FCN",

            "Type":
                "Dual Channel"

        }


        record["Range Error"] = (
            cv["oof_mean_range_error"]
        )


        record["Velocity Error"] = (
            cv["oof_mean_velocity_error"]
        )


        record["Strict Hit Rate"] = (

            cv["fold_refined_strict_mean"]

        )


        records.append(record)



    # ------------------
    # Refiner
    # ------------------

    if cv:


        record = {


            "Model":
                "Dual FCN + Refiner",

            "Type":
                "Proposed"

        }



        record["Range Error"] = (

            cv["oof_mean_range_error"]

        )


        record["Velocity Error"] = (

            cv["oof_mean_velocity_error"]

        )


        record["Strict Hit Rate"] = (

            cv["oof_strict_hit_rate"]

        )


        record["Relaxed Hit Rate"] = (

            cv["oof_relaxed_hit_rate"]

        )


        records.append(record)



    df = pd.DataFrame(records)


    save_csv(
        df,
        "model_comparison.csv"
    )


    return df




# ==================================================
# 2. Refiner提升分析
# ==================================================

def generate_refiner_ablation():


    path = (

        TABLE_DIR /
        "cv_oof_predictions.csv"

    )


    if not path.exists():

        print(
            "缺少:",
            path
        )

        return



    df = pd.read_csv(path)



    result = pd.DataFrame({

        "Stage":[

            "Dual FCN coarse",

            "Dual FCN + Refiner"

        ],


        "Mean Range Error":[

            df["coarse_range_error"].mean(),

            df["refined_range_error"].mean()

        ],


        "Mean Velocity Error":[

            df["coarse_velocity_error"].mean(),

            df["refined_velocity_error"].mean()

        ],


        "Strict Hit Rate":[

            df["coarse_strict_hit"].mean(),

            df["refined_strict_hit"].mean()

        ],


        "Relaxed Hit Rate":[

            df["coarse_relaxed_hit"].mean(),

            df["refined_relaxed_hit"].mean()

        ]

    })


    save_csv(

        result,

        "refiner_ablation.csv"

    )


    return result




# ==================================================
# 3. 五折稳定性
# ==================================================

def generate_fold_analysis():


    path = (

        TABLE_DIR /
        "cv_fold_metrics.csv"

    )


    if not path.exists():

        return



    df = pd.read_csv(path)



    save_csv(

        df,

        "fold_metrics.csv"

    )



    plt.figure(

        figsize=(8,5)

    )


    plt.plot(

        df["cv_fold"],

        df["refined_strict_hit_rate"],

        marker="o"

    )


    plt.xlabel(

        "Fold"

    )


    plt.ylabel(

        "Strict Hit Rate"

    )


    plt.title(

        "5-fold Stability"

    )


    plt.grid()


    plt.tight_layout()



    plt.savefig(

        FIG_DIR /
        "fold_stability.png",

        dpi=300

    )


    plt.close()



# ==================================================
# 4. 误差分布
# ==================================================

def generate_error_distribution():


    path = (

        TABLE_DIR /
        "cv_oof_predictions.csv"

    )


    if not path.exists():

        return



    df = pd.read_csv(path)



    plt.figure(

        figsize=(8,5)

    )


    plt.hist(

        df["refined_range_error"],

        bins=30

    )


    plt.xlabel(

        "Range Error (gate)"

    )


    plt.ylabel(

        "Count"

    )


    plt.title(

        "Refined Range Error Distribution"

    )


    plt.tight_layout()



    plt.savefig(

        FIG_DIR /
        "range_error_distribution.png",

        dpi=300

    )


    plt.close()



# ==================================================
# 主程序
# ==================================================

def main():


    print(

        "\n====== Final Experiment Report ======\n"

    )


    generate_model_comparison()


    generate_refiner_ablation()


    generate_fold_analysis()


    generate_error_distribution()



    print(

        "\n全部完成"

    )

    print(

        "输出目录:",
        FINAL_DIR

    )



if __name__ == "__main__":

    main()