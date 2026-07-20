from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path("results/experiments")


MODELS = {
    "H": "detection_h_v4_fold",
    "V": "detection_v_v4_fold",
    "HV": "detection_hv_v4_fold",
    "DPG": "dpg_fcn_v4_fold",
}


def load(model, fold):

    path = (
        ROOT
        / f"{MODELS[model]}{fold:02d}_seed42"
        / "tables"
        / "test_predictions.csv"
    )

    if not path.exists():
        return None

    return pd.read_csv(path)



# ============================================================
# 1. Localization statistics
# ============================================================


print("\n")
print("="*100)
print("V4 SIX FOLD LOCALIZATION ERROR")
print("="*100)


rows=[]


for fold in range(1,7):

    for model in MODELS:

        df=load(model,fold)

        if df is None:
            continue


        pos=df[df.target_present==1]


        detected=pos[pos.correct_detection==1]


        rows.append(
            {
                "fold":fold,
                "model":model,

                "all_range_mae":
                    pos.range_error_gates.mean(),

                "all_velocity_mae":
                    pos.velocity_error_bins.mean(),

                "correct_range_mae":
                    detected.range_error_gates.mean()
                    if len(detected)>0 else np.nan,

                "correct_velocity_mae":
                    detected.velocity_error_bins.mean()
                    if len(detected)>0 else np.nan,


                "correct_count":
                    len(detected),

                "positive_count":
                    len(pos),
            }
        )



loc=pd.DataFrame(rows)


print("\nFULL")
print(
    loc.round(3)
    .to_string(index=False)
)



print("\nMEAN ± STD")


summary=(
    loc
    .groupby("model")
    .agg(
        all_range_mean=("all_range_mae","mean"),
        all_range_std=("all_range_mae","std"),

        all_velocity_mean=("all_velocity_mae","mean"),
        all_velocity_std=("all_velocity_mae","std"),

        correct_range_mean=("correct_range_mae","mean"),
        correct_range_std=("correct_range_mae","std"),

        correct_velocity_mean=("correct_velocity_mae","mean"),
        correct_velocity_std=("correct_velocity_mae","std"),
    )
)


print(
    summary.round(3)
)



# ============================================================
# 2. Rescue analysis
# ============================================================


print("\n")
print("="*100)
print("DPG RESCUE ANALYSIS")
print("="*100)



rescue_rows=[]


for fold in range(1,7):


    data={}


    for model in ["H","HV","DPG"]:

        data[model]=load(model,fold)


    if any(v is None for v in data.values()):
        continue


    # 只看目标样本

    h=data["H"]
    hv=data["HV"]
    dpg=data["DPG"]


    h=h[h.target_present==1].copy()
    hv=hv[hv.target_present==1].copy()
    dpg=dpg[dpg.target_present==1].copy()


    # sample_id 对齐

    h=h.set_index("sample_id")
    hv=hv.set_index("sample_id")
    dpg=dpg.set_index("sample_id")


    common=dpg.index


    h_correct=h.loc[common].correct_detection
    hv_correct=hv.loc[common].correct_detection
    dpg_correct=dpg.correct_detection


    rescue_rows.append(
        {
            "fold":fold,


            # H miss -> DPG hit

            "DPG_rescue_H":
                int(((h_correct==0)&(dpg_correct==1)).sum()),


            # HV miss -> DPG hit

            "DPG_rescue_HV":
                int(((hv_correct==0)&(dpg_correct==1)).sum()),


            # H hit -> DPG miss

            "DPG_loss_vs_H":
                int(((h_correct==1)&(dpg_correct==0)).sum()),


            # HV hit -> DPG miss

            "DPG_loss_vs_HV":
                int(((hv_correct==1)&(dpg_correct==0)).sum()),



            "total_targets":
                len(common),
        }
    )



res=pd.DataFrame(rescue_rows)


print("\nFULL")

print(res.to_string(index=False))



print("\nTOTAL")


print(
    res.drop(columns=["fold"])
    .sum()
)



print("\nAVERAGE PER FOLD")


print(
    res.drop(columns=["fold"])
    .mean()
    .round(3)
)


print("\nDONE")
