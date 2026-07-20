from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path("results/experiments")


MODELS = {
    "H": "detection_h_v4_fold",
    "V": "detection_v_v4_fold",
    "HV": "detection_hv_v4_fold",
    "DPG": "dpg_fcn_v4_fold",
}


def load_prediction(model, fold):

    exp = ROOT / f"{MODELS[model]}{fold:02d}_seed42"

    path = exp / "tables" / "test_predictions.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    return df



# ============================================================
# 1 ROC-AUC + Pd@FAR
# ============================================================

print("\n")
print("="*100)
print("ROC-AUC AND Pd@FALSE ALARM RATE")
print("="*100)


auc_rows=[]


for fold in range(1,7):

    for model in MODELS:

        df=load_prediction(model,fold)

        if df is None:
            continue


        y=df["target_present"].values
        score=df["score"].values


        auc=roc_auc_score(y,score)


        fpr,tpr,thr=roc_curve(y,score)


        row={
            "fold":fold,
            "model":model,
            "AUC":auc,
        }


        for target_far in [0.01,0.05,0.10]:

            idx=np.where(fpr<=target_far)[0]

            if len(idx):

                row[
                    f"Pd@Pfa{int(target_far*100)}%"
                ]=max(tpr[idx])

            else:
                row[
                    f"Pd@Pfa{int(target_far*100)}%"
                ]=0


        auc_rows.append(row)



auc_df=pd.DataFrame(auc_rows)


print(
    auc_df
    .groupby("model")
    .mean(numeric_only=True)
    .round(4)
)


print("\nFULL")
print(auc_df.round(4).to_string(index=False))



# ============================================================
# 2 DPG gate statistics
# ============================================================


print("\n")
print("="*100)
print("DPG GATE STATISTICS")
print("="*100)


gate_rows=[]


for fold in range(1,7):

    df=load_prediction("DPG",fold)

    if df is None:
        continue


    for group,name in [
        (df,"all"),
        (df[df.target_present==1],"positive"),
        (df[df.target_present==0],"background"),
        (df[df.false_alarm==1],"false_alarm"),
        (df[df.correct_detection==1],"correct"),
    ]:

        if len(group)==0:
            continue


        gate_rows.append(
            {
                "fold":fold,
                "group":name,
                "count":len(group),
                "gate_h_mean":group["gate_h"].mean(),
                "gate_v_mean":group["gate_v"].mean(),
                "gate_margin_mean":group["gate_margin"].mean(),
            }
        )


gate_df=pd.DataFrame(gate_rows)

print(
    gate_df.round(4).to_string(index=False)
)


print("\nAverage")
print(
    gate_df
    .groupby("group")
    .mean(numeric_only=True)
    .round(4)
)



# ============================================================
# 3 False alarm source analysis
# ============================================================


print("\n")
print("="*100)
print("DPG FALSE ALARM SOURCE ANALYSIS")
print("="*100)


for fold in range(1,7):

    df=load_prediction("DPG",fold)


    if df is None:
        continue


    fa=df[df.false_alarm==1]


    if len(fa)==0:
        continue


    print("\n")
    print("-"*80)
    print("FOLD",fold,
          "false alarms:",
          len(fa))


    print("\nsource_file:")
    print(
        fa.mat_path
        .apply(lambda x:Path(x).stem)
        .value_counts()
        .head(10)
    )


    print("\nbeam:")
    print(
        fa.beam_layer
        .value_counts()
    )


    print("\nazimuth:")
    print(
        fa.azimuth_deg
        .value_counts()
        .head(10)
    )


print("\nDONE")
