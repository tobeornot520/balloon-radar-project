from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


ROOT = Path("results/experiments")

OUT = Path(
    "results/analysis/v4_paper_figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = {
    "H":"detection_h_v4_fold",
    "V":"detection_v_v4_fold",
    "HV":"detection_hv_v4_fold",
    "DPG":"dpg_fcn_v4_fold",
}



def load(model,fold):

    path = (
        ROOT
        /
        f"{MODELS[model]}{fold:02d}_seed42"
        /
        "tables"
        /
        "test_predictions.csv"
    )

    return pd.read_csv(path)



# ==========================================================
# Figure 1 ROC Curve
# ==========================================================


plt.figure(
    figsize=(7,6)
)


for model in MODELS:


    ys=[]
    ss=[]

    for fold in range(1,7):

        df=load(model,fold)

        ys.extend(
            df.target_present.values
        )

        ss.extend(
            df.score.values
        )


    fpr,tpr,_=roc_curve(
        ys,
        ss
    )

    roc_auc=auc(
        fpr,
        tpr
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{model} AUC={roc_auc:.3f}"
    )


plt.xlabel(
    "False Alarm Rate"
)

plt.ylabel(
    "Detection Probability"
)


plt.xlim(
    0,0.2
)

plt.ylim(
    0.5,1
)


plt.grid()

plt.legend()

plt.tight_layout()


plt.savefig(
    OUT/"roc_curve.png",
    dpi=300
)

plt.close()



# ==========================================================
# Figure 2 Localization boxplot
# ==========================================================


range_data=[]
velocity_data=[]
labels=[]


for model in MODELS:

    values_r=[]
    values_v=[]

    for fold in range(1,7):

        df=load(model,fold)

        pos=df[
            df.target_present==1
        ]

        values_r.extend(
            pos.range_error_gates
        )

        values_v.extend(
            pos.velocity_error_bins
        )


    range_data.append(values_r)
    velocity_data.append(values_v)
    labels.append(model)



plt.figure(
    figsize=(7,5)
)

plt.boxplot(
    range_data,
    tick_labels=labels
)

plt.ylabel(
    "Range error (gate)"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    OUT/"range_error_boxplot.png",
    dpi=300
)

plt.close()



plt.figure(
    figsize=(7,5)
)


plt.boxplot(
    velocity_data,
    tick_labels=labels
)


plt.ylabel(
    "Velocity error (bin)"
)


plt.grid()

plt.tight_layout()


plt.savefig(
    OUT/"velocity_error_boxplot.png",
    dpi=300
)

plt.close()



# ==========================================================
# Figure 3 Rescue Analysis
# ==========================================================


rescue_H=[]
rescue_HV=[]


for fold in range(1,7):


    h=load("H",fold)
    hv=load("HV",fold)
    dpg=load("DPG",fold)


    h=h[
        h.target_present==1
    ].set_index(
        "sample_id"
    )


    hv=hv[
        hv.target_present==1
    ].set_index(
        "sample_id"
    )


    dpg=dpg[
        dpg.target_present==1
    ].set_index(
        "sample_id"
    )


    rescue_H.append(
        (
            (h.correct_detection==0)
            &
            (dpg.correct_detection==1)
        ).sum()
    )


    rescue_HV.append(
        (
            (hv.correct_detection==0)
            &
            (dpg.correct_detection==1)
        ).sum()
    )



plt.figure(
    figsize=(6,5)
)


x=np.arange(6)


plt.bar(
    x-0.2,
    rescue_H,
    width=0.4,
    label="vs H"
)


plt.bar(
    x+0.2,
    rescue_HV,
    width=0.4,
    label="vs HV"
)


plt.xlabel(
    "Fold"
)

plt.ylabel(
    "Recovered targets"
)

plt.legend()

plt.grid(
    axis="y"
)


plt.tight_layout()


plt.savefig(
    OUT/"target_rescue.png",
    dpi=300
)


plt.close()



# ==========================================================
# Figure 4 Gate distribution
# ==========================================================


gate=[]


for fold in range(1,7):

    df=load("DPG",fold)

    for name,data in [

        (
            "Target",
            df[df.target_present==1]
        ),

        (
            "Background",
            df[df.target_present==0]
        ),

        (
            "FalseAlarm",
            df[df.false_alarm==1]
        )

    ]:

        for x in data.gate_h:

            gate.append(
                {
                    "type":name,
                    "gate_h":x
                }
            )



gate=pd.DataFrame(gate)



plt.figure(
    figsize=(7,5)
)


plt.boxplot(
    [
        gate[
            gate.type=="Target"
        ].gate_h,

        gate[
            gate.type=="Background"
        ].gate_h,

        gate[
            gate.type=="FalseAlarm"
        ].gate_h,
    ],

    tick_labels=[
        "Target",
        "Background",
        "False Alarm"
    ]
)


plt.ylabel(
    "H branch gate weight"
)


plt.grid()

plt.tight_layout()


plt.savefig(
    OUT/"gate_distribution.png",
    dpi=300
)


plt.close()



print(
    "Figures saved:",
    OUT
)
