#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="formal")
    args = parser.parse_args()
    src = ROOT / "results/data_audit/roi_stage4_selected_sixfold_v1"
    out = src / f"paper_assets_{args.scope}"
    out.mkdir(parents=True, exist_ok=True)

    aggregate = pd.read_csv(src / f"aggregate_{args.scope}.csv")
    detail = pd.read_csv(src / f"detail_{args.scope}.csv")
    consistency = pd.read_csv(src / f"fold_consistency_{args.scope}.csv")
    labels = {
        "power2_baseline": "Power2 baseline",
        "power2_roi_power_control": "ROI power control",
        "power2_roi_ri4": "ROI RI4",
    }
    aggregate["display_name"] = aggregate["mode"].map(labels)
    detail["display_name"] = detail["mode"].map(labels)

    columns = [
        "display_name", "total_false_alarms", "mean_fixed_pfa",
        "mean_fixed_score_pd", "mean_fixed_joint_pd", "mean_auc",
        "mean_pauc_5pct", "mean_tpr_1pct", "mean_tpr_5pct",
        "delta_total_false_alarms_vs_power2",
        "delta_mean_joint_pd_vs_power2", "delta_mean_pauc_vs_power2",
    ]
    table = aggregate[columns]
    table.to_csv(out / "table_main.csv", index=False, encoding="utf-8-sig")
    (out / "table_main.md").write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8"
    )
    detail.to_csv(out / "table_fold_detail.csv", index=False, encoding="utf-8-sig")
    consistency.to_csv(out / "table_fold_consistency.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.bar(aggregate["display_name"], aggregate["total_false_alarms"])
    plt.ylabel("Total false alarms")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(out / "figure_total_false_alarms.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(aggregate))
    width = 0.35
    plt.bar(x - width / 2, aggregate["mean_auc"], width, label="ROC-AUC")
    plt.bar(x + width / 2, aggregate["mean_pauc_5pct"], width, label="pAUC@5% FPR")
    plt.xticks(x, aggregate["display_name"], rotation=18, ha="right")
    plt.ylim(0.6, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "figure_auc_pauc.png", dpi=220)
    plt.close()

    pivot = detail.pivot(index="fold", columns="display_name", values="false_alarms")
    plt.figure(figsize=(8, 5))
    pivot.plot(kind="line", marker="o", ax=plt.gca())
    plt.ylabel("False alarms")
    plt.tight_layout()
    plt.savefig(out / "figure_fold_false_alarms.png", dpi=220)
    plt.close()

    pivot = detail.pivot(
        index="fold", columns="display_name",
        values="test_background_exceed_q99_fraction",
    )
    plt.figure(figsize=(8, 5))
    pivot.plot(kind="line", marker="o", ax=plt.gca())
    plt.ylabel("Test background above validation Q99")
    plt.tight_layout()
    plt.savefig(out / "figure_threshold_transfer.png", dpi=220)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
