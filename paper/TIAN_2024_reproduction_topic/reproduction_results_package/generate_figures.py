from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
FIGURES = ROOT / "figures"


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_experiment_outcomes() -> None:
    names = [
        "Point-GT",
        "Random negatives",
        "Dense column negatives",
        "6-channel validation",
        "6-channel held-out",
    ]
    joint_pd = [0.4151, 0.2453, 0.1132, 0.9434, 0.5660]
    pfa = [0.0133, 0.0067, 0.0133, 0.0067, 0.8933]
    colors = ["#4C78A8", "#72A0C1", "#9CBAD0", "#59A14F", "#E15759"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    for axis, values, title in zip(
        axes,
        (joint_pd, pfa),
        ("Joint detection and localization rate", "Background false-alarm rate"),
        strict=True,
    ):
        bars = axis.bar(x, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Rate")
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Tian-style local reproduction: actual evaluation results", fontsize=14)
    save_figure(figure, "01_experiment_outcomes.png")


def plot_fixed_template() -> None:
    mean_map = pd.read_csv(
        EVIDENCE / "point_gt_target_probability_mean_map.csv"
    ).to_numpy(dtype=float)
    std_map = pd.read_csv(
        EVIDENCE / "point_gt_target_probability_std_map.csv"
    ).to_numpy(dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (mean_map, std_map),
        ("Mean target probability map", "Across-sample standard deviation"),
        strict=True,
    ):
        image = axis.imshow(values, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("Range output grid x")
        axis.set_ylabel("Doppler output grid y")
        axis.set_xticks(range(values.shape[1]))
        figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    figure.suptitle(
        "Point-GT output audit: two stable Doppler bands at range column x=4\n"
        "Mean correlation with the common template: 0.99818",
        fontsize=13,
    )
    save_figure(figure, "02_fixed_template_heatmap.png")


def plot_point_gt_localization() -> None:
    table = pd.read_csv(EVIDENCE / "point_gt_validation_predictions.csv")
    targets = table[(table["target_present"] == 1) & table["detected"].astype(bool)].copy()
    successful = targets["localization_ok"].astype(bool)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter(
        targets.loc[~successful, "true_velocity_index"],
        targets.loc[~successful, "pred_velocity_index"],
        color="#E15759",
        alpha=0.8,
        label=f"Localization failed ({int((~successful).sum())})",
    )
    axis.scatter(
        targets.loc[successful, "true_velocity_index"],
        targets.loc[successful, "pred_velocity_index"],
        color="#59A14F",
        alpha=0.9,
        label=f"Localization passed ({int(successful.sum())})",
    )
    limits = [0, 127]
    axis.plot(limits, limits, linestyle="--", color="#333333", label="Perfect prediction")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("True Doppler bin")
    axis.set_ylabel("Predicted Doppler bin")
    axis.set_title("Point-GT Fold 1: true versus predicted Doppler position")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    save_figure(figure, "03_point_gt_velocity_localization.png")


def plot_background_shift() -> None:
    validation = pd.read_csv(EVIDENCE / "thesis_adapter_validation_predictions.csv")
    heldout = pd.read_csv(EVIDENCE / "thesis_adapter_heldout_predictions.csv")
    validation = validation[validation["target_present"] == 0]
    heldout = heldout[heldout["target_present"] == 0]
    threshold = float(validation["validation_threshold"].iloc[0])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(
        validation["peak_grid_y"],
        validation["peak_score"],
        s=22,
        alpha=0.65,
        color="#4C78A8",
        label="Validation background",
    )
    axes[0].scatter(
        heldout["peak_grid_y"],
        heldout["peak_score"],
        s=22,
        alpha=0.65,
        color="#E15759",
        label="Held-out background",
    )
    axes[0].axhline(
        threshold,
        color="#333333",
        linestyle="--",
        label=f"Frozen threshold = {threshold:.3f}",
    )
    axes[0].set_xlabel("Doppler output grid y")
    axes[0].set_ylabel("Peak score")
    axes[0].set_title("Background peak score and location")
    axes[0].set_xlim(-1, 32)
    axes[0].set_ylim(0, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower center")

    bins = np.arange(-0.5, 32.5, 1)
    axes[1].hist(
        validation["peak_grid_y"],
        bins=bins,
        alpha=0.7,
        color="#4C78A8",
        label="Validation background",
    )
    axes[1].hist(
        heldout["peak_grid_y"],
        bins=bins,
        alpha=0.65,
        color="#E15759",
        label="Held-out background",
    )
    axes[1].axvline(16, color="#333333", linestyle="--", label="Zero-Doppler row")
    axes[1].set_xlabel("Doppler output grid y")
    axes[1].set_ylabel("Background sample count")
    axes[1].set_title("Background peak-row distribution")
    axes[1].set_xlim(-1, 32)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")

    figure.suptitle(
        "Frozen 6-channel adapter: validation 1/150 false alarms, held-out 134/150",
        fontsize=13,
    )
    save_figure(figure, "04_background_group_shift.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    plot_experiment_outcomes()
    plot_fixed_template()
    plot_point_gt_localization()
    plot_background_shift()


if __name__ == "__main__":
    main()
