"""Figures for the cost-of-quality comparison: mean performance AND
cross-fold variance (the "cost of variability" signal) for all three model
families."""

import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

ORDER = ["RandomForest", "GradientBoosting", "CausalTransformer (final config)"]
COLORS = ["#8C8C8C", "#DD8452", "#4C72B0"]


def fig1_mean_performance():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "vs_baselines_fold_avg.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ["auroc_TWF", "auroc_OSF", "wear_r2"]):
        means = df.groupby("config")[col].mean().reindex(ORDER)
        stds = df.groupby("config")[col].std().reindex(ORDER)
        ax.bar(range(len(means)), means.values, yerr=stds.values, capsize=5, color=COLORS)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(ORDER, rotation=20, ha="right")
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Mean Performance +/- Cross-Fold SD: Random Forest vs. Gradient Boosting vs. Causal Transformer")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_mean_performance.png"), dpi=150)
    plt.close()


def fig2_cost_of_variability():
    """The paper's central point: variance itself is a cost. Plot cross-fold
    SD of wear R2 directly, since a volatile signal forces wider safety
    margins regardless of its mean."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "vs_baselines_fold_avg.csv"))
    stds = df.groupby("config")["wear_r2"].std().reindex(ORDER)

    plt.figure(figsize=(7, 5))
    plt.bar(range(len(stds)), stds.values, color=COLORS)
    plt.xticks(range(len(stds)), ORDER, rotation=20, ha="right")
    plt.ylabel("Cross-fold SD of Wear Regression R2")
    plt.title("Cost of Variability: Lower SD = More Predictable Quality Signal")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_cost_of_variability.png"), dpi=150)
    plt.close()


def fig3_per_fold_boxplot():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "vs_baselines_fold_avg.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in zip(axes, ["auroc_TWF", "auroc_OSF", "wear_r2"],
                               ["TWF AUROC", "OSF AUROC", "Wear Regression R2"]):
        data = [df[df["config"] == c][col].dropna().values for c in ORDER]
        ax.boxplot(data, tick_labels=ORDER)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Per-Fold Distribution (8 folds, 3 seeds averaged per fold)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_per_fold_boxplot.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig1_mean_performance()
    fig2_cost_of_variability()
    fig3_per_fold_boxplot()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
