"""
07_evaluate_and_compare.py
==========================
Compare all models on the same test set and generate comprehensive
evaluation plots and tables.

Reads results from each model's output directory and produces:
  1. Side-by-side metric comparison table
  2. Scatter plots (predicted vs actual)
  3. Error distribution analysis
  4. Difficulty-stratified performance
  5. Learning curves (if available)

Usage:
    python 07_evaluate_and_compare.py \
        --results_dir ../results \
        --output_dir ../results/comparison \
        --data_dir ../data
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================================
# Load results from each model
# =========================================================================

MODEL_NAMES = ["lomap", "mpnn", "gat", "ntm", "transformer"]
MODEL_DISPLAY = {
    "lomap": "LOMAP Baseline (GBM)",
    "mpnn": "MPNN",
    "gat": "GAT / AttentiveFP",
    "ntm": "NTM (Learned Metric)",
    "transformer": "Molecular Transformer",
}


def load_model_results(results_dir):
    """Load results.json and test predictions for all available models."""
    results = {}

    for name in MODEL_NAMES:
        model_dir = os.path.join(results_dir, name)
        results_path = os.path.join(model_dir, "results.json")
        preds_path = os.path.join(model_dir, "test_preds.npz")

        # LOMAP has multiple sub-models; use GBM
        if name == "lomap":
            preds_path = os.path.join(model_dir, "gbm_test_preds.npz")

        if not os.path.exists(results_path):
            print(f"  Skipping {name}: no results.json found")
            continue

        with open(results_path) as f:
            metrics = json.load(f)

        preds = None
        if os.path.exists(preds_path):
            data = np.load(preds_path)
            preds = {"y_true": data["y_true"], "y_pred": data["y_pred"]}

        results[name] = {"metrics": metrics, "preds": preds}
        print(f"  Loaded {name}: {MODEL_DISPLAY.get(name, name)}")

    return results


# =========================================================================
# 1. Metric Comparison Table
# =========================================================================

def metric_comparison_table(results, output_dir):
    """Generate a comparison table of all models."""
    print("\n" + "=" * 60)
    print("1. Metric Comparison")
    print("=" * 60)

    rows = []
    for name, data in results.items():
        m = data["metrics"]
        row = {
            "Model": MODEL_DISPLAY.get(name, name),
            "Test RMSE": m.get("test_rmse", None),
            "Test MAE": m.get("test_mae", None),
            "Test R²": m.get("test_r2", None),
            "Test Pearson r": m.get("test_pearson_r", None),
            "Test Spearman ρ": m.get("test_spearman_r", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    df = df.sort_values("Test RMSE")

    print(df.to_string(float_format="%.4f"))

    # Save as CSV and LaTeX
    df.to_csv(os.path.join(output_dir, "model_comparison.csv"), float_format="%.4f")
    with open(os.path.join(output_dir, "model_comparison.tex"), "w") as f:
        f.write(df.to_latex(float_format="%.4f"))

    return df


# =========================================================================
# 2. Scatter Plots
# =========================================================================

def scatter_plots(results, output_dir):
    """Predicted vs actual scatter plots for all models."""
    print("\n" + "=" * 60)
    print("2. Scatter Plots")
    print("=" * 60)

    models_with_preds = {k: v for k, v in results.items() if v["preds"] is not None}
    n_models = len(models_with_preds)

    if n_models == 0:
        print("  No predictions available.")
        return

    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, (name, data) in enumerate(models_with_preds.items()):
        ax = axes[idx // cols][idx % cols]
        y_true = data["preds"]["y_true"]
        y_pred = data["preds"]["y_pred"]

        ax.scatter(y_true, y_pred, s=3, alpha=0.2, c="steelblue")

        # Identity line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="y=x")

        # Stats
        r2 = r2_score(y_true, y_pred)
        pr, _ = pearsonr(y_true, y_pred)
        ax.set_xlabel("Actual Stderr Difference")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{MODEL_DISPLAY.get(name, name)}\nR²={r2:.3f}  r={pr:.3f}")
        ax.legend(loc="upper left")

    # Hide unused axes
    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_plots.png"), dpi=150)
    plt.close()
    print("  Saved scatter_plots.png")


# =========================================================================
# 3. Error Distribution
# =========================================================================

def error_distributions(results, output_dir):
    """Compare error distributions across models."""
    print("\n" + "=" * 60)
    print("3. Error Distributions")
    print("=" * 60)

    models_with_preds = {k: v for k, v in results.items() if v["preds"] is not None}
    if not models_with_preds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute error distributions
    ax = axes[0]
    for name, data in models_with_preds.items():
        errors = np.abs(data["preds"]["y_true"] - data["preds"]["y_pred"])
        ax.hist(errors, bins=50, alpha=0.4, label=MODEL_DISPLAY.get(name, name))
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Error Distribution")
    ax.legend(fontsize=8)

    # Signed error (bias check)
    ax = axes[1]
    for name, data in models_with_preds.items():
        errors = data["preds"]["y_pred"] - data["preds"]["y_true"]
        ax.hist(errors, bins=50, alpha=0.4, label=MODEL_DISPLAY.get(name, name))
    ax.set_xlabel("Signed Error (pred - actual)")
    ax.set_ylabel("Count")
    ax.set_title("Signed Error Distribution (Bias Check)")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distributions.png"), dpi=150)
    plt.close()
    print("  Saved error_distributions.png")


# =========================================================================
# 4. Difficulty-Stratified Performance
# =========================================================================

def stratified_performance(results, output_dir):
    """
    Evaluate how each model performs on easy vs hard transformations,
    stratified by |target| magnitude.
    """
    print("\n" + "=" * 60)
    print("4. Difficulty-Stratified Performance")
    print("=" * 60)

    models_with_preds = {k: v for k, v in results.items() if v["preds"] is not None}
    if not models_with_preds:
        return

    # Use the first model's y_true for stratification (all should be same)
    ref_name = list(models_with_preds.keys())[0]
    y_true = models_with_preds[ref_name]["preds"]["y_true"]
    abs_target = np.abs(y_true)

    # Define strata
    quantiles = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    thresholds = np.quantile(abs_target, quantiles)
    strata_names = [
        f"Q1 (|t|<{thresholds[1]:.3f})",
        f"Q2 ({thresholds[1]:.3f}-{thresholds[2]:.3f})",
        f"Q3 ({thresholds[2]:.3f}-{thresholds[3]:.3f})",
        f"Q4 ({thresholds[3]:.3f}-{thresholds[4]:.3f})",
    ]

    strata_data = []
    for i in range(len(quantiles) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == len(quantiles) - 2:
            mask = (abs_target >= lo) & (abs_target <= hi)
        else:
            mask = (abs_target >= lo) & (abs_target < hi)

        for name, data in models_with_preds.items():
            yt = data["preds"]["y_true"][mask]
            yp = data["preds"]["y_pred"][mask]
            if len(yt) < 5:
                continue
            strata_data.append({
                "Stratum": strata_names[i],
                "Model": MODEL_DISPLAY.get(name, name),
                "RMSE": np.sqrt(mean_squared_error(yt, yp)),
                "MAE": mean_absolute_error(yt, yp),
                "n": int(mask.sum()),
            })

    df_strata = pd.DataFrame(strata_data)
    print(df_strata.pivot(index="Stratum", columns="Model", values="RMSE")
          .to_string(float_format="%.4f"))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_pivot = df_strata.pivot(index="Stratum", columns="Model", values="RMSE")
    df_pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_ylabel("RMSE")
    ax.set_title("Performance by Difficulty Stratum")
    ax.legend(fontsize=8, loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stratified_performance.png"), dpi=150)
    plt.close()
    print("  Saved stratified_performance.png")

    df_strata.to_csv(os.path.join(output_dir, "stratified_performance.csv"), index=False)


# =========================================================================
# 5. Learning Curves
# =========================================================================

def learning_curves(results, output_dir):
    """Plot training and validation loss curves."""
    print("\n" + "=" * 60)
    print("5. Learning Curves")
    print("=" * 60)

    models_with_history = {
        k: v for k, v in results.items()
        if "history" in v["metrics"] and v["metrics"]["history"]
    }

    if not models_with_history:
        print("  No training history available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    for name, data in models_with_history.items():
        history = data["metrics"]["history"]
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        ax.plot(epochs, train_loss, label=MODEL_DISPLAY.get(name, name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("Training Loss Curves")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Validation loss
    ax = axes[1]
    for name, data in models_with_history.items():
        history = data["metrics"]["history"]
        epochs = [h["epoch"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        ax.plot(epochs, val_loss, label=MODEL_DISPLAY.get(name, name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_title("Validation Loss Curves")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=150)
    plt.close()
    print("  Saved learning_curves.png")


# =========================================================================
# 6. NTM-Specific Analysis
# =========================================================================

def ntm_specific_analysis(results, results_dir, output_dir):
    """Extra analysis specific to the NTM model."""
    print("\n" + "=" * 60)
    print("6. NTM-Specific Analysis")
    print("=" * 60)

    ntm_dir = os.path.join(results_dir, "ntm")
    metric_path = os.path.join(ntm_dir, "metric_tensor.npz")
    embed_path = os.path.join(ntm_dir, "test_embeddings.npz")

    if not os.path.exists(metric_path):
        print("  No NTM metric tensor found. Skipping.")
        return

    # Load metric tensor
    metric_data = np.load(metric_path)
    eigenvalues = metric_data["eigenvalues"]
    M = metric_data["M"]

    print(f"  Metric tensor shape: {M.shape}")
    print(f"  Anisotropy: {eigenvalues[0] / eigenvalues[-1]:.1f}")

    # Correlation: NTM distance vs prediction error (for all models)
    if os.path.exists(embed_path):
        embed_data = np.load(embed_path)
        d_m = embed_data["d_m"]
        targets = embed_data["targets"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NTM distance vs |target|
        ax = axes[0]
        ax.scatter(d_m, np.abs(targets), s=3, alpha=0.2, c="steelblue")
        pr, _ = pearsonr(d_m, np.abs(targets))
        ax.set_xlabel("NTM Distance d_M(A, B)")
        ax.set_ylabel("|Stderr Difference|")
        ax.set_title(f"NTM Distance vs |Target| (r={pr:.3f})")

        # Does NTM distance predict which pairs are hard?
        ax = axes[1]
        # Bin by NTM distance and compute mean absolute error per bin
        bins = np.percentile(d_m, np.linspace(0, 100, 21))
        bin_centers = []
        bin_abs_targets = []
        for i in range(len(bins) - 1):
            mask = (d_m >= bins[i]) & (d_m < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_abs_targets.append(np.abs(targets[mask]).mean())

        ax.bar(range(len(bin_centers)), bin_abs_targets, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(bin_centers)))
        ax.set_xticklabels([f"{c:.2f}" for c in bin_centers], rotation=45, fontsize=7)
        ax.set_xlabel("NTM Distance Bin")
        ax.set_ylabel("Mean |Target|")
        ax.set_title("Mean |Target| by NTM Distance Quantile")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ntm_distance_analysis.png"), dpi=150)
        plt.close()
        print("  Saved ntm_distance_analysis.png")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--output_dir", default="../results/comparison")
    parser.add_argument("--data_dir", default="../data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all model results
    print("Loading model results...")
    results = load_model_results(args.results_dir)

    if not results:
        print("No model results found. Run the model scripts first.")
        return

    # Run analyses
    df_comparison = metric_comparison_table(results, args.output_dir)
    scatter_plots(results, args.output_dir)
    error_distributions(results, args.output_dir)
    stratified_performance(results, args.output_dir)
    learning_curves(results, args.output_dir)
    ntm_specific_analysis(results, args.results_dir, args.output_dir)

    print(f"\nAll comparison results saved to {args.output_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
