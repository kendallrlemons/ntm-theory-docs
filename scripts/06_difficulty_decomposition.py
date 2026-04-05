"""
06_difficulty_decomposition.py
==============================
Decompose transformation difficulty using the learned NTM metric tensor.

Takes a trained NTM model and analyzes:
  1. Metric tensor eigenstructure (hard vs easy directions)
  2. Per-pair difficulty decomposition by eigendirection
  3. Molecular substructure attribution (which atoms/bonds drive difficulty)
  4. Cluster analysis of transformation types

Usage:
    python 06_difficulty_decomposition.py \
        --data_dir ../data \
        --ntm_model_dir ../results/ntm \
        --output_dir ../results/decomposition \
        --batch_size 256 \
        --seed 42
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    PairDataset, collate_pair, ATOM_DIM, BOND_DIM,
    smiles_to_graph, batch_graphs,
)
from importlib import import_module
# Import NTM model classes from 04_ntm_model.py
_ntm_mod = import_module("04_ntm_model")
NTMModel = _ntm_mod.NTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Metric Tensor Eigenstructure Analysis
# =========================================================================

def analyze_metric_eigenstructure(model, output_dir):
    """Analyze the learned metric tensor's eigenstructure."""
    print("=" * 60)
    print("1. Metric Tensor Eigenstructure")
    print("=" * 60)

    eigenvalues, eigenvectors = model.metric.eigendecomposition()
    eigenvalues = eigenvalues.cpu().numpy()
    eigenvectors = eigenvectors.cpu().numpy()

    # Summary stats
    print(f"  Dimension: {len(eigenvalues)}")
    print(f"  Eigenvalue range: [{eigenvalues[-1]:.6f}, {eigenvalues[0]:.6f}]")
    print(f"  Anisotropy (λ_max / λ_min): {eigenvalues[0] / eigenvalues[-1]:.1f}")
    print(f"  Condition number: {eigenvalues[0] / eigenvalues[-1]:.1f}")

    # Effective dimensionality: how many eigenvalues carry most of the "weight"
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    eff_dim_90 = np.searchsorted(cumulative, 0.9) + 1
    eff_dim_95 = np.searchsorted(cumulative, 0.95) + 1
    print(f"  Effective dim (90% variance): {eff_dim_90}")
    print(f"  Effective dim (95% variance): {eff_dim_95}")

    # Plot eigenvalue spectrum
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Spectrum
    ax = axes[0]
    ax.bar(range(len(eigenvalues)), eigenvalues, color="steelblue", alpha=0.8)
    ax.set_xlabel("Eigendirection index (sorted by magnitude)")
    ax.set_ylabel("Eigenvalue (difficulty weight)")
    ax.set_title("Metric Tensor Eigenvalue Spectrum")
    ax.set_yscale("log")

    # Cumulative
    ax = axes[1]
    ax.plot(range(len(cumulative)), cumulative, "b-", linewidth=2)
    ax.axhline(y=0.9, color="r", linestyle="--", label="90%")
    ax.axhline(y=0.95, color="orange", linestyle="--", label="95%")
    ax.set_xlabel("Number of eigendirections")
    ax.set_ylabel("Cumulative fraction of total difficulty")
    ax.set_title("Cumulative Eigenvalue Fraction")
    ax.legend()

    # Top eigenvalues ratio
    ax = axes[2]
    ratios = eigenvalues / eigenvalues[-1]
    n_show = min(20, len(ratios))
    ax.barh(range(n_show), ratios[:n_show], color="steelblue", alpha=0.8)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"v{i}" for i in range(n_show)])
    ax.set_xlabel("Relative difficulty (λ_i / λ_min)")
    ax.set_title(f"Top-{n_show} Hardest Directions")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eigenvalue_spectrum.png"), dpi=150)
    plt.close()

    return eigenvalues, eigenvectors


# =========================================================================
# 2. Per-Pair Difficulty Decomposition
# =========================================================================

@torch.no_grad()
def decompose_pair_difficulties(model, loader, device, eigenvalues, eigenvectors):
    """
    For each pair, decompose total difficulty into contributions
    from each eigendirection.
    """
    print("\n" + "=" * 60)
    print("2. Per-Pair Difficulty Decomposition")
    print("=" * 60)

    model.eval()
    evals_t = torch.tensor(eigenvalues, device=device)
    evecs_t = torch.tensor(eigenvectors, device=device)

    all_decompositions = []
    all_targets = []
    all_total_d = []

    for ga, gb, targets in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}

        h_a = model.encoder(ga)
        h_b = model.encoder(gb)
        delta_h = h_b - h_a  # (batch, dim)

        # Project onto eigenbasis
        projections = delta_h @ evecs_t  # (batch, dim)

        # Difficulty per direction: λ_i * (v_i · Δh)²
        difficulty_per_dir = evals_t * (projections ** 2)  # (batch, dim)

        # Total difficulty
        total_d = torch.sqrt(difficulty_per_dir.sum(dim=-1) + 1e-8)

        # Fraction per direction
        fractions = difficulty_per_dir / (difficulty_per_dir.sum(dim=-1, keepdim=True) + 1e-8)

        all_decompositions.append(fractions.cpu().numpy())
        all_targets.append(targets.numpy())
        all_total_d.append(total_d.cpu().numpy())

    decompositions = np.concatenate(all_decompositions)
    targets = np.concatenate(all_targets)
    total_d = np.concatenate(all_total_d)

    # Which eigendirections dominate?
    mean_fractions = decompositions.mean(axis=0)
    top_dirs = np.argsort(mean_fractions)[::-1]

    print(f"  Mean difficulty fraction by top eigendirections:")
    for i in top_dirs[:10]:
        print(f"    v{i}: {mean_fractions[i]:.4f} ({mean_fractions[i]*100:.1f}%)")

    # Correlation: total NTM distance vs target
    pr, pp = pearsonr(total_d, np.abs(targets))
    print(f"\n  Correlation |NTM distance| vs |target|: r={pr:.4f} (p={pp:.2e})")

    return decompositions, targets, total_d


# =========================================================================
# 3. Atom-Level Attribution
# =========================================================================

def atom_attribution(model, smi_a, smi_b, device, eigenvalues, eigenvectors):
    """
    Compute per-atom contribution to transformation difficulty.
    
    Uses gradient of NTM distance w.r.t. node features to identify
    which atoms in each molecule drive the difficulty.
    """
    graph_a = smiles_to_graph(smi_a)
    graph_b = smiles_to_graph(smi_b)
    if graph_a is None or graph_b is None:
        return None, None

    # Batch single graphs
    ga = batch_graphs([graph_a])
    gb = batch_graphs([graph_b])
    ga = {k: v.to(device) for k, v in ga.items()}
    gb = {k: v.to(device) for k, v in gb.items()}

    # Enable gradients on node features
    ga["node_feats"].requires_grad_(True)
    gb["node_feats"].requires_grad_(True)

    model.eval()
    h_a = model.encoder(ga)
    h_b = model.encoder(gb)
    d_m = model.metric.compute_distance(h_a, h_b)

    # Backprop to node features
    d_m.backward()

    attr_a = ga["node_feats"].grad.norm(dim=-1).detach().cpu().numpy()
    attr_b = gb["node_feats"].grad.norm(dim=-1).detach().cpu().numpy()

    # Normalize
    attr_a = attr_a / (attr_a.max() + 1e-8)
    attr_b = attr_b / (attr_b.max() + 1e-8)

    return attr_a, attr_b


def batch_atom_attribution(model, df, device, eigenvalues, eigenvectors, n_samples=100):
    """Run atom attribution on a sample of pairs."""
    print("\n" + "=" * 60)
    print("3. Atom-Level Attribution")
    print("=" * 60)

    col_a, col_b, col_t = df.columns
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    results = []
    for _, row in sample.iterrows():
        attr_a, attr_b = atom_attribution(
            model, row[col_a], row[col_b], device, eigenvalues, eigenvectors
        )
        if attr_a is not None:
            mol_a = Chem.MolFromSmiles(row[col_a])
            mol_b = Chem.MolFromSmiles(row[col_b])

            # Find most important atoms
            top_a = np.argsort(attr_a)[::-1][:3]
            top_b = np.argsort(attr_b)[::-1][:3]

            results.append({
                "smi_a": row[col_a],
                "smi_b": row[col_b],
                "target": row[col_t],
                "max_attr_a": float(attr_a.max()),
                "max_attr_b": float(attr_b.max()),
                "mean_attr_a": float(attr_a.mean()),
                "mean_attr_b": float(attr_b.mean()),
                "top_atoms_a": top_a.tolist(),
                "top_atoms_b": top_b.tolist(),
                "n_atoms_a": len(attr_a),
                "n_atoms_b": len(attr_b),
            })

    print(f"  Computed attributions for {len(results)} pairs")
    print(f"  Mean max attribution (Mol A): "
          f"{np.mean([r['max_attr_a'] for r in results]):.4f}")
    print(f"  Mean max attribution (Mol B): "
          f"{np.mean([r['max_attr_b'] for r in results]):.4f}")

    return results


# =========================================================================
# 4. Transformation Type Clustering
# =========================================================================

def cluster_transformations(decompositions, targets, output_dir, n_clusters=8):
    """
    Cluster transformations by their difficulty profile to identify
    distinct transformation types.
    """
    print("\n" + "=" * 60)
    print("4. Transformation Type Clustering")
    print("=" * 60)

    # Use top eigendirections for clustering
    n_top = min(20, decompositions.shape[1])
    features = decompositions[:, :n_top]

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Analyze each cluster
    print(f"  {n_clusters} clusters identified:")
    cluster_stats = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_decomp = decompositions[mask]
        cluster_targets = targets[mask]

        # Dominant direction for this cluster
        mean_profile = cluster_decomp.mean(axis=0)
        dominant_dir = np.argmax(mean_profile)

        stats = {
            "cluster": c,
            "size": int(mask.sum()),
            "dominant_direction": int(dominant_dir),
            "dominant_fraction": float(mean_profile[dominant_dir]),
            "mean_target": float(cluster_targets.mean()),
            "std_target": float(cluster_targets.std()),
        }
        cluster_stats.append(stats)

        print(f"    Cluster {c}: n={mask.sum():>6d}  "
              f"dominant=v{dominant_dir} ({mean_profile[dominant_dir]:.1%})  "
              f"mean_target={cluster_targets.mean():.4f}")

    # Visualize clusters in 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    scatter = ax.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels, cmap="tab10", s=5, alpha=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Transformation Types (by difficulty profile)")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    # Cluster profiles
    ax = axes[1]
    for c in range(n_clusters):
        mask = labels == c
        mean_profile = decompositions[mask, :n_top].mean(axis=0)
        ax.plot(range(n_top), mean_profile, label=f"C{c} (n={mask.sum()})", alpha=0.8)
    ax.set_xlabel("Eigendirection index")
    ax.set_ylabel("Mean difficulty fraction")
    ax.set_title("Cluster Difficulty Profiles")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transformation_clusters.png"), dpi=150)
    plt.close()

    return labels, cluster_stats


# =========================================================================
# 5. Summary Visualization
# =========================================================================

def summary_plots(decompositions, targets, total_d, output_dir):
    """Generate summary plots."""
    print("\n" + "=" * 60)
    print("5. Summary Visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # NTM distance vs target
    ax = axes[0, 0]
    ax.scatter(total_d, np.abs(targets), s=3, alpha=0.3, c="steelblue")
    pr, _ = pearsonr(total_d, np.abs(targets))
    ax.set_xlabel("NTM Distance d_M(A, B)")
    ax.set_ylabel("|Stderr Difference|")
    ax.set_title(f"NTM Distance vs Target (r={pr:.3f})")

    # Difficulty concentration: how many directions matter?
    ax = axes[0, 1]
    sorted_fracs = np.sort(decompositions, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_fracs, axis=1)
    for pct in [0.5, 0.8, 0.9, 0.95]:
        n_dirs = np.argmax(cumulative >= pct, axis=1) + 1
        ax.hist(n_dirs, bins=50, alpha=0.5, label=f"{pct:.0%} of difficulty")
    ax.set_xlabel("Number of eigendirections")
    ax.set_ylabel("Count")
    ax.set_title("How Concentrated Is Difficulty?")
    ax.legend()

    # Top eigendirection importance
    ax = axes[1, 0]
    top_k = min(20, decompositions.shape[1])
    mean_fracs = decompositions.mean(axis=0)[:top_k]
    ax.bar(range(top_k), mean_fracs, color="steelblue", alpha=0.8)
    ax.set_xlabel("Eigendirection index")
    ax.set_ylabel("Mean difficulty fraction")
    ax.set_title(f"Top-{top_k} Eigendirection Importance")

    # Distribution of difficulty by hard vs easy
    ax = axes[1, 1]
    hard_frac = decompositions[:, :5].sum(axis=1)  # top-5 hardest
    easy_frac = decompositions[:, -5:].sum(axis=1)  # top-5 easiest
    ax.scatter(hard_frac, easy_frac, c=np.abs(targets), cmap="viridis",
               s=5, alpha=0.3)
    ax.set_xlabel("Fraction in top-5 hard directions")
    ax.set_ylabel("Fraction in top-5 easy directions")
    ax.set_title("Hard vs Easy Direction Dominance")
    plt.colorbar(ax.collections[0], ax=ax, label="|Target|")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "difficulty_summary.png"), dpi=150)
    plt.close()

    print("  Saved summary plots")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--ntm_model_dir", default="../results/ntm")
    parser.add_argument("--output_dir", default="../results/decomposition")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")

    # Load trained NTM model
    print("Loading NTM model...")
    with open(os.path.join(args.ntm_model_dir, "results.json")) as f:
        ntm_args = json.load(f)["args"]

    model = NTMModel(
        ATOM_DIM, BOND_DIM,
        ntm_args["hidden_dim"], ntm_args["num_layers"],
        metric_type=ntm_args["metric_type"],
        dropout=ntm_args.get("dropout", 0.1),
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(args.ntm_model_dir, "best_model.pt"),
                    map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print("  Model loaded.")

    # Load test data
    print("Loading test data...")
    df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    ds_test = PairDataset(df_test)
    from torch.utils.data import DataLoader
    loader_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pair, num_workers=4, pin_memory=True,
    )

    # 1. Eigenstructure
    eigenvalues, eigenvectors = analyze_metric_eigenstructure(model, args.output_dir)

    # 2. Per-pair decomposition
    decompositions, targets, total_d = decompose_pair_difficulties(
        model, loader_test, DEVICE, eigenvalues, eigenvectors
    )

    # 3. Atom attribution
    attr_results = batch_atom_attribution(
        model, df_test, DEVICE, eigenvalues, eigenvectors, n_samples=200
    )

    # 4. Clustering
    labels, cluster_stats = cluster_transformations(
        decompositions, targets, args.output_dir, n_clusters=args.n_clusters
    )

    # 5. Summary plots
    summary_plots(decompositions, targets, total_d, args.output_dir)

    # Save all results
    np.savez(
        os.path.join(args.output_dir, "decompositions.npz"),
        decompositions=decompositions,
        targets=targets,
        total_d=total_d,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        cluster_labels=labels,
    )
    with open(os.path.join(args.output_dir, "cluster_stats.json"), "w") as f:
        json.dump(cluster_stats, f, indent=2)
    with open(os.path.join(args.output_dir, "atom_attributions.json"), "w") as f:
        json.dump(attr_results, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
