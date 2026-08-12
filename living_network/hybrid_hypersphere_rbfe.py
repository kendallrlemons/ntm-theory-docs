"""
hybrid_hypersphere_rbfe.py
===========================
Combines the two independently-validated ideas from hybrid_topo_rbfe.py and
hypersphere_rbfe.py into a single transformation-aware hyperspherical model,
and benchmarks it against both of its parents plus the LOMAP-like/Tanimoto
baselines.

Background
----------
- hybrid_topo_rbfe.py:  merges ligand A + ligand B into ONE hybrid-topology
  graph (via MCS) and encodes it with an MPNN -> single embedding per pair ->
  regression head predicts |SE|. Transformation-aware, but no hyperspherical
  geometry (flat Euclidean embedding, pure MSE loss).
- hypersphere_rbfe.py:  encodes ligand A and ligand B INDEPENDENTLY (no
  knowledge of the transformation partner) with an MPNN, L2-normalizes each
  onto the unit hypersphere, and trains with a 3-component loss (continuous
  contrastive + dispersion + regression).

This script reuses the hybrid-topology encoder (HybridGNN.encode(), imported
from hybrid_topo_rbfe.py) to get a single transformation-aware embedding per
pair, L2-normalizes it onto the hypersphere, and trains two variants:

  Phase 2 (ablation):  hybrid encoder + L2-norm + regression-only loss.
                        Isolates whether hyperspherical geometry ALONE (with
                        no contrastive/dispersion pressure) helps over the
                        flat-Euclidean hybrid_topo_rbfe.py baseline.
  Phase 3 (combined):  hybrid encoder + L2-norm + full 3-component loss.

Key architectural note — reformulated contrastive loss
--------------------------------------------------------
hypersphere_rbfe.py's continuous_contrastive_loss operates on TWO independent
per-ligand embeddings (h_a, h_b) within the SAME pair, targeting
cos(h_a, h_b) = cos(pi * SE_norm) (similar ligands -> low SE -> close on the
sphere). HybridGNN produces only ONE embedding per pair (the merged
transformation graph), so there is no "h_a"/"h_b" to compare within a pair.

Instead, batch_pairwise_contrastive_loss reformulates the same idea ACROSS
pairs in a batch: two different transformations i, j with similar predicted
difficulty (|SE_i - SE_j| small) should embed close together on the sphere;
transformations with very different difficulty should be far apart. Dispersion
(cluster-center spread) and the regression head both carry over unchanged.

Phases
------
1. Data loading & split (reuses hybrid_topo_rbfe.phase1_load for exact index
   alignment with the cached graphs_all.pt from mcs_precompute.py).
2. Train ablation model: hybrid encoder + L2-norm + regression-only.
3. Train combined model: hybrid encoder + L2-norm + full 3-component loss.
4. Geometric validation of the combined model (OOD score / regression vs |SE|,
   embedding visualization).
5. Six-way comparison on one fixed test sample: Tanimoto, LOMAP-like,
   independent hypersphere (loaded checkpoint), independent MCS/GNN (loaded
   checkpoint), hybrid+L2 ablation (this script), hybrid+hypersphere combined
   (this script).

Usage
-----
    python hybrid_hypersphere_rbfe.py \\
        --graph_cache_dir /path/to/mcs_precompute_output \\
        --hybrid_ckpt     /path/to/hybrid_topo_rbfe/phase3_hybrid/best.pt \\
        --hypersphere_ckpt /path/to/hypersphere_rbfe/phase3_hypersphere/best.pt \\
        --output_dir ./results_hybrid_hypersphere

Requires --hybrid_ckpt and --hypersphere_ckpt to have been trained with the
SAME --hidden_dim/--num_layers/--embed_dim/--dropout defaults as this script
(defaults match both parent scripts), since checkpoints only store weights,
not architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    roc_auc_score,
)

# Reuse code from the two parent scripts rather than duplicating it. Both
# modules guard their own execution behind `if __name__ == "__main__":`, so
# importing them only runs their module-level setup (device detection,
# constant/class definitions) — no training or CLI parsing happens on import.
import hybrid_topo_rbfe as hto
import hypersphere_rbfe as hyp

DEVICE = hto.DEVICE

# =============================================================================
# CONFIG — defaults
# =============================================================================

DEFAULT_INPUT       = hto.DEFAULT_INPUT
DEFAULT_OUTPUT_DIR  = "./results_hybrid_hypersphere"
DEFAULT_SAMPLE_SIZE = 0
DEFAULT_SEED        = 42
DEFAULT_PHASES      = "1,2,3,4,5"
DEFAULT_NUM_CLUSTERS = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",         default=DEFAULT_INPUT)
    p.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--phases",        default=DEFAULT_PHASES)
    p.add_argument("--sample_size",   type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed",          type=int, default=DEFAULT_SEED)
    p.add_argument("--val_frac",      type=float, default=0.1,
                   help="Passed through to hto.phase1_load, which requires it.")
    p.add_argument("--test_frac",     type=float, default=0.1,
                   help="Passed through to hto.phase1_load, which requires it.")
    p.add_argument("--hidden_dim",    type=int, default=128)
    p.add_argument("--num_layers",    type=int, default=4)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--batch_size",    type=int, default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--epochs",        type=int, default=60)
    p.add_argument("--patience",      type=int, default=20)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--mcs_timeout",   type=int, default=5,
                   help="Only used if --graph_cache_dir has no cached graphs.")
    p.add_argument("--num_clusters",  type=int, default=DEFAULT_NUM_CLUSTERS,
                   help="K for dispersion loss (combined model only)")
    p.add_argument("--cluster_ema",   type=float, default=0.9)
    p.add_argument("--w_contrastive", type=float, default=1.0,
                   help="Weight for the batch-pairwise contrastive loss "
                        "(combined model only; ablation model always uses 0)")
    p.add_argument("--w_dispersion",  type=float, default=0.5,
                   help="Weight for the dispersion loss "
                        "(combined model only; ablation model always uses 0)")
    p.add_argument("--w_regression",  type=float, default=1.0)
    p.add_argument("--graph_cache_dir", default=None,
                   help="Directory containing graphs_all.pt + cleaned.pkl "
                        "produced by mcs_precompute.py.")
    p.add_argument("--hybrid_ckpt",     default=None,
                   help="Path to a trained hybrid_topo_rbfe.py best.pt "
                        "checkpoint, used as the independent MCS/GNN "
                        "comparison arm in Phase 5.")
    p.add_argument("--hypersphere_ckpt", default=None,
                   help="Path to a trained hypersphere_rbfe.py best.pt "
                        "checkpoint, used as the independent hypersphere "
                        "comparison arm in Phase 5.")
    return p.parse_args()


def set_seed(seed: int):
    hto.set_seed(seed)


# =============================================================================
# Reformulated hyperspherical loss (batch-pairwise, single embedding/pair)
# =============================================================================

def batch_pairwise_contrastive_loss(emb: torch.Tensor, se: torch.Tensor,
                                     se_scale: float) -> torch.Tensor:
    """
    Generalizes hypersphere_rbfe.py's continuous_contrastive_loss to a single
    embedding per sample (rather than two, h_a/h_b, within a pair).

    For every pair of SAMPLES i, j in the batch (not to be confused with the
    ligand pairs A/B — here "i"/"j" index different transformations):
        target_cos(i, j) = cos(pi * |SE_norm_i - SE_norm_j|)
        pred_cos(i, j)   = emb_i . emb_j          (emb is L2-normalized)

    Transformations of similar difficulty should embed close together
    (target_cos -> 1 when |SE_i - SE_j| -> 0); transformations of very
    different difficulty should be pushed apart (target_cos -> -1 when
    |SE_i - SE_j| -> se_scale).
    """
    se_norm = (se / max(se_scale, 1e-8)).clamp(0, 1)
    diff = (se_norm.unsqueeze(0) - se_norm.unsqueeze(1)).abs()
    target_cos = torch.cos(np.pi * diff)
    pred_cos = emb @ emb.T
    B = emb.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb.device)
    mask = ~torch.eye(B, dtype=torch.bool, device=emb.device)
    return F.mse_loss(pred_cos[mask], target_cos[mask])


# =============================================================================
# Shared training loop for Phase 2 (ablation) and Phase 3 (combined)
# =============================================================================

def train_model(df: pd.DataFrame, args, output_dir: str,
                use_hyperspherical_losses: bool, run_name: str) -> Dict:
    """Train HybridGNN.encode() -> L2-norm -> regression head.

    If use_hyperspherical_losses is True, also trains cluster centers and
    adds the batch-pairwise contrastive + dispersion losses (the "combined"
    model). If False, trains with regression loss only (the "ablation":
    isolates whether L2-normalizing the hybrid embedding alone helps).
    """
    print("\n" + "=" * 70)
    print(f"PHASE — Training ({run_name})")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, f"phase_{run_name}")
    os.makedirs(phase_dir, exist_ok=True)

    train_orig = df[df.split == "train"].index
    val_orig   = df[df.split == "val"].index
    train_df   = df[df.split == "train"].reset_index(drop=True)
    val_df     = df[df.split == "val"].reset_index(drop=True)

    print("\n[1] Building data loaders")
    phase2_dir = (
        args.graph_cache_dir
        if args.graph_cache_dir
        else os.path.join(output_dir, "phase2_hybrid")
    )
    train_graphs = hto._load_graphs("train", phase2_dir, df_index=train_orig)
    val_graphs   = hto._load_graphs("val",   phase2_dir, df_index=val_orig)
    _, train_loader = hto._build_loader(train_df, args, shuffle=True,  graphs=train_graphs)
    _, val_loader   = hto._build_loader(val_df,   args, shuffle=False, graphs=val_graphs)

    se_scale = float(np.quantile(train_df["SE_abs"].values, 0.99))
    print(f"    SE scale (q99): {se_scale:.6f}")

    encoder = hto.HybridGNN(
        hto.ATOM_DIM, hto.BOND_DIM, args.hidden_dim, args.num_layers,
        args.embed_dim, args.dropout,
    ).to(DEVICE)
    clusters = hyp.ClusterCenters(
        args.num_clusters, args.embed_dim, ema=args.cluster_ema,
    ).to(DEVICE) if use_hyperspherical_losses else None

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"    Params: {n_params:,}  hyperspherical_losses={use_hyperspherical_losses}")

    optim = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=3, min_lr=1e-6,
    )
    best_val_rho = -float("inf")
    best_ckpt = os.path.join(phase_dir, "best.pt")
    patience_ctr = 0
    history = []

    w_c = args.w_contrastive if use_hyperspherical_losses else 0.0
    w_d = args.w_dispersion  if use_hyperspherical_losses else 0.0
    w_r = args.w_regression

    print("\n[2] Training")
    for epoch in range(args.epochs):
        encoder.train()
        t0 = time.time()
        sums = {"total": 0.0, "contrast": 0.0, "disp": 0.0, "reg": 0.0, "n": 0}

        for batch, y in train_loader:
            if batch is None:
                continue
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            y = y.to(DEVICE)

            emb = encoder.encode(batch)
            emb_n = F.normalize(emb, p=2, dim=-1)

            if use_hyperspherical_losses:
                with torch.no_grad():
                    clusters.update(emb_n.detach())
                L_c = batch_pairwise_contrastive_loss(emb_n, y, se_scale)
                L_d = clusters.dispersion_loss()
            else:
                L_c = torch.tensor(0.0, device=DEVICE)
                L_d = torch.tensor(0.0, device=DEVICE)

            pred = encoder.head(emb_n).squeeze(-1)
            L_r = F.mse_loss(pred, y)

            L = w_c * L_c + w_d * L_d + w_r * L_r

            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optim.step()

            bs = y.size(0)
            sums["total"]    += L.item() * bs
            sums["contrast"] += float(L_c.item()) * bs
            sums["disp"]     += float(L_d.item()) * bs
            sums["reg"]      += L_r.item() * bs
            sums["n"]        += bs

        # ---- Validation ----
        encoder.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch, y in val_loader:
                if batch is None:
                    continue
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                emb_n = F.normalize(encoder.encode(batch), p=2, dim=-1)
                pred = encoder.head(emb_n).squeeze(-1)
                val_preds.append(pred.cpu().numpy())
                val_true.append(y.numpy())

        val_preds = np.concatenate(val_preds)
        val_true  = np.concatenate(val_true)
        val_mse   = float(np.mean((val_preds - val_true) ** 2))
        val_rho   = float(spearmanr(val_preds, val_true).correlation)
        dt        = time.time() - t0

        prev_lr = optim.param_groups[0]["lr"]
        scheduler.step(val_rho)
        new_lr  = optim.param_groups[0]["lr"]
        lr_note = f"  lr↓{new_lr:.2e}" if new_lr < prev_lr else ""

        n = max(sums["n"], 1)
        rec = {
            "epoch": epoch + 1,
            "train_total": sums["total"] / n,
            "train_contrast": sums["contrast"] / n,
            "train_disp": sums["disp"] / n,
            "train_reg": sums["reg"] / n,
            "val_mse": val_mse,
            "val_rho": val_rho,
            "time_sec": dt,
        }
        history.append(rec)
        print(f"  ep {epoch+1:3d}  "
              f"train {rec['train_total']:.4f} "
              f"(c={rec['train_contrast']:.3f} d={rec['train_disp']:.3f} "
              f"r={rec['train_reg']:.4f})  "
              f"val_mse={val_mse:.4f}  val_ρ={val_rho:.3f}  ({dt:.0f}s){lr_note}")

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            ckpt = {
                "encoder": encoder.state_dict(),
                "args": vars(args),
                "se_scale": se_scale,
                "use_hyperspherical_losses": use_hyperspherical_losses,
            }
            if use_hyperspherical_losses:
                ckpt["clusters"] = clusters.state_dict()
            torch.save(ckpt, best_ckpt)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    with open(os.path.join(phase_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {
        "ckpt": best_ckpt,
        "phase_dir": phase_dir,
        "se_scale": se_scale,
        "use_hyperspherical_losses": use_hyperspherical_losses,
    }


def _load_combined_model(ckpt_path: str, args):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder = hto.HybridGNN(
        hto.ATOM_DIM, hto.BOND_DIM, args.hidden_dim, args.num_layers,
        args.embed_dim, args.dropout,
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    clusters = None
    if ckpt.get("use_hyperspherical_losses"):
        clusters = hyp.ClusterCenters(
            args.num_clusters, args.embed_dim, ema=args.cluster_ema,
        ).to(DEVICE)
        clusters.load_state_dict(ckpt["clusters"])
    return encoder, clusters, ckpt["se_scale"]


@torch.no_grad()
def _encode_all_combined(encoder, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (embeddings, true |SE|, regression predictions)."""
    EMB, Y, PRED = [], [], []
    for batch, y in loader:
        if batch is None:
            continue
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        emb_n = F.normalize(encoder.encode(batch), p=2, dim=-1)
        pred = encoder.head(emb_n).squeeze(-1)
        EMB.append(emb_n.cpu().numpy())
        Y.append(y.numpy())
        PRED.append(pred.cpu().numpy())
    return np.concatenate(EMB), np.concatenate(Y), np.concatenate(PRED)


# =============================================================================
# Phase 4 — Geometric validation (combined model only)
# =============================================================================

def phase4_validate(df: pd.DataFrame, args, ckpt_info: Dict, output_dir: str) -> Dict:
    print("\n" + "=" * 70)
    print("PHASE 4 — Geometric Validation (combined model)")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, "phase4_combined")
    os.makedirs(phase_dir, exist_ok=True)

    encoder, clusters, se_scale = _load_combined_model(ckpt_info["ckpt"], args)
    if clusters is None:
        print("    Skipping: this checkpoint was trained without hyperspherical "
              "losses (no cluster centers). Run Phase 4 on the combined model.")
        return {}

    test_orig = df[df.split == "test"].index
    test_df   = df[df.split == "test"].reset_index(drop=True)
    phase2_dir = (
        args.graph_cache_dir
        if args.graph_cache_dir
        else os.path.join(output_dir, "phase2_hybrid")
    )
    test_graphs = hto._load_graphs("test", phase2_dir, df_index=test_orig)
    _, test_loader = hto._build_loader(test_df, args, shuffle=False, graphs=test_graphs)

    print("\n[4.1] Encoding test set")
    EMB, Y, pred = _encode_all_combined(encoder, test_loader)

    centers = clusters.centers.detach().cpu().numpy()
    sims = EMB @ centers.T
    ood = 1.0 - sims.max(axis=-1)

    def _corr(x, y, label):
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        print(f"    {label:>25}:  Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.1e})  "
              f"Spearman ρ={sr.correlation:+.3f} (p={sr.pvalue:.1e})")
        return {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.correlation), "spearman_p": float(sr.pvalue)}

    print("\n[4.2] Correlations with |SE|:")
    results = {
        "ood_score":       _corr(ood, Y, "ood_score"),
        "regression_head": _corr(pred, Y, "regression_head"),
    }
    results["test_mae"] = float(mean_absolute_error(Y, pred))

    np.savez(os.path.join(phase_dir, "scores.npz"),
             emb=EMB, y=Y, pred=pred, ood=ood)
    with open(os.path.join(phase_dir, "correlations.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Visualization: PCA 2D scatter ----
    print("\n[4.3] Visualizing embedding geometry")
    pca = PCA(n_components=2, random_state=0).fit(EMB)
    emb_2d = pca.transform(EMB)
    centers_2d = pca.transform(centers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=ood, s=3, alpha=0.4, cmap="viridis")
    axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", s=120, marker="*",
                    edgecolor="black", linewidth=0.8, label="cluster centers")
    plt.colorbar(sc, ax=axes[0], label="OOD score")
    axes[0].set_title("Pair embeddings — colored by OOD")
    axes[0].legend()

    sc = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=Y, s=3, alpha=0.4, cmap="magma",
                         vmin=np.quantile(Y, 0.05), vmax=np.quantile(Y, 0.95))
    axes[1].scatter(centers_2d[:, 0], centers_2d[:, 1], c="cyan", s=120, marker="*",
                    edgecolor="black", linewidth=0.8)
    plt.colorbar(sc, ax=axes[1], label="|SE|")
    axes[1].set_title("Pair embeddings — colored by |SE|")
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "embedding_visualization.png"), dpi=140)
    plt.close()

    # ---- 3D interactive sphere (PCA-3 re-normalized) ----
    pca3 = PCA(n_components=3, random_state=0).fit(EMB)
    emb_3d = pca3.transform(EMB)
    emb_3d = emb_3d / (np.linalg.norm(emb_3d, axis=1, keepdims=True) + 1e-9)
    centers_3d = pca3.transform(centers)
    centers_3d = centers_3d / (np.linalg.norm(centers_3d, axis=1, keepdims=True) + 1e-9)

    n_plot = min(8000, len(emb_3d))
    idx = np.random.RandomState(0).choice(len(emb_3d), n_plot, replace=False)
    pts, pt_ood, pt_se = emb_3d[idx], ood[idx], Y[idx]

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    sx, sy, sz = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)

    def _sphere_traces(color, cbar_title):
        return [
            go.Surface(x=sx, y=sy, z=sz, opacity=0.08, showscale=False,
                      colorscale=[[0, "lightgray"], [1, "lightgray"]], hoverinfo="skip"),
            go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                        marker=dict(size=2.5, color=color, colorscale="Viridis",
                                   opacity=0.75, showscale=True,
                                   colorbar=dict(title=cbar_title)),
                        name="pair embeddings"),
            go.Scatter3d(x=centers_3d[:, 0], y=centers_3d[:, 1], z=centers_3d[:, 2],
                        mode="markers+text",
                        marker=dict(size=10, color="red", symbol="diamond",
                                   line=dict(color="black", width=2)),
                        text=[f"c{i}" for i in range(len(centers_3d))],
                        textposition="top center", name="cluster centers"),
        ]

    fig3 = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Colored by OOD score", "Colored by |SE|"),
        horizontal_spacing=0.05,
    )
    for t in _sphere_traces(pt_ood, "OOD"):
        fig3.add_trace(t, row=1, col=1)
    for t in _sphere_traces(np.clip(pt_se, 0, np.quantile(pt_se, 0.95)), "|SE|"):
        fig3.add_trace(t, row=1, col=2)
    fig3.update_layout(
        title="Combined model — pair-transformation embeddings on S²",
        height=700, width=1400, template="plotly_white",
        scene=dict(aspectmode="cube"), scene2=dict(aspectmode="cube"),
    )
    fig3.write_html(os.path.join(phase_dir, "sphere_3d_interactive.html"))
    print(f"    Saved embedding_visualization.png and sphere_3d_interactive.html")

    return results


# =============================================================================
# Phase 5 — Six-way comparison
# =============================================================================

def _predict_aligned(graphs: list, se_values: np.ndarray, predict_fn,
                     batch_size: int) -> np.ndarray:
    """Run predict_fn over `graphs` in chunks, preserving row alignment even
    when some entries are None (MCS/featurization failures). Returns an array
    of length len(graphs) with np.nan for rows that were None — this avoids
    the silent index-shift that DataLoader's collate-drop-None causes when
    predictions need to be compared row-for-row against other scores."""
    n = len(graphs)
    out = np.full(n, np.nan, dtype=np.float64)
    idx = 0
    while idx < n:
        chunk_idx = list(range(idx, min(idx + batch_size, n)))
        kept = [i for i in chunk_idx if graphs[i] is not None]
        if kept:
            batch_list = [(graphs[i], float(se_values[i])) for i in kept]
            batch, _ = hto._collate(batch_list)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                preds = predict_fn(batch)
            out[kept] = preds.detach().cpu().numpy()
        idx += batch_size
    return out


def _predict_hypersphere_aligned(eval_df: pd.DataFrame, encoder, head,
                                 batch_size: int) -> np.ndarray:
    """Row-aligned prediction using the independent hypersphere model, which
    encodes each ligand from raw SMILES (no cached hybrid graph)."""
    n = len(eval_df)
    out = np.full(n, np.nan, dtype=np.float64)
    smi_a_col, smi_b_col = hto.SMILES_A_COLUMN, hto.SMILES_B_COLUMN
    idx = 0
    while idx < n:
        chunk_idx = list(range(idx, min(idx + batch_size, n)))
        gas, gbs, kept = [], [], []
        for i in chunk_idx:
            row = eval_df.iloc[i]
            ga = hyp.smiles_to_graph(row[smi_a_col])
            gb = hyp.smiles_to_graph(row[smi_b_col])
            if ga is not None and gb is not None:
                gas.append(ga); gbs.append(gb); kept.append(i)
        if kept:
            ga_b = {k: v.to(DEVICE) for k, v in hyp.batch_graphs(gas).items()}
            gb_b = {k: v.to(DEVICE) for k, v in hyp.batch_graphs(gbs).items()}
            with torch.no_grad():
                h_a = encoder(ga_b)
                h_b = encoder(gb_b)
                pred = head(h_a, h_b)
            out[kept] = pred.cpu().numpy()
        idx += batch_size
    return out


def phase5_compare(df: pd.DataFrame, args, ablation_ckpt: Dict,
                   combined_ckpt: Dict, output_dir: str) -> Dict:
    print("\n" + "=" * 70)
    print("PHASE 5 — Six-Way Comparison")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, "phase5_compare")
    os.makedirs(phase_dir, exist_ok=True)

    # ---- One fixed eval sample, shared across every method ----
    test_df  = df[df.split == "test"].reset_index(drop=True)
    n_eval   = min(20_000, len(test_df))
    eval_pos = test_df.sample(n=n_eval, random_state=args.seed).index
    eval_df  = test_df.loc[eval_pos].reset_index(drop=True)
    test_orig = df[df.split == "test"].index
    eval_orig = test_orig[eval_pos]
    Y_true = eval_df["SE_abs"].values.astype(np.float64)
    print(f"    Evaluating on {n_eval:,} test pairs (shared across all methods)")

    smi_a_col, smi_b_col = hto.SMILES_A_COLUMN, hto.SMILES_B_COLUMN

    # ---- Structural baselines ----
    print("\n[5.1] Computing Tanimoto + real-LOMAP baselines")
    tani = np.full(n_eval, np.nan)
    lomap_score = np.full(n_eval, np.nan)
    for i in range(n_eval):
        row = eval_df.iloc[i]
        tani[i]        = hto._tanimoto_score(row[smi_a_col], row[smi_b_col])
        lomap_score[i] = hto._lomap_like_score(row[smi_a_col], row[smi_b_col], args.mcs_timeout)
        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{n_eval}")

    # ---- Hybrid-graph-based methods (share the same cached graphs) ----
    phase2_dir = (
        args.graph_cache_dir
        if args.graph_cache_dir
        else os.path.join(output_dir, "phase2_hybrid")
    )
    eval_graphs = hto._load_graphs("test", phase2_dir, df_index=eval_orig)
    if eval_graphs is None:
        raise RuntimeError(
            "No cached graphs found. Set --graph_cache_dir to the output of "
            "mcs_precompute.py."
        )

    print("\n[5.2] Independent MCS/GNN (hybrid_topo_rbfe.py)")
    pred_mcs = np.full(n_eval, np.nan)
    if args.hybrid_ckpt:
        hybrid_model = hto.HybridGNN(
            hto.ATOM_DIM, hto.BOND_DIM, args.hidden_dim, args.num_layers,
            args.embed_dim, args.dropout,
        ).to(DEVICE)
        hybrid_ckpt = torch.load(args.hybrid_ckpt, map_location=DEVICE, weights_only=False)
        hybrid_model.load_state_dict(hybrid_ckpt["model"])
        hybrid_model.eval()
        pred_mcs = _predict_aligned(eval_graphs, Y_true, hybrid_model, args.batch_size)
    else:
        print("    Skipping (--hybrid_ckpt not provided)")

    print("\n[5.3] Independent hypersphere (hypersphere_rbfe.py)")
    pred_hyp = np.full(n_eval, np.nan)
    if args.hypersphere_ckpt:
        hyp_encoder, hyp_head, _, _, _ = hyp._load_model(args.hypersphere_ckpt, args)
        pred_hyp = _predict_hypersphere_aligned(eval_df, hyp_encoder, hyp_head, args.batch_size)
    else:
        print("    Skipping (--hypersphere_ckpt not provided)")

    print("\n[5.4] Hybrid + L2-norm (ablation, this script's Phase 2)")
    abl_encoder, _, _ = _load_combined_model(ablation_ckpt["ckpt"], args)
    pred_abl = _predict_aligned(
        eval_graphs, Y_true,
        lambda b: abl_encoder.head(F.normalize(abl_encoder.encode(b), p=2, dim=-1)).squeeze(-1),
        args.batch_size,
    )

    print("\n[5.5] Hybrid + Hypersphere (combined, this script's Phase 3)")
    comb_encoder, _, _ = _load_combined_model(combined_ckpt["ckpt"], args)
    pred_comb = _predict_aligned(
        eval_graphs, Y_true,
        lambda b: comb_encoder.head(F.normalize(comb_encoder.encode(b), p=2, dim=-1)).squeeze(-1),
        args.batch_size,
    )

    # ---- Common valid mask across ALL six scores ----
    all_scores = {
        "Tanimoto (1-sim)":              tani,
        "LOMAP (real, 1-sim)":           lomap_score,
        "Hypersphere (independent)":     pred_hyp,
        "MCS/GNN (independent)":         pred_mcs,
        "Hybrid + L2-norm (ablation)":   pred_abl,
        "Hybrid + Hypersphere (combined)": pred_comb,
    }
    valid = np.ones(n_eval, dtype=bool)
    for name, s in all_scores.items():
        valid &= ~np.isnan(s)
    print(f"\n    {valid.sum():,}/{n_eval:,} rows valid across all six methods")
    Y_valid = Y_true[valid]

    thresh = np.quantile(Y_valid, 0.75)
    y_bin = (Y_valid >= thresh).astype(int)

    def _evaluate(score, name):
        s = score[valid]
        pr = pearsonr(s, Y_valid)
        sr = spearmanr(s, Y_valid)
        try:
            auroc = roc_auc_score(y_bin, s)
            auprc = average_precision_score(y_bin, s)
            s_thresh = np.quantile(s, 0.75)
            mcc = matthews_corrcoef(y_bin, (s >= s_thresh).astype(int))
        except Exception:
            auroc = auprc = mcc = np.nan
        return {
            "method": name,
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.correlation), "spearman_p": float(sr.pvalue),
            "auroc_hard_top25": float(auroc),
            "auprc_hard_top25": float(auprc),
            "mcc_hard_top25": float(mcc),
        }

    rows = [_evaluate(s, name) for name, s in all_scores.items()]
    comp_df = pd.DataFrame(rows).set_index("method")
    print("\n[5.6] Results:")
    print(comp_df.round(4).to_string())
    comp_df.to_csv(os.path.join(phase_dir, "six_way_comparison.csv"))

    _metrics = [
        ("spearman_r",       "Spearman ρ"),
        ("auroc_hard_top25", "AUROC (top 25% hard)"),
        ("auprc_hard_top25", "AUPRC (top 25% hard)"),
        ("mcc_hard_top25",   "MCC (top 25% hard)"),
    ]
    n_m = len(comp_df)
    _colors = ["gray", "gray", "darkorange", "steelblue", "seagreen", "mediumpurple"][:n_m]
    fig, axes = plt.subplots(1, len(_metrics), figsize=(22, 5))
    for ax, (metric, title) in zip(axes, _metrics):
        vals = comp_df[metric].values
        ax.bar(range(n_m), vals, color=_colors, edgecolor="black")
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(comp_df.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.axhline(y=0, color="black", linewidth=0.5)
    plt.suptitle("Six-Way Comparison — Hybrid+Hypersphere vs Parents vs Baselines", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "six_way_comparison.png"), dpi=140)
    plt.close()
    print(f"    Saved six_way_comparison.png / .csv")

    with open(os.path.join(phase_dir, "results.json"), "w") as f:
        json.dump(comp_df.to_dict(orient="index"), f, indent=2)

    return {"comparison": comp_df.to_dict(orient="index"), "n_valid": int(valid.sum())}


# =============================================================================
# Main
# =============================================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)  # flush on every newline (SLURM-safe)
    args   = parse_args()
    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Phases: {phases}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    df = None
    ablation_ckpt = None
    combined_ckpt = None

    # Phase 1: reuse hybrid_topo_rbfe.py's loader for exact index alignment
    # with the cached graphs_all.pt.
    df = hto.phase1_load(args, args.output_dir)

    if 2 in phases:
        ablation_ckpt = train_model(df, args, args.output_dir,
                                    use_hyperspherical_losses=False,
                                    run_name="ablation")
    else:
        ck = os.path.join(args.output_dir, "phase_ablation", "best.pt")
        if os.path.exists(ck):
            ablation_ckpt = {"ckpt": ck, "phase_dir": os.path.dirname(ck)}

    if 3 in phases:
        combined_ckpt = train_model(df, args, args.output_dir,
                                    use_hyperspherical_losses=True,
                                    run_name="combined")
    else:
        ck = os.path.join(args.output_dir, "phase_combined", "best.pt")
        if os.path.exists(ck):
            combined_ckpt = {"ckpt": ck, "phase_dir": os.path.dirname(ck)}

    if 4 in phases:
        if combined_ckpt is None:
            raise RuntimeError("Phase 4 requires a trained combined checkpoint (Phase 3).")
        phase4_validate(df, args, combined_ckpt, args.output_dir)

    if 5 in phases:
        if ablation_ckpt is None or combined_ckpt is None:
            raise RuntimeError(
                "Phase 5 requires both the ablation (Phase 2) and combined "
                "(Phase 3) checkpoints."
            )
        phase5_compare(df, args, ablation_ckpt, combined_ckpt, args.output_dir)

    print("\nAll requested phases complete.")


if __name__ == "__main__":
    main()
