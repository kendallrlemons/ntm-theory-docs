"""
04_ntm_model.py
===============
Neural Thermodynamic Metric (NTM) model with learned metric tensor.

This is the core model from the NTM theory: a GNN encoder maps molecules to
embeddings, then a learned positive-definite metric tensor M defines the
Riemannian distance that predicts transformation difficulty.

Key features:
  - Learned metric tensor M (full or diagonal)
  - Riemannian distance d_M(A,B) = sqrt((h_B - h_A)^T M (h_B - h_A))
  - Eigendecomposition of M reveals hard/easy transformation directions
  - Embeddings can be extracted for downstream analysis

Usage:
    python 04_ntm_model.py \
        --data_dir ../data \
        --output_dir ../results/ntm \
        --metric_type full \
        --epochs 100 \
        --batch_size 256 \
        --lr 1e-3 \
        --hidden_dim 128 \
        --num_layers 4 \
        --seed 42
"""

import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Re-use dataset/featurization from MPNN script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    PairDataset, collate_pair, ATOM_DIM, BOND_DIM,
    smiles_to_graph, batch_graphs, MPNNLayer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# NTM Encoder (GNN backbone)
# =========================================================================

class NTMEncoder(nn.Module):
    """
    GNN encoder that maps molecular graphs to embeddings on the manifold.
    
    This is φ(x) → h ∈ ℝ^d from the theory.
    """

    def __init__(self, atom_dim, bond_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(atom_dim, hidden_dim)
        self.edge_embed = nn.Linear(bond_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # Project to embedding dimension
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph):
        x = self.node_embed(graph["node_feats"])
        edge_attr = self.edge_embed(graph["edge_feats"])
        edge_index = graph["edge_index"]

        for layer, norm in zip(self.layers, self.norms):
            x_new = layer(x, edge_index, edge_attr)
            x = norm(x + self.dropout(x_new))

        # Global mean pooling
        batch = graph["batch"]
        num_graphs = batch.max().item() + 1
        pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        pooled.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
        pooled = pooled / counts.clamp(min=1)

        return self.embed_proj(pooled)  # (batch, hidden_dim)


# =========================================================================
# Learned Metric Tensor
# =========================================================================

class LearnedMetricTensor(nn.Module):
    """
    Learned positive-definite metric tensor M.
    
    Parameterized via Cholesky decomposition: M = L L^T
    This guarantees M is always positive semi-definite.
    
    Options:
      - 'diagonal': M = diag(exp(w))  — fast, interpretable
      - 'full': M = L L^T             — expressive, captures correlations
    """

    def __init__(self, dim: int, metric_type: str = "full"):
        super().__init__()
        self.dim = dim
        self.metric_type = metric_type

        if metric_type == "diagonal":
            # Log-weights for diagonal metric
            self.log_diag = nn.Parameter(torch.zeros(dim))
        elif metric_type == "full":
            # Cholesky factor L (lower triangular)
            self.L_diag = nn.Parameter(torch.ones(dim))
            n_lower = dim * (dim - 1) // 2
            self.L_lower = nn.Parameter(torch.zeros(n_lower) * 0.01)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

    def get_metric_matrix(self) -> torch.Tensor:
        """Return M as a (dim, dim) matrix."""
        if self.metric_type == "diagonal":
            return torch.diag(torch.exp(self.log_diag))
        else:
            L = torch.zeros(self.dim, self.dim, device=self.L_diag.device)
            # Positive diagonal via softplus
            L.diagonal().copy_(F.softplus(self.L_diag) + 0.01)
            # Lower triangular entries
            idx = torch.tril_indices(self.dim, self.dim, offset=-1)
            L[idx[0], idx[1]] = self.L_lower
            return L @ L.T

    def compute_distance(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """
        Compute NTM Riemannian distance: d_M(A,B) = sqrt(Δh^T M Δh).
        
        Args:
            h_a: (batch, dim) embeddings of molecule A
            h_b: (batch, dim) embeddings of molecule B
        Returns:
            (batch,) distances
        """
        delta = h_b - h_a  # (batch, dim)

        if self.metric_type == "diagonal":
            weights = torch.exp(self.log_diag)
            d_sq = torch.sum(delta ** 2 * weights, dim=-1)
        else:
            M = self.get_metric_matrix()
            # d² = Δh^T M Δh = sum_ij Δh_i M_ij Δh_j
            d_sq = torch.sum(delta * (delta @ M), dim=-1)

        return torch.sqrt(d_sq + 1e-8)

    def eigendecomposition(self):
        """
        Eigendecompose M to identify hard/easy directions.
        
        Returns:
            eigenvalues: (dim,) sorted descending (hardest first)
            eigenvectors: (dim, dim) columns are eigenvectors
        """
        M = self.get_metric_matrix().detach()
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        idx = torch.argsort(eigenvalues, descending=True)
        return eigenvalues[idx], eigenvectors[:, idx]


# =========================================================================
# Full NTM Model
# =========================================================================

class NTMModel(nn.Module):
    """
    Neural Thermodynamic Metric model.
    
    Architecture:
      1. GNN encoder φ maps molecules to embeddings
      2. Learned metric tensor M defines distance
      3. Prediction head maps (distance, embeddings) → target
    
    Loss includes:
      - MSE on predictions
      - Metric regularization (encourage meaningful eigenstructure)
    """

    def __init__(
        self,
        atom_dim, bond_dim, hidden_dim, num_layers,
        metric_type="full", dropout=0.1,
    ):
        super().__init__()
        self.encoder = NTMEncoder(atom_dim, bond_dim, hidden_dim, num_layers, dropout)
        self.metric = LearnedMetricTensor(hidden_dim, metric_type)

        # Prediction head: distance + embedding features → prediction
        self.pred_head = nn.Sequential(
            nn.Linear(1 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph_a, graph_b):
        h_a = self.encoder(graph_a)
        h_b = self.encoder(graph_b)

        # NTM distance
        d_m = self.metric.compute_distance(h_a, h_b).unsqueeze(-1)  # (batch, 1)

        # Combine distance with embedding difference and sum
        features = torch.cat([d_m, h_b - h_a, h_a + h_b], dim=-1)

        return self.pred_head(features).squeeze(-1)

    def get_embeddings(self, graph_a, graph_b):
        """Extract embeddings for downstream analysis."""
        with torch.no_grad():
            h_a = self.encoder(graph_a)
            h_b = self.encoder(graph_b)
        return h_a, h_b

    def get_ntm_distance(self, graph_a, graph_b):
        """Compute NTM distance between pairs."""
        with torch.no_grad():
            h_a = self.encoder(graph_a)
            h_b = self.encoder(graph_b)
            return self.metric.compute_distance(h_a, h_b)

    def metric_regularization_loss(self, lambda_reg=0.01):
        """
        Regularize metric tensor:
          - Encourage eigenvalues to be spread (not all equal)
          - Prevent degenerate metric (eigenvalues too small or too large)
        """
        M = self.metric.get_metric_matrix()
        eigenvalues = torch.linalg.eigvalsh(M)

        # Encourage spread: minimize negative entropy of normalized eigenvalues
        probs = eigenvalues / (eigenvalues.sum() + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()

        # Prevent degenerate eigenvalues
        log_eig = torch.log(eigenvalues + 1e-8)
        spread_penalty = log_eig.var()

        return lambda_reg * (-entropy + 0.1 * spread_penalty)


# =========================================================================
# Training
# =========================================================================

def train_epoch(model, loader, optimizer, device, metric_reg=0.01):
    model.train()
    total_loss, n = 0, 0
    for ga, gb, targets in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(ga, gb)
        loss = F.mse_loss(preds, targets)

        # Add metric regularization
        reg_loss = model.metric_regularization_loss(metric_reg)
        total = loss + reg_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, targets = [], []
    for ga, gb, t in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}
        preds.append(model(ga, gb).cpu().numpy())
        targets.append(t.numpy())
    return np.concatenate(preds), np.concatenate(targets)


@torch.no_grad()
def extract_embeddings_and_distances(model, loader, device):
    """Extract all embeddings and NTM distances for analysis."""
    model.eval()
    all_h_a, all_h_b, all_d_m, all_targets = [], [], [], []

    for ga, gb, t in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}

        h_a = model.encoder(ga)
        h_b = model.encoder(gb)
        d_m = model.metric.compute_distance(h_a, h_b)

        all_h_a.append(h_a.cpu().numpy())
        all_h_b.append(h_b.cpu().numpy())
        all_d_m.append(d_m.cpu().numpy())
        all_targets.append(t.numpy())

    return {
        "h_a": np.concatenate(all_h_a),
        "h_b": np.concatenate(all_h_b),
        "d_m": np.concatenate(all_d_m),
        "targets": np.concatenate(all_targets),
    }


def evaluate(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
          f"Pearson={pr:.4f}  Spearman={sr:.4f}")
    return {f"{prefix}{k}": v for k, v in
            {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pr, "spearman_r": sr}.items()}


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/ntm")
    parser.add_argument("--metric_type", default="full", choices=["diagonal", "full"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--metric_reg", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")
    print(f"Metric type: {args.metric_type}")

    # Load
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    print("Building datasets...")
    ds_train = PairDataset(df_train)
    ds_val = PairDataset(df_val)
    ds_test = PairDataset(df_test)

    kw = dict(collate_fn=collate_pair, num_workers=4, pin_memory=True)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **kw)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **kw)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **kw)

    model = NTMModel(
        ATOM_DIM, BOND_DIM, args.hidden_dim, args.num_layers,
        metric_type=args.metric_type, dropout=args.dropout,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader_train, optimizer, DEVICE, args.metric_reg)
        y_vp, y_vt = eval_epoch(model, loader_val, DEVICE)
        val_loss = mean_squared_error(y_vt, y_vp)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if epoch % 5 == 0 or epoch == 1:
            # Log metric tensor stats
            evals, evecs = model.metric.eigendecomposition()
            anisotropy = (evals[0] / evals[-1]).item()
            print(f"Epoch {epoch:3d}: train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"anisotropy={anisotropy:.1f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best and evaluate
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                     weights_only=True))

    print("\nFinal Validation:")
    y_vp, y_vt = eval_epoch(model, loader_val, DEVICE)
    val_metrics = evaluate(y_vt, y_vp, "val_")

    print("\nFinal Test:")
    y_tp, y_tt = eval_epoch(model, loader_test, DEVICE)
    test_metrics = evaluate(y_tt, y_tp, "test_")

    # Extract embeddings and distances for analysis
    print("\nExtracting embeddings and NTM distances...")
    test_data = extract_embeddings_and_distances(model, loader_test, DEVICE)
    np.savez(os.path.join(args.output_dir, "test_embeddings.npz"), **test_data)

    # Save metric tensor and eigendecomposition
    print("\nMetric tensor analysis:")
    evals, evecs = model.metric.eigendecomposition()
    M = model.metric.get_metric_matrix().detach().cpu().numpy()

    print(f"  Eigenvalue range: [{evals[-1].item():.4f}, {evals[0].item():.4f}]")
    print(f"  Anisotropy ratio: {(evals[0] / evals[-1]).item():.1f}")
    print(f"  Condition number: {(evals[0] / evals[-1]).item():.1f}")
    print(f"  Top-5 eigenvalues: {evals[:5].tolist()}")

    np.savez(
        os.path.join(args.output_dir, "metric_tensor.npz"),
        M=M,
        eigenvalues=evals.cpu().numpy(),
        eigenvectors=evecs.cpu().numpy(),
    )

    # Save results
    results = {
        **val_metrics, **test_metrics,
        "args": vars(args),
        "history": history,
        "metric_anisotropy": (evals[0] / evals[-1]).item(),
        "metric_eigenvalue_range": [evals[-1].item(), evals[0].item()],
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.output_dir, "test_preds.npz"), y_true=y_tt, y_pred=y_tp)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
