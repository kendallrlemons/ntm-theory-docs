"""
02_mpnn_model.py
================
Message Passing Neural Network (MPNN) for predicting FEP stderr differences.

Encodes each molecule as a graph, applies message passing layers,
pools to graph-level embeddings, then predicts from the pair.

Usage:
    python 02_mpnn_model.py \
        --data_dir ../data \
        --output_dir ../results/mpnn \
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
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    PairDataset, collate_pair, batch_graphs, smiles_to_graph,
    ATOM_DIM, BOND_DIM, MPNNLayer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# MPNN Model
# =========================================================================

class MPNN(nn.Module):
    """Full MPNN encoder for a single molecule → embedding."""

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

    def forward(self, graph):
        x = self.node_embed(graph["node_feats"])
        edge_attr = self.edge_embed(graph["edge_feats"])
        edge_index = graph["edge_index"]

        for layer, norm in zip(self.layers, self.norms):
            x_new = layer(x, edge_index, edge_attr)
            x = norm(x + self.dropout(x_new))  # residual

        # Global mean pooling
        batch = graph["batch"]
        num_graphs = batch.max().item() + 1
        pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        pooled.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
        pooled = pooled / counts.clamp(min=1)

        return pooled  # (batch, hidden_dim)


class PairMPNN(nn.Module):
    """Pair-level model: encode A and B, predict stderr difference."""

    def __init__(self, atom_dim, bond_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.encoder = MPNN(atom_dim, bond_dim, hidden_dim, num_layers, dropout)

        # Prediction head takes: h_a, h_b, h_b - h_a, h_a * h_b
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph_a, graph_b):
        h_a = self.encoder(graph_a)
        h_b = self.encoder(graph_b)

        pair_repr = torch.cat([h_a, h_b, h_b - h_a, h_a * h_b], dim=-1)
        return self.pred_head(pair_repr).squeeze(-1)

    def get_embeddings(self, graph_a, graph_b):
        """Return embeddings for analysis."""
        with torch.no_grad():
            h_a = self.encoder(graph_a)
            h_b = self.encoder(graph_b)
        return h_a, h_b


# =========================================================================
# Training loop
# =========================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for ga, gb, targets in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(ga, gb)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for ga, gb, targets in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}
        preds = model(ga, gb)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    metrics = {
        f"{prefix}rmse": rmse, f"{prefix}mae": mae, f"{prefix}r2": r2,
        f"{prefix}pearson_r": pr, f"{prefix}spearman_r": sr,
    }
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
          f"Pearson={pr:.4f}  Spearman={sr:.4f}")
    return metrics


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/mpnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    print("Building datasets...")
    ds_train = PairDataset(df_train)
    ds_val = PairDataset(df_val)
    ds_test = PairDataset(df_test)

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pair, num_workers=4, pin_memory=True,
    )
    loader_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pair, num_workers=4, pin_memory=True,
    )
    loader_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pair, num_workers=4, pin_memory=True,
    )

    # Model
    model = PairMPNN(
        atom_dim=ATOM_DIM, bond_dim=BOND_DIM,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-6
    )

    # Training
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader_train, optimizer, DEVICE)
        y_val_pred, y_val_true = eval_epoch(model, loader_val, DEVICE)
        val_loss = mean_squared_error(y_val_true, y_val_pred)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
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

    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                     weights_only=True))

    print("\nFinal Validation:")
    y_val_pred, y_val_true = eval_epoch(model, loader_val, DEVICE)
    val_metrics = evaluate(y_val_true, y_val_pred, "val_")

    print("\nFinal Test:")
    y_test_pred, y_test_true = eval_epoch(model, loader_test, DEVICE)
    test_metrics = evaluate(y_test_true, y_test_pred, "test_")

    # Save
    results = {**val_metrics, **test_metrics, "args": vars(args), "history": history}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(args.output_dir, "test_preds.npz"),
             y_true=y_test_true, y_pred=y_test_pred)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
