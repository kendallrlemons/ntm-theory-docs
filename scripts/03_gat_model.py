"""
03_gat_model.py
===============
Graph Attention Network (GAT) / AttentiveFP-style model for predicting
FEP stderr differences.

Uses multi-head attention over molecular graphs to learn which atoms/bonds
matter most for transformation difficulty.

Usage:
    python 03_gat_model.py \
        --data_dir ../data \
        --output_dir ../results/gat \
        --epochs 100 \
        --batch_size 256 \
        --lr 1e-3 \
        --hidden_dim 128 \
        --num_layers 4 \
        --num_heads 4 \
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
from shared_utils import PairDataset, collate_pair, ATOM_DIM, BOND_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# GAT Model
# =========================================================================

class GATLayer(nn.Module):
    """Multi-head Graph Attention Layer."""

    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W_node = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, num_heads, bias=False)

        # Attention parameters (per head)
        self.attn_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        src, dst = edge_index

        # Project nodes → (N, heads, head_dim)
        h = self.W_node(x).view(N, self.num_heads, self.head_dim)

        # Attention scores
        attn_src_score = (h[src] * self.attn_src).sum(dim=-1)  # (E, heads)
        attn_dst_score = (h[dst] * self.attn_dst).sum(dim=-1)  # (E, heads)
        edge_score = self.W_edge(edge_attr)                     # (E, heads)

        attn = self.leaky_relu(attn_src_score + attn_dst_score + edge_score)

        # Softmax per destination node
        attn_max = torch.zeros(N, self.num_heads, device=x.device)
        attn_max.index_reduce_(0, dst, attn, "amax", include_self=True)
        attn = torch.exp(attn - attn_max[dst])

        attn_sum = torch.zeros(N, self.num_heads, device=x.device)
        attn_sum.index_add_(0, dst, attn)
        attn = attn / (attn_sum[dst] + 1e-8)
        attn = self.dropout(attn)

        # Weighted message aggregation
        msg = h[src] * attn.unsqueeze(-1)  # (E, heads, head_dim)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst, msg)

        # Reshape and residual
        out = out.view(N, -1)  # (N, out_dim)
        out = self.norm(out + self.W_node(x))  # residual + norm

        return out


class GATEncoder(nn.Module):
    """Full GAT encoder: molecular graph → embedding."""

    def __init__(self, atom_dim, bond_dim, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(atom_dim, hidden_dim)
        self.edge_embed = nn.Linear(bond_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATLayer(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)
            )

        self.dropout = nn.Dropout(dropout)

        # Attentive readout (AttentiveFP-style)
        self.readout_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph):
        x = self.node_embed(graph["node_feats"])
        edge_attr = self.edge_embed(graph["edge_feats"])
        edge_index = graph["edge_index"]

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(F.relu(x))

        # Attentive pooling
        batch = graph["batch"]
        attn_weights = self.readout_attn(x).squeeze(-1)  # (N,)

        # Softmax per graph
        attn_max = torch.zeros(batch.max() + 1, device=x.device)
        attn_max.index_reduce_(0, batch, attn_weights, "amax", include_self=True)
        attn_weights = torch.exp(attn_weights - attn_max[batch])
        attn_sum = torch.zeros(batch.max() + 1, device=x.device)
        attn_sum.index_add_(0, batch, attn_weights)
        attn_weights = attn_weights / (attn_sum[batch] + 1e-8)

        # Weighted sum
        weighted = x * attn_weights.unsqueeze(-1)
        num_graphs = batch.max().item() + 1
        pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
        pooled.index_add_(0, batch, weighted)

        return pooled


class PairGAT(nn.Module):
    """Pair-level GAT model."""

    def __init__(self, atom_dim, bond_dim, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = GATEncoder(
            atom_dim, bond_dim, hidden_dim, num_layers, num_heads, dropout
        )
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
        with torch.no_grad():
            return self.encoder(graph_a), self.encoder(graph_b)


# =========================================================================
# Training (same structure as MPNN)
# =========================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
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
    preds, targets = [], []
    for ga, gb, t in loader:
        ga = {k: v.to(device) for k, v in ga.items()}
        gb = {k: v.to(device) for k, v in gb.items()}
        preds.append(model(ga, gb).cpu().numpy())
        targets.append(t.numpy())
    return np.concatenate(preds), np.concatenate(targets)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/gat")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")

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

    model = PairGAT(
        ATOM_DIM, BOND_DIM, args.hidden_dim, args.num_layers,
        args.num_heads, args.dropout,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader_train, optimizer, DEVICE)
        y_vp, y_vt = eval_epoch(model, loader_val, DEVICE)
        val_loss = mean_squared_error(y_vt, y_vp)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                     weights_only=True))

    print("\nFinal Validation:")
    y_vp, y_vt = eval_epoch(model, loader_val, DEVICE)
    val_metrics = evaluate(y_vt, y_vp, "val_")

    print("\nFinal Test:")
    y_tp, y_tt = eval_epoch(model, loader_test, DEVICE)
    test_metrics = evaluate(y_tt, y_tp, "test_")

    results = {**val_metrics, **test_metrics, "args": vars(args), "history": history}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.output_dir, "test_preds.npz"), y_true=y_tt, y_pred=y_tp)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
