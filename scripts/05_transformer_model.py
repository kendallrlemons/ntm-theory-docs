"""
05_transformer_model.py
=======================
Molecular Transformer for predicting FEP stderr differences.

Operates directly on SMILES strings using tokenization + self-attention.
Cross-attention between molecule A and B captures pairwise interactions.

Usage:
    python 05_transformer_model.py \
        --data_dir ../data \
        --output_dir ../results/transformer \
        --epochs 100 \
        --batch_size 128 \
        --lr 5e-4 \
        --d_model 256 \
        --nhead 8 \
        --num_layers 4 \
        --seed 42
"""

import argparse
import os
import json
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# SMILES Tokenizer
# =========================================================================

SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

class SMILESTokenizer:
    """Character-level SMILES tokenizer with learned vocabulary."""

    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

    def __init__(self, max_len=128):
        self.max_len = max_len
        self.token2id = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.fitted = False

    def fit(self, smiles_list):
        """Build vocabulary from SMILES list."""
        vocab = set()
        for smi in smiles_list:
            tokens = SMILES_REGEX.findall(smi)
            vocab.update(tokens)

        for token in sorted(vocab):
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

        self.fitted = True
        print(f"  Vocabulary size: {len(self.token2id)}")
        return self

    def encode(self, smi: str) -> list:
        """Encode SMILES to token IDs."""
        tokens = SMILES_REGEX.findall(smi)
        ids = [self.token2id.get("[CLS]")]
        for t in tokens[: self.max_len - 2]:
            ids.append(self.token2id.get(t, self.token2id["[UNK]"]))
        ids.append(self.token2id.get("[SEP]"))
        return ids

    def pad(self, ids: list) -> tuple:
        """Pad to max_len, return (ids, attention_mask)."""
        pad_id = self.token2id["[PAD]"]
        n = len(ids)
        padded = ids + [pad_id] * (self.max_len - n)
        mask = [1] * n + [0] * (self.max_len - n)
        return padded[: self.max_len], mask[: self.max_len]

    @property
    def vocab_size(self):
        return len(self.token2id)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.token2id, f)

    def load(self, path):
        with open(path) as f:
            self.token2id = json.load(f)
        self.id2token = {int(i): t for t, i in self.token2id.items()}
        self.fitted = True
        return self


# =========================================================================
# Dataset
# =========================================================================

class SMILESPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SMILESTokenizer):
        col_a, col_b, col_t = df.columns
        self.smiles_a = df[col_a].tolist()
        self.smiles_b = df[col_b].tolist()
        self.targets = df[col_t].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        ids_a, mask_a = self.tokenizer.pad(self.tokenizer.encode(self.smiles_a[idx]))
        ids_b, mask_b = self.tokenizer.pad(self.tokenizer.encode(self.smiles_b[idx]))
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(mask_a, dtype=torch.bool),
            torch.tensor(ids_b, dtype=torch.long),
            torch.tensor(mask_b, dtype=torch.bool),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# =========================================================================
# Model
# =========================================================================

class MolecularTransformer(nn.Module):
    """
    Transformer encoder for SMILES strings with cross-attention
    between molecule pairs.
    """

    def __init__(
        self, vocab_size, d_model=256, nhead=8, num_layers=4,
        dim_feedforward=512, max_len=128, dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Token + positional embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Self-attention encoder (shared for both molecules)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention layer (A attends to B and vice versa)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def encode_smiles(self, token_ids, attn_mask):
        """Encode a SMILES sequence → (batch, d_model) embedding."""
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(token_ids) + self.pos_embed(positions)

        # Transformer expects mask where True = ignore
        src_key_padding_mask = ~attn_mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # CLS token pooling (first token)
        return x, x[:, 0, :]  # full sequence, CLS embedding

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        seq_a, cls_a = self.encode_smiles(ids_a, mask_a)
        seq_b, cls_b = self.encode_smiles(ids_b, mask_b)

        # Cross-attention: A attends to B
        cross_a, _ = self.cross_attn(
            seq_a, seq_b, seq_b,
            key_padding_mask=~mask_b,
        )
        cross_a = self.cross_norm(seq_a + cross_a)
        cross_cls_a = cross_a[:, 0, :]

        # Cross-attention: B attends to A
        cross_b, _ = self.cross_attn(
            seq_b, seq_a, seq_a,
            key_padding_mask=~mask_a,
        )
        cross_b = self.cross_norm(seq_b + cross_b)
        cross_cls_b = cross_b[:, 0, :]

        # Pair representation
        pair = torch.cat([
            cross_cls_a, cross_cls_b,
            cross_cls_b - cross_cls_a,
            cross_cls_a * cross_cls_b,
        ], dim=-1)

        return self.pred_head(pair).squeeze(-1)


# =========================================================================
# Training
# =========================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for ids_a, mask_a, ids_b, mask_b, targets in loader:
        ids_a, mask_a = ids_a.to(device), mask_a.to(device)
        ids_b, mask_b = ids_b.to(device), mask_b.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(ids_a, mask_a, ids_b, mask_b)
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
    for ids_a, mask_a, ids_b, mask_b, t in loader:
        ids_a, mask_a = ids_a.to(device), mask_a.to(device)
        ids_b, mask_b = ids_b.to(device), mask_b.to(device)
        preds.append(model(ids_a, mask_a, ids_b, mask_b).cpu().numpy())
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


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
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

    # Build tokenizer
    print("Building tokenizer...")
    col_a, col_b = df_train.columns[0], df_train.columns[1]
    all_smiles = pd.concat([df_train[col_a], df_train[col_b]]).unique().tolist()
    tokenizer = SMILESTokenizer(max_len=args.max_len).fit(all_smiles)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    # Datasets
    print("Building datasets...")
    ds_train = SMILESPairDataset(df_train, tokenizer)
    ds_val = SMILESPairDataset(df_val, tokenizer)
    ds_test = SMILESPairDataset(df_test, tokenizer)

    kw = dict(num_workers=4, pin_memory=True)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **kw)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **kw)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **kw)

    model = MolecularTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, max_len=args.max_len,
        dropout=args.dropout,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader_train, optimizer, DEVICE)
        y_vp, y_vt = eval_epoch(model, loader_val, DEVICE)
        val_loss = mean_squared_error(y_vt, y_vp)
        scheduler.step()
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
