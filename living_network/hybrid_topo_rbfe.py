"""
hybrid_topo_rbfe.py
===================
Hybrid-topology GNN for predicting RBFE absolute standard error (|SE|).

Builds a single merged molecular graph per pair by finding the MCS between
ligand A and ligand B, then encoding core / A-unique / B-unique atoms with
endpoint-membership flags. A single MPNN encodes the full hybrid graph; a
regression head predicts |SE| directly via pure MSE.

Design contrast with hypersphere_rbfe.py
-----------------------------------------
- Input: one merged hybrid-topology graph per pair (vs. two separate graphs)
- Geometry: flat Euclidean embedding (no L2 norm / hypersphere)
- Loss: pure MSE regression (no contrastive / dispersion terms)
- OOD: none (regression-only baseline / complement to the hypersphere model)

Phases
------
1. Data loading, cleaning, and train/val/test split
2. Featurization smoke-test (hybrid graph construction check)
3. GNN training with MSE loss
4. Evaluation vs Tanimoto and LOMAP-like baselines

Usage
-----
    python hybrid_topo_rbfe.py

Edit the CONFIG block below or pass CLI overrides.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
from rdkit import RDLogger

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# Device
# =============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[device] CUDA available — using {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    _reasons = []
    if not torch.backends.cuda.is_built():
        _reasons.append("PyTorch was not built with CUDA support")
    else:
        _reasons.append("PyTorch has CUDA support but no GPU was found")
        try:
            import subprocess, shutil
            if shutil.which("nvidia-smi"):
                _smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if _smi.returncode == 0 and _smi.stdout.strip():
                    _reasons.append(
                        f"nvidia-smi sees: {_smi.stdout.strip()!r} "
                        "(driver/CUDA version mismatch?)"
                    )
                else:
                    _reasons.append("nvidia-smi found but reported no GPUs")
            else:
                _reasons.append("nvidia-smi not found on PATH")
        except Exception as _e:
            _reasons.append(f"could not run nvidia-smi ({_e})")
    print(f"[device] CUDA unavailable — falling back to CPU. Reason(s): "
          + "; ".join(_reasons))

# =============================================================================
# CONFIG — hardcoded defaults (override via CLI if desired)
# =============================================================================

SE_COLUMN        = "Difference in first_pass_free_energy_stderr"
SMILES_A_COLUMN  = "Compound Smiles 1"
SMILES_B_COLUMN  = "Compound Smiles 2"

DEFAULT_INPUT       = "/Users/lemonsk/Downloads/compound_smiles_stderr_differences.csv"
DEFAULT_OUTPUT_DIR  = "./results_hybrid"
DEFAULT_SAMPLE_SIZE = 0          # 0 = keep all rows after cleaning
DEFAULT_SEED        = 42
DEFAULT_PHASES      = "1,2,3,4"

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default=DEFAULT_INPUT)
    p.add_argument("--output_dir",   default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--phases",       default=DEFAULT_PHASES)
    p.add_argument("--sample_size",  type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--val_frac",     type=float, default=0.1)
    p.add_argument("--test_frac",    type=float, default=0.1)
    p.add_argument("--hidden_dim",   type=int, default=128)
    p.add_argument("--num_layers",   type=int, default=4)
    p.add_argument("--embed_dim",    type=int, default=128)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--epochs",       type=int, default=60)
    p.add_argument("--patience",     type=int, default=20,
                   help="Early-stop patience in epochs, measured against val "
                        "Spearman rho. Should exceed the LR scheduler's own "
                        "patience (3) so LR decay gets a chance to help first.")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    p.add_argument("--mcs_timeout",  type=int, default=5,
                   help="MCS search timeout per pair (seconds)")
    p.add_argument("--precompute",       action="store_true",
                   help="Pre-build all hybrid graphs in Phase 2 and cache to disk. "
                        "Eliminates per-epoch MCS cost; recommended for large datasets.")
    p.add_argument("--graph_cache_dir",  default=None,
                   help="Directory containing graphs_all.pt and cleaned.pkl produced "
                        "by mcs_precompute.py.  If omitted, falls back to "
                        "{output_dir}/phase2_hybrid (legacy location).")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Atom / bond vocabularies  (same as hypersphere_rbfe.py)
# =============================================================================

ATOM_VOCAB = {
    "atomic_num":    list(range(1, 120)),
    "degree":        list(range(0, 7)),
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": list(range(0, 5)),
}

BOND_VOCAB = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def _one_hot(v, vocab):
    x = [0] * (len(vocab) + 1)
    if v in vocab:
        x[vocab.index(v)] = 1
    else:
        x[-1] = 1
    return x


def _base_atom_features(atom) -> List[float]:
    """149-dim atom features (same as hypersphere_rbfe.py)."""
    return (
        _one_hot(atom.GetAtomicNum(),    ATOM_VOCAB["atomic_num"])
        + _one_hot(atom.GetDegree(),     ATOM_VOCAB["degree"])
        + _one_hot(atom.GetFormalCharge(), ATOM_VOCAB["formal_charge"])
        + _one_hot(atom.GetHybridization(), ATOM_VOCAB["hybridization"])
        + _one_hot(atom.GetTotalNumHs(), ATOM_VOCAB["num_hs"])
        + [int(atom.GetIsAromatic()),
           int(atom.IsInRing()),
           float(atom.GetNumRadicalElectrons())]
    )


def _bond_features(bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_VOCAB["bond_type"])
        + [int(bond.GetIsConjugated()),
           int(bond.IsInRing()),
           int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    )


# Endpoint flags: [is_core, is_a_unique, is_b_unique]
# ATOM_DIM = 149 base features + 3 endpoint flags = 152
_BASE_DIM = (
    len(ATOM_VOCAB["atomic_num"]) + 1
    + len(ATOM_VOCAB["degree"]) + 1
    + len(ATOM_VOCAB["formal_charge"]) + 1
    + len(ATOM_VOCAB["hybridization"]) + 1
    + len(ATOM_VOCAB["num_hs"]) + 1
    + 3
)
ENDPOINT_DIM = 3  # [core, A-unique, B-unique]
ATOM_DIM = _BASE_DIM + ENDPOINT_DIM
BOND_DIM = len(BOND_VOCAB["bond_type"]) + 1 + 3


# =============================================================================
# Hybrid graph construction
# =============================================================================

def build_hybrid_graph(smi_a: str, smi_b: str,
                       mcs_timeout: int = 5) -> Optional[Dict]:
    """
    Build a single merged hybrid-topology graph for a ligand pair.

    Node features (ATOM_DIM = 152):
        - 149-dim base atom features (same vocab as hypersphere_rbfe.py)
        - 3-dim endpoint flag: [1,0,0]=core  [0,1,0]=A-unique  [0,0,1]=B-unique

    The MCS forms the shared 'alchemical core'; atoms unique to each endpoint
    are attached to it via their original bonds. If MCS search fails or yields
    no atoms, the two molecules are concatenated with no shared core (all
    A atoms flagged A-unique, all B atoms flagged B-unique).

    Returns a dict with keys: node_feats, edge_index, edge_feats,
                               n_core, n_a_unique, n_b_unique
    or None if either SMILES is invalid.
    """
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None

    # ---- Find MCS ----
    try:
        res = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            completeRingsOnly=False,
            timeout=mcs_timeout,
        )
        mcs_mol = Chem.MolFromSmarts(res.smartsString) if res.numAtoms > 0 else None
    except Exception:
        mcs_mol = None

    # ---- Atom index mappings ----
    if mcs_mol is not None and mcs_mol.GetNumAtoms() > 0:
        match_a = mol_a.GetSubstructMatch(mcs_mol)
        match_b = mol_b.GetSubstructMatch(mcs_mol)
    else:
        match_a = ()
        match_b = ()

    core_a_idx = set(match_a)   # atom indices in mol_a that are in MCS
    core_b_idx = set(match_b)   # atom indices in mol_b that are in MCS
    uniq_a_idx = [i for i in range(mol_a.GetNumAtoms()) if i not in core_a_idx]
    uniq_b_idx = [i for i in range(mol_b.GetNumAtoms()) if i not in core_b_idx]

    n_core     = len(match_a)
    n_a_unique = len(uniq_a_idx)
    n_b_unique = len(uniq_b_idx)

    # ---- Build node → merged-index maps ----
    # Core nodes: 0 .. n_core-1       (use mol_a's atom objects)
    # A-unique:   n_core .. n_core+n_a_unique-1
    # B-unique:   n_core+n_a_unique .. total-1

    a_to_merged: Dict[int, int] = {}
    for merged_idx, a_idx in enumerate(match_a):
        a_to_merged[a_idx] = merged_idx                          # core
    for local_idx, a_idx in enumerate(uniq_a_idx):
        a_to_merged[a_idx] = n_core + local_idx                  # A-unique

    b_to_merged: Dict[int, int] = {}
    for merged_idx, b_idx in enumerate(match_b):
        b_to_merged[b_idx] = merged_idx                          # core (same slot)
    for local_idx, b_idx in enumerate(uniq_b_idx):
        b_to_merged[b_idx] = n_core + n_a_unique + local_idx     # B-unique

    total_nodes = n_core + n_a_unique + n_b_unique
    if total_nodes == 0:
        return None

    # ---- Node features ----
    node_feats = []

    # Core nodes (from mol_a's atom objects)
    for a_idx in match_a:
        base = _base_atom_features(mol_a.GetAtomWithIdx(a_idx))
        node_feats.append(base + [1, 0, 0])   # flag: core

    # A-unique nodes
    for a_idx in uniq_a_idx:
        base = _base_atom_features(mol_a.GetAtomWithIdx(a_idx))
        node_feats.append(base + [0, 1, 0])   # flag: A-unique

    # B-unique nodes
    for b_idx in uniq_b_idx:
        base = _base_atom_features(mol_b.GetAtomWithIdx(b_idx))
        node_feats.append(base + [0, 0, 1])   # flag: B-unique

    # ---- Edge list ----
    edge_src, edge_dst, edge_feats = [], [], []

    def _add_bond(src: int, dst: int, bond):
        bf = _bond_features(bond)
        edge_src.extend([src, dst])
        edge_dst.extend([dst, src])
        edge_feats.extend([bf, bf])

    # All bonds in mol_a (covers core-core, core-A-unique, A-unique-A-unique)
    for bond in mol_a.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        _add_bond(a_to_merged[i], a_to_merged[j], bond)

    # Bonds in mol_b that involve at least one B-unique atom
    # (core-core bonds already added from mol_a; skip them)
    for bond in mol_b.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        i_is_b_uniq = i not in core_b_idx
        j_is_b_uniq = j not in core_b_idx
        if i_is_b_uniq or j_is_b_uniq:
            _add_bond(b_to_merged[i], b_to_merged[j], bond)

    if not node_feats:
        return None

    return {
        "node_feats":  torch.tensor(node_feats, dtype=torch.float),
        "edge_index":  torch.tensor([edge_src, edge_dst], dtype=torch.long)
                       if edge_src else torch.zeros((2, 0), dtype=torch.long),
        "edge_feats":  torch.tensor(edge_feats, dtype=torch.float)
                       if edge_feats else torch.zeros((0, BOND_DIM), dtype=torch.float),
        "n_core":      n_core,
        "n_a_unique":  n_a_unique,
        "n_b_unique":  n_b_unique,
    }


# =============================================================================
# Dataset
# =============================================================================

def _collate(batch):
    """Batch a list of (graph_dict, y) tuples with index offsets."""
    node_feats_list, edge_index_list, edge_feats_list, ys = [], [], [], []
    offset = 0
    for g, y in batch:
        n = g["node_feats"].size(0)
        node_feats_list.append(g["node_feats"])
        edge_index_list.append(g["edge_index"] + offset)
        edge_feats_list.append(g["edge_feats"])
        ys.append(y)
        offset += n
    return {
        "node_feats": torch.cat(node_feats_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_feats": torch.cat(edge_feats_list, dim=0),
        "batch_ptr":  torch.tensor(
            [0] + [b["node_feats"].size(0) for b, _ in batch], dtype=torch.long
        ).cumsum(0),
    }, torch.tensor(ys, dtype=torch.float)


class HybridPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mcs_timeout: int = 5):
        self.df = df.reset_index(drop=True)
        self.mcs_timeout = mcs_timeout
        self._cache: Dict[Tuple[str, str], Optional[Dict]] = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi_a, smi_b = row[SMILES_A_COLUMN], row[SMILES_B_COLUMN]
        key = (smi_a, smi_b)
        if key not in self._cache:
            self._cache[key] = build_hybrid_graph(smi_a, smi_b, self.mcs_timeout)
        g = self._cache[key]
        y = float(row["SE_abs"])
        return g, y

    @staticmethod
    def collate_fn(batch):
        valid = [(g, y) for g, y in batch if g is not None]
        if not valid:
            return None, None
        return _collate(valid)


class PrecomputedHybridDataset(Dataset):
    """Wraps a pre-built list of graph dicts (parallel to df rows)."""
    def __init__(self, graphs: list, df: pd.DataFrame):
        self.graphs = graphs
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.graphs[idx], float(self.df.iloc[idx]["SE_abs"])

    @staticmethod
    def collate_fn(batch):
        valid = [(g, y) for g, y in batch if g is not None]
        if not valid:
            return None, None
        return _collate(valid)


def precompute_graphs(df: pd.DataFrame, split: str, args, phase_dir: str) -> str:
    """Build all hybrid graphs for one split once and save to disk."""
    out_path = os.path.join(phase_dir, f"graphs_{split}.pt")
    if os.path.exists(out_path):
        print(f"    [cache hit] {out_path}")
        return out_path
    split_df = df[df.split == split].reset_index(drop=True)
    graphs, t0 = [], time.time()
    for i, row in split_df.iterrows():
        if i % 10_000 == 0:
            print(f"      {split}: {i:,}/{len(split_df):,}  "
                  f"({time.time()-t0:.0f}s elapsed)")
        graphs.append(
            build_hybrid_graph(row[SMILES_A_COLUMN], row[SMILES_B_COLUMN],
                               args.mcs_timeout)
        )
    torch.save(graphs, out_path)
    gb = os.path.getsize(out_path) / 1e9
    print(f"    Saved {len(graphs):,} {split} graphs → {out_path} ({gb:.2f} GB)")
    return out_path


def _load_graphs(split: str, phase2_dir: str,
                 df_index=None) -> Optional[list]:
    """Return a graph list for the given split, or None if no cache exists.

    Checks two locations in order:
    1. graphs_all.pt (produced by mcs_precompute.py) — sliced by df_index.
    2. graphs_{split}.pt (legacy per-split files).
    """
    all_path   = os.path.join(phase2_dir, "graphs_all.pt")
    split_path = os.path.join(phase2_dir, f"graphs_{split}.pt")

    if os.path.exists(all_path):
        print(f"    Loading graphs_all.pt and slicing for {split} ...")
        all_graphs = torch.load(all_path, weights_only=False)
        if df_index is None:
            return all_graphs
        return [all_graphs[i] for i in df_index]

    if os.path.exists(split_path):
        print(f"    Loading precomputed {split} graphs from {split_path} ...")
        return torch.load(split_path, weights_only=False)

    return None


def _build_loader(df, args, shuffle: bool, graphs: Optional[list] = None):
    if graphs is not None:
        ds = PrecomputedHybridDataset(graphs, df)
        num_workers = 0   # list already in memory; no benefit from workers
        collate = PrecomputedHybridDataset.collate_fn
    else:
        ds = HybridPairDataset(df, mcs_timeout=args.mcs_timeout)
        num_workers = args.num_workers
        collate = HybridPairDataset.collate_fn
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


# =============================================================================
# Model
# =============================================================================

class MPNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        if src.numel() == 0:
            return x
        m = self.msg(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        agg = torch.zeros(x.size(0), m.size(-1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(m), m)
        return self.norm(self.gru(agg, x))


class HybridGNN(nn.Module):
    """Single MPNN over the merged hybrid-topology graph → regression."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int,
                 num_layers: int, embed_dim: int, dropout: float):
        super().__init__()
        self.node_embed = nn.Linear(atom_dim, hidden_dim)
        self.edge_embed = nn.Linear(bond_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, batch: Dict) -> torch.Tensor:
        """Return the pooled/projected embedding (pre-regression-head).

        Exposed separately from forward() so other scripts can reuse this
        encoder (e.g. to project the embedding onto a hypersphere) without
        duplicating the message-passing + pooling logic.
        """
        x = self.node_embed(batch["node_feats"])
        e = self.edge_embed(batch["edge_feats"])
        ei = batch["edge_index"]
        for layer in self.layers:
            x = self.dropout(layer(x, ei, e))
        # Mean pool per graph using batch_ptr
        ptr = batch["batch_ptr"]
        graphs = []
        for i in range(len(ptr) - 1):
            graphs.append(x[ptr[i]:ptr[i + 1]].mean(0))
        g = torch.stack(graphs, dim=0)
        return self.proj(g)

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)


# =============================================================================
# Phase 1 — Data loading and split
# =============================================================================

def phase1_load(args, output_dir: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 1 — Data Loading & Split")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, "phase1_hybrid")
    os.makedirs(phase_dir, exist_ok=True)
    pkl = os.path.join(phase_dir, "cleaned_split.pkl")

    if os.path.exists(pkl):
        print(f"    Loading cached split from {pkl}")
        return pd.read_pickle(pkl)

    print(f"\n[1.1] Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"    Raw rows: {len(df):,}")

    df = df[[SMILES_A_COLUMN, SMILES_B_COLUMN, SE_COLUMN]].dropna()
    df = df.drop_duplicates(subset=[SMILES_A_COLUMN, SMILES_B_COLUMN])
    df["SE_abs"] = df[SE_COLUMN].abs()
    df = df[df["SE_abs"] > 0].reset_index(drop=True)
    print(f"    After cleaning: {len(df):,}")

    # Subsample
    sample_size = int(args.sample_size)
    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
        print(f"    Subsampled to {sample_size:,} rows")

    # SE distribution
    q = df["SE_abs"].quantile([0.01, 0.25, 0.50, 0.75, 0.95, 0.99])
    print(f"\n[1.2] |SE| distribution:  "
          f"q50={q[0.50]:.4f}  q75={q[0.75]:.4f}  q99={q[0.99]:.4f}")

    # Grouped split by smi_a
    groups = df[SMILES_A_COLUMN].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    n = len(groups)
    n_val  = max(1, int(n * args.val_frac))
    n_test = max(1, int(n * args.test_frac))
    val_g  = set(groups[n - n_val - n_test : n - n_test])
    test_g = set(groups[n - n_test :])

    df["split"] = "train"
    df.loc[df[SMILES_A_COLUMN].isin(val_g),  "split"] = "val"
    df.loc[df[SMILES_A_COLUMN].isin(test_g), "split"] = "test"
    print(f"\n[1.3] Split: "
          f"train={( df.split=='train').sum():,}  "
          f"val={(df.split=='val').sum():,}  "
          f"test={(df.split=='test').sum():,}")

    df.to_pickle(pkl)
    return df


# =============================================================================
# Phase 2 — Featurization smoke-test
# =============================================================================

def phase2_check(df: pd.DataFrame, args, output_dir: str):
    print("\n" + "=" * 70)
    print("PHASE 2 — Hybrid Graph Construction Check")
    print("=" * 70)

    sample = df.sample(min(200, len(df)), random_state=args.seed)
    ok, failed, core_sizes, uniq_sizes = 0, 0, [], []
    for _, row in sample.iterrows():
        g = build_hybrid_graph(row[SMILES_A_COLUMN], row[SMILES_B_COLUMN],
                               args.mcs_timeout)
        if g is not None:
            ok += 1
            core_sizes.append(g["n_core"])
            uniq_sizes.append(g["n_a_unique"] + g["n_b_unique"])
        else:
            failed += 1

    print(f"    Built {ok}/{ok+failed} hybrid graphs successfully")
    if core_sizes:
        print(f"    MCS core size:  mean={np.mean(core_sizes):.1f}  "
              f"min={min(core_sizes)}  max={max(core_sizes)}")
        print(f"    Unique atoms:   mean={np.mean(uniq_sizes):.1f}  "
              f"min={min(uniq_sizes)}  max={max(uniq_sizes)}")
    print(f"    ATOM_DIM={ATOM_DIM}  BOND_DIM={BOND_DIM}")

    phase_dir = os.path.join(output_dir, "phase2_hybrid")
    os.makedirs(phase_dir, exist_ok=True)

    if core_sizes:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(core_sizes,  bins=20, color="steelblue",  edgecolor="black")
        axes[0].set_xlabel("MCS core atoms"); axes[0].set_ylabel("Count")
        axes[0].set_title("MCS Core Size Distribution")
        axes[1].hist(uniq_sizes, bins=20, color="darkorange", edgecolor="black")
        axes[1].set_xlabel("Total unique atoms (A+B)"); axes[1].set_ylabel("Count")
        axes[1].set_title("Unique-Atom Count Distribution")
        plt.suptitle("Hybrid Graph Topology Statistics", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(phase_dir, "graph_topology_stats.png"), dpi=140)
        plt.close()
        print(f"    Saved topology stats to {phase_dir}/graph_topology_stats.png")

    if args.precompute:
        print("\n[2.2] Precomputing hybrid graphs for all splits (--precompute)")
        for split in ("train", "val", "test"):
            precompute_graphs(df, split, args, phase_dir)


# =============================================================================
# Phase 3 — Training
# =============================================================================

def phase3_train(df: pd.DataFrame, args, output_dir: str) -> str:
    print("\n" + "=" * 70)
    print("PHASE 3 — Hybrid-Topology GNN Training")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, "phase3_hybrid")
    os.makedirs(phase_dir, exist_ok=True)

    train_orig = df[df.split == "train"].index
    val_orig   = df[df.split == "val"].index
    train_df   = df[df.split == "train"].reset_index(drop=True)
    val_df     = df[df.split == "val"].reset_index(drop=True)

    print("\n[3.1] Building data loaders")
    phase2_dir = (
        args.graph_cache_dir
        if args.graph_cache_dir
        else os.path.join(output_dir, "phase2_hybrid")
    )
    train_graphs = _load_graphs("train", phase2_dir, df_index=train_orig)
    val_graphs   = _load_graphs("val",   phase2_dir, df_index=val_orig)
    _, train_loader = _build_loader(train_df, args, shuffle=True,  graphs=train_graphs)
    _, val_loader   = _build_loader(val_df,   args, shuffle=False, graphs=val_graphs)

    model = HybridGNN(
        ATOM_DIM, BOND_DIM, args.hidden_dim, args.num_layers,
        args.embed_dim, args.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=3, min_lr=1e-6,
    )
    best_val_rho = -float("inf")
    best_ckpt = os.path.join(phase_dir, "best.pt")
    patience_ctr = 0
    history = []

    print("\n[3.2] Training")
    n_train_batches = len(train_loader)
    LOG_EVERY_BATCH = max(1, n_train_batches // 20)  # ~20 progress lines/epoch
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, total_n = 0.0, 0

        for bi, (batch, y) in enumerate(train_loader):
            if batch is None:
                continue
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            y = y.to(DEVICE)
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += loss.item() * y.size(0)
            total_n    += y.size(0)

            if (bi + 1) % LOG_EVERY_BATCH == 0 or (bi + 1) == n_train_batches:
                dt_so_far = time.time() - t0
                rate      = (bi + 1) / max(dt_so_far, 1e-6)
                print(f"    ep {epoch+1:3d}  batch {bi+1:,}/{n_train_batches:,}  "
                      f"running_mse={total_loss/max(total_n,1):.4f}  "
                      f"{rate:.1f} batch/s  ({dt_so_far:.0f}s elapsed)")

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch, y in val_loader:
                if batch is None:
                    continue
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                pred = model(batch)
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

        print(f"  ep {epoch+1:3d}  "
              f"train_mse={total_loss/max(total_n,1):.4f}  "
              f"val_mse={val_mse:.4f}  val_ρ={val_rho:.3f}  ({dt:.0f}s){lr_note}")

        history.append({"epoch": epoch+1, "val_mse": val_mse, "val_rho": val_rho})

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_ckpt)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    import json
    with open(os.path.join(phase_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Training curve plot
    epochs_done = [r["epoch"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_done, [r["val_mse"] for r in history], color="steelblue")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val MSE"); ax1.set_title("Validation MSE")
    ax2.plot(epochs_done, [r["val_rho"] for r in history], color="seagreen")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Spearman ρ"); ax2.set_title("Validation Spearman ρ")
    plt.suptitle("Hybrid-Topology GNN Training Curve", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "training_curve.png"), dpi=140)
    plt.close()
    print(f"    Saved training curve to {phase_dir}/training_curve.png")

    return best_ckpt


# =============================================================================
# Phase 4 — Evaluation vs baselines
# =============================================================================

def _tanimoto_score(smi_a: str, smi_b: str) -> float:
    ma = Chem.MolFromSmiles(smi_a)
    mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return np.nan
    fp_a = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return 1.0 - DataStructs.TanimotoSimilarity(fp_a, fp_b)


def _lomap_like_score(smi_a: str, smi_b: str, mcs_timeout: int = 5) -> float:
    """Real LOMAP difficulty score (1 - LOMAP similarity), using the actual
    `lomap` + `gufe` packages (LomapAtomMapper + default_lomap_score) rather
    than a hand-rolled approximation. LOMAP is the baseline we're trying to
    beat, so this needs to reflect the real algorithm's MCS-mapping + rule
    penalties (ring-breaking, hybridization/charge changes, etc.), not just
    MCS-coverage averaged with molecular-weight similarity.

    Requires `pip install lomap gufe` (conda-forge: lomap2, gufe).
    Returns np.nan if either SMILES fails to parse/embed or if no atom
    mapping is found (e.g. completely disconnected scaffolds).
    """
    try:
        import gufe
        import lomap
    except ImportError as e:
        raise ImportError(
            "Real LOMAP scoring requires the `lomap` and `gufe` packages "
            "(conda-forge: lomap2, gufe). Install them, e.g.:\n"
            "    conda install -c conda-forge lomap2 gufe"
        ) from e

    def _prep(smi: str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = Chem.AddHs(m)
        try:
            if AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE) != 0:
                return None
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            return None
        return m

    ma = _prep(smi_a)
    mb = _prep(smi_b)
    if ma is None or mb is None:
        return np.nan

    try:
        comp_a = gufe.SmallMoleculeComponent(rdkit=ma)
        comp_b = gufe.SmallMoleculeComponent(rdkit=mb)
        mapper = lomap.LomapAtomMapper(time=mcs_timeout, threed=False)
        mapping = next(iter(mapper.suggest_mappings(comp_a, comp_b)), None)
        if mapping is None:
            return np.nan
        score = lomap.default_lomap_score(mapping)  # 0 (terrible) .. 1 (great)
    except Exception:
        return np.nan

    return 1.0 - float(score)


def phase4_evaluate(df: pd.DataFrame, args, ckpt_path: str, output_dir: str):
    print("\n" + "=" * 70)
    print("PHASE 4 — Evaluation vs Baselines")
    print("=" * 70)

    import json
    phase_dir = os.path.join(output_dir, "phase4_hybrid")
    os.makedirs(phase_dir, exist_ok=True)

    test_df  = df[df.split == "test"].reset_index(drop=True)
    n_eval   = min(20_000, len(test_df))
    eval_pos = test_df.sample(n=n_eval, random_state=args.seed).index  # positions within test_df
    eval_df  = test_df.loc[eval_pos].reset_index(drop=True)
    print(f"    Evaluating on {n_eval:,} test pairs")

    # ---- Load model ----
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = HybridGNN(
        ATOM_DIM, BOND_DIM, args.hidden_dim, args.num_layers,
        args.embed_dim, args.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- Model predictions ----
    print("\n[4.1] Computing model predictions")
    phase2_dir  = (
        args.graph_cache_dir
        if args.graph_cache_dir
        else os.path.join(output_dir, "phase2_hybrid")
    )
    test_orig   = df[df.split == "test"].index
    eval_orig   = test_orig[eval_pos]  # correctly maps sampled rows -> original df indices
    test_graphs = _load_graphs("test", phase2_dir, df_index=eval_orig)
    _, loader = _build_loader(eval_df, args, shuffle=False, graphs=test_graphs)
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch, y in loader:
            if batch is None:
                continue
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            pred = model(batch)
            all_preds.append(pred.cpu().numpy())
            all_true.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)

    # ---- Baselines ----
    print("\n[4.2] Computing Tanimoto + real-LOMAP baselines")
    tani_scores, lomap_scores = [], []
    for i, row in eval_df.iterrows():
        if i % 2000 == 0:
            print(f"      {i}/{n_eval}")
        tani_scores.append(_tanimoto_score(row[SMILES_A_COLUMN], row[SMILES_B_COLUMN]))
        lomap_scores.append(_lomap_like_score(row[SMILES_A_COLUMN], row[SMILES_B_COLUMN],
                                               args.mcs_timeout))
    tani_scores  = np.array(tani_scores, dtype=float)
    lomap_scores = np.array(lomap_scores, dtype=float)

    # ---- Filter valid rows ----
    valid = (
        np.isfinite(all_preds) & np.isfinite(all_true)
        & np.isfinite(tani_scores) & np.isfinite(lomap_scores)
    )
    Y_true      = all_true[valid]
    pred        = all_preds[valid]
    tani_scores = tani_scores[valid]
    lomap_scores= lomap_scores[valid]

    thresh = np.quantile(Y_true, 0.75)
    y_bin  = (Y_true >= thresh).astype(int)

    def _evaluate(score, name):
        pr = pearsonr(score, Y_true)
        sr = spearmanr(score, Y_true)
        try:
            auroc = roc_auc_score(y_bin, score)
            auprc = average_precision_score(y_bin, score)
            s_thresh = np.quantile(score, 0.75)
            mcc = matthews_corrcoef(y_bin, (score >= s_thresh).astype(int))
        except Exception:
            auroc = auprc = mcc = np.nan
        return {
            "method":            name,
            "pearson_r":         float(pr.statistic),
            "pearson_p":         float(pr.pvalue),
            "spearman_r":        float(sr.correlation),
            "spearman_p":        float(sr.pvalue),
            "auroc_hard_top25":  float(auroc),
            "auprc_hard_top25":  float(auprc),
            "mcc_hard_top25":    float(mcc),
        }

    rows = [
        _evaluate(tani_scores,   "Tanimoto (1-sim)"),
        _evaluate(lomap_scores,  "LOMAP (real, 1-sim)"),
        _evaluate(pred,          "Hybrid-topo GNN"),
    ]
    comp_df = pd.DataFrame(rows).set_index("method")
    print("\n[4.3] Results:")
    print(comp_df.round(4).to_string())
    comp_df.to_csv(os.path.join(phase_dir, "baseline_comparison.csv"))

    # ---- Bar chart ----
    _metrics = [
        ("spearman_r",       "Spearman ρ"),
        ("auroc_hard_top25", "AUROC (top 25% hard)"),
        ("auprc_hard_top25", "AUPRC (top 25% hard)"),
        ("mcc_hard_top25",   "MCC (top 25% hard)"),
    ]
    n_m = len(comp_df)
    _colors = ["gray", "gray", "steelblue"][:n_m]
    fig, axes = plt.subplots(1, len(_metrics), figsize=(20, 5))
    for ax, (metric, title) in zip(axes, _metrics):
        vals = comp_df[metric].values
        ax.bar(range(n_m), vals, color=_colors, edgecolor="black")
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(comp_df.index, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.axhline(y=0, color="black", linewidth=0.5)
    plt.suptitle("Hybrid-Topology GNN vs Baselines", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "baseline_comparison.png"), dpi=140)
    plt.close()
    print(f"    Saved plot to {phase_dir}/baseline_comparison.png")

    with open(os.path.join(phase_dir, "results.json"), "w") as f:
        json.dump(comp_df.to_dict(orient="index"), f, indent=2)

    # Pred vs actual scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(Y_true, pred, alpha=0.3, s=4, color="steelblue", rasterized=True)
    lim = max(Y_true.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x")
    ax.set_xlabel("|SE| actual"); ax.set_ylabel("|SE| predicted")
    ax.set_title(f"Hybrid GNN: predicted vs actual\nSpearman ρ={float(spearmanr(pred, Y_true).correlation):.3f}")
    ax.legend(fontsize=8)

    # Residual histogram
    ax2 = axes[1]
    residuals = pred - Y_true
    ax2.hist(residuals, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Residual (pred − actual)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual Distribution\nmean={residuals.mean():.4f}  std={residuals.std():.4f}")

    plt.suptitle("Hybrid-Topology GNN — Test Set", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "prediction_scatter.png"), dpi=140)
    plt.close()
    print(f"    Saved scatter to {phase_dir}/prediction_scatter.png")


# =============================================================================
# Main
# =============================================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)  # flush on every newline (SLURM-safe)
    args   = parse_args()
    phases = [int(p) for p in args.phases.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Phases: {phases}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    df      = None
    ckpt    = None

    if 1 in phases:
        df = phase1_load(args, args.output_dir)
    else:
        pkl = os.path.join(args.output_dir, "phase1_hybrid", "cleaned_split.pkl")
        if os.path.exists(pkl):
            df = pd.read_pickle(pkl)

    if 2 in phases:
        if df is None:
            raise RuntimeError("Phase 2 requires Phase 1 artifacts.")
        phase2_check(df, args, args.output_dir)

    if 3 in phases:
        if df is None:
            raise RuntimeError("Phase 3 requires Phase 1 artifacts.")
        ckpt = phase3_train(df, args, args.output_dir)
    else:
        ck = os.path.join(args.output_dir, "phase3_hybrid", "best.pt")
        if os.path.exists(ck):
            ckpt = ck

    if 4 in phases:
        if ckpt is None:
            raise RuntimeError("Phase 4 requires a trained checkpoint from Phase 3.")
        if df is None:
            raise RuntimeError("Phase 4 requires Phase 1 artifacts.")
        phase4_evaluate(df, args, ckpt, args.output_dir)

    print("\nAll requested phases complete.")


if __name__ == "__main__":
    main()
