"""
shared_utils.py
===============
Shared molecular graph featurization, dataset, and collation utilities
used by all GNN-based model scripts (MPNN, GAT, NTM).

This avoids code duplication and ensures consistent featurization.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# =========================================================================
# Atom / Bond featurization
# =========================================================================

ATOM_FEATURES = {
    "atomic_num": list(range(1, 120)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": [0, 1, 2, 3, 4],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def one_hot(val, vocab):
    vec = [0] * (len(vocab) + 1)
    if val in vocab:
        vec[vocab.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom) -> list:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )


def bond_features(bond) -> list:
    return (
        one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"])
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    )


ATOM_DIM = (
    len(ATOM_FEATURES["atomic_num"]) + 1
    + len(ATOM_FEATURES["degree"]) + 1
    + len(ATOM_FEATURES["formal_charge"]) + 1
    + len(ATOM_FEATURES["hybridization"]) + 1
    + len(ATOM_FEATURES["num_hs"]) + 1
    + 2  # aromatic, in_ring
)

BOND_DIM = len(BOND_FEATURES["bond_type"]) + 1 + 2  # conjugated, in_ring


# =========================================================================
# SMILES → Graph
# =========================================================================

def smiles_to_graph(smi: str):
    """Convert SMILES to a graph dict with node/edge features."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    if len(node_feats) == 0:
        return None

    edge_index = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_feats.extend([bf, bf])

    # Self-loops for isolated atoms
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_feats = [[0] * BOND_DIM]

    return {
        "node_feats": torch.tensor(node_feats, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long).T,
        "edge_feats": torch.tensor(edge_feats, dtype=torch.float32),
        "num_nodes": len(node_feats),
    }


# =========================================================================
# Batching
# =========================================================================

def batch_graphs(graphs):
    """Batch a list of graph dicts into a single batched graph."""
    node_feats = []
    edge_index = []
    edge_feats = []
    batch_vec = []
    offset = 0

    for i, g in enumerate(graphs):
        n = g["num_nodes"]
        node_feats.append(g["node_feats"])
        edge_index.append(g["edge_index"] + offset)
        edge_feats.append(g["edge_feats"])
        batch_vec.append(torch.full((n,), i, dtype=torch.long))
        offset += n

    return {
        "node_feats": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_feats": torch.cat(edge_feats, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
    }


def collate_pair(batch):
    """Collate pairs of graphs into batched tensors."""
    graphs_a, graphs_b, targets = zip(*batch)
    return (
        batch_graphs(graphs_a),
        batch_graphs(graphs_b),
        torch.stack(targets),
    )


# =========================================================================
# Dataset
# =========================================================================

class PairDataset(Dataset):
    """Dataset of molecular pairs with precomputed graphs."""

    def __init__(self, df):
        col_a, col_b, col_t = df.columns
        self.smiles_a = df[col_a].tolist()
        self.smiles_b = df[col_b].tolist()
        self.targets = df[col_t].tolist()

        # Precompute graphs with cache (many Mol A are repeated)
        self._cache = {}
        self.valid_indices = []
        for idx in range(len(self.smiles_a)):
            ga = self._get_graph(self.smiles_a[idx])
            gb = self._get_graph(self.smiles_b[idx])
            if ga is not None and gb is not None:
                self.valid_indices.append(idx)

        print(f"  Valid pairs: {len(self.valid_indices)}/{len(self.smiles_a)}")

    def _get_graph(self, smi):
        if smi not in self._cache:
            self._cache[smi] = smiles_to_graph(smi)
        return self._cache[smi]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        ga = self._get_graph(self.smiles_a[real_idx])
        gb = self._get_graph(self.smiles_b[real_idx])
        target = self.targets[real_idx]
        return ga, gb, torch.tensor(target, dtype=torch.float32)


# =========================================================================
# MPNN Layer (shared by MPNN and NTM encoders)
# =========================================================================

class MPNNLayer(nn.Module):
    """Single message-passing layer."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_fn = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        messages = self.message_fn(msg_input)

        agg = torch.zeros(x.size(0), messages.size(1), device=x.device)
        agg.index_add_(0, dst, messages)

        x_new = self.update_fn(agg, x)
        return x_new
