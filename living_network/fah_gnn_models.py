"""Adapters for the frozen Fah benchmark; no split or training logic."""
import torch
from torch import nn
import hypersphere_rbfe as sphere
import hybrid_topo_rbfe as hybrid


class FahGNN(nn.Module):
    def __init__(self, architecture, pilot=False, hidden_dim=128, embed_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.architecture = architecture
        self.pilot = pilot
        if architecture == 'hypersphere':
            self.encoder = sphere.HypersphereEncoder(sphere.ATOM_DIM, sphere.BOND_DIM, hidden_dim, num_layers, embed_dim, dropout, l2_normalize=True)
            input_dim = 2 * embed_dim + 1
        elif architecture == 'hybrid':
            self.encoder = hybrid.HybridGNN(hybrid.ATOM_DIM, hybrid.BOND_DIM, hidden_dim, num_layers, embed_dim, dropout)
            self.encoder.head = nn.Identity()
            input_dim = embed_dim
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.head = nn.Sequential(nn.Linear(input_dim + int(pilot), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)) if architecture == 'hypersphere' else nn.Sequential(nn.Linear(input_dim + int(pilot), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, graphs, pilot=None):
        if self.architecture == 'hypersphere':
            ha, hb = self.encoder(graphs[0]), self.encoder(graphs[1])
            cosine = (ha * hb).sum(-1)
            features = torch.cat([(ha - hb).abs(), ha * hb, cosine[:, None]], dim=-1)
        else:
            features = self.encoder.encode(graphs)
            cosine = None
        if self.pilot:
            if pilot is None:
                raise ValueError('Pilot input required')
            features = torch.cat([features, pilot[:, None]], dim=-1)
        elif pilot is not None:
            raise ValueError('Structure-only model cannot accept pilot inputs')
        return self.head(features).squeeze(-1), cosine


def collate_graphs(records, architecture, device):
    if architecture == 'hypersphere':
        a, b = zip(*records)
        batches = (sphere.batch_graphs(a), sphere.batch_graphs(b))
        return tuple({k: v.to(device) for k, v in batch.items()} for batch in batches)
    batch, _ = hybrid._collate([(graph, 0.0) for graph in records])
    return {k: v.to(device) for k, v in batch.items()}
