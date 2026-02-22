"""
Neural Thermodynamic Metric (NTM) Core Implementation

This module provides the core classes and functions for the NTM model,
including molecular graph conversion, GNN encoding, metric tensor learning,
and geodesic computation.

Usage:
    from ntm_core import NeuralThermodynamicMetric, mol_to_graph, compute_geodesic
    
    model = NeuralThermodynamicMetric(hidden_dim=64)
    graph_a = mol_to_graph("CCO")
    graph_b = mol_to_graph("CCCO")
    
    h_a = model.encode(graph_a)
    h_b = model.encode(graph_b)
    distance = model.compute_distance(h_a, h_b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def mol_to_graph(smiles: str, device: str = 'cpu') -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert a SMILES string to a graph representation for GNN processing.
    
    Args:
        smiles: SMILES string of the molecule
        device: Device to place tensors on ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing:
            - node_features: (n_atoms, 7) atom feature tensor
            - edge_index: (2, n_edges) edge connectivity
            - edge_features: (n_edges, 3) bond feature tensor
            - batch: (n_atoms,) batch assignment (all zeros for single molecule)
        
        Returns None if SMILES is invalid or RDKit is not available.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES conversion. Install with: pip install rdkit")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node (atom) features
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),           # Atomic number
            atom.GetDegree(),              # Number of bonds
            atom.GetFormalCharge(),        # Formal charge
            int(atom.GetHybridization()),  # Hybridization state
            int(atom.GetIsAromatic()),     # Is aromatic
            atom.GetTotalNumHs(),          # Total hydrogen count
            int(atom.IsInRing())           # Is in a ring
        ]
        node_features.append(features)
    
    # Edge (bond) features and connectivity
    edge_index = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Add both directions (undirected graph)
        edge_index.extend([[i, j], [j, i]])
        
        bond_feat = [
            float(bond.GetBondTypeAsDouble()),  # Bond order (1, 1.5, 2, 3)
            int(bond.GetIsConjugated()),        # Is conjugated
            int(bond.IsInRing())                # Is in ring
        ]
        edge_features.extend([bond_feat, bond_feat])
    
    # Handle molecules with no bonds
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_features = [[0.0, 0, 0]]
    
    return {
        'node_features': torch.tensor(node_features, dtype=torch.float32, device=device),
        'edge_index': torch.tensor(edge_index, dtype=torch.long, device=device).T,
        'edge_features': torch.tensor(edge_features, dtype=torch.float32, device=device),
        'batch': torch.zeros(len(node_features), dtype=torch.long, device=device)
    }


def to_device(graph: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move all tensors in a graph dictionary to the specified device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in graph.items()}


class GraphConvLayer(nn.Module):
    """
    Graph convolution layer with edge features and attention.
    
    Implements message passing with:
    - Edge-conditioned messages
    - Attention-weighted aggregation
    - Residual connections with LayerNorm
    """
    
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        
        # Message network: [h_dst || h_src - h_dst || edge_feat] -> message
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Attention for message weighting
        self.attention = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Update with residual
        self.update_net = nn.Linear(in_dim + out_dim, out_dim) if in_dim != out_dim else None
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the graph convolution layer.
        
        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Edge connectivity (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
        
        Returns:
            Updated node features (n_nodes, out_dim)
        """
        src, dst = edge_index
        
        # Compute messages
        h_src, h_dst = x[src], x[dst]
        message_input = torch.cat([h_dst, h_src - h_dst, edge_attr], dim=-1)
        messages = self.message_net(message_input)
        
        # Attention weights (simplified softmax over all edges)
        attn_scores = self.attention(messages)
        attn_weights = F.softmax(attn_scores, dim=0)
        
        # Aggregate messages
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages * attn_weights)
        
        # Update with residual
        if self.update_net is not None:
            out = self.update_net(torch.cat([x, aggregated], dim=-1))
        else:
            out = x + aggregated
        
        return self.norm(out)


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular embedding.
    
    Architecture:
        1. Initial embedding of node and edge features
        2. Multiple message passing layers
        3. Global mean pooling
        4. Output projection
    """
    
    def __init__(self, node_dim: int = 7, edge_dim: int = 3, 
                 hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        
        # Initial embeddings
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a molecular graph to a fixed-size embedding.
        
        Args:
            node_features: (n_nodes, node_dim) atom features
            edge_index: (2, n_edges) bond connectivity
            edge_features: (n_edges, edge_dim) bond features
            batch: (n_nodes,) batch assignment for each atom
        
        Returns:
            Graph-level embedding (batch_size, hidden_dim)
        """
        # Initial embeddings
        x = self.node_embed(node_features)
        edge_attr = self.edge_embed(edge_features)
        
        # Message passing
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Global mean pooling per graph in batch
        unique_batches = batch.unique()
        pooled = torch.stack([x[batch == b].mean(dim=0) for b in unique_batches])
        
        # Output projection
        return self.output_norm(self.output_proj(pooled))


class MetricTensor(nn.Module):
    """
    Learnable Riemannian metric tensor.
    
    Supports two parameterizations:
        - 'diagonal': M = diag(exp(m_1), ..., exp(m_d)) - efficient, fewer parameters
        - 'full': M = L @ L^T where L is lower triangular - captures correlations
    """
    
    def __init__(self, dim: int, metric_type: str = 'diagonal'):
        super().__init__()
        self.dim = dim
        self.metric_type = metric_type
        
        if metric_type == 'diagonal':
            self.log_weights = nn.Parameter(torch.zeros(dim))
        else:
            self.L_diag = nn.Parameter(torch.ones(dim))
            n_lower = dim * (dim - 1) // 2
            self.L_lower = nn.Parameter(torch.zeros(n_lower))
    
    def get_weights(self) -> torch.Tensor:
        """Return diagonal metric weights."""
        if self.metric_type == 'diagonal':
            return torch.exp(self.log_weights)
        else:
            return torch.diag(self.get_full_metric())
    
    def get_full_metric(self) -> torch.Tensor:
        """Return full metric tensor M."""
        if self.metric_type == 'diagonal':
            return torch.diag(torch.exp(self.log_weights))
        else:
            L = torch.zeros(self.dim, self.dim, device=self.L_diag.device)
            L.diagonal().copy_(F.softplus(self.L_diag) + 0.1)
            idx = torch.tril_indices(self.dim, self.dim, offset=-1)
            L[idx[0], idx[1]] = self.L_lower
            return L @ L.T
    
    def compute_distance(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """
        Compute Riemannian distance between embeddings.
        
        d_M(a, b) = sqrt((h_b - h_a)^T M (h_b - h_a))
        """
        diff = h_b - h_a
        
        if self.metric_type == 'diagonal':
            weights = torch.exp(self.log_weights)
            d_sq = torch.sum(diff ** 2 * weights, dim=-1)
        else:
            M = self.get_full_metric()
            d_sq = torch.sum(diff * (diff @ M), dim=-1)
        
        return torch.sqrt(d_sq + 1e-8)


class NeuralThermodynamicMetric(nn.Module):
    """
    Neural Thermodynamic Metric for predicting RBFE difficulty.
    
    The model learns a Riemannian metric on molecular embedding space where
    geodesic distance correlates with transformation difficulty.
    
    Architecture:
        1. GNN Encoder: Molecular graph -> embedding h ∈ R^d
        2. Metric Tensor: Learned positive-definite matrix M
        3. Residual Path: Direct embedding features for asymmetric effects
        4. Prediction Head: Final difficulty prediction
    
    Args:
        node_dim: Dimension of node features (default: 7)
        edge_dim: Dimension of edge features (default: 3)
        hidden_dim: Embedding dimension (default: 64)
        num_layers: Number of GNN layers (default: 3)
        metric_type: 'diagonal' or 'full' (default: 'diagonal')
    """
    
    def __init__(self, node_dim: int = 7, edge_dim: int = 3, 
                 hidden_dim: int = 64, num_layers: int = 3,
                 metric_type: str = 'diagonal'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.metric_type = metric_type
        
        # GNN Encoder
        self.gnn = MolecularGNN(node_dim, edge_dim, hidden_dim, num_layers)
        
        # Metric Tensor
        self.metric = MetricTensor(hidden_dim, metric_type)
        
        # Residual path: [riemannian_dist | h_diff | h_sum]
        residual_dim = 1 + 2 * hidden_dim
        self.residual_net = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def encode(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a molecular graph to embedding."""
        return self.gnn(
            graph['node_features'],
            graph['edge_index'],
            graph['edge_features'],
            graph['batch']
        )
    
    def get_metric_weights(self) -> torch.Tensor:
        """Return diagonal metric weights."""
        return self.metric.get_weights()
    
    def get_full_metric(self) -> torch.Tensor:
        """Return full metric tensor."""
        return self.metric.get_full_metric()
    
    def compute_distance(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """Compute Riemannian distance between embeddings."""
        return self.metric.compute_distance(h_a, h_b)
    
    def forward(self, graph_a: Dict[str, torch.Tensor], 
                graph_b: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict transformation difficulty for a molecular pair.
        
        Args:
            graph_a: Graph dictionary for molecule A
            graph_b: Graph dictionary for molecule B
        
        Returns:
            Predicted difficulty score (batch_size,)
        """
        # Encode both molecules
        h_a = self.encode(graph_a)
        h_b = self.encode(graph_b)
        
        # Riemannian distance
        riemannian_dist = self.compute_distance(h_a, h_b).unsqueeze(-1)
        
        # Residual features
        h_diff = h_b - h_a
        h_sum = h_a + h_b
        
        # Combine and predict
        residual_input = torch.cat([riemannian_dist, h_diff, h_sum], dim=-1)
        residual_features = self.residual_net(residual_input)
        
        return self.pred_head(residual_features).squeeze(-1)


def compute_geodesic(model: NeuralThermodynamicMetric, 
                     h_start: torch.Tensor, h_end: torch.Tensor,
                     n_points: int = 50, n_iterations: int = 200,
                     lr: float = 0.01) -> Tuple[np.ndarray, float, float]:
    """
    Compute the geodesic path between two molecular embeddings.
    
    The geodesic minimizes path length under the learned Riemannian metric.
    
    Args:
        model: NTM model (used for metric tensor)
        h_start: Starting embedding (1, d)
        h_end: Ending embedding (1, d)
        n_points: Number of points along the path
        n_iterations: Optimization iterations
        lr: Learning rate for waypoint optimization
    
    Returns:
        path: (n_points, d) numpy array of waypoints
        geodesic_length: Total path length under the metric
        euclidean_length: Straight-line Euclidean distance
    """
    device = h_start.device
    
    # Initialize path as straight line
    t = torch.linspace(0, 1, n_points, device=device).unsqueeze(1)
    h_start_flat = h_start.squeeze(0)
    h_end_flat = h_end.squeeze(0)
    
    initial_path = h_start_flat * (1 - t) + h_end_flat * t
    
    # Optimize intermediate waypoints
    waypoints = nn.Parameter(initial_path[1:-1].clone())
    optimizer = torch.optim.Adam([waypoints], lr=lr)
    
    M = model.get_full_metric().detach()
    
    for _ in range(n_iterations):
        optimizer.zero_grad()
        
        full_path = torch.cat([
            h_start_flat.unsqueeze(0),
            waypoints,
            h_end_flat.unsqueeze(0)
        ], dim=0)
        
        segments = full_path[1:] - full_path[:-1]
        segment_lengths = torch.sqrt(
            torch.sum(segments * (segments @ M), dim=1) + 1e-8
        )
        
        total_length = segment_lengths.sum()
        total_length.backward()
        optimizer.step()
    
    # Final path
    with torch.no_grad():
        final_path = torch.cat([
            h_start_flat.unsqueeze(0),
            waypoints,
            h_end_flat.unsqueeze(0)
        ], dim=0)
        
        segments = final_path[1:] - final_path[:-1]
        segment_lengths = torch.sqrt(
            torch.sum(segments * (segments @ M), dim=1) + 1e-8
        )
        geodesic_length = segment_lengths.sum().item()
        euclidean_length = torch.norm(h_end_flat - h_start_flat).item()
    
    return final_path.cpu().numpy(), geodesic_length, euclidean_length


def compute_energy_landscape(model: NeuralThermodynamicMetric,
                             h_start: torch.Tensor, h_end: torch.Tensor,
                             n_grid: int = 30) -> Tuple[np.ndarray, ...]:
    """
    Compute energy landscape between two molecular embeddings.
    
    Creates a 2D projection of the embedding space with energy values
    representing transformation difficulty.
    
    Args:
        model: NTM model
        h_start: Starting embedding (1, d)
        h_end: Ending embedding (1, d)
        n_grid: Grid resolution
    
    Returns:
        xx, yy: Grid coordinates
        energy: Energy values at each grid point
        v1, v2: Basis vectors for the 2D projection
    """
    device = h_start.device
    
    # Orthonormal basis
    v1 = (h_end - h_start).squeeze()
    v1 = v1 / (torch.norm(v1) + 1e-8)
    
    torch.manual_seed(42)
    random_vec = torch.randn_like(v1)
    v2 = random_vec - torch.dot(random_vec, v1) * v1
    v2 = v2 / (torch.norm(v2) + 1e-8)
    
    # Grid
    center = (h_start + h_end).squeeze() / 2
    span = torch.norm(h_end - h_start).item() * 2.0
    
    x = torch.linspace(-span/2, span/2, n_grid, device=device)
    y = torch.linspace(-span/2, span/2, n_grid, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Project metric to 2D
    M = model.get_full_metric().detach()
    basis = torch.stack([v1, v2], dim=1)
    M_2d = basis.T @ M @ basis
    
    eigvals, _ = torch.linalg.eigh(M_2d)
    eigvals = eigvals.clamp(min=0.1)
    
    # Endpoint positions in 2D
    h_start_2d = torch.tensor([
        torch.dot((h_start.squeeze() - center), v1).item(),
        torch.dot((h_start.squeeze() - center), v2).item()
    ], device=device)
    h_end_2d = torch.tensor([
        torch.dot((h_end.squeeze() - center), v1).item(),
        torch.dot((h_end.squeeze() - center), v2).item()
    ], device=device)
    
    # Compute energy
    energy = torch.zeros(n_grid, n_grid, device=device)
    sigma = span / 6
    
    for i in range(n_grid):
        for j in range(n_grid):
            pt = torch.tensor([xx[i, j].item(), yy[i, j].item()], device=device)
            
            # Wells at endpoints
            d_start = torch.norm(pt - h_start_2d)
            d_end = torch.norm(pt - h_end_2d)
            well_start = -1.5 * torch.exp(-d_start**2 / (2 * sigma**2))
            well_end = -1.5 * torch.exp(-d_end**2 / (2 * sigma**2))
            
            # Barrier at midpoint
            midpoint = (h_start_2d + h_end_2d) / 2
            pt_centered = pt - midpoint
            metric_dist_sq = pt_centered @ M_2d @ pt_centered
            barrier_height = 0.8 * (eigvals[1] / eigvals[0]).sqrt()
            barrier = barrier_height * torch.exp(-metric_dist_sq / (2 * (sigma * 0.7)**2))
            
            # Curvature and saddle
            metric_curvature = 0.3 * torch.sqrt(pt @ M_2d @ pt + 1e-8)
            saddle = -0.2 * torch.exp(-pt[1]**2 / sigma**2)
            
            energy[i, j] = metric_curvature + well_start + well_end + barrier + saddle
    
    # Normalize
    energy = energy - energy.min()
    energy = energy / (energy.max() + 1e-8)
    
    return (xx.cpu().numpy(), yy.cpu().numpy(), energy.cpu().numpy(),
            v1.cpu().numpy(), v2.cpu().numpy())
