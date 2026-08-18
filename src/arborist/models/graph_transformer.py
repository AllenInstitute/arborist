"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph transformer for contextualizing curve embeddings over skeleton topology.

"""

import torch
import torch.nn as nn


class GraphTransformer(nn.Module):
    """
    Stack of graph-masked transformer layers over a set of node features.

    Restricts each node's attention to itself and its 1-hop graph neighbors,
    coupling transformer expressiveness with explicit graph topology.
    """

    def __init__(self, latent_dim, n_heads=4, n_layers=3, d_ff=128, dropout=0.1):
        """
        Parameters
        ----------
        latent_dim : int
            Node feature dimension (must equal CurveEncoder latent_dim when
            used inside ArboristModel).
        n_heads : int, optional
            Number of attention heads. Default is 4.
        n_layers : int, optional
            Number of transformer layers. Default is 3.
        d_ff : int, optional
            Feed-forward hidden dimension. Default is 128.
        dropout : float, optional
            Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def _build_attn_mask(self, n, edge_index, device):
        """
        True at (i, j) blocks attention from node i to node j.
        Every node attends to itself and all immediate neighbors.
        """
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        if edge_index.shape[1] > 0:
            mask[edge_index[0], edge_index[1]] = False
        return mask

    def forward(self, x, edge_index):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim) — one feature vector per graph node (curve).
        edge_index : torch.Tensor
            Shape (2, E), dtype long — bidirectional line-graph adjacency
            as returned by GraphDataset.

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim) — enriched node features.
        """
        attn_mask = self._build_attn_mask(x.shape[0], edge_index, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)


class GraphTransformerLayer(nn.Module):
    """
    One layer of graph-masked multi-head self-attention with pre-norm and FFN.
    """

    def __init__(self, latent_dim, n_heads, d_ff, dropout=0.1):
        """
        Parameters
        ----------
        latent_dim : int
            Node feature dimension.
        n_heads : int
            Number of attention heads.
        d_ff : int
            Feed-forward hidden dimension.
        dropout : float, optional
            Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim) — one feature vector per graph node.
        attn_mask : torch.Tensor, optional
            Shape (n, n) BoolTensor; True at (i, j) blocks attention from
            node i to node j. Default is None (full attention).

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim).
        """
        h = self.norm1(x).unsqueeze(0)                # (1, n, latent_dim)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout(h.squeeze(0))            # (n, latent_dim)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
