"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph transformer for contextualizing curve embeddings over skeleton topology.

Two structural signals are injected before/during attention:

  1. Global PE  — root-distance PE over the topological graph (see below).
  2. Relative PE — learnable per-head bias indexed by clamped line-graph hop
                   distance (0=self, 1..max_rel_dist=hop count,
                   max_rel_dist+1=far). Unreachable pairs (disconnected
                   components) receive -inf. The old bool attention mask is
                   subsumed by this float bias.


Root-Distance Global Positional Encoding
----------------------------------------
Each TreeSample is a rooted subgraph, where the root marks the structurally
significant node (e.g. a branch point or endpoint of interest). Because the
model's decision is inherently relative to that root, curves should be
characterized by their position within the tree with respect to it — not in
isolation. Two measures of distance from the root are read at each curve's
proximal and distal endpoints: hop count (number of curves on the path) and
cable length (log-compressed by the caller; see Arborist.geom_scale). The four
values are projected to latent_dim and added to the curve embeddings before
attention, encoding both a curve's depth and its orientation relative to the
root. Both distances are precomputed per sample in the dataset worker.

"""

import torch
import torch.nn as nn


class GraphTransformer(nn.Module):
    """
    Stack of graph-transformer layers with root-distance global PE and
    hop-distance relative PE.
    """

    def __init__(
        self,
        latent_dim,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.1,
        max_rel_dist=4,
    ):
        super().__init__()
        self.max_rel_dist = max_rel_dist

        # Global PE: hop count and compressed cable length from the root at
        # both endpoints (4 values) projected to latent_dim
        self.pe_proj = nn.Linear(4, latent_dim)

        # Relative PE: one learned scalar bias per head per distance bucket
        # Buckets: 0=self, 1..max_rel_dist=hop count, max_rel_dist+1=far/unreachable
        self.rel_pe_emb = nn.Embedding(max_rel_dist + 2, n_heads)

        # Edge-type PE: per-head bias for parent→child (0), child→parent (1),
        # sibling (2), and no-direct-edge (3)
        self.edge_type_emb = nn.Embedding(4, n_heads)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    # --- Positional encoding helpers ---
    def _simple_pe(self, topo_hops, topo_cable, node_endpoints, device):
        """
        Hop count and compressed cable length from the root, read at each
        node's (proximal, distal) endpoints. Returns (n, 4).
        """
        hops = torch.as_tensor(topo_hops, dtype=torch.float32, device=device)
        cable = torch.as_tensor(topo_cable, dtype=torch.float32, device=device)
        ends = torch.as_tensor(node_endpoints, dtype=torch.long, device=device)
        return torch.cat([hops[ends], cable[ends]], dim=1)

    def _edge_type_bias(self, edge_index, edge_type, n, device):
        """Per-head attention bias from line-graph edge types. (n_heads, n, n)"""
        # Default: no-direct-edge (3) for all pairs; fill direct edges with their type
        type_mat = torch.full((n, n), 3, dtype=torch.long, device=device)
        if edge_index.shape[1] > 0:
            et = torch.as_tensor(edge_type, dtype=torch.long, device=device)
            type_mat[edge_index[0], edge_index[1]] = et
        return self.edge_type_emb(type_mat).permute(2, 0, 1)  # (n_heads, n, n)

    def _rel_pe_bias(self, hop_dist, device):
        """Float attention bias (n_heads, n, n) from a hop distance ndarray or tensor."""
        dist_t = torch.as_tensor(hop_dist, dtype=torch.long, device=device)
        bucket = dist_t.clamp(min=0, max=self.max_rel_dist + 1)
        bucket[dist_t < 0] = self.max_rel_dist + 1
        bias = self.rel_pe_emb(bucket).permute(2, 0, 1)  # (n_heads, n, n)
        bias = bias.masked_fill(dist_t.unsqueeze(0) < 0, float('-inf'))
        return bias

    # --- Forward ---
    def forward(self, x, edge_index, edge_type, hop_dist, topo_hops, topo_cable, node_endpoints):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim).
        edge_index : torch.Tensor
            Shape (2, E), dtype long — bidirectional line-graph adjacency.
        edge_type : array-like
            Shape (E,) line-graph edge direction types, parallel to edge_index.
        hop_dist : array-like
            Shape (n, n) line-graph hop distances, -1 = unreachable.
        topo_hops, topo_cable : array-like
            Shape (n_topo,) hop count and compressed cable length from the
            root to each topological node.
        node_endpoints : numpy.ndarray or torch.Tensor
            Shape (n, 2) — (proximal, distal) topological node index of each
            attention node.

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim).
        """
        n, device = x.shape[0], x.device

        # Add the global PE from the root before any attention
        x = x + self.pe_proj(self._simple_pe(topo_hops, topo_cable, node_endpoints, device))

        # Per-head float attention bias: hop distance + edge direction type
        attn_bias = self._rel_pe_bias(hop_dist, device)        # (n_heads, n, n)
        attn_bias = attn_bias + self._edge_type_bias(edge_index, edge_type, n, device)

        for layer in self.layers:
            x = layer(x, attn_bias)
        return self.norm(x)


class GraphTransformerLayer(nn.Module):
    """
    One layer of graph-transformer attention with pre-norm and FFN.
    Accepts a float attn_bias (n_heads, n, n) added to attention logits.
    """

    def __init__(self, latent_dim, n_heads, d_ff, dropout=0.1):
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

    def forward(self, x, attn_bias=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim).
        attn_bias : torch.Tensor, optional
            Shape (n_heads, n, n) float — added to raw attention logits.
            -inf entries block attention. Default is None (full attention).
        """
        h = self.norm1(x).unsqueeze(0)                 # (1, n, latent_dim)
        h, _ = self.attn(h, h, h, attn_mask=attn_bias)
        x = x + self.dropout(h.squeeze(0))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
