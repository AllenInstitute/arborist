"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph transformer for contextualizing curve embeddings over skeleton topology.

Two structural signals are injected before/during attention:

  1. Global PE  — heat diffusion PE from the root over the topological graph
                  (see below).
  2. Relative PE — learnable per-head bias indexed by clamped line-graph hop
                   distance (0=self, 1..max_rel_dist=hop count,
                   max_rel_dist+1=far). Unreachable pairs (disconnected
                   components) receive -inf. The old bool attention mask is
                   subsumed by this float bias.


Heat Diffusion Global Positional Encoding
-----------------------------------------
Each TreeSample is a rooted subgraph, where the root marks the structurally
significant node (e.g. a branch point or endpoint of interest). Because the
model's decision is inherently relative to that root, curves should be
characterized by their position within the tree with respect to it — not in
isolation. Heat diffusion PE captures this by propagating a unit impulse from
the root over the topological graph (nodes = root, branch points, leaves;
edges = curves) and reading out how much heat reaches each topological node v
at k time scales:

    H_t(root, v) = Σⱼ exp(−λⱼ · t) · φⱼ(root) · φⱼ(v)

where λⱼ and φⱼ are the eigenvalues and eigenvectors of the normalized graph
Laplacian. Heat is normalized so the root reads 1 at every scale, making the
encoding comparable across trees of different size. Each curve then reads the
heat at its proximal and distal endpoints, giving a 2k-dimensional encoding of
both its depth and its orientation relative to the root:

  - Small t: only nodes adjacent to the root are warm — encodes local
    proximity to the root (1–2 hops).
  - Large t: heat has diffused across the tree — encodes broad distance
    and whether a node is in the same connected component as the root.

The k time scales are initialized log-spaced at [1, 2, 4, ..., 2^(k-1)] and
parameterized in log-space (t = exp(log_t)) so they remain positive and are
learned end-to-end, allowing the model to discover which diffusion scales are
most informative for the task. The eigendecomposition is done once per sample
in the dataset worker; the model only contracts the basis with the learned
time scales.

"""

import math

import numpy as np
import torch
import torch.nn as nn


class GraphTransformer(nn.Module):
    """
    Stack of graph-transformer layers with Laplacian global PE and
    hop-distance relative PE.
    """

    def __init__(
        self,
        latent_dim,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.1,
        n_pe_components=4,
        max_rel_dist=4,
    ):
        super().__init__()
        self.n_pe_components = n_pe_components
        self.max_rel_dist = max_rel_dist
        self.n_heads = n_heads

        # Global PE: k learnable diffusion times (log-space for positivity),
        # initialized at [1, 2, 4, ..., 2^(k-1)]; heat at a curve's two
        # endpoints (2k values) is projected to latent_dim
        self.log_times = nn.Parameter(
            math.log(2) * torch.arange(n_pe_components, dtype=torch.float32)
        )
        self.pe_proj = nn.Linear(2 * n_pe_components, latent_dim)

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
    def _heat_diffusion_pe(self, heat_lam, heat_basis, node_endpoints, device):
        """
        Heat kernel from the root over the topological graph at k learned
        time scales, read out at each node's (proximal, distal) endpoints.

        Parameters
        ----------
        heat_lam : array-like, shape (n_topo,)
        heat_basis : array-like, shape (n_topo, n_topo)
        node_endpoints : array-like, shape (n, 2)
            Topological node indices per attention node. The CLS node uses
            (root, root).

        Returns (n, 2k).
        """
        lam = torch.as_tensor(heat_lam, dtype=torch.float32, device=device)
        basis = torch.as_tensor(heat_basis, dtype=torch.float32, device=device)
        ends = torch.as_tensor(node_endpoints, dtype=torch.long, device=device)

        times = self.log_times.exp()                                      # (k,)
        heat = basis @ torch.exp(-lam[:, None] * times[None, :])          # (n_topo, k)
        heat = heat / heat[0:1].clamp(min=1e-8)                          # root ≡ 1 per scale
        return heat[ends].reshape(ends.shape[0], -1)                      # (n, 2k)

    def _hop_dist_matrix(self, n, edge_index):
        """BFS fallback — returns int8 ndarray. Only used when hop_dist not precomputed."""
        dist = np.full((n, n), -1, dtype=np.int8)
        np.fill_diagonal(dist, 0)
        if edge_index.shape[1] > 0:
            adj = [[] for _ in range(n)]
            for a, b in edge_index.cpu().t().tolist():
                adj[a].append(b)
            for src in range(n):
                queue, seen, d = [src], {src}, 0
                while queue:
                    nxt = []
                    for u in queue:
                        dist[src, u] = d
                        for v in adj[u]:
                            if v not in seen:
                                seen.add(v)
                                nxt.append(v)
                    queue, d = nxt, d + 1
        return dist

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
    def forward(self, x, edge_index, heat_lam, heat_basis, node_endpoints, edge_type=None, hop_dist=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim).
        edge_index : torch.Tensor
            Shape (2, E), dtype long — bidirectional line-graph adjacency.
        heat_lam : numpy.ndarray
            Topological-graph Laplacian eigenvalues (n_topo,). From TreeSample.
        heat_basis : numpy.ndarray
            Root-projected eigenbasis (n_topo, n_topo). From TreeSample.
        node_endpoints : numpy.ndarray or torch.Tensor
            Shape (n, 2) — (proximal, distal) topological node index of each
            attention node.
        edge_type : array-like or None
            Shape (E,) line-graph edge direction types, parallel to edge_index.
        hop_dist : array-like or None
            Shape (n, n) line-graph hop distances, -1 = unreachable. BFS
            fallback if None.

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim).
        """
        n, device = x.shape[0], x.device

        # Add heat diffusion PE from the root before any attention
        x = x + self.pe_proj(
            self._heat_diffusion_pe(heat_lam, heat_basis, node_endpoints, device)
        )

        # Build per-head float attention bias: hop distance + edge direction type
        if hop_dist is None:
            hop_dist = self._hop_dist_matrix(n, edge_index)
        attn_bias = self._rel_pe_bias(hop_dist, device)        # (n_heads, n, n)
        if edge_type is not None:
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
