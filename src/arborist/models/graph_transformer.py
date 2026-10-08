"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph transformer for contextualizing curve embeddings over skeleton topology.

Two structural signals are computed from the line-graph edge_index and
injected before/during attention:

  1. Global PE  — heat diffusion PE from the graph root (see below).
  2. Relative PE — learnable per-head bias indexed by clamped hop distance
                   (0=self, 1..max_rel_dist=hop count, max_rel_dist+1=far).
                   Unreachable pairs (disconnected components) receive -inf.
                   The old bool attention mask is subsumed by this float bias.


Heat Diffusion Global Positional Encoding
-----------------------------------------
Each TreeSample is a rooted subgraph, where the root marks the structurally
significant node (e.g. a branch point or endpoint of interest). Because the
model's decision is inherently relative to that root, curves should be
characterized by their position within the tree with respect to it — not in
isolation. Heat diffusion PE captures this by propagating a unit impulse from
the root and reading out how much heat reaches each curve v at k time scales:

    H_t(root, v) = Σⱼ exp(−λⱼ · t) · φⱼ(root) · φⱼ(v)

where λⱼ and φⱼ are the eigenvalues and eigenvectors of the normalized graph
Laplacian. This gives each curve a k-dimensional encoding of its proximity to
the root across spatial scales:

  - Small t: only immediately adjacent curves are warm — encodes local
    proximity to the root (1–2 hops).
  - Large t: heat has diffused across the tree — encodes broad distance
    and whether a node is in the same connected component as the root.

The k time scales are initialized log-spaced at [1, 2, 4, ..., 2^(k-1)] and
parameterized in log-space (t = exp(log_t)) so they remain positive and are
learned end-to-end, allowing the model to discover which diffusion scales are
most informative for the task.

"""

import math

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
        # initialized at [1, 2, 4, ..., 2^(k-1)], then projected to latent_dim
        self.log_times = nn.Parameter(
            math.log(2) * torch.arange(n_pe_components, dtype=torch.float32)
        )
        self.pe_proj = nn.Linear(n_pe_components, latent_dim)

        # Relative PE: one learned scalar bias per head per distance bucket
        # Buckets: 0=self, 1..max_rel_dist=hop count, max_rel_dist+1=far/unreachable
        self.rel_pe_emb = nn.Embedding(max_rel_dist + 2, n_heads)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    # --- Positional encoding helpers ---
    def _heat_diffusion_pe(self, n, edge_index, root_indices, device, heat_lam=None, heat_basis=None):
        """
        Heat kernel from root at k log-spaced time scales.

        Uses precomputed (heat_lam, heat_basis) when available — eigendecomposition
        runs once in the dataset worker, not per forward pass. Falls back to
        on-the-fly computation when those are None.

        Returns (n, k).
        """
        k = self.n_pe_components
        if n <= 1:
            return torch.zeros(n, k, device=device)

        if heat_lam is not None and heat_basis is not None:
            # Use basis precomputed in the dataset worker
            lam = torch.as_tensor(heat_lam, dtype=torch.float32, device=device)    # (n,)
            basis = torch.as_tensor(heat_basis, dtype=torch.float32, device=device) # (n, n)
        else:
            # Build normalized Laplacian and decompose on the fly
            adj = torch.zeros(n, n, device=device)
            if edge_index.shape[1] > 0:
                adj[edge_index[0], edge_index[1]] = 1.0
            deg = adj.sum(dim=1).clamp(min=1e-8)
            d_inv_sqrt = deg.pow(-0.5)
            L = torch.eye(n, device=device) - d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
            try:
                lam, eigvecs = torch.linalg.eigh(L)
            except Exception:
                return torch.zeros(n, k, device=device)
            # Project eigenvectors onto root to form the pre-multiplied basis
            if root_indices:
                idx = torch.tensor(root_indices, dtype=torch.long, device=device)
                root_proj = eigvecs[idx].mean(dim=0)
            else:
                root_proj = eigvecs[0]
            basis = eigvecs * root_proj[None, :]

        # Contract basis with k learned diffusion time scales → per-curve PE
        times = self.log_times.exp().to(device)                          # (k,)
        exp_weights = torch.exp(-lam[:, None] * times[None, :])          # (n, k)
        return basis @ exp_weights                                        # (n, k)

    def _hop_dist_matrix(self, n, edge_index):
        """BFS all-pairs shortest-path distances; -1 for unreachable pairs."""
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        if edge_index.shape[1] > 0:
            adj = [[] for _ in range(n)]
            for a, b in edge_index.cpu().t().tolist():
                adj[a].append(b)
            for src in range(n):
                queue, seen, d = [src], {src}, 0
                while queue:
                    nxt = []
                    for u in queue:
                        dist[src][u] = d
                        for v in adj[u]:
                            if v not in seen:
                                seen.add(v)
                                nxt.append(v)
                    queue, d = nxt, d + 1
        return dist

    def _rel_pe_bias(self, dist_list, n, device):
        """Float attention bias (n_heads, n, n) from hop distances."""
        dist_t = torch.tensor(dist_list, dtype=torch.long, device=device)
        bucket = dist_t.clamp(min=0, max=self.max_rel_dist + 1)
        bucket[dist_t < 0] = self.max_rel_dist + 1        # unreachable
        bias = self.rel_pe_emb(bucket).permute(2, 0, 1)  # (n_heads, n, n)
        bias = bias.masked_fill(dist_t.unsqueeze(0) < 0, float('-inf'))
        return bias

    # --- Forward ---
    def forward(self, x, edge_index, root_indices=None, heat_lam=None, heat_basis=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim).
        edge_index : torch.Tensor
            Shape (2, E), dtype long — bidirectional line-graph adjacency.
        root_indices : list[int] or None
            Curve indices adjacent to the proposal endpoint. Used as the heat
            diffusion source. Falls back to node 0 if None.
        heat_lam : numpy.ndarray or None
            Precomputed Laplacian eigenvalues (n,). Passed from TreeSample.
        heat_basis : numpy.ndarray or None
            Precomputed projected eigenbasis (n, n). Passed from TreeSample.

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim).
        """
        n, device = x.shape[0], x.device

        # Add heat diffusion PE from the root before any attention
        x = x + self.pe_proj(
            self._heat_diffusion_pe(n, edge_index, root_indices, device, heat_lam, heat_basis)
        )

        # Build per-head float attention bias from hop distances
        dist_list = self._hop_dist_matrix(n, edge_index)
        attn_bias = self._rel_pe_bias(dist_list, n, device)  # (n_heads, n, n)

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
