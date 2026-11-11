"""
Created on Mon Nov 2 10:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Two-stage transformer over curve embeddings for hierarchical global pooling.
Curves are first partitioned into connected subtrees (groups). Stage 1 gives
each group its own class token and runs self-attention within the group;
stage 2 runs self-attention between the root and the group class tokens and
reads the global tree embedding from the root. Group context is broadcast
back to the curves so per-curve embeddings keep a view of the whole tree.

Group class tokens reuse the positional encodings of the flat
GraphTransformer (arborist.models.positional_encoding): each is positioned
like its group's head curve, the member closest to the root, so the global
and relative quantities of that curve describe the group's placement and
its LCA-based distances to the root and to the other groups.

"""

import numpy as np
import torch
import torch.nn as nn

from arborist.models.graph_transformer import GraphTransformerLayer
from arborist.models.positional_encoding import GlobalPE, RelativePE


def partition_curves(parents, lengths, max_cable=None):
    """
    Partitions the curves of a TreeSample into connected subtrees.

    Walks the curves leaf-first and merges each curve into its parent's
    group while the group's total cable stays within max_cable, so groups
    are connected and roughly balanced in cable. Curves adjacent to the
    root always start their own group. With max_cable None the groups are
    the whole subtrees under each root-adjacent curve.

    Parameters
    ----------
    parents : numpy.ndarray
        Shape (T,) int. Parent token of each token (see TreeSample.parents).
    lengths : numpy.ndarray
        Shape (T,) float. Cable length of each token's curve; entry 0 (the
        root) is ignored.
    max_cable : float, optional
        Largest total cable (microns) a group may hold. Default is None.

    Returns
    -------
    group_of : numpy.ndarray
        Shape (T,) int. Group index of each token; -1 for the root.
    heads : numpy.ndarray
        Shape (G,) int. Token index of each group's head curve.
    """
    T = len(parents)
    head = np.arange(T)
    acc = np.asarray(lengths, dtype=np.float64).copy()
    limit = np.inf if max_cable is None else max_cable
    for k in range(T - 1, 0, -1):
        p = parents[k]
        if p > 0 and acc[p] + acc[k] <= limit:
            head[k] = p
            acc[p] += acc[k]
    for k in range(1, T):
        head[k] = head[head[k]]
    heads = np.unique(head[1:])
    group_of = np.full(T, -1, dtype=np.int64)
    group_of[1:] = np.searchsorted(heads, head[1:])
    return group_of, heads


class HierarchicalGraphTransformer(nn.Module):
    """
    Two-stage transformer over curve embeddings: attention within groups of
    curves, each read out by its own class token, attention between the
    root and the group class tokens, and a broadcast of group context back
    to the curves. Same inputs and outputs as GraphTransformer, plus the
    grouping.

    Parameters
    ----------
    latent_dim : int
        Token dimension.
    n_heads : int, optional
        Attention heads in every layer. Default is 4.
    n_local_layers : int, optional
        Stage-1 layers, attending within each group. Default is 1.
    n_global_layers : int, optional
        Stage-2 layers, attending between group tokens. Default is 3.
    d_ff : int, optional
        Feed-forward width. Default is 128.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    geom_scale : float, optional
        Reference length (microns) for the positional encodings. Default is
        10.0.
    """

    def __init__(
        self,
        latent_dim,
        n_heads=4,
        n_local_layers=1,
        n_global_layers=3,
        d_ff=128,
        dropout=0.1,
        geom_scale=10.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.global_pe = GlobalPE(latent_dim, geom_scale=geom_scale)
        self.relative_pe = RelativePE(n_heads, geom_scale=geom_scale)
        self.group_cls = nn.Parameter(torch.zeros(latent_dim))
        self.local_layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_local_layers)
        ])
        self.global_layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_global_layers)
        ])
        self.broadcast = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, x, endpoints, hops, cable, rel_hops, rel_cable, group_of, heads
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (T, latent_dim). Root token first, then one per curve.
        endpoints, hops, cable : torch.Tensor
            Per-token inputs to GlobalPE; see its forward.
        rel_hops, rel_cable : torch.Tensor
            Pairwise inputs to RelativePE; see its forward.
        group_of : torch.Tensor
            Shape (T,) long. Group of each token, -1 for the root.
        heads : torch.Tensor
            Shape (G,) long. Head token of each group.

        Returns
        -------
        torch.Tensor
            Shape (T, latent_dim). Root output is the global tree embedding.
        """
        pe = self.global_pe(endpoints, hops, cable)
        x = x + pe
        attn_bias = self.relative_pe(endpoints[:, 1], rel_hops, rel_cable)

        # Stage 1: each group is a padded row led by its class token, which
        # takes the position of the group's head curve
        members, padding = self._pad_groups(group_of, len(heads))
        tokens = torch.cat([heads[:, None], members], dim=1)   # (G, m+1)
        padding = torch.cat([padding[:, :1] & False, padding], dim=1)
        h = x[tokens]
        h[:, 0] = self.group_cls + pe[heads]
        bias = self._gather_bias(attn_bias, tokens, padding)
        for layer in self.local_layers:
            h = layer(h, bias)
        groups = h[:, 0]                                       # (G, d)

        # Stage 2: the root attends with the group class tokens
        sel = torch.cat([heads.new_zeros(1), heads])
        g = torch.cat([x[:1], groups], dim=0)
        g_bias = attn_bias[:, sel[:, None], sel[None, :]]
        for layer in self.global_layers:
            g = layer(g, g_bias)

        # Broadcast group context back down to the curve tokens
        keep = ~padding[:, 1:]
        order = members[keep]
        out = x.new_empty(x.shape)
        out[0] = g[0]
        out[order] = h[:, 1:][keep] + self.broadcast(g[1:][group_of[order]])
        return self.norm(out)

    def _gather_bias(self, attn_bias, tokens, padding):
        """
        Restricts the (n_heads, T, T) attention bias to each padded group,
        masking padded keys, as one (G * n_heads, m, m) attention mask.
        """
        G, m = tokens.shape
        bias = attn_bias[:, tokens[:, :, None], tokens[:, None, :]]
        bias = bias.permute(1, 0, 2, 3)                        # (G, H, m, m)
        masked = padding[:, None, None, :].expand(-1, self.n_heads, m, -1)
        return bias.masked_fill(masked, float("-inf")).flatten(0, 1)

    @staticmethod
    def _pad_groups(group_of, n_groups):
        """
        Indexes the tokens of each group into a padded (G, m) matrix, where
        padding slots point at token 0 and are flagged in the mask.
        """
        counts = torch.bincount(group_of[1:], minlength=n_groups)
        m = int(counts.max())
        order = torch.argsort(group_of[1:], stable=True) + 1
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(len(order), device=group_of.device)
        slot = slot - starts[group_of[order]]
        members = group_of.new_zeros(n_groups, m)
        padding = torch.ones(
            n_groups, m, dtype=torch.bool, device=group_of.device
        )
        members[group_of[order], slot] = order
        padding[group_of[order], slot] = False
        return members, padding
