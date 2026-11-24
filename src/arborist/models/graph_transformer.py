"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Transformers over curve embeddings, one token per curve plus the root.
GraphTransformer attends over all tokens; SubtreeGraphTransformer attends
within each subtree, then between the subtrees and the root.

"""

import numpy as np
import torch
import torch.nn as nn

from arborist.models.positional_encoding import GlobalPE, RelativePE


class GraphTransformer(nn.Module):
    """
    Stack of transformer layers over curve embeddings with global and
    relative positional encodings.
    """

    def __init__(
        self,
        latent_dim,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.1,
        geom_scale=10.0,
    ):
        super().__init__()
        self.global_pe = GlobalPE(latent_dim, geom_scale=geom_scale)
        self.relative_pe = RelativePE(n_heads, geom_scale=geom_scale)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, endpoints, hops, cable, rel_hops, rel_cable):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim). One embedding per token.
        endpoints, hops, cable : torch.Tensor
            Per-token inputs to GlobalPE; see its forward.
        rel_hops, rel_cable : torch.Tensor
            Pairwise inputs to RelativePE; see its forward.

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim).
        """
        x = x + self.global_pe(endpoints, hops, cable)
        attn_bias = self.relative_pe(endpoints[:, 1], rel_hops, rel_cable)
        for layer in self.layers:
            x = layer(x, attn_bias)
        return self.norm(x)


class GraphTransformerLayer(nn.Module):
    """One pre-norm self-attention + FFN layer over the (n, latent_dim) token set."""

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

    def forward(self, x, attn_bias):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (n, latent_dim), or (B, n, latent_dim) for B independent
            token sets.
        attn_bias : torch.Tensor
            Shape (n_heads, n, n) float, or (B * n_heads, n, n) for batched
            input, added to the attention logits.
        """
        h = self.norm1(x)
        unbatched = h.dim() == 2
        if unbatched:
            h = h.unsqueeze(0)                         # (1, n, latent_dim)
        h, _ = self.attn(h, h, h, attn_mask=attn_bias)
        if unbatched:
            h = h.squeeze(0)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


def partition_curves(parents):
    """
    Partitions the curves of a TreeSample into the subtrees under each
    root-adjacent curve.

    Parameters
    ----------
    parents : numpy.ndarray
        Shape (T,) int. Parent token of each token (see TreeSample.parents);
        parents precede their children.

    Returns
    -------
    group_of : numpy.ndarray
        Shape (T,) int. Subtree index of each token; -1 for the root.
    heads : numpy.ndarray
        Shape (G,) int. Token index of each subtree's root-adjacent curve.
    """
    T = len(parents)
    head = np.arange(T)
    for k in range(1, T):
        if parents[k] > 0:
            head[k] = head[parents[k]]
    heads = np.unique(head[1:])
    group_of = np.full(T, -1, dtype=np.int64)
    group_of[1:] = np.searchsorted(heads, head[1:])
    return group_of, heads


class SubtreeGraphTransformer(GraphTransformer):
    """
    GraphTransformer with a subtree level between the curves and the root.
    Curves are partitioned into connected subtrees (``partition_curves``).
    Stage 1 attends within each subtree under its own class token; stage 2,
    the inherited layer stack, attends between the root and the subtree
    tokens; subtree context is then broadcast back to the curves so the
    per-curve embeddings keep a view of the whole tree. Same inputs and
    outputs as GraphTransformer, plus the grouping.

    Each subtree's class token takes the position of its head curve, so the
    positional encodings describe the subtree's placement and its distances
    to the root and to the other subtrees.

    Parameters
    ----------
    latent_dim : int
        Token dimension.
    n_heads : int, optional
        Attention heads in every layer. Default is 4.
    n_layers : int, optional
        Stage-2 layers, attending between the root and the subtrees.
        Default is 3.
    n_local_layers : int, optional
        Stage-1 layers, attending within each subtree. Default is 1.
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
        n_layers=3,
        n_local_layers=1,
        d_ff=128,
        dropout=0.1,
        geom_scale=10.0,
    ):
        super().__init__(latent_dim, n_heads, n_layers, d_ff, dropout, geom_scale)
        self.n_heads = n_heads
        self.group_cls = nn.Parameter(torch.zeros(latent_dim))
        self.local_layers = nn.ModuleList([
            GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
            for _ in range(n_local_layers)
        ])
        self.broadcast = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

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
            Shape (T,) long. Subtree of each token, -1 for the root.
        heads : torch.Tensor
            Shape (G,) long. Head token of each subtree.

        Returns
        -------
        torch.Tensor
            Shape (T, latent_dim). Root output is the global tree embedding.
        """
        pe = self.global_pe(endpoints, hops, cable)
        x = x + pe
        attn_bias = self.relative_pe(endpoints[:, 1], rel_hops, rel_cable)

        # Stage 1: each subtree is a padded row led by its class token, which
        # takes the position of the subtree's head curve
        members, padding = self._pad_groups(group_of, len(heads))
        tokens = torch.cat([heads[:, None], members], dim=1)   # (G, m+1)
        padding = torch.cat([padding[:, :1] & False, padding], dim=1)
        h = x[tokens]
        h[:, 0] = self.group_cls + pe[heads]
        bias = self._gather_bias(attn_bias, tokens, padding)
        for layer in self.local_layers:
            h = layer(h, bias)
        groups = h[:, 0]                                       # (G, d)

        # Stage 2: the root attends with the subtree class tokens
        sel = torch.cat([heads.new_zeros(1), heads])
        g = torch.cat([x[:1], groups], dim=0)
        g_bias = attn_bias[:, sel[:, None], sel[None, :]]
        for layer in self.layers:
            g = layer(g, g_bias)

        # Broadcast subtree context back down to the curve tokens
        keep = ~padding[:, 1:]
        order = members[keep]
        out = x.new_empty(x.shape)
        out[0] = g[0]
        out[order] = h[:, 1:][keep] + self.broadcast(g[1:][group_of[order]])
        return self.norm(out)

    def _gather_bias(self, attn_bias, tokens, padding):
        """
        Restricts the (n_heads, T, T) attention bias to each padded subtree,
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
        Indexes the tokens of each subtree into a padded (G, m) matrix, where
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
