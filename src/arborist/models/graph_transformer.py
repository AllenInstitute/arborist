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
        d_ff=128,
        dropout=0.1,
        geom_scale=10.0,
        n_heads=4,
        n_layers=3,
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
    subtree_of : numpy.ndarray
        Shape (T,) int. Subtree index of each token; -1 for the root.
    roots : numpy.ndarray
        Shape (S,) int. Token index of each subtree's root-adjacent curve.
    """
    T = len(parents)
    root = np.arange(T)
    for k in range(1, T):
        if parents[k] > 0:
            root[k] = root[parents[k]]
    roots = np.unique(root[1:])
    subtree_of = np.full(T, -1, dtype=np.int64)
    subtree_of[1:] = np.searchsorted(roots, root[1:])
    return subtree_of, roots


class SubtreeGraphTransformer(GraphTransformer):
    """
    GraphTransformer that attends within each subtree under its own class
    token, then between the subtree tokens and the root, and broadcasts the
    result back to the curves.

    Each subtree's class token takes the position of its head curve, so the
    positional encodings describe the subtree's placement and its distances
    to the root and to the other subtrees.

    Parameters
    ----------
    latent_dim : int
        Token dimension.
    d_ff : int, optional
        Feed-forward width. Default is 128.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    geom_scale : float, optional
        Reference length (microns) for the positional encodings. Default is
        10.0.
    n_heads : int, optional
        Attention heads in every layer. Default is 4.
    n_layers : int, optional
        Stage-2 layers, attending between the root and the subtrees.
        Default is 3.
    """

    def __init__(
        self,
        latent_dim,
        d_ff=128,
        dropout=0.1,
        geom_scale=10.0,
        n_heads=4,
        n_layers=3,
    ):
        super().__init__(latent_dim, d_ff, dropout, geom_scale, n_heads, n_layers)
        self.n_heads = n_heads
        self.subtree_cls = nn.Parameter(torch.zeros(latent_dim))
        self.local_layer = GraphTransformerLayer(latent_dim, n_heads, d_ff, dropout)
        self.broadcast = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self, x, endpoints, hops, cable, rel_hops, rel_cable, subtree_of, roots
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
        subtree_of : torch.Tensor
            Shape (T,) long. Subtree of each token, -1 for the root.
        roots : torch.Tensor
            Shape (S,) long. Root-adjacent curve token of each subtree.

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
        members, padding = self._pad_subtrees(subtree_of, len(roots))
        tokens = torch.cat([roots[:, None], members], dim=1)   # (S, m+1)
        padding = torch.cat([padding[:, :1] & False, padding], dim=1)
        h = x[tokens]
        h[:, 0] = self.subtree_cls + pe[roots]
        bias = self._gather_bias(attn_bias, tokens, padding)
        h = self.local_layer(h, bias)
        subtrees = h[:, 0]                                     # (S, d)

        # Stage 2: the root attends with the subtree class tokens
        sel = torch.cat([roots.new_zeros(1), roots])
        g = torch.cat([x[:1], subtrees], dim=0)
        g_bias = attn_bias[:, sel[:, None], sel[None, :]]
        for layer in self.layers:
            g = layer(g, g_bias)

        # Broadcast subtree context back down to the curve tokens
        keep = ~padding[:, 1:]
        order = members[keep]
        out = x.new_empty(x.shape)
        out[0] = g[0]
        out[order] = h[:, 1:][keep] + self.broadcast(g[1:][subtree_of[order]])
        return self.norm(out)

    def _gather_bias(self, attn_bias, tokens, padding):
        """
        Restricts the (n_heads, T, T) attention bias to each padded subtree,
        masking padded keys, as one (S * n_heads, m, m) attention mask.
        """
        S, m = tokens.shape
        bias = attn_bias[:, tokens[:, :, None], tokens[:, None, :]]
        bias = bias.permute(1, 0, 2, 3)                        # (S, H, m, m)
        masked = padding[:, None, None, :].expand(-1, self.n_heads, m, -1)
        return bias.masked_fill(masked, float("-inf")).flatten(0, 1)

    @staticmethod
    def _pad_subtrees(subtree_of, n_subtrees):
        """
        Indexes the tokens of each subtree into a padded (S, m) matrix, where
        padding slots point at token 0 and are flagged in the mask.
        """
        counts = torch.bincount(subtree_of[1:], minlength=n_subtrees)
        m = int(counts.max())
        order = torch.argsort(subtree_of[1:], stable=True) + 1
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(len(order), device=subtree_of.device)
        slot = slot - starts[subtree_of[order]]
        members = subtree_of.new_zeros(n_subtrees, m)
        padding = torch.ones(
            n_subtrees, m, dtype=torch.bool, device=subtree_of.device
        )
        members[subtree_of[order], slot] = order
        padding[subtree_of[order], slot] = False
        return members, padding
