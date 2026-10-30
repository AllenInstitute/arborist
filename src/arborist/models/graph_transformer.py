"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Transformer for contextualizing curve embeddings over skeleton topology. Each
token is one curve (plus the root as token 0) and attends to every other
token; tree structure enters through the positional encodings in
arborist.models.positional_encoding: a global PE added to the tokens and a
relative PE added to the attention logits.

"""

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
            Shape (n, latent_dim).
        attn_bias : torch.Tensor
            Shape (n_heads, n, n) float, added to the attention logits.
        """
        h = self.norm1(x).unsqueeze(0)                 # (1, n, latent_dim)
        h, _ = self.attn(h, h, h, attn_mask=attn_bias)
        x = x + self.dropout(h.squeeze(0))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
