"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

End-to-end ArboristModel for neuron morphology encoding.

"""

import json
import torch
import torch.nn as nn

from arborist.models.curve_transformer import CurveEncoder
from arborist.models.graph_transformer import GraphTransformer


class Arborist(nn.Module):
    """
    End-to-end neuron morphology encoder.

    Encodes a TreeSample in three stages:
      1. CurveEncoder   — each irreducible path → latent vector z_i
      2. GraphTransformer — message-pass over the line graph of the skeleton
                           so each curve sees its neighboring branches
      3. Readout      — average over all curve embeddings → global tree z

    Parameters
    ----------
    curve_segment_len : int, optional
        Points per curve segment for CurveEncoder. Default is 10.
    curve_d_token : int, optional
        CurveEncoder token dimension. Default is 128.
    curve_n_heads : int, optional
        Attention heads in CurveEncoder. Default is 4.
    curve_n_layers : int, optional
        Transformer layers in CurveEncoder. Default is 4.
    curve_d_ff : int, optional
        CurveEncoder feed-forward dimension. Default is 128.
    graph_n_heads : int, optional
        Attention heads in GraphTransformer. Default is 4.
    graph_n_layers : int, optional
        Transformer layers in GraphTransformer. Default is 3.
    graph_d_ff : int, optional
        GraphTransformer feed-forward dimension. Default is 128.
    latent_dim : int, optional
        Shared latent dimension: CurveEncoder output = GraphTransformer input.
        Default is 64.
    dropout : float, optional
        Dropout probability shared across both sub-models. Default is 0.1.
    """

    def __init__(
        self,
        curve_segment_len=10,
        curve_d_token=64,
        curve_n_heads=4,
        curve_n_layers=4,
        curve_d_ff=128,
        graph_n_heads=4,
        graph_n_layers=3,
        graph_d_ff=128,
        latent_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        self.config = {
            "curve_segment_len": curve_segment_len,
            "curve_d_token": curve_d_token,
            "curve_n_heads": curve_n_heads,
            "curve_n_layers": curve_n_layers,
            "curve_d_ff": curve_d_ff,
            "graph_n_heads": graph_n_heads,
            "graph_n_layers": graph_n_layers,
            "graph_d_ff": graph_d_ff,
            "latent_dim": latent_dim,
            "dropout": dropout,
        }
        self.curve_encoder = CurveEncoder(
            segment_len=curve_segment_len,
            d_token=curve_d_token,
            n_heads=curve_n_heads,
            n_layers=curve_n_layers,
            d_ff=curve_d_ff,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.graph_transformer = GraphTransformer(
            latent_dim=latent_dim,
            n_heads=graph_n_heads,
            n_layers=graph_n_layers,
            d_ff=graph_d_ff,
            dropout=dropout,
        )

    def _collate_curves(self, curves):
        device = next(self.parameters()).device
        lengths = [len(c) for c in curves]
        n_max = max(max(lengths), self.curve_encoder.segment_len)
        B = len(curves)
        diffs = torch.zeros(B, n_max, 3, device=device)
        mask = torch.ones(B, n_max, dtype=torch.bool, device=device)
        for i, (c, l) in enumerate(zip(curves, lengths)):
            diffs[i, :l] = torch.as_tensor(c, dtype=torch.float32, device=device)
            mask[i, :l] = False
        return diffs, mask

    @torch._dynamo.disable
    def encode(self, sample):
        """
        Encodes a TreeSample into per-curve and global tree embeddings.

        Parameters
        ----------
        sample : TreeSample
            A rooted subgraph as returned by GraphDataset.__getitem__.

        Returns
        -------
        z_tree : torch.Tensor
            Shape (latent_dim,) — global tree embedding, mean-pooled over
            all curve embeddings after graph contextualization.
        z_curves : torch.Tensor
            Shape (n_curves, latent_dim) — per-curve embeddings after the
            GraphTransformer contextualizes each curve by its neighbors.
        """
        device = next(self.parameters()).device

        if not sample.curves:
            empty = torch.zeros(self.config["latent_dim"], device=device)
            return empty, empty.unsqueeze(0)

        # Encode curves
        diffs, mask = self._collate_curves(sample.curves)
        z, _ = self.curve_encoder(diffs, mask)

        # Encode graph
        edge_index = torch.as_tensor(
            sample.edge_index, dtype=torch.long, device=device
        )
        z_curves = self.graph_transformer(z, edge_index)
        z_tree = z_curves.mean(dim=0)
        return z_tree, z_curves

    def forward(self, sample):
        return self.encode(sample)

    def save(self, path):
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
