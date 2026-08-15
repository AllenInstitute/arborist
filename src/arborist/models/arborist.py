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
      3. Mean-pool      — average over all curve embeddings → global tree z

    Parameters
    ----------
    segment_len : int, optional
        Points per curve segment for CurveEncoder. Default is 10.
    d_token : int, optional
        CurveEncoder token dimension. Default is 128.
    curve_n_heads : int, optional
        Attention heads in CurveEncoder. Default is 4.
    curve_n_layers : int, optional
        Transformer layers in CurveEncoder. Default is 4.
    d_ff_curve : int, optional
        CurveEncoder feed-forward dimension. Default is 256.
    latent_dim : int, optional
        Shared latent dimension: CurveEncoder output = GraphTransformer input.
        Default is 64.
    graph_n_heads : int, optional
        Attention heads in GraphTransformer. Default is 4.
    graph_n_layers : int, optional
        Transformer layers in GraphTransformer. Default is 3.
    d_ff_graph : int, optional
        GraphTransformer feed-forward dimension. Default is 256.
    dropout : float, optional
        Dropout probability shared across both sub-models. Default is 0.1.
    """

    def __init__(
        self,
        segment_len=10,
        d_token=128,
        curve_n_heads=4,
        curve_n_layers=4,
        d_ff_curve=256,
        latent_dim=64,
        graph_n_heads=4,
        graph_n_layers=3,
        d_ff_graph=256,
        dropout=0.1,
    ):
        super().__init__()
        self.config = {
            "segment_len": segment_len,
            "d_token": d_token,
            "curve_n_heads": curve_n_heads,
            "curve_n_layers": curve_n_layers,
            "d_ff_curve": d_ff_curve,
            "latent_dim": latent_dim,
            "graph_n_heads": graph_n_heads,
            "graph_n_layers": graph_n_layers,
            "d_ff_graph": d_ff_graph,
            "dropout": dropout,
        }
        self.curve_encoder = CurveEncoder(
            segment_len=segment_len,
            d_token=d_token,
            n_heads=curve_n_heads,
            n_layers=curve_n_layers,
            d_ff=d_ff_curve,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.graph_transformer = GraphTransformer(
            d_model=latent_dim,
            n_heads=graph_n_heads,
            n_layers=graph_n_layers,
            d_ff=d_ff_graph,
            dropout=dropout,
        )

    def _collate_curves(self, curves):
        device = next(self.parameters()).device
        lengths = [len(c) for c in curves]
        n_max = max(lengths)
        B = len(curves)
        diffs = torch.zeros(B, n_max, 3, device=device)
        mask = torch.ones(B, n_max, dtype=torch.bool, device=device)
        for i, (c, l) in enumerate(zip(curves, lengths)):
            diffs[i, :l] = torch.as_tensor(c, dtype=torch.float32, device=device)
            mask[i, :l] = False
        return diffs, mask

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

        # Encode all curves in parallel (treat each as an independent batch item)
        diffs, mask = self._collate_curves(sample.curves)   # (n_curves, N_max, 3)
        z, _ = self.curve_encoder(diffs, mask)              # (n_curves, latent_dim)

        # Contextualize via graph topology
        edge_index = torch.as_tensor(
            sample.edge_index, dtype=torch.long, device=device
        )
        z_curves = self.graph_transformer(z, edge_index)    # (n_curves, latent_dim)

        z_tree = z_curves.mean(dim=0)                       # (latent_dim,)
        return z_tree, z_curves

    def forward(self, sample):
        return self.encode(sample)

    def save_config(self, path):
        with open(path, "w") as f:
            json.dump(self.config, f)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
