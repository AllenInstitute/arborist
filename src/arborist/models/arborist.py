"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

End-to-end ArboristModel for neuron morphology encoding.

"""

import numpy as np
import torch
import torch.nn as nn

from arborist.data.datasets import collate_curves
from arborist.models.curve_transformer import CurveEncoder
from arborist.models.graph_transformer import GraphTransformer
from arborist.models.hierarchical_transformer import (
    HierarchicalGraphTransformer,
    partition_curves,
)


class Arborist(nn.Module):
    """
    End-to-end neuron morphology encoder.

    Encodes a TreeSample in three stages:
      1. CurveEncoder   — each irreducible path → latent vector z_i (shape
                          only; translation-invariant)
      2. GraphTransformer — attention across all curves of the subgraph, with
                           each curve's placement and distance from the root
                           added as a global PE
      3. Readout      — CLS token output → global tree z

    Parameters
    ----------
    curve_nodes_per_token : int, optional
        Nodes per token for CurveEncoder. Default is 10.
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
    geom_scale : float, optional
        Reference length r0 (microns) for log compression in the positional
        encoding; see GlobalPE. Default is 10.0.
    pooling : str, optional
        How the global tree embedding is read out. "cls": one flat attention
        stack over all curves, read from the root token. "hierarchical":
        attention within connected groups of curves, each read out by its
        own class token, then attention between the root and the group
        tokens; see HierarchicalGraphTransformer. Default is "cls".
    pool_cable : float, optional
        Hierarchical pooling only. Largest total cable (microns) per group.
        None groups by the subtree under each root-adjacent curve. Default
        is None.
    pool_local_layers : int, optional
        Hierarchical pooling only. Attention layers within each group;
        graph_n_layers then counts the layers between groups. Default is 1.
    """

    def __init__(
        self,
        curve_nodes_per_token=10,
        curve_d_token=64,
        curve_n_heads=4,
        curve_n_layers=4,
        curve_d_ff=128,
        graph_n_heads=4,
        graph_n_layers=3,
        graph_d_ff=128,
        latent_dim=64,
        dropout=0.1,
        geom_scale=10.0,
        pooling="cls",
        pool_cable=None,
        pool_local_layers=1,
    ):
        super().__init__()
        if pooling not in ("cls", "hierarchical"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.config = {
            "curve_nodes_per_token": curve_nodes_per_token,
            "curve_d_token": curve_d_token,
            "curve_n_heads": curve_n_heads,
            "curve_n_layers": curve_n_layers,
            "curve_d_ff": curve_d_ff,
            "graph_n_heads": graph_n_heads,
            "graph_n_layers": graph_n_layers,
            "graph_d_ff": graph_d_ff,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "geom_scale": geom_scale,
            "pooling": pooling,
            "pool_cable": pool_cable,
            "pool_local_layers": pool_local_layers,
        }
        self.curve_encoder = CurveEncoder(
            nodes_per_token=curve_nodes_per_token,
            d_token=curve_d_token,
            n_heads=curve_n_heads,
            n_layers=curve_n_layers,
            d_ff=curve_d_ff,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        if pooling == "hierarchical":
            self.graph_transformer = HierarchicalGraphTransformer(
                latent_dim=latent_dim,
                n_heads=graph_n_heads,
                n_local_layers=pool_local_layers,
                n_global_layers=graph_n_layers,
                d_ff=graph_d_ff,
                dropout=dropout,
                geom_scale=geom_scale,
            )
        else:
            self.graph_transformer = GraphTransformer(
                latent_dim=latent_dim,
                n_heads=graph_n_heads,
                n_layers=graph_n_layers,
                d_ff=graph_d_ff,
                dropout=dropout,
                geom_scale=geom_scale,
            )
        self.cls_token = nn.Parameter(torch.zeros(latent_dim))

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
            Shape (latent_dim,) — global tree embedding from the CLS token
            after graph contextualization.
        z_curves : torch.Tensor
            Shape (n_curves, latent_dim) — per-curve embeddings after the
            GraphTransformer contextualizes each curve by its neighbors.
        """
        device = next(self.parameters()).device

        if not sample.curves:
            empty = torch.zeros(self.config["latent_dim"], device=device)
            return empty, empty.unsqueeze(0)

        # Embed each curve independently via the sequence transformer
        diffs, mask = collate_curves(
            sample.curves, min_len=self.curve_encoder.nodes_per_token, device=device
        )
        z, _ = self.curve_encoder(diffs, mask)

        # Prepend the CLS token as token 0 (the root), matching the token
        # order of the sample's positional arrays
        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        kwargs = {}
        if self.config["pooling"] == "hierarchical":
            group_of, heads = self.group_curves(sample)
            kwargs["group_of"] = torch.as_tensor(group_of, device=device)
            kwargs["heads"] = torch.as_tensor(heads, device=device)

        z_out = self.graph_transformer(
            torch.cat([self.cls_token.unsqueeze(0), z], dim=0),
            endpoints=t(sample.endpoints),
            hops=t(sample.hops),
            cable=t(sample.cable),
            rel_hops=t(sample.rel_hops),
            rel_cable=t(sample.rel_cable),
            **kwargs,
        )

        # CLS token output is the global tree embedding; remaining nodes are per-curve
        z_tree = z_out[0]
        z_curves = z_out[1:]
        return z_tree, z_curves

    def group_curves(self, sample):
        """
        Partitions the sample's curves into the connected groups used by
        hierarchical pooling; see partition_curves.

        Returns
        -------
        group_of : numpy.ndarray
            Shape (T,) int. Group of each token, -1 for the root.
        heads : numpy.ndarray
            Shape (G,) int. Head token of each group.
        """
        parents = getattr(sample, "parents", None)
        if parents is None:
            # Older samples: the parent is the one token one hop upstream
            # and zero hops downstream of each curve's distal endpoint
            is_parent = (sample.rel_hops[..., 0] == 1) & (
                sample.rel_hops[..., 1] == 0
            )
            parents = is_parent.argmax(axis=1)
            parents[0] = -1
        T = len(parents)
        lengths = sample.rel_cable[np.arange(T), parents, 0]
        lengths[0] = 0
        return partition_curves(parents, lengths, self.config["pool_cable"])

    def forward(self, sample):
        return self.encode(sample)

    def save(self, path):
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
