"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

End-to-end Arborist model for neuron morphology encoding. ``Arborist``
encodes one TreeSample into a global and per-curve embeddings;
``ArboristEncoder`` encodes a batch and reads out one vector per tree sample.

"""

import torch
import torch.nn as nn

from arborist.data.datasets import collate_curves
from arborist.models.curve_transformer import CurveEncoder
from arborist.models.graph_transformer import (
    GraphTransformer,
    SubtreeGraphTransformer,
    partition_curves,
)

POOLINGS = ("flat", "subtree")


class Arborist(nn.Module):
    """
    End-to-end neuron morphology encoder.

    Encodes a TreeSample in three stages:
      1. CurveEncoder — each irreducible path → latent vector z_i (shape
                        only; translation-invariant)
      2. Graph transformer — attention across the curves of the subgraph,
                        with each curve's placement and distance from the
                        root as positional encodings. "flat" pooling attends
                        over all curves at once; "subtree" pooling attends
                        within connected subtrees, then between the subtrees
                        and the root.
      3. Readout — CLS token output → global tree z, per-curve outputs →
                   z_curves.

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
        "flat": attention over all curves at once. "subtree": attention
        within each subtree, then between the subtrees and the root. Default
        is "flat".
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
        pooling="flat",
    ):
        super().__init__()
        if pooling not in POOLINGS:
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
        if pooling == "subtree":
            self.graph_transformer = SubtreeGraphTransformer(
                latent_dim=latent_dim,
                d_ff=graph_d_ff,
                dropout=dropout,
                geom_scale=geom_scale,
                n_heads=graph_n_heads,
                n_layers=graph_n_layers,
            )
        else:
            self.graph_transformer = GraphTransformer(
                latent_dim=latent_dim,
                d_ff=graph_d_ff,
                dropout=dropout,
                geom_scale=geom_scale,
                n_heads=graph_n_heads,
                n_layers=graph_n_layers,
            )
        self.cls_token = nn.Parameter(torch.zeros(latent_dim))

    @torch._dynamo.disable
    def encode(self, tree_sample):
        """
        Encodes a TreeSample into per-curve and global tree embeddings.

        Parameters
        ----------
        tree_sample : TreeSample
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

        if not tree_sample.curves:
            empty = torch.zeros(self.config["latent_dim"], device=device)
            return empty, empty.unsqueeze(0)

        # Embed each curve independently via the sequence transformer
        diffs, mask = collate_curves(
            tree_sample.curves, min_len=self.curve_encoder.nodes_per_token, device=device
        )
        z, _ = self.curve_encoder(diffs, mask)

        # Prepend the CLS token as token 0 (the root), matching the token
        # order of the tree sample's positional arrays
        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        kwargs = {}
        if self.config["pooling"] == "subtree":
            subtree_of, roots = self.subtrees(tree_sample)
            kwargs["subtree_of"] = torch.as_tensor(subtree_of, device=device)
            kwargs["roots"] = torch.as_tensor(roots, device=device)

        z_out = self.graph_transformer(
            torch.cat([self.cls_token.unsqueeze(0), z], dim=0),
            endpoints=t(tree_sample.endpoints),
            hops=t(tree_sample.hops),
            cable=t(tree_sample.cable),
            rel_hops=t(tree_sample.rel_hops),
            rel_cable=t(tree_sample.rel_cable),
            **kwargs,
        )

        # CLS token output is the global tree embedding; remaining nodes are per-curve
        z_tree = z_out[0]
        z_curves = z_out[1:]
        return z_tree, z_curves

    def subtrees(self, tree_sample):
        """
        Partitions the tree sample's curves into the subtrees used by
        subtree pooling; see partition_curves.

        Returns
        -------
        subtree_of : numpy.ndarray
            Shape (T,) int. Subtree of each token, -1 for the root.
        roots : numpy.ndarray
            Shape (S,) int. Root-adjacent curve token of each subtree.
        """
        return partition_curves(tree_sample.parents)

    def forward(self, tree_sample):
        return self.encode(tree_sample)

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


class ArboristEncoder(nn.Module):
    """
    Arborist as an embedding module: encodes a batch of TreeSamples and
    reads out one vector per tree sample, for use as the morphology branch of a
    downstream model.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    readout : str, optional
        "cls" or "cls+root_curves" (CLS plus mean root curve embedding).
        Default is "cls+root_curves".
    **arborist_kwargs
        Forwarded to Arborist.
    """

    def __init__(self, latent_dim=64, readout="cls+root_curves", **arborist_kwargs):
        super().__init__()
        if readout not in ("cls", "cls+root_curves"):
            raise ValueError(f"Unknown readout: {readout}")
        self.config = {"latent_dim": latent_dim, "readout": readout, **arborist_kwargs}
        self.readout = readout
        self.embed_dim = 2 * latent_dim if readout == "cls+root_curves" else latent_dim
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)

    @torch._dynamo.disable
    def forward(self, tree_samples):
        """
        Parameters
        ----------
        tree_samples : List[TreeSample]
            Batch of tree samples, as returned by collate_trees.

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim), on this module's device.
        """
        with torch.amp.autocast("cuda", enabled=False):
            z = []
            for tree_sample in tree_samples:
                z_tree, z_curves = self.arborist.encode(tree_sample)
                if self.readout == "cls+root_curves":
                    idx = list(tree_sample.root_curve_indices or range(len(z_curves)))
                    z_tree = torch.cat([z_tree, z_curves[idx].mean(dim=0)])
                z.append(z_tree)
        return torch.stack(z)

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
