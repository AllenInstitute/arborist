"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

End-to-end Arborist model for neuron morphology encoding. ``Arborist``
encodes one TreeSample into a global and per-curve embeddings; ``readout``
turns that encoding into a single vector; ``ArboristEncoder`` does both for
a batch, for use as the morphology branch of a larger model.

"""

import numpy as np
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
# Pooling names saved before the subtree transformer was renamed
LEGACY_POOLING = {"cls": "flat", "hierarchical": "subtree"}


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
                   z_curves; see ``readout``.

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
        Attention pattern of the graph transformer. "flat": one attention
        stack over all curves (GraphTransformer). "subtree": attention
        within connected subtrees of curves, each read out by its own class
        token, then between the root and the subtree tokens
        (SubtreeGraphTransformer). Default is "flat".
    pool_cable : float, optional
        Subtree pooling only. Largest total cable (microns) per subtree.
        None uses the whole subtree under each root-adjacent curve. Default
        is None.
    pool_local_layers : int, optional
        Subtree pooling only. Attention layers within each subtree;
        graph_n_layers then counts the layers between subtrees. Default is 1.
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
        pool_cable=None,
        pool_local_layers=1,
    ):
        super().__init__()
        pooling = LEGACY_POOLING.get(pooling, pooling)
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
        if pooling == "subtree":
            self.graph_transformer = SubtreeGraphTransformer(
                latent_dim=latent_dim,
                n_heads=graph_n_heads,
                n_layers=graph_n_layers,
                n_local_layers=pool_local_layers,
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
        if self.config["pooling"] == "subtree":
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
        Partitions the sample's curves into the connected subtrees used by
        subtree pooling; see partition_curves.

        Returns
        -------
        group_of : numpy.ndarray
            Shape (T,) int. Subtree of each token, -1 for the root.
        heads : numpy.ndarray
            Shape (G,) int. Head token of each subtree.
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
        model.load_state_dict(upgrade_state_dict(checkpoint["state_dict"]))
        return model


def upgrade_state_dict(state_dict):
    """
    Renames the keys of a state_dict saved before SubtreeGraphTransformer
    inherited its between-subtree layers from GraphTransformer
    ("global_layers" -> "layers"). Works under any prefix, so models that
    hold an Arborist can call it on their own state_dict.
    """
    return {
        k.replace("graph_transformer.global_layers.", "graph_transformer.layers."): v
        for k, v in state_dict.items()
    }


# Width of each readout as a multiple of latent_dim
READOUT_WIDTH = {"cls": 1, "cls+root_curves": 2}
READOUTS = tuple(READOUT_WIDTH)


def readout(z_tree, z_curves, sample, kind="cls+root_curves"):
    """
    Reads one embedding out of an encoded TreeSample.

    Parameters
    ----------
    z_tree : torch.Tensor
        Shape (latent_dim,). Global tree embedding from ``Arborist.encode``.
    z_curves : torch.Tensor
        Shape (n_curves, latent_dim). Per-curve embeddings.
    sample : TreeSample
        The encoded sample.
    kind : str, optional
        "cls": the global embedding. "cls+root_curves": the global embedding
        concatenated with the mean embedding of the curves adjacent to the
        root. The root is the site of interest, so this reads its local
        geometry directly instead of only through attention into the CLS
        token. Default is "cls+root_curves".

    Returns
    -------
    torch.Tensor
        Shape (latent_dim * READOUT_WIDTH[kind],).
    """
    if kind == "cls":
        return z_tree
    if kind == "cls+root_curves":
        idx = list(sample.root_curve_indices or range(len(z_curves)))
        return torch.cat([z_tree, z_curves[idx].mean(dim=0)])
    raise ValueError(f"Unknown readout: {kind}")


class ArboristEncoder(nn.Module):
    """
    Arborist as an embedding module: encodes a batch of TreeSamples and
    applies a readout, giving one vector per sample, for use as the
    morphology branch of a downstream model.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    readout : str, optional
        One of READOUTS. Default is "cls+root_curves".
    **arborist_kwargs
        Forwarded to Arborist.
    """

    def __init__(self, latent_dim=64, readout="cls+root_curves", **arborist_kwargs):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"Unknown readout: {readout}")
        self.config = {"latent_dim": latent_dim, "readout": readout, **arborist_kwargs}
        self.readout = readout
        self.embed_dim = latent_dim * READOUT_WIDTH[readout]
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)

    @torch._dynamo.disable
    def forward(self, samples):
        """
        Parameters
        ----------
        samples : List[TreeSample]
            Batch of samples, as returned by collate_trees.

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim), on this module's device.
        """
        # Arborist builds fp32 tensors from numpy inside encode
        with torch.amp.autocast("cuda", enabled=False):
            z = [readout(*self.arborist.encode(s), s, self.readout) for s in samples]
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
