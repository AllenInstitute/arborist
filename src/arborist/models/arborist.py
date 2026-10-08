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
        graph_n_pe_components=4,
        graph_max_rel_dist=4,
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
            "graph_n_pe_components": graph_n_pe_components,
            "graph_max_rel_dist": graph_max_rel_dist,
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
            n_pe_components=graph_n_pe_components,
            max_rel_dist=graph_max_rel_dist,
        )
        self.cls_token = nn.Parameter(torch.zeros(latent_dim))
        self.root_proj = nn.Linear(1, latent_dim, bias=False)

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
        diffs, mask = self._collate_curves(sample.curves)
        z, _ = self.curve_encoder(diffs, mask)

        # Inject binary root flag so root-adjacent curves are distinguishable before attention
        root_indices = sample.root_curve_indices
        if root_indices:
            is_root = torch.zeros(len(sample.curves), 1, device=device)
            is_root[root_indices] = 1.0
            z = z + self.root_proj(is_root)

        # Prepend CLS token; shift curve edge_index by 1 to make room
        edge_index = torch.as_tensor(sample.edge_index, dtype=torch.long, device=device)
        edge_index_ext = edge_index + 1 if edge_index.shape[1] > 0 else edge_index
        z_all = torch.cat([self.cls_token.unsqueeze(0), z], dim=0)

        # Connect CLS (node 0) to root-adjacent curves so it acts as the virtual root
        root_shifted = [r + 1 for r in root_indices] if root_indices else list(range(1, len(sample.curves) + 1))
        cls_src = torch.zeros(len(root_shifted), dtype=torch.long, device=device)
        cls_dst = torch.tensor(root_shifted, dtype=torch.long, device=device)
        cls_edges = torch.stack([torch.cat([cls_src, cls_dst]), torch.cat([cls_dst, cls_src])])
        edge_index_ext = torch.cat([edge_index_ext, cls_edges], dim=1)

        # Extend edge_type: original types shift cleanly; CLS→root is P2C, root→CLS is C2P
        orig_et = getattr(sample, "edge_type", None)
        if orig_et is not None:
            cls_et = torch.cat([
                torch.zeros(len(root_shifted), dtype=torch.long, device=device),  # CLS→root: P2C
                torch.ones(len(root_shifted), dtype=torch.long, device=device),   # root→CLS: C2P
            ])
            edge_type_ext = torch.cat([
                torch.as_tensor(orig_et, dtype=torch.long, device=device), cls_et
            ])
        else:
            edge_type_ext = None

        # Contextualize all nodes; heat PE diffuses from CLS (index 0) as the root
        z_out = self.graph_transformer(
            z_all, edge_index_ext,
            root_indices=[0],
            heat_lam=None,
            heat_basis=None,
            edge_type=edge_type_ext,
        )

        # CLS token output is the global tree embedding; remaining nodes are per-curve
        z_tree = z_out[0]
        z_curves = z_out[1:]
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
