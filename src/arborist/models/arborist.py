"""
Created on Mon Aug 6 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

End-to-end ArboristModel for neuron morphology encoding.

"""

import json
import numpy as np
import torch
import torch.nn as nn

from arborist.models.curve_transformer import CurveEncoder
from arborist.models.graph_transformer import GraphTransformer


class Arborist(nn.Module):
    """
    End-to-end neuron morphology encoder.

    Encodes a TreeSample in three stages:
      1. CurveEncoder   — each irreducible path → latent vector z_i (shape
                          only; translation-invariant), plus a projection of
                          the curve's placement in the tree frame
      2. GraphTransformer — message-pass over the line graph of the skeleton
                           so each curve sees its neighboring branches
      3. Readout      — CLS token output → global tree z

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
    geom_scale : float, optional
        Reference length r0 (microns) for log-radial compression of the
        per-curve placement features. Endpoint positions keep their direction
        while the radius becomes log1p(r / r0); cable length becomes
        log1p(L / r0). Resolution is near-linear below r0 and logarithmic
        above, so bounded subgraphs and whole-brain arbors share one
        order-one feature range. Default is 10.0.
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
        geom_scale=10.0,
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
            "geom_scale": geom_scale,
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

        # Curve placement in the tree frame (proximal xyz, distal xyz, length)
        self.geom_scale = geom_scale
        self.geom_proj = nn.Sequential(
            nn.Linear(7, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _compress_geom(self, geom):
        """Log-radial compression of (n, 7) placement features; see geom_scale."""
        pos = geom[:, :6].reshape(-1, 2, 3)
        r = pos.norm(dim=-1, keepdim=True)
        pos = pos * torch.log1p(r / self.geom_scale) / r.clamp(min=1e-6)
        length = torch.log1p(geom[:, 6:] / self.geom_scale)
        return torch.cat([pos.reshape(-1, 6), length], dim=1)

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

        # Add each curve's placement in the shared tree frame; the curve
        # encoder itself is translation-invariant and only sees shape
        geom = torch.as_tensor(sample.curve_geom, dtype=torch.float32, device=device)
        z = z + self.geom_proj(self._compress_geom(geom))

        root_indices = sample.root_curve_indices

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

        # Extend precomputed hop_dist to include CLS as node 0
        orig_hop = getattr(sample, "hop_dist", None)
        if orig_hop is not None:
            orig = torch.as_tensor(orig_hop, dtype=torch.long, device=device)  # (n, n)
            n_c = orig.shape[0]
            hop_ext = torch.full((n_c + 1, n_c + 1), -1, dtype=torch.long, device=device)
            hop_ext[0, 0] = 0
            hop_ext[1:, 1:] = orig
            # CLS row: 1 + min dist from any root curve to each curve
            root_t = torch.tensor(root_indices or list(range(n_c)), dtype=torch.long, device=device)
            min_d = orig[root_t].clone()
            min_d[min_d < 0] = 127          # sentinel for unreachable before min
            cls_row, _ = min_d.min(dim=0)   # (n_c,)
            cls_row = cls_row + 1
            cls_row[cls_row > 127] = -1     # propagate unreachable
            hop_ext[0, 1:] = cls_row
            hop_ext[1:, 0] = cls_row
        else:
            hop_ext = None

        # CLS sits at the topological root (node 0); curves read heat at their endpoints
        node_endpoints = np.concatenate(
            [np.zeros((1, 2), dtype=np.int64), np.asarray(sample.curve_endpoints, dtype=np.int64)]
        )

        z_out = self.graph_transformer(
            z_all, edge_index_ext,
            heat_lam=sample.heat_lam,
            heat_basis=sample.heat_basis,
            node_endpoints=node_endpoints,
            edge_type=edge_type_ext,
            hop_dist=hop_ext,
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
