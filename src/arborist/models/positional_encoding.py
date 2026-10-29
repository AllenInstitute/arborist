"""
Created on Wed Oct 28 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Positional encodings for the GraphTransformer. Curve embeddings from the
CurveEncoder are translation invariant, so everything about where a curve
sits in the tree enters through these encodings. Both encodings describe the
same three things — geometric placement, topological distance (hops), and
geometric distance (cable) — once relative to the root (GlobalPE) and once
between every pair of tokens (RelativePE).

Tokens are the root (token 0) followed by the curves. Global quantities are
anchored at a curve's proximal endpoint, so hops is the branch order.
Relative quantities are anchored at the distal endpoint, which maps each
curve to its own topological node (and the root token to the root), so that
siblings are distinguishable from a token and itself.

"""

import torch
import torch.nn as nn


def log_radial(x, scale):
    """Unit direction times log1p(r / scale) for vectors in the last dim."""
    r = x.norm(dim=-1, keepdim=True)
    return x * torch.log1p(r / scale) / r.clamp(min=1e-6)


class GlobalPE(nn.Module):
    """
    Where each token sits relative to the root. Three parts are computed per
    token, concatenated, and projected to latent_dim:

      placement (6)
          Proximal and distal endpoint positions in the subgraph frame with
          the root at the origin, each as unit direction times
          log1p(r / geom_scale).
      topological_distance (1)
          Number of curves on the path from the root to the proximal
          endpoint, i.e. the branch order, compressed as log1p(hops).
      geometric_distance (1)
          Cable length from the root to the proximal endpoint, compressed as
          log1p(cable / geom_scale).

    The root token has all-zero inputs.

    Parameters
    ----------
    latent_dim : int
        Output dimension, matching the curve embeddings it is added to.
    geom_scale : float, optional
        Reference length r0 (microns) for log compression. Default is 10.0.
    """

    def __init__(self, latent_dim, geom_scale=10.0):
        super().__init__()
        self.geom_scale = geom_scale
        self.proj = nn.Sequential(
            nn.Linear(8, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def placement(self, endpoints):
        """(n, 2, 3) proximal/distal xyz -> (n, 6) log-radial positions."""
        return log_radial(endpoints, self.geom_scale).reshape(endpoints.shape[0], 6)

    def topological_distance(self, hops):
        """(n,) curves from root to proximal endpoint -> (n, 1) log-compressed."""
        return torch.log1p(hops).unsqueeze(1)

    def geometric_distance(self, cable):
        """(n,) cable from root to proximal endpoint -> (n, 1) log-compressed."""
        return torch.log1p(cable / self.geom_scale).unsqueeze(1)

    def forward(self, endpoints, hops, cable):
        """
        Parameters
        ----------
        endpoints : torch.Tensor
            Shape (n, 2, 3). Proximal and distal endpoint xyz per token.
        hops : torch.Tensor
            Shape (n,). Curves from the root to the proximal endpoint.
        cable : torch.Tensor
            Shape (n,). Cable length (microns) from the root to the proximal
            endpoint.

        Returns
        -------
        torch.Tensor
            Shape (n, latent_dim).
        """
        return self.proj(torch.cat([
            self.placement(endpoints),
            self.topological_distance(hops),
            self.geometric_distance(cable),
        ], dim=1))


class RelativePE(nn.Module):
    """
    Where each token sits relative to every other token, as a per-head bias
    added to the attention logits. Three parts are computed per ordered pair
    (i, j), concatenated, and projected to one scalar per head:

      placement (3)
          Vector from token i's anchor to token j's anchor, as unit direction
          times log1p(r / geom_scale).
      topological_distance (2)
          (upstream, downstream) hops: curves from i up to the lowest common
          ancestor, then down to j, each compressed as log1p(hops).
          Mirror-symmetric, rel(j, i) = (d, u): self is (0, 0),
          parent->child (0, 1), siblings (1, 1).
      geometric_distance (2)
          (upstream, downstream) cable along the same path, each compressed
          as log1p(cable / geom_scale). Their sum is the tree path distance.

    Parameters
    ----------
    n_heads : int
        Number of attention heads to bias.
    geom_scale : float, optional
        Reference length r0 (microns) for log compression. Default is 10.0.
    d_hidden : int, optional
        Hidden width of the pairwise projection. Default is 32.
    """

    def __init__(self, n_heads, geom_scale=10.0, d_hidden=32):
        super().__init__()
        self.geom_scale = geom_scale
        self.proj = nn.Sequential(
            nn.Linear(7, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_heads),
        )

    def placement(self, anchors):
        """(n, 3) anchor xyz -> (n, n, 3) log-radial vectors from i to j."""
        return log_radial(anchors[None, :, :] - anchors[:, None, :], self.geom_scale)

    def topological_distance(self, rel_hops):
        """(n, n, 2) (upstream, downstream) hops -> log-compressed."""
        return torch.log1p(rel_hops)

    def geometric_distance(self, rel_cable):
        """(n, n, 2) (upstream, downstream) cable -> log-compressed."""
        return torch.log1p(rel_cable / self.geom_scale)

    def forward(self, anchors, rel_hops, rel_cable):
        """
        Parameters
        ----------
        anchors : torch.Tensor
            Shape (n, 3). Distal endpoint xyz per token (origin for the root).
        rel_hops, rel_cable : torch.Tensor
            Shape (n, n, 2). (upstream, downstream) hops and cable between
            every ordered pair of tokens.

        Returns
        -------
        torch.Tensor
            Shape (n_heads, n, n) attention bias.
        """
        pair = torch.cat([
            self.placement(anchors),
            self.topological_distance(rel_hops),
            self.geometric_distance(rel_cable),
        ], dim=-1)
        return self.proj(pair).permute(2, 0, 1)
