"""
Created on Mon June 8 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Curve and rooted-subgraph datasets over a SkeletonGraph, and the TreeSample
container consumed by the Arborist model.

"""

from copy import deepcopy
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import pandas as pd
import torch

from arborist.utils.graph_utils import topological_decomposition
from arborist.utils.util import write_json


# --- Dataset Classes ---
class CurveDataset(Dataset):

    def __init__(
        self,
        graph,
        brain_id=None,
        max_length=np.inf,
        segment_len=10,
        transform=None,
    ):
        # Instance attributes
        self.brain_id = brain_id
        self.max_length = max_length
        self.segment_len = segment_len
        self.transform = transform

        # Core data structures
        self.graph = graph
        self.paths = self.get_valid_paths()

    def get_valid_paths(self):
        paths = list()
        for p in self.irreducible_paths():
            length = self.path_length(p)
            if length < self.max_length and len(p) > self.segment_len:
                paths.append(p)
        return paths

    # --- Get Examples ---
    def __getitem__(self, i):
        # Get path
        curve = deepcopy(self.node_xyz[self.paths[i]])
        if self.transform:
            curve = self.transform(curve)

        # Normalize — backward pass avoids aliasing without a copy
        curve -= curve[0]
        curve[-1:0:-1] -= curve[-2::-1]
        return curve

    # --- Helpers ---
    def curve_lengths(self):
        return np.array([self.path_length(p) for p in self.paths])

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        lengths = self.curve_lengths()
        num_neurons = self.graph.num_connected_components()
        return (
            f"CurveDataset("
            f"\n   brain_id={self.brain_id}, "
            f"\n   num_neurons={num_neurons}, "
            f"\n   num_curves={len(self)}, "
            f"\n   min_length={np.min(lengths):.2f}, "
            f"\n   mean_length={np.mean(lengths):.2f}, "
            f"\n   max_length={np.max(lengths):.2f},"
            f"\n)"
        )


class GraphDataset(Dataset):
    """
    Dataset over rooted subgraphs of a single SkeletonGraph.

    Each item corresponds to one root node. __getitem__ extracts the
    subgraph within "max_depth" microns, decomposes it into irreducible paths
    via topological decomposition, computes first-order finite differences for
    each path, and returns a TreeSample with those curves and their placement
    relative to the root.
    """

    def __init__(self, graph, root_nodes, max_depth=None, graph_transform=None):
        """
        Instantiates a GraphDataset object.

        Parameters
        ----------
        graph : SkeletonGraph
            The full skeleton graph to sample from.
        root_nodes : List[int]
            One root node per dataset item.
        max_depth : float
            Depth in microns for rooted subgraph extraction.
        graph_transform : callable, optional
            Applied to the full (N_nodes, 3) node_xyz array of the rooted
            subgraph before path decomposition (e.g. GraphTransforms). Because
            it acts on all node coordinates at once, every curve in the
            subgraph receives the same rotation and mirror flip, preserving
            inter-curve spatial relationships. Default is None.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.graph = graph
        self.root_nodes = root_nodes
        self.max_depth = max_depth
        self.graph_transform = graph_transform
        self.config = {
            "max_depth": max_depth,
            "graph_transform": type(graph_transform).__name__ if graph_transform else None,
        }


    def __getitem__(self, i):
        # Extract tree sample components
        root = self.root_nodes[i]
        subgraph = self.graph.rooted_subgraph(root, self.max_depth)
        if self.graph_transform:
            subgraph.node_xyz = self.graph_transform(subgraph.node_xyz)
        _, paths, topo_edge_index = topological_decomposition(subgraph)

        # Translate and difference each curve
        curves = []
        for path in paths:
            xyz = subgraph.node_xyz[path].copy()
            xyz -= xyz[0]
            xyz[-1:0:-1] -= xyz[-2::-1]
            curves.append(xyz)

        return build_tree_sample(curves, topo_edge_index)

    def __len__(self):
        return len(self.root_nodes)

    def save_config(self, path):
        """
        Saves dataset parameters to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        write_json(path, self.config)


@dataclass(repr=False)
class TreeSample:
    """
    A rooted subgraph ready to pass through CurveEncoder then GraphTransformer.
    Construct via ``build_tree_sample``.

    The curves themselves are translation invariant; the positional arrays
    say where each curve sits relative to the root and to every other curve
    and feed the positional encodings (see arborist.models.positional_encoding).
    They are indexed by token: token 0 is the root and ``curves[k]`` is token
    ``k + 1``, with T = n_curves + 1 tokens in all.

    Attributes
    ----------
    curves : List[numpy.ndarray]
        One array per irreducible path, each of shape (N_i, 3). Values are
        first-order finite differences with a leading zero row, matching the
        convention expected by CurveEncoder.
    root_curve_indices : List[int]
        Indices into ``curves`` of the curves adjacent to the root.
    endpoints : numpy.ndarray
        Shape (T, 2, 3) float32. Proximal and distal endpoint xyz of each
        token in the subgraph frame with the root at the origin; proximal is
        the endpoint closer to the root. Both are the origin for the root.
    hops : numpy.ndarray
        Shape (T,) float32. Number of curves on the path from the root to
        each token's proximal endpoint (branch order).
    cable : numpy.ndarray
        Shape (T,) float32. Cable length (microns) along the path from the
        root to each token's proximal endpoint.
    rel_hops : numpy.ndarray
        Shape (T, T, 2) float32. For tokens (i, j), anchored at their distal
        endpoints: curves from i up to the lowest common ancestor, then down
        to j. rel_hops[j, i] is rel_hops[i, j] reversed.
    rel_cable : numpy.ndarray
        Shape (T, T, 2) float32. Cable length (microns) along the same
        upstream and downstream path segments.
    target : any, optional
        Supervised label for this sample (scalar, array, or dict). Stored
        alongside the sample so DataLoader batches can be self-contained.
    """

    curves: list
    root_curve_indices: list
    endpoints: np.ndarray
    hops: np.ndarray
    cable: np.ndarray
    rel_hops: np.ndarray
    rel_cable: np.ndarray
    target: object = None

    def __repr__(self):
        return f"TreeSample(n_curves={len(self.curves)})"


def build_tree_sample(curves, topo_edge_index, target=None):
    """
    Build a fully-annotated TreeSample from preprocessed curves and topology.

    Walks the tree once to place every curve's endpoints in the root frame and
    accumulate hop count and cable length from the root, then derives the
    pairwise (upstream, downstream) hops and cable between tokens from their
    lowest common ancestors, so model forward passes skip all Python-side
    computation. All callers that construct TreeSamples should go through
    this factory to stay consistent.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Finite-difference curve arrays (already translated and differenced),
        each of shape (N_i, 3).
    topo_edge_index : List[Tuple[int, int]]
        Topological graph edges, one per curve, as (proximal, distal) pairs
        with node 0 the root and every curve listed after its parent, as
        produced by ``topological_decomposition``.
    target : any, optional
        Supervised label to embed in the sample.

    Returns
    -------
    TreeSample
    """
    n = len(curves)
    topo_idx = np.asarray(topo_edge_index, dtype=np.int64).reshape(n, 2)
    n_topo = 1 + int(topo_idx.max()) if n else 1

    # Walk parent-first: a curve's distal node is its proximal node plus the
    # sum of its step vectors, one more hop, and its cable length further out.
    # ancestors[v] marks v and every node on its path to the root.
    pos = np.zeros((n_topo, 3))
    hops = np.zeros(n_topo, dtype=np.float32)
    cable = np.zeros(n_topo, dtype=np.float32)
    ancestors = np.eye(n_topo, dtype=bool)
    visited = np.zeros(n_topo, dtype=bool)
    visited[0] = True
    for k, (p, d) in enumerate(topo_idx):
        if not visited[p] or visited[d]:
            raise ValueError(
                f"topo_edge_index must list (proximal, distal) edges parent-first "
                f"from root 0; curve {k} = ({p}, {d}) violates this"
            )
        visited[d] = True
        steps = np.asarray(curves[k], dtype=np.float64)
        pos[d] = pos[p] + steps.sum(axis=0)
        hops[d] = hops[p] + 1
        cable[d] = cable[p] + np.linalg.norm(steps, axis=1).sum()
        ancestors[d] |= ancestors[p]

    # Tokens: the root, then each curve. Global arrays anchor at the proximal
    # node; relative arrays anchor at the distal node, which is unique per curve.
    proximal = np.concatenate([[0], topo_idx[:, 0]])
    distal = np.concatenate([[0], topo_idx[:, 1]])
    endpoints = np.stack([pos[proximal], pos[distal]], axis=1)

    # Lowest common ancestor of each token pair: the deepest shared ancestor
    anc = ancestors[distal]                                        # (T, n_topo)
    common = anc[:, None, :] & anc[None, :, :]                     # (T, T, n_topo)
    lca = (common * (hops + 1)).argmax(axis=-1)                    # (T, T)
    up, down = hops[distal][:, None] - hops[lca], hops[distal][None, :] - hops[lca]
    rel_hops = np.stack([up, down], axis=-1)
    up, down = cable[distal][:, None] - cable[lca], cable[distal][None, :] - cable[lca]
    rel_cable = np.stack([up, down], axis=-1)

    return TreeSample(
        curves=curves,
        root_curve_indices=[k for k in range(n) if topo_idx[k, 0] == 0],
        endpoints=endpoints.astype(np.float32),
        hops=hops[proximal],
        cable=cable[proximal],
        rel_hops=rel_hops.astype(np.float32),
        rel_cable=rel_cable.astype(np.float32),
        target=target,
    )


class DatasetCollection(Dataset):
    """
    A flat, indexable view over multiple datasets (one per brain/specimen).

    Parameters
    ----------
    datasets : List[Dataset]
        Constituent datasets to combine.
    weight_fn : callable, optional
        Maps a dataset to a 1-D array of per-item sampling weights. Used by
        samplers for non-uniform drawing (e.g. length-weighted curve sampling).
        Defaults to uniform weights when None. Default is None.
    is_val : bool, optional
        If True, precomputes a fixed set of examples at construction time.
        Default is False.
    n_val_examples : int, optional
        Number of validation examples to precompute. Default is 1000.
    seed : int, optional
        Random seed for reproducible val set. Default is 42.
    """

    def __init__(
        self,
        datasets,
        weight_fn=None,
        is_val=False,
        n_val_examples=1000,
        seed=42,
    ):
        self.datasets = datasets
        self.is_val = is_val
        self.n_val_examples = n_val_examples
        self.seed = seed
        self._build_index(weight_fn)
        if is_val:
            self.val_examples = self._precompute_val(n_val_examples, seed)

    def _build_index(self, weight_fn):
        rows = []
        for ds_idx, dataset in enumerate(self.datasets):
            n = len(dataset)
            weights = weight_fn(dataset) if weight_fn is not None else np.ones(n)
            rows.append(pd.DataFrame({
                "ds_idx": np.full(n, ds_idx, dtype=int),
                "item_idx": np.arange(n),
                "weight": weights,
            }))
        self.index = pd.concat(rows, ignore_index=True)

    def _precompute_val(self, n, seed):
        rng = np.random.default_rng(seed)
        p = self.index["weight"].values / self.index["weight"].sum()
        idxs = rng.choice(len(self.index), size=n, replace=False, p=p)
        examples = []
        for i in idxs:
            ds_idx = self.index["ds_idx"][i]
            item_idx = self.index["item_idx"][i]
            ds = self.datasets[ds_idx]
            # Suppress augmentation so val items are fixed and unaugmented
            orig = ds.__dict__.get("graph_transform")
            if orig is not None:
                ds.__dict__["graph_transform"] = None
            examples.append(ds[item_idx])
            if orig is not None:
                ds.__dict__["graph_transform"] = orig
        return examples

    def __getitem__(self, i):
        if self.is_val:
            return self.val_examples[i]
        ds_idx = self.index["ds_idx"][i]
        item_idx = self.index["item_idx"][i]
        return self.datasets[ds_idx][item_idx]

    def __len__(self):
        if self.is_val:
            return len(self.val_examples)
        return len(self.index)

    def save_config(self, path):
        """
        Saves collection and dataset parameters to a single JSON file.

        Dataset parameters are taken from the first constituent dataset
        (all datasets in a collection share the same configuration).

        Parameters
        ----------
        path : str
            Destination file path.
        """
        config = {
            "n_datasets": len(self.datasets),
            "is_val": self.is_val,
            "n_val_examples": self.n_val_examples,
            "seed": self.seed,
        }
        if self.datasets and hasattr(self.datasets[0], "config"):
            config.update(self.datasets[0].config)
        write_json(path, config)

    def curve_lengths(self):
        return np.concatenate([ds.curve_lengths() for ds in self.datasets])

    def __repr__(self):
        return (
            f"DatasetCollection("
            f"num_datasets={len(self.datasets)}, "
            f"num_items={len(self.index)})"
        )


# --- DataLoader Classes ---
class CurveSampler(Sampler):

    def __init__(self, dataset, examples_per_epoch):
        """
        Parameters
        ----------
        dataset : CurveDatasetCollection
            Dataset to sample from.
        """
        self.dataset = dataset
        self.examples_per_epoch = examples_per_epoch

    def __iter__(self):
        idxs = self.dataset.index.sample(
            self.examples_per_epoch, replace=True, weights="weight"
        ).index
        return iter(np.array(idxs))

    def __len__(self):
        return self.examples_per_epoch


def collate_curves(curves, min_len=1, device=None):
    """
    Pads a list of curves to the longest in the batch and generates an
    attention mask.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Each of shape (N_i, 3), where N_i can vary.
    min_len : int, optional
        Minimum padded length, e.g. CurveEncoder's segment_len. Default is 1.
    device : torch.device, optional
        Device for the returned tensors. Default is None (CPU).

    Returns
    -------
    padded : torch.Tensor
        Shape (B, N_max, 3), zero-padded.
    mask : torch.Tensor
        Shape (B, N_max), True where padding.
    """
    lengths = [len(c) for c in curves]
    n_max = max(max(lengths), min_len)
    B = len(curves)

    padded = torch.zeros(B, n_max, 3, device=device)
    mask = torch.ones(B, n_max, dtype=torch.bool, device=device)
    for i, (c, l) in enumerate(zip(curves, lengths)):
        padded[i, :l] = torch.as_tensor(c, dtype=torch.float32, device=device)
        mask[i, :l] = False
    return padded, mask


def build_dataloader(
    dataset,
    batch_size=32,
    examples_per_epoch=5000,
    num_workers=0,
    use_sampler=True,
):
    """
    Builds a DataLoader for a DatasetCollection that samples with respect
    to curve length.

    Parameters
    ----------
    dataset : DatasetCollection
        Dataset to input to dataloader.
    batch_size : int, optional
        Number of curves per batch. Default is 32.
    examples_per_epoch : int, optional
        Number of examples per epoch. Default is 5000.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 0.

    Returns
    -------
    DataLoader
    """
    if use_sampler:
        sampler = CurveSampler(dataset, examples_per_epoch)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_curves,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )


def collate_trees(samples):
    """
    Collate a list of TreeSamples into a batch.

    Trees vary in the number of curves, so they cannot be naively stacked.
    The samples are returned as a plain list; targets, when present, are
    stacked into a float tensor. The model is expected to loop over the list
    and accumulate gradients across the batch.

    Returns
    -------
    samples : List[TreeSample]
    targets : torch.Tensor or None
        Shape (B,) or (B, ...) if all samples carry a target, else None.
    """
    targets = None
    if samples[0].target is not None:
        targets = torch.stack([
            torch.as_tensor(s.target, dtype=torch.float32) for s in samples
        ])
    return samples, targets


def build_graph_dataloader(dataset, batch_size=4, num_workers=0, shuffle=True):
    """
    DataLoader for GraphDataset — each batch is (list[TreeSample], targets).

    Parameters
    ----------
    dataset : GraphDataset
    batch_size : int, optional
        Default is 4.
    num_workers : int, optional
        Default is 0.
    shuffle : bool, optional
        Default is True.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_trees,
        num_workers=num_workers,
        shuffle=shuffle,
    )
