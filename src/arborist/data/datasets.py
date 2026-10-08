"""
Created on Mon June 8 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import pandas as pd
import torch

from arborist.utils.graph_utils import topological_decomposition
from arborist.utils.util import write_json


# Line-graph edge direction types (stored on TreeSample.edge_type)
PARENT_TO_CHILD = 0   # curve i is closer to root than curve j
CHILD_TO_PARENT = 1   # curve i is farther from root than curve j
SIBLING = 2           # curves i and j share the same parent branch point
NO_EDGE = 3           # default for pairs with no direct line-graph edge


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
    each path, and returns a TreeSample with those curves and the line-graph
    connectivity between them.
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


class TreeSample:
    """
    A rooted subgraph ready to pass through CurveEncoder then GraphTransformer.

    Attributes
    ----------
    curves : List[numpy.ndarray]
        One array per irreducible path, each of shape (N_i, 3). Values are
        first-order finite differences with a leading zero row, matching the
        convention expected by CurveEncoder.
    edge_index : numpy.ndarray
        Shape (2, E), dtype int64. Line-graph adjacency: two curves share an
        edge when they meet at a topological node (branch point or leaf),
        so message passing over this graph communicates between neighboring
        branches.
    root_curve_indices : List[int] or None
        Indices of curves adjacent to the root topological node.
    heat_lam : numpy.ndarray or None
        Shape (n,). Eigenvalues of the normalized line-graph Laplacian,
        precomputed so the model only contracts with learned time scales.
    heat_basis : numpy.ndarray or None
        Shape (n, n). Pre-multiplied eigenbasis: basis[v, j] = phi_j(v) *
        phi_j_root_mean, so H_t(root, v) = sum_j basis[v,j]*exp(-lam_j*t).
    is_root : numpy.ndarray or None
        Shape (n,) float32. 1.0 for root-adjacent curves, 0.0 otherwise.
    edge_type : numpy.ndarray or None
        Shape (E,) int64. Direction type for each line-graph edge, parallel
        to edge_index. Values are PARENT_TO_CHILD (0), CHILD_TO_PARENT (1),
        SIBLING (2). Used by GraphTransformer as per-head attention biases.
    hop_dist : numpy.ndarray or None
        Shape (n, n) int8. All-pairs BFS hop distances on the line graph,
        precomputed so the model forward skips Python BFS entirely.
        -1 encodes unreachable pairs.
    target : any, optional
        Supervised label for this sample (scalar, array, or dict). Stored
        alongside the sample so DataLoader batches can be self-contained.
    """

    def __init__(
        self,
        curves,
        edge_index,
        root_curve_indices=None,
        heat_lam=None,
        heat_basis=None,
        is_root=None,
        edge_type=None,
        hop_dist=None,
        target=None,
    ):
        self.curves = curves
        self.edge_index = edge_index
        self.root_curve_indices = root_curve_indices
        self.heat_lam = heat_lam
        self.heat_basis = heat_basis
        self.is_root = is_root
        self.edge_type = edge_type
        self.hop_dist = hop_dist
        self.target = target

    def __repr__(self):
        return (
            f"TreeSample("
            f"n_curves={len(self.curves)}, "
            f"n_edges={self.edge_index.shape[1]})"
        )


def _topo_depths(topo_edge_index):
    """BFS from topological node 0; returns {node: depth} for all reachable nodes."""
    adj = defaultdict(list)
    for u, v in topo_edge_index:
        adj[u].append(v)
        adj[v].append(u)
    depth = {0: 0}
    queue = [0]
    while queue:
        nxt = []
        for u in queue:
            for v in adj[u]:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    nxt.append(v)
        queue = nxt
    return depth


def _build_line_graph(topo_edge_index, topo_depths):
    """
    Build the line-graph edge_index and a parallel edge_type array.

    For each pair of curves (i, j) sharing a topological node t the edge
    type is determined by whether t is the proximal (root-side) or distal
    (leaf-side) endpoint of each curve:

      PARENT_TO_CHILD  t == distal[i]   and t == proximal[j]
      CHILD_TO_PARENT  t == proximal[i] and t == distal[j]
      SIBLING          t == proximal[i] and t == proximal[j]
    """
    # Determine proximal/distal endpoint of each curve from BFS depths
    proximal, distal = {}, {}
    for k, (u, v) in enumerate(topo_edge_index):
        if topo_depths.get(u, 0) <= topo_depths.get(v, 0):
            proximal[k], distal[k] = u, v
        else:
            proximal[k], distal[k] = v, u

    # Group curves by shared topological node
    topo_to_curves = defaultdict(list)
    for k, (u, v) in enumerate(topo_edge_index):
        topo_to_curves[u].append(k)
        topo_to_curves[v].append(k)

    src, dst, types = [], [], []
    for t, curves_at_t in topo_to_curves.items():
        for i in curves_at_t:
            for j in curves_at_t:
                if i == j:
                    continue
                src.append(i)
                dst.append(j)
                if t == distal[i] and t == proximal[j]:
                    types.append(PARENT_TO_CHILD)
                elif t == proximal[i] and t == distal[j]:
                    types.append(CHILD_TO_PARENT)
                else:
                    types.append(SIBLING)

    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.array([src, dst], dtype=np.int64), np.array(types, dtype=np.int64)


def build_tree_sample(curves, topo_edge_index, target=None):
    """
    Build a fully-annotated TreeSample from preprocessed curves and topology.

    Computes every structural field the model uses — edge types, heat-diffusion
    basis, hop distances, and root flags — so model forward passes skip all
    Python-side computation. All callers that construct TreeSamples should go
    through this factory to stay consistent.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Finite-difference curve arrays (already translated and differenced),
        each of shape (N_i, 3).
    topo_edge_index : List[Tuple[int, int]]
        Topological graph edges, one per curve, as (src_topo, dst_topo) pairs.
        Node 0 is the root.
    target : any, optional
        Supervised label to embed in the sample.

    Returns
    -------
    TreeSample
    """
    topo_depths = _topo_depths(topo_edge_index)
    edge_index, edge_type = _build_line_graph(topo_edge_index, topo_depths)
    n = len(curves)
    root_curve_indices = [k for k, (u, v) in enumerate(topo_edge_index) if 0 in (u, v)]
    heat_lam, heat_basis = _compute_heat_basis(n, edge_index, root_curve_indices)
    hop_dist = _compute_hop_dist(n, edge_index)
    is_root = np.zeros(n, dtype=np.float32)
    if root_curve_indices:
        is_root[root_curve_indices] = 1.0
    return TreeSample(
        curves=curves,
        edge_index=edge_index,
        root_curve_indices=root_curve_indices,
        heat_lam=heat_lam,
        heat_basis=heat_basis,
        is_root=is_root,
        edge_type=edge_type,
        hop_dist=hop_dist,
        target=target,
    )


def _compute_hop_dist(n, edge_index):
    """BFS all-pairs shortest paths on the line graph. int8 array, -1 = unreachable."""
    dist = np.full((n, n), -1, dtype=np.int8)
    np.fill_diagonal(dist, 0)
    if n <= 1 or edge_index.shape[1] == 0:
        return dist
    adj = [[] for _ in range(n)]
    for u, v in edge_index.T.tolist():
        adj[u].append(v)
    for src in range(n):
        queue, seen, d = [src], {src}, 0
        while queue:
            nxt = []
            for u in queue:
                dist[src, u] = d
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            queue, d = nxt, d + 1
    return dist


def _compute_heat_basis(n, edge_index, root_indices):
    """
    Precompute the normalized Laplacian eigenbasis projected onto the root.

    Returns lam (n,) and basis (n, n) where basis[v, j] = phi_j(v) *
    phi_j_root_mean, so the model only needs to contract with learned time
    scales: H_t(root, v) = sum_j basis[v, j] * exp(-lam_j * t).
    """
    if n <= 1:
        return np.zeros(n, dtype=np.float32), np.zeros((n, n), dtype=np.float32)

    # Normalized Laplacian of the line graph
    adj = np.zeros((n, n), dtype=np.float64)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(axis=1).clip(min=1e-8)
    d_inv_sqrt = deg ** -0.5
    L = np.eye(n) - d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    # Eigendecomposition — sorted ascending by eigenvalue
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        return np.zeros(n, dtype=np.float32), np.zeros((n, n), dtype=np.float32)

    # Project eigenvectors onto root so basis[v,j] = phi_j(v) * phi_j(root)
    root_proj = eigvecs[root_indices].mean(axis=0) if root_indices else eigvecs[0]
    basis = eigvecs * root_proj[None, :]
    return eigvals.astype(np.float32), basis.astype(np.float32)


def _build_line_graph_edge_index(topo_edge_index):
    """
    Converts topological-graph edge pairs into line-graph edge pairs.

    In the topological graph each node is a branching/leaf point and each
    edge is an irreducible path (a curve). In the line graph each curve
    becomes a node and two curve-nodes are connected when they share a
    topological endpoint.

    Parameters
    ----------
    topo_edge_index : List[Tuple[int, int]]
        Edges of the topological graph as (src_topo_idx, dst_topo_idx) pairs,
        parallel to the list of curves.

    Returns
    -------
    numpy.ndarray
        Shape (2, E), int64.
    """
    topo_to_curves = defaultdict(list)
    for curve_idx, (u, v) in enumerate(topo_edge_index):
        topo_to_curves[u].append(curve_idx)
        topo_to_curves[v].append(curve_idx)

    src, dst = [], []
    for neighbors in topo_to_curves.values():
        for i in neighbors:
            for j in neighbors:
                if i != j:
                    src.append(i)
                    dst.append(j)

    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


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
            examples.append(self.datasets[ds_idx][item_idx])
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


def collate_curves(curves):
    """
    Pads a list of curves to the longest in the batch and generates an
    attention mask.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Each of shape (N_i, 3), where N_i can vary.

    Returns
    -------
    padded : torch.Tensor
        Shape (B, N_max, 3), zero-padded.
    mask : torch.Tensor
        Shape (B, N_max), True where padding.
    """
    lengths = [len(c) for c in curves]
    n_max = max(lengths)
    B = len(curves)

    padded = torch.zeros(B, n_max, 3)
    mask = torch.ones(B, n_max, dtype=torch.bool)
    for i, (c, l) in enumerate(zip(curves, lengths)):
        padded[i, :l] = torch.tensor(c)
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
    Builds a DataLoader for a PathsDatasetCollection that samples with respect
    to curve length.

    Parameters
    ----------
    dataset : PathsDatasetCollection
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
