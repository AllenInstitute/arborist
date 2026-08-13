"""
Created on Mon June 8 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, Sampler

import networkx as nx
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

        # Normalize
        curve -= curve[0]
        curve[1:] -= curve[:-1]
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
        num_neurons = nx.number_connected_components(self.graph)
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

    def __init__(self, graph, root_nodes, max_depth=None, transform=None, graph_transform=None):
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
        transform : callable, optional
            Applied to each raw xyz array before differencing (e.g.
            CurveTransforms for augmentation). Default is None.
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
        self.transform = transform
        self.graph_transform = graph_transform
        self.config = {
            "max_depth": max_depth,
            "transform": type(transform).__name__ if transform else None,
            "graph_transform": type(graph_transform).__name__ if graph_transform else None,
        }


    def __getitem__(self, i):
        # Extract tree sample components
        root = self.root_nodes[i]
        subgraph = self.graph.rooted_subgraph(root, self.max_depth)
        if self.graph_transform:
            subgraph.node_xyz = self.graph_transform(subgraph.node_xyz)
        _, paths, topo_edge_index = topological_decomposition(subgraph)

        # Create list of curves
        curves = []
        for path in paths:
            xyz = subgraph.node_xyz[path].copy()
            if self.transform:
                xyz = self.transform(xyz)
            xyz -= xyz[0]
            xyz[1:] -= xyz[:-1].copy()
            curves.append(xyz)

        # Create TreeSample
        edge_index = _build_line_graph_edge_index(topo_edge_index)
        return TreeSample(curves=curves, edge_index=edge_index)

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
    """

    def __init__(self, curves, edge_index):
        self.curves = curves
        self.edge_index = edge_index

    def __repr__(self):
        return (
            f"TreeSample("
            f"n_curves={len(self.curves)}, "
            f"n_edges={self.edge_index.shape[1]})"
        )


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
        idxs = rng.choice(len(self.index), size=n, replace=False)
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
