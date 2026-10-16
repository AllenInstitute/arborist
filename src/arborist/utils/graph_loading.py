"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Loads SWC files and builds the irreducible graph representations used by
SkeletonGraph.

"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import networkx as nx
import numpy as np

from arborist.utils import geometry
from arborist.utils import swc_loading as swc_util

SOMA_TYPE = 1


class GraphLoader:
    """
    Reads SWC files in parallel and extracts irreducible graph components
    ready to be loaded into a SkeletonGraph.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_cable_length=40.0,
        node_spacing=1,
        prefetch=128,
        prune_depth=24.0,
        verbose=False,
    ):
        self.min_cable_length = min_cable_length
        self.node_spacing = node_spacing
        self.prefetch = prefetch
        self.prune_depth = prune_depth
        self.swc_reader = swc_util.Reader(anisotropy, verbose)
        self.verbose = verbose

    def __call__(self, swc_pointer):
        """
        Processes SWC files in parallel and returns a deque of irreducible
        component dicts.

        Parameters
        ----------
        swc_pointer : str
            Path to SWC files (local directory, ZIP, or cloud prefix).

        Returns
        -------
        deque[dict]
            Each dict has keys ``"swc_id"``, ``"nodes"``, ``"edges"``,
            ``"is_soma"``.
        """
        swc_dicts = self.swc_reader(swc_pointer)

        # Most SWCs are far below min_cable_length; the raw path length is an
        # upper bound on the cable length that "load" computes (pruning only
        # removes edges), so they can be dropped before being dispatched.
        swc_dicts = [
            d for d in swc_dicts
            if self.raw_cable_length(d) >= self.min_cable_length
        ]

        # One task per SWC spends more on pickling and IPC than on the work
        # itself (~1 ms per SWC), so SWCs are dispatched in chunks.
        irreducibles = deque()
        with ProcessPoolExecutor() as executor:
            chunksize = max(1, min(256, len(swc_dicts) // (8 * executor._max_workers)))
            results = executor.map(self.load, swc_dicts, chunksize=chunksize)
            if self.verbose:
                results = tqdm(results, total=len(swc_dicts), desc="Load Graphs")
            for result in results:
                if result:
                    irreducibles.append(result)
        return irreducibles

    def raw_cable_length(self, swc_dict):
        """
        Sums the edge lengths of an SWC file as written, without building a
        graph. This bounds the cable length of its irreducible components
        from above.

        Parameters
        ----------
        swc_dict : dict
            Parsed contents of one SWC file.

        Returns
        -------
        float
        """
        ids, pids, xyz = swc_dict["id"], swc_dict["pid"], swc_dict["xyz"]
        order = np.argsort(ids)
        parent = order[np.clip(np.searchsorted(ids[order], pids), 0, len(ids) - 1)]
        has_parent = ids[parent] == pids
        edges = xyz[has_parent] - xyz[parent[has_parent]]
        return float(np.linalg.norm(edges, axis=1).sum())

    def load(self, swc_dict):
        """
        Extracts irreducible components from a single SWC dictionary.

        Parameters
        ----------
        swc_dict : dict
            Parsed contents of one SWC file.

        Returns
        -------
        dict or None
        """
        graph = swc_util.to_graph(swc_dict)
        prune_branches(graph, self.prune_depth)
        irreducibles = self.get_irreducibles(graph)
        if irreducibles:
            irreducibles["is_soma"] = len(swc_dict["soma_nodes"]) > 0
            irreducibles["swc_id"] = swc_dict["swc_name"]
            return irreducibles
        return None

    def get_irreducibles(self, graph):
        """
        Identifies and returns the irreducible structure of a connected graph.

        Parameters
        ----------
        graph : networkx.Graph
            Dense per-SWC graph as returned by ``swc_util.to_graph``.

        Returns
        -------
        dict or None
        """
        leaf = find_leaf(graph)
        irr_nodes = {leaf}
        irr_edges = dict()

        radius = graph.graph["radius"]
        node_type = graph.graph["type"]
        xyz = graph.graph["xyz"]

        root, cable_length = None, 0
        for i, j in nx.dfs_edges(graph, source=leaf):
            if root is None:
                root, edge_length = i, 0
                attrs = {
                    "radius": [radius[i]],
                    "type": [node_type[i]],
                    "xyz": [xyz[i]],
                }

            edge_length += np.linalg.norm(xyz[i] - xyz[j])
            attrs["radius"].append(radius[j])
            attrs["type"].append(node_type[j])
            attrs["xyz"].append(xyz[j])

            if graph.degree[j] != 2:
                cable_length += edge_length
                irr_nodes.add(j)
                attrs = _to_numpy(attrs)
                n_pts = int(edge_length / self.node_spacing)
                self._resample_edge(graph, attrs, (root, j), n_pts)
                irr_edges[(root, j)] = attrs
                root = None

        # Reject curvy line fragments where endpoints are too close
        if len(irr_nodes) == 2:
            t0, t1 = irr_nodes
            endpoint_dist = np.linalg.norm(xyz[t0] - xyz[t1])
            if endpoint_dist / cable_length < 0.5:
                return None

        if cable_length >= self.min_cable_length:
            return {
                "nodes": _set_node_attrs(graph, irr_nodes),
                "edges": _set_edge_attrs(graph, irr_edges),
            }
        return None

    def _resample_edge(self, graph, attrs, edge, n_pts):
        attrs["xyz"] = geometry.resample_curve_3d(attrs["xyz"], n_pts=n_pts)
        attrs["radius"] = geometry.resample_curve_1d(attrs["radius"], n_pts)
        attrs["type"] = _resample_labels(attrs["type"], len(attrs["xyz"]))
        graph.graph["xyz"][edge[0]] = attrs["xyz"][0]
        graph.graph["xyz"][edge[1]] = attrs["xyz"][-1]


# --- Helpers ---

def count_nodes(irr_list):
    """Counts the total number of dense nodes across all irreducible dicts."""
    n = 0
    for irr in irr_list:
        n += len(irr["nodes"])
        for attrs in irr["edges"].values():
            n += len(attrs["xyz"]) - 2
    return n


def find_leaf(graph):
    """Returns a leaf node (degree == 1) from the graph, or None."""
    for i in graph.nodes:
        if graph.degree[i] == 1:
            return i
    return None


def prune_branches(graph, depth):
    """
    Removes branches shorter than ``depth`` microns from leaf to the nearest
    branching node.
    """
    xyz = graph.graph["xyz"]
    changed = True
    while changed:
        changed = False
        for leaf in [i for i in graph.nodes if graph.degree[i] == 1]:
            branch, length = [leaf], 0
            for i, j in nx.dfs_edges(graph, source=leaf):
                length += np.linalg.norm(xyz[i] - xyz[j])
                if length > depth:
                    break
                if graph.degree(j) == 2:
                    branch.append(j)
                elif graph.degree(j) > 2:
                    graph.remove_nodes_from(branch)
                    changed = True
                    break


def _resample_labels(labels, n_pts):
    """
    Resamples a categorical label array to "n_pts" entries.

    Compartment codes are categorical, so they are resampled by nearest
    neighbor rather than spline-interpolated like radius and xyz, which would
    invent codes that never appeared in the SWC file. The one exception is
    the soma: a resampled point lying between a soma node and a neurite node
    is on the neurite, so it takes the neurite's code. Nearest neighbor alone
    labels the first half of every soma-to-neurite edge as soma, which spreads
    a single-node soma over tens of microns. The edge's endpoints keep their
    own codes, and points between two soma nodes stay soma.
    """
    t = np.linspace(0, len(labels) - 1, n_pts)
    out = labels[np.rint(t).astype(int)]
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(labels) - 1)
    interior = np.zeros(n_pts, dtype=bool)
    interior[1:-1] = True
    fix = interior & (out == SOMA_TYPE) & (labels[lo] != labels[hi])
    out[fix] = np.where(labels[lo[fix]] == SOMA_TYPE, labels[hi[fix]], labels[lo[fix]])
    return out


def _set_node_attrs(graph, nodes):
    xyz, radius = graph.graph["xyz"], graph.graph["radius"]
    node_type = graph.graph["type"]
    return {
        i: {"radius": radius[i], "type": node_type[i], "xyz": xyz[i]}
        for i in nodes
    }


def _set_edge_attrs(graph, attrs):
    for e in attrs:
        i, j = e
        attrs[e]["xyz"][0] = graph.graph["xyz"][i]
        attrs[e]["xyz"][-1] = graph.graph["xyz"][j]
    return attrs


def _to_numpy(attrs):
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    attrs["type"] = np.array(attrs["type"], dtype=np.int8)
    return attrs
