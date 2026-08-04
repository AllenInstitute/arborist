"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Loads SWC files and builds the irreducible graph representations used by
SkeletonGraph.

"""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from tqdm import tqdm

import networkx as nx
import numpy as np

from arborist.utils import geometry
from arborist.utils import swc_loading as swc_util


class GraphLoader:
    """
    Reads SWC files in parallel and extracts irreducible graph components
    ready to be loaded into a SkeletonGraph.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_cable_length=40.0,
        min_swc_pts=1,
        node_spacing=1,
        prefetch=128,
        prune_depth=24.0,
        verbose=False,
    ):
        self.min_cable_length = min_cable_length
        self.min_swc_pts = min_swc_pts
        self.node_spacing = node_spacing
        self.prefetch = prefetch
        self.prune_depth = prune_depth
        self.swc_reader = swc_util.Reader(anisotropy, min_swc_pts, verbose)
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
        swc_dicts = deque(
            d for d in swc_dicts if len(d["xyz"]) > self.min_swc_pts
        )
        if self.verbose:
            pbar = tqdm(total=len(swc_dicts), desc="Load Graphs")

        pending = set()
        irreducibles = deque()
        with ProcessPoolExecutor() as executor:
            while len(pending) < self.prefetch and swc_dicts:
                pending.add(executor.submit(self.load, swc_dicts.pop()))

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result:
                        irreducibles.append(result)
                    if self.verbose:
                        pbar.update(1)
                    if swc_dicts:
                        pending.add(
                            executor.submit(self.load, swc_dicts.pop())
                        )
        return irreducibles

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
        xyz = graph.graph["xyz"]

        root, cable_length = None, 0
        for i, j in nx.dfs_edges(graph, source=leaf):
            if root is None:
                root, edge_length = i, 0
                attrs = {"radius": [radius[i]], "xyz": [xyz[i]]}

            edge_length += np.linalg.norm(xyz[i] - xyz[j])
            attrs["radius"].append(radius[j])
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


def _set_node_attrs(graph, nodes):
    xyz, radius = graph.graph["xyz"], graph.graph["radius"]
    return {i: {"radius": radius[i], "xyz": xyz[i]} for i in nodes}


def _set_edge_attrs(graph, attrs):
    for e in attrs:
        i, j = e
        attrs[e]["xyz"][0] = graph.graph["xyz"][i]
        attrs[e]["xyz"][-1] = graph.graph["xyz"][j]
    return attrs


def _to_numpy(attrs):
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    return attrs
