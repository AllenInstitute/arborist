"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of NetworkX.Graph called "SkeletonGraph"
that represents a tree-structured graph with 3D spatial geometry.

"""

from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import StringIO
from scipy.spatial import KDTree
from scipy.spatial import distance
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import zipfile as zf

from arborist.utils.graph_loading import GraphLoader, count_nodes


class SkeletonGraph(nx.Graph):
    """
    A custom subclass of NetworkX tailored for graphs constructed from SWC
    files, where each connected component represents a single SWC file.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        node_spacing=1,
        verbose=False,
    ):
        """
        Instantiates a SkeletonGraph object.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors. Default is
            (1.0, 1.0, 1.0).
        node_spacing : float, optional
            Distance (in microns) between neighboring nodes. Default is 1μm.
        verbose : bool, optional
            Indication of whether to display progress bars. Default is False.
        """
        super().__init__()
        self.anisotropy = np.array(anisotropy)
        self.component_id_to_swc_id = dict()
        self.kdtree = None
        self.node_spacing = node_spacing
        self.verbose = verbose

    # --- Load ---
    def load(self, swc_pointer):
        """
        Loads SWC files into the graph.

        Parameters
        ----------
        swc_pointer : str
            Path to SWC files (local directory, ZIP, or cloud prefix).
        """
        irreducibles = GraphLoader(
            anisotropy=tuple(self.anisotropy),
            node_spacing=self.node_spacing,
            verbose=self.verbose,
        )(swc_pointer)

        num_nodes = count_nodes(irreducibles)
        self.node_component_id = np.zeros((num_nodes,), dtype=int)
        self.node_radius = np.zeros((num_nodes,), dtype=np.float16)
        self.node_xyz = np.zeros((num_nodes, 3), dtype=np.float32)

        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

        self.check_swc_ids()
        self.set_kdtree()

    # --- Build Graph ---
    def add_connected_component(self, irreducibles, component_id):
        """
        Adds a new connected component to the graph.

        Parameters
        ----------
        irreducibles : dict
            Dictionary with the following required fields:
                - "swc_id": SWC ID of the component.
                - "nodes": dictionary of node attributes.
                - "edges": dictionary of edge attributes.
        component_id : int
            Unique identifier for the connected component being added.
        """
        self.component_id_to_swc_id[component_id] = irreducibles["swc_id"]
        node_id_mapping = self._add_nodes(irreducibles["nodes"], component_id)
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs, component_id)

    def _add_nodes(self, node_dict, component_id):
        """
        Adds nodes to the graph from a dictionary of node attributes.

        Parameters
        ----------
        node_dict : dict
            Dictionary mapping original node IDs to their attributes. Each
            value must be a dictionary containing the keys "radius" and "xyz".
        component_id : str
            Connected component ID.

        Returns
        -------
        node_id_mapping : Dict[int, int]
            Mapping from original node IDs to new graph node IDs.
        """
        node_id_mapping = dict()
        for node_id, attrs in node_dict.items():
            new_id = self.number_of_nodes()
            self.node_component_id[new_id] = component_id
            self.node_radius[new_id] = attrs["radius"]
            self.node_xyz[new_id] = attrs["xyz"]
            self.add_node(new_id)
            node_id_mapping[node_id] = new_id
        return node_id_mapping

    def _add_edge(self, edge_id, attrs, component_id):
        """
        Adds an edge to the graph.

        Parameters
        ----------
        edge_id : Tuple[int]
            Edge to be added.
        attrs : dict
            Edge attribute dictionary from an SWC file.
        component_id : int
            Connected component ID.
        """
        i, j = edge_id
        dist_i = distance.euclidean(self.node_xyz[i], attrs["xyz"][0])
        dist_j = distance.euclidean(self.node_xyz[j], attrs["xyz"][0])
        start, end = (i, j) if dist_i < dist_j else (j, i)

        iterator = zip(attrs["radius"], attrs["xyz"])
        for cnt, (radius, xyz) in enumerate(iterator):
            if cnt > 0 and cnt < len(attrs["xyz"]) - 1:
                new_id = self.number_of_nodes()
                if cnt == 1:
                    self.add_edge(start, new_id)
                else:
                    self.add_edge(new_id, new_id - 1)
                self.node_component_id[new_id] = component_id
                self.node_radius[new_id] = radius
                self.node_xyz[new_id] = xyz
        self.add_edge(new_id, end)

    # --- Update Structure ---
    def check_swc_ids(self):
        visited = set()
        for nodes in map(list, nx.connected_components(self)):
            swc_id = self.node_swc_id(nodes[0])
            if swc_id not in visited:
                visited.add(swc_id)
                continue
            swc_ids = set(self.component_id_to_swc_id.values())
            while swc_id in visited and swc_id in swc_ids:
                swc_name, cnt = swc_id.split(".")
                swc_id = f"{swc_name}.{int(cnt) + 1}"
            component_id = self.node_component_id[nodes[0]]
            self.component_id_to_swc_id[component_id] = swc_id
            visited.add(swc_id)

    def reassign_component_ids(self):
        """
        Reassigns component IDs for all connected components in the graph.
        """
        self.check_swc_ids()
        component_id_to_swc_id = dict()
        for i, nodes in enumerate(nx.connected_components(self), start=1):
            nodes = np.array(list(nodes), dtype=int)
            component_id_to_swc_id[i] = self.node_swc_id(nodes[0])
            self.node_component_id[nodes] = i
        self.component_id_to_swc_id = component_id_to_swc_id

    def relabel_nodes(self):
        """
        Reassigns contiguous node IDs and updates all dependent structures.
        """
        old_node_ids = np.array(self.nodes, dtype=int)
        new_node_ids = np.arange(len(old_node_ids))
        old_to_new = dict(zip(old_node_ids, new_node_ids))
        old_edge_ids = list(self.edges)

        self.clear()
        self.add_nodes_from(new_node_ids)
        for i, j in old_edge_ids:
            self.add_edge(old_to_new[i], old_to_new[j])

        self.node_radius = self.node_radius[old_node_ids]
        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]

        self.reassign_component_ids()
        self.set_kdtree()
        assert len(self.node_xyz) == self.number_of_nodes()
        return old_to_new

    def remove_nodes(self, nodes, relabel_nodes=True):
        """
        Removes nodes from the graph.

        Parameters
        ----------
        nodes : container
            Node IDs to remove.
        relabel_nodes : bool, optional
            Indication of whether to relabel nodes after removal. Default is
            True.
        """
        self.remove_nodes_from(nodes)
        if relabel_nodes:
            self.relabel_nodes()

    def remove_small_components(self, min_size=20, relabel_nodes=True):
        rm_nodes = list()
        for nodes in map(list, nx.connected_components(self)):
            length = self.cable_length(max_depth=min_size, root=nodes[0])
            if length < min_size:
                rm_nodes.extend(nodes)
        self.remove_nodes_from(rm_nodes)
        if relabel_nodes:
            self.relabel_nodes()

    def remove_nearby_nodes(self, roots, max_dist=5.0):
        """
        Removes nodes within a given radius from a set of root nodes.

        Parameters
        ----------
        roots : List[int]
            Root nodes.
        max_dist : float, optional
            Maximum distance within which nodes are removed. Default is 5.0.
        """
        nodes = set()
        while len(roots) > 0:
            root = roots.pop()
            queue = [(root, 0)]
            visited = {root}
            while len(queue) > 0:
                i, dist_i = queue.pop()
                visited.add(i)
                for j in self.neighbors(i):
                    dist_j = dist_i + self.dist(i, j)
                    if j not in visited and dist_j <= max_dist:
                        queue.append((j, dist_j))
                    elif j not in visited and self.degree[j] > 2:
                        queue.append((j, dist_i))
            nodes = nodes.union(visited)
        self.remove_nodes_from(nodes)

    # --- Writer ---
    def to_zipped_swcs(self, zip_path, use_radius=False):
        """
        Writes the graph to a ZIP archive of SWC files.

        Parameters
        ----------
        zip_path : str
            Path to ZIP archive.
        use_radius : bool, optional
            Indication of whether to use node radius or default 2μm. Default
            is False.
        """
        self.check_swc_ids()
        with zf.ZipFile(zip_path, "w") as zipfile:
            for nodes in map(list, nx.connected_components(self)):
                self.component_to_zipped_swc(zipfile, nodes[0], use_radius)

    def to_zipped_swcs_multithreaded(self, output_dir, use_radius=True):
        """
        Writes the graph to ZIP archives using multithreading.

        Parameters
        ----------
        output_dir : str
            Path to output directory.
        use_radius : bool, optional
            Indication of whether to use node radius or default 2μm. Default
            is True.
        """
        def create_job(batch, i):
            zip_path = os.path.join(output_dir, f"{i}.zip")
            thread = executor.submit(
                self._batch_to_zipped_swcs, batch, zip_path, use_radius
            )
            return thread

        n = nx.number_connected_components(self)
        batch_size = max(1, n / 1000) if n > 10**4 else n
        os.makedirs(output_dir, exist_ok=True)

        self.check_swc_ids()
        with ThreadPoolExecutor() as executor:
            batch, threads, zip_cnt = list(), list(), 0
            for i, nodes in enumerate(nx.connected_components(self)):
                batch.append(nodes)
                if len(batch) >= batch_size:
                    threads.append(create_job(list(batch), zip_cnt))
                    batch = list()
                    zip_cnt += 1
            if batch:
                threads.append(create_job(list(batch), zip_cnt + 1))

            pbar = tqdm(total=len(threads), desc="Write SWCs")
            for thread in as_completed(threads):
                thread.result()
                pbar.update(1)

    def _batch_to_zipped_swcs(self, nodes_list, zip_path, use_radius):
        with zf.ZipFile(zip_path, "w") as zipfile:
            for nodes in map(list, nodes_list):
                self.component_to_zipped_swc(zipfile, nodes[0], use_radius)

    def component_to_zipped_swc(self, zipfile, root, use_radius=False):
        """
        Writes the connected component containing the given root node to a
        zipped SWC file.

        Parameters
        ----------
        zipfile : zipfile.ZipFile
            ZipFile object that will store the generated SWC file.
        root : int
            Root node of connected component.
        use_radius : bool, optional
            Indication of whether to preserve radii or use default 2μm.
            Default is False.
        """
        def write_entry(node, parent):
            x, y, z = self.node_xyz[node]
            r = self.node_radius[node] if use_radius else 2
            node_to_idx[node] = cnt
            text_buffer.write(
                f"\n{cnt} 2 {x} {y} {z} {r} {node_to_idx[parent]}"
            )

        with StringIO() as text_buffer:
            text_buffer.write("# id, type, z, y, x, r, pid")
            cnt = 1
            node_to_idx = defaultdict(lambda: -1)
            write_entry(root, -1)
            for i, j in nx.dfs_edges(self, source=root):
                cnt += 1
                write_entry(j, i)

            # Deduplicate filename within zip
            swc_id = self.node_swc_id(root)
            name = f"{swc_id}.swc"
            suffix = 0
            while name in zipfile.namelist():
                name = f"{swc_id}.{suffix}.swc"
                suffix += 1
            zipfile.writestr(name, text_buffer.getvalue())

    # --- Helpers ---
    def branching_nodes(self):
        """
        Returns all branching nodes (degree > 2).
        """
        return [i for i in self.nodes if self.degree[i] > 2]

    def cable_length(self, max_depth=np.inf, root=None):
        """
        Computes cable length of the graph or a connected component.

        Parameters
        ----------
        max_depth : float, optional
            Maximum depth before stopping. Default is np.inf.
        root : int, optional
            Node in the component to measure. Default is None.

        Returns
        -------
        float
            Cable length.
        """
        cable_length = 0
        for i, j in nx.dfs_edges(self, source=root):
            cable_length += self.dist(i, j)
            if cable_length > max_depth:
                break
        return cable_length

    def clip_to_skeleton(self, gt_graph, dist):
        """
        Removes nodes more than "dist" microns from "gt_graph".

        Parameters
        ----------
        gt_graph : SkeletonGraph
            Ground truth graph used as clipping reference.
        dist : float
            Distance threshold in microns.
        """
        d_gt, _ = gt_graph.kdtree.query(self.node_xyz)
        nodes = np.where(d_gt > dist)[0]
        self.remove_nodes_from(nodes)
        for nodes in list(nx.connected_components(self)):
            if len(nodes) < 30:
                self.remove_nodes_from(nodes)
        self.relabel_nodes()

    def closest_node(self, xyz):
        """
        Finds the closest node to the given xyz coordinate.

        Parameters
        ----------
        xyz : ArrayLike
            Coordinate to query.

        Returns
        -------
        int
            Closest node.
        """
        assert self.kdtree, "KD-Tree attribute has not been set!"
        _, node = self.kdtree.query(xyz)
        return node

    def component_id_from_swc_id(self, query_swc_id):
        """
        Gets the component ID for the given SWC ID.
        """
        for component_id, swc_id in self.component_id_to_swc_id.items():
            if query_swc_id == swc_id:
                return component_id
        raise ValueError(f"SWC ID={query_swc_id} not found")

    def connected_nodes(self, root):
        """
        Gets all nodes connected to the given root node.
        """
        queue = [root]
        visited = set({root})
        while queue:
            i = queue.pop()
            for j in self.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)
        return visited

    def directed_path(self, start_node, next_node, max_depth=np.inf):
        queue = [(next_node, 0)]
        visited = [start_node, next_node]
        while queue:
            i, dist_i = queue.pop()
            if self.degree[i] != 2:
                return visited
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_depth and j not in visited:
                    queue.append((j, dist_j))
                    visited.append(j)
        return visited

    def dist(self, i, j):
        """
        Computes Euclidean distance between nodes i and j.
        """
        return distance.euclidean(self.node_xyz[i], self.node_xyz[j])

    def find_connecting_path(self, nodes):
        """
        Finds the path connecting the given set of nodes.
        """
        connecting_nodes = set()
        for target in nodes[1:]:
            path = nx.shortest_path(self, source=nodes[0], target=target)
            connecting_nodes = connecting_nodes.union(set(path))
        return connecting_nodes

    def find_nearby_branching_node(self, root, max_depth=16):
        queue = [(root, 0)]
        visited = {root}
        while queue:
            i, d_i = queue.pop(0)
            if self.degree[i] >= 3:
                return i
            for j in self.neighbors(i):
                d_j = d_i + self.dist(i, j)
                if j not in visited and d_j < max_depth:
                    queue.append((j, d_j))
                    visited.add(j)
        return root

    def get_irreducible_edge(self, node):
        """
        Finds the irreducible edge containing the given node.
        """
        assert self.degree[node] < 3
        edge = list()
        queue = [node]
        visited = set(queue)
        while queue:
            i = queue.pop()
            if self.degree[i] != 2:
                edge.append(i)
                continue
            for j in self.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)
        assert len(edge) == 2
        return edge

    def irreducible_nodes(self):
        """
        Returns the set of irreducible nodes (degree != 2).
        """
        return {i for i in map(int, self.nodes) if self.degree[i] != 2}

    def irreducible_paths(self):
        """
        Extracts non-branching paths between irreducible nodes.

        Returns
        -------
        List[numpy.ndarray]
            Each entry is an ordered array of node IDs forming a path between
            two irreducible nodes, inclusive of both endpoints.
        """
        irreducible = {n for n in self.nodes if self.degree(n) != 2}
        paths = []
        visited_edges = set()
        for source in irreducible:
            for nb in self.neighbors(source):
                edge = frozenset((source, nb))
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                path = [source, nb]
                prev, curr = source, nb
                while curr not in irreducible:
                    nxt = next(n for n in self.neighbors(curr) if n != prev)
                    edge = frozenset((curr, nxt))
                    visited_edges.add(edge)
                    path.append(nxt)
                    prev, curr = curr, nxt
                paths.append(np.array(path, dtype=int))
        return paths

    def leaf_nodes(self):
        """
        Returns all leaf nodes (degree == 1).
        """
        return [i for i in self.nodes if self.degree[i] == 1]

    def midpoint(self, i, j):
        return np.mean([self.node_xyz[i], self.node_xyz[j]], axis=0)

    def node_segment_id(self, node):
        return self.node_swc_id(node).split(".")[0]

    def node_swc_id(self, i):
        component_id = self.node_component_id[i]
        swc_id = self.component_id_to_swc_id[component_id]
        return swc_id if "." in swc_id else f"{swc_id}.0"

    def nodes_with_component_id(self, component_id):
        return set(np.where(self.node_component_id == component_id)[0])

    def nodes_with_segment_id(self, query_segment_id):
        nodes = set()
        query_id = str(query_segment_id)
        for swc_id in self.swc_ids():
            segment_id = swc_id.split(".")[0]
            if segment_id == query_id:
                component_id = self.component_id_from_swc_id(swc_id)
                nodes = nodes.union(self.nodes_with_component_id(component_id))
        return nodes

    def nodes_within_distance(self, root, max_dist):
        """
        Gets nodes connected to root up to max_dist microns.
        """
        queue = [(root, 0)]
        visited = {root}
        while queue:
            i, dist_i = queue.pop()
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_dist and j not in visited:
                    queue.append((j, dist_j))
                    visited.add(j)
        return list(visited)

    def path_from_leaf(self, leaf, max_depth=np.inf):
        """
        Gets the path emanating from a leaf up to max_depth microns.
        """
        queue = [(leaf, 0)]
        path = [leaf]
        while queue:
            i, dist_i = queue.pop()
            if self.degree[i] != 2 and dist_i > 0:
                return path
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_depth and j not in path:
                    queue.append((j, dist_j))
                    path.append(j)
        return path

    def path_length(self, path):
        """
        Computes the Euclidean length of a node path.

        Parameters
        ----------
        path : List[int]
            List of node IDs forming a path.

        Returns
        -------
        float
        """
        if len(path) > 1:
            return np.linalg.norm(
                np.diff(self.node_xyz[path], axis=0), axis=1
            ).sum()
        return 0

    def path_thru_node(self, i, max_depth=np.inf):
        if self.degree[i] == 0:
            return [i]
        elif self.degree[i] == 1:
            return self.path_from_leaf(i, max_depth)
        else:
            assert self.degree[i] == 2
            j, k = self.neighbors(i)
            path_ij = self.directed_path(i, j, max_depth=max_depth)
            path_ik = self.directed_path(i, k, max_depth=max_depth)
            return path_ij[::-1] + path_ik[1:]

    def rooted_subgraph(self, root, radius):
        """
        Gets a rooted subgraph within radius microns.

        Parameters
        ----------
        root : int
            Root node ID.
        radius : float
            Depth in microns.

        Returns
        -------
        SkeletonGraph
        """
        subgraph = self.__class__(anisotropy=self.anisotropy)
        subgraph.add_node(0)
        idxs = [root]
        node_mapping = {root: 0}
        queue = [(root, 0)]
        visited = {root}
        while queue:
            i, dist_i = queue.pop()
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if j not in visited and dist_j < radius:
                    node_mapping[j] = subgraph.number_of_nodes()
                    subgraph.add_edge(node_mapping[i], node_mapping[j])
                    queue.append((j, dist_j))
                    visited.add(j)
                    idxs.append(j)
        idxs = np.array(idxs, dtype=int)
        subgraph.node_radius = self.node_radius[idxs]
        subgraph.node_xyz = self.node_xyz[idxs]
        return subgraph

    def set_kdtree(self):
        """
        Initializes KD-Tree from node xyz coordinates.
        """
        self.kdtree = KDTree(self.node_xyz)

    def swc_ids(self):
        """
        Returns the set of all unique SWC IDs in the graph.
        """
        return set(self.component_id_to_swc_id.values())

    def __repr__(self):
        n_components = format(nx.number_connected_components(self), ",")
        n_nodes = format(self.number_of_nodes(), ",")
        n_edges = format(self.number_of_edges(), ",")
        return (
            f"   SkeletonGraph(\n"
            f"      num_connected_components={n_components},\n"
            f"      num_nodes={n_nodes},\n"
            f"      num_edges={n_edges},\n"
            f"   )"
        )
