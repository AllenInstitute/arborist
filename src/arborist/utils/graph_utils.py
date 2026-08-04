"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General-purpose graph utilities for tree-structured graphs.

"""

import networkx as nx


def cycle_exists(graph):
    """Returns True if the graph contains a cycle."""
    try:
        nx.find_cycle(graph)
        return True
    except nx.exception.NetworkXNoCycle:
        return False


def edges_to_line_graph(edges):
    """Constructs a NetworkX line graph from a list of edges."""
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return nx.line_graph(graph)


def topological_decomposition(subgraph, root=0):
    """
    Decomposes a rooted subgraph into paths between topological nodes
    (root, branching nodes, and leaves).

    Parameters
    ----------
    subgraph : SkeletonGraph
        Rooted subgraph as returned by ``SkeletonGraph.rooted_subgraph``.
    root : int, optional
        Node ID of the subgraph root. Default is 0.

    Returns
    -------
    topo_nodes : List[int]
        Subgraph node IDs that are topological points (root first).
    paths : List[List[int]]
        Node ID chains (inclusive of both endpoints), root-to-leaf order,
        parallel to ``edge_index``.
    edge_index : List[Tuple[int, int]]
        Edges of the reduced topological graph as index pairs into
        ``topo_nodes``.
    """
    def is_topo(i):
        deg = subgraph.degree[i]
        return i == root or deg == 1 or deg >= 3

    topo_nodes = [i for i in range(subgraph.number_of_nodes()) if is_topo(i)]
    topo_idx = {n: k for k, n in enumerate(topo_nodes)}

    paths, edge_index = [], []
    visited = {root}
    stack = [(root, root, [root])]

    while stack:
        parent, curr, path = stack.pop()
        for nb in subgraph.neighbors(curr):
            if nb == parent or nb in visited:
                continue
            visited.add(nb)
            if is_topo(nb):
                paths.append(path + [nb])
                edge_index.append((topo_idx[path[0]], topo_idx[nb]))
                stack.append((curr, nb, [nb]))
            else:
                stack.append((curr, nb, path + [nb]))

    return topo_nodes, paths, edge_index
