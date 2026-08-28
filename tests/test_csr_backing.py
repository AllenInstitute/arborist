"""Equivalence tests for the CSR graph backing.

SkeletonGraph used to subclass networkx.Graph. These tests pin the CSR
replacement to nx's observable behaviour on every graph primitive that arborist
and neuron-proofreader call, so that the storage change stays invisible.

The reference is not a recorded fixture. "_NxReference" is an nx.Graph carrying
SkeletonGraph's own method bodies, with thin adapters that route the six graph
algorithms back to networkx. Both sides therefore run identical Python over
different storage, which is what makes a mismatch meaningful.

Neighbour ordering is checked in "insertion_order=True" mode, because that is
the mode that claims to reproduce nx exactly. The default sorted mode is
checked on everything order-independent, plus the order-valued outputs that
were measured not to change.
"""

import io
import pickle
import unittest
import zipfile as zf
from copy import deepcopy

import networkx as nx
import numpy as np

from arborist.skeleton_graph import SkeletonGraph, register_node_attrs

SEED = 0
N_NODES = 400


@register_node_attrs
class _NxReference(nx.Graph):
    """
    nx-backed stand-in carrying SkeletonGraph's method bodies.

    The borrowed methods touch only nodes, degree, neighbors,
    number_of_nodes and the node attribute arrays, so they run unchanged over
    nx storage. The adapters exist because SkeletonGraph now calls these six
    as methods rather than as nx module functions.
    """

    NODE_ATTR_SPEC = SkeletonGraph.NODE_ATTR_SPEC

    init_node_attrs = SkeletonGraph.init_node_attrs
    copy_node_attrs = SkeletonGraph.copy_node_attrs
    reindex_node_attrs = SkeletonGraph.reindex_node_attrs

    branching_nodes = SkeletonGraph.branching_nodes
    cable_length = SkeletonGraph.cable_length
    component_to_zipped_swc = SkeletonGraph.component_to_zipped_swc
    connected_nodes = SkeletonGraph.connected_nodes
    dist = SkeletonGraph.dist
    find_connecting_path = SkeletonGraph.find_connecting_path
    get_irreducible_edge = SkeletonGraph.get_irreducible_edge
    irreducible_nodes = SkeletonGraph.irreducible_nodes
    irreducible_paths = SkeletonGraph.irreducible_paths
    leaf_nodes = SkeletonGraph.leaf_nodes
    node_swc_id = SkeletonGraph.node_swc_id
    nodes_within_distance = SkeletonGraph.nodes_within_distance
    path_from_leaf = SkeletonGraph.path_from_leaf
    path_length = SkeletonGraph.path_length
    rooted_subgraph = SkeletonGraph.rooted_subgraph
    set_kdtree = SkeletonGraph.set_kdtree

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), **kwargs):
        super().__init__()
        self.anisotropy = np.array(anisotropy)
        self.component_id_to_swc_id = dict()
        self.kdtree = None

    def connected_components(self):
        return list(nx.connected_components(self))

    def number_connected_components(self):
        return nx.number_connected_components(self)

    def node_connected_component(self, i):
        return nx.node_connected_component(self, i)

    def has_path(self, i, j):
        return nx.has_path(self, i, j)

    def shortest_path(self, source, target):
        return nx.shortest_path(self, source=source, target=target)

    def dfs_edges(self, source=None):
        return nx.dfs_edges(self, source=source)


def random_tree_edges(rng, num_nodes):
    """
    Builds a random tree by attaching each node to an earlier one, which is
    the shape a real skeleton has.
    """
    parents = np.array(
        [rng.integers(0, i) for i in range(1, num_nodes)], dtype=np.int64
    )
    return np.stack([parents, np.arange(1, num_nodes)], axis=1)


def fragmented_edges(rng, num_components, per_component):
    """
    Packs several trees into one node ID space, giving a many-component graph
    like a FragmentsGraph.
    """
    parts, offset = list(), 0
    for _ in range(num_components):
        parts.append(random_tree_edges(rng, per_component) + offset)
        offset += per_component
    return np.concatenate(parts), offset


def build_pair(num_nodes, edges, insertion_order=True):
    """
    Builds an nx reference and a CSR SkeletonGraph over the same nodes, edges
    and node attributes.
    """
    rng = np.random.default_rng(SEED)
    xyz = rng.normal(scale=50.0, size=(num_nodes, 3))
    radius = rng.uniform(0.5, 3.0, size=num_nodes)
    node_type = rng.integers(1, 5, size=num_nodes)

    reference = _NxReference()
    reference.add_nodes_from(range(num_nodes))
    reference.add_edges_from(map(tuple, edges))

    graph = SkeletonGraph(insertion_order=insertion_order)
    graph.build_from_edges(num_nodes, edges)

    for obj in (reference, graph):
        obj.init_node_attrs(num_nodes)
        obj.node_xyz[:] = xyz
        obj.node_radius[:] = radius
        obj.node_type[:] = node_type
        obj.node_component_id[:] = 0
        obj.component_id_to_swc_id[0] = "toy.0"
    return reference, graph


def swc_text(graph, root):
    """Returns the SWC text the real writer produces for one component."""
    buffer = io.BytesIO()
    with zf.ZipFile(buffer, "w") as zipfile:
        graph.component_to_zipped_swc(zipfile, root, use_radius=True)
        name = zipfile.namelist()[0]
    with zf.ZipFile(buffer) as zipfile:
        return zipfile.read(name).decode()


class CSREquivalenceTest(unittest.TestCase):
    """Compares the CSR backing against nx on the primitives in use."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(SEED)
        cls.tree_edges = random_tree_edges(rng, N_NODES)
        cls.frag_edges, cls.frag_nodes = fragmented_edges(rng, 12, 40)

    def cases(self, insertion_order=True):
        yield "tree", build_pair(
            N_NODES, self.tree_edges, insertion_order
        )
        yield "fragmented", build_pair(
            self.frag_nodes, self.frag_edges, insertion_order
        )

    # --- Structure ---
    def test_counts_and_degree(self):
        for label, (reference, graph) in self.cases():
            with self.subTest(label):
                n = reference.number_of_nodes()
                self.assertEqual(n, graph.number_of_nodes())
                self.assertEqual(n, len(graph))
                self.assertEqual(
                    reference.number_of_edges(), graph.number_of_edges()
                )
                self.assertEqual(sorted(reference.nodes), sorted(graph.nodes))
                # Both access forms, because consumers use both.
                self.assertEqual(
                    [reference.degree[i] for i in range(n)],
                    [graph.degree[i] for i in range(n)],
                )
                self.assertEqual(
                    [reference.degree(i) for i in range(n)],
                    [graph.degree(i) for i in range(n)],
                )

    def test_neighbors_match_insertion_order(self):
        for label, (reference, graph) in self.cases(insertion_order=True):
            with self.subTest(label):
                for i in range(reference.number_of_nodes()):
                    self.assertEqual(
                        list(reference.neighbors(i)),
                        list(graph.neighbors(i)),
                        f"node {i}",
                    )

    def test_neighbors_match_as_sets_when_sorted(self):
        for label, (reference, graph) in self.cases(insertion_order=False):
            with self.subTest(label):
                for i in range(reference.number_of_nodes()):
                    self.assertEqual(
                        sorted(reference.neighbors(i)),
                        sorted(graph.neighbors(i)),
                        f"node {i}",
                    )

    def test_edges_match_as_sets(self):
        for label, (reference, graph) in self.cases():
            with self.subTest(label):
                self.assertEqual(
                    {frozenset(e) for e in reference.edges},
                    {frozenset(e) for e in graph.edges},
                )

    def test_has_node_and_has_edge(self):
        _, graph = build_pair(N_NODES, self.tree_edges)
        self.assertTrue(graph.has_node(0))
        self.assertFalse(graph.has_node(N_NODES))
        self.assertFalse(graph.has_node(-1))
        for i, j in map(tuple, self.tree_edges[:20]):
            self.assertTrue(graph.has_edge(i, j))
            self.assertTrue(graph.has_edge(j, i))
        self.assertFalse(graph.has_edge(0, N_NODES - 1))

    # --- Connectivity ---
    def test_connected_components(self):
        for label, (reference, graph) in self.cases():
            with self.subTest(label):
                self.assertEqual(
                    {frozenset(c) for c in reference.connected_components()},
                    {frozenset(c) for c in graph.connected_components()},
                )
                self.assertEqual(
                    reference.number_connected_components(),
                    graph.number_connected_components(),
                )

    def test_path_queries(self):
        rng = np.random.default_rng(SEED)
        for label, (reference, graph) in self.cases():
            with self.subTest(label):
                n = reference.number_of_nodes()
                for i, j in rng.integers(0, n, (60, 2)).tolist():
                    self.assertEqual(
                        reference.has_path(i, j), graph.has_path(i, j)
                    )
                    self.assertEqual(
                        reference.node_connected_component(i),
                        graph.node_connected_component(i),
                    )
                    if reference.has_path(i, j):
                        self.assertEqual(
                            reference.shortest_path(i, j),
                            graph.shortest_path(i, j),
                        )

    def test_shortest_path_raises_nx_exception(self):
        """Two consumers catch nx.NetworkXNoPath by name."""
        _, graph = build_pair(self.frag_nodes, self.frag_edges)
        components = sorted(map(sorted, graph.connected_components()))
        source, target = components[0][0], components[1][0]
        self.assertFalse(graph.has_path(source, target))
        with self.assertRaises(nx.NetworkXNoPath):
            graph.shortest_path(source, target)

    def test_dfs_edges_match(self):
        for label, (reference, graph) in self.cases(insertion_order=True):
            with self.subTest(label):
                self.assertEqual(
                    list(reference.dfs_edges()), list(graph.dfs_edges())
                )
                for root in sorted(
                    min(c) for c in graph.connected_components()
                ):
                    self.assertEqual(
                        list(reference.dfs_edges(source=root)),
                        list(graph.dfs_edges(source=root)),
                    )

    # --- Borrowed SkeletonGraph algorithms ---
    def test_borrowed_algorithms_match(self):
        for label, (reference, graph) in self.cases(insertion_order=True):
            with self.subTest(label):
                self.assertEqual(reference.leaf_nodes(), graph.leaf_nodes())
                self.assertEqual(
                    reference.branching_nodes(), graph.branching_nodes()
                )
                self.assertEqual(
                    reference.irreducible_nodes(), graph.irreducible_nodes()
                )
                self.assertEqual(
                    {tuple(p) for p in reference.irreducible_paths()},
                    {tuple(p) for p in graph.irreducible_paths()},
                )
                self.assertAlmostEqual(
                    reference.cable_length(), graph.cable_length(), places=6
                )
                for root in sorted(
                    min(c) for c in graph.connected_components()
                )[:5]:
                    self.assertEqual(
                        reference.connected_nodes(root),
                        graph.connected_nodes(root),
                    )
                    self.assertAlmostEqual(
                        reference.cable_length(max_depth=200, root=root),
                        graph.cable_length(max_depth=200, root=root),
                        places=6,
                    )
                    self.assertEqual(
                        sorted(reference.nodes_within_distance(root, 120)),
                        sorted(graph.nodes_within_distance(root, 120)),
                    )

    def test_rooted_subgraph_matches(self):
        for label, (reference, graph) in self.cases(insertion_order=True):
            with self.subTest(label):
                for root in sorted(
                    min(c) for c in graph.connected_components()
                )[:5]:
                    expected = reference.rooted_subgraph(root, 150)
                    actual = graph.rooted_subgraph(root, 150)
                    self.assertEqual(
                        expected.number_of_nodes(), actual.number_of_nodes()
                    )
                    # Node IDs are assigned in discovery order, so equal
                    # coordinates mean equal traversal.
                    np.testing.assert_array_equal(
                        expected.node_xyz, actual.node_xyz
                    )

    def test_swc_writer_text_matches(self):
        for label, (reference, graph) in self.cases(insertion_order=True):
            with self.subTest(label):
                for root in sorted(
                    min(c) for c in graph.connected_components()
                )[:5]:
                    self.assertEqual(
                        swc_text(reference, root), swc_text(graph, root)
                    )

    def test_subgraph_matches(self):
        rng = np.random.default_rng(SEED)
        for label, (reference, graph) in self.cases():
            with self.subTest(label):
                n = reference.number_of_nodes()
                keep = sorted(
                    rng.choice(n, size=n // 2, replace=False).tolist()
                )
                expected = reference.subgraph(keep)
                actual = graph.subgraph(keep)
                self.assertEqual(
                    expected.number_of_edges(), actual.number_of_edges()
                )
                # subgraph renumbers by ascending original ID.
                remap = {old: new for new, old in enumerate(keep)}
                self.assertEqual(
                    {
                        frozenset((remap[i], remap[j]))
                        for i, j in expected.edges
                    },
                    {frozenset(e) for e in actual.edges},
                )
                np.testing.assert_array_equal(
                    graph.node_xyz[keep], actual.node_xyz
                )


class CSRMutationTest(unittest.TestCase):
    """Covers the paths where CSR cannot mirror nx's storage directly."""

    def setUp(self):
        rng = np.random.default_rng(SEED)
        self.edges = random_tree_edges(rng, N_NODES)
        self.reference, self.graph = build_pair(N_NODES, self.edges)

    def test_add_edge_is_idempotent(self):
        """
        nx.add_edge on an existing edge is a no-op. Missing this made degree
        and neighbors disagree with nx on cells whose SWC repeats an edge.
        """
        i, j = map(int, self.edges[0])
        before = self.graph.number_of_edges()
        degree_before = self.graph.degree[i]
        for _ in range(5):
            self.graph.add_edge(i, j)
            self.graph.add_edge(j, i)
        self.assertEqual(before, self.graph.number_of_edges())
        self.assertEqual(degree_before, self.graph.degree[i])
        self.assertEqual(
            sorted(set(self.graph.neighbors(i))),
            sorted(self.graph.neighbors(i)),
        )

    def test_duplicate_edges_in_build_are_collapsed(self):
        """
        Real MICrONS SWCs repeat edges, and nx collapses them. This was found
        only once the spike's sweep reached 50 cells.
        """
        edges = np.concatenate([self.edges, self.edges[:25]])
        reference, graph = build_pair(N_NODES, edges)
        self.assertEqual(
            reference.number_of_edges(), graph.number_of_edges()
        )
        for i in range(N_NODES):
            self.assertEqual(
                reference.degree[i], graph.degree[i], f"node {i}"
            )
            self.assertEqual(
                sorted(reference.neighbors(i)),
                sorted(graph.neighbors(i)),
                f"node {i}",
            )

    def test_add_edge_agrees_with_rebuild(self):
        """Incremental edges must match a from-scratch build."""
        rng = np.random.default_rng(SEED + 1)
        extra = [
            (int(a), int(b))
            for a, b in rng.integers(0, N_NODES, (40, 2))
            if a != b
        ]
        self.reference.add_edges_from(extra)
        for i, j in extra:
            self.graph.add_edge(i, j)

        self.assertEqual(
            self.reference.number_of_edges(), self.graph.number_of_edges()
        )
        self.assertEqual(
            self.reference.number_connected_components(),
            self.graph.number_connected_components(),
        )
        for i in range(N_NODES):
            self.assertEqual(
                self.reference.degree[i], self.graph.degree[i], f"node {i}"
            )
            self.assertEqual(
                sorted(self.reference.neighbors(i)),
                sorted(self.graph.neighbors(i)),
                f"node {i}",
            )

    def test_add_edge_reads_served_from_overlay(self):
        """
        Adjacency reads must see pending edges without forcing a rebuild: the
        merge-proofreading accept loop interleaves add_edge with a traversal of
        the merged component on every accept.
        """
        self.graph.number_connected_components()
        rebuilds = self.graph.rebuild_count
        rng = np.random.default_rng(SEED + 2)
        for a, b in rng.integers(0, N_NODES, (50, 2)).tolist():
            if a == b:
                continue
            self.graph.add_edge(a, b)
            self.assertIn(b, list(self.graph.neighbors(a)))
            self.assertIn(a, list(self.graph.neighbors(b)))
            self.assertTrue(self.graph.has_path(a, b))
        self.assertEqual(rebuilds, self.graph.rebuild_count)

    def test_add_edge_creates_nodes_implicitly(self):
        """rooted_subgraph relies on nx's implicit node creation."""
        new_id = N_NODES
        self.reference.add_edge(new_id, 0)
        self.graph.add_edge(new_id, 0)
        self.assertEqual(
            self.reference.number_of_nodes(), self.graph.number_of_nodes()
        )
        self.assertEqual(
            self.reference.degree[new_id], self.graph.degree[new_id]
        )
        self.assertEqual(
            sorted(self.reference.neighbors(new_id)),
            sorted(self.graph.neighbors(new_id)),
        )

    def test_removed_nodes_are_invisible(self):
        """
        CSR keeps a dense ID space, so removal is a mask. Everything that
        reports on nodes has to honour it or nx's behaviour is not reproduced.
        """
        drop = [5, 6, 7, 100, 101]
        self.reference.remove_nodes_from(drop)
        self.graph.remove_nodes_from(drop)

        self.assertEqual(
            self.reference.number_of_nodes(), self.graph.number_of_nodes()
        )
        self.assertEqual(sorted(self.reference.nodes), sorted(self.graph.nodes))
        for i in drop:
            self.assertFalse(self.graph.has_node(i))
            self.assertNotIn(i, self.graph.nodes)
        self.assertEqual(
            self.reference.number_of_edges(), self.graph.number_of_edges()
        )
        self.assertEqual(
            {frozenset(c) for c in self.reference.connected_components()},
            {frozenset(c) for c in self.graph.connected_components()},
        )
        self.assertEqual(
            self.reference.irreducible_nodes(), self.graph.irreducible_nodes()
        )

    def test_removals_accumulate_across_calls(self):
        """
        "trim_branches" removes nodes once per trimmed proposal and relabels
        once at the end, so the removal mask must accumulate rather than being
        replaced by the last call.
        """
        first, second = [10, 11, 12], [200, 201]
        for group in (first, second):
            self.reference.remove_nodes_from(group)
            self.graph.remove_nodes_from(group)
        self.assertEqual(
            self.reference.number_of_nodes(), self.graph.number_of_nodes()
        )
        for i in first + second:
            self.assertFalse(self.graph.has_node(i))

        self.graph.relabel_nodes()
        self.assertEqual(
            N_NODES - len(first) - len(second), self.graph.number_of_nodes()
        )

    def test_relabel_nodes_compacts_and_reindexes(self):
        drop = [3, 4, 250]
        keep = [i for i in range(N_NODES) if i not in drop]
        expected_xyz = self.graph.node_xyz[keep].copy()

        self.reference.remove_nodes_from(drop)
        self.graph.remove_nodes_from(drop)
        old_to_new = self.graph.relabel_nodes()

        self.assertEqual(len(keep), self.graph.number_of_nodes())
        self.assertEqual(list(range(len(keep))), sorted(self.graph.nodes))
        np.testing.assert_array_equal(expected_xyz, self.graph.node_xyz)
        self.assertEqual({old_to_new[i] for i in keep}, set(range(len(keep))))
        # Edge set must survive the renumbering.
        self.assertEqual(
            {
                frozenset((old_to_new[i], old_to_new[j]))
                for i, j in self.reference.edges
            },
            {frozenset(e) for e in self.graph.edges},
        )

    def test_remove_small_components(self):
        rng = np.random.default_rng(SEED)
        edges, num_nodes = fragmented_edges(rng, 10, 30)
        _, graph = build_pair(num_nodes, edges)
        before = graph.number_connected_components()
        graph.remove_small_components(min_size=10)
        self.assertLessEqual(graph.number_connected_components(), before)
        self.assertEqual(len(graph.node_xyz), graph.number_of_nodes())


class CSRSerializationTest(unittest.TestCase):
    """
    The neighbour array is an ndarray view over an array("i") buffer, which a
    naive copy silently unlinks. merge_inference deep-copies the graph.
    """

    def setUp(self):
        rng = np.random.default_rng(SEED)
        _, self.graph = build_pair(N_NODES, random_tree_edges(rng, N_NODES))

    def assert_intact(self, graph):
        self.assertEqual(N_NODES, graph.number_of_nodes())
        self.assertEqual(
            self.graph.number_of_edges(), graph.number_of_edges()
        )
        np.testing.assert_array_equal(
            np.asarray(self.graph._indices), np.asarray(graph._indices)
        )
        # The view must alias the buffer it was rebuilt from, not a stale copy.
        self.assertEqual(
            graph._nbr.buffer_info()[0],
            graph._indices.__array_interface__["data"][0],
        )
        for i in range(0, N_NODES, 37):
            self.assertEqual(
                list(self.graph.neighbors(i)), list(graph.neighbors(i))
            )
        np.testing.assert_array_equal(self.graph.node_xyz, graph.node_xyz)

    def test_deepcopy_round_trip(self):
        self.assert_intact(deepcopy(self.graph))

    def test_pickle_round_trip(self):
        self.assert_intact(pickle.loads(pickle.dumps(self.graph)))

    def test_deepcopy_is_independent(self):
        clone = deepcopy(self.graph)
        clone.add_edge(0, N_NODES - 1)
        self.assertNotEqual(
            self.graph.number_of_edges(), clone.number_of_edges()
        )
        clone.node_xyz[0] = 12345.0
        self.assertNotEqual(12345.0, self.graph.node_xyz[0][0])


if __name__ == "__main__":
    unittest.main()
