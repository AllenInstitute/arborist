"""Tests for the SkeletonGraph node attribute registry.

The registry replaces the individual node_component_id/node_radius/node_xyz
instance attributes with a single store fronted by properties. These tests pin
down the access patterns used across arborist and its subclasses, so that the
storage change stays invisible to callers.
"""

import copy
import os
import shutil
import tempfile
import unittest

import numpy as np

from arborist.skeleton_graph import SkeletonGraph


def write_toy_swc(path):
    """
    Writes an SWC file with a trunk and two branches, long enough to survive
    the loader's pruning and minimum cable length filters.

    Parameters
    ----------
    path : str
        Path that SWC file is written to.
    """
    lines = ["# id type x y z r pid"]
    node_id = 0

    def add(node_type, x, y, z, parent_id):
        nonlocal node_id
        node_id += 1
        lines.append(f"{node_id} {node_type} {x} {y} {z} 1.0 {parent_id}")
        return node_id

    root = add(1, 0, 0, 0, -1)
    node = root
    for x in range(1, 61):
        node = add(3, x, 0, 0, node)

    branch = node
    node = branch
    for y in range(1, 61):
        node = add(3, 60, y, 0, node)

    node = branch
    for y in range(1, 61):
        node = add(2, 60, -y, 0, node)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class NodeAttrRegistryTest(unittest.TestCase):
    """Tests that node attributes behave like plain NumPy arrays."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        # SWC name must contain a "." so that check_swc_ids can deduplicate
        cls.swc_path = os.path.join(cls.tmp_dir, "toy.0.swc")
        write_toy_swc(cls.swc_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def build(self, cls=SkeletonGraph):
        graph = cls()
        graph.load(self.swc_path)
        return graph

    def test_load_allocates_every_registered_attr(self):
        graph = self.build()
        n = graph.number_of_nodes()
        self.assertGreater(n, 0)
        self.assertEqual(graph.node_component_id.shape, (n,))
        self.assertEqual(graph.node_radius.shape, (n,))
        self.assertEqual(graph.node_xyz.shape, (n, 3))
        self.assertEqual(graph.node_component_id.dtype, np.dtype(int))
        self.assertEqual(graph.node_radius.dtype, np.float16)
        self.assertEqual(graph.node_xyz.dtype, np.float32)

    def test_load_populates_geometry(self):
        graph = self.build()
        self.assertTrue(np.all(graph.node_radius > 0))
        self.assertTrue(np.all(graph.node_component_id == 0))
        # trunk runs 0..60 along x, branches span -60..60 along y
        self.assertAlmostEqual(float(graph.node_xyz[:, 0].max()), 60, places=3)
        self.assertAlmostEqual(float(graph.node_xyz[:, 1].max()), 60, places=3)
        self.assertAlmostEqual(
            float(graph.node_xyz[:, 1].min()), -60, places=3
        )

    def test_read_patterns(self):
        graph = self.build()
        idxs = np.array([0, 5, 11, 17])
        self.assertIsInstance(float(graph.node_radius[3]), float)
        self.assertEqual(graph.node_xyz[idxs].shape, (4, 3))
        self.assertEqual(graph.node_xyz[idxs, 0].shape, (4,))
        self.assertEqual(graph.node_component_id[idxs].shape, (4,))
        self.assertTrue(np.all(graph.node_xyz[graph.node_radius > 0].shape[0]))

    def test_in_place_item_assignment(self):
        graph = self.build()
        idxs = np.array([0, 5, 11])
        graph.node_component_id[idxs] = 7
        self.assertTrue(np.all(graph.node_component_id[idxs] == 7))
        graph.node_radius[2] = 6
        self.assertEqual(float(graph.node_radius[2]), 6)
        graph.node_xyz[1] = [1, 2, 3]
        np.testing.assert_allclose(graph.node_xyz[1], [1, 2, 3])

    def test_whole_array_assignment(self):
        graph = self.build()
        n = graph.number_of_nodes()
        graph.node_radius = np.ones((n,), dtype=np.float16)
        self.assertEqual(float(np.sum(graph.node_radius.astype(float))), n)

        graph.node_xyz = graph.node_xyz * 2.0
        self.assertEqual(graph.node_xyz.dtype, np.float32)

    def test_whole_array_assignment_does_not_enforce_dtype(self):
        """Callers replace node_radius with float64; that must still work."""
        graph = self.build()
        n = graph.number_of_nodes()
        graph.node_radius = 10 * np.maximum(np.zeros(n, dtype=np.float64), 0.1)
        self.assertEqual(graph.node_radius.dtype, np.float64)

    def test_getattr_setattr_by_name(self):
        """Pins the contract used by FragmentsGraph.resize_node_attr."""
        graph = self.build()
        for attr_name, shape in [
            ("node_radius", None),
            ("node_xyz", 3),
            ("node_component_id", None),
        ]:
            arr = getattr(graph, attr_name)
            new_shape = (len(arr) + 5,) if shape is None else (len(arr) + 5, 3)
            grown = np.empty(new_shape, dtype=arr.dtype)
            grown[: len(arr)] = arr
            setattr(graph, attr_name, grown)
            self.assertEqual(len(getattr(graph, attr_name)), len(arr) + 5)

    def test_relabel_keeps_attrs_aligned(self):
        graph = self.build()
        leaf = next(i for i in graph.nodes if graph.degree[i] == 1)
        expected_xyz = np.array(
            [graph.node_xyz[i] for i in graph.nodes if i != leaf]
        )

        graph.remove_nodes([leaf])

        n = graph.number_of_nodes()
        self.assertEqual(len(graph.node_xyz), n)
        self.assertEqual(len(graph.node_radius), n)
        self.assertEqual(len(graph.node_component_id), n)
        np.testing.assert_allclose(graph.node_xyz, expected_xyz)

    def test_rooted_subgraph_carries_every_attr(self):
        graph = self.build()
        subgraph = graph.rooted_subgraph(0, 20)
        n = subgraph.number_of_nodes()
        self.assertGreater(n, 1)
        self.assertEqual(len(subgraph.node_xyz), n)
        self.assertEqual(len(subgraph.node_radius), n)
        # regression: component IDs used to be dropped by rooted_subgraph
        self.assertEqual(len(subgraph.node_component_id), n)

    def test_subclass_extends_spec(self):
        class TaggedGraph(SkeletonGraph):
            NODE_ATTR_SPEC = {
                **SkeletonGraph.NODE_ATTR_SPEC,
                "tag": (np.uint8, ()),
            }

        graph = self.build(cls=TaggedGraph)
        n = graph.number_of_nodes()
        self.assertEqual(graph.node_tag.shape, (n,))
        self.assertEqual(graph.node_tag.dtype, np.uint8)
        self.assertEqual(graph.node_xyz.shape, (n, 3))

        graph.node_tag[3] = 2
        leaf = next(i for i in graph.nodes if graph.degree[i] == 1)
        graph.remove_nodes([leaf])
        self.assertEqual(len(graph.node_tag), graph.number_of_nodes())

    def test_unset_attr_raises_attribute_error(self):
        graph = SkeletonGraph()
        with self.assertRaises(AttributeError):
            graph.node_radius

    def test_deepcopy(self):
        graph = self.build()
        clone = copy.deepcopy(graph)
        np.testing.assert_allclose(clone.node_xyz, graph.node_xyz)
        clone.node_xyz[0] = [9, 9, 9]
        self.assertFalse(np.allclose(clone.node_xyz[0], graph.node_xyz[0]))

    def test_copy_node_attrs_full(self):
        graph = self.build()
        target = SkeletonGraph()
        graph.copy_node_attrs(target)
        np.testing.assert_allclose(target.node_xyz, graph.node_xyz)
        target.node_xyz[0] = [9, 9, 9]
        self.assertFalse(np.allclose(target.node_xyz[0], graph.node_xyz[0]))


if __name__ == "__main__":
    unittest.main()
