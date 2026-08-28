"""Tests for SWC node type propagation into SkeletonGraph.

The SWC type column used to be read only to flag somas and then discarded.
These tests cover carrying it through the loader onto graph nodes and back out
of the writer, and pin the requirement that it is resampled by nearest
neighbor rather than by spline.
"""

import os
import shutil
import tempfile
import unittest
import zipfile as zf

import numpy as np

from arborist.skeleton_graph import SkeletonGraph
from arborist.utils import geometry

from tests.test_node_attrs import write_toy_swc

# write_toy_swc uses soma for the root, basal for the trunk and +y branch, and
# axon for the -y branch
SOMA, AXON, BASAL = 1, 2, 3


class ResampleLabelsTest(unittest.TestCase):
    """Tests for nearest neighbor resampling of categorical labels."""

    def test_length_and_dtype(self):
        labels = np.array([2, 2, 3, 3], dtype=np.uint8)
        out = geometry.resample_labels_1d(labels, 9)
        self.assertEqual(len(out), 9)
        self.assertEqual(out.dtype, np.uint8)

    def test_only_returns_input_labels(self):
        labels = np.array([2, 2, 4], dtype=np.uint8)
        out = geometry.resample_labels_1d(labels, 17)
        self.assertTrue(set(np.unique(out)).issubset({2, 4}))

    def test_endpoints_preserved(self):
        labels = np.array([1, 3, 3, 2], dtype=np.uint8)
        out = geometry.resample_labels_1d(labels, 12)
        self.assertEqual(out[0], 1)
        self.assertEqual(out[-1], 2)

    def test_single_label(self):
        labels = np.array([3], dtype=np.uint8)
        out = geometry.resample_labels_1d(labels, 6)
        self.assertEqual(len(out), 6)
        self.assertTrue(np.all(out == 3))

    def test_downsample(self):
        labels = np.array([2] * 10 + [3] * 10, dtype=np.uint8)
        out = geometry.resample_labels_1d(labels, 4)
        self.assertEqual(len(out), 4)
        self.assertTrue(set(np.unique(out)).issubset({2, 3}))

    def test_spline_resampling_would_invent_labels(self):
        """
        Motivates resample_labels_1d over resample_curve_1d. Splining a path
        that is only axon and basal dendrite undershoots into the soma type,
        labelling mid-dendrite nodes as soma.
        """
        labels = np.array([AXON, AXON, BASAL, BASAL, AXON], dtype=np.float32)
        spline = geometry.resample_curve_1d(labels, 20)
        self.assertLess(spline.min(), AXON)
        self.assertIn(SOMA, set(spline.astype(np.uint8).tolist()))

        nearest = geometry.resample_labels_1d(
            labels.astype(np.uint8), 20
        )
        self.assertTrue(set(np.unique(nearest)).issubset({AXON, BASAL}))


class NodeTypeTest(unittest.TestCase):
    """Tests that node types survive loading, resampling, and writing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.swc_path = os.path.join(cls.tmp_dir, "toy.0.swc")
        write_toy_swc(cls.swc_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def build(self):
        graph = SkeletonGraph()
        graph.load(self.swc_path)
        return graph

    def test_node_type_is_registered(self):
        graph = self.build()
        n = graph.number_of_nodes()
        self.assertEqual(graph.node_type.shape, (n,))
        self.assertEqual(graph.node_type.dtype, np.uint8)

    def test_no_invalid_type_codes(self):
        """Regression guard: spline resampling invents codes not in the SWC."""
        graph = self.build()
        found = set(int(t) for t in np.unique(graph.node_type))
        self.assertTrue(found.issubset({SOMA, AXON, BASAL}), found)

    def test_both_arbor_types_present(self):
        graph = self.build()
        found = set(int(t) for t in np.unique(graph.node_type))
        self.assertIn(AXON, found)
        self.assertIn(BASAL, found)

    def test_types_follow_geometry(self):
        """The -y branch is axon, the +y branch is basal."""
        graph = self.build()
        y = graph.node_xyz[:, 1]
        below = graph.node_type[y < -5]
        above = graph.node_type[y > 5]
        self.assertGreater(len(below), 0)
        self.assertGreater(len(above), 0)
        self.assertTrue(np.all(below == AXON))
        self.assertTrue(np.all(above == BASAL))

    def test_soma_node_type_preserved(self):
        graph = self.build()
        self.assertIn(SOMA, set(int(t) for t in np.unique(graph.node_type)))

    def test_type_survives_relabel(self):
        graph = self.build()
        leaf = next(i for i in graph.nodes if graph.degree[i] == 1)
        expected = np.array(
            [graph.node_type[i] for i in graph.nodes if i != leaf]
        )
        graph.remove_nodes([leaf])
        np.testing.assert_array_equal(graph.node_type, expected)

    def test_rooted_subgraph_carries_type(self):
        graph = self.build()
        subgraph = graph.rooted_subgraph(0, 20)
        self.assertEqual(
            len(subgraph.node_type), subgraph.number_of_nodes()
        )

    def test_writer_emits_stored_type(self):
        graph = self.build()
        zip_path = os.path.join(self.tmp_dir, "out.zip")
        graph.to_zipped_swcs(zip_path)

        types = list()
        with zf.ZipFile(zip_path) as zipfile:
            for name in zipfile.namelist():
                content = zipfile.read(name).decode().splitlines()
                for line in content:
                    if line.startswith("#") or not line.strip():
                        continue
                    types.append(int(line.split()[1]))

        self.assertGreater(len(types), 0)
        found = set(types)
        self.assertTrue(found.issubset({SOMA, AXON, BASAL}), found)
        self.assertIn(AXON, found)
        self.assertIn(BASAL, found)

    def test_writer_falls_back_for_untyped_nodes(self):
        """Graphs not loaded from SWC keep the previous type 2 output."""
        graph = self.build()
        graph.node_type = np.zeros_like(graph.node_type)
        zip_path = os.path.join(self.tmp_dir, "untyped.zip")
        graph.to_zipped_swcs(zip_path)

        with zf.ZipFile(zip_path) as zipfile:
            name = zipfile.namelist()[0]
            content = zipfile.read(name).decode().splitlines()
        types = {
            int(line.split()[1])
            for line in content
            if line.strip() and not line.startswith("#")
        }
        self.assertEqual(types, {AXON})

    def test_grow_node_attrs_preserves_and_zero_fills(self):
        graph = self.build()
        n = graph.number_of_nodes()
        expected = graph.node_type.copy()

        graph.grow_node_attrs(n + 4)

        self.assertEqual(len(graph.node_type), n + 4)
        self.assertEqual(len(graph.node_xyz), n + 4)
        np.testing.assert_array_equal(graph.node_type[:n], expected)
        self.assertTrue(np.all(graph.node_type[n:] == 0))


if __name__ == "__main__":
    unittest.main()
