"""Tests for compartment-label resampling in the SWC graph loader."""

import os
import tempfile
import unittest

import numpy as np

from arborist.skeleton_graph import SkeletonGraph
from arborist.utils.graph_loading import _resample_labels


class ResampleLabelsTest(unittest.TestCase):
    """A soma is a point: resampling must not spread it along a neurite."""

    def test_soma_to_neurite_edge_keeps_one_soma_point(self):
        out = _resample_labels(np.array([1, 3, 3], dtype=np.uint8), 20)
        self.assertEqual(out[0], 1)
        self.assertTrue(np.all(out[1:] == 3))

    def test_two_node_edge(self):
        out = _resample_labels(np.array([1, 2], dtype=np.uint8), 50)
        self.assertEqual(out[0], 1)
        self.assertTrue(np.all(out[1:] == 2))

    def test_neurite_to_soma_edge(self):
        out = _resample_labels(np.array([4, 4, 1], dtype=np.uint8), 20)
        self.assertEqual(out[-1], 1)
        self.assertTrue(np.all(out[:-1] == 4))

    def test_points_between_soma_nodes_stay_soma(self):
        out = _resample_labels(np.array([1, 1, 1, 3], dtype=np.uint8), 31)
        t = np.linspace(0, 3, 31)
        np.testing.assert_array_equal(out[t < 2], 1)
        np.testing.assert_array_equal(out[t > 2], 3)

    def test_neurite_labels_still_nearest_neighbour(self):
        labels = np.array([3, 3, 2, 2], dtype=np.uint8)
        idxs = np.rint(np.linspace(0, 3, 13)).astype(int)
        np.testing.assert_array_equal(_resample_labels(labels, 13), labels[idxs])


class LoadSomaTest(unittest.TestCase):
    """End to end: a single-node soma loads as a single soma node."""

    def test_single_soma_node_survives_load(self):
        # Soma at the origin, a 60 um stem, then two 60 um branches.
        rows = [(1, 1, 0, 0, 0, -1)]
        for k in range(1, 4):
            rows.append((len(rows) + 1, 3, 20.0 * k, 0, 0, len(rows)))
        fork = len(rows)
        for sign in (1, -1):
            parent = fork
            for k in range(1, 4):
                rows.append((len(rows) + 1, 3, 60 + 20.0 * k, sign * 20.0 * k, 0, parent))
                parent = len(rows)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "cell.swc"), "w") as f:
                for n, t, x, y, z, p in rows:
                    f.write(f"{n} {t} {x} {y} {z} 1.0 {p}\n")
            graph = SkeletonGraph()
            graph.load(d)
        soma = np.flatnonzero(graph.node_feats["type"] == 1)
        self.assertEqual(len(soma), 1)
        np.testing.assert_allclose(graph.node_xyz[soma[0]], 0.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
