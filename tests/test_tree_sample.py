"""Tests for build_tree_sample and its lowest-common-ancestor helper."""

import unittest

import numpy as np

from arborist.data.datasets import _lowest_common_ancestors, build_tree_sample


def random_tree(n, rng, min_pts=3, max_pts=30):
    """Random parent-first topology with root 0 and n finite-difference curves."""
    edges = [(int(rng.integers(0, d)), d) for d in range(1, n + 1)]
    curves = [
        np.vstack([
            np.zeros(3), rng.normal(size=(int(rng.integers(min_pts, max_pts)), 3))
        ])
        for _ in range(n)
    ]
    return curves, edges


def reference_lca(ancestors, depths):
    """O(n^3) reference: deepest node in the intersection of ancestor sets."""
    common = ancestors[:, None, :] & ancestors[None, :, :]
    return (common * (depths + 1)).argmax(axis=-1)


class LowestCommonAncestorsTest(unittest.TestCase):
    """_lowest_common_ancestors matches the ancestor-set intersection."""

    def test_matches_reference(self):
        """Agrees with the reference on random trees of several sizes."""
        rng = np.random.default_rng(0)
        for n in (1, 2, 5, 20, 200):
            _, edges = random_tree(n, rng)
            anc = np.eye(n + 1, dtype=bool)
            depths = np.zeros(n + 1, dtype=np.float32)
            for p, d in edges:
                anc[d] |= anc[p]
                depths[d] = depths[p] + 1
            np.testing.assert_array_equal(
                _lowest_common_ancestors(anc, depths), reference_lca(anc, depths)
            )

    def test_diagonal_is_self(self):
        """A node is its own lowest common ancestor."""
        anc = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
        lca = _lowest_common_ancestors(anc, np.array([0, 1, 2]))
        np.testing.assert_array_equal(np.diag(lca), [0, 1, 2])


class BuildTreeSampleTest(unittest.TestCase):
    """TreeSample fields are consistent with the topology."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.curves, self.edges = random_tree(25, self.rng)
        self.sample = build_tree_sample(self.curves, self.edges)

    def test_parents(self):
        """Each curve's parent token ends where the curve starts."""
        topo = np.asarray(self.edges)
        proximal = np.concatenate([[0], topo[:, 0]])
        distal = np.concatenate([[0], topo[:, 1]])
        parents = self.sample.parents
        self.assertEqual(parents[0], -1)
        np.testing.assert_array_equal(distal[parents[1:]], proximal[1:])
        for k in self.sample.root_curve_indices:
            self.assertEqual(parents[k + 1], 0)

    def test_relative_arrays(self):
        """Parent-child pairs are (1, 0) hops apart and one curve length."""
        s = self.sample
        for i in range(1, len(s.parents)):
            p = s.parents[i]
            np.testing.assert_array_equal(s.rel_hops[i, p], [1, 0])
            np.testing.assert_array_equal(s.rel_hops[p, i], [0, 1])
            length = np.linalg.norm(self.curves[i - 1], axis=1).sum()
            self.assertAlmostEqual(s.rel_cable[i, p, 0], length, places=4)
            self.assertAlmostEqual(s.rel_cable[i, p, 1], 0.0)

    def test_rejects_child_first_order(self):
        """Edges listed before their parent raise."""
        with self.assertRaises(ValueError):
            build_tree_sample(self.curves[:2], [(1, 2), (0, 1)])

    def test_empty(self):
        """A sample with no curves has only the root token."""
        s = build_tree_sample([], [])
        self.assertEqual(s.endpoints.shape, (1, 2, 3))
        self.assertEqual(s.rel_hops.shape, (1, 1, 2))
        np.testing.assert_array_equal(s.parents, [-1])


if __name__ == "__main__":
    unittest.main()
