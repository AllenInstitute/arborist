"""Tests for hierarchical pooling in Arborist."""

import os
import tempfile
import unittest

import numpy as np
import torch

from arborist.data.datasets import TreeSample, build_tree_sample
from arborist.models.arborist import Arborist
from arborist.models.hierarchical_transformer import partition_curves
from tests.test_tree_sample import random_tree


def curve_lengths(sample):
    """Cable length of each token's own curve; 0 for the root."""
    T = len(sample.parents)
    lengths = sample.rel_cable[np.arange(T), sample.parents, 0]
    lengths[0] = 0
    return lengths


class PartitionCurvesTest(unittest.TestCase):
    """partition_curves makes connected, cable-bounded groups."""

    def setUp(self):
        rng = np.random.default_rng(2)
        self.sample = build_tree_sample(*random_tree(40, rng))
        self.lengths = curve_lengths(self.sample)

    def test_root_children_start_groups(self):
        """With no cable limit, groups are the subtrees under the root."""
        group_of, heads = partition_curves(self.sample.parents, self.lengths)
        self.assertEqual(group_of[0], -1)
        self.assertEqual(
            sorted(heads), sorted(k + 1 for k in self.sample.root_curve_indices)
        )
        for k in range(1, len(group_of)):
            p = self.sample.parents[k]
            if p > 0:
                self.assertEqual(group_of[k], group_of[p])

    def test_cable_limit(self):
        """Groups are connected and stay within the cable limit."""
        limit = 3 * self.lengths.mean()
        group_of, heads = partition_curves(
            self.sample.parents, self.lengths, max_cable=limit
        )
        self.assertEqual(len(heads), group_of.max() + 1)
        for g, head in enumerate(heads):
            members = np.nonzero(group_of == g)[0]
            self.assertIn(head, members)
            total = self.lengths[members].sum()
            self.assertTrue(total <= limit or len(members) == 1)
            for k in members:
                if k != head:
                    self.assertEqual(group_of[self.sample.parents[k]], g)
            self.assertNotEqual(group_of[self.sample.parents[head]], g)

    def test_tiny_limit_is_one_curve_per_group(self):
        """A limit below every curve length isolates each curve."""
        group_of, heads = partition_curves(
            self.sample.parents, self.lengths, max_cable=0.0
        )
        self.assertEqual(len(heads), len(self.sample.curves))
        np.testing.assert_array_equal(heads, np.arange(1, len(group_of)))


class HierarchicalArboristTest(unittest.TestCase):
    """Arborist with pooling="hierarchical" runs end to end."""

    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(3)
        self.sample = build_tree_sample(*random_tree(30, self.rng, min_pts=11))

    def test_rejects_unknown_pooling(self):
        """An unknown pooling name raises."""
        with self.assertRaises(ValueError):
            Arborist(pooling="max")

    def test_shapes_and_gradients(self):
        """Every pooling returns the flat shapes and trains every parameter."""
        for kwargs in (
            {"pooling": "cls"},
            {"pooling": "hierarchical"},
            {"pooling": "hierarchical", "pool_cable": 50.0, "pool_local_layers": 2},
        ):
            model = Arborist(latent_dim=16, **kwargs)
            z_tree, z_curves = model.encode(self.sample)
            self.assertEqual(z_tree.shape, (16,))
            self.assertEqual(z_curves.shape, (30, 16))
            (z_tree.sum() + z_curves.sum()).backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)

    def test_single_curve_and_empty(self):
        """Degenerate trees encode without error."""
        model = Arborist(latent_dim=16, pooling="hierarchical")
        one = build_tree_sample(*random_tree(1, self.rng, min_pts=11))
        self.assertEqual(model.encode(one)[1].shape, (1, 16))
        empty = TreeSample([], [], None, None, None, None, None)
        z_tree, z_curves = model.encode(empty)
        self.assertEqual(z_curves.shape, (1, 16))

    def test_group_curves_without_parents(self):
        """Samples built before TreeSample.parents existed group identically."""
        model = Arborist(latent_dim=16, pooling="hierarchical", pool_cable=50.0)
        s = self.sample
        legacy = TreeSample(
            s.curves, s.root_curve_indices, s.endpoints, s.hops, s.cable,
            s.rel_hops, s.rel_cable,
        )
        group_of, heads = model.group_curves(s)
        legacy_group_of, legacy_heads = model.group_curves(legacy)
        np.testing.assert_array_equal(group_of, legacy_group_of)
        np.testing.assert_array_equal(heads, legacy_heads)

    def test_curves_outside_group_do_not_mix_in_stage_one(self):
        """Stage 1 is local: curve outputs depend on other groups only via
        the broadcast group tokens, so with the broadcast zeroed a change
        to one group leaves the other groups' curves unchanged."""
        model = Arborist(latent_dim=16, pooling="hierarchical").eval()
        for p in model.graph_transformer.broadcast.parameters():
            torch.nn.init.zeros_(p)
        group_of, _ = model.group_curves(self.sample)
        with torch.no_grad():
            base = model.encode(self.sample)[1]
            curves = [c.copy() for c in self.sample.curves]
            k = np.nonzero(group_of == 0)[0][0]
            curves[k - 1] = curves[k - 1] * 2.0
            other = build_tree_sample(curves, self.sample_edges())
            changed = model.encode(other)[1]
        same_group = group_of[1:] == 0
        self.assertTrue(torch.allclose(base[~same_group], changed[~same_group], atol=1e-5))
        self.assertFalse(torch.allclose(base[same_group], changed[same_group], atol=1e-5))

    def sample_edges(self):
        """Recover (proximal, distal) topology from the sample's parents."""
        distal = np.arange(len(self.sample.parents))
        return [(int(distal[p]), int(d)) for d, p in enumerate(self.sample.parents) if p >= 0]

    def test_save_load(self):
        """Config round-trips and legacy configs load as flat models."""
        model = Arborist(latent_dim=16, pooling="hierarchical", pool_cable=75.0)
        flat = Arborist(latent_dim=16)
        with tempfile.TemporaryDirectory() as d:
            model.save(os.path.join(d, "h.pth"))
            loaded = Arborist.load(os.path.join(d, "h.pth"))
            self.assertEqual(loaded.config["pooling"], "hierarchical")
            self.assertEqual(loaded.config["pool_cable"], 75.0)
            legacy = {
                "config": {
                    k: v for k, v in flat.config.items() if not k.startswith("pool")
                },
                "state_dict": flat.state_dict(),
            }
            torch.save(legacy, os.path.join(d, "legacy.pth"))
            self.assertEqual(
                Arborist.load(os.path.join(d, "legacy.pth")).config["pooling"], "cls"
            )


if __name__ == "__main__":
    unittest.main()
