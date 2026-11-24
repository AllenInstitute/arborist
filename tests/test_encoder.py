"""Tests for readouts and ArboristEncoder."""

import os
import tempfile
import unittest

import numpy as np
import torch

from arborist.data.datasets import build_tree_sample
from arborist.models.arborist import READOUTS, Arborist, ArboristEncoder, readout
from tests.test_tree_sample import random_tree

LATENT = 16
SMALL = dict(
    curve_d_token=16, curve_n_layers=1, curve_d_ff=32, graph_n_layers=1,
    graph_d_ff=32, dropout=0.0,
)


def make_samples(n):
    rng = np.random.default_rng(0)
    return [build_tree_sample(*random_tree(8, rng)) for _ in range(n)]


class ReadoutTest(unittest.TestCase):
    def setUp(self):
        self.arborist = Arborist(latent_dim=LATENT, **SMALL).eval()
        self.sample = make_samples(1)[0]
        with torch.no_grad():
            self.z_tree, self.z_curves = self.arborist.encode(self.sample)

    def test_cls_is_the_global_embedding(self):
        torch.testing.assert_close(
            readout(self.z_tree, self.z_curves, self.sample, "cls"), self.z_tree
        )

    def test_cls_root_curves_concatenates_root_curve_mean(self):
        z = readout(self.z_tree, self.z_curves, self.sample, "cls+root_curves")
        self.assertEqual(z.shape, (2 * LATENT,))
        torch.testing.assert_close(z[:LATENT], self.z_tree)
        root_mean = self.z_curves[list(self.sample.root_curve_indices)].mean(dim=0)
        torch.testing.assert_close(z[LATENT:], root_mean)

    def test_unknown_readout_raises(self):
        with self.assertRaises(ValueError):
            readout(self.z_tree, self.z_curves, self.sample, "mean")


class ArboristEncoderTest(unittest.TestCase):
    def test_batch_matches_per_sample_readout(self):
        samples = make_samples(3)
        for kind in READOUTS:
            encoder = ArboristEncoder(LATENT, readout=kind, **SMALL).eval()
            with torch.no_grad():
                z = encoder(samples)
                expected = torch.stack(
                    [readout(*encoder.arborist.encode(s), s, kind) for s in samples]
                )
            self.assertEqual(z.shape, (3, encoder.embed_dim))
            torch.testing.assert_close(z, expected)

    def test_config_and_embed_dim(self):
        encoder = ArboristEncoder(LATENT, readout="cls", **SMALL)
        self.assertEqual(encoder.embed_dim, LATENT)
        self.assertEqual(encoder.config["readout"], "cls")
        self.assertEqual(encoder.config["curve_d_token"], 16)
        with self.assertRaises(ValueError):
            ArboristEncoder(LATENT, readout="mean")

    def test_save_load_roundtrip(self):
        samples = make_samples(2)
        encoder = ArboristEncoder(LATENT, **SMALL).eval()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e.pt")
            encoder.save(path)
            loaded = ArboristEncoder.load(path).eval()
        with torch.no_grad():
            torch.testing.assert_close(encoder(samples), loaded(samples))


if __name__ == "__main__":
    unittest.main()
