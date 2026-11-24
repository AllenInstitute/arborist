"""Tests for CurveEncoder tokenization."""

import unittest

import torch

from arborist.models.curve_transformer import CurveEncoder


class CurveEncoderTest(unittest.TestCase):
    """Curves that are not a whole number of segments keep their tail."""

    def setUp(self):
        torch.manual_seed(0)
        self.encoder = CurveEncoder(segment_len=10, d_token=16, latent_dim=8).eval()

    def test_partial_segment_is_encoded(self):
        """A 19-point curve yields two segment tokens, not one."""
        diffs = torch.randn(1, 19, 3)
        diffs[:, 0] = 0
        with torch.no_grad():
            z, tokens = self.encoder(diffs)
            z_head, tokens_head = self.encoder(diffs[:, :10])
        self.assertEqual(tokens.shape[1], 3)
        self.assertEqual(tokens_head.shape[1], 2)
        self.assertFalse(torch.allclose(z, z_head))

    def test_whole_segments_unchanged(self):
        """Padding is a no-op when the length is a multiple of segment_len."""
        diffs = torch.randn(2, 30, 3)
        diffs[:, 0] = 0
        with torch.no_grad():
            z, tokens = self.encoder(diffs)
        self.assertEqual(tokens.shape[1], 4)
        self.assertEqual(z.shape, (2, 8))

    def test_padding_mask_extended(self):
        """A padded batch encodes each curve as if it were alone."""
        a = torch.randn(1, 19, 3)
        b = torch.randn(1, 24, 3)
        a[:, 0] = b[:, 0] = 0
        batch = torch.zeros(2, 24, 3)
        batch[0, :19] = a[0]
        batch[1] = b[0]
        mask = torch.ones(2, 24, dtype=torch.bool)
        mask[0, :19] = False
        mask[1, :] = False
        with torch.no_grad():
            z_batch, _ = self.encoder(batch, mask)
            z_a, _ = self.encoder(a)
            z_b, _ = self.encoder(b)
        self.assertTrue(torch.allclose(z_batch[0], z_a[0], atol=1e-5))
        self.assertTrue(torch.allclose(z_batch[1], z_b[0], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
