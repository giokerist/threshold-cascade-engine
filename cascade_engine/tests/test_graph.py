"""Unit tests for graph generation module."""

from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph import (
    generate_erdos_renyi, generate_barabasi_albert,
    generate_watts_strogatz, generate_custom,
)


class TestGraphProperties(unittest.TestCase):

    def test_erdos_renyi_shape(self):
        A = generate_erdos_renyi(20, 0.3, seed=0)
        self.assertEqual(A.shape, (20, 20))

    def test_erdos_renyi_no_self_loops(self):
        A = generate_erdos_renyi(15, 0.5, seed=1)
        self.assertTrue(np.all(np.diag(A) == 0))

    def test_erdos_renyi_deterministic(self):
        A1 = generate_erdos_renyi(20, 0.3, seed=42)
        A2 = generate_erdos_renyi(20, 0.3, seed=42)
        np.testing.assert_array_equal(A1, A2)

    def test_barabasi_albert_shape(self):
        A = generate_barabasi_albert(30, 2, seed=0)
        self.assertEqual(A.shape, (30, 30))

    def test_barabasi_albert_symmetric(self):
        A = generate_barabasi_albert(20, 2, seed=5)
        np.testing.assert_array_equal(A, A.T)

    def test_watts_strogatz_shape(self):
        A = generate_watts_strogatz(25, 4, 0.1, seed=0)
        self.assertEqual(A.shape, (25, 25))

    def test_watts_strogatz_no_self_loops(self):
        A = generate_watts_strogatz(20, 4, 0.2, seed=3)
        self.assertTrue(np.all(np.diag(A) == 0))

    def test_custom_correct_edges(self):
        A = generate_custom(5, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(A[0, 1], 1)
        self.assertEqual(A[1, 0], 0)
        self.assertEqual(A[2, 3], 1)

    def test_custom_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            generate_custom(5, [(0, 5)])

    def test_adjacency_convention(self):
        """A[j,i]=1 means j->i; column i = in-neighbors of i."""
        A = generate_custom(3, [(0, 2)])
        self.assertEqual(A[0, 2], 1)
        self.assertEqual(A[2, 0], 0)
        self.assertEqual(A[:, 2].sum(), 1)

    def test_empty_graph(self):
        A = generate_custom(5, [])
        self.assertTrue(np.all(A == 0))


if __name__ == "__main__":
    unittest.main()
