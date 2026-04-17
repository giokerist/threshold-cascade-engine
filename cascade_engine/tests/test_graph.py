"""Unit tests for graph generation module."""

from __future__ import annotations
import sys, unittest, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cascade_engine.graph import (
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


class TestCustomGraphValidation(unittest.TestCase):
    """Tests for self-loop and duplicate-edge warning behavior."""

    def test_self_loop_stripped_from_adjacency(self):
        """Self-loop (i, i) must not appear in the adjacency matrix."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            A = generate_custom(4, [(0, 1), (2, 2), (1, 3)])
        self.assertTrue(np.all(np.diag(A) == 0),
                        "Diagonal must be zero — self-loops must be stripped")

    def test_self_loop_emits_warning(self):
        """A UserWarning must be emitted when self-loops are detected."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_custom(4, [(0, 1), (2, 2)])
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0,
                           "Expected a UserWarning for self-loop input")
        self.assertIn("self-loop", str(user_warnings[0].message).lower())

    def test_self_loop_valid_edges_preserved(self):
        """Non-loop edges must survive self-loop stripping unchanged."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            A = generate_custom(4, [(0, 1), (1, 1), (2, 3)])
        self.assertEqual(A[0, 1], 1)
        self.assertEqual(A[2, 3], 1)
        self.assertEqual(A[1, 1], 0)

    def test_duplicate_edges_collapsed(self):
        """Duplicate edges must appear as a single edge in the adjacency matrix."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            A = generate_custom(3, [(0, 1), (0, 1), (0, 1)])
        self.assertEqual(A[0, 1], 1,
                         "Edge (0,1) should be present exactly once")

    def test_duplicate_edges_emits_warning(self):
        """A UserWarning must be emitted when duplicate edges are detected."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_custom(3, [(0, 1), (1, 2), (0, 1)])
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0,
                           "Expected a UserWarning for duplicate edges")
        self.assertIn("duplicate", str(user_warnings[0].message).lower())

    def test_clean_edge_list_no_warnings(self):
        """No warnings should be emitted for a clean edge list."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_custom(5, [(0, 1), (1, 2), (3, 4)])
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 0,
                         "No warnings expected for a clean edge list")

    def test_zero_indegree_node_not_affected_by_cascade(self):
        """A node with no in-edges must not change state due to neighbor failures."""
        # Node 0 has no in-neighbors; nodes 1 and 2 both point to node 3.
        # Seeding 1 and 2 as failed should cascade to 3 but never touch 0.
        from cascade_engine.propagation import run_until_stable, STATE_FAILED
        A = generate_custom(4, [(1, 3), (2, 3)])
        theta_deg  = np.full(4, 0.3)
        theta_fail = np.full(4, 0.5)
        S0 = np.array([0, 2, 2, 0], dtype=np.int32)  # nodes 1 and 2 pre-failed
        final, _, _, _ = run_until_stable(S0, A, theta_deg, theta_fail)
        self.assertEqual(final[0], 0,
                         "Node 0 has zero in-degree and must remain operational")
        self.assertEqual(final[3], STATE_FAILED,
                         "Node 3 should fail due to both in-neighbors failing")


if __name__ == "__main__":
    unittest.main()
