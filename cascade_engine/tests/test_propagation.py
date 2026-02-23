"""
Unit tests for the cascade propagation engine.
Compatible with both pytest and unittest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from propagation import (
    run_until_stable,
    propagation_step,
    STATE_OPERATIONAL,
    STATE_DEGRADED,
    STATE_FAILED,
)
from graph import generate_custom
from metrics import fragility_index, cascade_size


def uniform_thresholds(n, deg, fail):
    return np.full(n, deg, dtype=np.float64), np.full(n, fail, dtype=np.float64)


class TestMonotonicity(unittest.TestCase):

    def test_states_never_decrease_random_graph(self):
        rng = np.random.default_rng(42)
        n = 30
        A = (rng.random((n, n)) < 0.15).astype(np.uint8)
        np.fill_diagonal(A, 0)
        S0 = rng.integers(0, 3, size=n, dtype=np.int32)
        td, tf = uniform_thresholds(n, 0.2, 0.6)
        _, _, history, _ = run_until_stable(S0, A, td, tf)
        for t in range(history.shape[0] - 1):
            self.assertTrue(np.all(history[t+1] >= history[t]))

    def test_monotonicity_many_seeds(self):
        n = 20
        rng = np.random.default_rng(0)
        A = (rng.random((n, n)) < 0.2).astype(np.uint8)
        np.fill_diagonal(A, 0)
        td, tf = uniform_thresholds(n, 0.25, 0.5)
        for seed in range(20):
            S0 = np.random.default_rng(seed).integers(0, 3, size=n, dtype=np.int32)
            _, _, history, _ = run_until_stable(S0, A, td, tf)
            for t in range(history.shape[0] - 1):
                self.assertTrue(np.all(history[t+1] >= history[t]))

    def test_propagation_step_monotone(self):
        n = 5
        A = generate_custom(n, [(0, 1), (1, 2)])
        S = np.array([2, 1, 0, 0, 0], dtype=np.int32)
        D = A.sum(axis=0).astype(np.float64)
        td, tf = uniform_thresholds(n, 0.5, 0.9)
        S_next = propagation_step(S, A, td, tf, D)
        self.assertTrue(np.all(S_next >= S))


class TestConvergenceBound(unittest.TestCase):

    def test_convergence_within_2n_steps(self):
        rng = np.random.default_rng(7)
        n = 50
        A = (rng.random((n, n)) < 0.1).astype(np.uint8)
        np.fill_diagonal(A, 0)
        td, tf = uniform_thresholds(n, 0.3, 0.6)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        _, steps, history, _ = run_until_stable(S0, A, td, tf)
        self.assertLessEqual(steps, 2 * n)
        self.assertEqual(history.shape[0] - 1, steps)

    def test_isolated_graph_zero_steps(self):
        n = 100
        A = np.zeros((n, n), dtype=np.uint8)
        td, tf = uniform_thresholds(n, 0.3, 0.7)
        S0 = np.zeros(n, dtype=np.int32)
        S0[5] = STATE_FAILED
        _, steps, _, _ = run_until_stable(S0, A, td, tf)
        self.assertEqual(steps, 0)

    def test_max_steps_terminates(self):
        rng = np.random.default_rng(3)
        n = 10
        A = (rng.random((n, n)) < 0.3).astype(np.uint8)
        np.fill_diagonal(A, 0)
        td, tf = uniform_thresholds(n, 0.1, 0.2)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        _, steps, _, _ = run_until_stable(S0, A, td, tf, max_steps=1)
        self.assertLessEqual(steps, 1)

    def test_already_stable_zero_steps(self):
        n = 5
        A = np.zeros((n, n), dtype=np.uint8)
        td, tf = uniform_thresholds(n, 0.5, 0.9)
        S0 = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        _, steps, history, _ = run_until_stable(S0, A, td, tf)
        self.assertEqual(steps, 0)
        np.testing.assert_array_equal(history[0], S0)


class TestChainGraph(unittest.TestCase):

    def _chain(self, n):
        return generate_custom(n, [(i, i+1) for i in range(n-1)])

    def test_full_propagation_zero_threshold(self):
        n = 8
        A = self._chain(n)
        td = np.zeros(n); tf = np.zeros(n)
        S0 = np.zeros(n, dtype=np.int32); S0[0] = STATE_FAILED
        final, _, _, _ = run_until_stable(S0, A, td, tf)
        self.assertTrue(np.all(final == STATE_FAILED), f"Got {final}")

    def test_no_propagation_unreachable_threshold(self):
        n = 6
        A = self._chain(n)
        td = np.full(n, 1.1); tf = np.full(n, 1.1)
        S0 = np.zeros(n, dtype=np.int32); S0[0] = STATE_FAILED
        final, steps, _, _ = run_until_stable(S0, A, td, tf)
        self.assertEqual(final[0], STATE_FAILED)
        for i in range(1, n):
            self.assertEqual(final[i], STATE_OPERATIONAL)
        self.assertEqual(steps, 0)

    def test_step_count_threshold_half(self):
        """Chain of n nodes with threshold=0.5 converges in exactly n-1 steps."""
        n = 5
        A = self._chain(n)
        td = np.full(n, 0.5); tf = np.full(n, 0.5)
        S0 = np.zeros(n, dtype=np.int32); S0[0] = STATE_FAILED
        final, steps, _, _ = run_until_stable(S0, A, td, tf)
        self.assertTrue(np.all(final == STATE_FAILED))
        self.assertEqual(steps, n - 1)

    def test_degraded_not_failed(self):
        """theta_fail > 1 (unreachable): only direct neighbors of failed node degrade.

        F_i counts FAILED in-neighbors only.  Node 1's parent (0) is failed ->
        F_1=1.0 >= td=0.5 -> node 1 degrades.  Node 2's parent (1) is only
        degraded (state 1, not 2), so F_2=0 -> node 2 stays operational.
        """
        n = 5
        A = self._chain(n)
        td = np.full(n, 0.5); tf = np.full(n, 1.5)
        S0 = np.zeros(n, dtype=np.int32); S0[0] = STATE_FAILED
        final, _, _, _ = run_until_stable(S0, A, td, tf)
        self.assertEqual(final[0], STATE_FAILED)
        # Only node 1 has a failed in-neighbor (node 0 is failed -> F_1=1.0>=0.5)
        self.assertEqual(final[1], STATE_DEGRADED)
        # Nodes 2..n-1: their in-neighbor is only degraded (not failed), so F_i=0
        for i in range(2, n):
            self.assertEqual(final[i], STATE_OPERATIONAL,
                             f"Node {i} expected OPERATIONAL (parent not failed)")


class TestZeroInDegree(unittest.TestCase):

    def test_source_nodes_unaffected(self):
        n = 5
        A = generate_custom(n, [(0, 1), (1, 2)])
        td = np.full(n, 0.1); tf = np.full(n, 0.1)
        S0 = np.zeros(n, dtype=np.int32); S0[2] = STATE_FAILED
        final, _, _, _ = run_until_stable(S0, A, td, tf)
        self.assertEqual(final[3], STATE_OPERATIONAL)
        self.assertEqual(final[4], STATE_OPERATIONAL)


class TestFragilityIndex(unittest.TestCase):

    def test_fi_self_counts(self):
        n = 6
        rng = np.random.default_rng(99)
        A = (rng.random((n, n)) < 0.2).astype(np.uint8)
        np.fill_diagonal(A, 0)
        fi = fragility_index(A, np.full(n, 0.5), np.full(n, 0.8))
        self.assertTrue(np.all(fi >= 1))

    def test_fi_upper_bound(self):
        n = 6
        rng = np.random.default_rng(13)
        A = (rng.random((n, n)) < 0.25).astype(np.uint8)
        np.fill_diagonal(A, 0)
        fi = fragility_index(A, np.full(n, 0.3), np.full(n, 0.5))
        self.assertTrue(np.all(fi <= n))

    def test_fi_fully_connected_zero_threshold(self):
        n = 5
        A = np.ones((n, n), dtype=np.uint8); np.fill_diagonal(A, 0)
        fi = fragility_index(A, np.zeros(n), np.zeros(n))
        self.assertTrue(np.all(fi == n))

    def test_fi_isolated_graph(self):
        n = 8
        A = np.zeros((n, n), dtype=np.uint8)
        fi = fragility_index(A, np.full(n, 0.5), np.full(n, 0.8))
        self.assertTrue(np.all(fi == 1))


class TestCascadeSize(unittest.TestCase):

    def test_all_operational(self):
        stats = cascade_size(np.zeros(10, dtype=np.int32))
        self.assertEqual(stats["n_failed"], 0)
        self.assertEqual(stats["frac_affected"], 0.0)

    def test_all_failed(self):
        stats = cascade_size(np.full(10, 2, dtype=np.int32))
        self.assertEqual(stats["n_failed"], 10)
        self.assertAlmostEqual(stats["frac_failed"], 1.0)

    def test_mixed(self):
        state = np.array([0, 1, 2, 2, 1, 0], dtype=np.int32)
        stats = cascade_size(state)
        self.assertEqual(stats["n_degraded"], 2)
        self.assertEqual(stats["n_failed"], 2)
        self.assertAlmostEqual(stats["frac_affected"], 4/6)


class TestInputValidation(unittest.TestCase):

    def test_mismatched_s0_raises(self):
        n = 5
        A = np.zeros((n, n), dtype=np.uint8)
        with self.assertRaises(ValueError):
            run_until_stable(np.zeros(n+1, dtype=np.int32), A, np.zeros(n), np.zeros(n))

    def test_non_square_A_raises(self):
        with self.assertRaises(ValueError):
            run_until_stable(
                np.zeros(5, dtype=np.int32),
                np.zeros((5, 6), dtype=np.uint8),
                np.zeros(5), np.zeros(5)
            )


if __name__ == "__main__":
    unittest.main()
