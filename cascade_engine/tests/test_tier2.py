"""
Unit tests for the Tier 2 stochastic extensions.

Covers:
  - Logistic (sigmoid) function correctness
  - Stochastic propagation: monotonicity, determinism with fixed seed
  - Monte Carlo: reproducibility, statistics, CI
  - Sensitivity analysis: structure and monotonicity expectations
  - Metrics extensions: RMSE, MAPE, Spearman, CI

Tier 1 tests are NOT removed or modified.  All Tier 1 tests remain in
test_propagation.py, test_graph.py, and test_config.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.random import default_rng

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cascade_engine.stochastic_propagation import (
    sigmoid,
    propagation_step_stochastic,
    run_until_stable_stochastic,
)
from cascade_engine.monte_carlo import run_monte_carlo, confidence_interval
from cascade_engine.sensitivity import threshold_sensitivity, sensitivity_aggregate_by_perturbation
from cascade_engine.metrics import rmse, mape, spearman_correlation, confidence_interval as metrics_ci
from cascade_engine.propagation import STATE_FAILED, STATE_DEGRADED, STATE_OPERATIONAL
from cascade_engine.graph import generate_custom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(n: int) -> np.ndarray:
    return generate_custom(n, [(i, i + 1) for i in range(n - 1)])


def _uniform_thresh(n: int, deg: float, fail: float):
    return np.full(n, deg, dtype=np.float64), np.full(n, fail, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sigmoid / logistic tests
# ---------------------------------------------------------------------------


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_zero_is_half(self):
        """sigmoid(0) must equal 0.5 exactly."""
        result = sigmoid(np.array([0.0]))
        self.assertAlmostEqual(float(result[0]), 0.5, places=12)

    def test_sigmoid_positive_above_half(self):
        """sigmoid(x) > 0.5 for x > 0."""
        x = np.array([0.1, 1.0, 5.0, 100.0])
        self.assertTrue(np.all(sigmoid(x) > 0.5))

    def test_sigmoid_negative_below_half(self):
        """sigmoid(x) < 0.5 for x < 0."""
        x = np.array([-0.1, -1.0, -5.0, -100.0])
        self.assertTrue(np.all(sigmoid(x) < 0.5))

    def test_sigmoid_output_in_unit_interval(self):
        """sigmoid output must be strictly in (0, 1)."""
        x = np.linspace(-500, 500, 1000)
        out = sigmoid(x)
        self.assertTrue(np.all(out > 0.0))
        self.assertTrue(np.all(out < 1.0))

    def test_sigmoid_large_positive_near_one(self):
        """sigmoid(large) ≈ 1."""
        result = float(sigmoid(np.array([1000.0]))[0])
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_sigmoid_large_negative_near_zero(self):
        """sigmoid(-large) ≈ 0."""
        result = float(sigmoid(np.array([-1000.0]))[0])
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_sigmoid_high_k_approximates_step(self):
        """With very high k, logistic probability at F > theta ≈ 1."""
        k = 1000.0
        theta_fail = np.array([0.5])
        F_above = np.array([0.6])
        F_below = np.array([0.4])
        p_above = float(sigmoid(k * (F_above - theta_fail))[0])
        p_below = float(sigmoid(k * (F_below - theta_fail))[0])
        self.assertGreater(p_above, 0.999)
        self.assertLess(p_below, 0.001)


# ---------------------------------------------------------------------------
# Stochastic propagation tests
# ---------------------------------------------------------------------------


class TestStochasticMonotonicity(unittest.TestCase):

    def test_states_never_decrease_chain(self):
        """States must be monotonically non-decreasing across all steps."""
        n = 10
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        rng = default_rng(42)
        _, _, history, _ = run_until_stable_stochastic(S0, A, td, tf, k=5.0, rng=rng)
        for t in range(history.shape[0] - 1):
            self.assertTrue(
                np.all(history[t + 1] >= history[t]),
                f"Monotonicity violated at step {t}",
            )

    def test_states_never_decrease_dense_graph(self):
        """Random dense graph: states must not decrease."""
        rng_gen = default_rng(7)
        n = 25
        A = (rng_gen.random((n, n)) < 0.3).astype(np.uint8)
        np.fill_diagonal(A, 0)
        td, tf = _uniform_thresh(n, 0.2, 0.5)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        rng = default_rng(99)
        _, _, history, _ = run_until_stable_stochastic(S0, A, td, tf, k=8.0, rng=rng)
        for t in range(history.shape[0] - 1):
            self.assertTrue(np.all(history[t + 1] >= history[t]))

    def test_deterministic_with_fixed_seed(self):
        """Same seed must produce identical final states across two calls."""
        n = 15
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        final1, _, _, _ = run_until_stable_stochastic(
            S0, A, td, tf, k=5.0, rng=default_rng(42)
        )
        final2, _, _, _ = run_until_stable_stochastic(
            S0, A, td, tf, k=5.0, rng=default_rng(42)
        )
        np.testing.assert_array_equal(final1, final2)

    def test_different_seeds_may_differ(self):
        """Different seeds should produce different outcomes in a borderline scenario."""
        # Use a star graph: node 0 (hub) seeded, nodes 1..n-1 depend on it.
        # With F=1.0 and theta_fail=0.5, p_fail = sigmoid(k*(1-0.5)).
        # At k=1.0 this gives sigmoid(0.5) ≈ 0.622 — genuinely stochastic.
        n = 6
        # Star: node 0 -> all others
        edges = [(0, i) for i in range(1, n)]
        A = generate_custom(n, edges)
        # Use low k so the logistic is in its probabilistic regime
        td = np.full(n, 0.3)
        tf = np.full(n, 0.5)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED   # hub fails; leaves face F=1.0, p≈0.62
        results = []
        for seed in range(40):
            final, _, _, _ = run_until_stable_stochastic(
                S0, A, td, tf, k=1.0, rng=default_rng(seed)
            )
            results.append(tuple(final.tolist()))
        # Should see more than one distinct outcome
        self.assertGreater(len(set(results)), 1)

    def test_isolated_graph_no_propagation(self):
        """Isolated nodes: only the seeded node changes state."""
        n = 10
        A = np.zeros((n, n), dtype=np.uint8)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        S0 = np.zeros(n, dtype=np.int32)
        S0[3] = STATE_FAILED
        final, steps, _, _ = run_until_stable_stochastic(
            S0, A, td, tf, k=5.0, rng=default_rng(0)
        )
        self.assertEqual(final[3], STATE_FAILED)
        for i in range(n):
            if i != 3:
                self.assertEqual(final[i], STATE_OPERATIONAL)
        self.assertEqual(steps, 0)

    def test_max_steps_terminates(self):
        """max_steps must prevent infinite loops."""
        n = 10
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.1, 0.15)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, steps, _, _ = run_until_stable_stochastic(
                S0, A, td, tf, k=3.0, rng=default_rng(0), max_steps=2
            )
        self.assertLessEqual(steps, 2)

    def test_high_k_matches_deterministic(self):
        """At very high k, stochastic results should match deterministic."""
        from cascade_engine.propagation import run_until_stable
        n = 12
        A = _chain(n)
        td = np.full(n, 0.4)
        tf = np.full(n, 0.5)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        det_final, _, _, _ = run_until_stable(S0, A, td, tf)
        # With k=1000, logistic is effectively a step function
        stoch_final, _, _, _ = run_until_stable_stochastic(
            S0, A, td, tf, k=1000.0, rng=default_rng(0)
        )
        np.testing.assert_array_equal(stoch_final, det_final)

    def test_invalid_k_raises(self):
        n = 5
        A = np.zeros((n, n), dtype=np.uint8)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        S0 = np.zeros(n, dtype=np.int32)
        with self.assertRaises(ValueError):
            run_until_stable_stochastic(S0, A, td, tf, k=-1.0, rng=default_rng(0))

    def test_shape_mismatch_raises(self):
        n = 5
        A = np.zeros((n, n), dtype=np.uint8)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        with self.assertRaises(ValueError):
            run_until_stable_stochastic(
                np.zeros(n + 1, dtype=np.int32), A, td, tf, k=5.0, rng=default_rng(0)
            )


class TestStochasticStep(unittest.TestCase):

    def test_step_output_in_valid_states(self):
        """All output states must be in {0, 1, 2}."""
        n = 20
        rng = default_rng(0)
        A = (rng.random((n, n)) < 0.2).astype(np.uint8)
        np.fill_diagonal(A, 0)
        S = rng.integers(0, 3, size=n, dtype=np.int32)
        D = A.sum(axis=0).astype(np.float64)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        S_next = propagation_step_stochastic(S, A, td, tf, D, k=5.0, rng=default_rng(1))
        self.assertTrue(np.all((S_next >= 0) & (S_next <= 2)))

    def test_step_monotone(self):
        """Output of one step must be >= input."""
        n = 15
        A = _chain(n)
        S = np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        D = A.sum(axis=0).astype(np.float64)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        S_next = propagation_step_stochastic(S, A, td, tf, D, k=5.0, rng=default_rng(0))
        self.assertTrue(np.all(S_next >= S))


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------


class TestMonteCarlo(unittest.TestCase):

    def _small_setup(self):
        n = 10
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        return A, td, tf, n

    def test_reproducibility_fixed_seed(self):
        """Same master seed must produce identical distributions."""
        A, td, tf, n = self._small_setup()
        r1 = run_monte_carlo(A, td, tf, seed_node=0, trials=20, seed=42, k=5.0)
        r2 = run_monte_carlo(A, td, tf, seed_node=0, trials=20, seed=42, k=5.0)
        np.testing.assert_array_equal(r1.cascade_sizes, r2.cascade_sizes)
        np.testing.assert_array_equal(r1.times_to_stability, r2.times_to_stability)

    def test_different_seeds_differ(self):
        """Different master seeds should produce different distributions.

        Uses a sparse random graph with low k and high thresholds so that
        genuine stochasticity is present in the outcomes (not all trials
        collapse to the same deterministic full-cascade result).
        """
        from cascade_engine.graph import generate_erdos_renyi
        n = 20
        A = generate_erdos_renyi(n, p=0.1, seed=7)
        td = np.full(n, 0.4)
        tf = np.full(n, 0.7)
        r1 = run_monte_carlo(A, td, tf, seed_node=0, trials=50, seed=1, k=2.0)
        r2 = run_monte_carlo(A, td, tf, seed_node=0, trials=50, seed=9999, k=2.0)
        # Different seeds must produce different raw cascade_sizes arrays
        self.assertFalse(np.array_equal(r1.cascade_sizes, r2.cascade_sizes))

    def test_cascade_sizes_in_unit_interval(self):
        """All cascade size fractions must be in [0, 1]."""
        A, td, tf, n = self._small_setup()
        r = run_monte_carlo(A, td, tf, seed_node=0, trials=30, seed=0, k=5.0)
        self.assertTrue(np.all(r.cascade_sizes >= 0.0))
        self.assertTrue(np.all(r.cascade_sizes <= 1.0))

    def test_mean_within_ci(self):
        """Sample mean must lie within its own 95% CI."""
        A, td, tf, n = self._small_setup()
        r = run_monte_carlo(A, td, tf, seed_node=0, trials=50, seed=7, k=5.0)
        self.assertLessEqual(r.ci_low, r.mean_cascade_size)
        self.assertGreaterEqual(r.ci_high, r.mean_cascade_size)

    def test_variance_non_negative(self):
        A, td, tf, n = self._small_setup()
        r = run_monte_carlo(A, td, tf, seed_node=0, trials=20, seed=0, k=5.0)
        self.assertGreaterEqual(r.variance_cascade_size, 0.0)

    def test_trials_count_correct(self):
        A, td, tf, n = self._small_setup()
        r = run_monte_carlo(A, td, tf, seed_node=0, trials=25, seed=0, k=5.0)
        self.assertEqual(len(r.cascade_sizes), 25)
        self.assertEqual(len(r.times_to_stability), 25)

    def test_too_few_trials_raises(self):
        A, td, tf, n = self._small_setup()
        with self.assertRaises(ValueError):
            run_monte_carlo(A, td, tf, seed_node=0, trials=1, seed=0, k=5.0)

    def test_invalid_seed_node_raises(self):
        A, td, tf, n = self._small_setup()
        with self.assertRaises(ValueError):
            run_monte_carlo(A, td, tf, seed_node=n + 5, trials=10, seed=0, k=5.0)

    def test_summary_dict_serialisable(self):
        """summary_dict must be JSON-serialisable."""
        import json
        A, td, tf, n = self._small_setup()
        r = run_monte_carlo(A, td, tf, seed_node=0, trials=10, seed=0, k=5.0)
        serialised = json.dumps(r.summary_dict())
        self.assertIsInstance(serialised, str)

    def test_isolated_seed_node_always_affected(self):
        """Seeded node must always be in cascade (FI >= 1/n)."""
        n = 8
        A = np.zeros((n, n), dtype=np.uint8)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        r = run_monte_carlo(A, td, tf, seed_node=0, trials=20, seed=0, k=5.0)
        min_frac = 1.0 / n
        self.assertTrue(
            np.all(r.cascade_sizes >= min_frac - 1e-9),
            f"Some trials had cascade_size < 1/n: min={np.min(r.cascade_sizes)}"
        )


# ---------------------------------------------------------------------------
# Confidence interval tests
# ---------------------------------------------------------------------------


class TestConfidenceInterval(unittest.TestCase):

    def test_ci_contains_true_mean_normal(self):
        """95% CI should contain the true mean for a normal sample."""
        rng = np.random.default_rng(42)
        # 1000 trials: 95% CI should contain true mean ~95% of the time
        # We just check one instance here for determinism
        samples = rng.normal(loc=0.5, scale=0.1, size=100)
        lo, hi = confidence_interval(samples, confidence=0.95)
        self.assertLess(lo, 0.5 + 0.05)   # reasonable bounds
        self.assertGreater(hi, 0.5 - 0.05)
        self.assertLess(lo, hi)

    def test_ci_symmetric_around_mean(self):
        """CI should be roughly symmetric around the sample mean."""
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=0.5, scale=0.05, size=200)
        lo, hi = confidence_interval(samples, confidence=0.95)
        mean = float(np.mean(samples))
        self.assertAlmostEqual(mean - lo, hi - mean, places=5)

    def test_wider_ci_lower_confidence(self):
        """Lower confidence level must give narrower CI."""
        rng = np.random.default_rng(1)
        samples = rng.normal(0.5, 0.1, 100)
        lo90, hi90 = confidence_interval(samples, confidence=0.90)
        lo99, hi99 = confidence_interval(samples, confidence=0.99)
        self.assertLess(hi90 - lo90, hi99 - lo99)

    def test_ci_narrows_with_more_samples(self):
        """More samples should give a narrower CI."""
        rng = np.random.default_rng(2)
        small = rng.normal(0.5, 0.1, 20)
        large = rng.normal(0.5, 0.1, 500)
        lo_s, hi_s = confidence_interval(small)
        lo_l, hi_l = confidence_interval(large)
        self.assertLess(hi_l - lo_l, hi_s - lo_s)

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            confidence_interval(np.array([0.5]))

    def test_invalid_confidence_raises(self):
        with self.assertRaises(ValueError):
            confidence_interval(np.array([0.5, 0.6]), confidence=1.5)


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------


class TestSensitivity(unittest.TestCase):

    def _setup(self):
        n = 8
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.5)
        return A, td, tf, n

    def test_returns_correct_number_of_records(self):
        A, td, tf, n = self._setup()
        perturbations = [-0.1, 0.0, 0.1]
        seed_nodes = [0, 3]
        points = threshold_sensitivity(
            A, td, tf, perturbation_values=perturbations, seed_nodes=seed_nodes
        )
        self.assertEqual(len(points), len(perturbations) * len(seed_nodes))

    def test_zero_perturbation_matches_baseline(self):
        """delta=0 should match an unperturbed deterministic run."""
        from cascade_engine.propagation import run_until_stable
        from cascade_engine.propagation import STATE_DEGRADED as SD
        n = 8
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.5)
        points = threshold_sensitivity(
            A, td, tf, perturbation_values=[0.0], seed_nodes=[0], mode="deterministic"
        )
        self.assertEqual(len(points), 1)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        final, _, _, _ = run_until_stable(S0, A, td, tf)
        expected_frac = float(np.sum(final >= SD)) / n
        self.assertAlmostEqual(points[0].cascade_size_fraction, expected_frac, places=9)

    def test_higher_threshold_smaller_or_equal_cascade(self):
        """Increasing theta_fail should not increase (or only slightly increase) cascade."""
        n = 10
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.5)
        perturbations = [-0.2, -0.1, 0.0, 0.1, 0.2]
        points = threshold_sensitivity(
            A, td, tf,
            perturbation_values=perturbations,
            seed_nodes=[0],
            mode="deterministic",
        )
        cascade_sizes = [p.cascade_size_fraction for p in points]
        # Cascade should be non-increasing as perturbation increases
        # (higher threshold = harder to fail = smaller cascade)
        for i in range(len(cascade_sizes) - 1):
            self.assertGreaterEqual(
                cascade_sizes[i] + 1e-9, cascade_sizes[i + 1],
                f"Cascade increased from δ={perturbations[i]} to δ={perturbations[i+1]}: "
                f"{cascade_sizes[i]} -> {cascade_sizes[i+1]}"
            )

    def test_stochastic_mode_produces_std(self):
        """Stochastic mode must produce non-negative std."""
        n = 8
        rng_gen = default_rng(0)
        A = (rng_gen.random((n, n)) < 0.3).astype(np.uint8)
        np.fill_diagonal(A, 0)
        td, tf = _uniform_thresh(n, 0.35, 0.55)
        points = threshold_sensitivity(
            A, td, tf,
            perturbation_values=[0.0],
            seed_nodes=[0],
            mode="stochastic",
            stochastic_trials=10,
            stochastic_k=5.0,
            seed=0,
        )
        self.assertGreaterEqual(points[0].std_cascade_size, 0.0)
        self.assertEqual(points[0].n_trials, 10)

    def test_stochastic_reproducible(self):
        """Same seed must produce same stochastic sensitivity results."""
        n = 8
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.5)
        kwargs = dict(
            perturbation_values=[0.0, 0.1],
            seed_nodes=[0],
            mode="stochastic",
            stochastic_trials=5,
            stochastic_k=5.0,
            seed=99,
        )
        p1 = threshold_sensitivity(A, td, tf, **kwargs)
        p2 = threshold_sensitivity(A, td, tf, **kwargs)
        for pt1, pt2 in zip(p1, p2):
            self.assertAlmostEqual(
                pt1.cascade_size_fraction, pt2.cascade_size_fraction, places=12
            )

    def test_clamp_mode_stored_on_point(self):
        """SensitivityPoint.clamp_mode must reflect the mode used."""
        A, td, tf, n = self._setup()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pts_floor = threshold_sensitivity(
                A, td, tf, [-0.5], seed_nodes=[0],
                mode="deterministic", clamp_mode="fail_floored_at_deg",
            )
            pts_drag = threshold_sensitivity(
                A, td, tf, [-0.5], seed_nodes=[0],
                mode="deterministic", clamp_mode="fail_drags_deg",
            )
        self.assertEqual(pts_floor[0].clamp_mode, "fail_floored_at_deg")
        self.assertEqual(pts_drag[0].clamp_mode, "fail_drags_deg")

    def test_fail_floored_keeps_theta_deg_constant(self):
        """fail_floored_at_deg cascade <= fail_drags_deg cascade for large negative delta."""
        import warnings
        A, td, tf, n = self._setup()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pts_floor = threshold_sensitivity(
                A, td, tf, [-0.5], seed_nodes=[0],
                mode="deterministic", clamp_mode="fail_floored_at_deg",
            )
            pts_drag = threshold_sensitivity(
                A, td, tf, [-0.5], seed_nodes=[0],
                mode="deterministic", clamp_mode="fail_drags_deg",
            )
        # Dragging theta_deg down must produce >= cascade size vs flooring theta_fail
        self.assertGreaterEqual(
            pts_drag[0].cascade_size_fraction,
            pts_floor[0].cascade_size_fraction,
        )

    def test_invalid_clamp_mode_raises(self):
        """Unknown clamp_mode must raise ValueError."""
        A, td, tf, n = self._setup()
        with self.assertRaises(ValueError):
            threshold_sensitivity(
                A, td, tf, [-0.1], seed_nodes=[0],
                mode="deterministic", clamp_mode="bad_mode",
            )

    def test_consecutive_stable_steps_accepted(self):
        """consecutive_stable_steps must be accepted without error in stochastic mode."""
        A, td, tf, n = self._setup()
        pts = threshold_sensitivity(
            A, td, tf, [0.0], seed_nodes=[0],
            mode="stochastic", stochastic_trials=5, seed=0,
            consecutive_stable_steps=5,
        )
        self.assertEqual(len(pts), 1)
        self.assertGreaterEqual(pts[0].cascade_size_fraction, 0.0)

    def test_aggregate_by_perturbation_length(self):
        A, td, tf, n = self._setup()
        perturbations = [-0.1, 0.0, 0.1]
        points = threshold_sensitivity(
            A, td, tf,
            perturbation_values=perturbations,
            seed_nodes=[0, 1, 2],
            mode="deterministic",
        )
        agg = sensitivity_aggregate_by_perturbation(points)
        self.assertEqual(len(agg), 3)

    def test_invalid_mode_raises(self):
        A, td, tf, n = self._setup()
        with self.assertRaises(ValueError):
            threshold_sensitivity(A, td, tf, [0.0], mode="invalid_mode")

    def test_large_negative_delta_warns_theta_deg_clamped(self):
        """A UserWarning must fire when large negative delta triggers clamping.

        Tests both clamp_mode values:
        - 'fail_floored_at_deg' (default): warning mentions floor/theta_deg
        - 'fail_drags_deg' (legacy):       warning mentions theta_deg dragged
        """
        import warnings
        A, td, tf, n = self._setup()
        # delta=-0.5 drops theta_fail to 0.0, which is below theta_deg=0.3

        # Default mode: fail_floored_at_deg
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            threshold_sensitivity(
                A, td, tf, [-0.5], seed_nodes=[0], mode="deterministic",
                clamp_mode="fail_floored_at_deg",
            )
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0,
                           "Expected a UserWarning with fail_floored_at_deg clamping")
        self.assertIn("theta_deg", str(user_warnings[0].message))

        # Legacy mode: fail_drags_deg
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            threshold_sensitivity(
                A, td, tf, [-0.5], seed_nodes=[0], mode="deterministic",
                clamp_mode="fail_drags_deg",
            )
        uw2 = [w for w in caught2 if issubclass(w.category, UserWarning)]
        self.assertGreater(len(uw2), 0,
                           "Expected a UserWarning with fail_drags_deg clamping")
        self.assertIn("theta_deg", str(uw2[0].message))

    def test_small_delta_no_theta_deg_warning(self):
        """No warning when theta_fail stays above theta_deg after perturbation."""
        import warnings
        A, td, tf, n = self._setup()
        # delta=+0.1 raises theta_fail -- no clamping needed
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            threshold_sensitivity(A, td, tf, [0.1], seed_nodes=[0], mode="deterministic")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 0,
                         "No warning expected when theta_fail stays above theta_deg")


# ---------------------------------------------------------------------------
# Extended metrics tests
# ---------------------------------------------------------------------------


class TestRMSE(unittest.TestCase):

    def test_identical_arrays_zero_rmse(self):
        x = np.array([0.1, 0.5, 0.9])
        self.assertAlmostEqual(rmse(x, x), 0.0, places=12)

    def test_known_value(self):
        pred = np.array([3.0, 4.0])
        obs = np.array([1.0, 2.0])
        # residuals = [2, 2]; RMSE = sqrt(mean([4,4])) = 2.0
        self.assertAlmostEqual(rmse(pred, obs), 2.0, places=12)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            rmse(np.array([]), np.array([]))

    def test_rmse_non_negative(self):
        rng = np.random.default_rng(0)
        x = rng.random(50)
        y = rng.random(50)
        self.assertGreaterEqual(rmse(x, y), 0.0)


class TestMAPE(unittest.TestCase):

    def test_identical_arrays_zero_mape(self):
        x = np.array([0.1, 0.5, 0.9])
        self.assertAlmostEqual(mape(x, x), 0.0, places=10)

    def test_known_value(self):
        pred = np.array([1.1, 2.2])
        obs = np.array([1.0, 2.0])
        # |0.1/1.0| + |0.2/2.0| = 0.1 + 0.1 = 0.2; mean = 0.1; * 100 = 10%
        self.assertAlmostEqual(mape(pred, obs), 10.0, places=10)

    def test_zero_observed_no_crash(self):
        """eps prevents division by zero."""
        pred = np.array([0.1, 0.2])
        obs = np.array([0.0, 0.2])
        result = mape(pred, obs)
        self.assertGreaterEqual(result, 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mape(np.array([1.0]), np.array([1.0, 2.0]))

    def test_mape_non_negative(self):
        rng = np.random.default_rng(5)
        x = rng.random(50)
        y = rng.random(50)
        self.assertGreaterEqual(mape(x, y), 0.0)


class TestSpearman(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        x = np.arange(10, dtype=np.float64)
        result = spearman_correlation(x, x)
        self.assertAlmostEqual(result["rho"], 1.0, places=10)

    def test_perfect_negative_correlation(self):
        x = np.arange(10, dtype=np.float64)
        result = spearman_correlation(x, x[::-1])
        self.assertAlmostEqual(result["rho"], -1.0, places=10)

    def test_rho_in_range(self):
        rng = np.random.default_rng(0)
        x = rng.random(50)
        y = rng.random(50)
        result = spearman_correlation(x, y)
        self.assertGreaterEqual(result["rho"], -1.0)
        self.assertLessEqual(result["rho"], 1.0)

    def test_p_value_range(self):
        rng = np.random.default_rng(1)
        x = rng.random(30)
        y = rng.random(30)
        result = spearman_correlation(x, y)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            spearman_correlation(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_elements_raises(self):
        with self.assertRaises(ValueError):
            spearman_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_returns_dict_keys(self):
        x = np.arange(5, dtype=np.float64)
        result = spearman_correlation(x, x)
        self.assertIn("rho", result)
        self.assertIn("p_value", result)


class TestMetricsCI(unittest.TestCase):
    """Tests for confidence_interval in metrics.py (independent of monte_carlo.py)."""

    def test_ci_low_less_than_high(self):
        samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        lo, hi = metrics_ci(samples)
        self.assertLess(lo, hi)

    def test_ci_contains_sample_mean(self):
        rng = np.random.default_rng(10)
        samples = rng.normal(0.5, 0.1, 100)
        lo, hi = metrics_ci(samples, confidence=0.95)
        mean = float(np.mean(samples))
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            metrics_ci(np.array([0.5]))

    def test_invalid_confidence_raises(self):
        with self.assertRaises(ValueError):
            metrics_ci(np.array([0.5, 0.6, 0.7]), confidence=0.0)


# ---------------------------------------------------------------------------
# Integration: Tier 1 unchanged by Tier 2 imports
# ---------------------------------------------------------------------------


class TestTier1Unchanged(unittest.TestCase):
    """Smoke tests ensuring Tier 1 deterministic functions still work correctly
    after the Tier 2 additions to metrics.py."""

    def test_fragility_index_still_works(self):
        from cascade_engine.metrics import fragility_index
        n = 6
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.4, 0.5)
        fi = fragility_index(A, td, tf)
        self.assertEqual(len(fi), n)
        self.assertTrue(np.all(fi >= 1))

    def test_cascade_size_still_works(self):
        from cascade_engine.metrics import cascade_size
        state = np.array([0, 1, 2, 2, 0], dtype=np.int32)
        stats = cascade_size(state)
        self.assertEqual(stats["n_failed"], 2)
        self.assertEqual(stats["n_degraded"], 1)

    def test_deterministic_propagation_unchanged(self):
        from cascade_engine.propagation import run_until_stable
        n = 5
        A = _chain(n)
        td = np.zeros(n)
        tf = np.zeros(n)
        S0 = np.zeros(n, dtype=np.int32)
        S0[0] = STATE_FAILED
        final, _, _, _ = run_until_stable(S0, A, td, tf)
        self.assertTrue(np.all(final == STATE_FAILED))


class TestUtils(unittest.TestCase):
    """Tests for cascade_engine.utils."""

    def test_confidence_interval_shared(self):
        """utils.confidence_interval and metrics.confidence_interval must agree."""
        from cascade_engine.utils import confidence_interval as utils_ci
        from cascade_engine.metrics import confidence_interval as metrics_ci
        samples = np.random.default_rng(0).normal(0, 1, 100)
        lo1, hi1 = utils_ci(samples)
        lo2, hi2 = metrics_ci(samples)
        self.assertAlmostEqual(lo1, lo2, places=12)
        self.assertAlmostEqual(hi1, hi2, places=12)

    def test_make_trial_rngs_count(self):
        """make_trial_rngs must return exactly n_trials Generators."""
        from cascade_engine.utils import make_trial_rngs
        rngs = make_trial_rngs(master_seed=42, n_trials=10)
        self.assertEqual(len(rngs), 10)

    def test_make_trial_rngs_independent(self):
        """Different trial RNGs must produce different random draws."""
        from cascade_engine.utils import make_trial_rngs
        rngs = make_trial_rngs(master_seed=0, n_trials=5)
        draws = [rng.random() for rng in rngs]
        # All distinct (probability of collision in float64 is negligible)
        self.assertEqual(len(set(draws)), 5)

    def test_make_trial_rngs_reproducible(self):
        """Same master_seed must produce same sequence of draws."""
        from cascade_engine.utils import make_trial_rngs
        rngs1 = make_trial_rngs(42, 5)
        rngs2 = make_trial_rngs(42, 5)
        for r1, r2 in zip(rngs1, rngs2):
            self.assertEqual(r1.random(), r2.random())

    def test_make_node_trial_rngs_shape(self):
        """make_node_trial_rngs must return (n_nodes, n_trials) grid."""
        from cascade_engine.utils import make_node_trial_rngs
        grid = make_node_trial_rngs(0, n_nodes=4, n_trials=6)
        self.assertEqual(len(grid), 4)
        for row in grid:
            self.assertEqual(len(row), 6)


class TestConsecutiveStableSteps(unittest.TestCase):
    """Tests for the F-001 fix: multi-step convergence criterion."""

    def _setup(self):
        n = 8
        A = _chain(n)
        td, tf = _uniform_thresh(n, 0.3, 0.6)
        return A, td, tf, n

    def test_invalid_consecutive_steps_raises(self):
        """consecutive_stable_steps < 1 must raise ValueError."""
        from cascade_engine.stochastic_propagation import run_until_stable_stochastic
        A, td, tf, n = self._setup()
        rng = np.random.default_rng(0)
        S0 = np.zeros(n, dtype=np.int32); S0[0] = 2
        with self.assertRaises(ValueError):
            run_until_stable_stochastic(S0, A, td, tf, k=5.0, rng=rng,
                                        consecutive_stable_steps=0)

    def test_single_step_vs_multi_step(self):
        """3-step convergence must return cascade_size >= 1-step convergence."""
        from cascade_engine.stochastic_propagation import run_until_stable_stochastic
        A, td, tf, n = self._setup()
        S0 = np.zeros(n, dtype=np.int32); S0[0] = 2

        rng1 = np.random.default_rng(5)
        final1, _, _, _ = run_until_stable_stochastic(S0, A, td, tf, k=3.0, rng=rng1,
                                                       consecutive_stable_steps=1)
        rng3 = np.random.default_rng(5)
        final3, _, _, _ = run_until_stable_stochastic(S0, A, td, tf, k=3.0, rng=rng3,
                                                       consecutive_stable_steps=3)
        # With more patience, we get >= affected nodes (never fewer due to monotonicity)
        self.assertGreaterEqual(
            int(np.sum(final3 >= 1)),
            int(np.sum(final1 >= 1)),
        )

    def test_monte_carlo_accepts_consecutive_stable_steps(self):
        """run_monte_carlo must forward consecutive_stable_steps without error."""
        from cascade_engine.monte_carlo import run_monte_carlo
        A, td, tf, n = self._setup()
        result = run_monte_carlo(A, td, tf, seed_node=0, trials=5, seed=0,
                                 k=5.0, consecutive_stable_steps=5)
        self.assertEqual(result.consecutive_stable_steps, 5)
        self.assertEqual(len(result.cascade_sizes), 5)


if __name__ == "__main__":
    unittest.main()
