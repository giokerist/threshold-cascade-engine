"""
Monte Carlo experiment harness for the cascade propagation engine.

Runs multiple independent trials of the stochastic cascade and returns
distributional statistics.  Each trial uses a statistically-independent
Generator derived via numpy SeedSequence.spawn() (F-006 fix), ensuring
reproducibility and independence that the simple integer-offset approach
could not guarantee.

Design principles
-----------------
* No global RNG state: every trial creates its own Generator instance
  via SeedSequence.spawn() (replaces seed + trial integer offset).
* No Python loops over nodes (delegated to stochastic_propagation).
* Results are immutable numpy arrays -- no mutation between trials.
* Confidence intervals use scipy.stats.t (t-distribution, two-tailed),
  imported from shared utils.py (F-004 fix -- no more duplication).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import SeedSequence, default_rng

from .stochastic_propagation import run_until_stable_stochastic
from .propagation import STATE_DEGRADED
from .utils import confidence_interval


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloResult:
    """Immutable container for Monte Carlo experiment results.

    Attributes
    ----------
    cascade_sizes : np.ndarray, shape (trials,)
        Raw fraction of nodes affected per trial (affected = degraded + failed).
    times_to_stability : np.ndarray, shape (trials,), dtype int
        Number of steps until convergence per trial.
    mean_cascade_size : float
        Mean fraction of nodes affected across trials.
    variance_cascade_size : float
        Sample variance of cascade size across trials.
    ci_low : float
        Lower bound of 95% confidence interval for mean cascade size.
    ci_high : float
        Upper bound of 95% confidence interval for mean cascade size.
    trials : int
        Number of trials executed.
    seed : int
        Master seed used to derive per-trial seeds.
    n_nodes : int
        Number of nodes in the graph.
    k : float
        Logistic steepness used.
    consecutive_stable_steps : int
        Convergence criterion passed to each trial.
    """
    cascade_sizes: np.ndarray
    times_to_stability: np.ndarray
    mean_cascade_size: float
    variance_cascade_size: float
    ci_low: float
    ci_high: float
    trials: int
    seed: int
    n_nodes: int
    k: float
    consecutive_stable_steps: int = 3

    def summary_dict(self) -> dict:
        """Return a JSON-serialisable summary (no arrays)."""
        return {
            "trials": self.trials,
            "seed": self.seed,
            "n_nodes": self.n_nodes,
            "k": self.k,
            "consecutive_stable_steps": self.consecutive_stable_steps,
            "mean_cascade_size": self.mean_cascade_size,
            "variance_cascade_size": self.variance_cascade_size,
            "ci_95_low": self.ci_low,
            "ci_95_high": self.ci_high,
            "min_cascade_size": float(np.min(self.cascade_sizes)),
            "max_cascade_size": float(np.max(self.cascade_sizes)),
            "median_cascade_size": float(np.median(self.cascade_sizes)),
            "mean_time_to_stability": float(np.mean(self.times_to_stability)),
        }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_monte_carlo(
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    seed_node: int,
    trials: int,
    seed: int,
    k: float = 10.0,
    max_steps: int | None = None,
    consecutive_stable_steps: int = 3,
) -> MonteCarloResult:
    """Run Monte Carlo trials seeding a single node as the initial failure.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix: A[j, i] = 1 iff j -> i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,)
        Failure threshold (logistic midpoint) per node.
    seed_node : int
        Index of the node to seed as failed at t=0.
    trials : int
        Number of independent Monte Carlo trials.
    seed : int
        Master seed.  Per-trial RNGs are derived via SeedSequence.spawn()
        (F-006 fix) for guaranteed statistical independence.
    k : float, optional
        Logistic steepness parameter (default 10.0).
    max_steps : int or None, optional
        Maximum steps per trial.
    consecutive_stable_steps : int, optional
        Consecutive quiet steps required for convergence (default 3).
        Passed through to run_until_stable_stochastic.

    Returns
    -------
    MonteCarloResult
        Frozen dataclass holding raw distributions and aggregate statistics.

    Raises
    ------
    ValueError
        If ``trials < 2`` (CI computation requires at least 2 samples) or
        ``seed_node`` is out of range.
    """
    n = A.shape[0]
    if trials < 2:
        raise ValueError(f"trials must be >= 2 for CI computation; got {trials}.")
    if not (0 <= seed_node < n):
        raise ValueError(f"seed_node {seed_node} out of range [0, {n - 1}].")

    # F-006: use SeedSequence.spawn() for statistically-independent trial RNGs
    ss = SeedSequence(seed)
    trial_rngs = [default_rng(child) for child in ss.spawn(trials)]

    cascade_sizes = np.empty(trials, dtype=np.float64)
    times_to_stability = np.empty(trials, dtype=np.int64)

    S0 = np.zeros(n, dtype=np.int32)
    S0[seed_node] = 2  # STATE_FAILED

    for trial in range(trials):
        final_state, t_stable, _, _ = run_until_stable_stochastic(
            S0, A, theta_deg, theta_fail,
            k=k, rng=trial_rngs[trial],
            max_steps=max_steps,
            consecutive_stable_steps=consecutive_stable_steps,
        )

        affected = int(np.sum(final_state >= STATE_DEGRADED))
        cascade_sizes[trial] = affected / n
        times_to_stability[trial] = t_stable

    mean_cs = float(np.mean(cascade_sizes))
    var_cs = float(np.var(cascade_sizes, ddof=1))
    ci_low, ci_high = confidence_interval(cascade_sizes)   # F-004: shared util

    return MonteCarloResult(
        cascade_sizes=cascade_sizes,
        times_to_stability=times_to_stability,
        mean_cascade_size=mean_cs,
        variance_cascade_size=var_cs,
        ci_low=ci_low,
        ci_high=ci_high,
        trials=trials,
        seed=seed,
        n_nodes=n,
        k=k,
        consecutive_stable_steps=consecutive_stable_steps,
    )


def run_monte_carlo_all_seeds(
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    trials: int,
    seed: int,
    k: float = 10.0,
    max_steps: int | None = None,
    consecutive_stable_steps: int = 3,
) -> list[MonteCarloResult]:
    """Run Monte Carlo trials for every node as initial seed.

    Useful for computing a stochastic fragility index across the full graph.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix.
    theta_deg : np.ndarray, shape (n,)
        Degradation thresholds.
    theta_fail : np.ndarray, shape (n,)
        Failure thresholds.
    trials : int
        Number of trials per seed node.
    seed : int
        Master seed.  Node-level seed blocks are derived by spawning n
        child SeedSequences from the master, then spawning trials children
        from each node SeedSequence (F-006 fix).
    k : float, optional
        Logistic steepness parameter (default 10.0).
    max_steps : int or None, optional
        Maximum steps per trial.
    consecutive_stable_steps : int, optional
        Consecutive quiet steps required for convergence (default 3).

    Returns
    -------
    list of MonteCarloResult
        One result per node (length n), in node index order.
    """
    n = A.shape[0]
    # F-006: two-level SeedSequence hierarchy for full independence
    root_ss = SeedSequence(seed)
    node_sss = root_ss.spawn(n)

    results: list[MonteCarloResult] = []
    for node, node_ss in enumerate(node_sss):
        # Each node gets its own master seed derived from the root
        node_master_seed = int(node_ss.generate_state(1)[0])
        result = run_monte_carlo(
            A, theta_deg, theta_fail,
            seed_node=node,
            trials=trials,
            seed=node_master_seed,
            k=k,
            max_steps=max_steps,
            consecutive_stable_steps=consecutive_stable_steps,
        )
        results.append(result)
    return results
