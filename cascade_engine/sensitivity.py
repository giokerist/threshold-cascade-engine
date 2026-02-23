"""
Sensitivity analysis module for the cascade propagation engine.

Quantifies how cascade size changes as failure thresholds are perturbed.
Supports both deterministic and stochastic propagation modes.

Design
------
* Pure functions: no global state, no side effects.
* All results are returned as structured dicts ready for CSV export.
* Stochastic mode averages over multiple trials per perturbation level.
* Perturbations are applied to a copy of the threshold array — originals
  are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng

from propagation import run_until_stable, STATE_DEGRADED
from stochastic_propagation import run_until_stable_stochastic


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityPoint:
    """Result for a single perturbation level and seed node.

    Attributes
    ----------
    perturbation : float
        The delta applied: theta_fail += perturbation.
    seed_node : int
        Which node was seeded as the initial failure.
    cascade_size_fraction : float
        Fraction of nodes affected (degraded + failed) at convergence.
    time_to_stability : int
        Steps until convergence.
    n_trials : int
        1 for deterministic, >= 2 for stochastic (averaged).
    std_cascade_size : float
        Standard deviation across trials (0.0 for deterministic).
    """
    perturbation: float
    seed_node: int
    cascade_size_fraction: float
    time_to_stability: float          # float to allow averages in stochastic mode
    n_trials: int
    std_cascade_size: float

    def to_dict(self) -> dict:
        return {
            "perturbation": self.perturbation,
            "seed_node": self.seed_node,
            "cascade_size_fraction": self.cascade_size_fraction,
            "time_to_stability": self.time_to_stability,
            "n_trials": self.n_trials,
            "std_cascade_size": self.std_cascade_size,
        }


# ---------------------------------------------------------------------------
# Core sensitivity runner
# ---------------------------------------------------------------------------


def threshold_sensitivity(
    A: np.ndarray,
    base_theta_deg: np.ndarray,
    base_theta_fail: np.ndarray,
    perturbation_values: list[float],
    seed_nodes: list[int] | None = None,
    mode: str = "deterministic",
    stochastic_trials: int = 30,
    stochastic_k: float = 10.0,
    seed: int = 0,
) -> list[SensitivityPoint]:
    """Run threshold sensitivity analysis.

    For each perturbation δ in ``perturbation_values``, the failure threshold
    array is adjusted:

        theta_fail_perturbed = clip(base_theta_fail + δ, 0, 1)

    The degradation threshold is clipped to remain <= the (possibly perturbed)
    failure threshold element-wise.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix: A[j, i] = 1 iff j → i.
    base_theta_deg : np.ndarray, shape (n,)
        Baseline degradation thresholds.
    base_theta_fail : np.ndarray, shape (n,)
        Baseline failure thresholds.
    perturbation_values : list of float
        Delta values to sweep (e.g., [-0.2, -0.1, 0.0, 0.1, 0.2]).
    seed_nodes : list of int or None, optional
        Which nodes to use as initial failures.  Defaults to all nodes.
    mode : {"deterministic", "stochastic"}, optional
        Propagation mode (default "deterministic").
    stochastic_trials : int, optional
        Number of Monte Carlo trials per (perturbation, seed_node) combination
        when mode == "stochastic" (default 30).
    stochastic_k : float, optional
        Logistic steepness for stochastic mode (default 10.0).
    seed : int, optional
        Master seed for stochastic trials (default 0).  Seeds are derived as
        ``seed + perturbation_idx * n_nodes * stochastic_trials
          + seed_node_idx * stochastic_trials + trial``.

    Returns
    -------
    list of SensitivityPoint
        One record per (perturbation, seed_node) combination, in order:
        outer loop = perturbations, inner loop = seed_nodes.

    Raises
    ------
    ValueError
        If mode is not "deterministic" or "stochastic".
    """
    if mode not in ("deterministic", "stochastic"):
        raise ValueError(f"mode must be 'deterministic' or 'stochastic'; got {mode!r}.")

    n = A.shape[0]
    if seed_nodes is None:
        seed_nodes = list(range(n))

    results: list[SensitivityPoint] = []

    for p_idx, delta in enumerate(perturbation_values):
        # Build perturbed thresholds — never mutate originals
        theta_fail_p = np.clip(base_theta_fail + delta, 0.0, 1.0)
        # Degradation must not exceed failure threshold
        theta_deg_p = np.minimum(base_theta_deg, theta_fail_p)

        for sn_idx, seed_node in enumerate(seed_nodes):
            S0 = np.zeros(n, dtype=np.int32)
            S0[seed_node] = 2  # STATE_FAILED

            if mode == "deterministic":
                final_state, t_stable, _, _ = run_until_stable(
                    S0, A, theta_deg_p, theta_fail_p
                )
                affected = int(np.sum(final_state >= STATE_DEGRADED))
                cs_frac = affected / n
                results.append(SensitivityPoint(
                    perturbation=delta,
                    seed_node=seed_node,
                    cascade_size_fraction=cs_frac,
                    time_to_stability=float(t_stable),
                    n_trials=1,
                    std_cascade_size=0.0,
                ))

            else:  # stochastic
                trial_cs = np.empty(stochastic_trials, dtype=np.float64)
                trial_ts = np.empty(stochastic_trials, dtype=np.float64)

                for trial in range(stochastic_trials):
                    # Isolated per-trial seed derived from master seed
                    trial_seed = (
                        seed
                        + p_idx * n * stochastic_trials
                        + sn_idx * stochastic_trials
                        + trial
                    )
                    trial_rng = default_rng(trial_seed)

                    final_state, t_stable, _, _ = run_until_stable_stochastic(
                        S0, A, theta_deg_p, theta_fail_p,
                        k=stochastic_k, rng=trial_rng,
                    )
                    affected = int(np.sum(final_state >= STATE_DEGRADED))
                    trial_cs[trial] = affected / n
                    trial_ts[trial] = t_stable

                results.append(SensitivityPoint(
                    perturbation=delta,
                    seed_node=seed_node,
                    cascade_size_fraction=float(np.mean(trial_cs)),
                    time_to_stability=float(np.mean(trial_ts)),
                    n_trials=stochastic_trials,
                    std_cascade_size=float(np.std(trial_cs, ddof=1)),
                ))

    return results


def sensitivity_to_records(points: list[SensitivityPoint]) -> list[dict]:
    """Convert a list of SensitivityPoints to a list of dicts for CSV export.

    Parameters
    ----------
    points : list of SensitivityPoint

    Returns
    -------
    list of dict
        Each dict has keys: perturbation, seed_node, cascade_size_fraction,
        time_to_stability, n_trials, std_cascade_size.
    """
    return [p.to_dict() for p in points]


def sensitivity_aggregate_by_perturbation(
    points: list[SensitivityPoint],
) -> list[dict]:
    """Aggregate sensitivity results by perturbation level.

    For each unique perturbation delta, compute the mean and std of cascade
    size across all seed nodes.

    Parameters
    ----------
    points : list of SensitivityPoint

    Returns
    -------
    list of dict
        Sorted by perturbation.  Each dict has keys:
        perturbation, mean_cascade_size, std_cascade_size, n_seed_nodes.
    """
    from collections import defaultdict
    grouped: dict[float, list[float]] = defaultdict(list)
    for p in points:
        grouped[p.perturbation].append(p.cascade_size_fraction)

    agg = []
    for delta in sorted(grouped.keys()):
        vals = np.array(grouped[delta])
        agg.append({
            "perturbation": delta,
            "mean_cascade_size": float(np.mean(vals)),
            "std_cascade_size": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seed_nodes": len(vals),
        })
    return agg
