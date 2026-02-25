"""
Sensitivity analysis module for the cascade propagation engine.

Quantifies how cascade size changes as failure thresholds are perturbed.
Supports both deterministic and stochastic propagation modes.

Threshold clamping (F-003 fix)
-------------------------------
When a negative perturbation delta reduces theta_fail_p below base_theta_deg
for some nodes, two strategies are available via the ``clamp_mode`` parameter:

``'fail_drags_deg'`` (original behaviour, kept for backward compatibility)
    theta_deg_p is dragged down to match theta_fail_p.  This means both
    thresholds change for affected nodes.  Reported cascade sizes for large
    negative deltas reflect *simultaneous* reductions in both theta_fail and
    theta_deg -- NOT a pure theta_fail sweep.  A UserWarning is always emitted
    when this occurs.

``'fail_floored_at_deg'`` (recommended for pure theta_fail analysis)
    theta_fail_p is floored at base_theta_deg element-wise.  theta_deg_p is
    held constant at base_theta_deg.  This produces a pure theta_fail sweep
    at the cost of a smaller effective range for negative deltas.  The actual
    applied perturbation may be smaller than delta for nodes where clamping
    activates.  A UserWarning is emitted reporting how many nodes were
    floor-clamped.

The default is ``'fail_floored_at_deg'`` for new analyses.  Use
``'fail_drags_deg'`` only when backward compatibility with existing results
is required, and always document which mode was used in reported results.

Design
------
* Pure functions: no global state, no side effects.
* All results are returned as structured dicts ready for CSV export.
* Stochastic mode averages over multiple trials per perturbation level.
* Perturbations are applied to a copy of the threshold array -- originals
  are never mutated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.random import SeedSequence, default_rng

from .propagation import run_until_stable, STATE_DEGRADED
from .stochastic_propagation import run_until_stable_stochastic


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityPoint:
    """Result for a single perturbation level and seed node.

    Attributes
    ----------
    perturbation : float
        The delta applied to theta_fail: theta_fail_p = clip(base + delta, 0, 1)
        (subject to clamping -- see clamp_mode in threshold_sensitivity).
    seed_node : int
        Which node was seeded as the initial failure.
    cascade_size_fraction : float
        Fraction of nodes affected (degraded + failed) at convergence.
    time_to_stability : int
        Steps until convergence (or mean across trials in stochastic mode).
    n_trials : int
        1 for deterministic, >= 2 for stochastic (averaged).
    std_cascade_size : float
        Standard deviation across trials (0.0 for deterministic).
    clamp_mode : str
        The clamping mode used: 'fail_floored_at_deg' or 'fail_drags_deg'.
    n_nodes_clamped : int
        Number of nodes where clamping activated for this perturbation level.
    """
    perturbation: float
    seed_node: int
    cascade_size_fraction: float
    time_to_stability: float
    n_trials: int
    std_cascade_size: float
    clamp_mode: str = "fail_floored_at_deg"
    n_nodes_clamped: int = 0

    def to_dict(self) -> dict:
        return {
            "perturbation": self.perturbation,
            "seed_node": self.seed_node,
            "cascade_size_fraction": self.cascade_size_fraction,
            "time_to_stability": self.time_to_stability,
            "n_trials": self.n_trials,
            "std_cascade_size": self.std_cascade_size,
            "clamp_mode": self.clamp_mode,
            "n_nodes_clamped": self.n_nodes_clamped,
        }


# ---------------------------------------------------------------------------
# Threshold clamping helpers
# ---------------------------------------------------------------------------


def _apply_perturbation(
    base_theta_deg: np.ndarray,
    base_theta_fail: np.ndarray,
    delta: float,
    clamp_mode: str,
    warn_stacklevel: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply delta to theta_fail and clamp according to clamp_mode.

    Parameters
    ----------
    base_theta_deg : np.ndarray
        Baseline degradation thresholds (never mutated).
    base_theta_fail : np.ndarray
        Baseline failure thresholds (never mutated).
    delta : float
        Perturbation to add to theta_fail.
    clamp_mode : str
        'fail_floored_at_deg' or 'fail_drags_deg'.
    warn_stacklevel : int
        Passed to warnings.warn.

    Returns
    -------
    theta_deg_p : np.ndarray
        Perturbed/clamped degradation thresholds.
    theta_fail_p : np.ndarray
        Perturbed/clamped failure thresholds.
    n_clamped : int
        Number of nodes where clamping activated.
    """
    if clamp_mode not in ("fail_floored_at_deg", "fail_drags_deg"):
        raise ValueError(
            f"clamp_mode must be 'fail_floored_at_deg' or 'fail_drags_deg'; "
            f"got {clamp_mode!r}."
        )

    theta_fail_raw = np.clip(base_theta_fail + delta, 0.0, 1.0)

    # Identify nodes where raw perturbed theta_fail drops below theta_deg
    clamp_mask = theta_fail_raw < base_theta_deg
    n_clamped = int(np.sum(clamp_mask))

    if clamp_mode == "fail_floored_at_deg":
        # Floor theta_fail at theta_deg; theta_deg stays constant.
        # This is a PURE theta_fail sweep -- theta_deg is never modified.
        theta_fail_p = np.maximum(theta_fail_raw, base_theta_deg)
        theta_deg_p = base_theta_deg.copy()
        if n_clamped > 0:
            warnings.warn(
                f"threshold_sensitivity [fail_floored_at_deg]: delta={delta:.4f} "
                f"would have pushed theta_fail below theta_deg for {n_clamped} "
                f"node(s). theta_fail_p has been floored at base_theta_deg for "
                f"those nodes. theta_deg is unchanged -- this remains a pure "
                f"theta_fail perturbation. The effective perturbation for clamped "
                f"nodes is smaller than delta.",
                UserWarning,
                stacklevel=warn_stacklevel,
            )
    else:
        # 'fail_drags_deg': drag theta_deg down to match theta_fail.
        # WARNING: both thresholds change -- this is NOT a pure theta_fail sweep.
        theta_fail_p = theta_fail_raw
        theta_deg_p = np.minimum(base_theta_deg, theta_fail_p)
        if n_clamped > 0:
            warnings.warn(
                f"threshold_sensitivity [fail_drags_deg]: delta={delta:.4f} "
                f"caused theta_fail_p to drop below base_theta_deg for "
                f"{n_clamped} node(s). theta_deg has been dragged down to match. "
                f"Results for this perturbation level reflect changes in BOTH "
                f"theta_fail AND theta_deg -- NOT a pure theta_fail sweep. "
                f"Consider switching to clamp_mode='fail_floored_at_deg' for "
                f"unbiased theta_fail sensitivity analysis.",
                UserWarning,
                stacklevel=warn_stacklevel,
            )

    return theta_deg_p, theta_fail_p, n_clamped


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
    clamp_mode: str = "fail_floored_at_deg",
    consecutive_stable_steps: int = 3,
) -> list[SensitivityPoint]:
    """Run threshold sensitivity analysis.

    For each perturbation delta in ``perturbation_values``, the failure
    threshold array is adjusted:

        theta_fail_perturbed = clip(base_theta_fail + delta, 0, 1)

    The degradation threshold is then handled according to ``clamp_mode``
    (see module docstring for full explanation).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix: A[j, i] = 1 iff j -> i.
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
        Monte Carlo trials per (perturbation, seed_node) combination
        when mode == "stochastic" (default 30).
    stochastic_k : float, optional
        Logistic steepness for stochastic mode (default 10.0).
    seed : int, optional
        Master seed for stochastic trials (default 0).  Per-trial RNGs are
        derived via SeedSequence.spawn() for statistical independence.
    clamp_mode : str, optional
        How to handle negative deltas that push theta_fail below theta_deg.
        'fail_floored_at_deg' (default) keeps theta_deg fixed and floors
        theta_fail -- pure theta_fail sweep, recommended.
        'fail_drags_deg' drags theta_deg down -- biased, kept for backward
        compatibility only.
    consecutive_stable_steps : int, optional
        Passed to run_until_stable_stochastic (default 3).

    Returns
    -------
    list of SensitivityPoint
        One record per (perturbation, seed_node) combination.

    Raises
    ------
    ValueError
        If mode or clamp_mode are not recognised values.
    """
    if mode not in ("deterministic", "stochastic"):
        raise ValueError(f"mode must be 'deterministic' or 'stochastic'; got {mode!r}.")

    n = A.shape[0]
    if seed_nodes is None:
        seed_nodes = list(range(n))

    # F-006: build a 3-level SeedSequence hierarchy for stochastic mode:
    #   root -> perturbation -> seed_node -> trial
    # IMPORTANT: all spawn() calls must happen OUTSIDE the loops they index,
    # because SeedSequence.spawn() is stateful — each call consumes entropy
    # from the parent and produces a new, distinct set of children. Calling
    # spawn() inside an inner loop causes the same p_idx to resolve to a
    # different parent seed on every sn_idx iteration, breaking independence.
    node_ss_grid: list[list] = []
    if mode == "stochastic":
        root_ss = SeedSequence(seed)
        pert_ss_list = root_ss.spawn(len(perturbation_values))
        node_ss_grid = [
            pert_ss.spawn(len(seed_nodes))
            for pert_ss in pert_ss_list
        ]

    results: list[SensitivityPoint] = []

    for p_idx, delta in enumerate(perturbation_values):
        theta_deg_p, theta_fail_p, n_clamped = _apply_perturbation(
            base_theta_deg, base_theta_fail, delta,
            clamp_mode=clamp_mode,
            warn_stacklevel=2,
        )

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
                    clamp_mode=clamp_mode,
                    n_nodes_clamped=n_clamped,
                ))

            else:  # stochastic
                # F-006: use the pre-spawned hierarchy — node_ss_grid[p_idx][sn_idx]
                # was derived outside the loops so each (perturbation, seed_node)
                # pair gets a unique, independent SeedSequence as intended.
                trial_rngs = [
                    default_rng(child)
                    for child in node_ss_grid[p_idx][sn_idx].spawn(stochastic_trials)
                ]

                trial_cs = np.empty(stochastic_trials, dtype=np.float64)
                trial_ts = np.empty(stochastic_trials, dtype=np.float64)

                for trial in range(stochastic_trials):
                    final_state, t_stable, _, _ = run_until_stable_stochastic(
                        S0, A, theta_deg_p, theta_fail_p,
                        k=stochastic_k,
                        rng=trial_rngs[trial],
                        consecutive_stable_steps=consecutive_stable_steps,
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
                    clamp_mode=clamp_mode,
                    n_nodes_clamped=n_clamped,
                ))

    return results


def sensitivity_to_records(points: list[SensitivityPoint]) -> list[dict]:
    """Convert a list of SensitivityPoints to a list of dicts for CSV export."""
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
        perturbation, mean_cascade_size, std_cascade_size, n_seed_nodes,
        clamp_mode, n_nodes_clamped_mean.
    """
    from collections import defaultdict
    grouped: dict[float, list] = defaultdict(list)
    for p in points:
        grouped[p.perturbation].append(p)

    agg = []
    for delta in sorted(grouped.keys()):
        pts = grouped[delta]
        vals = np.array([p.cascade_size_fraction for p in pts])
        agg.append({
            "perturbation": delta,
            "mean_cascade_size": float(np.mean(vals)),
            "std_cascade_size": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seed_nodes": len(vals),
            "clamp_mode": pts[0].clamp_mode,
            "n_nodes_clamped": pts[0].n_nodes_clamped,
        })
    return agg
