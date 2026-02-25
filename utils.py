"""
Shared utilities for the cascade propagation engine.

Centralises helpers that would otherwise be duplicated across modules.
Currently provides:
  - confidence_interval()  : t-distribution CI for a sample mean
  - SeedSequence-based RNG spawning utilities (F-006)

All functions are pure (no global state).
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def confidence_interval(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a confidence interval for the population mean via t-distribution.

    Parameters
    ----------
    samples : np.ndarray, shape (m,)
        Sample array with m >= 2.
    confidence : float, optional
        Confidence level in (0, 1).  Default 0.95.

    Returns
    -------
    (ci_low, ci_high) : tuple of float
        Lower and upper bounds.

    Raises
    ------
    ValueError
        If m < 2 or confidence is not in (0, 1).
    """
    m = len(samples)
    if m < 2:
        raise ValueError("Need at least 2 samples for CI computation.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
    mean = float(np.mean(samples))
    se = float(stats.sem(samples))
    # Guard: zero-variance sample has a degenerate but well-defined CI.
    if se == 0.0:
        return mean, mean
    interval = stats.t.interval(confidence, df=m - 1, loc=mean, scale=se)
    return float(interval[0]), float(interval[1])


# ---------------------------------------------------------------------------
# F-006: SeedSequence-based RNG spawning
# ---------------------------------------------------------------------------


def make_trial_rngs(master_seed: int, n_trials: int) -> list[Generator]:
    """Spawn *n_trials* statistically-independent Generators from a master seed.

    Uses ``numpy.random.SeedSequence.spawn()`` which derives child seeds via a
    hash-based algorithm, guaranteeing statistical independence between streams
    — unlike the simple ``default_rng(seed + i)`` integer-offset approach.

    Parameters
    ----------
    master_seed : int
        The top-level seed.  The same master_seed always produces the same
        sequence of Generators, ensuring full reproducibility.
    n_trials : int
        How many independent Generator instances to create.

    Returns
    -------
    list of Generator
        Length-n_trials list of seeded Generators, ready for use.
    """
    ss = SeedSequence(master_seed)
    return [default_rng(child) for child in ss.spawn(n_trials)]


def make_node_trial_rngs(master_seed: int, n_nodes: int, n_trials: int) -> list[list[Generator]]:
    """Spawn a 2-D grid of independent Generators: [node][trial].

    The master SeedSequence is first split into n_nodes node-level streams;
    each node stream is then split into n_trials trial-level streams.  This
    guarantees independence both *across nodes* and *across trials within a
    node* — replacing the ``seed + node * trials + trial`` integer-offset
    approach.

    Parameters
    ----------
    master_seed : int
        Top-level master seed.
    n_nodes : int
        Number of node-level streams to produce.
    n_trials : int
        Number of trial-level streams per node.

    Returns
    -------
    list of list of Generator
        Shape (n_nodes, n_trials).
    """
    root_ss = SeedSequence(master_seed)
    node_sss = root_ss.spawn(n_nodes)
    return [
        [default_rng(child) for child in node_ss.spawn(n_trials)]
        for node_ss in node_sss
    ]
