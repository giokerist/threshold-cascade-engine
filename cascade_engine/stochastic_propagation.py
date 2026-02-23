"""
Stochastic cascade propagation engine.

Extends the deterministic threshold model with a logistic (sigmoid) failure
probability model.  All logic is kept strictly separate from the deterministic
engine in propagation.py to preserve Tier 1 integrity.

State encoding mirrors propagation.py:
    0 = operational
    1 = degraded
    2 = failed

Update rule (stochastic mode)
------------------------------
For each node i at time t, let F_i(t) be the failed dependency ratio
(identical computation to deterministic mode):

    F_i(t) = (# failed in-neighbors) / D_i    (0 if D_i == 0)

Degradation uses the same hard threshold as deterministic mode:
    degrade if F_i >= theta_deg[i]

Failure uses a logistic probability instead of a hard threshold:
    P(S_i -> 2) = sigmoid( k * (F_i - theta_fail[i]) )

where sigmoid(x) = 1 / (1 + exp(-x))  and k > 0 controls steepness.

At k → ∞, this recovers the deterministic rule.

Monotonicity is preserved: once degraded or failed, a node cannot revert.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from propagation import (
    STATE_OPERATIONAL,
    STATE_DEGRADED,
    STATE_FAILED,
    MonotonicityViolation,
)


# ---------------------------------------------------------------------------
# Logistic helper
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid / logistic function.

    Uses ``scipy.special.expit`` which avoids overflow/underflow at extreme
    inputs.  Output is clipped to the open interval (0, 1) using float64
    machine epsilon, ensuring no exact 0 or 1 values are returned.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape).

    Returns
    -------
    np.ndarray
        Values strictly in (0, 1) of the same shape as x.
    """
    from scipy.special import expit
    result = expit(x)
    # At float64 extremes expit may saturate to exactly 0.0 or 1.0.
    # Use np.finfo.eps (machine epsilon ≈ 2.2e-16) so the upper clip
    # 1.0 - eps is the largest float64 strictly less than 1.0.
    _eps = np.finfo(np.float64).eps
    return np.clip(result, np.finfo(np.float64).tiny, 1.0 - _eps)


# ---------------------------------------------------------------------------
# Single stochastic propagation step
# ---------------------------------------------------------------------------


def propagation_step_stochastic(
    S: np.ndarray,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    D: np.ndarray,
    k: float,
    rng: Generator,
) -> np.ndarray:
    """Compute one synchronous stochastic propagation step.

    Degradation uses a hard threshold (same as deterministic).
    Failure uses logistic probability sampling.

    Parameters
    ----------
    S : np.ndarray, shape (n,), dtype int
        Current state vector.  Values in {0, 1, 2}.
    A : np.ndarray, shape (n, n), dtype uint8
        Adjacency matrix: A[j, i] = 1 iff j → i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,)
        Failure threshold (logistic midpoint) per node.
    D : np.ndarray, shape (n,)
        In-degree per node (precomputed).
    k : float
        Logistic steepness parameter (k > 0).  Higher k → closer to
        deterministic step; k = 10 is a reasonable default.
    rng : Generator
        Seeded numpy Generator for reproducible sampling.

    Returns
    -------
    np.ndarray, shape (n,), dtype int
        Updated state vector (monotonically non-decreasing w.r.t. S).

    Notes
    -----
    Vectorized: no Python loops over nodes.
    Sampling is a single rng.random(n) call, keeping RNG state advancement
    consistent regardless of node count.
    """
    # ---- Compute failed dependency ratio F_i (identical to deterministic) --
    failed_mask = (S == STATE_FAILED).astype(np.float64)       # (n,)
    failed_in: np.ndarray = A.T @ failed_mask                   # (n,)
    D_safe = np.where(D > 0, D, 1.0)
    F: np.ndarray = np.where(D > 0, failed_in / D_safe, 0.0)   # (n,)

    # ---- Degradation: hard threshold (same as deterministic) ----------------
    # Only applies to nodes with at least one in-neighbor (D_i > 0)
    will_degrade = (F >= theta_deg) & (D > 0)                  # (n,) bool

    # ---- Failure: logistic probability sampling -----------------------------
    # P(fail) = sigmoid(k * (F_i - theta_fail_i)), but ONLY for nodes with
    # D_i > 0.  Nodes with no in-neighbors have no dependency signal and
    # cannot be triggered by the cascade (F_i is undefined / 0 for them).
    p_fail: np.ndarray = sigmoid(k * (F - theta_fail))         # (n,)
    p_fail = np.where(D > 0, p_fail, 0.0)                      # zero out isolates
    uniform_draws: np.ndarray = rng.random(S.shape[0])         # (n,) single call
    will_fail = uniform_draws < p_fail                         # (n,) bool

    # ---- Build candidate new state vector -----------------------------------
    new_state = np.full(S.shape[0], STATE_OPERATIONAL, dtype=np.int32)
    new_state = np.where(will_degrade, STATE_DEGRADED, new_state)
    new_state = np.where(will_fail, STATE_FAILED, new_state)

    # ---- Enforce monotonicity -----------------------------------------------
    result: np.ndarray = np.maximum(S, new_state).astype(np.int32)
    return result


# ---------------------------------------------------------------------------
# Full stochastic cascade runner
# ---------------------------------------------------------------------------


def run_until_stable_stochastic(
    S0: np.ndarray,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    k: float,
    rng: Generator,
    max_steps: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray, bool]:
    """Run the stochastic cascade until the state vector stabilises.

    Stabilisation is declared when no state changes occur in a step.  Because
    the model is stochastic, a stable step does not guarantee global
    convergence; however, the monotonicity guarantee means states cannot
    revert, so eventual absorption is certain in finite time.

    Parameters
    ----------
    S0 : np.ndarray, shape (n,), dtype int-like
        Initial state vector.  Values in {0, 1, 2}.
    A : np.ndarray, shape (n, n)
        Adjacency matrix: A[j, i] = 1 iff j → i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,)
        Logistic midpoint (failure threshold) per node.
    k : float
        Logistic steepness parameter (k > 0).
    rng : Generator
        Seeded numpy Generator.  Caller is responsible for seeding to ensure
        reproducibility.  The Generator is advanced in-place each step.
    max_steps : int or None, optional
        Maximum number of propagation steps.  Defaults to ``4 * n`` to allow
        for probabilistic settling (more than the deterministic 2n bound).

    Returns
    -------
    final_state : np.ndarray, shape (n,)
        State vector at convergence (or at truncation if max_steps was reached).
    time_to_stability : int
        Number of steps taken until no further changes were observed.
    full_state_history : np.ndarray, shape (T+1, n)
        Complete state history including the initial state at row 0.
    convergence_reached : bool
        ``True`` if the state vector stabilised before ``max_steps`` was
        exhausted; ``False`` if the loop was cut short.

    Raises
    ------
    MonotonicityViolation
        If a node's state decreases between consecutive steps (should never
        occur given the maximum enforcement, but kept as a safety guard).
    ValueError
        If input arrays are inconsistent in shape.
    """
    import warnings

    n = int(A.shape[0])
    if A.shape != (n, n):
        raise ValueError(f"A must be square; got shape {A.shape}.")
    if S0.shape != (n,):
        raise ValueError(f"S0 must have shape ({n},); got {S0.shape}.")
    if theta_deg.shape != (n,):
        raise ValueError(f"theta_deg must have shape ({n},); got {theta_deg.shape}.")
    if theta_fail.shape != (n,):
        raise ValueError(f"theta_fail must have shape ({n},); got {theta_fail.shape}.")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}.")

    if max_steps is None:
        max_steps = 4 * n

    D: np.ndarray = A.sum(axis=0).astype(np.float64)

    S_current = np.array(S0, dtype=np.int32)
    history: list[np.ndarray] = [S_current.copy()]

    steps_taken: int = 0
    convergence_reached: bool = False
    for _ in range(max_steps):
        S_next = propagation_step_stochastic(
            S_current, A, theta_deg, theta_fail, D, k, rng
        )

        if np.any(S_next < S_current):
            violators = np.where(S_next < S_current)[0].tolist()
            raise MonotonicityViolation(
                f"Monotonicity violated at step {steps_taken + 1} "
                f"for nodes: {violators}"
            )

        history.append(S_next.copy())
        steps_taken += 1

        if np.array_equal(S_next, S_current):
            history.pop()
            convergence_reached = True
            break

        S_current = S_next
    else:
        warnings.warn(
            f"run_until_stable_stochastic: max_steps={max_steps} reached without "
            "convergence. Returning partial result. Consider increasing max_steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    full_history = np.array(history, dtype=np.int32)
    time_to_stability = full_history.shape[0] - 1

    return S_current, time_to_stability, full_history, convergence_reached
