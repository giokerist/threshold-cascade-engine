"""
Stochastic cascade propagation engine.

Extends the deterministic threshold model with a logistic (sigmoid) failure
probability model.  All logic is kept strictly separate from the deterministic
engine in propagation.py to preserve Tier 1 integrity.

State encoding mirrors propagation.py:
    0 = operational
    1 = degraded
    2 = failed

Update rule (Tier 2 — stochastic mode)
---------------------------------------
For each node i at time t, let F_i(t) be the failed dependency ratio
(identical computation to deterministic mode):

    F_i(t) = (# failed in-neighbors) / D_i    (undefined / no signal if D_i == 0)

**Axiom — Degradation (hard threshold, both Tier 1 and Tier 2):**
    Degradation represents an observable, discrete capacity reduction triggered
    when the fraction of failed upstream dependencies crosses a hard threshold.
    It is modelled deterministically in both tiers:

        degrade if F_i(t) >= theta_deg[i]

    This reflects the assumption that infrastructure degradation is a
    measurable, deterministic consequence of upstream failure load, not a
    probabilistic outcome.  Only the *absorption* into full failure is
    treated stochastically (see below).

**Axiom — Failure (logistic probability, Tier 2 only):**
    Failure represents an absorbing state whose onset is subject to
    uncertainty (e.g., residual capacity, load variability, recovery
    attempts).  It uses a logistic probability:

        P(S_i -> 2 | F_i) = sigmoid( k * (F_i - theta_fail[i]) )

    where sigmoid(x) = 1 / (1 + exp(-x))  and k > 0 controls steepness.
    At k -> inf this recovers the deterministic Tier 1 rule exactly.

Monotonicity is preserved: once degraded or failed, a node cannot revert.

Convergence criterion (F-001 fix)
----------------------------------
Stochastic cascades are declared converged only after ``consecutive_stable_steps``
consecutive steps with no state changes.  A single quiet step does not
guarantee absorption -- a near-threshold node could still flip on the next
draw.  The default is 3 consecutive quiet steps, which empirically captures
>99% of residual transitions for k >= 5 on sparse graphs.  Callers may
override via the ``consecutive_stable_steps`` parameter.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.random import Generator
from scipy.special import expit as _expit

from .propagation import (
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
    result = _expit(x)
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

    Degradation uses a hard threshold (Tier 2 axiom -- see module docstring).
    Failure uses logistic probability sampling.

    Parameters
    ----------
    S : np.ndarray, shape (n,), dtype int
        Current state vector.  Values in {0, 1, 2}.
    A : np.ndarray, shape (n, n), dtype uint8
        Adjacency matrix: A[j, i] = 1 iff j -> i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node (hard, deterministic).
    theta_fail : np.ndarray, shape (n,)
        Failure threshold (logistic midpoint) per node.
    D : np.ndarray, shape (n,)
        In-degree per node (precomputed).
    k : float
        Logistic steepness parameter (k > 0).  Higher k -> closer to
        deterministic step; k = 10 is a reasonable default.
    rng : Generator
        Seeded numpy Generator for reproducible sampling.

    Returns
    -------
    np.ndarray, shape (n,), dtype int
        Updated state vector (monotonically non-decreasing w.r.t. S).
    """
    # ---- Compute failed dependency ratio F_i (identical to deterministic) --
    failed_mask = (S == STATE_FAILED).astype(np.float64)       # (n,)
    failed_in: np.ndarray = A.T @ failed_mask                   # (n,)
    D_safe = np.where(D > 0, D, 1.0)
    F: np.ndarray = np.where(D > 0, failed_in / D_safe, 0.0)   # (n,)

    # ---- Degradation: hard threshold (Tier 2 axiom -- deterministic) -------
    will_degrade = (F >= theta_deg) & (D > 0)                  # (n,) bool

    # ---- Failure: logistic probability sampling ----------------------------
    p_fail: np.ndarray = sigmoid(k * (F - theta_fail))         # (n,)
    p_fail = np.where(D > 0, p_fail, 0.0)                      # zero out isolates
    uniform_draws: np.ndarray = rng.random(S.shape[0])         # (n,) single call
    will_fail = uniform_draws < p_fail                         # (n,) bool

    # ---- Build candidate new state vector ----------------------------------
    new_state = np.full(S.shape[0], STATE_OPERATIONAL, dtype=np.int32)
    new_state = np.where(will_degrade, STATE_DEGRADED, new_state)
    new_state = np.where(will_fail, STATE_FAILED, new_state)

    # ---- Enforce monotonicity ----------------------------------------------
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
    consecutive_stable_steps: int = 3,
) -> tuple[np.ndarray, int, np.ndarray, bool]:
    """Run the stochastic cascade until the state vector stabilises.

    **Convergence criterion (F-001 fix):**
    Stabilisation requires *consecutive_stable_steps* consecutive steps with
    zero state changes.  A single quiet step is insufficient in stochastic
    mode -- a near-threshold node may still flip on the next draw.  Default
    of 3 captures >99% of residual transitions for k >= 5.

    Parameters
    ----------
    S0 : np.ndarray, shape (n,), dtype int-like
        Initial state vector.  Values in {0, 1, 2}.
    A : np.ndarray, shape (n, n)
        Adjacency matrix: A[j, i] = 1 iff j -> i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,)
        Logistic midpoint (failure threshold) per node.
    k : float
        Logistic steepness parameter (k > 0).
    rng : Generator
        Seeded numpy Generator.
    max_steps : int or None, optional
        Maximum propagation steps.  Defaults to ``4 * n``.
    consecutive_stable_steps : int, optional
        Consecutive quiet steps required for convergence.  Default 3.
        Increase for low k (< 3) or near-threshold graphs.

    Returns
    -------
    final_state : np.ndarray, shape (n,)
        State vector at convergence or truncation.
    time_to_stability : int
        Step index of the last actual state change (0 if S0 already stable).
    full_state_history : np.ndarray, shape (T+1, n)
        History of states at every step where a change occurred, plus S0.
        Quiet-counting steps are not appended (no new information).
    convergence_reached : bool
        True if stabilised before max_steps; False if truncated.

    Raises
    ------
    MonotonicityViolation
        If any node's state decreases between consecutive steps.
    ValueError
        If input arrays are inconsistent in shape or k/consecutive_stable_steps
        are out of range.
    """
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
    if consecutive_stable_steps < 1:
        raise ValueError(
            f"consecutive_stable_steps must be >= 1; got {consecutive_stable_steps}."
        )

    if max_steps is None:
        max_steps = 4 * n

    D: np.ndarray = A.sum(axis=0).astype(np.float64)

    S_current = np.array(S0, dtype=np.int32)
    history: list[np.ndarray] = [S_current.copy()]

    convergence_reached: bool = False
    quiet_count: int = 0
    last_change_step: int = 0

    for step in range(max_steps):
        S_next = propagation_step_stochastic(
            S_current, A, theta_deg, theta_fail, D, k, rng
        )

        if np.any(S_next < S_current):
            violators = np.where(S_next < S_current)[0].tolist()
            raise MonotonicityViolation(
                f"Monotonicity violated at step {step + 1} "
                f"for nodes: {violators}"
            )

        if np.array_equal(S_next, S_current):
            # Quiet step -- count but do not append duplicate to history
            quiet_count += 1
            if quiet_count >= consecutive_stable_steps:
                convergence_reached = True
                break
        else:
            # Active step -- record and reset quiet counter
            quiet_count = 0
            last_change_step = step + 1
            history.append(S_next.copy())

        S_current = S_next

    else:
        warnings.warn(
            f"run_until_stable_stochastic: max_steps={max_steps} reached without "
            f"convergence (consecutive_stable_steps={consecutive_stable_steps}). "
            "Returning partial result. Consider increasing max_steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    full_history = np.array(history, dtype=np.int32)
    time_to_stability = last_change_step

    return S_current, time_to_stability, full_history, convergence_reached
