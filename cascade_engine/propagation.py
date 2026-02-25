"""
Core cascade propagation engine.

Implements a vectorized, synchronous, discrete-time threshold model over a
directed graph.  States are monotonically non-decreasing:

    0 = operational
    1 = degraded
    2 = failed

All functions are pure (no global mutable state) and deterministic.
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_OPERATIONAL: int = 0
STATE_DEGRADED: int = 1
STATE_FAILED: int = 2


# ---------------------------------------------------------------------------
# Monotonicity guard
# ---------------------------------------------------------------------------


class MonotonicityViolation(RuntimeError):
    """Raised when a node's state decreases between time steps."""


# ---------------------------------------------------------------------------
# Single propagation step
# ---------------------------------------------------------------------------


def propagation_step(
    S: np.ndarray,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Compute one synchronous propagation step.

    Parameters
    ----------
    S : np.ndarray, shape (n,), dtype int
        Current state vector.  Values in {0, 1, 2}.
    A : np.ndarray, shape (n, n), dtype uint8
        Adjacency matrix with convention A[j, i] = 1 iff j → i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold for each node.
    theta_fail : np.ndarray, shape (n,)
        Failure threshold for each node.
    D : np.ndarray, shape (n,)
        In-degree for each node (precomputed for efficiency).

    Returns
    -------
    np.ndarray, shape (n,), dtype int
        Updated state vector (monotonically non-decreasing w.r.t. S).

    Notes
    -----
    The failed-fraction F_i is computed as:

        failed_neighbors_i = sum_j  A[j, i]  *  (S[j] == 2)
        F_i = failed_neighbors_i / D_i   (undefined / no signal if D_i == 0)

    Update rule (applied simultaneously for all i):

        if   D_i == 0              ->  new_state = S[i]  (no cascade signal; isolate)
        elif F_i >= theta_fail[i]  ->  new_state = 2
        elif F_i >= theta_deg[i]   ->  new_state = 1
        else                       ->  new_state = S[i]  (unchanged)

    Only FAILED (state=2) in-neighbors contribute to F_i.
    Degraded (state=1) in-neighbors exert zero cascade pressure.
    Monotonicity is enforced via element-wise maximum.
    """
    n = S.shape[0]

    # Boolean mask: which nodes are currently failed (state == 2)
    failed_mask = (S == STATE_FAILED).astype(np.float64)  # shape (n,)

    # failed_count[i] = number of failed in-neighbors of i
    # A[:, i] gives the in-neighbors of i; dot with failed_mask sums failed ones.
    # Vectorised: (n,n).T @ (n,) = (n,)  but we need column sums weighted by failed_mask
    # A has A[j,i]=1 iff j->i, so column i = in-neighbors of i.
    # failed_in[i] = A[:, i] . failed_mask = sum_j A[j,i] * failed_mask[j]
    # Equivalently: (A.T @ failed_mask)[i]
    failed_in: np.ndarray = A.T @ failed_mask  # shape (n,)

    # Compute failed-fraction; set numerically to 0 for isolates (D_i == 0) so
    # the division is safe, but the (D > 0) guard below ensures isolates never
    # transition — F_i is semantically undefined (no signal) when D_i == 0.
    D_safe = np.where(D > 0, D, 1.0)
    F: np.ndarray = np.where(D > 0, failed_in / D_safe, 0.0)  # shape (n,)

    # Compute candidate new states.
    # Guard with (D > 0): isolates have no cascade signal (F_i semantically
    # undefined) and must never transition via the cascade mechanism.
    # This matches the stochastic engine's explicit (D > 0) guard and the README.
    new_state = np.full(n, STATE_OPERATIONAL, dtype=np.int32)
    new_state = np.where((F >= theta_deg) & (D > 0), STATE_DEGRADED, new_state)
    new_state = np.where((F >= theta_fail) & (D > 0), STATE_FAILED, new_state)

    # Enforce monotonicity: state can never decrease
    result: np.ndarray = np.maximum(S, new_state).astype(np.int32)
    return result


# ---------------------------------------------------------------------------
# Full cascade runner
# ---------------------------------------------------------------------------


def run_until_stable(
    S0: np.ndarray,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    max_steps: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray, bool]:
    """Run the cascade until the state vector stabilises or max_steps is reached.

    Parameters
    ----------
    S0 : np.ndarray, shape (n,), dtype int-like
        Initial state vector.  Values in {0, 1, 2}.
    A : np.ndarray, shape (n, n)
        Adjacency matrix with convention A[j, i] = 1 iff j → i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,)
        Failure threshold per node.
    max_steps : int or None, optional
        Maximum number of propagation steps.  Defaults to ``2 * n``, which is
        the theoretical upper bound for this class of monotone systems.

    Returns
    -------
    final_state : np.ndarray, shape (n,)
        State vector at convergence (or at truncation if max_steps was reached).
    time_to_stability : int
        Number of steps taken until the state stopped changing (0 if S0 is
        already stable).
    full_state_history : np.ndarray, shape (T+1, n)
        Complete state history including the initial state at row 0.
    convergence_reached : bool
        ``True`` if the state vector stabilised before ``max_steps`` was
        exhausted; ``False`` if the loop was cut short.  With the default
        ``max_steps = 2 * n`` this is always ``True`` for monotone systems,
        but callers passing a custom ``max_steps`` should check this flag.

    Raises
    ------
    MonotonicityViolation
        If any node's state decreases between consecutive time steps.
    ValueError
        If input arrays are inconsistent in shape.
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

    if max_steps is None:
        max_steps = 2 * n

    # Precompute in-degrees (constant across steps)
    D: np.ndarray = A.sum(axis=0).astype(np.float64)  # shape (n,)

    S_current = np.array(S0, dtype=np.int32)
    history: list[np.ndarray] = [S_current.copy()]

    convergence_reached: bool = False
    for _ in range(max_steps):
        S_next = propagation_step(S_current, A, theta_deg, theta_fail, D)

        # Monotonicity check — step index is len(history) before append
        if np.any(S_next < S_current):
            violators = np.where(S_next < S_current)[0].tolist()
            raise MonotonicityViolation(
                f"Monotonicity violated at step {len(history)} "
                f"for nodes: {violators}"
            )

        history.append(S_next.copy())

        if np.array_equal(S_next, S_current):
            # Stable: remove the duplicate final snapshot so history length
            # equals (meaningful steps taken + 1 for S0).
            history.pop()
            convergence_reached = True
            break

        S_current = S_next
    else:
        warnings.warn(
            f"run_until_stable: max_steps={max_steps} reached without convergence. "
            "Returning partial result. Consider increasing max_steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    full_history = np.array(history, dtype=np.int32)  # shape (T+1, n)
    # time_to_stability = number of steps until first stable state
    time_to_stability = full_history.shape[0] - 1

    return S_current, time_to_stability, full_history, convergence_reached
