"""
propagation_fast.py — Vectorized Sparse Propagation with Numba JIT
===================================================================

High-performance replacement for ``propagation.py`` targeting 100K+ nodes
and 1M+ edges on a 4-core Linux machine.

Architecture
------------
Two paths, both operating on the same ``(A_T, in_degree)`` CSR pair:

1. **Numba JIT path** (``_update_kernel_numba``):
   The innermost state-update loop is compiled by Numba with
   ``@njit(parallel=True, fastmath=True)``.  The outer ``prange`` distributes
   rows (nodes) across all available CPU threads; the inner ``range`` walks
   the CSR column index array for each node sequentially.  All inputs are
   typed NumPy arrays — no Python objects enter the JIT region, bypassing
   the GIL entirely.

2. **SciPy sparse path** (``propagation_step_sparse``):
   Uses a single ``A_T.dot(failed_vec)`` BLAS-backed matvec to compute
   in-neighbour failure counts in one vectorised call, then applies
   ``np.where`` for the threshold decisions.  This path is faster for
   moderate graphs and serves as the stochastic engine's backbone.

The linear-algebra propagation formula
---------------------------------------
The deterministic cascade rule is:

    F = (A_T · S_t^{fail}) / in_degree
    S_{t+1} = Φ(F − Θ_fail)·2 + Φ(F − Θ_deg)·(1 − Φ(F − Θ_fail))

Where:
  * ``A_T``          — CSR transpose of the adjacency matrix (row i = in-neighbours of i)
  * ``S_t^{fail}``   — binary float64 vector: 1.0 if node is in STATE_FAILED (2), else 0.0
  * ``Θ_fail``       — failure threshold vector (shape n)
  * ``Φ``            — Heaviside step function: Φ(x) = 1 if x >= 0, else 0

For the degraded state, the same pattern is applied with Θ_deg.

The full update rule (both states) becomes::

    F = A_T · S_t^{fail} / in_degree          # failed fraction per node
    new_state = Φ(F - Θ_fail) * 2             # 2 if F >= Θ_fail
              + Φ(F - Θ_deg) * (1 - Φ(F - Θ_fail))  # 1 if Θ_deg <= F < Θ_fail
    S_{t+1} = max(S_t, new_state)             # monotonicity

Numba warm-up
-------------
On first import, this module calls ``_warmup_numba()`` which runs the JIT
kernel on a tiny 4-node graph.  This amortises the ~1s compilation overhead
before any user-facing call.

Usage
-----
>>> from cascade_engine.graph_sparse import build_sparse_graph
>>> from cascade_engine.propagation_fast import run_until_stable_fast
>>> A_T, D = build_sparse_graph(src_ids, tgt_ids, n_nodes)
>>> final, t, history, converged = run_until_stable_fast(
...     S0, A_T, D, theta_deg, theta_fail, progress_callback=my_cb
... )
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

# Lazy Numba import — if numba is not installed, fall back to pure numpy.
try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    numba = None  # type: ignore[assignment]

from .propagation import (
    STATE_OPERATIONAL,
    STATE_DEGRADED,
    STATE_FAILED,
    MonotonicityViolation,
)
from .progress import ProgressCallback, null_callback


# ---------------------------------------------------------------------------
# Numba JIT kernel — compiled once, reused for every step
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _update_kernel_numba(
        at_data: np.ndarray,      # float32 (nnz,)  — CSR A_T.data
        at_indices: np.ndarray,   # int32   (nnz,)  — CSR A_T.indices
        at_indptr: np.ndarray,    # int32   (n+1,)  — CSR A_T.indptr
        failed_state: np.ndarray, # float64 (n,)    — 1.0 iff node is FAILED
        in_degree: np.ndarray,    # float64 (n,)    — in-degree per node
        theta_deg: np.ndarray,    # float64 (n,)    — degradation threshold
        theta_fail: np.ndarray,   # float64 (n,)    — failure threshold
        S_current: np.ndarray,    # int32   (n,)    — current state vector
    ) -> np.ndarray:
        """Inner state-update kernel compiled by Numba with full parallelism.

        The outer ``prange`` distributes one row (node) per thread using all
        available CPU cores.  The inner ``range`` walks the CSR column index
        slice for that node sequentially.  Every write targets a unique index
        in ``S_next``, so there are no data races.

        The GIL is released for the entire duration of this function because
        all inputs are typed NumPy arrays and there are no Python object
        accesses inside the JIT region.

        Update rule per node i
        ----------------------
        1. Traverse in-neighbours (CSR row i of A_T): accumulate weighted
           failed count ``failed_count_i = Σ_{k in row} at_data[k] * failed_state[at_indices[k]]``
        2. Compute failure fraction: ``F_i = failed_count_i / in_degree[i]``
        3. Apply thresholds (with isolate guard ``in_degree[i] > 0``):
           * ``F_i >= theta_fail[i]`` → candidate = FAILED (2)
           * ``F_i >= theta_deg[i]``  → candidate = DEGRADED (1)
           * otherwise                → candidate = OPERATIONAL (0)
        4. Monotonicity: ``S_next[i] = max(S_current[i], candidate)``
        """
        n = len(at_indptr) - 1
        S_next = np.empty(n, dtype=np.int32)

        for i in prange(n):  # parallel outer loop — one row per thread
            deg_i = in_degree[i]

            if deg_i == 0.0:
                # Isolate: no cascade signal; state can only ratchet upward
                S_next[i] = S_current[i]
                continue

            # Dot product: sum of at_data[k] * failed_state[at_indices[k]]
            # for k in row i of A_T (= in-neighbours of i)
            failed_count = 0.0
            row_start = at_indptr[i]
            row_end = at_indptr[i + 1]
            for k in range(row_start, row_end):
                failed_count += at_data[k] * failed_state[at_indices[k]]

            F_i = failed_count / deg_i

            # Threshold decisions (closed-form, branch-free)
            if F_i >= theta_fail[i]:
                candidate = np.int32(2)    # STATE_FAILED
            elif F_i >= theta_deg[i]:
                candidate = np.int32(1)    # STATE_DEGRADED
            else:
                candidate = np.int32(0)    # STATE_OPERATIONAL

            # Monotonicity: state can never decrease
            cur = S_current[i]
            S_next[i] = candidate if candidate > cur else cur

        return S_next

else:
    # Pure-NumPy fallback (used when Numba is not installed)
    def _update_kernel_numba(  # type: ignore[misc]
        at_data, at_indices, at_indptr,
        failed_state, in_degree, theta_deg, theta_fail, S_current,
    ) -> np.ndarray:
        """NumPy fallback for _update_kernel_numba (no Numba)."""
        # Reconstruct CSR matvec manually using numpy
        # This is slower but functionally identical.
        n = len(at_indptr) - 1
        failed_counts = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for k in range(at_indptr[i], at_indptr[i + 1]):
                failed_counts[i] += at_data[k] * failed_state[at_indices[k]]

        D_safe = np.where(in_degree > 0, in_degree, 1.0)
        F = np.where(in_degree > 0, failed_counts / D_safe, 0.0)

        new_state = np.zeros(n, dtype=np.int32)
        new_state = np.where((F >= theta_deg) & (in_degree > 0), 1, new_state)
        new_state = np.where((F >= theta_fail) & (in_degree > 0), 2, new_state)

        return np.maximum(S_current, new_state).astype(np.int32)


# ---------------------------------------------------------------------------
# SciPy sparse path — vectorised, no Numba required
# ---------------------------------------------------------------------------


def propagation_step_sparse(
    S: np.ndarray,
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
) -> np.ndarray:
    """One synchronous propagation step using SciPy CSR sparse matvec.

    Implements the linear-algebra formula::

        F = (A_T · S_failed) / in_degree
        S_{t+1} = Φ(F − Θ_fail)*2 + Φ(F − Θ_deg)*(1 − Φ(F − Θ_fail))
        S_{t+1} = max(S_t, S_{t+1})    # monotonicity

    The single ``A_T.dot()`` call dispatches to SciPy's BLAS-backed CSR
    matrix-vector product, which is typically faster than the Numba path for
    moderate graphs (<50K nodes) because it uses Intel MKL or OpenBLAS.
    The Numba path is preferred for very large graphs where memory access
    patterns benefit from per-thread caching.

    Parameters
    ----------
    S : np.ndarray, shape (n,), dtype int32
        Current state vector.
    A_T : csr_matrix, shape (n, n)
        Transpose adjacency matrix: row i = in-neighbours of i.
    in_degree : np.ndarray, shape (n,), dtype float64
        In-degree per node (precomputed).
    theta_deg : np.ndarray, shape (n,), dtype float64
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,), dtype float64
        Failure threshold per node.

    Returns
    -------
    np.ndarray, shape (n,), dtype int32
        Updated state vector (monotonically non-decreasing).
    """
    # --- Compute failure fraction F_i = (A_T · failed_vec)[i] / in_degree[i] ---
    failed_vec = (S == STATE_FAILED).astype(np.float64)  # (n,)
    failed_counts = A_T.dot(failed_vec)                   # (n,) — sparse matvec

    D_safe = np.where(in_degree > 0, in_degree, 1.0)
    F: np.ndarray = np.where(in_degree > 0, failed_counts / D_safe, 0.0)

    # --- Apply thresholds via vectorised np.where (Φ function) ---
    has_signal = in_degree > 0
    new_state = np.zeros(len(S), dtype=np.int32)
    new_state = np.where(has_signal & (F >= theta_deg), STATE_DEGRADED, new_state)
    new_state = np.where(has_signal & (F >= theta_fail), STATE_FAILED, new_state)

    # --- Monotonicity: S_{t+1} = max(S_t, new_state) ---
    return np.maximum(S, new_state).astype(np.int32)


def propagation_step_fast(
    S: np.ndarray,
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    use_numba: bool = True,
) -> np.ndarray:
    """Dispatch to the Numba JIT or SciPy sparse path based on availability.

    Parameters
    ----------
    S : np.ndarray, shape (n,), dtype int32
        Current state vector.
    A_T : csr_matrix, shape (n, n)
        Transpose adjacency matrix in CSR.
    in_degree : np.ndarray, shape (n,), dtype float64
    theta_deg : np.ndarray, shape (n,), dtype float64
    theta_fail : np.ndarray, shape (n,), dtype float64
    use_numba : bool, optional
        If ``True`` (default), use the Numba JIT kernel when available.

    Returns
    -------
    np.ndarray, shape (n,), dtype int32
    """
    if use_numba and _HAS_NUMBA:
        failed_vec = (S == STATE_FAILED).astype(np.float64)
        return _update_kernel_numba(
            A_T.data.astype(np.float32),
            A_T.indices.astype(np.int32),
            A_T.indptr.astype(np.int32),
            failed_vec,
            in_degree.astype(np.float64),
            theta_deg.astype(np.float64),
            theta_fail.astype(np.float64),
            S.astype(np.int32),
        )
    return propagation_step_sparse(S, A_T, in_degree, theta_deg, theta_fail)


# ---------------------------------------------------------------------------
# Full deterministic cascade runner
# ---------------------------------------------------------------------------


def run_until_stable_fast(
    S0: np.ndarray,
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    max_steps: Optional[int] = None,
    use_numba: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[np.ndarray, int, np.ndarray, bool]:
    """Run the deterministic cascade until convergence using the sparse engine.

    Implements the linear-algebra rule ``S_{t+1} = Φ(A_T·S_t − Θ)`` and is
    designed to handle 100K-node / 1M-edge graphs within seconds.

    Parameters
    ----------
    S0 : np.ndarray, shape (n,), dtype int-like
        Initial state vector.  Values in {0, 1, 2}.
    A_T : csr_matrix, shape (n, n)
        Transpose adjacency matrix from ``graph_sparse.build_sparse_graph``.
    in_degree : np.ndarray, shape (n,), dtype float64
        In-degree per node.
    theta_deg : np.ndarray, shape (n,), dtype float64
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,), dtype float64
        Failure threshold per node.
    max_steps : int or None, optional
        Cap on propagation steps.  Defaults to ``2 * n`` (theoretical maximum
        for monotone systems).
    use_numba : bool, optional
        Use the Numba JIT kernel (default ``True``).  Set to ``False`` to
        force the SciPy sparse path (e.g. for profiling comparisons).
    progress_callback : ProgressCallback or None, optional
        Called with ``(percent, message)`` after each propagation step.

    Returns
    -------
    final_state : np.ndarray, shape (n,), dtype int32
    time_to_stability : int
        Step index of the last actual state change.
    full_state_history : np.ndarray, shape (T+1, n), dtype int32
        State at every step that produced at least one change, plus S0.
    convergence_reached : bool

    Raises
    ------
    MonotonicityViolation
        If any node's state decreases between steps (indicates a bug).
    ValueError
        On shape mismatches.
    """
    n = A_T.shape[0]
    _validate_inputs(n, S0, theta_deg, theta_fail)

    if max_steps is None:
        max_steps = 2 * n

    cb = progress_callback or null_callback

    # Pre-cast arrays to the expected dtypes once (avoids repeated casting)
    at_data = A_T.data.astype(np.float32)
    at_indices = A_T.indices.astype(np.int32)
    at_indptr = A_T.indptr.astype(np.int32)
    D = in_degree.astype(np.float64)
    t_deg = theta_deg.astype(np.float64)
    t_fail = theta_fail.astype(np.float64)

    S_current = np.array(S0, dtype=np.int32)
    history: list[np.ndarray] = [S_current.copy()]
    last_change_step = 0
    convergence_reached = False

    for step in range(max_steps):
        if use_numba and _HAS_NUMBA:
            failed_vec = (S_current == STATE_FAILED).astype(np.float64)
            S_next = _update_kernel_numba(
                at_data, at_indices, at_indptr,
                failed_vec, D, t_deg, t_fail, S_current,
            )
        else:
            S_next = propagation_step_sparse(
                S_current, A_T, D, t_deg, t_fail
            )

        # Monotonicity guard
        if np.any(S_next < S_current):
            violators = np.where(S_next < S_current)[0].tolist()
            raise MonotonicityViolation(
                f"Monotonicity violated at step {step + 1} for nodes: {violators}"
            )

        pct = 100.0 * (step + 1) / max_steps
        cb(pct, f"Step {step + 1} — nodes changed: {int(np.sum(S_next != S_current))}")

        history.append(S_next.copy())

        if np.array_equal(S_next, S_current):
            history.pop()
            convergence_reached = True
            cb(100.0, f"Converged at step {step + 1}")
            break

        last_change_step = step + 1
        S_current = S_next

    else:
        warnings.warn(
            f"run_until_stable_fast: max_steps={max_steps} reached without convergence.",
            RuntimeWarning,
            stacklevel=2,
        )

    return (
        S_current,
        last_change_step,
        np.array(history, dtype=np.int32),
        convergence_reached,
    )


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------


def _validate_inputs(
    n: int,
    S0: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
) -> None:
    if S0.shape != (n,):
        raise ValueError(f"S0 must have shape ({n},); got {S0.shape}.")
    if theta_deg.shape != (n,):
        raise ValueError(f"theta_deg must have shape ({n},); got {theta_deg.shape}.")
    if theta_fail.shape != (n,):
        raise ValueError(f"theta_fail must have shape ({n},); got {theta_fail.shape}.")


# ---------------------------------------------------------------------------
# Numba warm-up (runs on module import to amortise first-call compilation)
# ---------------------------------------------------------------------------


def _warmup_numba() -> None:
    """Run the Numba JIT kernel on a tiny graph to trigger compilation.

    This amortises the ~0.5–2s LLVM compilation time before any real run.
    Called automatically at the bottom of this module.
    """
    if not _HAS_NUMBA:
        return
    _n = 4
    _at_data = np.ones(4, dtype=np.float32)
    _at_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    _at_indptr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    _failed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _D = np.ones(_n, dtype=np.float64)
    _t_d = np.full(_n, 0.3, dtype=np.float64)
    _t_f = np.full(_n, 0.6, dtype=np.float64)
    _S = np.array([2, 0, 0, 0], dtype=np.int32)
    _update_kernel_numba(
        _at_data, _at_indices, _at_indptr,
        _failed, _D, _t_d, _t_f, _S,
    )


# Trigger Numba compilation at import time.
# This runs in a background thread so it does not block the caller.
import threading as _threading

_warmup_thread = _threading.Thread(target=_warmup_numba, daemon=True, name="numba-warmup")
_warmup_thread.start()
