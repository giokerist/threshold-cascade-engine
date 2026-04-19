"""
monte_carlo_parallel.py — Multiprocessing Monte Carlo Engine
=============================================================

Distributes Monte Carlo iterations across all available CPU cores using
``concurrent.futures.ProcessPoolExecutor``.  Each worker process handles an
independent batch of seed integers, with its own statistically-isolated RNG
derived via ``numpy.random.SeedSequence.spawn()``.

Architecture
------------
::

    Main Process
      │
      ├── build_or_load_sparse_graph(...)        ← one-time, cached
      │
      ├── ProcessPoolExecutor (n workers)
      │     ├── _worker_init(A_T_arrays, thresholds, …)  [one call per worker]
      │     │   └── stores graph arrays in module-level global (no re-pickle)
      │     │
      │     ├── Worker 0: _run_mc_batch(seed_node, seeds[0:250])          # run_monte_carlo_parallel
      │     ├── Worker 1: _run_node_trials(node_idx=3, node_seed, trials) # run_monte_carlo_all_seeds_parallel
      │     └── ...
      │
      └── collects results + fires progress_callback

Serialisation notes
-------------------
* The CSR matrix is never pickled as a SciPy object — only its raw
  ``(data, indices, indptr)`` NumPy arrays are passed to the initializer.
  NumPy arrays use a compact buffer protocol, so a 1M-edge graph is ~12 MB
  of initializer payload (once per worker, not once per task).
* Within each worker the stochastic propagation uses the pure-NumPy / SciPy
  path (``propagation_step_stochastic`` from ``stochastic_propagation.py``)
  because Numba cannot be shared across ``fork``-spawned processes safely
  without a re-compilation step.  The per-trial cost is dominated by the
  ``A_T.dot()`` sparse matvec, which is BLAS-backed and very fast.

Progress reporting
------------------
The main process fires ``progress_callback(pct, msg)`` in the ``as_completed``
loop, once per completed batch.  Granularity = 100% / n_batches.

Usage
-----
>>> from cascade_engine.monte_carlo_parallel import run_monte_carlo_parallel
>>> result = run_monte_carlo_parallel(
...     A_T, in_degree, theta_deg, theta_fail,
...     seed_node=42, trials=1000, seed=0,
...     n_workers=4, progress_callback=my_cb,
... )
>>> print(result.summary_dict())
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy.sparse import csr_matrix

from .monte_carlo import MonteCarloResult
from .propagation import STATE_DEGRADED
from .utils import confidence_interval
from .progress import ProgressCallback, null_callback


# ---------------------------------------------------------------------------
# Worker-process global state (populated by _worker_init)
# ---------------------------------------------------------------------------

# Module-level dict — set once per worker process by the initializer.
# Using a plain dict (not a global) avoids name collisions with reload.
_WORKER: dict = {}


def _worker_init(
    at_data: np.ndarray,
    at_indices: np.ndarray,
    at_indptr: np.ndarray,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    n: int,
    k: float,
    max_steps: int,
    consecutive_stable_steps: int,
) -> None:
    """Called once per worker process to store graph + config in module globals.

    Parameters are NumPy arrays (fast pickle via buffer protocol).  The CSR
    matrix is reconstructed from its three component arrays so that
    ``_WORKER["A_T"]`` is a valid ``csr_matrix`` inside the worker.
    """
    from scipy.sparse import csr_matrix as _csr

    A_T_local = _csr(
        (at_data, at_indices, at_indptr),
        shape=(n, n),
        dtype=np.float32,
    )
    A_T_local.sort_indices()

    _WORKER["A_T"] = A_T_local
    _WORKER["in_degree"] = in_degree
    _WORKER["theta_deg"] = theta_deg
    _WORKER["theta_fail"] = theta_fail
    _WORKER["n"] = n
    _WORKER["k"] = k
    _WORKER["max_steps"] = max_steps
    _WORKER["css"] = consecutive_stable_steps


def _propagation_step_stochastic_sparse(
    S: np.ndarray,
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    k: float,
    rng,
) -> np.ndarray:
    """One stochastic propagation step using A_T.dot() (sparse-native).

    This is the worker-local replacement for ``propagation_step_stochastic``
    from ``stochastic_propagation.py``, which expects a dense matrix and calls
    ``A.T @ vec``.  Here we work directly with A_T (the CSR transpose) and
    call ``A_T.dot(vec)`` — equivalent but sparse-safe.

    Degradation: hard threshold (deterministic).
    Failure:     logistic probability sampling.
    """
    from scipy.special import expit as _expit

    STATE_FAILED_LOCAL = 2
    STATE_DEGRADED_LOCAL = 1

    # Failed fraction F_i = (A_T · failed_mask)[i] / in_degree[i]
    failed_mask = (S == STATE_FAILED_LOCAL).astype(np.float64)
    failed_counts = A_T.dot(failed_mask)                  # CSR matvec (n,)
    D_safe = np.where(in_degree > 0, in_degree, 1.0)
    F = np.where(in_degree > 0, failed_counts / D_safe, 0.0)

    # Degradation: hard threshold (Tier 2 axiom — deterministic)
    will_degrade = (F >= theta_deg) & (in_degree > 0)

    # Failure: logistic probability
    eps = np.finfo(np.float64).eps
    p_fail = np.clip(_expit(k * (F - theta_fail)), np.finfo(np.float64).tiny, 1.0 - eps)
    p_fail = np.where(in_degree > 0, p_fail, 0.0)
    u = rng.random(len(S))
    will_fail = u < p_fail

    # Build candidate state and enforce monotonicity
    new_state = np.zeros(len(S), dtype=np.int32)
    new_state = np.where(will_degrade, STATE_DEGRADED_LOCAL, new_state)
    new_state = np.where(will_fail, STATE_FAILED_LOCAL, new_state)
    return np.maximum(S, new_state).astype(np.int32)


def _run_stochastic_sparse(
    S0: np.ndarray,
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    k: float,
    rng,
    max_steps: int,
    css: int,
) -> Tuple[np.ndarray, int]:
    """Run stochastic cascade to convergence using A_T.dot() (worker-local).

    Returns (final_state, time_to_stability).
    """
    STATE_FAILED_LOCAL = 2
    n = len(S0)
    S_cur = S0.copy()
    last_change = 0
    quiet = 0

    for step in range(max_steps):
        S_next = _propagation_step_stochastic_sparse(
            S_cur, A_T, in_degree, theta_deg, theta_fail, k, rng
        )
        if np.array_equal(S_next, S_cur):
            quiet += 1
            if quiet >= css:
                break
        else:
            quiet = 0
            last_change = step + 1
        S_cur = S_next

    return S_cur, last_change


def _run_mc_batch(
    args: Tuple[int, List[int]]
) -> List[Tuple[float, int]]:
    """Worker task: run a batch of seeds for a single seed node.

    Parameters
    ----------
    args : (seed_node, seeds)
        * ``seed_node`` — index of the node seeded as STATE_FAILED at t=0.
        * ``seeds``     — list of integer seeds for per-trial RNGs.

    Returns
    -------
    results : list of (cascade_size_fraction, time_to_stability)
        One tuple per seed in the batch.
    """
    seed_node, seeds = args
    g = _WORKER
    n: int = g["n"]
    A_T: csr_matrix = g["A_T"]
    in_degree: np.ndarray = g["in_degree"]
    theta_deg: np.ndarray = g["theta_deg"]
    theta_fail: np.ndarray = g["theta_fail"]
    k: float = g["k"]
    max_steps: int = g["max_steps"]
    css: int = g["css"]

    S0 = np.zeros(n, dtype=np.int32)
    S0[seed_node] = 2   # STATE_FAILED

    STATE_DEGRADED_LOCAL = 1
    results: List[Tuple[float, int]] = []
    for seed in seeds:
        rng = default_rng(seed)
        final_state, t_stable = _run_stochastic_sparse(
            S0, A_T, in_degree, theta_deg, theta_fail,
            k=k, rng=rng,
            max_steps=max_steps,
            css=css,
        )
        affected = int(np.sum(final_state >= STATE_DEGRADED_LOCAL))
        results.append((float(affected) / n, int(t_stable)))

    return results


def _run_node_trials(
    args: Tuple[int, int, int]
) -> Tuple[int, List[Tuple[float, int]]]:
    """Worker task: run all Monte Carlo trials for one node.

    Takes three plain integers — no list of seeds, no large pickled payload.
    The worker generates its own per-trial RNGs from ``node_seed`` via
    SeedSequence, keeping all seed arithmetic inside the worker process.

    Parameters
    ----------
    args : (node_idx, node_seed, trials)
        * ``node_idx``  — index of the node seeded as STATE_FAILED at t=0.
        * ``node_seed`` — integer seed for this node's SeedSequence.
        * ``trials``    — number of independent trials to run.

    Returns
    -------
    (node_idx, results)
        ``results`` is a list of ``(cascade_size_fraction, time_to_stability)``
        tuples, one per trial.
    """
    node_idx, node_seed, trials = args
    g = _WORKER
    n: int = g["n"]
    A_T: csr_matrix = g["A_T"]
    in_degree: np.ndarray = g["in_degree"]
    theta_deg: np.ndarray = g["theta_deg"]
    theta_fail: np.ndarray = g["theta_fail"]
    k: float = g["k"]
    max_steps: int = g["max_steps"]
    css: int = g["css"]

    # Generate per-trial RNGs locally — avoids pickling a list of seeds
    ss = SeedSequence(node_seed)
    trial_rngs = [default_rng(child) for child in ss.spawn(trials)]

    S0 = np.zeros(n, dtype=np.int32)
    S0[node_idx] = 2   # STATE_FAILED

    STATE_DEGRADED_LOCAL = 1
    results: List[Tuple[float, int]] = []
    for rng in trial_rngs:
        final_state, t_stable = _run_stochastic_sparse(
            S0, A_T, in_degree, theta_deg, theta_fail,
            k=k, rng=rng,
            max_steps=max_steps,
            css=css,
        )
        affected = int(np.sum(final_state >= STATE_DEGRADED_LOCAL))
        results.append((float(affected) / n, int(t_stable)))

    return node_idx, results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_monte_carlo_parallel(
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    seed_node: int,
    trials: int,
    seed: int,
    k: float = 10.0,
    max_steps: Optional[int] = None,
    consecutive_stable_steps: int = 3,
    n_workers: int = 4,
    progress_callback: Optional[ProgressCallback] = None,
) -> MonteCarloResult:
    """Run Monte Carlo trials in parallel across *n_workers* CPU cores.

    Each trial independently seeds *seed_node* as STATE_FAILED at t=0 and
    runs the stochastic propagation until convergence.  Trials are batched
    evenly across workers; each worker receives a single initializer call
    (cheap, ~1 × pickle of graph arrays) and then processes its batch
    sequentially.

    Parameters
    ----------
    A_T : csr_matrix, shape (n, n)
        Transpose adjacency matrix from ``graph_sparse.build_sparse_graph``.
    in_degree : np.ndarray, shape (n,), dtype float64
        In-degree per node.
    theta_deg : np.ndarray, shape (n,), dtype float64
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,), dtype float64
        Failure threshold per node.
    seed_node : int
        Index of the node seeded as STATE_FAILED at t=0.
    trials : int
        Total number of independent Monte Carlo trials (>= 2).
    seed : int
        Master random seed.  Per-trial seeds are derived via two-level
        ``SeedSequence.spawn()`` for guaranteed statistical independence.
    k : float, optional
        Logistic steepness parameter (default 10.0).
    max_steps : int or None, optional
        Per-trial step cap.  Defaults to ``4 * n``.
    consecutive_stable_steps : int, optional
        Quiet steps required for convergence (default 3).
    n_workers : int, optional
        Number of parallel worker processes.  Default 4 (for a 4-core box).
        Set to 1 for single-process debugging.
    progress_callback : ProgressCallback or None, optional
        Called as ``(percent, message)`` from the main process after each
        worker batch completes.  Batch granularity = 100% / n_workers.

    Returns
    -------
    MonteCarloResult
        Frozen dataclass with raw distributions and aggregate statistics.

    Raises
    ------
    ValueError
        If ``trials < 2`` or ``seed_node`` is out of range.
    """
    n = A_T.shape[0]
    if trials < 2:
        raise ValueError(f"trials must be >= 2; got {trials}.")
    if not (0 <= seed_node < n):
        raise ValueError(f"seed_node {seed_node} out of range [0, {n - 1}].")

    if max_steps is None:
        max_steps = 4 * n

    cb = progress_callback or null_callback

    # ------------------------------------------------------------------
    # 1. Derive per-trial seeds via two-level SeedSequence hierarchy
    #    (guarantees statistical independence — F-006 pattern)
    # ------------------------------------------------------------------
    root_ss = SeedSequence(seed)
    trial_child_seeds: list[int] = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in root_ss.spawn(trials)
    ]

    # ------------------------------------------------------------------
    # 2. Split trials into n_workers batches (as even as possible)
    # ------------------------------------------------------------------
    n_workers = max(1, min(n_workers, trials))
    batch_size = (trials + n_workers - 1) // n_workers  # ceiling division
    batches: list[list[int]] = []
    for i in range(0, trials, batch_size):
        batches.append(trial_child_seeds[i : i + batch_size])

    # ------------------------------------------------------------------
    # 3. Prepare initializer arguments (raw numpy arrays — fast pickle)
    # ------------------------------------------------------------------
    at_data = A_T.data.astype(np.float32)
    at_indices = A_T.indices.astype(np.int32)
    at_indptr = A_T.indptr.astype(np.int32)
    D = in_degree.astype(np.float64)
    t_deg = theta_deg.astype(np.float64)
    t_fail = theta_fail.astype(np.float64)

    init_args = (
        at_data, at_indices, at_indptr,
        D, t_deg, t_fail,
        n, float(k), int(max_steps), int(consecutive_stable_steps),
    )

    # ------------------------------------------------------------------
    # 4. Dispatch batches to worker pool
    # ------------------------------------------------------------------
    # Use 'fork' start method on Linux for faster worker startup.
    # On macOS / Windows the default ('spawn') is used automatically.
    _ctx = _get_mp_context()

    cascade_sizes = np.empty(trials, dtype=np.float64)
    times_to_stability = np.empty(trials, dtype=np.int64)
    idx = 0

    cb(0.0, f"Dispatching {trials} trials across {len(batches)} batches …")

    if n_workers == 1 or trials <= 8:
        # Serial fallback — useful for debugging and very small trial counts
        _worker_init(*init_args)
        for b_idx, batch in enumerate(batches):
            batch_results = _run_mc_batch((seed_node, batch))
            for cs, ts in batch_results:
                cascade_sizes[idx] = cs
                times_to_stability[idx] = ts
                idx += 1
            pct = 100.0 * (b_idx + 1) / len(batches)
            cb(pct, f"Batch {b_idx + 1}/{len(batches)} done ({len(batch)} trials)")
    else:
        # Parallel path
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_ctx,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            # Map each batch to a future, keep batch index for ordering
            future_to_batch: dict = {
                pool.submit(_run_mc_batch, (seed_node, batch)): b_idx
                for b_idx, batch in enumerate(batches)
            }

            # Collect results as futures complete (order may vary)
            batch_results_ordered = [None] * len(batches)
            for future in as_completed(future_to_batch):
                b_idx = future_to_batch[future]
                try:
                    batch_results_ordered[b_idx] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Worker batch {b_idx} failed: {exc}"
                    ) from exc

                done_count = sum(
                    1 for r in batch_results_ordered if r is not None
                )
                pct = 100.0 * done_count / len(batches)
                cb(pct, f"Batch {done_count}/{len(batches)} done")

        # Flatten results preserving per-batch ordering
        for batch_res in batch_results_ordered:
            for cs, ts in batch_res:
                cascade_sizes[idx] = cs
                times_to_stability[idx] = ts
                idx += 1

    assert idx == trials, f"Expected {trials} results, got {idx}"

    # ------------------------------------------------------------------
    # 5. Aggregate statistics
    # ------------------------------------------------------------------
    mean_cs = float(np.mean(cascade_sizes))
    var_cs = float(np.var(cascade_sizes, ddof=1))
    ci_low, ci_high = confidence_interval(cascade_sizes)

    cb(100.0, "Monte Carlo complete")

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


def run_monte_carlo_all_seeds_parallel(
    A_T: csr_matrix,
    in_degree: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    trials: int,
    seed: int,
    k: float = 10.0,
    max_steps: Optional[int] = None,
    consecutive_stable_steps: int = 3,
    n_workers: int = 4,
    progress_callback: Optional[ProgressCallback] = None,
) -> list:
    """Run parallel Monte Carlo for every node as initial failure seed.

    Distributes **both** the node-iteration loop and the per-node trial
    batches across ``n_workers`` processes.  For *n* nodes × *trials*
    trials this is the most compute-intensive operation in the engine.

    Parameters
    ----------
    A_T : csr_matrix
    in_degree : np.ndarray
    theta_deg : np.ndarray
    theta_fail : np.ndarray
    trials : int
    seed : int
    k : float, optional
    max_steps : int or None, optional
    consecutive_stable_steps : int, optional
    n_workers : int, optional
    progress_callback : ProgressCallback or None, optional

    Returns
    -------
    list of MonteCarloResult
        One result per node, in node-index order.
    """
    n = A_T.shape[0]
    cb = progress_callback or null_callback
    results: list = []

    # ------------------------------------------------------------------
    # 1. Derive one integer seed per node — O(n) SeedSequence ops, fast.
    #    Per-trial seeds are generated inside each worker, not here.
    # ------------------------------------------------------------------
    root_ss = SeedSequence(seed)
    node_seeds: list[int] = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in root_ss.spawn(n)
    ]

    # ------------------------------------------------------------------
    # 2. One task per node — payload is three plain ints, trivial to pickle.
    #    Total tasks = n (not n × n_workers), submission is instantaneous.
    # ------------------------------------------------------------------
    if max_steps is None:
        max_steps = 4 * n

    task_args = [(node_idx, node_seeds[node_idx], trials) for node_idx in range(n)]
    total_tasks = n
    cb(0.0, f"Submitting {total_tasks:,} node tasks across {n_workers} workers…")

    # ------------------------------------------------------------------
    # 3. Prepare CSR raw arrays for the pool initializer (sent once)
    # ------------------------------------------------------------------
    at_data    = A_T.data.astype(np.float32)
    at_indices = A_T.indices.astype(np.int32)
    at_indptr  = A_T.indptr.astype(np.int32)
    D          = in_degree.astype(np.float64)
    t_deg      = theta_deg.astype(np.float64)
    t_fail     = theta_fail.astype(np.float64)

    init_args = (
        at_data, at_indices, at_indptr,
        D, t_deg, t_fail,
        n, float(k), int(max_steps), int(consecutive_stable_steps),
    )

    # ------------------------------------------------------------------
    # 4. Single pool — all node tasks submitted at once, collected as done
    # ------------------------------------------------------------------
    raw_results: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    completed = 0
    _ctx = _get_mp_context()

    if n_workers == 1:
        _worker_init(*init_args)
        for args in task_args:
            node_idx, node_results = _run_node_trials(args)
            raw_results[node_idx].extend(node_results)
            completed += 1
            pct = 100.0 * completed / total_tasks
            cb(pct, f"Node {completed:,}/{n:,} — {pct:.1f}% complete")
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_ctx,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            future_to_node = {
                pool.submit(_run_node_trials, args): args[0]
                for args in task_args
            }
            for future in as_completed(future_to_node):
                try:
                    node_idx, node_results = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Worker failed on node {future_to_node[future]}: {exc}"
                    ) from exc
                raw_results[node_idx].extend(node_results)
                completed += 1
                pct = 100.0 * completed / total_tasks
                cb(
                    pct,
                    f"Node {completed:,}/{n:,} — {pct:.1f}% complete",
                )

    cb(100.0, "All-seeds Monte Carlo complete")

    # ------------------------------------------------------------------
    # 5. Assemble one MonteCarloResult per node
    # ------------------------------------------------------------------
    mc_results: list = []
    for node_idx in range(n):
        pairs = raw_results[node_idx]
        cascade_sizes = np.array([p[0] for p in pairs], dtype=np.float64)
        times         = np.array([p[1] for p in pairs], dtype=np.int64)
        mean_cs = float(np.mean(cascade_sizes))
        var_cs  = float(np.var(cascade_sizes, ddof=1)) if len(cascade_sizes) > 1 else 0.0
        ci_low, ci_high = (
            confidence_interval(cascade_sizes)
            if len(cascade_sizes) >= 2 else (mean_cs, mean_cs)
        )
        mc_results.append(MonteCarloResult(
            cascade_sizes=cascade_sizes,
            times_to_stability=times,
            mean_cascade_size=mean_cs,
            variance_cascade_size=var_cs,
            ci_low=ci_low,
            ci_high=ci_high,
            trials=len(pairs),
            seed=seed,
            n_nodes=n,
            k=k,
            consecutive_stable_steps=consecutive_stable_steps,
        ))

    return mc_results


# ---------------------------------------------------------------------------
# Multiprocessing context helper
# ---------------------------------------------------------------------------


def _get_mp_context():
    """Return a ``multiprocessing`` context appropriate for the OS.

    * Linux   → ``'fork'``  (fastest; COW semantics mean no extra data copy)
    * macOS   → ``'spawn'`` (fork is unsafe with multi-threaded processes)
    * Windows → ``'spawn'`` (fork not available)
    """
    import platform
    system = platform.system()
    if system == "Linux":
        return _mp.get_context("fork")
    else:
        return _mp.get_context("spawn")
