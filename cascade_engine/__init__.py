"""
cascade_engine — Threshold-Based Cascade Propagation Engine
============================================================

Supports two propagation modes:

  Tier 1 (Deterministic)
      Hard-threshold update rule with monotonic three-state logic.
      Fully reproducible given a fixed graph and threshold arrays.

  Tier 2 (Stochastic)
      Logistic (sigmoid) failure probability with Monte Carlo wrapper.
      Per-trial RNG isolation via SeedSequence for statistical independence.

High-performance path (100K+ nodes / 1M+ edges)
------------------------------------------------
For large real-world graphs, use the fast sub-package:

>>> from cascade_engine.ingestion import ingest_edgelist
>>> from cascade_engine.graph_sparse import build_or_load_sparse_graph
>>> from cascade_engine.propagation_fast import run_until_stable_fast
>>> from cascade_engine.monte_carlo_parallel import run_monte_carlo_parallel
>>> from cascade_engine.progress import print_callback

Quick start (classic synthetic graphs)
---------------------------------------
>>> from cascade_engine.propagation import run_until_stable
>>> from cascade_engine.graph import generate_erdos_renyi
>>> import numpy as np
>>> A = generate_erdos_renyi(n=50, p=0.1, seed=42)
>>> n = A.shape[0]
>>> theta_deg  = np.full(n, 0.3)
>>> theta_fail = np.full(n, 0.6)
>>> S0 = np.zeros(n, dtype=np.int32)
>>> S0[0] = 2  # seed node 0 as failed
>>> final, steps, history, converged = run_until_stable(S0, A, theta_deg, theta_fail)
"""

# ---------------------------------------------------------------------------
# Legacy API (Tier 1 + Tier 2 — unchanged, backward compatible)
# ---------------------------------------------------------------------------
from .propagation import (
    run_until_stable,
    propagation_step,
    MonotonicityViolation,
    STATE_OPERATIONAL,
    STATE_DEGRADED,
    STATE_FAILED,
)
from .stochastic_propagation import (
    run_until_stable_stochastic,
    propagation_step_stochastic,
    sigmoid,
)
from .monte_carlo import run_monte_carlo, run_monte_carlo_all_seeds, MonteCarloResult
from .metrics import fragility_index, fragility_summary, cascade_size
from .sensitivity import threshold_sensitivity, SensitivityPoint
from .graph import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_custom,
)
from .utils import confidence_interval, make_trial_rngs, make_node_trial_rngs

# ---------------------------------------------------------------------------
# High-performance API (sparse engine — new in this refactor)
# ---------------------------------------------------------------------------
from .ingestion import ingest_edgelist, IngestResult, save_lookup_table, load_lookup_table
from .graph_sparse import (
    build_sparse_graph,
    build_or_load_sparse_graph,
    sparse_from_dense,
    cache_sparse_graph,
    load_sparse_graph,
    graph_info,
)
from .propagation_fast import (
    run_until_stable_fast,
    propagation_step_fast,
    propagation_step_sparse,
)
from .monte_carlo_parallel import (
    run_monte_carlo_parallel,
    run_monte_carlo_all_seeds_parallel,
)
from .progress import (
    ProgressTracker,
    ProgressCallback,
    MultiprocessingProgressProxy,
    ProgressListener,
    null_callback,
    print_callback,
    make_tqdm_callback,
)

__all__ = [
    # ------------------------------------------------------------------ #
    # Legacy — propagation                                                 #
    # ------------------------------------------------------------------ #
    "run_until_stable", "propagation_step", "MonotonicityViolation",
    "STATE_OPERATIONAL", "STATE_DEGRADED", "STATE_FAILED",
    # Legacy — stochastic
    "run_until_stable_stochastic", "propagation_step_stochastic", "sigmoid",
    # Legacy — monte carlo
    "run_monte_carlo", "run_monte_carlo_all_seeds", "MonteCarloResult",
    # Legacy — metrics
    "fragility_index", "fragility_summary", "cascade_size",
    # Legacy — sensitivity
    "threshold_sensitivity", "SensitivityPoint",
    # Legacy — graph generators
    "generate_erdos_renyi", "generate_barabasi_albert",
    "generate_watts_strogatz", "generate_custom",
    # Legacy — utils
    "confidence_interval", "make_trial_rngs", "make_node_trial_rngs",
    # ------------------------------------------------------------------ #
    # High-performance sparse engine                                       #
    # ------------------------------------------------------------------ #
    # Ingestion
    "ingest_edgelist", "IngestResult", "save_lookup_table", "load_lookup_table",
    # Sparse graph
    "build_sparse_graph", "build_or_load_sparse_graph", "sparse_from_dense",
    "cache_sparse_graph", "load_sparse_graph", "graph_info",
    # Fast propagation
    "run_until_stable_fast", "propagation_step_fast", "propagation_step_sparse",
    # Parallel Monte Carlo
    "run_monte_carlo_parallel", "run_monte_carlo_all_seeds_parallel",
    # Progress
    "ProgressTracker", "ProgressCallback", "MultiprocessingProgressProxy",
    "ProgressListener", "null_callback", "print_callback", "make_tqdm_callback",
]
