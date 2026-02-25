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

Quick start
-----------
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

__all__ = [
    # propagation
    "run_until_stable", "propagation_step", "MonotonicityViolation",
    "STATE_OPERATIONAL", "STATE_DEGRADED", "STATE_FAILED",
    # stochastic propagation
    "run_until_stable_stochastic", "propagation_step_stochastic", "sigmoid",
    # monte carlo
    "run_monte_carlo", "run_monte_carlo_all_seeds", "MonteCarloResult",
    # metrics
    "fragility_index", "fragility_summary", "cascade_size",
    # sensitivity
    "threshold_sensitivity", "SensitivityPoint",
    # graph
    "generate_erdos_renyi", "generate_barabasi_albert",
    "generate_watts_strogatz", "generate_custom",
    # utils
    "confidence_interval", "make_trial_rngs", "make_node_trial_rngs",
]
