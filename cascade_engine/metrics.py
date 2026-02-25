"""
Metrics module for the cascade propagation engine.

All metrics are pure functions with no global state.  The fragility index
reuses the propagation engine cleanly via ``run_until_stable``.

Tier 2 additions
----------------
* ``rmse``                  – RMSE between predicted and observed vectors.
* ``mape``                  – Mean absolute percentage error.
* ``spearman_correlation``  – Spearman rank correlation via scipy.stats.
* ``confidence_interval``   – General CI utility (t-distribution), consistent with monte_carlo.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.stats as _scipy_stats

from .propagation import run_until_stable, STATE_FAILED, STATE_DEGRADED
from .utils import confidence_interval as _shared_confidence_interval


# ---------------------------------------------------------------------------
# Basic cascade metrics
# ---------------------------------------------------------------------------


def cascade_size(final_state: np.ndarray) -> dict[str, int | float]:
    """Compute cascade size statistics from a final state vector.

    Parameters
    ----------
    final_state : np.ndarray, shape (n,)
        Final state array with values in {0, 1, 2}.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``n_total`` : total number of nodes
        - ``n_degraded`` : nodes in state 1
        - ``n_failed`` : nodes in state 2
        - ``n_affected`` : nodes in state 1 or 2
        - ``frac_degraded`` : fraction of nodes degraded
        - ``frac_failed`` : fraction of nodes failed
        - ``frac_affected`` : fraction of nodes affected (degraded or failed)
    """
    n = int(final_state.shape[0])
    n_failed = int(np.sum(final_state == STATE_FAILED))
    n_degraded = int(np.sum(final_state == STATE_DEGRADED))
    n_affected = n_degraded + n_failed

    return {
        "n_total": n,
        "n_degraded": n_degraded,
        "n_failed": n_failed,
        "n_affected": n_affected,
        "frac_degraded": n_degraded / n if n > 0 else 0.0,
        "frac_failed": n_failed / n if n > 0 else 0.0,
        "frac_affected": n_affected / n if n > 0 else 0.0,
    }


def time_to_stability(history: np.ndarray) -> int:
    """Extract the number of propagation steps until stabilisation.

    .. deprecated::
        This function is redundant.  ``run_until_stable`` already returns
        ``time_to_stability`` directly as its second return value, computed
        as ``full_history.shape[0] - 1``.  Prefer reading that value directly
        rather than calling this wrapper.

    Parameters
    ----------
    history : np.ndarray, shape (T+1, n)
        Full state history as returned by ``run_until_stable``.  Row 0 is
        the initial state; subsequent rows are post-step states.

    Returns
    -------
    int
        Number of steps taken (= T).
    """
    warnings.warn(
        "metrics.time_to_stability() is deprecated and will be removed in a "
        "future release. Read run_until_stable()'s second return value directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return int(history.shape[0]) - 1


# ---------------------------------------------------------------------------
# Fragility index
# ---------------------------------------------------------------------------


def fragility_index(
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    max_steps: int | None = None,
) -> np.ndarray:
    """Compute the fragility index for every node.

    For each node i, the fragility index FI[i] is the total number of nodes
    that end up affected (state >= 1) when node i alone is seeded as failed
    (state = 2) and all others start operational (state = 0).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Adjacency matrix with convention A[j, i] = 1 iff j → i.
    theta_deg : np.ndarray, shape (n,)
        Degradation threshold per node.
    theta_fail : np.ndarray, shape (n,)
        Failure threshold per node.
    max_steps : int or None, optional
        Passed through to ``run_until_stable``.  Defaults to ``2 * n``.

    Returns
    -------
    np.ndarray, shape (n,), dtype int
        FI[i] = total affected (degraded + failed) nodes when i is seeded.
        Includes node i itself.

    Notes
    -----
    This function runs ``n`` independent cascades.  For large graphs this can
    be expensive; consider parallelisation at the caller level for n >> 1000.
    """
    n = A.shape[0]
    fi = np.empty(n, dtype=np.int64)

    for i in range(n):
        S0 = np.zeros(n, dtype=np.int32)
        S0[i] = STATE_FAILED

        final_state, _, _, _ = run_until_stable(
            S0, A, theta_deg, theta_fail, max_steps=max_steps
        )

        fi[i] = int(np.sum(final_state >= STATE_DEGRADED))

    return fi


# ---------------------------------------------------------------------------
# Aggregate fragility summary
# ---------------------------------------------------------------------------


def fragility_summary(fi: np.ndarray) -> dict[str, float]:
    """Summarise a fragility index vector.

    Parameters
    ----------
    fi : np.ndarray, shape (n,)
        Fragility index vector as returned by ``fragility_index``.

    Returns
    -------
    dict
        Summary statistics: ``mean``, ``std``, ``min``, ``max``, ``median``,
        ``p90`` (90th percentile).
    """
    return {
        "mean": float(np.mean(fi)),
        "std": float(np.std(fi)),
        "min": float(np.min(fi)),
        "max": float(np.max(fi)),
        "median": float(np.median(fi)),
        "p90": float(np.percentile(fi, 90)),
    }


# ---------------------------------------------------------------------------
# Tier 2: Statistical comparison metrics
# ---------------------------------------------------------------------------


def rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Compute Root Mean Square Error between predicted and observed values.

    Parameters
    ----------
    predicted : np.ndarray, shape (m,)
        Simulated / predicted cascade sizes (fractions or counts).
    observed : np.ndarray, shape (m,)
        Hardware-observed or reference cascade sizes.

    Returns
    -------
    float
        RMSE value (same units as inputs).

    Raises
    ------
    ValueError
        If arrays differ in shape or are empty.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError(
            f"predicted and observed must have the same shape; "
            f"got {predicted.shape} vs {observed.shape}."
        )
    if predicted.size == 0:
        raise ValueError("Input arrays must not be empty.")
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def mape(predicted: np.ndarray, observed: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error.

    MAPE = (1/m) * sum( |predicted_i - observed_i| / max(|observed_i|, eps) ) * 100

    Parameters
    ----------
    predicted : np.ndarray, shape (m,)
        Predicted values.
    observed : np.ndarray, shape (m,)
        Reference / observed values.
    eps : float, optional
        Small constant added to denominators to avoid division by zero for
        zero-valued observations (default 1e-8).

    Returns
    -------
    float
        MAPE in percent [0, ∞).

    Raises
    ------
    ValueError
        If arrays differ in shape or are empty.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError(
            f"predicted and observed must have the same shape; "
            f"got {predicted.shape} vs {observed.shape}."
        )
    if predicted.size == 0:
        raise ValueError("Input arrays must not be empty.")
    denom = np.maximum(np.abs(observed), eps)
    return float(np.mean(np.abs(predicted - observed) / denom) * 100.0)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute Spearman rank correlation coefficient and p-value.

    Used to compare simulated fragility rankings against observed impact
    rankings from hardware experiments.

    Parameters
    ----------
    x : np.ndarray, shape (m,)
        First ranking vector (e.g., simulated fragility indices).
    y : np.ndarray, shape (m,)
        Second ranking vector (e.g., observed cascade sizes).

    Returns
    -------
    dict with keys:
        - ``rho``     : Spearman correlation coefficient in [-1, 1].
        - ``p_value`` : Two-tailed p-value for the null hypothesis rho == 0.

    Raises
    ------
    ValueError
        If arrays differ in shape or have fewer than 3 elements.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape} vs {y.shape}."
        )
    if x.size < 3:
        raise ValueError("Spearman correlation requires at least 3 elements.")
    result = _scipy_stats.spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def confidence_interval(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a confidence interval for the population mean via t-distribution.

    Delegates to utils.confidence_interval (F-004 fix: single source of truth).

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
    """
    return _shared_confidence_interval(samples, confidence)
