"""
Configuration loader for the cascade propagation engine.

Loads JSON config files, validates fields, and generates threshold arrays
reproducibly using numpy random Generator with a fixed seed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ConfigDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

GRAPH_TYPES = {"erdos_renyi", "barabasi_albert", "watts_strogatz", "custom"}
THRESHOLD_TYPES = {"uniform", "normal"}
PROPAGATION_MODES = {"deterministic", "stochastic"}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ConfigDict:
    """Load and validate a JSON configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    ConfigDict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If required fields are missing or values are invalid.
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as fh:
        cfg: ConfigDict = json.load(fh)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ConfigDict) -> None:
    """Validate top-level config fields.

    Parameters
    ----------
    cfg : ConfigDict
        Raw configuration dictionary to validate.

    Raises
    ------
    ValueError
        On any validation failure.
    """
    required_top = {"graph", "thresholds", "seed", "propagation_mode"}
    missing = required_top - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    graph_cfg = cfg["graph"]
    if "type" not in graph_cfg:
        raise ValueError("graph.type is required")
    if graph_cfg["type"] not in GRAPH_TYPES:
        raise ValueError(
            f"graph.type must be one of {GRAPH_TYPES}, got {graph_cfg['type']!r}"
        )

    # Type-specific required parameters
    _GRAPH_REQUIRED_PARAMS: dict[str, list[str]] = {
        "erdos_renyi":      ["n", "p"],
        "barabasi_albert":  ["n", "m"],
        "watts_strogatz":   ["n", "k", "p"],
        "custom":           ["n", "edges"],
    }
    gtype = graph_cfg["type"]
    missing_graph_params = [
        p for p in _GRAPH_REQUIRED_PARAMS[gtype] if p not in graph_cfg
    ]
    if missing_graph_params:
        raise ValueError(
            f"graph config for type {gtype!r} is missing required "
            f"parameter(s): {missing_graph_params}"
        )

    thresh_cfg = cfg["thresholds"]
    if "type" not in thresh_cfg:
        raise ValueError("thresholds.type is required")
    if thresh_cfg["type"] not in THRESHOLD_TYPES:
        raise ValueError(
            f"thresholds.type must be one of {THRESHOLD_TYPES}, got {thresh_cfg['type']!r}"
        )

    if cfg["propagation_mode"] not in PROPAGATION_MODES:
        raise ValueError(
            f"propagation_mode must be one of {PROPAGATION_MODES}, "
            f"got {cfg['propagation_mode']!r}"
        )


# ---------------------------------------------------------------------------
# Threshold generation
# ---------------------------------------------------------------------------


def generate_thresholds(
    n: int,
    thresh_cfg: ConfigDict,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate degradation and failure threshold arrays.

    Parameters
    ----------
    n : int
        Number of nodes.
    thresh_cfg : ConfigDict
        Threshold sub-config with keys ``type``, and distribution parameters.
        For ``uniform``: ``deg_low``, ``deg_high``, ``fail_low``, ``fail_high``.
        For ``normal``: ``deg_mean``, ``deg_std``, ``fail_mean``, ``fail_std``.
    rng : Generator
        Seeded numpy random Generator for reproducibility.

    Returns
    -------
    theta_deg : np.ndarray, shape (n,)
        Degradation thresholds clipped to [0, 1].
    theta_fail : np.ndarray, shape (n,)
        Failure thresholds clipped to [0, 1], always >= theta_deg element-wise.
        A ``UserWarning`` is emitted for any node where the raw draw required
        correction (theta_fail raised to match theta_deg).

    Raises
    ------
    ValueError
        If an unsupported threshold type is specified.
    """
    t = thresh_cfg["type"]

    if t == "uniform":
        theta_deg = rng.uniform(
            thresh_cfg.get("deg_low", 0.1),
            thresh_cfg.get("deg_high", 0.5),
            size=n,
        )
        theta_fail = rng.uniform(
            thresh_cfg.get("fail_low", 0.5),
            thresh_cfg.get("fail_high", 0.9),
            size=n,
        )
    elif t == "normal":
        theta_deg = rng.normal(
            thresh_cfg.get("deg_mean", 0.3),
            thresh_cfg.get("deg_std", 0.1),
            size=n,
        )
        theta_fail = rng.normal(
            thresh_cfg.get("fail_mean", 0.6),
            thresh_cfg.get("fail_std", 0.1),
            size=n,
        )
    else:
        raise ValueError(f"Unsupported threshold type: {t!r}")

    theta_deg = np.clip(theta_deg, 0.0, 1.0)
    theta_fail = np.clip(theta_fail, 0.0, 1.0)

    # Enforce theta_fail >= theta_deg element-wise.
    # Where the raw draw violates this (theta_fail < theta_deg), we correct by
    # clamping theta_fail up to theta_deg.  This can happen with normal
    # distributions whose tails overlap, or with misconfigured uniform ranges.
    # We emit a warning so callers are aware their thresholds were modified.
    violation_mask = theta_fail < theta_deg
    n_violations = int(np.sum(violation_mask))
    if n_violations > 0:
        violating_nodes = np.where(violation_mask)[0].tolist()
        warnings.warn(
            f"generate_thresholds: theta_fail < theta_deg for {n_violations} node(s) "
            f"(indices: {violating_nodes[:10]}{'...' if n_violations > 10 else ''}). "
            f"theta_fail has been raised to match theta_deg for those nodes. "
            f"To suppress this, ensure fail_low >= deg_high (uniform) or "
            f"fail_mean - 3*fail_std >= deg_mean + 3*deg_std (normal).",
            UserWarning,
            stacklevel=2,
        )
        theta_fail = np.maximum(theta_fail, theta_deg)

    # Warn when any threshold is exactly 0.0: nodes with in-edges and a zero
    # threshold will transition immediately at step 1 even with no failed
    # in-neighbors (F_i >= 0 is always true).  This is mathematically correct
    # but almost never the intended behaviour in infrastructure models.
    n_zero_deg  = int(np.sum(theta_deg  == 0.0))
    n_zero_fail = int(np.sum(theta_fail == 0.0))
    if n_zero_deg > 0 or n_zero_fail > 0:
        warnings.warn(
            f"generate_thresholds: {n_zero_deg} node(s) have theta_deg == 0.0 and "
            f"{n_zero_fail} node(s) have theta_fail == 0.0 after clipping. "
            f"Any such node with at least one in-edge will transition immediately "
            f"at step 1 regardless of neighbor states (F_i >= 0 is always satisfied). "
            f"Use theta_deg > 0 and theta_fail > 0 in realistic infrastructure models.",
            UserWarning,
            stacklevel=2,
        )

    return theta_deg, theta_fail


def build_rng(cfg: ConfigDict) -> Generator:
    """Build a seeded numpy Generator from a config dict.

    Parameters
    ----------
    cfg : ConfigDict
        Configuration dictionary containing ``seed`` (int).

    Returns
    -------
    Generator
        A seeded numpy random Generator.
    """
    return default_rng(int(cfg["seed"]))
