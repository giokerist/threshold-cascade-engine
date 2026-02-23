"""
Configuration loader for the cascade propagation engine.

Loads JSON config files, validates fields, and generates threshold arrays
reproducibly using numpy random Generator with a fixed seed.
"""

from __future__ import annotations

import json
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

    # Enforce theta_fail >= theta_deg element-wise
    theta_fail = np.maximum(theta_fail, theta_deg)

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
