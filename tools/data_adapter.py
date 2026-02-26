#!/usr/bin/env python3
"""
data_adapter.py — Consultancy Wrapper: Ingestor
================================================
Convert a CSV edgelist into a cascade_engine-compatible JSON configuration file.

The CSV must contain at minimum two columns: ``source`` and ``target``.
An optional ``weight`` column is accepted and recorded in output metadata,
but the current engine is unweighted — weights are stored for documentation
purposes only and do not affect simulation logic.

Node IDs in the CSV may be any non-negative integers or string labels.
They are re-mapped to a compact 0-indexed integer space automatically.

Usage examples
--------------
# Minimal — all thresholds at their defaults
python3 data_adapter.py edges.csv --output config_custom.json

# Full control
python3 data_adapter.py edges.csv \\
    --output config_custom.json \\
    --mode stochastic \\
    --threshold-type uniform \\
    --deg-low 0.1 --deg-high 0.35 \\
    --fail-low 0.5 --fail-high 0.85 \\
    --seed 42 \\
    --stochastic-k 10.0 \\
    --monte-carlo-trials 50 \\
    --enable-sensitivity

# Normal threshold distribution
python3 data_adapter.py edges.csv \\
    --output config_normal.json \\
    --threshold-type normal \\
    --deg-mean 0.3 --deg-std 0.08 \\
    --fail-mean 0.65 --fail-std 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"source", "target"}
OPTIONAL_WEIGHT_COLUMN = "weight"
DEFAULT_SENSITIVITY_CONFIG = {
    "perturbation_values": [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
    "max_seed_nodes": 20,
    "stochastic_trials": 20,
}


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


def load_edgelist(csv_path: Path) -> pd.DataFrame:
    """Load and validate an edgelist CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with columns at least ``source`` and ``target``.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing, or the file is empty / malformed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV '{csv_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV '{csv_path}' contains no rows.")

    # Normalise column names: strip whitespace and lower-case
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}."
        )

    # Drop fully-NA rows
    before = len(df)
    df = df.dropna(subset=["source", "target"])
    dropped = before - len(df)
    if dropped:
        print(f"  [data_adapter] Warning: dropped {dropped} row(s) with missing source/target.", file=sys.stderr)

    if df.empty:
        raise ValueError("No valid edges remain after dropping NA rows.")

    return df


# ---------------------------------------------------------------------------
# Node re-mapping
# ---------------------------------------------------------------------------


def build_node_mapping(df: pd.DataFrame) -> dict:
    """Build a compact 0-indexed mapping from raw node labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``source`` and ``target`` columns.

    Returns
    -------
    dict
        ``{"mapping": {raw_label: int_id, ...}, "n": int, "remapped": bool}``
        ``remapped`` is True when the raw labels were not already a compact
        zero-based integer sequence (0, 1, …, n-1).
    """
    all_nodes_raw = pd.concat([df["source"], df["target"]]).unique()
    # Attempt integer conversion for numeric node IDs
    try:
        all_nodes_sorted = sorted(int(x) for x in all_nodes_raw)
        raw_is_int = True
    except (ValueError, TypeError):
        all_nodes_sorted = sorted(str(x) for x in all_nodes_raw)
        raw_is_int = False

    n = len(all_nodes_sorted)

    if raw_is_int and all_nodes_sorted == list(range(n)):
        # Already compact 0-indexed integers — no remapping needed
        mapping = {v: v for v in all_nodes_sorted}
        remapped = False
    else:
        mapping = {raw: idx for idx, raw in enumerate(all_nodes_sorted)}
        remapped = True

    return {"mapping": mapping, "n": n, "remapped": remapped}


# ---------------------------------------------------------------------------
# Edge processing
# ---------------------------------------------------------------------------


def build_edge_list(
    df: pd.DataFrame,
    node_mapping: dict,
) -> tuple[list[list[int]], int, int]:
    """Convert raw CSV rows to a compact integer edge list.

    Self-loops and duplicate directed edges are detected and reported.

    Parameters
    ----------
    df : pd.DataFrame
        Raw edgelist DataFrame.
    node_mapping : dict
        Output of :func:`build_node_mapping`.

    Returns
    -------
    edges : list of [int, int]
        Directed edge list as [source, target] pairs (0-indexed).
    n_self_loops : int
        Number of self-loops stripped.
    n_duplicates : int
        Number of duplicate edges collapsed.
    """
    mapping = node_mapping["mapping"]

    # Normalise raw IDs to the type used in the mapping
    def _resolve(raw):
        try:
            key = int(raw)
        except (ValueError, TypeError):
            key = str(raw)
        return mapping[key]

    raw_edges = [(_resolve(r["source"]), _resolve(r["target"])) for _, r in df.iterrows()]

    # Self-loop detection
    self_loops = [(u, v) for u, v in raw_edges if u == v]
    clean = [(u, v) for u, v in raw_edges if u != v]

    # Duplicate detection
    seen: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int]] = []
    deduped: list[list[int]] = []
    for u, v in clean:
        if (u, v) in seen:
            duplicates.append((u, v))
        else:
            seen.add((u, v))
            deduped.append([u, v])

    return deduped, len(self_loops), len(duplicates)


# ---------------------------------------------------------------------------
# Threshold config builder
# ---------------------------------------------------------------------------


def build_threshold_config(args: argparse.Namespace) -> dict:
    """Build the ``thresholds`` sub-config from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
        Threshold sub-config ready for JSON serialisation.

    Raises
    ------
    ValueError
        On invalid threshold parameter combinations.
    """
    t = args.threshold_type

    if t == "uniform":
        if args.deg_low >= args.deg_high:
            raise ValueError(
                f"--deg-low ({args.deg_low}) must be strictly less than "
                f"--deg-high ({args.deg_high})."
            )
        if args.fail_low >= args.fail_high:
            raise ValueError(
                f"--fail-low ({args.fail_low}) must be strictly less than "
                f"--fail-high ({args.fail_high})."
            )
        if args.fail_low < args.deg_high:
            print(
                f"  [data_adapter] Warning: fail_low ({args.fail_low}) < deg_high ({args.deg_high}). "
                "Threshold ranges overlap — some nodes may require theta_fail correction at runtime.",
                file=sys.stderr,
            )
        return {
            "type": "uniform",
            "deg_low": args.deg_low,
            "deg_high": args.deg_high,
            "fail_low": args.fail_low,
            "fail_high": args.fail_high,
        }

    if t == "normal":
        for attr, name in [
            ("deg_mean", "--deg-mean"),
            ("deg_std", "--deg-std"),
            ("fail_mean", "--fail-mean"),
            ("fail_std", "--fail-std"),
        ]:
            if getattr(args, attr) is None:
                raise ValueError(
                    f"--threshold-type normal requires {name}. "
                    "Provide --deg-mean, --deg-std, --fail-mean, --fail-std."
                )
        if args.deg_std <= 0 or args.fail_std <= 0:
            raise ValueError("Standard deviations (--deg-std, --fail-std) must be > 0.")
        return {
            "type": "normal",
            "deg_mean": args.deg_mean,
            "deg_std": args.deg_std,
            "fail_mean": args.fail_mean,
            "fail_std": args.fail_std,
        }

    raise ValueError(f"Unknown --threshold-type: {t!r}. Must be 'uniform' or 'normal'.")


# ---------------------------------------------------------------------------
# Config assembler
# ---------------------------------------------------------------------------


def assemble_config(
    edge_list: list[list[int]],
    n: int,
    threshold_cfg: dict,
    args: argparse.Namespace,
    adapter_meta: dict,
) -> dict:
    """Assemble the full engine JSON config.

    Parameters
    ----------
    edge_list : list of [int, int]
        Directed edge pairs (0-indexed).
    n : int
        Total number of nodes.
    threshold_cfg : dict
        Threshold sub-config.
    args : argparse.Namespace
        CLI arguments.
    adapter_meta : dict
        Provenance metadata (source file, remapping stats, etc.).

    Returns
    -------
    dict
        Complete engine config ready for ``json.dumps``.
    """
    cfg: dict = {
        "graph": {
            "type": "custom",
            "n": n,
            "edges": edge_list,
        },
        "thresholds": threshold_cfg,
        "seed": args.seed,
        "propagation_mode": args.mode,
        # Provenance block (informational; engine ignores unknown top-level keys)
        "_adapter_meta": adapter_meta,
    }

    if args.mode == "stochastic":
        cfg["stochastic_k"] = args.stochastic_k
        cfg["monte_carlo_trials"] = args.monte_carlo_trials

    if args.enable_sensitivity:
        cfg["sensitivity_analysis"] = True
        cfg["sensitivity_config"] = DEFAULT_SENSITIVITY_CONFIG.copy()
    else:
        cfg["sensitivity_analysis"] = False

    return cfg


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------


def print_summary(
    csv_path: Path,
    output_path: Path,
    df: pd.DataFrame,
    node_info: dict,
    edge_list: list,
    n_self_loops: int,
    n_duplicates: int,
) -> None:
    """Print a concise ingestion summary to stdout."""
    n = node_info["n"]
    sep = "─" * 56
    print(sep)
    print("  data_adapter — Ingestion Summary")
    print(sep)
    print(f"  Source CSV       : {csv_path}")
    print(f"  Output config    : {output_path}")
    print(f"  Raw CSV rows     : {len(df)}")
    print(f"  Nodes detected   : {n}")
    print(f"  Directed edges   : {len(edge_list)}")
    print(f"  Self-loops dropped   : {n_self_loops}")
    print(f"  Duplicate edges collapsed : {n_duplicates}")
    if node_info["remapped"]:
        print(f"  Node ID remapping : YES — labels mapped to 0–{n - 1}")
    else:
        print(f"  Node ID remapping : NO  — already compact 0–{n - 1}")
    has_weight = OPTIONAL_WEIGHT_COLUMN in df.columns
    print(f"  Weight column    : {'PRESENT (stored in metadata)' if has_weight else 'ABSENT'}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a CSV edgelist to a cascade_engine JSON configuration file.\n\n"
            "Required CSV columns: source, target\n"
            "Optional CSV column:  weight (recorded in metadata, not used by engine)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Positional ---
    p.add_argument("input", type=Path, help="Path to the input CSV edgelist file.")

    # --- Output ---
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("config_custom.json"),
        metavar="PATH",
        help="Path for the output JSON config file (default: config_custom.json).",
    )

    # --- Simulation mode ---
    p.add_argument(
        "--mode",
        choices=["deterministic", "stochastic"],
        default="deterministic",
        help="Propagation mode (default: deterministic).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for graph and threshold generation (default: 42).",
    )

    # --- Threshold type ---
    p.add_argument(
        "--threshold-type",
        dest="threshold_type",
        choices=["uniform", "normal"],
        default="uniform",
        help="Threshold distribution type (default: uniform).",
    )

    # Uniform params
    grp_u = p.add_argument_group("Uniform threshold parameters")
    grp_u.add_argument("--deg-low",  dest="deg_low",  type=float, default=0.1)
    grp_u.add_argument("--deg-high", dest="deg_high", type=float, default=0.4)
    grp_u.add_argument("--fail-low", dest="fail_low", type=float, default=0.5)
    grp_u.add_argument("--fail-high",dest="fail_high",type=float, default=0.9)

    # Normal params
    grp_n = p.add_argument_group("Normal threshold parameters")
    grp_n.add_argument("--deg-mean",  dest="deg_mean",  type=float, default=None)
    grp_n.add_argument("--deg-std",   dest="deg_std",   type=float, default=None)
    grp_n.add_argument("--fail-mean", dest="fail_mean", type=float, default=None)
    grp_n.add_argument("--fail-std",  dest="fail_std",  type=float, default=None)

    # --- Stochastic params ---
    grp_s = p.add_argument_group("Stochastic / Monte Carlo parameters (--mode stochastic)")
    grp_s.add_argument(
        "--stochastic-k",
        dest="stochastic_k",
        type=float,
        default=10.0,
        help="Logistic steepness k > 0 (default: 10.0).",
    )
    grp_s.add_argument(
        "--monte-carlo-trials",
        dest="monte_carlo_trials",
        type=int,
        default=50,
        help="Monte Carlo trials per seed node (default: 50).",
    )

    # --- Sensitivity ---
    p.add_argument(
        "--enable-sensitivity",
        dest="enable_sensitivity",
        action="store_true",
        help="Enable sensitivity analysis block in the output config.",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    print(f"  [data_adapter] Reading CSV: {args.input}")
    try:
        df = load_edgelist(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Build node mapping ─────────────────────────────────────────────────
    try:
        node_info = build_node_mapping(df)
    except Exception as exc:
        print(f"ERROR building node mapping: {exc}", file=sys.stderr)
        sys.exit(1)

    n = node_info["n"]
    print(f"  [data_adapter] Detected {n} unique nodes.")

    # ── 3. Build edge list ────────────────────────────────────────────────────
    try:
        edge_list, n_self_loops, n_duplicates = build_edge_list(df, node_info)
    except Exception as exc:
        print(f"ERROR building edge list: {exc}", file=sys.stderr)
        sys.exit(1)

    if n_self_loops:
        print(f"  [data_adapter] Warning: {n_self_loops} self-loop(s) stripped.", file=sys.stderr)
    if n_duplicates:
        print(f"  [data_adapter] Warning: {n_duplicates} duplicate edge(s) collapsed.", file=sys.stderr)

    # ── 4. Build threshold config ─────────────────────────────────────────────
    try:
        threshold_cfg = build_threshold_config(args)
    except ValueError as exc:
        print(f"ERROR in threshold parameters: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── 5. Validate stochastic params ─────────────────────────────────────────
    if args.mode == "stochastic":
        if args.stochastic_k <= 0:
            print("ERROR: --stochastic-k must be > 0.", file=sys.stderr)
            sys.exit(1)
        if args.monte_carlo_trials < 2:
            print("ERROR: --monte-carlo-trials must be >= 2.", file=sys.stderr)
            sys.exit(1)

    # ── 6. Provenance metadata ────────────────────────────────────────────────
    has_weight = OPTIONAL_WEIGHT_COLUMN in df.columns
    adapter_meta = {
        "source_csv": str(args.input.resolve()),
        "n_raw_rows": len(df),
        "n_nodes": n,
        "n_edges": len(edge_list),
        "n_self_loops_stripped": n_self_loops,
        "n_duplicates_collapsed": n_duplicates,
        "node_remapped": node_info["remapped"],
        "has_weight_column": has_weight,
        "weight_note": (
            "Weight column present but not used by the cascade engine. "
            "Stored here for documentation only."
            if has_weight else None
        ),
        "node_mapping_sample": (
            dict(list(node_info["mapping"].items())[:10])
            if node_info["remapped"] else "identity"
        ),
    }

    # ── 7. Assemble and write config ──────────────────────────────────────────
    cfg = assemble_config(edge_list, n, threshold_cfg, args, adapter_meta)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg, indent=2))

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print_summary(args.input, output_path, df, node_info, edge_list, n_self_loops, n_duplicates)
    print(f"\n  Config written to: {output_path.resolve()}")
    print(f"  Run with: python3 -m cascade_engine.runner {output_path} --output-dir results/\n")


if __name__ == "__main__":
    main()
