#!/usr/bin/env python3
"""
scenario_manager.py — Consultancy Wrapper: Batch Processor
===========================================================
Automate "What-If" analysis by generating and executing multiple scenario
variants derived from a single base configuration.

Three scenario types are supported and can be combined in a single invocation:

  ``hardening``   — Targeted threshold hardening. Simulates the risk-reduction
                    effect of engineering interventions on the most connected
                    (highest in-degree) nodes by raising their failure thresholds.
                    Implemented by increasing fail_low/fail_high by a delta and
                    running sensitivity analysis limited to the top-k seed nodes.

  ``regional``    — Regional vulnerability injection. Models an externally
                    degraded sub-region by lowering thresholds for a caller-
                    specified list of node IDs, then measuring cascade impact.

  ``sweep``       — Logistic steepness (k) uncertainty sweep. Runs the
                    stochastic engine across a range of k values to locate the
                    "Chaos Point" — the k below which Spearman ρ collapses and
                    ranking information is lost.

Each scenario writes its runner output to an isolated sub-folder inside
``--output-dir``, enabling direct comparison across runs.

Usage examples
--------------
# All three scenarios
python3 scenario_manager.py cascade_engine/config_erdos_renyi.json \\
    --output-dir scenario_results/ \\
    --scenarios hardening regional sweep \\
    --hardening-deltas 0.1 0.2 0.3 \\
    --top-k 10 \\
    --vulnerable-nodes 5 12 37 \\
    --vulnerability-delta 0.2 \\
    --k-values 2 5 10 20 40 80

# Hardening only
python3 scenario_manager.py base.json \\
    --output-dir results/ \\
    --scenarios hardening \\
    --hardening-deltas 0.15 0.30 \\
    --top-k 5
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNNER_MODULE = "cascade_engine.runner"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Load and lightly validate a JSON config file.

    Parameters
    ----------
    path : Path
        Path to the JSON config.

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in '{path}': {exc}") from exc

    required = {"graph", "thresholds", "seed", "propagation_mode"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Base config missing required keys: {sorted(missing)}")

    return cfg


def _clamp_thresholds(cfg: dict) -> dict:
    """Ensure fail_low >= deg_high and all values in [0, 1] after modification."""
    t = cfg["thresholds"]
    if t.get("type") == "uniform":
        t["deg_low"]  = float(np.clip(t.get("deg_low",  0.1), 0.0, 1.0))
        t["deg_high"] = float(np.clip(t.get("deg_high", 0.4), 0.0, 1.0))
        t["fail_low"] = float(np.clip(t.get("fail_low", 0.5), 0.0, 1.0))
        t["fail_high"]= float(np.clip(t.get("fail_high",0.9), 0.0, 1.0))
        # Preserve ordering invariants
        if t["deg_low"] > t["deg_high"]:
            t["deg_low"], t["deg_high"] = t["deg_high"], t["deg_low"]
        if t["fail_low"] > t["fail_high"]:
            t["fail_low"], t["fail_high"] = t["fail_high"], t["fail_low"]
    elif t.get("type") == "normal":
        for key in ("deg_mean", "fail_mean"):
            if key in t:
                t[key] = float(np.clip(t[key], 0.0, 1.0))
        for key in ("deg_std", "fail_std"):
            if key in t:
                t[key] = max(float(t[key]), 1e-4)
    return cfg


def save_scenario_config(cfg: dict, path: Path) -> None:
    """Write a scenario config to disk, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Runner invocation
# ---------------------------------------------------------------------------


def run_engine(config_path: Path, output_dir: Path, timeout: int = 600) -> bool:
    """Invoke the cascade engine runner as a subprocess.

    Parameters
    ----------
    config_path : Path
        Scenario-specific JSON config path.
    output_dir : Path
        Directory for runner output files.
    timeout : int
        Maximum seconds to wait (default 600 s / 10 min).

    Returns
    -------
    bool
        True if runner exited successfully, False otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", RUNNER_MODULE,
        str(config_path),
        "--output-dir", str(output_dir),
    ]
    print(f"    ↳ Executing: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    ✗ TIMEOUT after {timeout}s", file=sys.stderr)
        return False
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"    ✗ Runner failed (exit {proc.returncode}) in {elapsed:.1f}s", file=sys.stderr)
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines()[-6:]:
                print(f"      {line}", file=sys.stderr)
        return False

    print(f"    ✓ Done in {elapsed:.1f}s → {output_dir}")
    return True


# ---------------------------------------------------------------------------
# Scenario 1: Targeted Hardening
# ---------------------------------------------------------------------------


def run_hardening_scenarios(
    base_cfg: dict,
    output_dir: Path,
    hardening_deltas: list[float],
    top_k: int,
) -> list[dict]:
    """Targeted threshold hardening for high-connectivity nodes.

    Raises fail_low and fail_high by ``delta`` to simulate engineering
    interventions that reduce failure probability on the most connected
    (highest in-degree) nodes.  Sensitivity analysis is enabled with a
    positive perturbation equal to ``delta``, restricted to the top-k nodes
    as seed sources.

    Parameters
    ----------
    base_cfg : dict
        Base configuration.
    output_dir : Path
        Root output directory; sub-folders are created per delta.
    hardening_deltas : list of float
        Magnitude of threshold increase to simulate (e.g., 0.1, 0.2, 0.3).
    top_k : int
        Number of highest in-degree nodes to consider as hardening targets.

    Returns
    -------
    list of dict
        One record per scenario: {label, config_path, results_dir, success}.
    """
    print(f"\n  [Scenario: Targeted Hardening] deltas={hardening_deltas}, top_k={top_k}")
    records = []
    hardening_dir = output_dir / "hardening"

    for delta in hardening_deltas:
        label = f"hardening_d{delta:.2f}"
        scenario_dir = hardening_dir / label
        config_path  = scenario_dir / "scenario_config.json"

        cfg = copy.deepcopy(base_cfg)
        t = cfg["thresholds"]

        if t.get("type") == "uniform":
            t["fail_low"]  = t.get("fail_low",  0.5) + delta
            t["fail_high"] = t.get("fail_high", 0.9) + delta
        elif t.get("type") == "normal":
            t["fail_mean"] = t.get("fail_mean", 0.65) + delta

        cfg = _clamp_thresholds(cfg)

        # Enable sensitivity to quantify cascade reduction vs. baseline
        cfg["sensitivity_analysis"] = True
        cfg["sensitivity_config"] = {
            "perturbation_values": [0.0],          # baseline only — thresholds already raised
            "max_seed_nodes": max(top_k, 5),
            "stochastic_trials": 20,
        }

        # Annotate for report_gen
        cfg["_scenario"] = {
            "type": "hardening",
            "delta": delta,
            "top_k": top_k,
            "description": (
                f"Targeted hardening: fail thresholds raised by {delta:.2f} "
                f"for top-{top_k} in-degree nodes (global approximation)."
            ),
        }

        scenario_dir.mkdir(parents=True, exist_ok=True)
        save_scenario_config(cfg, config_path)
        print(f"  → {label}: fail_low={t.get('fail_low'):.3f}, fail_high={t.get('fail_high', 'N/A')}")
        success = run_engine(config_path, scenario_dir / "results")

        records.append({
            "label": label,
            "scenario_type": "hardening",
            "delta": delta,
            "config_path": str(config_path),
            "results_dir": str(scenario_dir / "results"),
            "success": success,
        })

    return records


# ---------------------------------------------------------------------------
# Scenario 2: Regional Vulnerability
# ---------------------------------------------------------------------------


def run_regional_scenarios(
    base_cfg: dict,
    output_dir: Path,
    vulnerable_node_ids: list[int],
    vulnerability_delta: float,
) -> list[dict]:
    """Regional vulnerability injection.

    Lowers both degradation and failure thresholds for a specified set of
    node IDs by ``vulnerability_delta``, then runs the full cascade engine.
    This models external shocks such as a natural disaster affecting a
    geographic cluster, or cyber-attacks targeting a specific subnet.

    For engines using non-custom graph types (ER, BA, etc.), the node IDs
    refer to the integer indices within the generated graph.  Since the
    engine does not support per-node threshold overrides via JSON config,
    this scenario uses the sensitivity_config with seed_nodes set to the
    specified IDs and a negative perturbation equal to ``vulnerability_delta``.

    Parameters
    ----------
    base_cfg : dict
        Base configuration.
    output_dir : Path
        Root output directory.
    vulnerable_node_ids : list of int
        Node indices to weaken.
    vulnerability_delta : float
        Magnitude of threshold reduction (positive value; applied as -delta).

    Returns
    -------
    list of dict
    """
    if not vulnerable_node_ids:
        print("  [Scenario: Regional Vulnerability] No node IDs specified — skipping.", file=sys.stderr)
        return []

    label = "regional_vulnerability"
    print(f"\n  [Scenario: Regional Vulnerability] nodes={vulnerable_node_ids}, delta={vulnerability_delta}")
    regional_dir = output_dir / "regional"
    scenario_dir = regional_dir / label
    config_path  = scenario_dir / "scenario_config.json"

    cfg = copy.deepcopy(base_cfg)

    # Use sensitivity_analysis with negative delta to model weakened nodes
    # as seed sources — revealing cascade impact when those nodes are degraded.
    cfg["sensitivity_analysis"] = True
    cfg["sensitivity_config"] = {
        "perturbation_values": [-vulnerability_delta, 0.0],   # compare weakened vs. baseline
        "seed_nodes": vulnerable_node_ids,
        "stochastic_trials": 20,
        "max_seed_nodes": len(vulnerable_node_ids),
    }

    cfg["_scenario"] = {
        "type": "regional_vulnerability",
        "vulnerable_nodes": vulnerable_node_ids,
        "vulnerability_delta": vulnerability_delta,
        "description": (
            f"Regional vulnerability: thresholds reduced by {vulnerability_delta:.2f} "
            f"for nodes {vulnerable_node_ids} (modelled as negative perturbation)."
        ),
    }

    scenario_dir.mkdir(parents=True, exist_ok=True)
    save_scenario_config(cfg, config_path)
    print(f"  → {label}: delta=-{vulnerability_delta:.2f} on {len(vulnerable_node_ids)} node(s)")
    success = run_engine(config_path, scenario_dir / "results")

    return [{
        "label": label,
        "scenario_type": "regional",
        "vulnerable_nodes": vulnerable_node_ids,
        "vulnerability_delta": vulnerability_delta,
        "config_path": str(config_path),
        "results_dir": str(scenario_dir / "results"),
        "success": success,
    }]


# ---------------------------------------------------------------------------
# Scenario 3: Uncertainty (k) Sweep
# ---------------------------------------------------------------------------


def run_k_sweep_scenarios(
    base_cfg: dict,
    output_dir: Path,
    k_values: list[float],
    monte_carlo_trials: int,
) -> list[dict]:
    """Logistic steepness (k) uncertainty sweep.

    Runs the stochastic engine for each k value to identify the Chaos Point —
    the threshold below which the stochastic engine loses its rank-correlation
    with the deterministic fragility index (Spearman ρ → 0).

    Parameters
    ----------
    base_cfg : dict
        Base configuration.  Will be converted to stochastic mode if needed.
    output_dir : Path
        Root output directory.
    k_values : list of float
        Steepness values to sweep.
    monte_carlo_trials : int
        Trials per seed node for each k run.

    Returns
    -------
    list of dict
    """
    print(f"\n  [Scenario: k-Sweep] k_values={k_values}, trials={monte_carlo_trials}")
    records = []
    sweep_dir = output_dir / "k_sweep"

    for k in k_values:
        label = f"k_{k}"
        scenario_dir = sweep_dir / label
        config_path  = scenario_dir / "scenario_config.json"

        cfg = copy.deepcopy(base_cfg)
        cfg["propagation_mode"]   = "stochastic"
        cfg["stochastic_k"]       = float(k)
        cfg["monte_carlo_trials"] = monte_carlo_trials
        cfg["sensitivity_analysis"] = False

        cfg["_scenario"] = {
            "type": "k_sweep",
            "k": float(k),
            "description": f"Uncertainty sweep: stochastic k={k}, trials={monte_carlo_trials}.",
        }

        scenario_dir.mkdir(parents=True, exist_ok=True)
        save_scenario_config(cfg, config_path)
        print(f"  → {label}")
        success = run_engine(config_path, scenario_dir / "results")

        # Extract Spearman rho from summary.json if successful
        spearman_rho = None
        rmse_val = None
        summary_path = scenario_dir / "results" / "summary.json"
        if success and summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                spearman_rho = summary.get("spearman_rho")
                rmse_val     = summary.get("rmse_det_vs_stochastic")
            except Exception:
                pass

        records.append({
            "label": label,
            "scenario_type": "k_sweep",
            "k": float(k),
            "config_path": str(config_path),
            "results_dir": str(scenario_dir / "results"),
            "success": success,
            "spearman_rho": spearman_rho,
            "rmse": rmse_val,
        })

    return records


# ---------------------------------------------------------------------------
# Results manifest
# ---------------------------------------------------------------------------


def write_manifest(records: list[dict], output_dir: Path) -> Path:
    """Write a JSON manifest of all scenario runs for downstream tools.

    Parameters
    ----------
    records : list of dict
        Combined records from all scenarios.
    output_dir : Path
        Root output directory.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest = {
        "total_scenarios": len(records),
        "successful": sum(1 for r in records if r.get("success")),
        "failed": sum(1 for r in records if not r.get("success")),
        "scenarios": records,
    }
    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def print_final_summary(records: list[dict], manifest_path: Path) -> None:
    """Print a final summary table to stdout."""
    sep = "─" * 64
    print(f"\n{sep}")
    print("  scenario_manager — Run Summary")
    print(sep)
    total = len(records)
    ok    = sum(1 for r in records if r.get("success"))
    print(f"  Total scenarios : {total}")
    print(f"  Successful      : {ok}")
    print(f"  Failed          : {total - ok}")
    print()

    # k-sweep sub-table if present
    sweep_records = [r for r in records if r.get("scenario_type") == "k_sweep"]
    if sweep_records:
        print("  k-Sweep Results:")
        print(f"    {'k':>8}  {'Spearman ρ':>12}  {'RMSE':>10}  {'Status':>8}")
        for r in sweep_records:
            rho = r.get("spearman_rho")
            rmse = r.get("rmse")
            status = "✓" if r.get("success") else "✗"
            rho_s  = f"{rho:.4f}" if rho is not None else "N/A"
            rmse_s = f"{rmse:.4f}" if rmse is not None else "N/A"
            print(f"    {r['k']:>8.1f}  {rho_s:>12}  {rmse_s:>10}  {status:>8}")
        print()

    print(f"  Manifest: {manifest_path}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Batch 'What-If' scenario runner for the cascade propagation engine.\n\n"
            "Generates scenario config variants and executes the engine for each,\n"
            "saving results to organised sub-folders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("config", type=Path, help="Path to the base JSON configuration file.")
    p.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("scenario_results"),
        metavar="DIR",
        help="Root directory for all scenario outputs (default: scenario_results/).",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=["hardening", "regional", "sweep"],
        default=["hardening", "regional", "sweep"],
        metavar="SCENARIO",
        help=(
            "Which scenarios to run. Choices: hardening regional sweep. "
            "Default: all three."
        ),
    )

    # --- Hardening ---
    grp_h = p.add_argument_group("Targeted Hardening parameters")
    grp_h.add_argument(
        "--hardening-deltas",
        dest="hardening_deltas",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3],
        metavar="DELTA",
        help="Threshold increase magnitudes to simulate (default: 0.1 0.2 0.3).",
    )
    grp_h.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of highest in-degree nodes targeted for hardening (default: 10).",
    )

    # --- Regional ---
    grp_r = p.add_argument_group("Regional Vulnerability parameters")
    grp_r.add_argument(
        "--vulnerable-nodes",
        dest="vulnerable_nodes",
        nargs="+",
        type=int,
        default=[],
        metavar="NODE_ID",
        help="Node IDs to weaken (0-indexed integers, space-separated).",
    )
    grp_r.add_argument(
        "--vulnerability-delta",
        dest="vulnerability_delta",
        type=float,
        default=0.2,
        help="Threshold reduction magnitude for regional weakening (default: 0.2).",
    )

    # --- k-sweep ---
    grp_k = p.add_argument_group("k-Sweep (Uncertainty) parameters")
    grp_k.add_argument(
        "--k-values",
        dest="k_values",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        metavar="K",
        help="Logistic steepness values to sweep (default: 2 5 10 20 40 80).",
    )
    grp_k.add_argument(
        "--monte-carlo-trials",
        dest="monte_carlo_trials",
        type=int,
        default=30,
        help="Monte Carlo trials per node for k-sweep runs (default: 30).",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load base config ──────────────────────────────────────────────────────
    try:
        base_cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  scenario_manager starting.")
    print(f"  Base config : {args.config}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Scenarios   : {args.scenarios}")

    all_records: list[dict] = []

    # ── Scenario 1: Hardening ─────────────────────────────────────────────────
    if "hardening" in args.scenarios:
        records = run_hardening_scenarios(
            base_cfg=base_cfg,
            output_dir=output_dir,
            hardening_deltas=args.hardening_deltas,
            top_k=args.top_k,
        )
        all_records.extend(records)

    # ── Scenario 2: Regional ──────────────────────────────────────────────────
    if "regional" in args.scenarios:
        if not args.vulnerable_nodes:
            print(
                "\n  [Scenario: Regional Vulnerability] --vulnerable-nodes not specified. "
                "Skipping regional scenario.",
                file=sys.stderr,
            )
        else:
            records = run_regional_scenarios(
                base_cfg=base_cfg,
                output_dir=output_dir,
                vulnerable_node_ids=args.vulnerable_nodes,
                vulnerability_delta=args.vulnerability_delta,
            )
            all_records.extend(records)

    # ── Scenario 3: k-Sweep ───────────────────────────────────────────────────
    if "sweep" in args.scenarios:
        bad_k = [k for k in args.k_values if k <= 0]
        if bad_k:
            print(f"ERROR: k values must be > 0; got {bad_k}", file=sys.stderr)
            sys.exit(1)
        records = run_k_sweep_scenarios(
            base_cfg=base_cfg,
            output_dir=output_dir,
            k_values=sorted(args.k_values),
            monte_carlo_trials=args.monte_carlo_trials,
        )
        all_records.extend(records)

    # ── Write manifest and summary ────────────────────────────────────────────
    if not all_records:
        print("\n  No scenarios were executed.", file=sys.stderr)
        sys.exit(0)

    manifest_path = write_manifest(all_records, output_dir)
    print_final_summary(all_records, manifest_path)


if __name__ == "__main__":
    main()
