"""
Runner script for the cascade propagation engine (Tier 1 + Tier 2).

Loads a JSON config, builds the graph and thresholds, and dispatches to the
appropriate experiment pipeline based on ``propagation_mode``.

Tier 1 (deterministic)
    Fragility index experiment; outputs fragility_results.csv and summary.json.

Tier 2 (stochastic)
    Monte Carlo experiment; outputs monte_carlo_distribution.csv,
    fragility_results.csv, and optionally sensitivity_results.csv.

Usage
-----
    python runner.py config.json [--output-dir results/]

All outputs are written to the specified directory.  A config snapshot
with SHA-256 hash is always saved alongside results for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from .config import load_config, build_rng, generate_thresholds
from .graph import graph_from_config
from .metrics import (
    fragility_index,
    fragility_summary,
    cascade_size,
    rmse,
    mape,
    spearman_correlation,
)
from .propagation import run_until_stable
from .monte_carlo import run_monte_carlo_all_seeds, MonteCarloResult
from .sensitivity import (
    threshold_sensitivity,
    sensitivity_to_records,
    sensitivity_aggregate_by_perturbation,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cascade propagation engine — Tier 1 + Tier 2 runner."
    )
    parser.add_argument("config", help="Path to JSON configuration file.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files (default: results/).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _config_hash(cfg: dict) -> str:
    """Compute a SHA-256 hash of the JSON-serialised config for reproducibility."""
    serialised = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _nan_to_none(v):
    """Replace float NaN with None for valid JSON serialisation."""
    try:
        import math
        return None if (isinstance(v, float) and math.isnan(v)) else v
    except TypeError:
        return v


def _save_config_snapshot(output_dir: Path, cfg: dict) -> None:
    snapshot = {
        "config": cfg,
        "sha256": _config_hash(cfg),
    }
    (output_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2))


# ---------------------------------------------------------------------------
# Shared graph / threshold construction
# ---------------------------------------------------------------------------


def _build_experiment(
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build adjacency matrix and threshold arrays from config.

    Returns
    -------
    A, theta_deg, theta_fail, in_degree — all numpy arrays.
    """
    graph_cfg = dict(cfg["graph"])
    if "seed" not in graph_cfg:
        graph_cfg["seed"] = int(cfg["seed"])

    A = graph_from_config(graph_cfg)
    n = A.shape[0]
    in_degree = A.sum(axis=0).astype(np.int64)

    rng = build_rng(cfg)
    theta_deg, theta_fail = generate_thresholds(n, cfg["thresholds"], rng)

    return A, theta_deg, theta_fail, in_degree


# ---------------------------------------------------------------------------
# Tier 1: deterministic pipeline
# ---------------------------------------------------------------------------


def _run_deterministic(
    cfg: dict,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    in_degree: np.ndarray,
    output_dir: Path,
) -> None:
    """Execute the Tier 1 deterministic experiment pipeline."""
    n = A.shape[0]
    print(f"[Tier 1] Deterministic | n={n} nodes")

    # Fragility index
    t0 = time.perf_counter()
    fi = fragility_index(A, theta_deg, theta_fail)
    elapsed = time.perf_counter() - t0
    fi_sum = fragility_summary(fi)

    # F-007 fix: report top-k worst-case cascades by FI and by in-degree
    # rather than only argmax(FI), for richer thesis result tables.
    top_k = min(5, n)
    top_fi_nodes    = np.argsort(fi)[::-1][:top_k].tolist()
    top_indeg_nodes = np.argsort(in_degree)[::-1][:top_k].tolist()
    topk_seed_nodes = list(dict.fromkeys(top_fi_nodes + top_indeg_nodes))

    # Build per-seed cascade stats for the top-k set
    topk_results = []
    for sn in topk_seed_nodes:
        S_seed = np.zeros(n, dtype=np.int32)
        S_seed[sn] = 2
        sn_final, _, _, _ = run_until_stable(S_seed, A, theta_deg, theta_fail)
        sn_cs = cascade_size(sn_final)
        topk_results.append({
            "seed_node": sn,
            "fragility_index": int(fi[sn]),
            "in_degree": int(in_degree[sn]),
            "n_affected": sn_cs["n_affected"],
            "frac_affected": round(sn_cs["frac_affected"], 4),
            "n_failed": sn_cs["n_failed"],
            "n_degraded": sn_cs["n_degraded"],
        })

    # Keep the highest-FI node as the primary "worst case" for summary fields.
    # topk_results already contains the cascade stats for this node — reuse them
    # directly to avoid running an identical cascade a second time.
    highest_fi_node = top_fi_nodes[0]
    full_cs = next(r for r in topk_results if r["seed_node"] == highest_fi_node)

    # topk_cascade_results.csv  (F-007 fix)
    _write_csv(
        output_dir / "topk_cascade_results.csv",
        ["seed_node", "fragility_index", "in_degree", "n_affected",
         "frac_affected", "n_failed", "n_degraded"],
        topk_results,
    )

    # fragility_results.csv
    _write_csv(
        output_dir / "fragility_results.csv",
        ["node_id", "fragility_index", "theta_deg", "theta_fail", "in_degree"],
        [
            {
                "node_id": i,
                "fragility_index": int(fi[i]),
                "theta_deg": float(theta_deg[i]),
                "theta_fail": float(theta_fail[i]),
                "in_degree": int(in_degree[i]),
            }
            for i in range(n)
        ],
    )

    # results_summary.csv
    summary_kv = {
        "mode": "deterministic",
        "n_nodes": n,
        "elapsed_fragility_s": round(elapsed, 4),
        "fi_mean": round(fi_sum["mean"], 4),
        "fi_std": round(fi_sum["std"], 4),
        "fi_min": fi_sum["min"],
        "fi_median": fi_sum["median"],
        "fi_p90": fi_sum["p90"],
        "fi_max": fi_sum["max"],
        "full_cascade_n_affected": full_cs["n_affected"],
        "full_cascade_frac_affected": round(full_cs["frac_affected"], 4),
        "full_cascade_seed_node": highest_fi_node,
    }
    _write_csv(
        output_dir / "results_summary.csv",
        ["metric", "value"],
        [{"metric": metric_key, "value": v} for metric_key, v in summary_kv.items()],
    )

    # summary.json
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "deterministic",
                "n_nodes": n,
                "elapsed_seconds": elapsed,
                "fragility_index": fi_sum,
                "worst_case_cascade": {**full_cs, "seed_node": highest_fi_node},
            },
            indent=2,
        )
    )

    _print_det_summary(n, elapsed, fi_sum, full_cs, highest_fi_node, topk_results)


def _print_det_summary(n, elapsed, fi_sum, full_cs, highest_fi_node, topk_results=None):
    sep = "-" * 58
    print(sep)
    print("  Cascade Propagation Engine — Tier 1 (Deterministic)")
    print(sep)
    print(f"  Nodes                 : {n}")
    print(f"  Elapsed (fragility)   : {elapsed:.2f}s")
    print()
    print("  Fragility Index Statistics (affected nodes per seed)")
    print(f"    Mean   : {fi_sum['mean']:.2f}")
    print(f"    Std    : {fi_sum['std']:.2f}")
    print(f"    Min    : {fi_sum['min']:.0f}")
    print(f"    Median : {fi_sum['median']:.0f}")
    print(f"    P90    : {fi_sum['p90']:.0f}")
    print(f"    Max    : {fi_sum['max']:.0f}")
    print()
    print(f"  Worst-Case Cascade (highest-FI node = {highest_fi_node} seeded)")
    print(f"    Affected : {full_cs['n_affected']} / {n}")
    print(f"    Failed   : {full_cs['n_failed']} / {n}")
    print(f"    Degraded : {full_cs['n_degraded']} / {n}")
    if topk_results:
        print()
        print("  Top-k Worst-Case Cascades (by FI + in-degree)")
        hdr = f"  {'Node':>6}  {'FI':>6}  {'InDeg':>6}  {'Affected':>9}  {'Frac':>6}"
        print(hdr)
        for r in topk_results:
            print(f"  {r['seed_node']:>6}  {r['fragility_index']:>6}  {r['in_degree']:>6}  {r['n_affected']:>9}  {r['frac_affected']:>6.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Tier 2: stochastic + Monte Carlo pipeline
# ---------------------------------------------------------------------------


def _run_stochastic(
    cfg: dict,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    in_degree: np.ndarray,
    output_dir: Path,
) -> None:
    """Execute the Tier 2 stochastic / Monte Carlo experiment pipeline."""
    n = A.shape[0]
    trials: int = int(cfg.get("monte_carlo_trials", 50))
    k: float = float(cfg.get("stochastic_k", 10.0))
    master_seed: int = int(cfg["seed"])

    print(f"[Tier 2] Stochastic | n={n} | trials/node={trials} | k={k}")

    # Monte Carlo across all seed nodes
    t0 = time.perf_counter()
    mc_results: list[MonteCarloResult] = run_monte_carlo_all_seeds(
        A, theta_deg, theta_fail,
        trials=trials,
        seed=master_seed,
        k=k,
    )
    elapsed_mc = time.perf_counter() - t0

    mc_mean_cs = np.array([r.mean_cascade_size for r in mc_results])
    mc_ci_low = np.array([r.ci_low for r in mc_results])
    mc_ci_high = np.array([r.ci_high for r in mc_results])
    mc_var = np.array([r.variance_cascade_size for r in mc_results])

    # Deterministic fragility for comparison
    t1 = time.perf_counter()
    fi_det = fragility_index(A, theta_deg, theta_fail)
    elapsed_det = time.perf_counter() - t1
    fi_det_frac = fi_det / n

    # Comparison metrics
    rmse_val = rmse(fi_det_frac, mc_mean_cs)
    mape_val = mape(fi_det_frac, mc_mean_cs)
    spearman = spearman_correlation(fi_det_frac, mc_mean_cs)

    # fragility_results.csv (combined det + stochastic)
    _write_csv(
        output_dir / "fragility_results.csv",
        [
            "node_id", "det_fragility_index", "det_fragility_frac",
            "stochastic_mean_cascade", "stochastic_variance",
            "ci_95_low", "ci_95_high",
            "theta_deg", "theta_fail", "in_degree",
        ],
        [
            {
                "node_id": i,
                "det_fragility_index": int(fi_det[i]),
                "det_fragility_frac": float(fi_det_frac[i]),
                "stochastic_mean_cascade": float(mc_mean_cs[i]),
                "stochastic_variance": float(mc_var[i]),
                "ci_95_low": float(mc_ci_low[i]),
                "ci_95_high": float(mc_ci_high[i]),
                "theta_deg": float(theta_deg[i]),
                "theta_fail": float(theta_fail[i]),
                "in_degree": int(in_degree[i]),
            }
            for i in range(n)
        ],
    )

    # monte_carlo_distribution.csv (raw trial-level data)
    mc_rows = []
    for i, r in enumerate(mc_results):
        for trial_idx, (cs, ts) in enumerate(
            zip(r.cascade_sizes.tolist(), r.times_to_stability.tolist())
        ):
            mc_rows.append({
                "node_id": i,
                "trial": trial_idx,
                "cascade_size_frac": round(float(cs), 6),
                "time_to_stability": int(ts),
            })
    _write_csv(
        output_dir / "monte_carlo_distribution.csv",
        ["node_id", "trial", "cascade_size_frac", "time_to_stability"],
        mc_rows,
    )

    # results_summary.csv
    summary_kv = {
        "mode": "stochastic",
        "n_nodes": n,
        "monte_carlo_trials": trials,
        "stochastic_k": k,
        "elapsed_monte_carlo_s": round(elapsed_mc, 4),
        "elapsed_det_fragility_s": round(elapsed_det, 4),
        "rmse_det_vs_stochastic": round(rmse_val, 6),
        "mape_det_vs_stochastic_pct": round(mape_val, 4),
        "spearman_rho": round(spearman["rho"], 6),
        "spearman_p_value": round(spearman["p_value"], 6),
        "stochastic_mean_cascade_mean": round(float(np.mean(mc_mean_cs)), 4),
        "stochastic_mean_cascade_std": round(float(np.std(mc_mean_cs)), 4),
    }
    _write_csv(
        output_dir / "results_summary.csv",
        ["metric", "value"],
        [{"metric": metric_key, "value": v} for metric_key, v in summary_kv.items()],
    )
    # summary.json — replace any NaN values with None so output is valid JSON.
    # NaN arises when spearman_rho is undefined (e.g. all cascade sizes identical
    # at very low k), causing json.dumps to emit the non-standard NaN literal.
    summary_kv_serialisable = {metric_key: _nan_to_none(v) for metric_key, v in summary_kv.items()}
    (output_dir / "summary.json").write_text(json.dumps(summary_kv_serialisable, indent=2))

    _print_stochastic_summary(n, trials, k, elapsed_mc, mc_mean_cs, rmse_val, mape_val, spearman)

    # Optional sensitivity analysis
    if cfg.get("sensitivity_analysis", False):
        _run_sensitivity(cfg, A, theta_deg, theta_fail, output_dir, mode="stochastic", k=k)


def _print_stochastic_summary(n, trials, k, elapsed, mc_mean_cs, rmse_val, mape_val, spearman):
    sep = "-" * 58
    print(sep)
    print("  Cascade Propagation Engine — Tier 2 (Stochastic / MC)")
    print(sep)
    print(f"  Nodes              : {n}")
    print(f"  Trials / node      : {trials}")
    print(f"  k (steepness)      : {k}")
    print(f"  Elapsed (MC)       : {elapsed:.2f}s")
    print()
    print("  Stochastic Fragility (mean cascade size per seed node)")
    print(f"    Mean  : {np.mean(mc_mean_cs):.4f}")
    print(f"    Std   : {np.std(mc_mean_cs):.4f}")
    print(f"    Min   : {np.min(mc_mean_cs):.4f}")
    print(f"    Max   : {np.max(mc_mean_cs):.4f}")
    print()
    print("  Det vs Stochastic Comparison")
    print(f"    RMSE          : {rmse_val:.6f}")
    print(f"    MAPE          : {mape_val:.4f}%")
    print(f"    Spearman ρ    : {spearman['rho']:.4f}  (p={spearman['p_value']:.4f})")
    print(sep)


# ---------------------------------------------------------------------------
# Sensitivity sub-pipeline (deterministic or stochastic)
# ---------------------------------------------------------------------------


def _run_sensitivity(
    cfg: dict,
    A: np.ndarray,
    theta_deg: np.ndarray,
    theta_fail: np.ndarray,
    output_dir: Path,
    mode: str = "deterministic",
    k: float = 10.0,
) -> None:
    """Run threshold sensitivity analysis and write results."""
    n = A.shape[0]
    sens_cfg: dict = cfg.get("sensitivity_config", {})

    perturbations: list[float] = sens_cfg.get(
        "perturbation_values", [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    )
    # Default: up to 20 highest in-degree nodes as seeds
    max_seed_nodes: int = int(sens_cfg.get("max_seed_nodes", min(n, 20)))
    in_deg = A.sum(axis=0).astype(np.int64)
    seed_nodes = list(np.argsort(in_deg)[::-1][:max_seed_nodes].tolist())

    stochastic_trials: int = int(sens_cfg.get("stochastic_trials", 20))
    # Use a well-separated seed for sensitivity so its SeedSequence tree is
    # statistically independent from the Monte Carlo SeedSequence tree (F-006).
    master_seed: int = int(cfg["seed"]) + 10_000_000

    print(
        f"[Sensitivity] mode={mode} | {len(perturbations)} perturbations | "
        f"{len(seed_nodes)} seed nodes"
    )
    t0 = time.perf_counter()
    points = threshold_sensitivity(
        A, theta_deg, theta_fail,
        perturbation_values=perturbations,
        seed_nodes=seed_nodes,
        mode=mode,
        stochastic_trials=stochastic_trials,
        stochastic_k=k,
        seed=master_seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"[Sensitivity] Done in {elapsed:.2f}s — {len(points)} records")

    # Per-point CSV
    _write_csv(
        output_dir / "sensitivity_results.csv",
        [
            "perturbation", "seed_node", "cascade_size_fraction",
            "time_to_stability", "n_trials", "std_cascade_size",
            "clamp_mode", "n_nodes_clamped",
        ],
        sensitivity_to_records(points),
    )

    # Aggregated CSV
    _write_csv(
        output_dir / "sensitivity_aggregate.csv",
        ["perturbation", "mean_cascade_size", "std_cascade_size", "n_seed_nodes",
         "clamp_mode", "n_nodes_clamped"],
        sensitivity_aggregate_by_perturbation(points),
    )
    print("[Sensitivity] Wrote sensitivity_results.csv and sensitivity_aggregate.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    cfg = load_config(args.config)
    _save_config_snapshot(output_dir, cfg)

    # Log experiment metadata
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "config_file": str(Path(args.config).resolve()),
                "output_dir": str(output_dir.resolve()),
                "config_sha256": _config_hash(cfg),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )

    A, theta_deg, theta_fail, in_degree = _build_experiment(cfg)
    mode: str = str(cfg.get("propagation_mode", "deterministic"))

    if mode == "deterministic":
        _run_deterministic(cfg, A, theta_deg, theta_fail, in_degree, output_dir)
        if cfg.get("sensitivity_analysis", False):
            _run_sensitivity(cfg, A, theta_deg, theta_fail, output_dir, mode="deterministic")
    elif mode == "stochastic":
        _run_stochastic(cfg, A, theta_deg, theta_fail, in_degree, output_dir)
    else:
        print(f"ERROR: Unknown propagation_mode {mode!r}. Must be 'deterministic' or 'stochastic'.")
        sys.exit(1)

    print(f"  Results saved to : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
