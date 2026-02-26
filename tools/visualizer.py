#!/usr/bin/env python3
"""
visualizer.py — Consultancy Wrapper: Dashboard
===============================================
Generate publication-quality, board-ready visuals from cascade_engine outputs.

Three plot types are produced:

  1. **Risk Heatmap** — Network graph with nodes coloured and sized by their
     Fragility Index.  Requires a config snapshot (``config_snapshot.json``)
     to reconstruct the graph topology.

  2. **Ghost Hub Plot** — Scatter plot of In-Degree (X) vs. Fragility Index (Y).
     Nodes in the upper-left quadrant have high FI despite low connectivity —
     "ghost hubs" that are dangerous but not obviously so.

  3. **Convergence Curve** — Spearman ρ vs. logistic steepness k, generated
     from a k-sweep scenario directory.  Reveals the Chaos Point below which
     the stochastic engine's rankings diverge from the deterministic baseline.

Usage examples
--------------
# All three plots from a deterministic run
python3 visualizer.py \\
    --results-dir results_er/ \\
    --output-dir visuals/

# Stochastic results + convergence curve from k-sweep
python3 visualizer.py \\
    --results-dir results_stoch/ \\
    --output-dir visuals/ \\
    --k-sweep-dir scenario_results/k_sweep/

# Specify DPI and file format
python3 visualizer.py \\
    --results-dir results_er/ \\
    --output-dir visuals/ \\
    --dpi 150 \\
    --fmt pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive; safe for headless/CI environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

# NetworkX is used only for the Risk Heatmap; import guarded for graceful degradation
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE_RISK   = "YlOrRd"        # Low → high fragility: yellow → red
PALETTE_DEGREE = "Blues"
NODE_ALPHA     = 0.85
EDGE_ALPHA     = 0.25
GHOST_COLOR    = "#C0392B"       # Highlight colour for ghost hub annotations
SAFE_COLOR     = "#2ECC71"
SPINE_COLOR    = "#CCCCCC"
BACKGROUND     = "#FAFAFA"
ACCENT         = "#2C3E50"
FONTFAMILY     = "DejaVu Sans"

sns.set_theme(style="whitegrid", font=FONTFAMILY)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_fragility_csv(results_dir: Path) -> pd.DataFrame:
    """Load fragility_results.csv, handling both deterministic and stochastic schemas.

    Parameters
    ----------
    results_dir : Path

    Returns
    -------
    pd.DataFrame with normalised columns:
        ``node_id``, ``fragility_index``, ``in_degree``,
        ``theta_deg``, ``theta_fail``
        and optionally ``stochastic_mean_cascade``.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    csv_path = results_dir / "fragility_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"fragility_results.csv not found in: {results_dir}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"fragility_results.csv in '{results_dir}' is empty.")

    # Normalise: stochastic schema uses det_fragility_index
    if "det_fragility_index" in df.columns and "fragility_index" not in df.columns:
        df = df.rename(columns={"det_fragility_index": "fragility_index"})

    required = {"node_id", "fragility_index", "in_degree"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"fragility_results.csv is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}."
        )

    return df


def load_summary_json(results_dir: Path) -> dict:
    """Load summary.json, returning an empty dict if not found."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text())
    except Exception as exc:
        warnings.warn(f"Could not parse summary.json: {exc}")
        return {}


def load_config_snapshot(results_dir: Path) -> dict | None:
    """Load config_snapshot.json, returning None if not found."""
    snap_path = results_dir / "config_snapshot.json"
    if not snap_path.exists():
        return None
    try:
        return json.loads(snap_path.read_text())
    except Exception as exc:
        warnings.warn(f"Could not parse config_snapshot.json: {exc}")
        return None


def rebuild_graph(config_snapshot: dict) -> "nx.DiGraph | None":
    """Reconstruct a NetworkX DiGraph from a config snapshot.

    Parameters
    ----------
    config_snapshot : dict
        Parsed config_snapshot.json.

    Returns
    -------
    nx.DiGraph or None
        None if reconstruction fails or networkx is unavailable.
    """
    if not HAS_NX:
        return None
    try:
        cfg = config_snapshot.get("config", {})
        graph_cfg = cfg.get("graph", {})
        gtype = graph_cfg.get("type")
        n     = int(graph_cfg.get("n", 0))
        seed  = int(graph_cfg.get("seed", 0))

        if gtype == "erdos_renyi":
            return nx.gnp_random_graph(n, float(graph_cfg["p"]), seed=seed, directed=True)
        if gtype == "barabasi_albert":
            G_und = nx.barabasi_albert_graph(n, int(graph_cfg["m"]), seed=seed)
            return G_und.to_directed()
        if gtype == "watts_strogatz":
            G_und = nx.watts_strogatz_graph(n, int(graph_cfg["k"]), float(graph_cfg["p"]), seed=seed)
            return G_und.to_directed()
        if gtype == "custom":
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            for edge in graph_cfg.get("edges", []):
                u, v = int(edge[0]), int(edge[1])
                if u != v:
                    G.add_edge(u, v)
            return G
    except Exception as exc:
        warnings.warn(f"Graph reconstruction failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Plot 1: Risk Heatmap
# ---------------------------------------------------------------------------


def plot_risk_heatmap(
    df: pd.DataFrame,
    results_dir: Path,
    output_path: Path,
    dpi: int = 120,
    max_nodes_labels: int = 10,
) -> bool:
    """Network graph with nodes coloured by Fragility Index.

    Parameters
    ----------
    df : pd.DataFrame
        Fragility data (from :func:`load_fragility_csv`).
    results_dir : Path
        Used to load config_snapshot for topology reconstruction.
    output_path : Path
        Destination PNG/PDF path.
    dpi : int
        Output resolution.
    max_nodes_labels : int
        Maximum number of node labels to draw (top FI nodes only).

    Returns
    -------
    bool
        True if the heatmap was produced successfully, False if topology
        reconstruction failed (Ghost Hub + Convergence plots still run).
    """
    if not HAS_NX:
        print("  [visualizer] networkx not available — skipping Risk Heatmap.", file=sys.stderr)
        return False

    snap = load_config_snapshot(results_dir)
    G = rebuild_graph(snap) if snap else None

    if G is None:
        print("  [visualizer] Could not rebuild graph topology — skipping Risk Heatmap.", file=sys.stderr)
        return False

    # Ensure every node in the DataFrame is in the graph
    graph_nodes = set(G.nodes())
    df_nodes    = set(df["node_id"].tolist())
    extra = df_nodes - graph_nodes
    if extra:
        G.add_nodes_from(extra)

    # Remove self-loops for clean visualisation
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    n = len(G.nodes())

    # Build node colour and size arrays indexed by node_id
    fi_map  = dict(zip(df["node_id"], df["fragility_index"]))
    fi_vals = np.array([fi_map.get(node, 0) for node in sorted(G.nodes())])
    fi_norm = mcolors.Normalize(vmin=fi_vals.min(), vmax=max(fi_vals.max(), 1))
    cmap    = matplotlib.colormaps[PALETTE_RISK]
    node_colors = [cmap(fi_norm(fi_map.get(node, 0))) for node in sorted(G.nodes())]
    node_sizes  = [
        80 + 600 * fi_norm(fi_map.get(node, 0))
        for node in sorted(G.nodes())
    ]

    # Layout — spring for small graphs, spectral for large
    if n <= 200:
        pos = nx.spring_layout(G, seed=42, k=2.0 / max(np.sqrt(n), 1))
    else:
        try:
            pos = nx.spectral_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

    # Top-FI node labels
    top_fi_nodes = df.nlargest(max_nodes_labels, "fragility_index")["node_id"].tolist()
    labels = {node: str(node) for node in top_fi_nodes}

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # Draw edges first
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=EDGE_ALPHA,
        edge_color=SPINE_COLOR,
        arrows=True,
        arrowsize=8,
        width=0.6,
        connectionstyle="arc3,rad=0.08",
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=NODE_ALPHA,
    )

    # Labels for top nodes
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7,
        font_color="white",
        font_weight="bold",
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=fi_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fragility Index (affected nodes when seeded)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Infrastructure Risk Heatmap — Cascade Fragility Index",
        fontsize=15, fontweight="bold", color=ACCENT, pad=16,
    )
    ax.axis("off")

    # Annotation: worst node
    top_node = df.loc[df["fragility_index"].idxmax()]
    ax.annotate(
        f"Highest risk: Node {int(top_node['node_id'])}  (FI={int(top_node['fragility_index'])})",
        xy=(0.01, 0.02), xycoords="axes fraction",
        fontsize=9, color=GHOST_COLOR, style="italic",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [visualizer] Risk Heatmap → {output_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 2: Ghost Hub Plot
# ---------------------------------------------------------------------------


def plot_ghost_hub(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 120,
    top_n_annotate: int = 8,
) -> None:
    """Scatter plot of In-Degree vs. Fragility Index.

    Nodes in the upper-left region have high FI despite low connectivity —
    "ghost hubs" that are disproportionately dangerous.  Nodes in the lower-
    right are highly connected but relatively safe.

    Parameters
    ----------
    df : pd.DataFrame
        Fragility data.
    output_path : Path
        Destination path.
    dpi : int
    top_n_annotate : int
        Annotate the top-N highest-FI nodes by name.
    """
    fi_col = "fragility_index"
    id_col = "in_degree"

    fi_vals = df[fi_col].values
    id_vals = df[id_col].values

    fi_norm = mcolors.Normalize(vmin=fi_vals.min(), vmax=max(fi_vals.max(), 1))
    cmap    = matplotlib.colormaps[PALETTE_RISK]
    colors  = [cmap(fi_norm(v)) for v in fi_vals]
    sizes   = 40 + 400 * fi_norm(fi_vals)

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    scatter = ax.scatter(
        id_vals, fi_vals,
        c=colors, s=sizes,
        alpha=0.80, edgecolors="white", linewidths=0.6, zorder=3,
    )

    # Reference lines: medians
    med_id = float(np.median(id_vals))
    med_fi = float(np.median(fi_vals))
    ax.axvline(med_id, color=SPINE_COLOR, linestyle="--", linewidth=1.2, label=f"Median In-Degree ({med_id:.0f})", zorder=2)
    ax.axhline(med_fi, color=SPINE_COLOR, linestyle=":",  linewidth=1.2, label=f"Median FI ({med_fi:.0f})", zorder=2)

    # Ghost hub quadrant shading (low in-degree, high FI)
    ax.fill_betweenx(
        y=[med_fi, fi_vals.max() * 1.05],
        x1=0, x2=med_id,
        alpha=0.07, color=GHOST_COLOR, label="Ghost Hub Zone",
    )

    # Annotate top-N by FI
    top_nodes = df.nlargest(top_n_annotate, fi_col)
    texts = []
    for _, row in top_nodes.iterrows():
        ax.annotate(
            f"  N{int(row['node_id'])}",
            xy=(row[id_col], row[fi_col]),
            fontsize=8, color=GHOST_COLOR, fontweight="bold",
            zorder=5,
        )

    # Colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=fi_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fragility Index", fontsize=10)

    ax.set_xlabel("In-Degree (number of upstream dependencies)", fontsize=12, color=ACCENT)
    ax.set_ylabel("Fragility Index (cascade impact when seeded)", fontsize=12, color=ACCENT)
    ax.set_title(
        "Ghost Hub Analysis — In-Degree vs. Fragility Index\n"
        "Upper-left quadrant: high-risk nodes with few visible connections",
        fontsize=13, fontweight="bold", color=ACCENT, pad=14,
    )
    ax.legend(fontsize=9, framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [visualizer] Ghost Hub Plot → {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Convergence Curve (k-sweep)
# ---------------------------------------------------------------------------


def load_k_sweep_data(k_sweep_dir: Path) -> pd.DataFrame:
    """Collect Spearman ρ and RMSE from each k-sweep sub-folder.

    Expects the k-sweep directory structure produced by scenario_manager.py:
    ``k_sweep_dir / k_{value} / results / summary.json``

    Or the legacy structure from k_sweep_analysis.py:
    ``k_sweep_dir / k_{value} / summary.json``

    Parameters
    ----------
    k_sweep_dir : Path

    Returns
    -------
    pd.DataFrame with columns: ``k``, ``spearman_rho``, ``rmse``.
    """
    records = []

    if not k_sweep_dir.exists():
        raise FileNotFoundError(f"k-sweep directory not found: {k_sweep_dir}")

    # Collect all summary.json files under the k_sweep_dir tree
    for sub in sorted(k_sweep_dir.iterdir()):
        if not sub.is_dir():
            continue

        # Try scenario_manager layout first, then legacy
        candidates = [
            sub / "results" / "summary.json",
            sub / "summary.json",
        ]
        summary_path = next((c for c in candidates if c.exists()), None)
        if summary_path is None:
            continue

        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue

        rho  = summary.get("spearman_rho")
        rmse = summary.get("rmse_det_vs_stochastic")

        # Infer k from folder name (e.g., "k_10.0" or "k_10")
        k = None
        try:
            k_str = sub.name.replace("k_", "").strip()
            k = float(k_str)
        except ValueError:
            # Try reading from config snapshot
            snap_candidates = [
                sub / "results" / "config_snapshot.json",
                sub / "config_snapshot.json",
            ]
            for snap_p in snap_candidates:
                if snap_p.exists():
                    try:
                        snap = json.loads(snap_p.read_text())
                        k = float(snap.get("config", {}).get("stochastic_k", 0))
                        break
                    except Exception:
                        pass

        if k is not None:
            # Normalise None/NaN spearman (flat distributions at very low k,
            # or engine-serialised null via _nan_to_none) → 0.0 before appending.
            # The outer guard previously blocked None rho, silently dropping the
            # record. Now we normalise first so every valid k point is retained.
            if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                rho = 0.0
            records.append({"k": k, "spearman_rho": float(rho), "rmse": float(rmse) if rmse is not None else np.nan})

    if not records:
        raise ValueError(f"No valid k-sweep summary files found under: {k_sweep_dir}")

    df = pd.DataFrame(records).sort_values("k").reset_index(drop=True)
    return df


def plot_convergence_curve(
    k_sweep_dir: Path,
    output_path: Path,
    dpi: int = 120,
) -> None:
    """Two-panel Convergence Curve: Spearman ρ and RMSE vs. k.

    Parameters
    ----------
    k_sweep_dir : Path
        Directory containing k-sweep sub-folders.
    output_path : Path
    dpi : int
    """
    try:
        df_k = load_k_sweep_data(k_sweep_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [visualizer] Convergence Curve skipped: {exc}", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BACKGROUND)
    for ax in (ax1, ax2):
        ax.set_facecolor(BACKGROUND)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)

    k_vals = df_k["k"].values
    rho_vals = df_k["spearman_rho"].values
    rmse_vals = df_k["rmse"].values

    # ── Spearman ρ ──────────────────────────────────────────────────────────
    ax1.plot(k_vals, rho_vals, marker="o", color="#2980B9", linewidth=2.2,
             markersize=8, markerfacecolor="white", markeredgewidth=2, zorder=4)
    ax1.fill_between(k_vals, rho_vals, alpha=0.12, color="#2980B9")
    ax1.axhline(1.0, color=SAFE_COLOR,  linestyle="--", linewidth=1, alpha=0.7, label="ρ = 1.0 (perfect rank agreement)")
    ax1.axhline(0.0, color=GHOST_COLOR, linestyle=":",  linewidth=1, alpha=0.7, label="ρ = 0.0 (Chaos Point)")

    # Annotate chaos point if visible
    chaos_mask = rho_vals < 0.2
    if chaos_mask.any():
        chaos_k = k_vals[chaos_mask].max()
        ax1.axvspan(k_vals.min(), chaos_k, alpha=0.07, color=GHOST_COLOR, label=f"Chaos Zone (k ≤ {chaos_k:.0f})")

    ax1.set_xlabel("Logistic Steepness (k)", fontsize=12, color=ACCENT)
    ax1.set_ylabel("Spearman ρ", fontsize=12, color=ACCENT)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Convergence: Spearman ρ → 1.0 as k → ∞\n(Stochastic rankings align with deterministic FI)", fontsize=11, fontweight="bold", color=ACCENT)
    ax1.legend(fontsize=8, framealpha=0.85)
    ax1.grid(True, alpha=0.4)

    # ── RMSE ────────────────────────────────────────────────────────────────
    valid_rmse = ~np.isnan(rmse_vals)
    if valid_rmse.any():
        ax2.plot(k_vals[valid_rmse], rmse_vals[valid_rmse],
                 marker="s", color="#E74C3C", linewidth=2.2,
                 markersize=8, markerfacecolor="white", markeredgewidth=2, zorder=4)
        ax2.fill_between(k_vals[valid_rmse], rmse_vals[valid_rmse], alpha=0.12, color="#E74C3C")
        ax2.axhline(0.0, color=SAFE_COLOR, linestyle="--", linewidth=1, alpha=0.7, label="RMSE = 0 (perfect agreement)")

    ax2.set_xlabel("Logistic Steepness (k)", fontsize=12, color=ACCENT)
    ax2.set_ylabel("RMSE (det. FI fraction vs. stochastic mean)", fontsize=12, color=ACCENT)
    ax2.set_title("Prediction Error: RMSE → 0 as k → ∞\n(Stochastic results converge to deterministic baseline)", fontsize=11, fontweight="bold", color=ACCENT)
    ax2.legend(fontsize=8, framealpha=0.85)
    ax2.grid(True, alpha=0.4)

    fig.suptitle("Uncertainty Sweep — Stochastic Convergence Analysis", fontsize=14, fontweight="bold", color=ACCENT, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [visualizer] Convergence Curve → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate board-ready visuals from cascade_engine output directories.\n\n"
            "Produces up to three plots:\n"
            "  1. Risk Heatmap (network topology coloured by Fragility Index)\n"
            "  2. Ghost Hub Plot (In-Degree vs. FI scatter)\n"
            "  3. Convergence Curve (Spearman ρ vs. k from a k-sweep)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--results-dir", "-r",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to a cascade_engine output directory (must contain fragility_results.csv).",
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("visuals"),
        metavar="DIR",
        help="Directory for output images (default: visuals/).",
    )
    p.add_argument(
        "--k-sweep-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Root directory of a k-sweep run (e.g., scenario_results/k_sweep/). "
            "Required for the Convergence Curve plot. If omitted, the curve is skipped."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output image resolution in DPI (default: 120).",
    )
    p.add_argument(
        "--fmt",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output image format (default: png).",
    )
    p.add_argument(
        "--top-n-annotate",
        dest="top_n_annotate",
        type=int,
        default=8,
        help="Number of highest-FI nodes to annotate on the Ghost Hub Plot (default: 8).",
    )
    p.add_argument(
        "--skip-heatmap",
        dest="skip_heatmap",
        action="store_true",
        help="Skip the Risk Heatmap (useful when graph topology cannot be reconstructed).",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir  = args.output_dir
    fmt         = args.fmt

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df = load_fragility_csv(results_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  [visualizer] Loaded fragility data: {len(df)} nodes from {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    # ── Plot 1: Risk Heatmap ──────────────────────────────────────────────────
    if not args.skip_heatmap:
        hm_path = output_dir / f"risk_heatmap.{fmt}"
        ok = plot_risk_heatmap(df, results_dir, hm_path, dpi=args.dpi)
        if ok:
            produced.append(hm_path)

    # ── Plot 2: Ghost Hub ─────────────────────────────────────────────────────
    gh_path = output_dir / f"ghost_hub_plot.{fmt}"
    plot_ghost_hub(df, gh_path, dpi=args.dpi, top_n_annotate=args.top_n_annotate)
    produced.append(gh_path)

    # ── Plot 3: Convergence Curve ─────────────────────────────────────────────
    if args.k_sweep_dir is not None:
        cc_path = output_dir / f"convergence_curve.{fmt}"
        plot_convergence_curve(args.k_sweep_dir, cc_path, dpi=args.dpi)
        produced.append(cc_path)
    else:
        print("  [visualizer] --k-sweep-dir not provided — Convergence Curve skipped.")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "─" * 54
    print(f"\n{sep}")
    print("  visualizer — Output Summary")
    print(sep)
    for p in produced:
        print(f"  ✓  {p}")
    if not produced:
        print("  No plots were produced.", file=sys.stderr)
    print(sep)


if __name__ == "__main__":
    main()
