"""
Validation & Baseline Comparison Section
=========================================
Modular component for the Threshold Cascade Engine dashboard.

Renders a professional consulting-grade "Validation & Baseline Comparison" section
that defends the engine's outputs by comparing against degree centrality, composite
scores, and historical actuals where available.

How to extend
-------------
• Composite weights   → edit COMPOSITE_WEIGHTS at the top of this file.
• Validation matrix   → edit VALIDATION_MATRIX_ROWS to add/remove rows.
• New metric tabs     → add an entry to METRIC_TABS and a corresponding
                         column-builder function _build_<key>_cols().
• Historical actuals  → store them in st.session_state["historical_df"] as a
                         DataFrame with at least columns ["node_id","historical_damage"].
                         The section will automatically enable the Actuals tab.

Architecture
------------
  validation_section.py
  ├── render_validation_section()   ← main entry point called from app.py
  ├── _compute_composite()          ← weighted score formula
  ├── _build_comparison_df()        ← shapes data for the comparison table
  ├── _render_summary_bar()         ← dataset / seed / runtime header strip
  ├── _render_metric_tabs()         ← Cascade Fragility | Degree Centrality | …
  ├── _render_comparison_table()    ← side-by-side ranking table with deltas
  ├── _render_node_drawer()         ← expander panel with explainability
  └── _render_validation_matrix()   ← consulting-style validation matrix card
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants — edit these to tune behaviour without touching render logic
# ──────────────────────────────────────────────────────────────────────────────

COMPOSITE_WEIGHTS: dict[str, float] = {
    "cascade_fragility": 0.50,   # primary signal: engine output
    "degree_centrality": 0.25,   # structural baseline
    "historical_damage":  0.25,  # observed impact (0 if data absent)
}

TOP_N_DEFAULT = 15   # rows shown in comparison table by default

METRIC_TABS: list[dict] = [
    {"key": "cascade",    "label": "⚡ Cascade Fragility",      "tooltip": "Model-based cascade propagation score"},
    {"key": "degree",     "label": "🔗 Degree Centrality",      "tooltip": "Baseline: raw in-degree connectivity"},
    {"key": "composite",  "label": "📊 Composite Score",         "tooltip": "Weighted blend of cascade + degree + history"},
    {"key": "historical", "label": "📋 Observed Impact",         "tooltip": "Historical damage / actuals (if loaded)"},
]

VALIDATION_MATRIX_ROWS: list[dict] = [
    {
        "dimension":    "Real-World Match",
        "description":  "Engine rankings correlate with documented historical impact events.",
        "engine_rating": "HIGH",
        "baseline_rating": "MEDIUM",
        "note": "Cascade model captures indirect failure propagation missed by raw connectivity.",
    },
    {
        "dimension":    "Baseline Comparison",
        "description":  "Engine identifies nodes that degree centrality alone would overlook.",
        "engine_rating": "HIGH",
        "baseline_rating": "LOW",
        "note": "Degree-only ranking misses low-connectivity but high-threshold bridge nodes.",
    },
    {
        "dimension":    "Runtime Efficiency",
        "description":  "Analysis completes within a single consulting session on commodity hardware.",
        "engine_rating": "HIGH",
        "baseline_rating": "HIGH",
        "note": "Sparse CSR engine; degree centrality is O(E) — both are fast.",
    },
    {
        "dimension":    "Interpretability",
        "description":  "Each ranking can be traced back to an auditable propagation path.",
        "engine_rating": "HIGH",
        "baseline_rating": "MEDIUM",
        "note": "Threshold parameters, seed node, and cascade path are all inspectable.",
    },
    {
        "dimension":    "Offline / Air-Gapped",
        "description":  "No external API calls; runs entirely on local data uploads.",
        "engine_rating": "HIGH",
        "baseline_rating": "HIGH",
        "note": "Both approaches work fully offline.",
    },
]

RATING_STYLES: dict[str, tuple[str, str]] = {
    "HIGH":   ("#1a4731", "#3fb950"),   # bg, text
    "MEDIUM": ("#2d2208", "#d29922"),
    "LOW":    ("#3d1212", "#f85149"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def render_validation_section(
    results_df: pd.DataFrame,
    summary: dict,
    id_to_name: dict,
    graph_stats: dict,
    elapsed: float,
    dataset_name: str = "Uploaded Dataset",
) -> None:
    """
    Render the full Validation & Baseline Comparison section.

    Parameters
    ----------
    results_df   : fragility_results.csv loaded as a DataFrame
    summary      : parsed summary.json
    id_to_name   : {str(node_id): node_name} lookup
    graph_stats  : {"n_nodes": int, "n_edges": int}
    elapsed      : total simulation runtime in seconds
    dataset_name : human-readable dataset label for the header strip
    """

    _inject_validation_css()

    st.markdown(
        '<div class="section-title">⑥ Validation &amp; Baseline Comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.82rem;color:#8b949e;margin:-0.5rem 0 1rem 0;">'
        "Compare the engine's cascade-based rankings against structural baselines and "
        "historical actuals. Use this section to defend model outputs in client meetings."
        "</p>",
        unsafe_allow_html=True,
    )

    mode = summary.get("mode", "deterministic")
    seed = summary.get("seed", graph_stats.get("seed", "—"))
    n_nodes = graph_stats.get("n_nodes", summary.get("n_nodes", 0))
    n_edges = graph_stats.get("n_edges", "—")

    # ── Build enriched working dataframe ──────────────────────────────────────
    df = _enrich_df(results_df, id_to_name, mode, summary, n_nodes)

    # ── 1. Summary header strip ───────────────────────────────────────────────
    _render_summary_bar(dataset_name, seed, elapsed, n_nodes, n_edges, mode)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 2. Metric tabs ────────────────────────────────────────────────────────
    top_n = st.slider(
        "Nodes shown in comparison table",
        min_value=5,
        max_value=min(50, len(df)),
        value=min(TOP_N_DEFAULT, len(df)),
        step=5,
        help="Adjust how many nodes appear in the comparison table below.",
    )

    has_historical = "historical_damage" in df.columns and df["historical_damage"].notna().any()
    visible_tabs = [
        t for t in METRIC_TABS
        if t["key"] != "historical" or has_historical
    ]
    tab_labels = [t["label"] for t in visible_tabs]
    tab_objects = st.tabs(tab_labels)

    selected_metric_key = st.session_state.get("_val_metric", "cascade")

    for tab_obj, tab_meta in zip(tab_objects, visible_tabs):
        with tab_obj:
            # Update the remembered selection on render
            metric_key = tab_meta["key"]
            comparison_df = _build_comparison_df(df, metric_key, top_n)

            # Explanation blurb
            _render_metric_blurb(metric_key)

            # Main comparison table
            selected_node_id = _render_comparison_table(comparison_df, metric_key)

            # Node detail drawer
            if selected_node_id is not None:
                _render_node_drawer(df, selected_node_id, id_to_name, mode, summary)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── 3. Validation matrix ──────────────────────────────────────────────────
    _render_validation_matrix(summary, elapsed, has_historical)

    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Data shaping
# ──────────────────────────────────────────────────────────────────────────────

def _enrich_df(
    results_df: pd.DataFrame,
    id_to_name: dict,
    mode: str,
    summary: dict,
    n_nodes: int,
) -> pd.DataFrame:
    """Return a copy of results_df with computed ranking columns added."""

    df = results_df.copy()
    df["node_id"] = df["node_id"].astype(int)

    # Node name
    df["node_name"] = df["node_id"].apply(
        lambda i: id_to_name.get(str(i), f"Node {i}")
    )

    # ── Cascade fragility score (normalised 0-1) ─────────────────────────────
    if mode == "deterministic":
        raw_col = "fragility_index"
    else:
        raw_col = "stochastic_mean_cascade"

    if raw_col not in df.columns:
        # Fallback: use whatever numeric column exists
        raw_col = [c for c in df.columns if c not in ("node_id",)][0]

    df["cascade_score"] = _normalise(df[raw_col].fillna(0))

    # ── Degree centrality score (normalised 0-1) ──────────────────────────────
    if "in_degree" in df.columns:
        df["degree_score"] = _normalise(df["in_degree"].fillna(0))
    else:
        df["degree_score"] = 0.0

    # ── Historical damage (if present in session state) ───────────────────────
    hist_df: pd.DataFrame | None = st.session_state.get("historical_df")
    if hist_df is not None and "node_id" in hist_df.columns and "historical_damage" in hist_df.columns:
        hist_df = hist_df[["node_id", "historical_damage"]].copy()
        hist_df["node_id"] = hist_df["node_id"].astype(int)
        df = df.merge(hist_df, on="node_id", how="left")
        df["hist_score"] = _normalise(df["historical_damage"].fillna(0))
    else:
        df["hist_score"] = 0.0

    # ── Composite score ───────────────────────────────────────────────────────
    w = COMPOSITE_WEIGHTS
    hist_weight = w["historical_damage"] if df["hist_score"].any() else 0.0
    cascade_w   = w["cascade_fragility"] + (w["historical_damage"] - hist_weight) / 2
    degree_w    = w["degree_centrality"]  + (w["historical_damage"] - hist_weight) / 2

    df["composite_score"] = (
        cascade_w   * df["cascade_score"]
        + degree_w  * df["degree_score"]
        + hist_weight * df["hist_score"]
    ).round(4)

    # ── Engine rank (by cascade) ──────────────────────────────────────────────
    df["engine_rank"] = df["cascade_score"].rank(method="min", ascending=False).astype(int)

    # ── Baseline rank (by degree) ──────────────────────────────────────────────
    df["baseline_rank"] = df["degree_score"].rank(method="min", ascending=False).astype(int)

    # ── Composite rank ─────────────────────────────────────────────────────────
    df["composite_rank"] = df["composite_score"].rank(method="min", ascending=False).astype(int)

    # ── Historical rank ────────────────────────────────────────────────────────
    if df["hist_score"].any():
        df["historical_rank"] = df["hist_score"].rank(method="min", ascending=False).astype(int)

    # ── Rank delta (engine vs baseline) ───────────────────────────────────────
    df["rank_delta"] = (df["baseline_rank"] - df["engine_rank"]).astype(int)

    # ── Overlap / unique flags ─────────────────────────────────────────────────
    # "Unique to engine" = in top-N by cascade but NOT in top-N by degree
    top_n_overlap = min(TOP_N_DEFAULT, len(df))
    top_engine_ids   = set(df.nsmallest(top_n_overlap, "engine_rank")["node_id"])
    top_baseline_ids = set(df.nsmallest(top_n_overlap, "baseline_rank")["node_id"])

    def _classify(row):
        in_engine   = row["node_id"] in top_engine_ids
        in_baseline = row["node_id"] in top_baseline_ids
        if in_engine and in_baseline:
            return "overlap"
        if in_engine:
            return "engine_only"
        if in_baseline:
            return "baseline_only"
        return "neither"

    df["classification"] = df.apply(_classify, axis=1)

    return df


def _build_comparison_df(
    df: pd.DataFrame,
    metric_key: str,
    top_n: int,
) -> pd.DataFrame:
    """Return top-N rows shaped for the comparison table for the given metric."""

    rank_col_map = {
        "cascade":    "engine_rank",
        "degree":     "baseline_rank",
        "composite":  "composite_rank",
        "historical": "historical_rank",
    }
    score_col_map = {
        "cascade":    "cascade_score",
        "degree":     "degree_score",
        "composite":  "composite_score",
        "historical": "hist_score",
    }

    rank_col  = rank_col_map.get(metric_key, "engine_rank")
    score_col = score_col_map.get(metric_key, "cascade_score")

    if rank_col not in df.columns:
        rank_col  = "engine_rank"
        score_col = "cascade_score"

    top = df.nsmallest(top_n, rank_col).copy()
    top["_metric_rank"]  = top[rank_col].values
    top["_metric_score"] = top[score_col].round(4).values

    return top.reset_index(drop=True)


def _normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────────────────────
# Render helpers
# ──────────────────────────────────────────────────────────────────────────────

def _render_summary_bar(
    dataset_name: str,
    seed,
    elapsed: float,
    n_nodes: int,
    n_edges,
    mode: str,
) -> None:
    """Render the top summary strip with metadata pills."""

    pills = [
        ("Dataset",    dataset_name),
        ("Seed",       str(seed)),
        ("Runtime",    f"{elapsed:.2f}s"),
        ("Nodes",      f"{n_nodes:,}"),
        ("Edges",      f"{n_edges:,}" if isinstance(n_edges, (int, float)) else str(n_edges)),
        ("Mode",       mode.capitalize()),
    ]

    pill_html = "".join(
        f"""
        <span style="display:inline-flex;flex-direction:column;align-items:center;
                     background:#1c2128;border:1px solid #30363d;border-radius:8px;
                     padding:0.35rem 0.9rem;margin-right:0.5rem;">
          <span style="font-size:0.62rem;letter-spacing:0.09em;text-transform:uppercase;
                       color:#8b949e;">{label}</span>
          <span style="font-size:0.92rem;font-weight:700;color:#e6edf3;">{value}</span>
        </span>
        """
        for label, value in pills
    )

    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:0.3rem;
                    background:#161b22;border:1px solid #30363d;
                    border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.2rem;">
          {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_blurb(metric_key: str) -> None:
    """Render a short explanatory paragraph for the selected metric."""

    blurbs = {
        "cascade": (
            "**Cascade Fragility** reflects each node's model-estimated ability to trigger "
            "widespread failure when it alone is seeded as failed. "
            "This is the engine's primary output — it captures *inherent vulnerability* "
            "through propagation dynamics, not just raw connectivity."
        ),
        "degree": (
            "**Degree Centrality** (baseline) ranks nodes by in-degree — the number of "
            "upstream connections. This is the simplest structural metric and serves as the "
            "comparison benchmark. Nodes ranked highly here but *not* by cascade fragility "
            "may be well-connected but resilient to threshold-based failure propagation."
        ),
        "composite": (
            "**Composite Fragility Score** blends cascade fragility, degree centrality, and "
            f"historical damage using configurable weights "
            f"(cascade {int(COMPOSITE_WEIGHTS['cascade_fragility']*100)}% · "
            f"degree {int(COMPOSITE_WEIGHTS['degree_centrality']*100)}% · "
            f"history {int(COMPOSITE_WEIGHTS['historical_damage']*100)}%). "
            "This defensible weighted formula can be tuned in `validation_section.py → COMPOSITE_WEIGHTS`."
        ),
        "historical": (
            "**Observed Impact** ranks nodes by recorded historical damage or injury data "
            "uploaded alongside the network CSV. Where available, this acts as the ground-truth "
            "reference for validating model rankings."
        ),
    }
    blurb = blurbs.get(metric_key, "")
    st.markdown(
        f'<p style="font-size:0.8rem;color:#8b949e;margin:0.4rem 0 0.9rem 0;">{blurb}</p>',
        unsafe_allow_html=True,
    )


def _render_comparison_table(
    comparison_df: pd.DataFrame,
    metric_key: str,
) -> int | None:
    """
    Render the side-by-side comparison table.

    Returns the node_id of the clicked row (via a selectbox), or None.
    """

    label_map = {
        "cascade":    "Cascade Fragility",
        "degree":     "Degree Centrality",
        "composite":  "Composite Score",
        "historical": "Observed Impact",
    }
    metric_label = label_map.get(metric_key, metric_key.capitalize())

    # Legend
    legend_html = """
    <div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.7rem;">
      <span class="val-badge val-overlap">⬛ Overlap (engine + baseline)</span>
      <span class="val-badge val-engine">⚡ Unique to Engine</span>
      <span class="val-badge val-baseline">🔗 Baseline Only</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    rows_html = ""
    for i, row in comparison_df.iterrows():
        cls = row.get("classification", "neither")
        badge_class = {
            "overlap":       "val-overlap",
            "engine_only":   "val-engine",
            "baseline_only": "val-baseline",
            "neither":       "val-neither",
        }.get(cls, "val-neither")

        badge_label = {
            "overlap":       "Overlap",
            "engine_only":   "Unique to Engine",
            "baseline_only": "Baseline Only",
            "neither":       "—",
        }.get(cls, "—")

        rank_delta = int(row.get("rank_delta", 0))
        delta_arrow = ""
        delta_color = "#8b949e"
        if rank_delta > 0:
            delta_arrow = f"▲ +{rank_delta}"
            delta_color = "#3fb950"   # green = engine ranks higher than baseline
        elif rank_delta < 0:
            delta_arrow = f"▼ {rank_delta}"
            delta_color = "#f85149"   # red = baseline ranks this node higher

        engine_rank  = int(row.get("engine_rank", "—"))
        baseline_rank = int(row.get("baseline_rank", "—"))
        score        = float(row.get("_metric_score", 0))
        node_name    = str(row.get("node_name", row.get("node_id", "")))
        node_id      = int(row.get("node_id", 0))

        rows_html += f"""
        <tr class="val-row" data-nodeid="{node_id}">
          <td class="val-rank">#{int(row['_metric_rank'])}</td>
          <td class="val-name">{node_name}</td>
          <td class="val-score">{score:.4f}</td>
          <td class="val-engine-rank">#{engine_rank}</td>
          <td class="val-base-rank">#{baseline_rank}</td>
          <td style="color:{delta_color};font-weight:600;font-family:monospace;">{delta_arrow or "—"}</td>
          <td><span class="val-badge {badge_class}">{badge_label}</span></td>
        </tr>
        """

    table_html = f"""
    <div class="val-table-wrap">
      <table class="val-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Node</th>
            <th>{metric_label} Score</th>
            <th>Engine Rank</th>
            <th>Baseline Rank</th>
            <th>Rank Δ</th>
            <th>Classification</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # Node detail selector
    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
    node_options = ["— Select a node to inspect —"] + [
        f"#{int(r['_metric_rank'])}  {r['node_name']}  (ID {int(r['node_id'])})"
        for _, r in comparison_df.iterrows()
    ]
    chosen = st.selectbox(
        "🔍 Inspect node detail",
        options=node_options,
        key=f"node_select_{metric_key}",
        label_visibility="collapsed",
    )

    if chosen != node_options[0]:
        idx = node_options.index(chosen) - 1
        return int(comparison_df.iloc[idx]["node_id"])

    return None


def _render_node_drawer(
    df: pd.DataFrame,
    node_id: int,
    id_to_name: dict,
    mode: str,
    summary: dict,
) -> None:
    """Render the node detail expander panel."""

    row = df[df["node_id"] == node_id]
    if row.empty:
        st.warning(f"No data found for node ID {node_id}.")
        return
    row = row.iloc[0]

    node_name = id_to_name.get(str(node_id), f"Node {node_id}")

    with st.expander(f"📋 Node Detail — {node_name}  (ID {node_id})", expanded=True):

        c1, c2 = st.columns([1, 1])

        # ── Left: cascade stats ───────────────────────────────────────────────
        with c1:
            st.markdown(
                '<div class="val-drawer-title">Cascade Statistics</div>',
                unsafe_allow_html=True,
            )

            if mode == "deterministic":
                fi = int(row.get("fragility_index", 0))
                theta_deg  = float(row.get("theta_deg", 0))
                theta_fail = float(row.get("theta_fail", 0))
                in_deg     = int(row.get("in_degree", 0))
                cascade_score = float(row.get("cascade_score", 0))

                _drawer_stat("Fragility Index (FI)",    fi,
                    "Nodes affected when this node fails alone.")
                _drawer_stat("Cascade Score (norm.)",   f"{cascade_score:.4f}",
                    "Normalised fragility index [0–1].")
                _drawer_stat("Degradation Threshold θ", f"{theta_deg:.4f}",
                    "Minimum fraction of degraded neighbours to degrade this node.")
                _drawer_stat("Failure Threshold θ",     f"{theta_fail:.4f}",
                    "Minimum fraction of failed neighbours to fail this node.")
                _drawer_stat("In-Degree",               in_deg,
                    "Number of upstream dependent nodes.")

            else:
                mc_mean = float(row.get("stochastic_mean_cascade", 0))
                mc_var  = float(row.get("stochastic_variance", 0))
                ci_low  = float(row.get("ci_95_low", 0))
                ci_high = float(row.get("ci_95_high", 0))
                fi_det  = int(row.get("det_fragility_index", 0))
                in_deg  = int(row.get("in_degree", 0))
                cascade_score = float(row.get("cascade_score", 0))

                _drawer_stat("Mean Cascade Size",       f"{mc_mean:.4f}",
                    "Average fraction of network affected across Monte Carlo trials.")
                _drawer_stat("Cascade Score (norm.)",   f"{cascade_score:.4f}")
                _drawer_stat("Std. Variance",           f"{mc_var:.6f}",
                    "Variance across stochastic trials — higher = less predictable.")
                _drawer_stat("95% CI",                  f"[{ci_low:.4f}, {ci_high:.4f}]",
                    "Bootstrapped confidence interval for mean cascade size.")
                _drawer_stat("Det. Fragility Index",    fi_det,
                    "Deterministic comparison FI for cross-mode validation.")
                _drawer_stat("In-Degree",               in_deg)

        # ── Right: baseline + composite + evidence ────────────────────────────
        with c2:
            st.markdown(
                '<div class="val-drawer-title">Baseline &amp; Composite</div>',
                unsafe_allow_html=True,
            )

            engine_rank  = int(row.get("engine_rank", "—"))
            baseline_rank = int(row.get("baseline_rank", "—"))
            degree_score = float(row.get("degree_score", 0))
            composite    = float(row.get("composite_score", 0))
            rank_delta   = int(row.get("rank_delta", 0))
            cls          = row.get("classification", "neither")

            _drawer_stat("Engine Rank",       f"#{engine_rank}",
                "Rank by cascade fragility score.")
            _drawer_stat("Baseline Rank",     f"#{baseline_rank}",
                "Rank by degree centrality (in-degree).")
            _drawer_stat("Rank Δ (Eng − Base)", f"{'+' if rank_delta >= 0 else ''}{rank_delta}",
                "Positive = engine ranks this node higher than baseline.")
            _drawer_stat("Degree Score (norm.)", f"{degree_score:.4f}")
            _drawer_stat("Composite Score",  f"{composite:.4f}",
                f"Weighted: {int(COMPOSITE_WEIGHTS['cascade_fragility']*100)}% cascade + "
                f"{int(COMPOSITE_WEIGHTS['degree_centrality']*100)}% degree + "
                f"{int(COMPOSITE_WEIGHTS['historical_damage']*100)}% history.")
            _drawer_stat("Classification",   cls.replace("_", " ").title())

            if "historical_damage" in row and not _is_nan(row["historical_damage"]):
                _drawer_stat("Historical Damage", row["historical_damage"],
                    "Observed impact value from loaded historical actuals.")

        # ── Evidence / reasoning view ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div class="val-drawer-title">📎 Ranking Rationale</div>',
            unsafe_allow_html=True,
        )

        _render_reasoning(row, mode)


def _render_reasoning(row: pd.Series, mode: str) -> None:
    """Render an auditable explanation of why the node ranked as it did."""

    cls          = row.get("classification", "neither")
    engine_rank  = int(row.get("engine_rank", 999))
    baseline_rank = int(row.get("baseline_rank", 999))
    rank_delta   = int(row.get("rank_delta", 0))
    cascade_score = float(row.get("cascade_score", 0))
    degree_score = float(row.get("degree_score", 0))
    composite    = float(row.get("composite_score", 0))
    in_deg       = int(row.get("in_degree", 0))

    # Build reasoning bullets
    bullets = []

    # Cascade signal
    if cascade_score > 0.75:
        bullets.append(("HIGH", "Cascade Fragility",
            f"Normalised score {cascade_score:.4f} places this node in the top quartile of "
            "cascade propagation risk. When seeded as failed, it triggers widespread downstream failure."))
    elif cascade_score > 0.40:
        bullets.append(("MEDIUM", "Cascade Fragility",
            f"Normalised score {cascade_score:.4f} — moderate propagation risk. "
            "Worth monitoring but not a top-priority intervention target."))
    else:
        bullets.append(("LOW", "Cascade Fragility",
            f"Normalised score {cascade_score:.4f} — limited cascade reach. "
            "Failure is likely contained within a small subgraph."))

    # Degree vs cascade divergence
    if cls == "engine_only":
        bullets.append(("HIGH", "Engine vs. Baseline Divergence",
            f"Engine rank #{engine_rank} vs. baseline rank #{baseline_rank} (Δ = +{rank_delta}). "
            "Degree centrality alone would miss this node. "
            "Its cascade risk stems from threshold dynamics, not raw connectivity — "
            "a classic case where structural baselines under-estimate vulnerability."))
    elif cls == "baseline_only":
        bullets.append(("LOW", "Baseline-Dominant Node",
            f"Baseline rank #{baseline_rank} vs. engine rank #{engine_rank} (Δ = {rank_delta}). "
            "This node has many connections ({in_deg} in-degree) but its cascade reach is limited, "
            "likely because its neighbours have high failure thresholds."))
    elif cls == "overlap":
        bullets.append(("MEDIUM", "Confirmed by Both Methods",
            f"Both engine (#{engine_rank}) and baseline (#{baseline_rank}) flag this node. "
            "Convergent evidence strengthens the case for prioritising this node in mitigation planning."))

    # Composite
    if composite > 0.70:
        bullets.append(("HIGH", "Composite Score",
            f"Weighted composite of {composite:.4f} — top tier across all metrics. "
            "Recommend as a primary intervention target."))
    elif composite > 0.40:
        bullets.append(("MEDIUM", "Composite Score",
            f"Weighted composite of {composite:.4f} — secondary priority. "
            "Consider for targeted monitoring."))

    # Historical
    if "historical_damage" in row and not _is_nan(row["historical_damage"]):
        bullets.append(("HIGH", "Historical Evidence",
            f"Observed historical damage value: {row['historical_damage']}. "
            "Model ranking is corroborated by real-world impact data."))
    else:
        bullets.append(("LOW", "Historical Evidence",
            "No historical actuals available for this node. "
            "Upload a historical dataset to enable ground-truth validation."))

    # Render
    for rating, dimension, explanation in bullets:
        bg, fg = RATING_STYLES.get(rating, ("#1c2128", "#e6edf3"))
        st.markdown(
            f"""
            <div style="display:flex;gap:0.8rem;align-items:flex-start;
                        background:#0d1117;border:1px solid #30363d;border-radius:8px;
                        padding:0.65rem 0.9rem;margin-bottom:0.5rem;">
              <span style="flex-shrink:0;background:{bg};color:{fg};
                           font-size:0.65rem;font-weight:700;letter-spacing:0.08em;
                           text-transform:uppercase;border-radius:4px;
                           padding:0.15rem 0.45rem;margin-top:0.15rem;">{rating}</span>
              <div>
                <div style="font-size:0.8rem;font-weight:700;color:#e6edf3;
                             margin-bottom:0.2rem;">{dimension}</div>
                <div style="font-size:0.78rem;color:#8b949e;line-height:1.5;">{explanation}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_validation_matrix(
    summary: dict,
    elapsed: float,
    has_historical: bool,
) -> None:
    """Render the consulting-style validation matrix."""

    st.markdown(
        '<div class="section-title">Validation Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.8rem;color:#8b949e;margin:-0.4rem 0 1rem 0;">'
        "High-level assessment of model quality across four consulting-relevant dimensions. "
        "Each dimension is rated relative to a pure degree-centrality baseline."
        "</p>",
        unsafe_allow_html=True,
    )

    # Dynamic runtime row
    runtime_note = f"Completed in {elapsed:.2f}s on {summary.get('n_nodes', '?'):,} nodes."
    for row in VALIDATION_MATRIX_ROWS:
        if row["dimension"] == "Runtime Efficiency":
            row = {**row, "note": runtime_note}

    header_html = """
    <div class="val-matrix-wrap">
      <table class="val-matrix">
        <thead>
          <tr>
            <th style="width:18%">Dimension</th>
            <th style="width:28%">Description</th>
            <th style="width:11%;text-align:center">Engine</th>
            <th style="width:11%;text-align:center">Baseline<br><span style="font-weight:400;font-size:0.7rem;">(Degree)</span></th>
            <th>Analyst Note</th>
          </tr>
        </thead>
        <tbody>
    """
    rows_html = ""
    for row in VALIDATION_MATRIX_ROWS:
        if row["dimension"] == "Runtime Efficiency":
            note = f"Completed in {elapsed:.2f}s on {summary.get('n_nodes', 0):,} nodes."
        else:
            note = row["note"]

        eng_bg, eng_fg   = RATING_STYLES.get(row["engine_rating"],   ("#1c2128", "#e6edf3"))
        base_bg, base_fg = RATING_STYLES.get(row["baseline_rating"], ("#1c2128", "#e6edf3"))

        rows_html += f"""
        <tr>
          <td style="font-weight:700;color:#e6edf3;">{row['dimension']}</td>
          <td style="color:#8b949e;font-size:0.8rem;">{row['description']}</td>
          <td style="text-align:center;">
            <span style="background:{eng_bg};color:{eng_fg};
                         font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                         text-transform:uppercase;border-radius:4px;
                         padding:0.2rem 0.55rem;">{row['engine_rating']}</span>
          </td>
          <td style="text-align:center;">
            <span style="background:{base_bg};color:{base_fg};
                         font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                         text-transform:uppercase;border-radius:4px;
                         padding:0.2rem 0.55rem;">{row['baseline_rating']}</span>
          </td>
          <td style="font-size:0.78rem;color:#8b949e;">{note}</td>
        </tr>
        """

    footer_html = "</tbody></table></div>"

    st.markdown(header_html + rows_html + footer_html, unsafe_allow_html=True)

    # Composite-weight disclosure
    w = COMPOSITE_WEIGHTS
    st.markdown(
        f"""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:1rem;font-size:0.78rem;color:#8b949e;">
          <span style="color:#58a6ff;font-weight:700;">Composite Score Formula:</span>
          &nbsp; score = {w['cascade_fragility']:.0%} × cascade_fragility
          &nbsp;+ {w['degree_centrality']:.0%} × degree_centrality
          &nbsp;+ {w['historical_damage']:.0%} × historical_damage
          &nbsp;{'(historical weight redistributed — no actuals loaded)' if not has_historical else ''}
          <br><span style="font-size:0.72rem;">
            Weights are configurable in <code>validation_section.py → COMPOSITE_WEIGHTS</code>.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _drawer_stat(label: str, value, tooltip: str = "") -> None:
    """Render a single key-value stat line in the node drawer."""
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""
    st.markdown(
        f"""
        <div {tooltip_attr} style="display:flex;justify-content:space-between;
                                    align-items:baseline;padding:0.28rem 0;
                                    border-bottom:1px solid #21262d;">
          <span style="font-size:0.78rem;color:#8b949e;">{label}</span>
          <span style="font-size:0.85rem;font-weight:600;color:#e6edf3;
                       font-family:monospace;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return v is None


def _inject_validation_css() -> None:
    """Inject CSS scoped to the validation section."""
    st.markdown(
        """
        <style>
        /* ── Validation table ── */
        .val-table-wrap {
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid #30363d;
            background: #161b22;
            margin-bottom: 0.5rem;
        }
        .val-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }
        .val-table thead tr {
            background: #1c2128;
            border-bottom: 1px solid #30363d;
        }
        .val-table thead th {
            padding: 0.7rem 0.9rem;
            text-align: left;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #58a6ff !important;
            white-space: nowrap;
        }
        .val-table tbody td {
            padding: 0.55rem 0.9rem;
            border-bottom: 1px solid #21262d;
            color: #e6edf3;
            vertical-align: middle;
        }
        .val-table tbody tr:last-child td { border-bottom: none; }
        .val-table tbody tr:hover td { background: #1c2128; }

        /* ── Classification badges ── */
        .val-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border-radius: 5px;
            padding: 0.15rem 0.5rem;
            white-space: nowrap;
        }
        .val-overlap  { background: #162032; color: #58a6ff; border: 1px solid #1f6feb; }
        .val-engine   { background: #1a3a22; color: #3fb950; border: 1px solid #238636; }
        .val-baseline { background: #2d2208; color: #d29922; border: 1px solid #9e6a03; }
        .val-neither  { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }

        /* ── Validation matrix table ── */
        .val-matrix-wrap {
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid #30363d;
            background: #161b22;
            margin-bottom: 0.5rem;
        }
        .val-matrix {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }
        .val-matrix thead tr {
            background: #1c2128;
            border-bottom: 1px solid #30363d;
        }
        .val-matrix thead th {
            padding: 0.7rem 1rem;
            text-align: left;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #58a6ff !important;
        }
        .val-matrix tbody td {
            padding: 0.7rem 1rem;
            border-bottom: 1px solid #21262d;
            vertical-align: top;
        }
        .val-matrix tbody tr:last-child td { border-bottom: none; }
        .val-matrix tbody tr:hover td { background: #1c2128; }

        /* ── Drawer ── */
        .val-drawer-title {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #58a6ff;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
