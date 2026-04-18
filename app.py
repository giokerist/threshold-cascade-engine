"""
Cascade Propagation Engine — Streamlit Dashboard
=================================================
Enterprise-grade GUI for the threshold-based cascade risk simulation toolkit.
Replaces the terminal workflow with a click-driven, dark-mode web dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Path bootstrap — ensures cascade_engine is importable from any working dir
# ──────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from cascade_engine.runner import (
    _build_experiment,
    _run_deterministic,
    _run_stochastic,
    _ensure_dir,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cascade Propagation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark enterprise theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* ── Section cards ── */
    .section-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #58a6ff;
        margin-bottom: 0.8rem;
    }

    /* ── Metric cards ── */
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1;
    }
    .metric-unit {
        font-size: 0.7rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* ── Worst-case highlight ── */
    .worst-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #1c1a2e 100%);
        border: 1px solid #6e40c9;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .worst-title { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #a371f7; }
    .worst-value { font-size: 2.4rem; font-weight: 800; color: #f0883e; line-height: 1; margin: 0.2rem 0; }
    .worst-sub   { font-size: 0.82rem; color: #8b949e; }

    /* ── Status banner ── */
    .status-ok  { background:#0f2419; border:1px solid #238636; color:#3fb950;
                  border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; }
    .status-err { background:#1f0f0f; border:1px solid #da3633; color:#f85149;
                  border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; }
    .status-run { background:#0f1a2e; border:1px solid #1f6feb; color:#58a6ff;
                  border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; }

    /* ── Run button ── */
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #238636 0%, #1a7f37 100%) !important;
        color: #ffffff !important;
        border: 1px solid #2ea043 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.55rem 1.5rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%) !important;
        border-color: #3fb950 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(46,160,67,0.35) !important;
    }

    /* ── Inputs ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #30363d !important;
        border-radius: 8px !important;
        background: #0d1117 !important;
    }

    /* ── Tables ── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    thead th { background: #1c2128 !important; color: #58a6ff !important; }
    tbody tr:hover td { background: #1c2128 !important; }

    /* ── Progress ── */
    [data-testid="stProgress"] > div > div { background-color: #238636 !important; }

    /* ── Logo / header ── */
    .app-header {
        display: flex; align-items: center; gap: 1rem;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 1.4rem;
    }
    .app-logo { font-size: 2rem; }
    .app-name  { font-size: 1.35rem; font-weight: 800; color: #e6edf3; letter-spacing: -0.01em; }
    .app-sub   { font-size: 0.8rem; color: #8b949e; margin-top: 0.1rem; }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #30363d; margin: 1rem 0; }

    /* ── Node badge ── */
    .node-badge {
        display: inline-block;
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 0.1rem 0.45rem;
        font-size: 0.78rem;
        font-family: monospace;
        color: #79c0ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
for _key, _default in [
    ("df_headers", []),
    ("sim_ran", False),
    ("results_df", None),
    ("summary_json", {}),
    ("node_lookup", {}),
    ("output_dir", None),
    ("run_error", None),
    ("graph_stats", {}),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = APP_DIR / "sim_output"
_ensure_dir(OUTPUT_DIR)


def _read_headers(uploaded_file) -> list[str]:
    """Read only the first row of the uploaded CSV to extract column names."""
    uploaded_file.seek(0)
    first_chunk = io.StringIO(uploaded_file.read(8192).decode("utf-8", errors="replace"))
    uploaded_file.seek(0)
    df = pd.read_csv(first_chunk, nrows=0)
    return list(df.columns)


def _compose_node_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Compose a single node label from one or more columns by joining with '|'.
    All values are cast to stripped strings before joining.

        cols = ["STATE", "CZ_NAME"]  →  "TX|HARRIS"
        cols = ["EVENT_TYPE"]        →  "Tornado"
    """
    parts = [df[c].astype(str).str.strip() for c in cols]
    if len(parts) == 1:
        return parts[0]
    return parts[0].str.cat(parts[1:], sep="|")


def _build_graph_from_csv(
    uploaded_file,
    src_cols: list[str],
    tgt_cols: list[str],
    fi_cols: list[str],
    status_cb,
) -> tuple[list[list[int, int]], dict[str, int], dict[int, str], int]:
    """
    Parse CSV, compose node labels from column groups, map to integer IDs,
    and return a deduplicated edge list ready for the cascade engine.

    Node label composition
    ----------------------
    src_cols = ["STATE", "CZ_NAME"]  →  source label: "TX|HARRIS"
    tgt_cols = ["EVENT_TYPE"]        →  target label: "Tornado"

    Returns
    -------
    edges      : list of [int, int] for the config JSON
    name_to_id : dict str → int
    id_to_name : dict int → str  (saved as node_lookup.json)
    n_nodes    : total distinct node count
    """
    status_cb("Reading CSV and building network…")

    uploaded_file.seek(0)
    all_cols_needed = list(dict.fromkeys(src_cols + tgt_cols + fi_cols))
    df = pd.read_csv(uploaded_file, usecols=all_cols_needed, low_memory=False)

    # Drop rows with NA in any selected column
    before = len(df)
    df = df.dropna(subset=all_cols_needed)
    dropped = before - len(df)
    status_cb(f"Dropped {dropped:,} rows with NA values. {len(df):,} valid rows remain.")

    # Compose composite node labels
    df["__src__"] = _compose_node_key(df, src_cols)
    df["__tgt__"] = _compose_node_key(df, tgt_cols)

    # Build sorted unique node list → deterministic 0-indexed mapping
    all_nodes = sorted(set(df["__src__"].tolist() + df["__tgt__"].tolist()))
    name_to_id = {name: idx for idx, name in enumerate(all_nodes)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    n_nodes = len(all_nodes)

    status_cb(
        f"Composed node labels from {src_cols} → {tgt_cols}. "
        f"Mapped {n_nodes:,} unique nodes to integer IDs."
    )

    # Unique directed edges (self-loops silently dropped)
    edge_pairs = df[["__src__", "__tgt__"]].drop_duplicates()
    edges = [
        [name_to_id[r["__src__"]], name_to_id[r["__tgt__"]]]
        for _, r in edge_pairs.iterrows()
        if r["__src__"] != r["__tgt__"]
    ]

    status_cb(f"Built {len(edges):,} unique directed edges.")
    return edges, name_to_id, id_to_name, n_nodes


def _generate_config(
    n_nodes: int,
    edges: list,
    deg_low: float,
    deg_high: float,
    fail_low: float,
    fail_high: float,
    seed: int,
    mode: str,
) -> dict:
    """Assemble the cascade_engine config dict."""
    return {
        "graph": {
            "type": "custom",
            "n": n_nodes,
            "edges": edges,
            "seed": seed,
        },
        "thresholds": {
            "type": "uniform",
            "deg_low": deg_low,
            "deg_high": deg_high,
            "fail_low": fail_low,
            "fail_high": fail_high,
        },
        "seed": seed,
        "propagation_mode": mode,
    }


def _run_simulation(cfg: dict, status_cb, progress_cb) -> Path:
    """
    Run the cascade engine in-process and return the output directory.
    """
    progress_cb(0.4)
    status_cb("Building adjacency matrix and threshold arrays…")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, theta_deg, theta_fail, in_degree = _build_experiment(cfg)

    progress_cb(0.65)
    mode = cfg.get("propagation_mode", "deterministic")
    status_cb(f"Running {mode} propagation on {A.shape[0]} nodes…")

    if mode == "deterministic":
        _run_deterministic(cfg, A, theta_deg, theta_fail, in_degree, OUTPUT_DIR)
    else:
        _run_stochastic(cfg, A, theta_deg, theta_fail, in_degree, OUTPUT_DIR)

    progress_cb(0.95)
    return OUTPUT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — logo + file upload + column mapping
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="app-header">
          <div class="app-logo">⚡</div>
          <div>
            <div class="app-name">Cascade Engine</div>
            <div class="app-sub">Infrastructure Risk Simulator</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 1: File Ingestion ────────────────────────────────────────────
    st.markdown('<div class="section-title">① Data Ingestion</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        help="Select a CSV file (e.g., flight data, power grid data). "
             "Only headers are read at this stage.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8b949e;word-break:break-all;'
            f'margin-top:0.4rem;">📄 {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )
        # Silently extract headers once per new file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_last_file_key") != file_key:
            st.session_state.df_headers = _read_headers(uploaded_file)
            st.session_state._last_file_key = file_key
            st.session_state.sim_ran = False   # reset results on new file

    st.markdown("---")

    # ── Section 2: Column Mapping ────────────────────────────────────────────
    st.markdown('<div class="section-title">② Column Mapping</div>', unsafe_allow_html=True)

    headers = st.session_state.df_headers
    _disabled = len(headers) == 0

    def _default_selection(hints: list[str]) -> list[str]:
        """Return headers whose names contain any of the hint substrings."""
        if _disabled:
            return []
        found = []
        for hint in hints:
            for h in headers:
                if hint.lower() in h.lower() and h not in found:
                    found.append(h)
                    break
        return found or (headers[:1] if headers else [])

    src_cols = st.multiselect(
        "Source node columns",
        options=headers,
        default=_default_selection(["origin", "source", "state", "from"]),
        disabled=_disabled,
        help="One or more columns composed into a single source-node label (joined with '|'). "
             "Example: STATE + CZ_NAME → 'TX|HARRIS'",
        placeholder="Select one or more columns…",
    )

    tgt_cols = st.multiselect(
        "Target node columns",
        options=headers,
        default=_default_selection(["dest", "target", "event_type", "to", "sink"]),
        disabled=_disabled,
        help="One or more columns composed into a single target-node label (joined with '|'). "
             "Example: EVENT_TYPE → 'Tornado'",
        placeholder="Select one or more columns…",
    )

    fi_cols = st.multiselect(
        "Failure indicator columns",
        options=headers,
        default=_default_selection(["delay", "damage", "cancel", "loss", "fail"]),
        max_selections=2,
        disabled=_disabled,
        help="1–2 columns used to filter rows (rows where any selected column is NA are dropped). "
             "Numeric damage/casualty fields recommended.",
        placeholder="Select 1 or 2 columns…",
    )

    # Live label preview
    if src_cols or tgt_cols:
        src_preview = "|".join(f"<{c}>" for c in src_cols) if src_cols else "?"
        tgt_preview = "|".join(f"<{c}>" for c in tgt_cols) if tgt_cols else "?"
        fi_preview  = " + ".join(fi_cols) if fi_cols else "none"
        st.markdown(
            f"""
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;
                        padding:0.55rem 0.8rem;margin-top:0.3rem;font-family:monospace;
                        font-size:0.75rem;color:#8b949e;line-height:1.6;">
              <span style="color:#79c0ff;">src</span> → {src_preview}<br>
              <span style="color:#79c0ff;">tgt</span> → {tgt_preview}<br>
              <span style="color:#79c0ff;">fi&nbsp;</span> → {fi_preview}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Section 3: Engine Parameters ────────────────────────────────────────
    st.markdown('<div class="section-title">③ Engine Configuration</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        deg_low  = st.number_input("deg_low",  value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                                    help="Lower bound for degradation threshold (uniform draw)")
        fail_low = st.number_input("fail_low", value=0.50, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                                    help="Lower bound for failure threshold (uniform draw)")
    with c2:
        deg_high  = st.number_input("deg_high",  value=0.40, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                                     help="Upper bound for degradation threshold")
        fail_high = st.number_input("fail_high", value=0.80, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                                     help="Upper bound for failure threshold")

    seed = st.number_input("Random Seed", value=42, min_value=0, step=1,
                            help="Integer seed for fully reproducible results")

    prop_mode = st.selectbox(
        "Propagation Mode",
        ["deterministic", "stochastic"],
        help="deterministic = Tier 1 hard-threshold; stochastic = Tier 2 Monte Carlo",
    )

    st.markdown("---")

    # ── Section 4: Run Button ────────────────────────────────────────────────
    st.markdown('<div class="section-title">④ Execution</div>', unsafe_allow_html=True)
    run_btn = st.button(
        "⚡  Run Simulation",
        disabled=uploaded_file is None or not src_cols or not tgt_cols or not fi_cols,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main panel header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding:0.2rem 0 1rem 0;">
      <h1 style="font-size:1.6rem;font-weight:800;color:#e6edf3;margin:0;">
        Cascade Propagation Engine
      </h1>
      <p style="font-size:0.85rem;color:#8b949e;margin:0.3rem 0 0 0;">
        Threshold-Based Infrastructure Risk Simulation Dashboard
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Idle state — show instructions
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.sim_ran and not run_btn:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, body in [
        (c1, "📂", "1. Upload CSV",
         "Use the sidebar to load your network dataset — flight routes, power grid edges, or any source→target CSV."),
        (c2, "🔧", "2. Map & Configure",
         "Map your columns to engine inputs. Tune degradation and failure thresholds to match your domain."),
        (c3, "⚡", "3. Run & Analyse",
         "Hit Run Simulation. The engine builds the graph, propagates cascades, and surfaces the most fragile nodes."),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="section-card" style="text-align:center;min-height:160px;">
                  <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
                  <div style="font-weight:700;color:#e6edf3;margin-bottom:0.4rem;">{title}</div>
                  <div style="font-size:0.82rem;color:#8b949e;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Simulation execution
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.sim_ran = False
    st.session_state.run_error = None

    status_box = st.empty()
    progress_bar = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    def status(msg: str):
        status_box.markdown(
            f'<div class="status-run">🔄 {msg}</div>', unsafe_allow_html=True
        )
        log_lines.append(f"• {msg}")
        log_box.markdown(
            "<div style='font-size:0.75rem;color:#8b949e;font-family:monospace;'>"
            + "<br>".join(log_lines[-6:])
            + "</div>",
            unsafe_allow_html=True,
        )

    def progress(v: float):
        progress_bar.progress(min(v, 1.0))

    try:
        t0 = time.perf_counter()
        progress(0.05)
        status("Parsing CSV and mapping node identifiers…")

        edges, name_to_id, id_to_name, n_nodes = _build_graph_from_csv(
            uploaded_file, src_cols, tgt_cols, fi_cols, status
        )

        if n_nodes < 2:
            raise ValueError("Need at least 2 distinct nodes to build a network.")
        if len(edges) == 0:
            raise ValueError("No valid edges found after filtering. Check your column mapping.")

        progress(0.25)

        # Save node_lookup.json
        node_lookup_path = OUTPUT_DIR / "node_lookup.json"
        node_lookup_payload = {str(k): v for k, v in id_to_name.items()}
        node_lookup_path.write_text(json.dumps(node_lookup_payload, indent=2))
        status(f"Saved node_lookup.json ({n_nodes} nodes).")

        progress(0.30)

        # Build & save config.json
        cfg = _generate_config(
            n_nodes, edges, deg_low, deg_high, fail_low, fail_high, seed, prop_mode
        )
        config_path = OUTPUT_DIR / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        status("Generated config.json.")

        # Store graph stats for display
        st.session_state.graph_stats = {
            "n_nodes": n_nodes,
            "n_edges": len(edges),
        }

        # Run engine
        _run_simulation(cfg, status, progress)

        elapsed = time.perf_counter() - t0

        # Load results
        results_path = OUTPUT_DIR / "fragility_results.csv"
        results_df = pd.read_csv(results_path)
        summary_path = OUTPUT_DIR / "summary.json"
        summary_json = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        st.session_state.results_df = results_df
        st.session_state.summary_json = summary_json
        st.session_state.node_lookup = node_lookup_payload
        st.session_state.output_dir = str(OUTPUT_DIR)
        st.session_state.sim_ran = True
        st.session_state.run_error = None
        st.session_state._elapsed = elapsed

        progress(1.0)
        status_box.markdown(
            f'<div class="status-ok">✅ Simulation complete in {elapsed:.2f}s</div>',
            unsafe_allow_html=True,
        )
        log_box.empty()

    except Exception as exc:
        st.session_state.run_error = str(exc)
        status_box.markdown(
            f'<div class="status-err">❌ {exc}</div>', unsafe_allow_html=True
        )
        progress_bar.empty()
        log_box.empty()
        st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Section 5: Results Dashboard
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.sim_ran:

    results_df: pd.DataFrame = st.session_state.results_df
    summary: dict = st.session_state.summary_json
    id_to_name: dict = st.session_state.node_lookup
    graph_stats: dict = st.session_state.graph_stats

    mode = summary.get("mode", "deterministic")

    # ── Summary metrics row ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Network Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    def _metric(col, label, value, unit=""):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="metric-unit">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _metric(m1, "Total Nodes",  f"{graph_stats.get('n_nodes', '—'):,}", "")
    _metric(m2, "Total Edges",  f"{graph_stats.get('n_edges', '—'):,}", "directed")
    _metric(m3, "Mode",         mode.capitalize(), "propagation")
    elapsed_s = getattr(st.session_state, "_elapsed", 0)
    _metric(m4, "Elapsed",      f"{elapsed_s:.2f}", "seconds")

    st.markdown("---")

    # ── Worst-case cascade card ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Worst-Case Cascade</div>', unsafe_allow_html=True)

    if mode == "deterministic":
        wc = summary.get("worst_case_cascade", {})
        wc_seed_id = wc.get("seed_node", "?")
        wc_name = id_to_name.get(str(wc_seed_id), str(wc_seed_id))
        wc_affected = wc.get("n_affected", "?")
        n_total = summary.get("n_nodes", graph_stats.get("n_nodes", 1))
        wc_frac = wc.get("frac_affected", 0)
        fi_stats = summary.get("fragility_index", {})
        fi_max = fi_stats.get("max", "—")
        fi_mean = round(fi_stats.get("mean", 0), 1)

        wc_c1, wc_c2 = st.columns([1, 2])
        with wc_c1:
            st.markdown(
                f"""
                <div class="worst-card">
                  <div class="worst-title">Max Cascade Size</div>
                  <div class="worst-value">{wc_affected}</div>
                  <div class="worst-sub">nodes affected ({wc_frac*100:.1f}% of network)</div>
                  <div style="margin-top:0.7rem;font-size:0.78rem;color:#8b949e;">
                    Seeded by: <span class="node-badge">{wc_name}</span>
                    &nbsp; (ID {wc_seed_id})
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with wc_c2:
            st.markdown(
                f"""
                <div class="section-card" style="height:100%;">
                  <div class="section-title">Fragility Index Statistics</div>
                  <div class="metric-row">
                    <div class="metric-card"><div class="metric-label">Mean FI</div>
                      <div class="metric-value" style="font-size:1.3rem;">{fi_mean}</div></div>
                    <div class="metric-card"><div class="metric-label">Max FI</div>
                      <div class="metric-value" style="font-size:1.3rem;color:#f0883e;">{fi_max}</div></div>
                    <div class="metric-card"><div class="metric-label">P90 FI</div>
                      <div class="metric-value" style="font-size:1.3rem;">{fi_stats.get('p90','—')}</div></div>
                    <div class="metric-card"><div class="metric-label">Median FI</div>
                      <div class="metric-value" style="font-size:1.3rem;">{fi_stats.get('median','—')}</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:  # stochastic
        mc_mean = round(summary.get("stochastic_mean_cascade_mean", 0), 4)
        mc_std  = round(summary.get("stochastic_mean_cascade_std", 0), 4)
        spearman = round(summary.get("spearman_rho", 0), 4) if summary.get("spearman_rho") else "N/A"
        trials = summary.get("monte_carlo_trials", "?")

        wc_c1, wc_c2 = st.columns(2)
        with wc_c1:
            st.markdown(
                f"""
                <div class="worst-card">
                  <div class="worst-title">Mean Cascade Size (Stochastic)</div>
                  <div class="worst-value">{mc_mean}</div>
                  <div class="worst-sub">fraction of network affected per seed node (avg)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with wc_c2:
            st.markdown(
                f"""
                <div class="section-card">
                  <div class="section-title">Monte Carlo Statistics</div>
                  <div class="metric-row">
                    <div class="metric-card"><div class="metric-label">Trials / Node</div>
                      <div class="metric-value" style="font-size:1.3rem;">{trials}</div></div>
                    <div class="metric-card"><div class="metric-label">Std Dev</div>
                      <div class="metric-value" style="font-size:1.3rem;">{mc_std}</div></div>
                    <div class="metric-card"><div class="metric-label">Spearman ρ</div>
                      <div class="metric-value" style="font-size:1.3rem;">{spearman}</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Top 10 Fragile Nodes table ───────────────────────────────────────────
    st.markdown('<div class="section-title">⑤ Top 10 Most Fragile Nodes</div>', unsafe_allow_html=True)

    sort_col = "fragility_index" if mode == "deterministic" else "stochastic_mean_cascade"
    if sort_col not in results_df.columns:
        sort_col = results_df.columns[1]   # fallback

    top10 = results_df.nlargest(10, sort_col).copy()

    # Cross-reference node names
    top10["node_name"] = top10["node_id"].apply(
        lambda i: id_to_name.get(str(int(i)), str(int(i)))
    )

    # Column display order
    if mode == "deterministic":
        display_cols = {
            "Rank": range(1, len(top10) + 1),
            "Node Name": top10["node_name"].values,
            "Node ID": top10["node_id"].astype(int).values,
            "Fragility Index": top10["fragility_index"].astype(int).values,
            "Degradation θ": top10["theta_deg"].round(4).values,
            "Failure θ": top10["theta_fail"].round(4).values,
            "In-Degree": top10["in_degree"].astype(int).values,
        }
    else:
        display_cols = {
            "Rank": range(1, len(top10) + 1),
            "Node Name": top10["node_name"].values,
            "Node ID": top10["node_id"].astype(int).values,
            "Mean Cascade": top10["stochastic_mean_cascade"].round(4).values,
            "Det. Fragility": top10["det_fragility_index"].astype(int).values,
            "Variance": top10["stochastic_variance"].round(6).values,
            "95% CI Low": top10["ci_95_low"].round(4).values,
            "95% CI High": top10["ci_95_high"].round(4).values,
            "In-Degree": top10["in_degree"].astype(int).values,
        }

    display_df = pd.DataFrame(display_cols)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Node Name": st.column_config.TextColumn("Node Name", width="medium"),
        },
    )

    st.markdown("---")

    # ── Fragility distribution chart ─────────────────────────────────────────
    st.markdown('<div class="section-title">Fragility Distribution</div>', unsafe_allow_html=True)

    chart_col = "fragility_index" if mode == "deterministic" else "stochastic_mean_cascade"
    chart_label = "Fragility Index" if mode == "deterministic" else "Mean Cascade Size"

    hist_vals = results_df[chart_col].dropna().values
    hist_df = pd.DataFrame({chart_label: hist_vals})
    st.bar_chart(hist_df[chart_label].value_counts().sort_index())

    st.markdown("---")

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

    dl_c1, dl_c2, dl_c3 = st.columns(3)

    with dl_c1:
        st.download_button(
            "📥 fragility_results.csv",
            data=(OUTPUT_DIR / "fragility_results.csv").read_bytes(),
            file_name="fragility_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_c2:
        st.download_button(
            "📥 node_lookup.json",
            data=(OUTPUT_DIR / "node_lookup.json").read_bytes(),
            file_name="node_lookup.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_c3:
        st.download_button(
            "📥 config.json",
            data=(OUTPUT_DIR / "config.json").read_bytes(),
            file_name="config.json",
            mime="application/json",
            use_container_width=True,
        )

    # summary.json + top-k if deterministic
    dl_c4, dl_c5, _ = st.columns(3)
    with dl_c4:
        if (OUTPUT_DIR / "summary.json").exists():
            st.download_button(
                "📥 summary.json",
                data=(OUTPUT_DIR / "summary.json").read_bytes(),
                file_name="summary.json",
                mime="application/json",
                use_container_width=True,
            )
    with dl_c5:
        topk_path = OUTPUT_DIR / "topk_cascade_results.csv"
        if topk_path.exists():
            st.download_button(
                "📥 topk_cascade_results.csv",
                data=topk_path.read_bytes(),
                file_name="topk_cascade_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
