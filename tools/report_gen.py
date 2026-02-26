#!/usr/bin/env python3
"""
report_gen.py — Consultancy Wrapper: Deliverable
=================================================
Aggregate cascade_engine output artefacts and visualiser PNGs into a single,
self-contained HTML or Markdown report suitable for board-level delivery.

Report sections
---------------
1. Executive Summary
   High-level risk posture: node count, mean FI, worst-case cascade fraction.

2. Top 5 Fragile Nodes
   Table of the five highest-FI nodes with their thresholds and in-degree.
   Cross-references the Ghost Hub status of each node.

3. Systemic Tipping Points
   If k-sweep data is available, identifies the Chaos Point — the k value
   below which Spearman ρ drops below 0.5 — and reports RMSE at each k.

4. Recommended Hardening Priority
   Actionable prioritisation list derived from FI and in-degree.
   Ghost Hubs (high FI, low in-degree) are flagged as non-obvious risks.

Usage examples
--------------
# HTML report (default)
python3 report_gen.py \\
    --results-dir results_er/ \\
    --visuals-dir visuals/ \\
    --output report.html

# Markdown report
python3 report_gen.py \\
    --results-dir results_er/ \\
    --visuals-dir visuals/ \\
    --output report.md \\
    --format markdown

# With k-sweep and scenario manifest
python3 report_gen.py \\
    --results-dir results_er/ \\
    --visuals-dir visuals/ \\
    --scenario-dir scenario_results/ \\
    --output report.html \\
    --org-name "Acme Infrastructure Ltd." \\
    --analyst "Risk Analytics Team"
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GHOST_HUB_THRESHOLD_PERCENTILE_FI = 60    # FI above this percentile → high FI
GHOST_HUB_THRESHOLD_PERCENTILE_ID = 40    # in-degree below this percentile → low degree
CHAOS_POINT_RHO_THRESHOLD          = 0.5  # Spearman ρ below this → chaotic regime
TOP_N_NODES                        = 5
HARDENING_PRIORITY_N               = 10


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_fragility(results_dir: Path) -> pd.DataFrame:
    """Load fragility_results.csv with schema normalisation.

    Handles both deterministic (``fragility_index``) and stochastic
    (``det_fragility_index``) schemas.

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

    if "det_fragility_index" in df.columns and "fragility_index" not in df.columns:
        df = df.rename(columns={"det_fragility_index": "fragility_index"})

    required = {"node_id", "fragility_index", "in_degree"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"fragility_results.csv missing columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}."
        )

    return df


def load_summary(results_dir: Path) -> dict:
    """Load summary.json, returning empty dict on failure."""
    p = results_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_config_snapshot(results_dir: Path) -> dict:
    """Load config_snapshot.json, returning empty dict on failure."""
    p = results_dir / "config_snapshot.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data.get("config", {})
    except Exception:
        return {}


def load_topk_csv(results_dir: Path) -> pd.DataFrame | None:
    """Load topk_cascade_results.csv if present (deterministic mode only)."""
    p = results_dir / "topk_cascade_results.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_k_sweep_summary(scenario_dir: Path) -> pd.DataFrame | None:
    """Walk a scenario directory to collect k-sweep Spearman/RMSE data."""
    k_sweep_root = scenario_dir / "k_sweep"
    if not k_sweep_root.exists():
        return None

    records = []
    for sub in sorted(k_sweep_root.iterdir()):
        if not sub.is_dir():
            continue
        for cand in [sub / "results" / "summary.json", sub / "summary.json"]:
            if cand.exists():
                try:
                    s = json.loads(cand.read_text())
                    k_str = sub.name.replace("k_", "")
                    k = float(k_str)
                    rho  = s.get("spearman_rho")
                    rmse = s.get("rmse_det_vs_stochastic")
                    # Normalise None (JSON null via _nan_to_none) and float NaN
                    # to 0.0 before appending so low-k points are never dropped.
                    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                        rho = 0.0
                    records.append({"k": k, "spearman_rho": float(rho), "rmse": rmse})
                except Exception:
                    pass
                break

    if not records:
        return None
    return pd.DataFrame(records).sort_values("k").reset_index(drop=True)


def load_scenario_manifest(scenario_dir: Path) -> dict | None:
    """Load scenario_manifest.json if present."""
    p = scenario_dir / "scenario_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def classify_ghost_hubs(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``is_ghost_hub`` boolean column to the fragility DataFrame.

    A Ghost Hub is a node with FI above the 60th percentile AND in-degree
    below the 40th percentile — high risk, low visibility.
    """
    fi_threshold = np.percentile(df["fragility_index"], GHOST_HUB_THRESHOLD_PERCENTILE_FI)
    id_threshold = np.percentile(df["in_degree"],       GHOST_HUB_THRESHOLD_PERCENTILE_ID)
    df = df.copy()
    df["is_ghost_hub"] = (df["fragility_index"] >= fi_threshold) & (df["in_degree"] <= id_threshold)
    return df


def find_chaos_point(df_k: pd.DataFrame) -> float | None:
    """Return the smallest k where Spearman ρ ≥ CHAOS_POINT_RHO_THRESHOLD.

    Returns None if ρ never reaches the threshold.
    """
    above = df_k[df_k["spearman_rho"] >= CHAOS_POINT_RHO_THRESHOLD]
    if above.empty:
        return None
    return float(above["k"].min())


def build_hardening_priority(df: pd.DataFrame, n: int = HARDENING_PRIORITY_N) -> pd.DataFrame:
    """Build a prioritised hardening list.

    Priority score = normalised FI + normalised in-degree.  Ghost hubs are
    boosted by an additional 0.5 to surface non-obvious risks above
    straightforward high-degree/high-FI nodes.
    """
    df = classify_ghost_hubs(df).copy()
    fi_max = max(df["fragility_index"].max(), 1)
    id_max = max(df["in_degree"].max(), 1)

    df["_fi_norm"] = df["fragility_index"] / fi_max
    df["_id_norm"] = df["in_degree"] / id_max
    # Ghost hub boost must exceed the maximum possible id_norm contribution (1.0)
    # so a ghost hub with equal FI always ranks above a high-degree non-ghost hub.
    df["priority_score"] = df["_fi_norm"] + df["_id_norm"] + df["is_ghost_hub"].astype(float) * 1.5

    top = df.nlargest(n, "priority_score")[
        ["node_id", "fragility_index", "in_degree", "is_ghost_hub", "priority_score"]
    ].reset_index(drop=True)
    top.index = top.index + 1  # 1-based rank
    return top


# ---------------------------------------------------------------------------
# Image embedding
# ---------------------------------------------------------------------------


def _embed_image_html(path: Path, alt: str, max_width: str = "100%") -> str:
    """Return an HTML <img> tag with the image base64-embedded."""
    if not path.exists():
        return f'<p class="missing-img"><em>[Image not found: {path.name}]</em></p>'
    suffix = path.suffix.lstrip(".").lower()
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml", "pdf": "application/pdf"}
    mime = mime_map.get(suffix, "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure style="text-align:center;margin:20px 0;">'
        f'<img src="data:{mime};base64,{data}" alt="{alt}" '
        f'style="max-width:{max_width};border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.15);">'
        f'<figcaption style="font-size:0.85em;color:#666;margin-top:6px;">{alt}</figcaption>'
        f'</figure>'
    )


def _embed_image_md(path: Path, alt: str) -> str:
    """Return a Markdown image reference (relative path, not embedded)."""
    if not path.exists():
        return f"*[Image not found: {path.name}]*"
    return f"![{alt}]({path})"


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

HTML_CSS = """
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0;
         background: #f4f6f9; color: #2c3e50; }
  .page { max-width: 960px; margin: 30px auto; background: #fff;
          border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.09); overflow: hidden; }
  .header { background: linear-gradient(135deg,#1a2a4a 0%,#2e4a7a 100%);
             color: #fff; padding: 40px 48px 32px; }
  .header h1 { margin: 0 0 6px; font-size: 1.9em; letter-spacing: 0.5px; }
  .header .subtitle { opacity: 0.8; font-size: 0.95em; margin-top: 4px; }
  .header .meta { margin-top: 16px; font-size: 0.85em; opacity: 0.65; }
  .content { padding: 36px 48px; }
  h2 { color: #1a2a4a; border-bottom: 2px solid #e0e6ef; padding-bottom: 8px;
       margin-top: 40px; font-size: 1.3em; letter-spacing: 0.3px; }
  h3 { color: #2e4a7a; margin-top: 24px; font-size: 1.05em; }
  p  { line-height: 1.7; color: #444; }
  table { width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 0.93em; }
  th { background: #1a2a4a; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  tr:nth-child(even) { background: #f0f4fa; }
  td { padding: 9px 14px; border-bottom: 1px solid #e0e6ef; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px;
           font-size:0.8em; font-weight:600; }
  .badge-ghost { background:#fde8e8; color:#c0392b; }
  .badge-safe  { background:#e8f8ef; color:#27ae60; }
  .badge-chaos { background:#fff3cd; color:#b7770d; }
  .kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 16px; margin: 20px 0; }
  .kv-card { background: #f0f4fa; border-radius: 8px; padding: 16px 18px;
              border-left: 4px solid #2e4a7a; }
  .kv-card .val { font-size: 1.6em; font-weight: 700; color: #1a2a4a; }
  .kv-card .lbl { font-size: 0.82em; color: #666; margin-top: 3px; }
  .alert { background: #fff3cd; border-left: 4px solid #f0a500; padding: 12px 16px;
            border-radius: 4px; margin: 16px 0; font-size: 0.93em; }
  .missing-img { color: #999; font-style: italic; text-align: center; padding: 16px; }
  .footer { background: #f0f4fa; padding: 18px 48px; font-size: 0.82em; color: #888;
             border-top: 1px solid #e0e6ef; }
  @media print { .page { box-shadow: none; } }
</style>
"""


def _kv_cards(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="kv-card"><div class="val">{v}</div><div class="lbl">{k}</div></div>'
        for k, v in items
    )
    return f'<div class="kv-grid">{cards}</div>'


def build_html_report(
    df: pd.DataFrame,
    summary: dict,
    config: dict,
    df_k: pd.DataFrame | None,
    manifest: dict | None,
    visuals_dir: Path | None,
    org_name: str,
    analyst: str,
) -> str:
    """Assemble the complete HTML report string."""
    now = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    n   = int(summary.get("n_nodes", len(df)))
    mode = summary.get("mode", config.get("propagation_mode", "deterministic"))
    fi_col = "fragility_index"

    df = classify_ghost_hubs(df)
    n_ghost = int(df["is_ghost_hub"].sum())
    top5 = df.nlargest(TOP_N_NODES, fi_col).reset_index(drop=True)
    priority = build_hardening_priority(df, HARDENING_PRIORITY_N)

    fi_mean = float(df[fi_col].mean())
    fi_max  = int(df[fi_col].max())
    worst_node = int(df.loc[df[fi_col].idxmax(), "node_id"])
    worst_frac = round(fi_max / max(n, 1), 4)

    # Spearman ρ from summary (stochastic mode)
    spearman_rho = summary.get("spearman_rho")
    rmse_val     = summary.get("rmse_det_vs_stochastic")

    # ── 1. Header ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Infrastructure Risk Audit Report — {org_name}</title>
{HTML_CSS}
</head>
<body>
<div class="page">
<div class="header">
  <h1>Infrastructure Risk Audit Report</h1>
  <div class="subtitle">Cascade Propagation Analysis — Threshold-Based Failure Modelling</div>
  <div class="meta">
    Organisation: <strong>{org_name}</strong> &nbsp;|&nbsp;
    Prepared by: {analyst} &nbsp;|&nbsp;
    Generated: {now}
  </div>
</div>
<div class="content">
"""

    # ── 2. Executive Summary ──────────────────────────────────────────────────
    html += "<h2>1. Executive Summary</h2>"

    risk_level = (
        "CRITICAL" if worst_frac >= 0.4
        else "HIGH"   if worst_frac >= 0.2
        else "MODERATE" if worst_frac >= 0.1
        else "LOW"
    )
    badge_cls = {
        "CRITICAL": "badge-ghost",
        "HIGH":     "badge-ghost",
        "MODERATE": "badge-chaos",
        "LOW":      "badge-safe",
    }[risk_level]

    html += f"""
<p>This report summarises a cascade-propagation risk audit of a <strong>{n}-node</strong>
infrastructure network modelled under the <strong>{mode}</strong> propagation regime.
The analysis computes each node's Fragility Index (FI) — the number of additional nodes
affected when that node alone is forced into a failed state.  Elevated FI values indicate
systemic amplifiers whose failure cascades broadly through the network.</p>

<p>Overall risk posture: <span class="badge {badge_cls}">{risk_level}</span></p>
"""

    html += _kv_cards([
        ("Nodes modelled", str(n)),
        ("Mean Fragility Index", f"{fi_mean:.2f}"),
        ("Max Fragility Index", str(fi_max)),
        ("Worst-case cascade", f"{worst_frac*100:.1f}% of network"),
        ("Ghost Hubs detected", str(n_ghost)),
        ("Propagation mode", mode.capitalize()),
    ])

    if spearman_rho is not None:
        html += f"""<p><strong>Stochastic validation:</strong> At the configured logistic
steepness k={summary.get('stochastic_k', '?')}, the stochastic Monte Carlo results
correlate with the deterministic fragility index at
Spearman&nbsp;ρ&nbsp;=&nbsp;<strong>{spearman_rho:.4f}</strong>
(RMSE&nbsp;=&nbsp;{rmse_val:.4f if rmse_val is not None else 'N/A'}).
{"A ρ above 0.8 indicates strong rank agreement — stochastic results are reliable." if spearman_rho >= 0.8 else "A ρ below 0.8 suggests high uncertainty; consider increasing k or trials."}</p>"""

    # Risk Heatmap
    if visuals_dir:
        for stem in ["risk_heatmap.png", "risk_heatmap.pdf", "risk_heatmap.svg"]:
            img = visuals_dir / stem
            if img.exists():
                html += _embed_image_html(img, "Risk Heatmap — Network coloured by Fragility Index")
                break

    # ── 3. Top 5 Fragile Nodes ────────────────────────────────────────────────
    html += "<h2>2. Top 5 Fragile Nodes</h2>"
    html += """<p>The table below lists the five nodes with the highest Fragility Index.
Seeding any of these nodes as the initial failure produces the largest downstream cascades.
Ghost Hub status indicates a node that is disproportionately dangerous relative to its
apparent connectivity.</p>"""

    html += "<table><tr><th>Rank</th><th>Node ID</th><th>Fragility Index</th>"
    html += "<th>In-Degree</th>"
    if "theta_fail" in df.columns:
        html += "<th>θ_fail</th>"
    html += "<th>Ghost Hub?</th><th>Cascade %</th></tr>"

    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        ghost_badge = (
            '<span class="badge badge-ghost">YES</span>'
            if row.get("is_ghost_hub") else
            '<span class="badge badge-safe">NO</span>'
        )
        fi = int(row[fi_col])
        theta_cell = (
            f"<td>{row['theta_fail']:.3f}</td>" if "theta_fail" in df.columns else ""
        )
        html += (
            f"<tr><td>{rank}</td><td><strong>{int(row['node_id'])}</strong></td>"
            f"<td>{fi}</td><td>{int(row['in_degree'])}</td>"
            f"{theta_cell}<td>{ghost_badge}</td>"
            f"<td>{fi / max(n, 1) * 100:.1f}%</td></tr>"
        )
    html += "</table>"

    # Ghost Hub Plot
    if visuals_dir:
        for stem in ["ghost_hub_plot.png", "ghost_hub_plot.pdf", "ghost_hub_plot.svg"]:
            img = visuals_dir / stem
            if img.exists():
                html += _embed_image_html(img, "Ghost Hub Analysis — In-Degree vs. Fragility Index")
                break

    # ── 4. Systemic Tipping Points ────────────────────────────────────────────
    html += "<h2>3. Systemic Tipping Points</h2>"

    if df_k is not None and not df_k.empty:
        chaos_k = find_chaos_point(df_k)
        html += f"""<p>The k-sweep analysis runs the stochastic engine across a range of
logistic steepness values (k) to map the transition from chaotic to convergent behaviour.
Below the <em>Chaos Point</em> — the minimum k at which Spearman&nbsp;ρ&nbsp;≥&nbsp;{CHAOS_POINT_RHO_THRESHOLD} —
Monte Carlo rankings diverge substantially from the deterministic baseline.</p>"""

        if chaos_k is not None:
            html += f"""
<div class="alert">
  <strong>Chaos Point identified at k = {chaos_k:.1f}.</strong>
  Stochastic rankings become reliable above this threshold.
  For operational risk planning, use k&nbsp;≥&nbsp;{chaos_k:.0f} to ensure
  Monte Carlo results faithfully reflect the deterministic fragility ranking.
</div>"""
        else:
            html += f"""<div class="alert">Spearman&nbsp;ρ did not exceed {CHAOS_POINT_RHO_THRESHOLD}
across the tested k range — consider extending the sweep to higher k values.</div>"""

        html += "<table><tr><th>k</th><th>Spearman ρ</th><th>RMSE</th><th>Regime</th></tr>"
        for _, row in df_k.iterrows():
            rho  = row["spearman_rho"]
            rmse = row.get("rmse")
            regime = (
                "Convergent" if rho >= 0.8
                else "Transitional" if rho >= CHAOS_POINT_RHO_THRESHOLD
                else "Chaotic"
            )
            regime_badge = (
                '<span class="badge badge-safe">Convergent</span>'   if regime == "Convergent" else
                '<span class="badge badge-chaos">Transitional</span>' if regime == "Transitional" else
                '<span class="badge badge-ghost">Chaotic</span>'
            )
            rmse_s = f"{rmse:.4f}" if rmse is not None and not np.isnan(rmse) else "N/A"
            html += f"<tr><td>{row['k']:.1f}</td><td>{rho:.4f}</td><td>{rmse_s}</td><td>{regime_badge}</td></tr>"
        html += "</table>"

        # Convergence Curve image
        if visuals_dir:
            for stem in ["convergence_curve.png", "convergence_curve.pdf"]:
                img = visuals_dir / stem
                if img.exists():
                    html += _embed_image_html(img, "Convergence Curve — Spearman ρ and RMSE vs. k")
                    break
    else:
        html += """<p>No k-sweep data was provided. To generate systemic tipping-point
analysis, run <code>scenario_manager.py</code> with the <code>sweep</code> scenario
and supply the output directory via <code>--scenario-dir</code>.</p>"""

    # ── 5. Recommended Hardening Priority ────────────────────────────────────
    html += "<h2>4. Recommended Hardening Priority</h2>"
    html += f"""<p>The following prioritisation ranks nodes by a composite score that
combines normalised Fragility Index, in-degree, and a Ghost Hub boost (+0.5 for nodes
with high FI but low connectivity). <strong>Ghost Hubs should be addressed first</strong>
— they represent non-obvious single points of failure that may be overlooked in
degree-based triage alone.</p>"""

    html += """<table><tr><th>Priority</th><th>Node ID</th><th>Fragility Index</th>
<th>In-Degree</th><th>Ghost Hub?</th><th>Priority Score</th><th>Recommended Action</th></tr>"""

    for rank, (_, row) in enumerate(priority.iterrows(), 1):
        ghost = bool(row["is_ghost_hub"])
        ghost_badge = (
            '<span class="badge badge-ghost">YES ⚠</span>' if ghost else
            '<span class="badge badge-safe">NO</span>'
        )
        fi  = int(row["fragility_index"])
        indeg = int(row["in_degree"])
        score = float(row["priority_score"])

        if ghost:
            action = "Isolate + threshold hardening (Ghost Hub — hidden risk)"
        elif fi >= fi_mean * 2:
            action = "Threshold hardening + redundancy (High FI)"
        elif indeg >= np.percentile(df["in_degree"], 80):
            action = "Dependency reduction (high connectivity hub)"
        else:
            action = "Standard monitoring + periodic review"

        html += (
            f"<tr><td><strong>{rank}</strong></td><td>{int(row['node_id'])}</td>"
            f"<td>{fi}</td><td>{indeg}</td><td>{ghost_badge}</td>"
            f"<td>{score:.3f}</td><td>{action}</td></tr>"
        )
    html += "</table>"

    # Scenario manifest summary if available
    if manifest:
        total  = manifest.get("total_scenarios", 0)
        ok     = manifest.get("successful", 0)
        html += f"""<h3>Scenario Analysis Summary</h3>
<p>{total} What-If scenarios were executed ({ok} successful, {total - ok} failed).
See the individual scenario output directories for detailed per-scenario fragility results.</p>"""

    # ── Footer ────────────────────────────────────────────────────────────────
    html += f"""
</div><!-- /content -->
<div class="footer">
  Generated by <strong>report_gen.py</strong> — Cascade Propagation Engine Consultancy Wrapper &nbsp;|&nbsp;
  {now} &nbsp;|&nbsp; Analyst: {analyst} &nbsp;|&nbsp; Organisation: {org_name}
  <br>
  <em>Confidential — For internal risk management use only.
  Results are model-dependent; complement with domain-expert review before operational decisions.</em>
</div>
</div><!-- /page -->
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def build_markdown_report(
    df: pd.DataFrame,
    summary: dict,
    config: dict,
    df_k: pd.DataFrame | None,
    manifest: dict | None,
    visuals_dir: Path | None,
    org_name: str,
    analyst: str,
) -> str:
    now  = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    n    = int(summary.get("n_nodes", len(df)))
    mode = summary.get("mode", config.get("propagation_mode", "deterministic"))
    fi_col = "fragility_index"

    df = classify_ghost_hubs(df)
    n_ghost = int(df["is_ghost_hub"].sum())
    top5    = df.nlargest(TOP_N_NODES, fi_col).reset_index(drop=True)
    priority= build_hardening_priority(df, HARDENING_PRIORITY_N)

    fi_mean = float(df[fi_col].mean())
    fi_max  = int(df[fi_col].max())
    worst_node = int(df.loc[df[fi_col].idxmax(), "node_id"])
    worst_frac = round(fi_max / max(n, 1) * 100, 1)

    spearman_rho = summary.get("spearman_rho")
    rmse_val     = summary.get("rmse_det_vs_stochastic")

    lines: list[str] = []

    def h(level: int, text: str): lines.append(f"\n{'#' * level} {text}\n")
    def p(text: str):              lines.append(f"{text}\n")
    def rule():                    lines.append("\n---\n")

    lines.append(f"# Infrastructure Risk Audit Report\n")
    lines.append(f"**Organisation:** {org_name}  \n**Analyst:** {analyst}  \n**Generated:** {now}\n")
    rule()

    # 1. Executive Summary
    h(2, "1. Executive Summary")
    p(
        f"This report covers a cascade-propagation risk audit of a **{n}-node** "
        f"infrastructure network under the **{mode}** propagation regime."
    )

    risk = (
        "🔴 CRITICAL" if worst_frac / 100 >= 0.4
        else "🟠 HIGH"    if worst_frac / 100 >= 0.2
        else "🟡 MODERATE" if worst_frac / 100 >= 0.1
        else "🟢 LOW"
    )

    p(f"**Overall Risk Posture:** {risk}")

    lines.append(f"""
| Metric | Value |
|--------|-------|
| Nodes modelled | {n} |
| Mean Fragility Index | {fi_mean:.2f} |
| Max Fragility Index | {fi_max} (Node {worst_node}) |
| Worst-case cascade | {worst_frac}% of network |
| Ghost Hubs detected | {n_ghost} |
| Propagation mode | {mode.capitalize()} |
""")

    if spearman_rho is not None:
        p(
            f"**Stochastic validation:** Spearman ρ = **{spearman_rho:.4f}**, "
            f"RMSE = {f'{rmse_val:.4f}' if rmse_val else 'N/A'}."
        )

    if visuals_dir:
        for stem in ["risk_heatmap.png", "risk_heatmap.pdf"]:
            img = visuals_dir / stem
            if img.exists():
                lines.append(_embed_image_md(img, "Risk Heatmap — Network coloured by Fragility Index"))
                lines.append("\n")
                break

    rule()

    # 2. Top 5 Fragile Nodes
    h(2, "2. Top 5 Fragile Nodes")

    hdr = "| Rank | Node ID | Fragility Index | In-Degree |"
    sep = "|------|---------|-----------------|-----------|"
    if "theta_fail" in df.columns:
        hdr += " θ_fail |"; sep += "--------|"
    hdr += " Ghost Hub? | Cascade % |"
    sep += "-----------|-----------|"
    lines.append(hdr + "\n" + sep)

    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        fi = int(row[fi_col])
        gh = "⚠ YES" if row.get("is_ghost_hub") else "NO"
        tf_cell = f" {row['theta_fail']:.3f} |" if "theta_fail" in df.columns else ""
        lines.append(
            f"| {rank} | **{int(row['node_id'])}** | {fi} | {int(row['in_degree'])} |"
            f"{tf_cell} {gh} | {fi / max(n, 1) * 100:.1f}% |"
        )
    lines.append("")

    if visuals_dir:
        for stem in ["ghost_hub_plot.png", "ghost_hub_plot.pdf"]:
            img = visuals_dir / stem
            if img.exists():
                lines.append(_embed_image_md(img, "Ghost Hub Analysis — In-Degree vs. Fragility Index"))
                lines.append("\n")
                break

    rule()

    # 3. Systemic Tipping Points
    h(2, "3. Systemic Tipping Points")
    if df_k is not None and not df_k.empty:
        chaos_k = find_chaos_point(df_k)
        p(f"Chaos Point (ρ ≥ {CHAOS_POINT_RHO_THRESHOLD}): **k = {chaos_k:.1f}**" if chaos_k else "Chaos Point not reached in tested k range.")

        lines.append("| k | Spearman ρ | RMSE | Regime |")
        lines.append("|---|-----------|------|--------|")
        for _, row in df_k.iterrows():
            rho   = row["spearman_rho"]
            rmse  = row.get("rmse")
            regime = "Convergent" if rho >= 0.8 else "Transitional" if rho >= CHAOS_POINT_RHO_THRESHOLD else "Chaotic"
            rmse_s = f"{rmse:.4f}" if rmse is not None and not np.isnan(rmse) else "N/A"
            lines.append(f"| {row['k']:.1f} | {rho:.4f} | {rmse_s} | {regime} |")
        lines.append("")

        if visuals_dir:
            for stem in ["convergence_curve.png", "convergence_curve.pdf"]:
                img = visuals_dir / stem
                if img.exists():
                    lines.append(_embed_image_md(img, "Convergence Curve"))
                    lines.append("\n")
                    break
    else:
        p("No k-sweep data available. Run `scenario_manager.py` with the `sweep` scenario.")

    rule()

    # 4. Recommended Hardening Priority
    h(2, "4. Recommended Hardening Priority")

    lines.append("| Priority | Node ID | FI | In-Degree | Ghost Hub | Score | Recommended Action |")
    lines.append("|----------|---------|----|-----------|-----------|----|-------------------|")

    for rank, (_, row) in enumerate(priority.iterrows(), 1):
        ghost = bool(row["is_ghost_hub"])
        gh_s  = "⚠ YES" if ghost else "NO"
        fi    = int(row["fragility_index"])
        indeg = int(row["in_degree"])
        score = float(row["priority_score"])
        if ghost:
            action = "Isolate + threshold hardening"
        elif fi >= fi_mean * 2:
            action = "Threshold hardening + redundancy"
        elif indeg >= np.percentile(df["in_degree"], 80):
            action = "Dependency reduction"
        else:
            action = "Standard monitoring"
        lines.append(f"| **{rank}** | {int(row['node_id'])} | {fi} | {indeg} | {gh_s} | {score:.3f} | {action} |")

    lines.append("")
    rule()
    lines.append(
        f"*Generated by report_gen.py — Cascade Propagation Engine Consultancy Wrapper.  \n"
        f"Confidential — complement model results with domain-expert review.*\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate cascade_engine output artefacts into a single risk audit report.\n\n"
            "Produces an HTML (default) or Markdown report with:\n"
            "  1. Executive Summary\n"
            "  2. Top 5 Fragile Nodes\n"
            "  3. Systemic Tipping Points\n"
            "  4. Recommended Hardening Priority"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--results-dir", "-r",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to a cascade_engine output directory.",
    )
    p.add_argument(
        "--visuals-dir", "-v",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing visualiser PNG/PDF outputs (optional).",
    )
    p.add_argument(
        "--scenario-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root scenario_manager output directory (for k-sweep and manifest data).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("report.html"),
        metavar="PATH",
        help="Output report file path (default: report.html).",
    )
    p.add_argument(
        "--format",
        choices=["html", "markdown"],
        default="html",
        help="Output format: html (default) or markdown.",
    )
    p.add_argument(
        "--org-name",
        dest="org_name",
        default="Infrastructure Client",
        metavar="NAME",
        help="Organisation name for the report header (default: 'Infrastructure Client').",
    )
    p.add_argument(
        "--analyst",
        default="Risk Analytics Team",
        metavar="NAME",
        help="Analyst name/team for the report (default: 'Risk Analytics Team').",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load required data ────────────────────────────────────────────────────
    try:
        df = load_fragility(args.results_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = load_summary(args.results_dir)
    config  = load_config_snapshot(args.results_dir)

    print(f"  [report_gen] Loaded {len(df)} nodes from: {args.results_dir}")

    # ── Optional k-sweep data ─────────────────────────────────────────────────
    df_k: pd.DataFrame | None = None
    if args.scenario_dir:
        df_k = load_k_sweep_summary(args.scenario_dir)
        if df_k is not None:
            print(f"  [report_gen] k-sweep data: {len(df_k)} k-values loaded from {args.scenario_dir}")
        else:
            print(f"  [report_gen] No k-sweep data found in {args.scenario_dir}")

    # ── Optional scenario manifest ────────────────────────────────────────────
    manifest: dict | None = None
    if args.scenario_dir:
        manifest = load_scenario_manifest(args.scenario_dir)

    # ── Build report ──────────────────────────────────────────────────────────
    fmt = args.format
    if fmt == "html":
        report_text = build_html_report(
            df, summary, config, df_k, manifest,
            args.visuals_dir, args.org_name, args.analyst,
        )
    else:
        report_text = build_markdown_report(
            df, summary, config, df_k, manifest,
            args.visuals_dir, args.org_name, args.analyst,
        )

    # ── Write output ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")

    sep = "─" * 54
    print(f"\n{sep}")
    print("  report_gen — Report Complete")
    print(sep)
    print(f"  Format   : {fmt}")
    print(f"  Output   : {args.output.resolve()}")
    print(f"  Nodes    : {len(df)}")
    fi_max = int(df["fragility_index"].max()) if "fragility_index" in df.columns else "?"
    print(f"  Max FI   : {fi_max}")
    if df_k is not None:
        print(f"  k-values : {len(df_k)}")
    print(sep)


if __name__ == "__main__":
    main()
