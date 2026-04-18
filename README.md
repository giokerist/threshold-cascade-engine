# Cascade Propagation Engine

A vectorised, two-tier threshold-based cascade propagation engine for directed graphs — built for single-user freelance risk consulting.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Setup](#2-setup)
3. [Core Concepts](#3-core-concepts)
4. [End-to-End Consulting Workflow](#4-end-to-end-consulting-workflow)
   - [Step 1 — Collect Client Data](#step-1--collect-client-data)
   - [Step 2 — Ingest Data (`data_adapter.py`)](#step-2--ingest-data-data_adapterpy)
   - [Step 3 — Run the Engine](#step-3--run-the-engine)
   - [Step 4 — Visualise Results](#step-4--visualise-results-visualizerpy)
   - [Step 5 — Generate the Client Report](#step-5--generate-the-client-report-report_genpy)
   - [Step 6 — What-If Scenario Analysis](#step-6--what-if-scenario-analysis-scenario_managerpy)
5. [Input Data Format](#5-input-data-format)
6. [Configuration Reference](#6-configuration-reference)
7. [Engine Output Files](#7-engine-output-files)
8. [Tier 1 — Deterministic Engine](#8-tier-1--deterministic-engine)
9. [Tier 2 — Stochastic Engine](#9-tier-2--stochastic-engine)
10. [Consulting Tools Reference](#10-consulting-tools-reference)
11. [Running Tests](#11-running-tests)
12. [Design Decisions and Constraints](#12-design-decisions-and-constraints)

---

## 1. Repository Layout

```
threshold-cascade-engine/
│
├── cascade_engine/               # Core engine package
│   ├── config.py                 # Config loading, validation, threshold generation
│   ├── graph.py                  # Graph generators (ER, BA, WS, custom)
│   ├── propagation.py            # Tier 1: deterministic synchronous propagation
│   ├── stochastic_propagation.py # Tier 2: logistic-probability stochastic propagation
│   ├── monte_carlo.py            # Monte Carlo harness (SeedSequence-based)
│   ├── sensitivity.py            # Threshold sensitivity analysis
│   ├── metrics.py                # Fragility Index, RMSE, MAPE, Spearman ρ, CI
│   ├── utils.py                  # Shared RNG helpers and CI utility
│   ├── runner.py                 # CLI entry point (python -m cascade_engine.runner)
│   ├── requirements.txt          # Core engine dependencies
│   └── tests/                   # Pytest test suite (Tier 1 + Tier 2)
│       ├── test_propagation.py
│       └── test_tier2.py
│
├── tools/                        # Consulting workflow utilities
│   ├── data_adapter.py           # CSV / Excel → engine JSON config
│   ├── visualizer.py             # Risk Heatmap, Ghost Hub plot, Convergence Curve
│   ├── report_gen.py             # HTML / Markdown risk audit report generator
│   ├── scenario_manager.py       # Batch What-If scenario runner
│   ├── requirements.txt          # Extra tool dependencies (pandas, seaborn, openpyxl)
│   └── tests/                   # Tool-layer tests
│       └── test_data_adapter.py
│
├── cascade_engine/
│   ├── config_erdos_renyi.json   # Example: deterministic ER graph config
│   ├── config_barabasi_albert.json
│   ├── config_watts_strogatz.json
│   └── config_stochastic_er.json # Example: stochastic run config
│
├── pyproject.toml
└── README.md
```

---

## 2. Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd threshold-cascade-engine

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install core engine dependencies
pip install -r cascade_engine/requirements.txt

# Install consulting tool dependencies (includes openpyxl for Excel)
pip install -r tools/requirements.txt

# Install package in editable mode (enables python -m cascade_engine.runner)
pip install -e .
```

> **Python requirement:** 3.10 or later.

---

## 3. Core Concepts

### States

Every node is in one of three states at each simulation step:

| State | Integer | Meaning |
|-------|---------|---------|
| Operational | `0` | Functioning normally |
| Degraded | `1` | Reduced capacity; contributes to neighbour pressure |
| Failed | `2` | Down; counted as pressure on downstream nodes |

### Threshold Model

Each node `i` has two thresholds drawn from a configurable distribution:

- **θ_deg** — fraction of failed in-neighbours required to degrade node `i`
- **θ_fail** — fraction of failed in-neighbours required to fail node `i`

At each step, the **failure fraction** `F_i` for node `i` is:

```
F_i = (number of FAILED in-neighbours of i) / (in-degree of i)
```

Transitions are **monotone** (a node never recovers):

```
Operational → Degraded  if F_i ≥ θ_deg
Degraded    → Failed    if F_i ≥ θ_fail
```

### Fragility Index (FI)

The FI of a node is the total number of nodes (including itself) affected when it alone is seeded as the initial failure. A high FI means seeding that node causes a large cascade.

### Ghost Hub

A node with **high FI but low in-degree** — dangerous but not obviously so from a connectivity perspective. Ghost Hubs are non-obvious single points of failure that degree-based triage misses.

---

## 4. End-to-End Consulting Workflow

This is the full pipeline from raw client data to a board-ready deliverable.

### Step 1 — Collect Client Data

Obtain from the client a **dependency or connection table** describing how components/systems/entities are linked. Every row is a directed dependency: `source` depends on (or can fail into) `target`.

Accepted formats:
- **CSV** — any commons-separated or tab-separated text file
- **Excel** (`.xlsx` / `.xls`) — uses the first sheet by default; use `--sheet-name` for multi-sheet workbooks

Minimum required columns:

| Column | Required | Description |
|--------|----------|-------------|
| `source` | ✅ | Origin node ID (integer or string label) |
| `target` | ✅ | Destination node ID |
| `weight` | optional | Edge weight (stored in config metadata; not used by engine) |

Column names are case-insensitive and whitespace-stripped automatically.

### Step 2 — Ingest Data (`data_adapter.py`)

Convert the client file into an engine-ready JSON configuration:

```bash
# From CSV
python3 tools/data_adapter.py client_edges.csv \
    --output configs/client_config.json

# From Excel (first sheet)
python3 tools/data_adapter.py client_data.xlsx \
    --output configs/client_config.json

# From Excel, specific sheet
python3 tools/data_adapter.py client_data.xlsx \
    --sheet-name "Dependencies" \
    --output configs/client_config.json

# Stochastic mode with custom thresholds
python3 tools/data_adapter.py client_edges.csv \
    --output configs/client_stoch.json \
    --mode stochastic \
    --threshold-type uniform \
    --fail-low 0.5 --fail-high 0.85 \
    --stochastic-k 10.0 \
    --monte-carlo-trials 50
```

The adapter will:
- Auto-detect CSV vs Excel by file extension
- Remap node labels (any string or integer) to a compact `0`–`N-1` integer space
- Remove self-loops and collapse duplicate edges
- Validate all required columns and emit clear error messages

**Example output summary:**

```
────────────────────────────────────────────────────────
  data_adapter — Ingestion Summary
────────────────────────────────────────────────────────
  Source file      : client_data.xlsx  [EXCEL]
  Output config    : configs/client_config.json
  Raw rows read    : 142
  Nodes detected   : 58
  Directed edges   : 139
  Self-loops dropped        : 2
  Duplicate edges collapsed : 1
  Node ID remapping : YES — labels mapped to 0–57
  Weight column    : PRESENT (stored in metadata)
────────────────────────────────────────────────────────
```

### Step 3 — Run the Engine

```bash
# Deterministic run
python3 -m cascade_engine.runner configs/client_config.json \
    --output-dir results/client/

# Stochastic run (if mode=stochastic in config)
python3 -m cascade_engine.runner configs/client_stoch.json \
    --output-dir results/client_stoch/
```

Output goes to `results/client/`, containing:
- `fragility_results.csv` — FI and metadata per node
- `summary.json` — aggregate statistics
- `config_snapshot.json` — copy of run config (for reproducibility)
- `topk_cascade_results.csv` — deterministic only; top-k worst cascades

### Step 4 — Visualise Results (`visualizer.py`)

```bash
# Generate Risk Heatmap + Ghost Hub plot
python3 tools/visualizer.py \
    --results-dir results/client/ \
    --output-dir visuals/

# Add Convergence Curve from a k-sweep
python3 tools/visualizer.py \
    --results-dir results/client/ \
    --output-dir visuals/ \
    --k-sweep-dir scenario_results/k_sweep/

# PDF output at 200 DPI
python3 tools/visualizer.py \
    --results-dir results/client/ \
    --output-dir visuals/ \
    --fmt pdf --dpi 200
```

Produces:
| File | Description |
|------|-------------|
| `risk_heatmap.png` | Network graph coloured and sized by Fragility Index |
| `ghost_hub_plot.png` | In-Degree vs FI scatter; Ghost Hub zone shaded |
| `convergence_curve.png` | Spearman ρ and RMSE vs k (if `--k-sweep-dir` supplied) |

### Step 5 — Generate the Client Report (`report_gen.py`)

Assembles a self-contained, board-ready HTML or Markdown report:

```bash
# HTML report (default) — embeds visuals inline as base64
python3 tools/report_gen.py \
    --results-dir results/client/ \
    --visuals-dir visuals/ \
    --output report_client.html \
    --org-name "Acme Infrastructure Ltd." \
    --analyst "Your Name"

# With scenario analysis included
python3 tools/report_gen.py \
    --results-dir results/client/ \
    --visuals-dir visuals/ \
    --scenario-dir scenario_results/ \
    --output report_client.html \
    --org-name "Acme Infrastructure Ltd."

# Markdown format (for Obsidian, Notion, etc.)
python3 tools/report_gen.py \
    --results-dir results/client/ \
    --output report_client.md \
    --format markdown
```

Report sections:
1. **Executive Summary** — risk posture badge, KV cards (node count, mean/max FI, ghost hubs, cascade %)
2. **Top 5 Fragile Nodes** — table with FI, in-degree, θ_fail, Ghost Hub flag, cascade percentage
3. **Systemic Tipping Points** — Chaos Point identification from k-sweep data
4. **Recommended Hardening Priority** — actionable ranked list with Ghost Hub boost and suggested interventions

### Step 6 — What-If Scenario Analysis (`scenario_manager.py`)

Run batched What-If scenarios on top of the base config:

```bash
# All three scenario types
python3 tools/scenario_manager.py configs/client_config.json \
    --output-dir scenario_results/ \
    --scenarios hardening regional sweep \
    --hardening-deltas 0.1 0.2 0.3 \
    --top-k 10 \
    --vulnerable-nodes 5 12 37 \
    --vulnerability-delta 0.2 \
    --k-values 2 5 10 20 40 80

# Hardening only
python3 tools/scenario_manager.py configs/client_config.json \
    --output-dir scenario_results/ \
    --scenarios hardening \
    --hardening-deltas 0.15 0.30 \
    --top-k 5

# k-sweep uncertainty analysis only
python3 tools/scenario_manager.py configs/client_config.json \
    --output-dir scenario_results/ \
    --scenarios sweep \
    --k-values 2 5 10 20 40 80 \
    --monte-carlo-trials 50
```

**Scenario types:**

| Scenario | Description |
|----------|-------------|
| `hardening` | Raises `fail_low`/`fail_high` by `--hardening-deltas` magnitude to simulate engineering interventions |
| `regional` | Uses `--vulnerable-nodes` as initial failure seeds with reduced global thresholds (simulates localised shock) |
| `sweep` | Runs stochastic engine across `--k-values` range to find the Chaos Point where Spearman ρ → 0 |

Results are written to isolated sub-folders under `--output-dir`, and a `scenario_manifest.json` index is written to the root.

---

## 5. Input Data Format

### Minimal CSV

```csv
source,target
A,B
B,C
C,D
D,A
A,C
```

### With Weights (stored in metadata, not used by engine)

```csv
source,target,weight
Server_1,DB_Primary,1.0
DB_Primary,App_Cluster,0.8
App_Cluster,Load_Balancer,0.9
```

### Numeric IDs

```csv
source,target
0,1
1,2
2,3
3,0
```

### Excel

Any `.xlsx` or `.xls` file with the same column structure. The first sheet is used by default. Use `--sheet-name "SheetName"` or `--sheet-name 1` (0-based index) for multi-sheet workbooks.

### Node ID Rules

- May be **any integer** (including non-contiguous: `10, 20, 50`) or **any string label** (`"Server_A"`, `"Region_3"`)
- Non-compact or string IDs are **automatically remapped** to `0`–`N-1` (sorted alphabetically for strings)
- The mapping is stored in the output config's `_adapter_meta.node_mapping_sample` field for traceability
- Self-loops (`source == target`) and duplicate edges are silently removed

---

## 6. Configuration Reference

Engine configs are JSON files. `data_adapter.py` generates these automatically. Example structure:

```json
{
  "graph": {
    "type": "custom",
    "n": 58,
    "edges": [[0, 1], [1, 2], ...],
    "seed": 42
  },
  "thresholds": {
    "type": "uniform",
    "deg_low": 0.1,
    "deg_high": 0.4,
    "fail_low": 0.5,
    "fail_high": 0.9
  },
  "seed": 42,
  "propagation_mode": "deterministic",
  "sensitivity_analysis": false,
  "_adapter_meta": { ... }
}
```

### Graph types

| `type` | Additional fields | Description |
|--------|-------------------|-------------|
| `custom` | `n`, `edges` | Client graph from data_adapter |
| `erdos_renyi` | `n`, `p` | Erdős–Rényi random graph |
| `barabasi_albert` | `n`, `m` | Barabási–Albert preferential attachment |
| `watts_strogatz` | `n`, `k`, `p` | Watts–Strogatz small-world |

### Threshold distributions

| `type` | Fields | Notes |
|--------|--------|-------|
| `uniform` | `deg_low`, `deg_high`, `fail_low`, `fail_high` | Must satisfy `deg_low ≤ deg_high`, `fail_low ≤ fail_high` |
| `normal` | `deg_mean`, `deg_std`, `fail_mean`, `fail_std` | `std > 0` required; values clipped to `[0,1]` |

### Stochastic-mode extra fields

```json
{
  "propagation_mode": "stochastic",
  "stochastic_k": 10.0,
  "monte_carlo_trials": 50
}
```

`stochastic_k` controls the logistic steepness. Higher `k` → closer to deterministic step function. Use the k-sweep scenario to find a safe operational value.

### Sensitivity analysis

```json
{
  "sensitivity_analysis": true,
  "sensitivity_config": {
    "perturbation_values": [-0.2, -0.1, 0.0, 0.1, 0.2],
    "max_seed_nodes": 20,
    "stochastic_trials": 20
  }
}
```

---

## 7. Engine Output Files

All files written to `--output-dir`:

| File | Mode | Description |
|------|------|-------------|
| `fragility_results.csv` | both | Per-node FI, in-degree, thresholds |
| `summary.json` | both | Aggregate stats; Spearman ρ / RMSE (stochastic) |
| `config_snapshot.json` | both | Full run config snapshot for reproducibility |
| `topk_cascade_results.csv` | deterministic | Top-k worst-case cascade detail |

### `fragility_results.csv` schema

**Deterministic:**
```
node_id, fragility_index, in_degree, theta_deg, theta_fail
```

**Stochastic:**
```
node_id, det_fragility_index, stochastic_mean_cascade, stochastic_std_cascade,
stochastic_ci_low, stochastic_ci_high, in_degree, theta_deg, theta_fail
```

---

## 8. Tier 1 — Deterministic Engine

The deterministic engine implements the **synchronous threshold model**:

1. Seed one node as `FAILED`
2. At each step, compute `F_i` for every node simultaneously using the *current* state
3. Apply threshold rules atomically; states only increase
4. Halt when no state changes (stable) or `max_steps` is reached

**Convergence guarantee:** A monotone system on a finite graph always converges. With N nodes, convergence occurs in at most N steps.

**Key invariant:** `A[j, i] = 1` encodes edge `j → i` (j influences i). In-degree of node `i` is `A[:, i].sum()`.

Run directly:

```bash
python3 -m cascade_engine.runner cascade_engine/config_erdos_renyi.json \
    --output-dir results/er/
```

---

## 9. Tier 2 — Stochastic Engine

Replaces the hard step function with a **logistic (sigmoid) failure probability**:

```
P(node i fails | F_i, θ_fail_i) = σ(k · (F_i − θ_fail_i))
```

where `σ(x) = 1 / (1 + exp(−x))` and `k` is the steepness parameter.

As `k → ∞`, the logistic approximates the deterministic step. As `k → 0`, outcomes become increasingly random.

**Monte Carlo:** The engine runs `monte_carlo_trials` independent trials per seed node (using `numpy.random.SeedSequence` for reproducible, statistically independent per-trial RNGs). It reports:
- Mean / Std cascade size
- 95% confidence interval
- Spearman ρ and RMSE vs. the deterministic Fragility Index

**Finding the Chaos Point:** Use `scenario_manager.py` with `--scenarios sweep` to find the minimum `k` at which Spearman ρ ≥ 0.5 (the engine's rankings become reliable). Use that `k` value for operational risk runs.

```bash
python3 -m cascade_engine.runner cascade_engine/config_stochastic_er.json \
    --output-dir results/stoch/
```

---

## 10. Consulting Tools Reference

### `data_adapter.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | — | Path to CSV or Excel edgelist |
| `--output` / `-o` | `config_custom.json` | Output JSON config path |
| `--sheet-name` | first sheet | Excel sheet name or 0-based index |
| `--mode` | `deterministic` | `deterministic` or `stochastic` |
| `--threshold-type` | `uniform` | `uniform` or `normal` |
| `--deg-low` | `0.1` | Uniform: lower bound for θ_deg |
| `--deg-high` | `0.4` | Uniform: upper bound for θ_deg |
| `--fail-low` | `0.5` | Uniform: lower bound for θ_fail |
| `--fail-high` | `0.9` | Uniform: upper bound for θ_fail |
| `--deg-mean` | — | Normal: mean for θ_deg |
| `--deg-std` | — | Normal: std dev for θ_deg |
| `--fail-mean` | — | Normal: mean for θ_fail |
| `--fail-std` | — | Normal: std dev for θ_fail |
| `--seed` | `42` | Global RNG seed |
| `--stochastic-k` | `10.0` | Logistic steepness (stochastic mode) |
| `--monte-carlo-trials` | `50` | Trials per seed node (stochastic mode) |
| `--enable-sensitivity` | off | Enable sensitivity analysis block |

### `visualizer.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` / `-r` | required | Engine output directory |
| `--output-dir` / `-o` | `visuals/` | Where to write images |
| `--k-sweep-dir` | — | k-sweep root for Convergence Curve |
| `--dpi` | `120` | Output image resolution |
| `--fmt` | `png` | `png`, `pdf`, or `svg` |
| `--top-n-annotate` | `8` | Nodes to annotate on Ghost Hub plot |
| `--skip-heatmap` | off | Skip Risk Heatmap (graph topology optional) |

### `report_gen.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` / `-r` | required | Engine output directory |
| `--visuals-dir` / `-v` | — | Directory with visualiser PNGs (optional) |
| `--scenario-dir` | — | scenario_manager output root (optional) |
| `--output` / `-o` | `report.html` | Output report path |
| `--format` | `html` | `html` or `markdown` |
| `--org-name` | `Infrastructure Client` | Organisation name for report header |
| `--analyst` | `Risk Analytics Team` | Analyst name for footer |

### `scenario_manager.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `config` | required | Base JSON config |
| `--output-dir` / `-o` | `scenario_results/` | Root output directory |
| `--scenarios` | all | Space-separated: `hardening regional sweep` |
| `--hardening-deltas` | `0.1 0.2 0.3` | Threshold increase magnitudes |
| `--top-k` | `10` | Top in-degree nodes for hardening |
| `--vulnerable-nodes` | — | Node IDs for regional scenario |
| `--vulnerability-delta` | `0.2` | Threshold reduction for regional scenario |
| `--k-values` | `2 5 10 20 40 80` | Steepness values for k-sweep |
| `--monte-carlo-trials` | `30` | Trials per node for k-sweep |

---

## 11. Running Tests

```bash
# Full test suite (engine + tools), from project root
python3 -m pytest cascade_engine/tests tools/tests -v

# Engine only
python3 -m pytest cascade_engine/tests -v

# Tools only
python3 -m pytest tools/tests -v

# With coverage (requires pytest-cov)
python3 -m pytest cascade_engine/tests tools/tests --cov=cascade_engine --cov=tools -v
```

Test inventory:

| Suite | Tests | Covers |
|-------|-------|--------|
| `test_propagation.py` | 25 | Monotonicity, convergence, chain graphs, FI, input validation |
| `test_tier2.py` | 103 | Sigmoid, stochastic propagation, Monte Carlo, CI, sensitivity, metrics |
| `test_data_adapter.py` | 40 | CSV + Excel ingestion, node mapping, edge building, threshold config, end-to-end |

---

## 12. Design Decisions and Constraints

- **Single-user internal tool.** No web interface, database, authentication, or multi-tenancy. Simplicity and directness are intentional.
- **Tier 1 / Tier 2 separation is strict.** Deterministic and stochastic engines are isolated modules. Tier 1 results are never used as starting points for stochastic runs — both always start from the same initial seed state.
- **Monotonicity is enforced.** Nodes never recover. This is a deliberate modelling choice for infrastructure failure cascades where recovery requires explicit intervention outside the model.
- **RNG independence via SeedSequence.** Each Monte Carlo trial gets an independent RNG spawned from a master `SeedSequence`. This prevents correlation between trials and makes results exactly reproducible given the same master seed.
- **Edge weights are not used.** The engine is unweighted. Weights from client data are stored in config metadata for documentation purposes only. A weighted extension would require a different threshold formulation.
- **Node ID remapping.** The engine requires 0-indexed compact integer node IDs. `data_adapter.py` handles this transparently — the mapping is stored in the config for traceability.
- **`A[j, i] = 1` convention.** The adjacency matrix encodes `j → i` (j influences i) in the `[row, col]` = `[j, i]` position. In-degree of node `i` = `A[:, i].sum()`.
- **Sensitivity analysis approximation.** The `hardening` scenario in `scenario_manager.py` raises thresholds globally, not per-node. For true per-node threshold changes, build a `custom` graph config limited to the affected sub-region.
# Cascade Propagation Engine

A deterministic and stochastic, vectorized, threshold-based cascade propagation engine for directed graphs. This README has been updated to reflect the current repository layout and CLI/pytest usage.

---

## Repository layout (current)

```
cascade_engine/
├── __init__.py
├── config.py                               # JSON config loader + threshold generation
├── graph.py                                # Graph generation (NetworkX → NumPy)
├── propagation.py                          # Deterministic propagation engine (Tier 1)
├── stochastic_propagation.py               # Logistic stochastic extension (Tier 2)
├── metrics.py                              # Cascade size, fragility, RMSE, MAPE, Spearman
├── monte_carlo.py                          # Monte Carlo harness + CI utilities (Tier 2)
├── sensitivity.py                          # Threshold sensitivity analysis (Tier 2)
├── runner.py                               # Unified CLI (Tier 1 + Tier 2)
├── utils.py                                # Shared utilities (CI, RNG spawning, helpers)
├── conftest.py                             # pytest fixtures (tests live inside package)
├── Makefile                                # convenience shortcuts (install, test, run)
├── requirements.txt
├── config_erdos_renyi.json                 # Tier 1: deterministic, ER graph
├── config_barabasi_albert.json             # Tier 1: deterministic, BA graph
├── config_stochastic_er.json               # Tier 2: stochastic, ER graph
├── config_stochastic_ba.json               # Tier 2: stochastic, BA + sensitivity
├── config_deterministic_with_sensitivity.json
└── tests/                                  # pytest tests (inside package)
    ├── test_config.py
    ├── test_graph.py
    ├── test_propagation.py
    └── test_tier2.py
```

> Note: tests now live inside `cascade_engine/tests` and `conftest.py` is bundled with the package. **pytest users** do not need a project-root `conftest.py` — `conftest.py` handles `sys.path` automatically. The `sys.path.insert` lines in each test file are load-bearing for `python3 -m unittest discover` and must not be removed.

---

## Quick setup

Install the runtime/test dependencies:

```bash
pip install -r cascade_engine/requirements.txt
```

(or run the Makefile `make install` from the `cascade_engine/` directory)

---

## Run tests

Recommended from the **repository root** (works consistently with CI):

```bash
# All tests (Tier 1 + Tier 2)
pytest cascade_engine/tests -v

# Tier 1 only
pytest cascade_engine/tests/test_config.py cascade_engine/tests/test_graph.py cascade_engine/tests/test_propagation.py -v

# Tier 2 only
pytest cascade_engine/tests/test_tier2.py -v
```

Alternatively, run the Makefile targets from inside `cascade_engine/` as documented in `Makefile`.

---

## Running experiments / examples

Two valid invocation styles are supported. Run either from the package directory (`cd cascade_engine`) or from repository root using the `-m` module form.

From inside the package directory:

```bash
python runner.py config_erdos_renyi.json --output-dir results_er/
python runner.py config_stochastic_er.json --output-dir results_stoch/
```

From the repository root (module invocation — preferred for reproducible environments):

```bash
python -m cascade_engine.runner cascade_engine/config_erdos_renyi.json --output-dir results_er/
python -m cascade_engine.runner cascade_engine/config_stochastic_er.json --output-dir results_stoch/
```

---

## Output files

The engine produces the following artifacts (file names are exact as produced by the `runner`):

* `fragility_results.csv` — Per-node: fragility index, thresholds, in-degree
* `topk_cascade_results.csv` — Top-k worst-case cascade stats by fragility index + in-degree (deterministic mode only)
* `results_summary.csv` — Aggregate key-value statistics
* `summary.json` — Structured JSON summary
* `config_snapshot.json` — Exact config + SHA-256 hash
* `experiment_metadata.json` — Timestamp, file paths, config hash
* `monte_carlo_distribution.csv` — Raw trial-level Monte Carlo results (Tier 2)
* `sensitivity_results.csv` / `sensitivity_aggregate.csv` — Sensitivity analysis outputs (when enabled)

All outputs are written into the `--output-dir` you provide to `runner.py`.

---

## Config schema (example)

```json
{
  "graph": {
    "type": "erdos_renyi | barabasi_albert | watts_strogatz | custom",
    "n": 100,
    "p": 0.08,
    "m": 3,
    "seed": 42
  },
  "thresholds": {
    "type": "uniform | normal",
    "deg_low": 0.1,  "deg_high": 0.4,
    "fail_low": 0.5, "fail_high": 0.9
  },
  "seed": 42,
  "propagation_mode": "deterministic | stochastic",

  "stochastic_k": 10.0,
  "monte_carlo_trials": 50,

  "sensitivity_analysis": false,
  "sensitivity_config": {
    "perturbation_values": [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
    "max_seed_nodes": 20,
    "stochastic_trials": 20
  }
}
```

---

## Model details (behaviour summary)

* **States:** `0` = operational, `1` = degraded, `2` = failed. States are monotonically non-decreasing (no recovery modelled).
* **Deterministic mode (Tier 1):** Hard threshold crossings based on fraction of **failed** in-neighbors.
* **Stochastic mode (Tier 2):** Degradation uses hard thresholds; failure uses a logistic probability `sigmoid(k * (F_i - theta_fail))`.
* **Zero in-degree (isolates):** Nodes with `D_i == 0` do not receive cascade pressure and will not change state from cascade logic unless explicitly seeded.

See the in-code docstrings for detailed mathematical notes and edge-case behaviour.

---

## RNG & reproducibility

* RNGs are created using NumPy `default_rng` and `SeedSequence` spawn patterns. The `utils` module centralises RNG spawning for per-node and per-trial streams.
* Experiments are reproducible from a single integer `seed` in the config; Monte Carlo trials use deterministic per-trial streams derived from that seed.

---

## Makefile shortcuts

The `Makefile` inside `cascade_engine/` contains convenient targets:

* `make install` — install requirements
* `make test` — run full test suite
* `make test-tier1` / `make test-tier2` — run respective test groups
* `make run-er` / `make run-ba` / `make run-ws` / `make run-stoch-er` / `make run-stoch-ba` — example experiment runs

---

## k-Sweep analysis (`k_sweep_analysis.py`)

The `k_sweep_analysis.py` script (at the repository root) validates that the stochastic engine converges to the deterministic engine as the logistic steepness parameter `k` increases. It sweeps over a range of `k` values, runs a full stochastic Monte Carlo experiment for each, and compares results against the deterministic fragility index using Spearman rank correlation (ρ) and RMSE.

### What it measures

For each `k` value, the script reports:

* **Spearman ρ** — rank correlation between stochastic mean cascade sizes and deterministic fragility fractions. Should approach 1.0 as k → ∞.
* **RMSE** — absolute prediction error between stochastic and deterministic results. Should approach 0 as k → ∞.

### Running the sweep

From the repository root:

```bash
python3 k_sweep_analysis.py
```

The script uses `cascade_engine/config_stochastic_er.json` as the base config and overwrites `stochastic_k` for each iteration. Results are printed to stdout and a plot is saved to `k_sweep_results.png`.

### Expected output

```
k-Value    | Spearman Rho    | RMSE
----------------------------------------
2          | 0.0000          | 0.9796
5          | 0.0000          | 0.9796
10         | 0.3624          | 0.1263
20         | 0.9795          | 0.0003
40         | 1.0000          | 0.0000
80         | 1.0000          | 0.0000
150        | 1.0000          | 0.0000
300        | 1.0000          | 0.0000
```

Low `k` values (2–5) produce a flat logistic curve so all nodes receive similar failure probabilities regardless of their F_i — the ranking signal is lost and ρ ≈ 0. By k = 20–40 the stochastic results converge to the deterministic rankings. This confirms Tier 2 is a strict generalisation of Tier 1.

### Output artefacts

* `results_k_sweep/k_{k}/` — full runner output directory for each k value (fragility CSVs, summary JSON, etc.)
* `k_sweep_results.png` — two-panel plot: Spearman ρ vs k and RMSE vs k

---



* Tests are structured into Tier 1 (deterministic) and Tier 2 (stochastic) groups — keep tests deterministic by fixing `seed` values where applicable.
* Avoid global NumPy RNG state; use `default_rng` and `SeedSequence` for reproducibility.
* If adding new output artifacts, add consistent metadata in `experiment_metadata.json` so downstream analysis tools can consume runs programmatically.

---

## Adjacency matrix convention

`A[j, i] = 1` ↔ directed edge `j → i` (column `i` = in-neighbors of node `i`).
