# Cascade Propagation Engine

A deterministic and stochastic, vectorized, threshold-based cascade propagation engine for directed graphs.

## Architecture

```
cascade_engine/
├── config.py                               # JSON config loader + threshold generation
├── graph.py                                # Graph generation (NetworkX → NumPy)
├── propagation.py                          # Deterministic propagation engine (Tier 1)
├── stochastic_propagation.py               # Logistic stochastic extension (Tier 2)
├── metrics.py                              # Cascade size, fragility, RMSE, MAPE, Spearman
├── monte_carlo.py                          # Monte Carlo harness + CI utilities (Tier 2)
├── sensitivity.py                          # Threshold sensitivity analysis (Tier 2)
├── runner.py                               # Unified CLI (Tier 1 + Tier 2)
│
├── config_erdos_renyi.json                 # Tier 1: deterministic, ER graph
├── config_barabasi_albert.json             # Tier 1: deterministic, BA graph
├── config_stochastic_er.json               # Tier 2: stochastic, ER graph
├── config_stochastic_ba.json               # Tier 2: stochastic, BA + sensitivity
├── config_deterministic_with_sensitivity.json
├── requirements.txt
│
├── test_propagation.py                     # Tier 1 tests (unchanged)
├── test_graph.py                           # Tier 1 tests (unchanged)
├── test_config.py                          # Tier 1 tests (unchanged)
└── test_tier2.py                           # Tier 2 tests (65 tests)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
# All 105 tests (Tier 1 + Tier 2) — run from cascade_engine/
pytest tests/ -v

# Tier 1 only
pytest tests/test_config.py tests/test_graph.py tests/test_propagation.py -v

# Tier 2 only
pytest tests/test_tier2.py -v
```

> **Note:** Always run from the `cascade_engine/` directory. `conftest.py` at the project root handles `sys.path` automatically so no install step is needed beyond `pip install -r requirements.txt`.

---

## Tier 1: Deterministic Mode

```bash
python runner.py config_erdos_renyi.json --output-dir results_er/
python runner.py config_barabasi_albert.json --output-dir results_ba/
```

### Output Files

| File | Description |
|------|-------------|
| `fragility_results.csv` | Per-node: fragility index, thresholds, in-degree |
| `results_summary.csv` | Aggregate key-value statistics |
| `summary.json` | Structured JSON summary |
| `config_snapshot.json` | Exact config + SHA-256 hash |
| `experiment_metadata.json` | Timestamp, file paths, config hash |

---

## Tier 2: Stochastic / Monte Carlo Mode

```bash
# Stochastic Monte Carlo
python runner.py config_stochastic_er.json --output-dir results_stoch/

# Stochastic + sensitivity analysis
python runner.py config_stochastic_ba.json --output-dir results_ba_sens/

# Deterministic + sensitivity analysis
python runner.py config_deterministic_with_sensitivity.json --output-dir results_det_sens/
```

### Additional Output Files (Tier 2)

| File | Description |
|------|-------------|
| `fragility_results.csv` | Per-node: deterministic + stochastic fragility, 95% CI |
| `monte_carlo_distribution.csv` | Raw trial data: node_id, trial, cascade_size_frac, time_to_stability |
| `sensitivity_results.csv` | Per (δ, seed_node): cascade_size, std, n_trials |
| `sensitivity_aggregate.csv` | Mean cascade per perturbation level |

---

## Config Schema

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
    "perturbation_values": [-0.2, -0.1, 0.0, 0.1, 0.2],
    "max_seed_nodes": 20,
    "stochastic_trials": 20
  }
}
```

---

## Model Details

### States
- `0` = operational  
- `1` = degraded  
- `2` = failed

States are **monotonically non-decreasing** (0 → 1 → 2 only). Recovery is not modeled.

---

### Deterministic Update Rule (Tier 1)

```
D_i    = in-degree of node i
F_i(t) = (# failed in-neighbors) / D_i    [0 if D_i == 0]

if   F_i >= theta_fail[i]  →  new_state = 2
elif F_i >= theta_deg[i]   →  new_state = 1
else                        →  state unchanged
```

---

### Stochastic Update Rule (Tier 2)

Degradation: same hard threshold as deterministic.

Failure: logistic probability (nodes with D_i == 0 are never cascade-triggered):

```
P(S_i → 2) = sigmoid( k * (F_i(t) − theta_fail[i]) )

sigmoid(x) = 1 / (1 + exp(−x))
```

- `k` controls steepness (recommended: 5–20)
- At `k → ∞`: recovers the deterministic hard threshold exactly

---

### Convergence

- Deterministic: ≤ `2n` steps (monotone bound)
- Stochastic: ≤ `4n` steps (monotone absorption guaranteed)

---

### Extended Metrics (Tier 2)

| Function | Description |
|----------|-------------|
| `rmse(predicted, observed)` | Root Mean Square Error |
| `mape(predicted, observed)` | Mean Absolute Percentage Error (%) |
| `spearman_correlation(x, y)` | Spearman ρ and two-tailed p-value |
| `confidence_interval(samples, confidence)` | t-distribution CI for population mean |

---

### RNG Discipline

- No global NumPy random state used anywhere
- Each Monte Carlo trial: `default_rng(master_seed + trial_index)`
- Per-node seed blocks: `default_rng(master_seed + node * trials + trial)`
- Sensitivity seeds: `seed + p_idx * n * trials + sn_idx * trials + trial`
- Full experiment is reproducible from a single `seed` integer in the config

---

## Adjacency Matrix Convention

`A[j, i] = 1` ↔ directed edge `j → i` (column `i` = in-neighbors of node `i`).
