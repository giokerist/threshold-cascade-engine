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

> Note: tests now live inside `cascade_engine/tests` and `conftest.py` is bundled with the package. You no longer need a project-root `conftest.py` or manual `sys.path` hacks to run the tests.

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
* `make run-er` / `make run-ba` / `make run-stoch-er` / `make run-stoch-ba` — example experiment runs

---

## Contributing / notes for maintainers

* Tests are structured into Tier 1 (deterministic) and Tier 2 (stochastic) groups — keep tests deterministic by fixing `seed` values where applicable.
* Avoid global NumPy RNG state; use `default_rng` and `SeedSequence` for reproducibility.
* If adding new output artifacts, add consistent metadata in `experiment_metadata.json` so downstream analysis tools can consume runs programmatically.

---

## Adjacency matrix convention

`A[j, i] = 1` ↔ directed edge `j → i` (column `i` = in-neighbors of node `i`).
