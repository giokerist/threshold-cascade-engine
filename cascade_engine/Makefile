PYTHON ?= python3

# Makefile for cascade_engine
# Run from the cascade_engine/ directory.

.PHONY: install test test-tier1 test-tier2 run-er run-ba run-stoch-er run-stoch-ba

install:
	pip install -r requirements.txt

# Run all tests (Tier 1 + Tier 2)
test:
	$(PYTHON) -m pytest tests/ -v

# Tier 1 tests only
test-tier1:
	$(PYTHON) -m pytest tests/test_config.py tests/test_graph.py tests/test_propagation.py -v

# Tier 2 tests only
test-tier2:
	$(PYTHON) -m pytest tests/test_tier2.py -v

# Example runs
run-er:
	$(PYTHON) runner.py config_erdos_renyi.json --output-dir results_er/

run-ba:
	$(PYTHON) runner.py config_barabasi_albert.json --output-dir results_ba/

run-ws:
	$(PYTHON) runner.py config_watts_strogatz.json --output-dir results_ws/

run-stoch-er:
	$(PYTHON) runner.py config_stochastic_er.json --output-dir results_stoch_er/

run-stoch-ba:
	$(PYTHON) runner.py config_stochastic_ba.json --output-dir results_stoch_ba/
