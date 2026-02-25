"""Unit tests for the configuration module."""

from __future__ import annotations
import json, sys, tempfile, unittest, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cascade_engine.config import load_config, build_rng, generate_thresholds

VALID_CFG = {
    "graph": {"type": "erdos_renyi", "n": 20, "p": 0.2},
    "thresholds": {"type": "uniform", "deg_low": 0.1, "deg_high": 0.4,
                   "fail_low": 0.5, "fail_high": 0.9},
    "seed": 42,
    "propagation_mode": "deterministic",
}

def _write_cfg(d):
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(d, f); f.close()
    return Path(f.name)


class TestLoadConfig(unittest.TestCase):

    def test_valid_config_loads(self):
        cfg = load_config(_write_cfg(VALID_CFG))
        self.assertEqual(cfg["seed"], 42)

    def test_missing_seed_raises(self):
        bad = {k: v for k, v in VALID_CFG.items() if k != "seed"}
        with self.assertRaises(ValueError):
            load_config(_write_cfg(bad))

    def test_invalid_graph_type_raises(self):
        bad = {**VALID_CFG, "graph": {"type": "unknown", "n": 10}}
        with self.assertRaises(ValueError):
            load_config(_write_cfg(bad))

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")


class TestThresholds(unittest.TestCase):

    def test_uniform_clipped_to_unit_interval(self):
        rng = np.random.default_rng(0)
        cfg = {"type": "uniform", "deg_low": 0.1, "deg_high": 0.4,
               "fail_low": 0.5, "fail_high": 0.9}
        td, tf = generate_thresholds(100, cfg, rng)
        self.assertTrue(np.all((td >= 0) & (td <= 1)))
        self.assertTrue(np.all((tf >= 0) & (tf <= 1)))

    def test_fail_geq_deg_always(self):
        # This config intentionally uses overlapping normal distributions so
        # theta_fail < theta_deg for many nodes.  The engine corrects this and
        # emits a UserWarning.  We suppress it here so it doesn't pollute test
        # output and so the test doesn't fail under -W error::UserWarning.
        import warnings
        rng = np.random.default_rng(7)
        cfg = {"type": "normal", "deg_mean": 0.6, "deg_std": 0.2,
               "fail_mean": 0.4, "fail_std": 0.2}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            td, tf = generate_thresholds(100, cfg, rng)
        self.assertTrue(np.all(tf >= td))

    def test_reproducibility(self):
        cfg = {"type": "uniform", "deg_low": 0.1, "deg_high": 0.4,
               "fail_low": 0.5, "fail_high": 0.9}
        td1, tf1 = generate_thresholds(50, cfg, np.random.default_rng(123))
        td2, tf2 = generate_thresholds(50, cfg, np.random.default_rng(123))
        np.testing.assert_array_equal(td1, td2)
        np.testing.assert_array_equal(tf1, tf2)

    def test_build_rng_from_config(self):
        cfg = {**VALID_CFG}
        rng = build_rng(cfg)
        val1 = rng.integers(0, 1000)
        rng2 = build_rng(cfg)
        val2 = rng2.integers(0, 1000)
        self.assertEqual(val1, val2)

    def test_threshold_correction_emits_warning(self):
        """UserWarning must be raised when theta_fail is corrected upward."""
        # Use normal distribution with heavily overlapping means to guarantee violations
        rng = np.random.default_rng(7)
        cfg = {"type": "normal", "deg_mean": 0.6, "deg_std": 0.2,
               "fail_mean": 0.4, "fail_std": 0.2}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            td, tf = generate_thresholds(100, cfg, rng)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0,
                           "Expected a UserWarning when theta_fail < theta_deg")
        self.assertIn("theta_fail", str(user_warnings[0].message))

    def test_no_warning_when_thresholds_valid(self):
        """No warning when fail range is strictly above deg range."""
        rng = np.random.default_rng(0)
        cfg = {"type": "uniform", "deg_low": 0.1, "deg_high": 0.4,
               "fail_low": 0.5, "fail_high": 0.9}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_thresholds(200, cfg, rng)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 0,
                         "No warning expected for non-overlapping threshold ranges")


if __name__ == "__main__":
    unittest.main()
