"""
Unit tests for tools/data_adapter.py.

Covers:
  - CSV ingestion (happy path, missing columns, empty file)
  - Excel ingestion (.xlsx) with openpyxl
  - Node mapping (identity, remapped, string labels)
  - Edge list construction (self-loop detection, duplicate collapsing)
  - _resolve() type robustness (int/float/str labels from pandas)
  - Threshold config building (uniform and normal)
  - Config assembly (deterministic and stochastic modes)
  - End-to-end: CSV → JSON → engine runnable
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.data_adapter import (
    load_edgelist,
    build_node_mapping,
    build_edge_list,
    build_threshold_config,
    assemble_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(rows: str) -> Path:
    """Write rows to a temp CSV file and return its Path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    f.write(rows)
    f.flush()
    f.close()
    return Path(f.name)


def _make_xlsx(df: pd.DataFrame, sheet: str = "Sheet1") -> Path:
    """Write a DataFrame to a temp Excel file and return its Path."""
    f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    f.close()
    path = Path(f.name)
    df.to_excel(path, index=False, sheet_name=sheet)
    return path


def _fake_args(
    mode: str = "deterministic",
    seed: int = 42,
    threshold_type: str = "uniform",
    deg_low: float = 0.1,
    deg_high: float = 0.4,
    fail_low: float = 0.5,
    fail_high: float = 0.9,
    deg_mean=None,
    deg_std=None,
    fail_mean=None,
    fail_std=None,
    stochastic_k: float = 10.0,
    monte_carlo_trials: int = 50,
    enable_sensitivity: bool = False,
):
    """Return a minimal argparse.Namespace for threshold config tests."""
    import argparse
    return argparse.Namespace(
        mode=mode,
        seed=seed,
        threshold_type=threshold_type,
        deg_low=deg_low,
        deg_high=deg_high,
        fail_low=fail_low,
        fail_high=fail_high,
        deg_mean=deg_mean,
        deg_std=deg_std,
        fail_mean=fail_mean,
        fail_std=fail_std,
        stochastic_k=stochastic_k,
        monte_carlo_trials=monte_carlo_trials,
        enable_sensitivity=enable_sensitivity,
    )


# ---------------------------------------------------------------------------
# load_edgelist — CSV
# ---------------------------------------------------------------------------


class TestLoadEdgelistCSV(unittest.TestCase):

    def test_valid_csv_returns_df_and_format(self):
        p = _make_csv("source,target\n0,1\n1,2\n2,0\n")
        df, fmt = load_edgelist(p)
        self.assertEqual(fmt, "csv")
        self.assertEqual(len(df), 3)
        self.assertIn("source", df.columns)
        self.assertIn("target", df.columns)
        p.unlink()

    def test_weight_column_preserved(self):
        p = _make_csv("source,target,weight\nA,B,1.0\nB,C,0.5\n")
        df, _ = load_edgelist(p)
        self.assertIn("weight", df.columns)
        p.unlink()

    def test_column_names_normalised_lower(self):
        p = _make_csv("  Source , TARGET \n0,1\n")
        df, _ = load_edgelist(p)
        self.assertIn("source", df.columns)
        self.assertIn("target", df.columns)
        p.unlink()

    def test_missing_source_raises(self):
        p = _make_csv("from,target\n0,1\n")
        with self.assertRaises(ValueError):
            load_edgelist(p)
        p.unlink()

    def test_missing_target_raises(self):
        p = _make_csv("source,to\n0,1\n")
        with self.assertRaises(ValueError):
            load_edgelist(p)
        p.unlink()

    def test_empty_raises(self):
        p = _make_csv("source,target\n")
        with self.assertRaises(ValueError):
            load_edgelist(p)
        p.unlink()

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_edgelist(Path("/tmp/__nonexistent_cascade_test__.csv"))

    def test_na_rows_dropped(self):
        p = _make_csv("source,target\n0,1\n,2\n2,\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df, _ = load_edgelist(p)
        self.assertEqual(len(df), 1)
        p.unlink()

    def test_all_na_raises(self):
        p = _make_csv("source,target\n,\n,\n")
        with self.assertRaises(ValueError):
            load_edgelist(p)
        p.unlink()


# ---------------------------------------------------------------------------
# load_edgelist — Excel
# ---------------------------------------------------------------------------


class TestLoadEdgelistExcel(unittest.TestCase):

    def test_valid_xlsx_returns_df_and_format(self):
        df_in = pd.DataFrame({"source": ["A", "B", "C"], "target": ["B", "C", "A"]})
        path = _make_xlsx(df_in)
        try:
            df, fmt = load_edgelist(path)
            self.assertEqual(fmt, "excel")
            self.assertEqual(len(df), 3)
            self.assertIn("source", df.columns)
        finally:
            path.unlink(missing_ok=True)

    def test_excel_with_weight_column(self):
        df_in = pd.DataFrame({
            "source": [0, 1, 2],
            "target": [1, 2, 0],
            "weight": [1.0, 0.8, 0.5],
        })
        path = _make_xlsx(df_in)
        try:
            df, fmt = load_edgelist(path)
            self.assertIn("weight", df.columns)
            self.assertEqual(fmt, "excel")
        finally:
            path.unlink(missing_ok=True)

    def test_excel_specific_sheet(self):
        path = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        path.close()
        path = Path(path.name)
        try:
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"source": ["X"], "target": ["Y"]}).to_excel(
                    writer, sheet_name="Data", index=False
                )
                pd.DataFrame({"irrelevant": [1, 2]}).to_excel(
                    writer, sheet_name="Other", index=False
                )
            df, fmt = load_edgelist(path, sheet_name="Data")
            self.assertEqual(len(df), 1)
            self.assertIn("source", df.columns)
        finally:
            path.unlink(missing_ok=True)

    def test_excel_missing_required_column_raises(self):
        df_in = pd.DataFrame({"from": ["A"], "to": ["B"]})
        path = _make_xlsx(df_in)
        try:
            with self.assertRaises(ValueError):
                load_edgelist(path)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# build_node_mapping
# ---------------------------------------------------------------------------


class TestBuildNodeMapping(unittest.TestCase):

    def test_compact_ints_no_remap(self):
        df = pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0]})
        info = build_node_mapping(df)
        self.assertFalse(info["remapped"])
        self.assertEqual(info["n"], 3)

    def test_non_compact_ints_remapped(self):
        df = pd.DataFrame({"source": [10, 20], "target": [20, 30]})
        info = build_node_mapping(df)
        self.assertTrue(info["remapped"])
        self.assertEqual(info["n"], 3)

    def test_string_labels_remapped(self):
        df = pd.DataFrame({"source": ["A", "B", "C"], "target": ["B", "C", "A"]})
        info = build_node_mapping(df)
        self.assertTrue(info["remapped"])
        self.assertEqual(info["n"], 3)
        # Labels should be sorted alphabetically → A=0, B=1, C=2
        self.assertEqual(info["mapping"]["A"], 0)

    def test_single_node_both_endpoints(self):
        df = pd.DataFrame({"source": ["X"], "target": ["X"]})
        info = build_node_mapping(df)
        self.assertEqual(info["n"], 1)

    def test_n_equals_unique_nodes(self):
        df = pd.DataFrame({"source": ["A", "A", "B"], "target": ["B", "C", "C"]})
        info = build_node_mapping(df)
        self.assertEqual(info["n"], 3)


# ---------------------------------------------------------------------------
# build_edge_list
# ---------------------------------------------------------------------------


class TestBuildEdgeList(unittest.TestCase):

    def _df_and_info(self, rows):
        df = pd.DataFrame(rows, columns=["source", "target"])
        info = build_node_mapping(df)
        return df, info

    def test_basic_edges_clean(self):
        df, info = self._df_and_info([(0, 1), (1, 2), (2, 0)])
        edges, n_sl, n_dup = build_edge_list(df, info)
        self.assertEqual(len(edges), 3)
        self.assertEqual(n_sl, 0)
        self.assertEqual(n_dup, 0)

    def test_self_loops_stripped(self):
        df, info = self._df_and_info([(0, 0), (0, 1), (1, 1)])
        edges, n_sl, n_dup = build_edge_list(df, info)
        self.assertEqual(n_sl, 2)
        self.assertEqual(len(edges), 1)

    def test_duplicates_collapsed(self):
        df, info = self._df_and_info([(0, 1), (0, 1), (1, 2)])
        edges, n_sl, n_dup = build_edge_list(df, info)
        self.assertEqual(n_dup, 1)
        self.assertEqual(len(edges), 2)

    def test_string_label_resolution(self):
        df = pd.DataFrame({"source": ["A", "B", "C"], "target": ["B", "C", "A"]})
        info = build_node_mapping(df)
        edges, n_sl, n_dup = build_edge_list(df, info)
        self.assertEqual(len(edges), 3)
        self.assertEqual(n_sl, 0)
        self.assertEqual(n_dup, 0)
        # All resolved indices must be in range [0, 2]
        for u, v in edges:
            self.assertIn(u, range(3))
            self.assertIn(v, range(3))

    def test_float_int_labels_from_pandas(self):
        """Pandas reads integer CSV columns as float64; _resolve must handle 1.0 → 1."""
        df = pd.DataFrame({"source": [0.0, 1.0, 2.0], "target": [1.0, 2.0, 0.0]})
        info = build_node_mapping(df)
        edges, n_sl, n_dup = build_edge_list(df, info)
        self.assertEqual(len(edges), 3)

    def test_edges_are_integer_pairs(self):
        df, info = self._df_and_info([(0, 1), (1, 2)])
        edges, _, _ = build_edge_list(df, info)
        for edge in edges:
            self.assertIsInstance(edge[0], int)
            self.assertIsInstance(edge[1], int)


# ---------------------------------------------------------------------------
# build_threshold_config
# ---------------------------------------------------------------------------


class TestBuildThresholdConfig(unittest.TestCase):

    def test_uniform_returns_correct_keys(self):
        args = _fake_args(threshold_type="uniform")
        cfg = build_threshold_config(args)
        self.assertEqual(cfg["type"], "uniform")
        for k in ("deg_low", "deg_high", "fail_low", "fail_high"):
            self.assertIn(k, cfg)

    def test_normal_returns_correct_keys(self):
        args = _fake_args(
            threshold_type="normal",
            deg_mean=0.3, deg_std=0.08,
            fail_mean=0.65, fail_std=0.10,
        )
        cfg = build_threshold_config(args)
        self.assertEqual(cfg["type"], "normal")
        for k in ("deg_mean", "deg_std", "fail_mean", "fail_std"):
            self.assertIn(k, cfg)

    def test_uniform_inverted_range_raises(self):
        args = _fake_args(threshold_type="uniform", deg_low=0.5, deg_high=0.1)
        with self.assertRaises(ValueError):
            build_threshold_config(args)

    def test_normal_missing_param_raises(self):
        args = _fake_args(threshold_type="normal")  # mean/std are None
        with self.assertRaises(ValueError):
            build_threshold_config(args)

    def test_normal_zero_std_raises(self):
        args = _fake_args(
            threshold_type="normal",
            deg_mean=0.3, deg_std=0.0,
            fail_mean=0.65, fail_std=0.10,
        )
        with self.assertRaises(ValueError):
            build_threshold_config(args)


# ---------------------------------------------------------------------------
# assemble_config
# ---------------------------------------------------------------------------


class TestAssembleConfig(unittest.TestCase):

    def _make_cfg(self, mode="deterministic", enable_sensitivity=False, stochastic=False):
        edges = [[0, 1], [1, 2], [2, 0]]
        n = 3
        thresh = {"type": "uniform", "deg_low": 0.1, "deg_high": 0.4,
                  "fail_low": 0.5, "fail_high": 0.9}
        args = _fake_args(mode=mode, enable_sensitivity=enable_sensitivity)
        meta = {"source_file": "test.csv", "file_format": "csv",
                "n_nodes": n, "n_edges": len(edges)}
        return assemble_config(edges, n, thresh, args, meta)

    def test_graph_type_is_custom(self):
        cfg = self._make_cfg()
        self.assertEqual(cfg["graph"]["type"], "custom")

    def test_graph_n_correct(self):
        cfg = self._make_cfg()
        self.assertEqual(cfg["graph"]["n"], 3)

    def test_edges_present(self):
        cfg = self._make_cfg()
        self.assertEqual(len(cfg["graph"]["edges"]), 3)

    def test_deterministic_mode(self):
        cfg = self._make_cfg(mode="deterministic")
        self.assertEqual(cfg["propagation_mode"], "deterministic")
        self.assertNotIn("stochastic_k", cfg)

    def test_stochastic_mode_has_k_and_trials(self):
        cfg = self._make_cfg(mode="stochastic")
        self.assertEqual(cfg["propagation_mode"], "stochastic")
        self.assertIn("stochastic_k", cfg)
        self.assertIn("monte_carlo_trials", cfg)

    def test_sensitivity_disabled(self):
        cfg = self._make_cfg(enable_sensitivity=False)
        self.assertFalse(cfg["sensitivity_analysis"])

    def test_sensitivity_enabled(self):
        cfg = self._make_cfg(enable_sensitivity=True)
        self.assertTrue(cfg["sensitivity_analysis"])
        self.assertIn("sensitivity_config", cfg)

    def test_config_is_json_serialisable(self):
        cfg = self._make_cfg()
        json_str = json.dumps(cfg)
        self.assertIsInstance(json_str, str)

    def test_adapter_meta_stored(self):
        cfg = self._make_cfg()
        self.assertIn("_adapter_meta", cfg)
        self.assertIn("file_format", cfg["_adapter_meta"])


# ---------------------------------------------------------------------------
# End-to-end: CSV → JSON config → cascade engine runner
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):

    def test_csv_to_config_to_runner(self):
        """Full pipeline: CSV edgelist → config JSON → engine produces fragility CSV."""
        import subprocess

        p = _make_csv("source,target\n0,1\n1,2\n2,3\n3,0\n1,3\n")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as cf:
            config_path = Path(cf.name)
        with tempfile.TemporaryDirectory() as results_dir:
            repo_root = Path(__file__).parent.parent.parent

            # Step 1: data_adapter → JSON
            r = subprocess.run(
                [
                    sys.executable, str(repo_root / "tools" / "data_adapter.py"),
                    str(p), "--output", str(config_path),
                ],
                capture_output=True, text=True, cwd=str(repo_root),
            )
            self.assertEqual(r.returncode, 0, msg=f"data_adapter failed:\n{r.stderr}")

            # Step 2: engine → results
            r2 = subprocess.run(
                [
                    sys.executable, "-m", "cascade_engine.runner",
                    str(config_path), "--output-dir", results_dir,
                ],
                capture_output=True, text=True, cwd=str(repo_root),
            )
            self.assertEqual(r2.returncode, 0, msg=f"runner failed:\n{r2.stderr}")

            # Step 3: verify output
            frag_csv = Path(results_dir) / "fragility_results.csv"
            self.assertTrue(frag_csv.exists(), "fragility_results.csv not produced")
            df = pd.read_csv(frag_csv)
            self.assertEqual(len(df), 4)  # 4 nodes
            self.assertIn("fragility_index", df.columns)

        p.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)

    def test_excel_to_config_to_runner(self):
        """Full pipeline: Excel edgelist → config JSON → engine produces fragility CSV."""
        import subprocess

        df_in = pd.DataFrame({
            "source": ["A", "B", "C", "D", "A"],
            "target": ["B", "C", "D", "A", "C"],
        })
        path = _make_xlsx(df_in)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as cf:
            config_path = Path(cf.name)
        with tempfile.TemporaryDirectory() as results_dir:
            repo_root = Path(__file__).parent.parent.parent

            r = subprocess.run(
                [
                    sys.executable, str(repo_root / "tools" / "data_adapter.py"),
                    str(path), "--output", str(config_path),
                ],
                capture_output=True, text=True, cwd=str(repo_root),
            )
            self.assertEqual(r.returncode, 0, msg=f"data_adapter (Excel) failed:\n{r.stderr}")

            r2 = subprocess.run(
                [
                    sys.executable, "-m", "cascade_engine.runner",
                    str(config_path), "--output-dir", results_dir,
                ],
                capture_output=True, text=True, cwd=str(repo_root),
            )
            self.assertEqual(r2.returncode, 0, msg=f"runner (Excel) failed:\n{r2.stderr}")

            frag_csv = Path(results_dir) / "fragility_results.csv"
            self.assertTrue(frag_csv.exists())
            df = pd.read_csv(frag_csv)
            self.assertEqual(len(df), 4)  # 4 unique nodes A,B,C,D

        path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
