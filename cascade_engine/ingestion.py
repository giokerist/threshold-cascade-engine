"""
ingestion.py — High-Performance CSV Ingestion with Integer Factorization
=========================================================================

Converts raw CSV edgelists (including multi-column composite keys) into compact
integer-ID representations for use by the sparse cascade engine.

Design principles
-----------------
* All string-to-integer mapping happens **once**, immediately after CSV load,
  via ``pd.factorize()``.  Everything downstream operates on ``int64`` arrays.
* A lightweight JSON-serialisable ``lookup_table`` is kept **only** for final
  UI reporting — it is never carried into the simulation hot path.
* Composite node keys (e.g. ``STATE + WFO + EPISODE_ID``) are supported by
  concatenating the columns before factorization.
* Optional numeric indicator columns (e.g. severity scores) are aggregated
  per node (mean) and returned as float64 arrays aligned to the node ID space.
* A stable ``edge_hash`` (SHA-256 of source/target int arrays) is returned for
  the graph-cache keying layer in ``graph_sparse.py``.

Usage
-----
>>> from cascade_engine.ingestion import ingest_edgelist
>>> result = ingest_edgelist(
...     "storm_events.csv",
...     source_col="STATE",
...     composite_cols=["STATE", "WFO", "EPISODE_ID"],
...     indicator_cols=["DEATHS_DIRECT"],
... )
>>> result.n_nodes, result.n_edges
(42317, 1_003_421)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Immutable product of ingesting a CSV edgelist.

    Attributes
    ----------
    source_ids : np.ndarray, shape (E,), dtype int64
        Integer IDs for the source end of each edge.
    target_ids : np.ndarray, shape (E,), dtype int64
        Integer IDs for the target end of each edge.
    n_nodes : int
        Total number of unique nodes (= max ID + 1).
    n_edges : int
        Total number of edges after deduplication.
    node_indicators : dict[str, np.ndarray]
        Mapping ``col_name → float64 array of shape (n_nodes,)`` holding
        per-node aggregated indicator values (mean).  Empty dict if no
        indicator columns were requested.
    lookup_table : dict[str, str]
        ``str(node_id) → original_label`` for UI reporting only.
        Never used inside the simulation hot path.
    edge_hash : str
        16-character hex prefix of the SHA-256 digest of the
        (source_ids ‖ target_ids) byte stream.  Used as a cache key in
        ``graph_sparse.build_or_load_sparse_graph``.
    composite_key_col : str or None
        Name of the composite key column that was created, or ``None`` if
        a single column was used directly.
    source_label_counts : dict[str, int]
        Number of edges emanating from each unique source label (for
        diagnostics / UI reporting).
    """

    source_ids: np.ndarray
    target_ids: np.ndarray
    n_nodes: int
    n_edges: int
    node_indicators: dict[str, np.ndarray]
    lookup_table: dict[str, str]
    edge_hash: str
    composite_key_col: str | None = None
    source_label_counts: dict[str, int] = field(default_factory=dict)

    def to_json_meta(self) -> dict[str, Any]:
        """Return a JSON-serialisable metadata dict (no large arrays)."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "edge_hash": self.edge_hash,
            "composite_key_col": self.composite_key_col,
            "indicator_columns": list(self.node_indicators.keys()),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_composite_col(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Concatenate *cols* into a single string key column (separator ``|``)."""
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("|".join, axis=1)


def _compute_edge_hash(src: np.ndarray, tgt: np.ndarray) -> str:
    """Return the first 16 hex chars of SHA-256(src_bytes ‖ tgt_bytes)."""
    h = hashlib.sha256()
    h.update(src.astype(np.int64).tobytes())
    h.update(tgt.astype(np.int64).tobytes())
    return h.hexdigest()[:16]


def _aggregate_indicators(
    df: pd.DataFrame,
    indicator_cols: Sequence[str],
    node_ids: np.ndarray,
    n_nodes: int,
) -> dict[str, np.ndarray]:
    """Compute per-node mean for each indicator column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw edge DataFrame (one row per edge).
    indicator_cols : sequence of str
        Columns to aggregate.
    node_ids : np.ndarray, shape (E,), dtype int64
        Node integer IDs aligned to the *source* end of each edge.
    n_nodes : int
        Total number of unique nodes.

    Returns
    -------
    dict[str, np.ndarray]
        ``col → float64 array of shape (n_nodes,)`` with per-node means.
        Nodes that appear in no edge for a given column receive ``np.nan``.
    """
    out: dict[str, np.ndarray] = {}
    for col in indicator_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        sums = np.zeros(n_nodes, dtype=np.float64)
        counts = np.zeros(n_nodes, dtype=np.float64)
        valid = ~np.isnan(vals)
        np.add.at(sums, node_ids[valid], vals[valid])
        np.add.at(counts, node_ids[valid], 1.0)
        result = np.where(counts > 0, sums / counts, np.nan)
        out[col] = result
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_edgelist(
    path: str | Path,
    source_col: str = "source",
    target_col: str = "target",
    composite_cols: list[str] | None = None,
    indicator_cols: list[str] | None = None,
    sep: str = ",",
    deduplicate: bool = True,
    drop_self_loops: bool = True,
    chunksize: int | None = None,
) -> IngestResult:
    """Ingest a CSV edgelist and map all node labels to compact integer IDs.

    This is the **entry point** for real-world data.  It must be called once
    after CSV load, before any graph construction or simulation.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file.
    source_col : str, optional
        Column containing edge source labels.  Default ``"source"``.
        If *composite_cols* is provided, this argument is ignored for node
        identity (but used for indicator aggregation if listed there).
    target_col : str, optional
        Column containing edge target labels.  Default ``"target"``.
        Must refer to node identity in the same label space as *source_col*
        (or the composite key).
    composite_cols : list of str or None, optional
        If provided, these columns are concatenated with ``|`` as separator
        to create a single composite node key before factorization.  Both
        the source and target sides are looked up in this unified key space.
        Example: ``["STATE", "WFO", "EPISODE_ID"]``.
        When *None*, ``source_col`` and ``target_col`` are used directly.
    indicator_cols : list of str or None, optional
        Numeric columns to aggregate per source node (mean).  Useful for
        carrying severity scores or frequency counts into threshold generation.
    sep : str, optional
        CSV field delimiter.  Default ``","`` (standard CSV).
    deduplicate : bool, optional
        If ``True`` (default), remove duplicate (source, target) pairs.
        Duplicate edges are not meaningful in a binary adjacency model.
    drop_self_loops : bool, optional
        If ``True`` (default), remove rows where source == target.
        Self-loops inflate failure pressure and corrupt fragility results.
    chunksize : int or None, optional
        If set, read the CSV in chunks of this size (memory-efficient for
        very large files).  ``None`` (default) reads the whole file at once.

    Returns
    -------
    IngestResult
        Container holding integer arrays, lookup table, and metadata.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing or the file is empty after filtering.

    Examples
    --------
    Single-column keys:
    >>> result = ingest_edgelist("edges.csv")

    Composite keys (NOAA storm events pattern):
    >>> result = ingest_edgelist(
    ...     "storm_events.csv",
    ...     source_col="SOURCE_LOCATION",
    ...     target_col="TARGET_LOCATION",
    ...     composite_cols=["STATE", "WFO", "EPISODE_ID"],
    ...     indicator_cols=["DEATHS_DIRECT", "INJURIES_DIRECT"],
    ... )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # ------------------------------------------------------------------
    # 1. Load CSV — chunked or full
    # ------------------------------------------------------------------
    if chunksize is not None:
        chunks = pd.read_csv(path, sep=sep, dtype=str, chunksize=chunksize)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep=sep, dtype=str)

    if df.empty:
        raise ValueError(f"CSV file '{path}' contains no rows.")

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # ------------------------------------------------------------------
    # 2. Resolve node identity columns
    # ------------------------------------------------------------------
    composite_key_col: str | None = None

    if composite_cols is not None:
        _missing = [c for c in composite_cols if c not in df.columns]
        if _missing:
            raise ValueError(
                f"Composite key columns missing from CSV: {_missing}. "
                f"Available: {df.columns.tolist()}"
            )
        composite_key_col = "_composite_key"
        df[composite_key_col] = _make_composite_col(df, composite_cols)
        _src_key_col = composite_key_col
        _tgt_key_col = composite_key_col  # same label space; need separate target
        # When composite keys are used, source and target must map to the same
        # label space.  The user must supply a separate target composite column
        # or rely on the engine using the same column for both ends.
        # For now: source side uses composite_key_col; target side also uses it
        # unless a separate target_composite_cols arg is added in a future version.
        # Typical use: each row is an edge event; the composite key encodes the
        # *affected entity* (= target).  Use source_col for the upstream entity.
        if source_col in df.columns:
            _src_key_col = source_col
        _tgt_key_col = composite_key_col
    else:
        for col in (source_col, target_col):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in CSV. "
                    f"Available: {df.columns.tolist()}"
                )
        _src_key_col = source_col
        _tgt_key_col = target_col

    # ------------------------------------------------------------------
    # 3. Build unified node label space & factorize
    # ------------------------------------------------------------------
    # Collect all node labels from both ends
    all_src_labels = df[_src_key_col].astype(str)
    all_tgt_labels = df[_tgt_key_col].astype(str)
    all_labels_series = pd.concat(
        [all_src_labels, all_tgt_labels], ignore_index=True
    )
    # pd.factorize assigns compact integer IDs 0..K-1 in first-occurrence order
    all_node_codes, node_uniques = pd.factorize(all_labels_series)
    n_half = len(df)
    src_codes_raw = all_node_codes[:n_half].astype(np.int64)
    tgt_codes_raw = all_node_codes[n_half:].astype(np.int64)

    # ------------------------------------------------------------------
    # 4. Filter: self-loops and duplicates
    # ------------------------------------------------------------------
    df = df.copy()
    df["_src_id"] = src_codes_raw
    df["_tgt_id"] = tgt_codes_raw

    if drop_self_loops:
        df = df[df["_src_id"] != df["_tgt_id"]]

    if deduplicate:
        df = df.drop_duplicates(subset=["_src_id", "_tgt_id"])

    if df.empty:
        raise ValueError(
            "No valid edges remain after filtering self-loops and duplicates."
        )

    source_ids = df["_src_id"].to_numpy(dtype=np.int64)
    target_ids = df["_tgt_id"].to_numpy(dtype=np.int64)

    # Re-compact IDs to 0..n_nodes-1 (some labels may have been dropped by
    # filtering — re-factorize the remaining pairs for a tight mapping)
    combined = pd.factorize(
        pd.concat(
            [pd.Series(source_ids), pd.Series(target_ids)], ignore_index=True
        )
    )
    final_codes, final_uniques = combined
    n_half2 = len(source_ids)
    source_ids = final_codes[:n_half2].astype(np.int64)
    target_ids = final_codes[n_half2:].astype(np.int64)

    # Map original labels (node_uniques) through the re-compaction
    # final_uniques contains the original integer codes from the first factorize
    # node_uniques[final_uniques[i]] = original string label for new node i
    node_labels_compact = node_uniques[final_uniques]

    n_nodes = int(len(node_labels_compact))
    n_edges = int(len(source_ids))

    # ------------------------------------------------------------------
    # 5. Build lightweight lookup table (str(id) → original label)
    # ------------------------------------------------------------------
    lookup_table: dict[str, str] = {
        str(i): str(label) for i, label in enumerate(node_labels_compact)
    }

    # ------------------------------------------------------------------
    # 6. Aggregate indicator columns per source node
    # ------------------------------------------------------------------
    node_indicators: dict[str, np.ndarray] = {}
    if indicator_cols:
        node_indicators = _aggregate_indicators(
            df, indicator_cols, source_ids, n_nodes
        )

    # ------------------------------------------------------------------
    # 7. Source label edge counts (for diagnostics)
    # ------------------------------------------------------------------
    src_id_series = pd.Series(source_ids)
    src_counts_raw = src_id_series.value_counts().to_dict()
    source_label_counts: dict[str, int] = {
        lookup_table.get(str(k), str(k)): int(v)
        for k, v in src_counts_raw.items()
    }

    # ------------------------------------------------------------------
    # 8. Edge hash for cache keying
    # ------------------------------------------------------------------
    edge_hash = _compute_edge_hash(source_ids, target_ids)

    return IngestResult(
        source_ids=source_ids,
        target_ids=target_ids,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_indicators=node_indicators,
        lookup_table=lookup_table,
        edge_hash=edge_hash,
        composite_key_col=composite_key_col,
        source_label_counts=source_label_counts,
    )


def save_lookup_table(lookup_table: dict[str, str], path: str | Path) -> None:
    """Persist the lookup table to a JSON file for UI reporting."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lookup_table, fh, indent=2)


def load_lookup_table(path: str | Path) -> dict[str, str]:
    """Load a previously saved lookup table from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
