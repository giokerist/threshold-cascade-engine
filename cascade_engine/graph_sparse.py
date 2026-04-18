"""
graph_sparse.py — Sparse Graph Construction with Lazy Caching
=============================================================

Builds and caches the SciPy CSR adjacency matrix from integer edge arrays.

Key design decisions
--------------------
* **CSR format for A.T**: We store the *transpose* of the adjacency matrix
  (``A_T``) so that ``A_T.dot(failed_vec)`` directly computes the failed
  in-neighbour count for every node in a single BLAS-backed matvec call.
  Convention: ``A[j, i] = 1 iff j → i``, so ``A_T[i, j] = 1 iff j → i``
  and ``row i of A_T`` = in-neighbours of node i.

* **Lazy caching via pickle**: The (``A_T``, ``in_degree``) pair is cached to
  disk using Python's ``pickle`` module (``HIGHEST_PROTOCOL`` for speed).  The
  cache key is the 16-char ``edge_hash`` from ``IngestResult``, which is
  derived from the SHA-256 of the raw source/target integer arrays.  Changing
  only ``deg_low`` / ``fail_high`` / etc. does **not** invalidate the cache.

* **Memory layout**: CSR data and indices are stored as ``float32`` /
  ``int32`` (not float64 / int64) to halve the working-set size for
  large graphs.  The Numba kernel in ``propagation_fast.py`` accepts both.

* **Thread safety**: Cache reads are safe for concurrent readers.  A
  write-then-rename pattern prevents corrupt reads if two processes build
  the same graph simultaneously.

Typical use
-----------
>>> from cascade_engine.ingestion import ingest_edgelist
>>> from cascade_engine.graph_sparse import build_or_load_sparse_graph
>>> ir = ingest_edgelist("edges.csv")
>>> A_T, in_degree = build_or_load_sparse_graph(
...     ir.source_ids, ir.target_ids, ir.n_nodes, edge_hash=ir.edge_hash
... )
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

# Optional: joblib.Memory for function-level memoisation
try:
    from joblib import Memory as _JoblibMemory
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SparseGraphTuple = Tuple[csr_matrix, np.ndarray]
"""(A_T, in_degree): A_T is a CSR matrix; in_degree is float64 array."""

_DEFAULT_CACHE_DIR = ".cascade_cache"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_sparse_graph(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    n_nodes: int,
) -> SparseGraphTuple:
    """Build a CSR sparse graph from integer edge arrays.

    Constructs ``A_T = A.T`` in Compressed Sparse Row format, where
    ``A[j, i] = 1 iff j → i`` (the engine's convention).  Row ``i`` of
    ``A_T`` therefore lists the in-neighbours of node ``i``.

    Parameters
    ----------
    source_ids : np.ndarray, shape (E,), dtype int-like
        Integer IDs for the source (upstream) node of each edge.
    target_ids : np.ndarray, shape (E,), dtype int-like
        Integer IDs for the target (downstream) node of each edge.
    n_nodes : int
        Number of nodes (= max ID + 1).

    Returns
    -------
    A_T : csr_matrix, shape (n_nodes, n_nodes)
        Transpose of the adjacency matrix in CSR.  ``A_T.dot(v)`` computes
        in-neighbour weighted sums for every node simultaneously.
    in_degree : np.ndarray, shape (n_nodes,), dtype float64
        In-degree of each node.  Used for normalisation (F_i = count / D_i).

    Notes
    -----
    The CSR ``data`` array is ``float32`` (all 1.0) to save memory.  The
    Numba kernel in ``propagation_fast.py`` operates on the raw arrays
    ``A_T.data``, ``A_T.indices``, ``A_T.indptr`` directly.

    Duplicate edges are eliminated by ``csr_matrix`` construction via
    ``sum_duplicates()``, which sums weights — for a binary graph, any
    weight > 0 indicates presence, so this is equivalent to deduplication.
    For clean input from ``ingest_edgelist`` (which already deduplicates),
    this is a no-op.
    """
    src = np.asarray(source_ids, dtype=np.int32)
    tgt = np.asarray(target_ids, dtype=np.int32)
    n = int(n_nodes)

    if src.shape != tgt.shape:
        raise ValueError(
            f"source_ids and target_ids must have the same shape; "
            f"got {src.shape} vs {tgt.shape}"
        )
    if len(src) == 0:
        raise ValueError("Edge arrays are empty — cannot build graph.")

    # Bounds check
    max_id = max(int(src.max()), int(tgt.max()))
    if max_id >= n:
        raise ValueError(
            f"Node ID {max_id} out of range [0, {n - 1}] (n_nodes={n})."
        )

    # Build A_T in CSR: A_T[i, j] = 1 iff j→i (i = row = destination)
    # scipy csr_matrix((data, (row, col)), shape) where row=tgt, col=src
    data = np.ones(len(src), dtype=np.float32)
    A_T = csr_matrix((data, (tgt, src)), shape=(n, n), dtype=np.float32)
    A_T.sum_duplicates()    # collapse any remaining duplicate edges
    A_T.eliminate_zeros()   # strip any 0 entries
    A_T.sort_indices()      # required for Numba CSR traversal

    # in_degree[i] = number of in-neighbours of node i = row nnz of A_T
    in_degree = np.diff(A_T.indptr).astype(np.float64)

    return A_T, in_degree


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, edge_hash: str) -> Path:
    return cache_dir / f"graph_{edge_hash}.pkl"


def _atomic_write(path: Path, obj: object) -> None:
    """Write *obj* to *path* atomically using a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file first, then rename (atomic on POSIX)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)          # atomic on Linux
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def cache_sparse_graph(
    A_T: csr_matrix,
    in_degree: np.ndarray,
    edge_hash: str,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> Path:
    """Persist a sparse graph to disk.

    Parameters
    ----------
    A_T : csr_matrix
        Transpose adjacency matrix.
    in_degree : np.ndarray
        In-degree array.
    edge_hash : str
        16-char hex key (from ``IngestResult.edge_hash``).
    cache_dir : str or Path, optional
        Directory for cache files.  Created if absent.

    Returns
    -------
    Path
        Path to the written cache file.
    """
    path = _cache_path(Path(cache_dir), edge_hash)
    payload = {"A_T": A_T, "in_degree": in_degree, "edge_hash": edge_hash}
    _atomic_write(path, payload)
    return path


def load_sparse_graph(
    edge_hash: str,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> SparseGraphTuple | None:
    """Load a cached sparse graph, or return ``None`` if not found.

    Parameters
    ----------
    edge_hash : str
        16-char hex key produced by ``IngestResult.edge_hash``.
    cache_dir : str or Path, optional
        Cache directory to search.

    Returns
    -------
    (A_T, in_degree) or None
    """
    path = _cache_path(Path(cache_dir), edge_hash)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload["A_T"], payload["in_degree"]
    except (pickle.UnpicklingError, KeyError, EOFError):
        # Corrupt or truncated file — delete and return None so caller rebuilds
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def build_or_load_sparse_graph(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    n_nodes: int,
    edge_hash: str | None = None,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> SparseGraphTuple:
    """Build or load the sparse graph, using the on-disk cache when possible.

    This is the **primary entry point** for graph construction.  If the cache
    contains a file matching *edge_hash*, the pre-built ``(A_T, in_degree)``
    pair is loaded from disk (typically <50ms for 1M-edge graphs) instead of
    being rebuilt from the raw arrays (~1–3s).

    **Cache invalidation rule**: The cache is keyed *only* by the edge
    structure (SHA-256 of source/target arrays).  Changing threshold parameters
    (``deg_low``, ``fail_high``, etc.) does **not** invalidate the cache.  Only
    adding/removing edges does.

    Parameters
    ----------
    source_ids : np.ndarray, shape (E,), dtype int-like
        Source node IDs.
    target_ids : np.ndarray, shape (E,), dtype int-like
        Target node IDs.
    n_nodes : int
        Total number of unique nodes.
    edge_hash : str or None, optional
        Pre-computed 16-char hex digest from ``IngestResult.edge_hash``.
        If *None*, a fresh hash is computed (one extra pass over the arrays).
    cache_dir : str or Path, optional
        Cache directory.  Default ``.cascade_cache`` in the working directory.
    force_rebuild : bool, optional
        If ``True``, skip the cache lookup and always rebuild.

    Returns
    -------
    A_T : csr_matrix
        Sparse adjacency transpose in CSR format.
    in_degree : np.ndarray, shape (n_nodes,), dtype float64
        In-degree per node.
    """
    if edge_hash is None:
        from .ingestion import _compute_edge_hash
        edge_hash = _compute_edge_hash(
            np.asarray(source_ids, dtype=np.int64),
            np.asarray(target_ids, dtype=np.int64),
        )

    if not force_rebuild:
        cached = load_sparse_graph(edge_hash, cache_dir=cache_dir)
        if cached is not None:
            return cached

    # Cache miss — build fresh
    A_T, in_degree = build_sparse_graph(source_ids, target_ids, n_nodes)

    # Write to cache (best-effort; don't fail if disk is read-only)
    try:
        cache_sparse_graph(A_T, in_degree, edge_hash, cache_dir=cache_dir)
    except OSError:
        pass

    return A_T, in_degree


# ---------------------------------------------------------------------------
# NetworkX compatibility shim
# ---------------------------------------------------------------------------


def sparse_from_dense(A: np.ndarray) -> SparseGraphTuple:
    """Convert a legacy dense NumPy adjacency matrix to the sparse format.

    Provided for backward compatibility with the existing ``graph.py``
    generators (Erdős–Rényi, Barabási–Albert, Watts–Strogatz) that still
    return dense matrices.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Dense adjacency matrix with convention ``A[j, i] = 1 iff j → i``.

    Returns
    -------
    (A_T, in_degree) : SparseGraphTuple
    """
    from scipy.sparse import csr_matrix as _csr

    n = A.shape[0]
    # A.T in CSR — row i = in-neighbours of i
    A_T = _csr(A.T.astype(np.float32))
    A_T.sort_indices()
    in_degree = np.array(A_T.sum(axis=1), dtype=np.float64).ravel()
    return A_T, in_degree


def graph_info(A_T: csr_matrix, in_degree: np.ndarray) -> dict:
    """Return a diagnostic summary dict for a sparse graph."""
    n = A_T.shape[0]
    nnz = A_T.nnz
    return {
        "n_nodes": n,
        "n_edges": nnz,
        "density": nnz / (n * (n - 1)) if n > 1 else 0.0,
        "mean_in_degree": float(in_degree.mean()),
        "max_in_degree": float(in_degree.max()),
        "n_isolates": int((in_degree == 0).sum()),
        "memory_mb": round(
            (A_T.data.nbytes + A_T.indices.nbytes + A_T.indptr.nbytes) / 1e6, 3
        ),
    }
