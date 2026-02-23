"""
Graph generation utilities for the cascade propagation engine.

Uses NetworkX only for graph construction; all outputs are NumPy adjacency
matrices.  Convention: A[j, i] = 1 means there is a directed edge j → i,
so column i of A contains the in-neighbors of node i.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _digraph_to_adjacency(G: nx.DiGraph, n: int) -> np.ndarray:
    """Convert a NetworkX DiGraph to a NumPy adjacency matrix.

    The matrix is stored with convention A[j, i] = 1 iff edge j → i exists,
    i.e. column i lists in-neighbours of node i.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph with integer node labels 0..n-1.
    n : int
        Expected number of nodes.

    Returns
    -------
    np.ndarray, shape (n, n), dtype uint8
        Adjacency matrix with the j→i convention.
    """
    # nx.to_numpy_array returns A_nx where A_nx[u, v] = 1 iff edge u→v.
    # We want A[j, i] = 1 iff j→i, which is the same layout, so no transpose.
    A_nx = nx.to_numpy_array(G, nodelist=list(range(n)), dtype=np.uint8)
    return A_nx


def _undirected_to_directed_adjacency(G: nx.Graph, n: int) -> np.ndarray:
    """Convert an undirected NetworkX graph to a symmetric directed adjacency matrix.

    Each undirected edge {u, v} becomes two directed edges u→v and v→u.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph.
    n : int
        Number of nodes.

    Returns
    -------
    np.ndarray, shape (n, n), dtype uint8
        Symmetric adjacency matrix.
    """
    return _digraph_to_adjacency(G.to_directed(), n)


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------


def generate_erdos_renyi(n: int, p: float, seed: int) -> np.ndarray:
    """Generate an Erdős–Rényi directed random graph G(n, p).

    Each directed edge (u, v) exists independently with probability p.
    Self-loops are excluded.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability in [0, 1].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, n), dtype uint8
        Adjacency matrix A where A[j, i] = 1 iff j → i.
    """
    G: nx.DiGraph = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    A = _digraph_to_adjacency(G, n)
    np.fill_diagonal(A, 0)
    return A


def generate_barabasi_albert(n: int, m: int, seed: int) -> np.ndarray:
    """Generate a Barabási–Albert preferential attachment graph.

    NetworkX produces an undirected BA graph; edges are symmetrised into a
    directed graph (each undirected edge becomes two directed edges).

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges to attach per new node (m ≥ 1, m < n).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, n), dtype uint8
        Symmetric adjacency matrix.
    """
    G: nx.Graph = nx.barabasi_albert_graph(n, m, seed=seed)
    A = _undirected_to_directed_adjacency(G, n)
    np.fill_diagonal(A, 0)
    return A


def generate_watts_strogatz(n: int, k: int, p: float, seed: int) -> np.ndarray:
    """Generate a Watts–Strogatz small-world graph.

    The undirected WS graph is symmetrised into a directed graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is joined to k nearest neighbours in the ring lattice.
    p : float
        Probability of rewiring each edge.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, n), dtype uint8
        Symmetric adjacency matrix.
    """
    G: nx.Graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
    A = _undirected_to_directed_adjacency(G, n)
    np.fill_diagonal(A, 0)
    return A


def generate_custom(n: int, edge_list: Sequence[tuple[int, int]]) -> np.ndarray:
    """Generate a graph from an explicit edge list.

    Parameters
    ----------
    n : int
        Number of nodes (labels must be in 0..n-1).
    edge_list : sequence of (int, int)
        Directed edges as (source, target) pairs.

    Returns
    -------
    np.ndarray, shape (n, n), dtype uint8
        Adjacency matrix A where A[j, i] = 1 iff j → i.

    Raises
    ------
    ValueError
        If any node index is out of range [0, n-1].
    """
    for u, v in edge_list:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(
                f"Edge ({u}, {v}) contains a node index out of range [0, {n - 1}]."
            )

    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_list)
    A = _digraph_to_adjacency(G, n)
    np.fill_diagonal(A, 0)
    return A


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def graph_from_config(graph_cfg: dict) -> np.ndarray:
    """Build an adjacency matrix from a graph config sub-dict.

    Parameters
    ----------
    graph_cfg : dict
        Must contain ``type`` and the parameters required by the chosen
        generator.  Supported types: ``erdos_renyi``, ``barabasi_albert``,
        ``watts_strogatz``, ``custom``.

    Returns
    -------
    np.ndarray
        Adjacency matrix.

    Raises
    ------
    ValueError
        For unsupported graph types or missing parameters.
    """
    gtype = graph_cfg["type"]
    n: int = int(graph_cfg["n"])
    seed: int = int(graph_cfg.get("seed", 0))

    if gtype == "erdos_renyi":
        return generate_erdos_renyi(n, float(graph_cfg["p"]), seed)
    if gtype == "barabasi_albert":
        return generate_barabasi_albert(n, int(graph_cfg["m"]), seed)
    if gtype == "watts_strogatz":
        return generate_watts_strogatz(
            n, int(graph_cfg["k"]), float(graph_cfg["p"]), seed
        )
    if gtype == "custom":
        return generate_custom(n, [tuple(e) for e in graph_cfg["edges"]])

    raise ValueError(f"Unsupported graph type: {gtype!r}")
