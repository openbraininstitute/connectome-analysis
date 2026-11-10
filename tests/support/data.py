"""Deterministic inputs for the behaviour tests.

Everything is drawn from ``numpy.random.RandomState``, whose streams are frozen across numpy releases,
so the same arguments give the same input on every numpy version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"


def rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def digraph(
    n: int, p: float, seed: int = 0, *, reciprocal: float = 0.0, dtype=bool, fmt: str = "csr"
):
    """Random digraph on ``n`` nodes without self-loops.

    Every ordered pair is an edge with probability ``p``; each edge then also gets its reverse with
    probability ``reciprocal``.
    """
    r = rng(seed)
    dense = r.random_sample((n, n)) < p
    if reciprocal:
        dense |= dense.T & (r.random_sample((n, n)) < reciprocal)
        dense |= (dense & (r.random_sample((n, n)) < reciprocal)).T
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


def planted(
    n: int, p: float, seed: int = 0, sizes=(6, 5, 4, 4, 3), *, dtype=bool, fmt: str = "csr"
):
    """``digraph(n, p)`` plus feedforward cliques of the given sizes, so higher-dimensional simplices exist."""
    r = rng(seed + 1000)
    dense = digraph(n, p, seed).toarray()
    for size in sizes:
        nodes = r.permutation(n)[:size]
        for i in range(size):
            for j in range(i + 1, size):
                dense[nodes[i], nodes[j]] = True
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


def weighted(n: int, p: float, seed: int = 0, *, integers: bool = True, fmt: str = "csr"):
    """Random weighted digraph; integer weights 1..5 (stored as float64) or uniform reals in (0, 1]."""
    base = digraph(n, p, seed)
    r = rng(seed + 2000)
    weights = (
        r.randint(1, 6, size=base.nnz).astype(float)
        if integers
        else 1.0 - r.random_sample(base.nnz)
    )
    out = sp.csr_matrix((weights, base.indices.copy(), base.indptr.copy()), shape=base.shape)
    return out.asformat(fmt)


def positions(n: int, seed: int = 0, dims: int = 3, scale: float = 500.0) -> np.ndarray:
    return rng(seed + 3000).random_sample((n, dims)) * scale


def node_table(n: int, seed: int = 0, n_mtypes: int = 4) -> pd.DataFrame:
    """Neuron properties: coordinates, depth, layer and a categorical-like ``mtype`` column."""
    r = rng(seed + 4000)
    xyz = r.random_sample((n, 3)) * 500.0
    return pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "depth": np.round(xyz[:, 1] / 10.0),
            "layer": r.randint(1, 7, size=n),
            "mtype": np.array([f"m{i}" for i in range(n_mtypes)])[r.randint(0, n_mtypes, size=n)],
        }
    )


def tutorial_connectome(n: int | None = None):
    """The real connectome from ``tutorials/`` (6719 neurons), optionally its first ``n`` neurons."""
    adj = sp.load_npz(TUTORIALS / "adj_r0c10.npz").tocsr()
    nodes = pd.read_hdf(TUTORIALS / "nodes_r0c10.h5")
    if n is not None:
        adj, nodes = adj[:n, :n].tocsr(), nodes.iloc[:n]
    return adj, nodes.reset_index(drop=True)
