"""Helpers shared by the ``randomization_*`` case modules (this module registers no cases).

``R``, ``RR`` and ``U`` are lazy proxies: the import happens when a case runs, so a module that the rewrite
moves breaks the cases that use it and not the collection of every case in the test suite.
"""

import importlib

import numpy as np
import scipy.sparse as sp
from support.data import digraph, positions

# Cases whose result depends on a stream that is not frozen across numpy/scipy releases (np.random.default_rng,
# scipy.sparse.random(random_state=Generator)).  A failure only in these cases hints at stream drift.
GEN = ("generator-stream",)
# Cases that depend on the bigrandomgraphs C++ generators (their own PCG32 streams, seeded per thread).
CPP = ("cpp-stream",)
GEN_CPP = GEN + CPP


class Lazy:
    """Module proxy that imports on first attribute access."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        return getattr(importlib.import_module(self._name), item)


R = Lazy("connalysis.randomization")  # the package: what users import
RR = Lazy("connalysis.randomization.randomization")  # the module the docs build
U = Lazy(
    "connalysis.randomization.rand_utils"
)  # helpers (not exported by the package, except one function)


# ----------------------------------------------------------------------------- small result summaries


def reciprocal_entries(m):
    """Number of stored entries that have their reverse edge too (twice the number of reciprocal pairs)."""
    b = sp.csr_matrix(m) != 0
    return int(b.multiply(b.T).nnz)


def undirected_edges(m):
    b = sp.csr_matrix(m) != 0
    return int((b + b.T).nnz // 2)


def shape_info(m):
    """Everything about a sparse result that a call without a (usable) seed can promise."""
    return {
        "cls": type(m).__name__,
        "shape": list(m.shape),
        "dtype": str(m.dtype),
        "diagonal_sum": float(m.diagonal().sum()),
        "data_values": sorted(set(np.unique(m.data).tolist())),
    }


def edge_info(m):
    """``shape_info`` plus the numbers of edges, which the models that fix them also promise without a seed."""
    return {
        **shape_info(m),
        "nnz": int(m.nnz),
        "reciprocal": reciprocal_entries(m),
        "undirected": undirected_edges(m),
    }


def is_superset(m, base):
    """True if every edge of ``base`` is an edge of ``m``."""
    a, b = sp.csr_matrix(m) != 0, sp.csr_matrix(base) != 0
    return bool((b.astype(int) - a.astype(int) > 0).nnz == 0)


# ----------------------------------------------------------------------------- inputs


def reciprocal_graph(n=30, p=0.15, seed=4, reciprocal=0.3, **kw):
    """Digraph with many reciprocal pairs (about two thirds of the edges)."""
    return digraph(n, p, seed=seed, reciprocal=reciprocal, **kw)


def skeleta_graph(seed=1):
    """30 nodes, a few reciprocal pairs and planted feedforward cliques: skeleta with reciprocal and one-way edges."""
    n = 30
    r = np.random.RandomState(seed + 77)
    dense = digraph(n, 0.05, seed, reciprocal=0.3).toarray()
    for size in (5, 4, 4):
        nodes = r.permutation(n)[:size]
        for i in range(size):
            for j in range(i + 1, size):
                dense[nodes[i], nodes[j]] = True
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense)


def clusters(sizes=(4, 5, 3), spacing=1000.0, seed=0):
    """Unit-cube clusters far apart along x: a step-like distance dependence then gives a deterministic graph."""
    r = np.random.RandomState(seed)
    return np.vstack(
        [r.random_sample((s, 3)) + np.array([k * spacing, 0.0, 0.0]) for k, s in enumerate(sizes)]
    )


def moderate_points(n=30, seed=1, scale=100.0):
    return positions(n, seed=seed, scale=scale)


def depth_values(n=30, seed=2):
    return positions(n, seed=seed, dims=1, scale=100.0)[:, 0]


# the "step" parameters of the exponential a * exp(-b * d): probability 1 within a cluster, 0 between clusters
STEP_A, STEP_B = 1e6, 1.0
