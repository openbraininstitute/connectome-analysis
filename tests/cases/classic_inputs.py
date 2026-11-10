"""Input builders shared by the ``classic`` cases.  This module registers no cases."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from support.data import rng


def from_edges(n, edges, dtype=bool, fmt="csr"):
    """Digraph on ``n`` nodes with the given ``(source, target)`` edges."""
    dense = np.zeros((n, n), dtype=dtype)
    for i, j in edges:
        dense[i, j] = 1
    return sp.csr_matrix(dense).asformat(fmt)


def complete(n, dtype=bool, fmt="csr"):
    """Complete digraph (every ordered pair of distinct nodes is an edge)."""
    return sp.csr_matrix((np.ones((n, n)) - np.eye(n)).astype(dtype)).asformat(fmt)


def multi_component(dtype=bool, fmt="csr"):
    """14 nodes in 6 weak components: a strongly connected 4-cycle with a chord (0-3), a reciprocal pair with a
    tail (4, 5, 6), a feed-forward triangle (7, 8, 9), two isolated nodes (10, 11) and a reciprocal pair (12, 13)."""
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (2, 0),
        (4, 5),
        (5, 4),
        (5, 6),
        (7, 8),
        (8, 9),
        (7, 9),
        (12, 13),
        (13, 12),
    ]
    return from_edges(14, edges, dtype, fmt)


def hub_graph(n, seed, mean_degree=5.0, reciprocal=0.3, dtype=bool, fmt="csr"):
    """Digraph with heavy-tailed in- and out-degrees (hubs that tend to connect to each other), no self-loops."""
    r = rng(seed)
    base = 1.0 + r.pareto(2.2, n)
    w_out = base * np.exp(0.3 * r.standard_normal(n))
    w_in = base * np.exp(0.3 * r.standard_normal(n))
    c = mean_degree / (n * w_out.mean() * w_in.mean())
    p = np.minimum(1.0, c * np.outer(w_out, w_in))
    dense = r.random_sample((n, n)) < p
    dense |= (dense & (r.random_sample((n, n)) < reciprocal)).T
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


def regular_digraph(n, k, seed, dtype=bool, fmt="csr"):
    """Every node has exactly ``k`` random out-neighbours (no self-loops), so the mean degree is exactly ``k``."""
    r = rng(seed)
    cols = np.concatenate(
        [r.choice(np.delete(np.arange(n), i), k, replace=False) for i in range(n)]
    )
    rows = np.repeat(np.arange(n), k)
    return sp.csr_matrix((np.ones(n * k, dtype=dtype), (rows, cols)), shape=(n, n)).asformat(fmt)


def spatial_digraph(
    nodes, seed, cols=("x", "y"), p0=0.6, scale=120.0, reciprocal=0.4, dtype=bool, fmt="csr"
):
    """Digraph whose connection probability decays with the distance of the nodes in ``cols``."""
    xyz = nodes[list(cols)].to_numpy()
    dist = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
    r = rng(seed)
    dense = r.random_sample(dist.shape) < p0 * np.exp(-dist / scale)
    dense |= (dense & (r.random_sample(dist.shape) < reciprocal)).T
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


def with_explicit_zeros(adj, every=3):
    """Copy of ``adj`` (csr) in which every ``every``-th stored entry is set to 0 but kept in the sparsity pattern."""
    out = sp.csr_matrix(adj, copy=True)
    out.data[::every] = 0
    assert out.nnz == adj.nnz
    return out


def with_integer_mean_degree(adj):
    """Drop the last few edges (in row order) so that the number of edges is a multiple of the number of nodes.

    The common-neighbour null model is only defined when the mean degree is an integer.
    """
    coo = adj.tocoo()
    keep = coo.nnz - coo.nnz % adj.shape[0]
    return sp.csr_matrix((coo.data[:keep], (coo.row[:keep], coo.col[:keep])), shape=adj.shape)


def common_neighbour_weights(adj, direction="efferent", seed=0):
    """Float weights ``1 + (number of common neighbours of the two ends) + noise`` on the edges of ``adj``."""
    b = adj.astype(int)
    cn = b @ b.T if direction == "efferent" else b.T @ b
    w = (adj.astype(float) + cn.multiply(adj)).tocsr()
    w.data = w.data + rng(seed).random_sample(w.nnz)
    return w


def block_scc_graph(sizes=(7, 5, 8, 4), seed=0, isolated=2):
    """Strongly connected blocks (a cycle plus random chords) chained by one feed-forward edge each, a few isolated
    nodes, with the node order shuffled."""
    r = rng(seed)
    n = sum(sizes) + isolated
    edges, blocks, start = [], [], 0
    for size in sizes:
        nodes = list(range(start, start + size))
        blocks.append(nodes)
        edges += [(nodes[i], nodes[(i + 1) % size]) for i in range(size)]
        for _ in range(size):
            a, b = r.choice(nodes, 2, replace=False)
            edges.append((int(a), int(b)))
        start += size
    edges += [(a[0], b[0]) for a, b in zip(blocks[:-1], blocks[1:])]
    perm = r.permutation(n)
    return from_edges(n, [(int(perm[i]), int(perm[j])) for i, j in edges])


def with_unsorted_indices(adj):
    """Copy of a csr/csc matrix whose indices are stored in descending order within every row/column."""
    out = adj.copy()
    for a, b in zip(out.indptr[:-1], out.indptr[1:]):
        out.indices[a:b] = out.indices[a:b][::-1].copy()
        out.data[a:b] = out.data[a:b][::-1].copy()
    out.has_sorted_indices = False
    assert not out.has_sorted_indices
    return out


def stored_state(m):
    """Copies of the three arrays a csr/csc matrix stores (to check that a call left its input untouched)."""
    return m.indptr.copy(), m.indices.copy(), m.data.copy()


def same_state(before, m):
    after = stored_state(m)
    return all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(before, after))
