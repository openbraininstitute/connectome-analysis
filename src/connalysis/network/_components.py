# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Strongly connected components, numbered the same way whatever the version of scipy.

scipy 1.18 replaced the algorithm behind ``csgraph.connected_components(connection="strong")``.  It finds the
same components, but numbers them in a different order, which reorders the labels and the sizes derived
from them.  Up to scipy 1.17 the numbering was that of the iterative Tarjan algorithm ported below, and that is
the numbering this package has always returned.  The port is only used when the installed scipy numbers
differently, because it is a lot slower than scipy.
"""

import functools

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

# A graph on 12 nodes and the labels of its strongly connected components in scipy <= 1.17
_PROBE_EDGES = [
    (1, 2), (1, 3), (1, 4), (2, 0), (2, 10), (3, 7), (3, 11), (4, 5), (5, 1), (5, 3), (5, 7),
    (5, 9), (6, 3), (6, 5), (6, 7), (6, 10), (7, 3), (7, 8), (8, 1), (8, 3), (10, 6), (11, 0),
]  # fmt: skip
_PROBE_LABELS = [0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2]


@functools.cache
def _scipy_numbers_like_before_1_18():
    rows, cols = zip(*_PROBE_EDGES)
    probe = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(12, 12))
    labels = csgraph.connected_components(probe, directed=True, connection="strong")[1]
    return labels.tolist() == _PROBE_LABELS


def _tarjan_numbering(indptr, indices, n):
    """Port of the iterative Tarjan algorithm of scipy <= 1.17: ``(number of components, labels)``."""
    void, end = -1, -2
    indptr, indices = indptr.tolist(), indices.tolist()
    lowlinks = [void] * n  # also holds the labels of the nodes that have a component
    stack_b = [void] * n
    # Forward pointers of the DFS stack; nodes are on the other stack (SS) only after leaving this one
    stack_f = ss = [void] * n
    ss_head = end
    index, label = 0, n - 1
    for start in range(n):
        if lowlinks[start] != void:
            continue
        stack_head = start
        stack_f[start] = stack_b[start] = end
        while stack_head != end:
            v = stack_head
            if lowlinks[v] == void:
                lowlinks[v] = index
                index += 1
                for j in range(indptr[v], indptr[v + 1]):
                    w = indices[j]
                    if lowlinks[w] == void:
                        if stack_f[w] != void:  # already in the stack: move it to the top
                            f, b = stack_f[w], stack_b[w]
                            if b != end:
                                stack_f[b] = f
                            if f != end:
                                stack_b[f] = b
                        stack_f[w] = stack_head
                        stack_b[w] = end
                        stack_b[stack_head] = w
                        stack_head = w
            else:
                stack_head = stack_f[v]
                if stack_head >= 0:
                    stack_b[stack_head] = end
                stack_f[v] = stack_b[v] = void
                is_root = True
                low_v = lowlinks[v]
                for j in range(indptr[v], indptr[v + 1]):
                    low_w = lowlinks[indices[j]]
                    if low_w < low_v:
                        low_v = low_w
                        is_root = False
                lowlinks[v] = low_v
                if is_root:
                    index -= 1
                    while ss_head != end and lowlinks[v] <= lowlinks[ss_head]:
                        w = ss_head
                        ss_head = ss[w]
                        ss[w] = void
                        lowlinks[w] = label
                        index -= 1
                    lowlinks[v] = label
                    label -= 1
                else:
                    ss[v] = ss_head
                    ss_head = v
    return (n - 1) - label, ((n - 1) - np.array(lowlinks, dtype=np.int64)).astype(np.int32)


def strong_components(matrix):
    """Number and labels of the strongly connected components of a graph given by a sparse matrix."""
    if _scipy_numbers_like_before_1_18():
        return csgraph.connected_components(matrix, directed=True, connection="strong")
    graph = sp.csr_matrix(matrix, dtype=np.float64)  # as scipy prepares it
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("compressed-sparse graph must be shape (N, N)")  # the message of scipy
    return _tarjan_numbering(graph.indptr, graph.indices, graph.shape[0])
