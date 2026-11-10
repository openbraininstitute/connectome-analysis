"""``neighbours`` and ``neighbourhood`` in ``connalysis.network.local`` (the "old code" section).

They are written for dense arrays (``topology.tcc``/``dc`` call ``neighbourhood`` with one).  With sparse input
``np.nonzero(matrix[v])[0]`` is the row index of the non-zero entries of a 1 x N matrix, i.e. zeros, so the
"neighbours" of any connected node are just ``[v, 0]``.  Those oddities are pinned, not endorsed.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, planted, tutorial_connectome, weighted

from connalysis.network import local as L


def _small(dtype=bool):
    """10 nodes with reciprocal pairs and no loops; node 8 is isolated."""
    dense = digraph(10, 0.25, seed=2, reciprocal=0.3).toarray()
    dense[8, :] = False
    dense[:, 8] = False
    return dense.astype(dtype)


def _loop_graph(fmt="csr"):
    """Weights chosen so that the 2 x 2 matrix on [v, 0] is informative: 0 <-> 3, a loop at 0 and at 3."""
    dense = np.zeros((6, 6))
    dense[0, 3], dense[3, 0], dense[3, 3], dense[0, 0] = 5.0, 2.0, 7.0, 9.0
    dense[1, 3], dense[3, 4], dense[2, 1] = 1.0, 3.0, 4.0
    return sp.csr_matrix(dense).asformat(fmt)


# =============================================================================================== neighbours


@case("local.neighbours.dense_nodes")
def _():
    dense = _small()
    return {v: L.neighbours(v, dense) for v in range(10)}


@case("local.neighbours.keyword_arguments")
def _():
    return L.neighbours(v=3, matrix=_small())


@case("local.neighbours.dense_dtypes")
def _():
    base = _small()
    wts = weighted(10, 0.25, seed=2).toarray()
    return {
        "bool": L.neighbours(3, base),
        "int": L.neighbours(3, base.astype(int)),
        "float": L.neighbours(3, base.astype(float)),
        "weighted": L.neighbours(3, wts),
        "negative": L.neighbours(3, -2.0 * base),
    }


@case("local.neighbours.dense_self_loop")
def _():
    # the center is listed twice when it has a loop
    dense = _small()
    dense[3, 3] = True
    return L.neighbours(3, dense)


@case("local.neighbours.dense_isolated")
def _():
    return L.neighbours(8, _small())


@case("local.neighbours.dense_source_and_sink")
def _():
    dense = np.zeros((6, 6), dtype=bool)
    dense[0, [1, 2, 3]] = True
    dense[[1, 4], 5] = True
    return {
        "source": L.neighbours(0, dense),
        "sink": L.neighbours(5, dense),
        "middle": L.neighbours(1, dense),
    }


@case("local.neighbours.dense_matrix_class")
def _():
    # for a np.matrix matrix[v] is 1 x N, so the same oddity as with sparse input appears
    return L.neighbours(3, np.matrix(_small()))


@case("local.neighbours.sparse_formats")
def _():
    # defect: only [v, 0] (v and a spurious 0) comes out for a connected node, [v] for an isolated one
    adj = sp.csr_matrix(_small())
    out = {}
    for fmt in ("csr", "csc", "lil", "dok"):
        out[fmt] = {v: L.neighbours(v, adj.asformat(fmt)) for v in (0, 3, 8)}
    return out


@case("local.neighbours.sparse_weighted")
def _():
    return {v: L.neighbours(v, weighted(10, 0.25, seed=2)) for v in (1, 6, 9)}


@case("local.neighbours.coo_input")
def _():
    # coo matrices cannot be indexed
    return L.neighbours(3, sp.coo_matrix(_small()))


@case("local.neighbours.bsr_input")
def _():
    return L.neighbours(3, sp.bsr_matrix(_small()))


@case("local.neighbours.node_outside_matrix")
def _():
    return L.neighbours(12, _small())


@case("local.neighbours.node_as_list")
def _():
    return L.neighbours([3], _small())


@case("local.neighbours.tutorial_dense")
def _():
    adj, _nodes = tutorial_connectome(300)
    dense = adj.toarray()
    return {v: L.neighbours(v, dense) for v in (0, 17, 150)}


# =============================================================================================== neighbourhood


@case("local.neighbourhood.dense_nodes")
def _():
    dense = _small()
    return {v: L.neighbourhood(v, dense) for v in (0, 3, 4, 9)}


@case("local.neighbourhood.keyword_arguments")
def _():
    return L.neighbourhood(v=3, matrix=_small())


@case("local.neighbourhood.dense_weighted")
def _():
    return {
        v: L.neighbourhood(v, weighted(10, 0.3, seed=51, integers=False).toarray()) for v in (2, 7)
    }


@case("local.neighbourhood.dense_planted")
def _():
    # the way topology.tcc and topology.dc use it: a dense matrix, the center comes first
    dense = planted(30, 0.08, seed=52, sizes=(5, 4)).toarray()
    return {v: L.neighbourhood(v, dense) for v in (1, 8, 20)}


@case("local.neighbourhood.tutorial_dense")
def _():
    adj, _nodes = tutorial_connectome(300)
    dense = adj.toarray()
    return {v: L.neighbourhood(v, dense) for v in (0, 17, 150, 299)}


@case("local.neighbourhood.dense_isolated_and_loop")
def _():
    dense = _small()
    looped = dense.copy()
    looped[3, 3] = True
    return {"isolated": L.neighbourhood(8, dense), "self_loop": L.neighbourhood(3, looped)}


@case("local.neighbourhood.dense_matrix_class")
def _():
    return L.neighbourhood(3, np.matrix(_small()))


@case("local.neighbourhood.sparse_formats")
def _():
    # the sub-matrix is taken on [v, 0] with fancy indexing, in the format of the input
    return {fmt: L.neighbourhood(3, _loop_graph(fmt)) for fmt in ("csr", "csc", "lil", "dok")}


@case("local.neighbourhood.sparse_node_zero")
def _():
    # for v = 0 the "neighbours" are [0, 0]
    return L.neighbourhood(0, _loop_graph("csr"))


@case("local.neighbourhood.sparse_isolated_node")
def _():
    return L.neighbourhood(5, _loop_graph("csr"))


@case("local.neighbourhood.sparse_bool")
def _():
    adj = _loop_graph("csr").astype(bool)
    return {3: L.neighbourhood(3, adj), 5: L.neighbourhood(5, adj)}


@case("local.neighbourhood.coo_input")
def _():
    return L.neighbourhood(3, sp.coo_matrix(_small()))


@case("local.neighbourhood.node_outside_matrix")
def _():
    return L.neighbourhood(12, _small())
