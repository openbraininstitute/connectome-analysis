"""Classical clustering coefficient ``ccc`` in ``connalysis.network.classic``."""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, planted, weighted

from cases.classic_inputs import complete, from_edges, hub_graph, with_explicit_zeros
from connalysis.network import classic as C


def _clustered(n=40, seed=1):
    # hubs, reciprocal pairs and triangles of every kind
    return hub_graph(n, seed=seed, mean_degree=6.0, reciprocal=0.5)


@case("classic.ccc.sparse_input")
def _():
    return C.ccc(_clustered())


@case("classic.ccc.dense_input")
def _():
    return C.ccc(_clustered().toarray())


@case("classic.ccc.larger_sparse_graph_with_planted_cliques")
def _():
    return C.ccc(planted(100, 0.05, seed=2, sizes=(8, 6, 5, 4)))


@case("classic.ccc.larger_dense_graph_with_planted_cliques")
def _():
    return C.ccc(planted(100, 0.05, seed=2, sizes=(8, 6, 5, 4)).toarray())


@case("classic.ccc.sparse_formats")
def _():
    base = _clustered(30, seed=3)
    return [
        C.ccc(m)
        for m in (base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float))
    ]


@case("classic.ccc.dense_dtypes")
def _():
    base = _clustered(30, seed=3).toarray()
    return [C.ccc(base.astype(int)), C.ccc(base.astype(float)), C.ccc(base.astype(np.int8))]


@case("classic.ccc.weights_are_binarised")
def _():
    adj = weighted(30, 0.15, seed=4)
    return [C.ccc(adj), C.ccc(adj.toarray()), C.ccc(adj.astype(bool))]


@case("classic.ccc.nodes_with_degree_below_two")
def _():
    # node 0: isolated; 1: a single out-edge; 2: a single in-edge; 3, 4, 5, 6: part of a reciprocal triangle with a tail
    edges = [(1, 3), (4, 2), (3, 4), (4, 3), (4, 5), (5, 4), (3, 5), (5, 3), (5, 6)]
    adj = from_edges(8, edges)
    return [C.ccc(adj), C.ccc(adj.toarray())]


@case("classic.ccc.path_graph")
def _():
    adj = from_edges(4, [(0, 1), (1, 2)])
    return [C.ccc(adj), C.ccc(adj.toarray())]


@case("classic.ccc.cyclic_and_transitive_triangles")
def _():
    cyclic = from_edges(3, [(0, 1), (1, 2), (2, 0)])
    transitive = from_edges(3, [(0, 1), (0, 2), (1, 2)])
    return [C.ccc(m) for m in (cyclic, transitive, cyclic.toarray(), transitive.toarray())]


@case("classic.ccc.fully_reciprocal_graph")
def _():
    # symmetric graph: the classical undirected clustering coefficient
    base = digraph(30, 0.15, seed=5)
    sym = (base + base.T).astype(bool)
    return [C.ccc(sym), C.ccc(sym.toarray())]


@case("classic.ccc.complete_graph")
def _():
    return [C.ccc(complete(5)), C.ccc(complete(5).toarray())]


@case("classic.ccc.no_edges")
def _():
    return [C.ccc(sp.csr_matrix((5, 5), dtype=bool)), C.ccc(np.zeros((5, 5), dtype=bool))]


@case("classic.ccc.single_node")
def _():
    return [C.ccc(sp.csr_matrix((1, 1), dtype=bool)), C.ccc(np.zeros((1, 1), dtype=bool))]


@case("classic.ccc.two_nodes")
def _():
    return [C.ccc(from_edges(2, e)) for e in ([(0, 1)], [(0, 1), (1, 0)])]


@case("classic.ccc.self_loops")
def _():
    base = _clustered(30, seed=3).tolil()
    for i in (0, 4, 11):
        base[i, i] = True
    base = base.tocsr()
    return [C.ccc(base), C.ccc(base.toarray())]


@case("classic.ccc.explicit_zeros")
def _():
    adj = with_explicit_zeros(_clustered(30, seed=3), every=3)
    return [C.ccc(adj), C.ccc(adj.toarray())]


@case("classic.ccc.non_square")
def _():
    return C.ccc(sp.csr_matrix(np.ones((3, 4), dtype=bool)))
