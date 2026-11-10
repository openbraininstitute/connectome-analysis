"""Matrix helpers and degree/simplex counting in ``connalysis.network.topology``."""

import numpy as np
from support.cases import case
from support.data import digraph, planted, weighted

from connalysis.network import topology as T


@case("topology.rc_submatrix.bool")
def _():
    return T.rc_submatrix(digraph(40, 0.15, seed=1, reciprocal=0.5))


@case("topology.rc_submatrix.weighted")
def _():
    return T.rc_submatrix(weighted(40, 0.2, seed=2, integers=False))


@case("topology.rc_submatrix.dense_input")
def _():
    return T.rc_submatrix(digraph(15, 0.3, seed=3, reciprocal=0.4).toarray())


@case("topology.underlying_undirected_matrix.csc_input")
def _():
    return T.underlying_undirected_matrix(digraph(30, 0.1, seed=4, fmt="csc"))


@case("topology.node_degree.bool_total")
def _():
    return T.node_degree(digraph(50, 0.1, seed=5))


@case("topology.node_degree.in_out")
def _():
    adj = digraph(50, 0.1, seed=5)
    return [T.node_degree(adj, direction=d) for d in ("IN", "OUT", ("IN", "OUT"), ("OUT", "IN"))]


@case("topology.node_degree.weighted")
def _():
    return T.node_degree(weighted(30, 0.2, seed=6), direction="IN", weighted=True)


@case("topology.node_degree.dense_input")
def _():
    return T.node_degree(digraph(20, 0.2, seed=7).toarray(), direction="OUT")


@case("topology.node_degree.bad_direction")
def _():
    return T.node_degree(digraph(10, 0.2, seed=8), direction="SIDEWAYS")


@case("topology.simplex_counts.directed")
def _():
    return T.simplex_counts(planted(60, 0.06, seed=9))


@case("topology.simplex_counts.types")
def _():
    adj = planted(40, 0.08, seed=10, sizes=(5, 4))
    return {
        t: T.simplex_counts(adj, simplex_type=t) for t in ("directed", "undirected", "reciprocal")
    }


@case("topology.simplex_counts.max_simplices_and_max_dim")
def _():
    adj = planted(40, 0.08, seed=11)
    return [T.simplex_counts(adj, max_simplices=True), T.simplex_counts(adj, max_dim=2)]


@case("topology.simplex_counts.source_nodes")
def _():
    return T.simplex_counts(planted(40, 0.08, seed=12), nodes=np.arange(0, 40, 3))


@case("topology.simplex_counts.diagonal_rejected")
def _():
    adj = digraph(10, 0.3, seed=13).tolil()
    adj[2, 2] = True
    return T.simplex_counts(adj)
