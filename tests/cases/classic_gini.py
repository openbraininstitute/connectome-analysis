"""Gini curve and coefficient in ``connalysis.network.classic``."""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import node_table, planted, weighted

from cases.classic_inputs import (
    complete,
    from_edges,
    hub_graph,
    regular_digraph,
    with_explicit_zeros,
)
from connalysis.network import classic as C

DIRECTIONS = ("efferent", "afferent")


def _nonsquare(seed=1, shape=(8, 12), p=0.3):
    return sp.csr_matrix(np.random.RandomState(seed).random_sample(shape) < p)


# ----------------------------------------------------------------------------- gini_curve


@case("classic.gini_curve.efferent_is_not_sorted")
def _():
    # for the out-degree the curve is a cumulative sum in node order: sorted() does not sort a 2-d array
    return C.gini_curve(hub_graph(50, seed=1), None, direction="efferent")


@case("classic.gini_curve.afferent_is_sorted")
def _():
    return C.gini_curve(hub_graph(50, seed=1), None, direction="afferent")


@case("classic.gini_curve.default_is_efferent")
def _():
    adj = planted(40, 0.06, seed=2, sizes=(6, 4))
    return [C.gini_curve(adj, None), C.gini_curve(adj, None, "efferent")]


@case("classic.gini_curve.neuron_properties_are_ignored")
def _():
    adj = planted(40, 0.06, seed=2, sizes=(6, 4))
    return [C.gini_curve(adj, node_table(40), direction=d) for d in DIRECTIONS]


@case("classic.gini_curve.weighted_matrix")
def _():
    adj = weighted(40, 0.1, seed=3)
    return {d: C.gini_curve(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.real_weights")
def _():
    adj = weighted(40, 0.1, seed=3, integers=False)
    return {d: C.gini_curve(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.input_formats")
def _():
    base = hub_graph(30, seed=4)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float)]
    return [[C.gini_curve(m, None, direction=d) for d in DIRECTIONS] for m in inputs]


@case("classic.gini_curve.sparse_array_input")
def _():
    # with a sparse array (not a matrix) the out-degree sum is 1-d and sorted() does sort it
    adj = sp.csr_array(hub_graph(50, seed=1))
    return {d: C.gini_curve(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.non_square_matrix")
def _():
    return {d: C.gini_curve(_nonsquare(), None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.explicit_zeros")
def _():
    adj = with_explicit_zeros(hub_graph(30, seed=4), every=3)
    return {d: C.gini_curve(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.equal_degrees")
def _():
    # a graph where every node has the same in- and out-degree: the curve is a straight line
    n = 12
    ring = sp.csr_matrix(
        sum(np.eye(n, k=0)[np.roll(np.arange(n), k)] for k in (1, 2, 5)).astype(bool)
    )
    return {d: C.gini_curve(ring, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.complete_graph")
def _():
    return {d: C.gini_curve(complete(6), None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.single_hub")
def _():
    # one node sends to every other node
    adj = from_edges(8, [(0, j) for j in range(1, 8)])
    return {d: C.gini_curve(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.no_edges")
def _():
    return {
        d: C.gini_curve(sp.csr_matrix((5, 5), dtype=bool), None, direction=d) for d in DIRECTIONS
    }


@case("classic.gini_curve.single_node")
def _():
    return {
        d: C.gini_curve(sp.csr_matrix((1, 1), dtype=bool), None, direction=d) for d in DIRECTIONS
    }


@case("classic.gini_curve.two_nodes")
def _():
    return {d: C.gini_curve(from_edges(2, [(0, 1)]), None, direction=d) for d in DIRECTIONS}


@case("classic.gini_curve.bad_direction")
def _():
    return C.gini_curve(hub_graph(20, seed=5), None, direction="sideways")


@case("classic.gini_curve.dense_input", tags=("fixed",))
def _():
    return C.gini_curve(hub_graph(20, seed=5).toarray(), None)


# ----------------------------------------------------------------------------- gini_coefficient


@case("classic.gini_coefficient.directions")
def _():
    adj = hub_graph(50, seed=1)
    return {d: C.gini_coefficient(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_coefficient.default_is_efferent")
def _():
    adj = planted(40, 0.06, seed=2, sizes=(6, 4))
    return [C.gini_coefficient(adj, None), C.gini_coefficient(adj, None, "efferent")]


@case("classic.gini_coefficient.weighted_matrix")
def _():
    adj = weighted(40, 0.1, seed=3)
    return {d: C.gini_coefficient(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_coefficient.input_formats")
def _():
    base = hub_graph(30, seed=4)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float)]
    return [[C.gini_coefficient(m, None, direction=d) for d in DIRECTIONS] for m in inputs]


@case("classic.gini_coefficient.non_square_matrix")
def _():
    return {d: C.gini_coefficient(_nonsquare(), None, direction=d) for d in DIRECTIONS}


@case("classic.gini_coefficient.complete_graph")
def _():
    return {d: C.gini_coefficient(complete(6), None, direction=d) for d in DIRECTIONS}


@case("classic.gini_coefficient.single_hub")
def _():
    adj = from_edges(8, [(0, j) for j in range(1, 8)])
    return {d: C.gini_coefficient(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.gini_coefficient.no_edges")
def _():
    return {
        d: C.gini_coefficient(sp.csr_matrix((5, 5), dtype=bool), None, direction=d)
        for d in DIRECTIONS
    }


@case("classic.gini_coefficient.single_node")
def _():
    return {
        d: C.gini_coefficient(sp.csr_matrix((1, 1), dtype=bool), None, direction=d)
        for d in DIRECTIONS
    }


@case("classic.gini_coefficient.bad_direction")
def _():
    return C.gini_coefficient(hub_graph(20, seed=5), None, direction="sideways")


# ----------------------------------------------------------------------------- _analytical_expected_gini_curve


@case("classic.analytical_expected_gini_curve.directions")
def _():
    adj = hub_graph(50, seed=1, mean_degree=8.0)
    return {d: C._analytical_expected_gini_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.analytical_expected_gini_curve.default_is_efferent")
def _():
    adj = hub_graph(40, seed=6, mean_degree=8.0)
    return [
        C._analytical_expected_gini_curve(adj),
        C._analytical_expected_gini_curve(adj, "efferent"),
    ]


@case("classic.analytical_expected_gini_curve.dense_graph")
def _():
    # a high connection probability puts the binomial mass in the middle of the range
    adj = hub_graph(30, seed=7, mean_degree=15.0)
    return {d: C._analytical_expected_gini_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.analytical_expected_gini_curve.non_square_matrix")
def _():
    return {d: C._analytical_expected_gini_curve(_nonsquare(), direction=d) for d in DIRECTIONS}


@case("classic.analytical_expected_gini_curve.weights_do_not_matter")
def _():
    adj = weighted(40, 0.2, seed=8)
    return [
        C._analytical_expected_gini_curve(adj, direction="efferent"),
        C._analytical_expected_gini_curve(adj.astype(bool), direction="efferent"),
    ]


@case("classic.analytical_expected_gini_curve.stored_zeros_count_as_edges")
def _():
    adj = with_explicit_zeros(hub_graph(30, seed=7, mean_degree=10.0), every=2)
    return {d: C._analytical_expected_gini_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.analytical_expected_gini_curve.input_formats")
def _():
    base = hub_graph(30, seed=7, mean_degree=10.0)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int)]
    return [C._analytical_expected_gini_curve(m, direction="afferent") for m in inputs]


@case("classic.analytical_expected_gini_curve.complete_graph")
def _():
    return {d: C._analytical_expected_gini_curve(complete(6), direction=d) for d in DIRECTIONS}


@case("classic.analytical_expected_gini_curve.no_edges")
def _():
    return {
        d: C._analytical_expected_gini_curve(sp.csr_matrix((5, 5), dtype=bool), direction=d)
        for d in DIRECTIONS
    }


@case("classic.analytical_expected_gini_curve.two_nodes")
def _():
    return {
        d: C._analytical_expected_gini_curve(from_edges(2, [(0, 1)]), direction=d)
        for d in DIRECTIONS
    }


@case("classic.analytical_expected_gini_curve.single_node", tags=("fixed",))
def _():
    return C._analytical_expected_gini_curve(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.analytical_expected_gini_curve.bad_direction", tags=("fixed",))
def _():
    # the original fails with an UnboundLocalError instead of reporting the direction
    return C._analytical_expected_gini_curve(hub_graph(20, seed=5), direction="sideways")


@case("classic.analytical_expected_gini_curve.dense_input", tags=("fixed",))
def _():
    return C._analytical_expected_gini_curve(hub_graph(20, seed=5).toarray())


# ----------------------------------------------------------------------------- normalized_gini_coefficient


@case("classic.normalized_gini_coefficient.directions")
def _():
    adj = hub_graph(50, seed=1, mean_degree=8.0)
    return {d: C.normalized_gini_coefficient(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.normalized_gini_coefficient.default_is_efferent")
def _():
    adj = hub_graph(40, seed=6, mean_degree=8.0)
    return [
        C.normalized_gini_coefficient(adj, None),
        C.normalized_gini_coefficient(adj, None, "efferent"),
    ]


@case("classic.normalized_gini_coefficient.heavy_tailed_graph")
def _():
    adj = planted(60, 0.05, seed=9, sizes=(10, 8, 6))
    return {d: C.normalized_gini_coefficient(adj, node_table(60), direction=d) for d in DIRECTIONS}


@case("classic.normalized_gini_coefficient.non_square_matrix")
def _():
    return {d: C.normalized_gini_coefficient(_nonsquare(), None, direction=d) for d in DIRECTIONS}


@case("classic.normalized_gini_coefficient.weighted_matrix")
def _():
    adj = weighted(40, 0.2, seed=8)
    return {d: C.normalized_gini_coefficient(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.normalized_gini_coefficient.regular_graph")
def _():
    adj = regular_digraph(30, 4, seed=10)
    return {d: C.normalized_gini_coefficient(adj, None, direction=d) for d in DIRECTIONS}


@case("classic.normalized_gini_coefficient.complete_graph")
def _():
    return {d: C.normalized_gini_coefficient(complete(6), None, direction=d) for d in DIRECTIONS}


@case("classic.normalized_gini_coefficient.two_nodes")
def _():
    return {
        d: C.normalized_gini_coefficient(from_edges(2, [(0, 1)]), None, direction=d)
        for d in DIRECTIONS
    }


@case("classic.normalized_gini_coefficient.no_edges")
def _():
    return {
        d: C.normalized_gini_coefficient(sp.csr_matrix((5, 5), dtype=bool), None, direction=d)
        for d in DIRECTIONS
    }


@case("classic.normalized_gini_coefficient.single_node", tags=("fixed",))
def _():
    return C.normalized_gini_coefficient(sp.csr_matrix((1, 1), dtype=bool), None)


@case("classic.normalized_gini_coefficient.bad_direction")
def _():
    return C.normalized_gini_coefficient(hub_graph(20, seed=5), None, direction="sideways")
