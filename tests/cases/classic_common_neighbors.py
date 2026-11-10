"""Common-neighbour statistics in ``connalysis.network.classic``."""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, node_table, rng, weighted

from cases.classic_inputs import (
    common_neighbour_weights,
    complete,
    from_edges,
    hub_graph,
    regular_digraph,
    with_explicit_zeros,
    with_integer_mean_degree,
)
from connalysis.network import classic as C

DIRECTIONS = ("efferent", "afferent")

# ----------------------------------------------------------------------------- distribution_number_of_common_neighbors


@case("classic.distribution_number_of_common_neighbors.directions")
def _():
    adj = hub_graph(50, seed=1)
    return {d: C.distribution_number_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.distribution_number_of_common_neighbors.default_is_efferent")
def _():
    adj = hub_graph(40, seed=2)
    return [
        C.distribution_number_of_common_neighbors(adj),
        C.distribution_number_of_common_neighbors(adj, None),
    ]


@case("classic.distribution_number_of_common_neighbors.neuron_properties_are_ignored")
def _():
    adj = hub_graph(40, seed=2)
    return C.distribution_number_of_common_neighbors(
        adj, neuron_properties=node_table(40), direction="afferent"
    )


@case("classic.distribution_number_of_common_neighbors.integer_weights_count_as_multiplicities")
def _():
    adj = weighted(40, 0.15, seed=3)
    return {d: C.distribution_number_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.distribution_number_of_common_neighbors.fractional_weights_are_truncated")
def _():
    adj = weighted(30, 0.2, seed=4, integers=False)
    return {d: C.distribution_number_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.distribution_number_of_common_neighbors.input_formats")
def _():
    base = hub_graph(30, seed=5)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float)]
    return [
        [C.distribution_number_of_common_neighbors(m, direction=d) for d in DIRECTIONS]
        for m in inputs
    ]


@case("classic.distribution_number_of_common_neighbors.explicit_zeros")
def _():
    adj = with_explicit_zeros(hub_graph(30, seed=5), every=3)
    return {d: C.distribution_number_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.distribution_number_of_common_neighbors.dense_input", tags=("fixed",))
def _():
    return C.distribution_number_of_common_neighbors(hub_graph(20, seed=6).toarray())


@case("classic.distribution_number_of_common_neighbors.non_square_matrix")
def _():
    adj = sp.csr_matrix(np.random.RandomState(7).random_sample((12, 20)) < 0.25)
    return {d: C.distribution_number_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.distribution_number_of_common_neighbors.no_edges")
def _():
    return {
        d: C.distribution_number_of_common_neighbors(sp.csr_matrix((5, 5), dtype=bool), direction=d)
        for d in DIRECTIONS
    }


@case("classic.distribution_number_of_common_neighbors.two_nodes")
def _():
    return [
        C.distribution_number_of_common_neighbors(from_edges(2, e), direction=d)
        for e in ([(0, 1)], [(0, 1), (1, 0)])
        for d in DIRECTIONS
    ]


@case("classic.distribution_number_of_common_neighbors.complete_graph")
def _():
    return {
        d: C.distribution_number_of_common_neighbors(complete(6), direction=d) for d in DIRECTIONS
    }


@case("classic.distribution_number_of_common_neighbors.single_node")
def _():
    return C.distribution_number_of_common_neighbors(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.distribution_number_of_common_neighbors.bad_direction")
def _():
    return C.distribution_number_of_common_neighbors(hub_graph(20, seed=6), direction="sideways")


# ----------------------------------------------------------------------------- normalized_distribution_of_common_neighbors


@case("classic.normalized_distribution_of_common_neighbors.regular_graph")
def _():
    adj = regular_digraph(30, 5, seed=11)
    return {d: C.normalized_distribution_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.normalized_distribution_of_common_neighbors.hub_graph_with_integer_mean_degree")
def _():
    adj = with_integer_mean_degree(hub_graph(50, seed=12, mean_degree=6.0))
    return {d: C.normalized_distribution_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.normalized_distribution_of_common_neighbors.default_is_efferent")
def _():
    adj = regular_digraph(24, 4, seed=13)
    return [
        C.normalized_distribution_of_common_neighbors(adj),
        C.normalized_distribution_of_common_neighbors(adj, None),
    ]


@case("classic.normalized_distribution_of_common_neighbors.neuron_properties_are_ignored")
def _():
    adj = regular_digraph(24, 4, seed=13)
    return C.normalized_distribution_of_common_neighbors(
        adj, neuron_properties=node_table(24), direction="afferent"
    )


@case("classic.normalized_distribution_of_common_neighbors.fractional_mean_degree_gives_nan")
def _():
    # the hypergeometric reference needs an integer mean degree; otherwise every value is NaN
    adj = hub_graph(40, seed=14)
    assert adj.nnz % 40 != 0
    return {d: C.normalized_distribution_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.normalized_distribution_of_common_neighbors.non_square_matrix")
def _():
    rows = np.repeat(np.arange(30), 4)
    cols = np.concatenate([rng(15).choice(40, 4, replace=False) for _ in range(30)])
    adj = sp.csr_matrix((np.ones(120, dtype=bool), (rows, cols)), shape=(30, 40))
    return {d: C.normalized_distribution_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.normalized_distribution_of_common_neighbors.input_formats")
def _():
    base = regular_digraph(20, 3, seed=16)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float)]
    return [C.normalized_distribution_of_common_neighbors(m, direction="afferent") for m in inputs]


@case("classic.normalized_distribution_of_common_neighbors.complete_graph")
def _():
    return {
        d: C.normalized_distribution_of_common_neighbors(complete(6), direction=d)
        for d in DIRECTIONS
    }


@case("classic.normalized_distribution_of_common_neighbors.no_edges")
def _():
    return {
        d: C.normalized_distribution_of_common_neighbors(
            sp.csr_matrix((5, 5), dtype=bool), direction=d
        )
        for d in DIRECTIONS
    }


@case("classic.normalized_distribution_of_common_neighbors.bad_direction")
def _():
    return C.normalized_distribution_of_common_neighbors(
        regular_digraph(20, 3, seed=16), direction="sideways"
    )


# ----------------------------------------------------------------------------- overexpression_of_common_neighbors


@case("classic.overexpression_of_common_neighbors.regular_graph")
def _():
    adj = regular_digraph(30, 5, seed=21)
    return {d: C.overexpression_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.overexpression_of_common_neighbors.hub_graph_with_integer_mean_degree")
def _():
    adj = with_integer_mean_degree(hub_graph(60, seed=22, mean_degree=6.0))
    return {d: C.overexpression_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.overexpression_of_common_neighbors.default_is_efferent")
def _():
    adj = with_integer_mean_degree(hub_graph(40, seed=23, mean_degree=5.0))
    return [
        C.overexpression_of_common_neighbors(adj),
        C.overexpression_of_common_neighbors(adj, None),
    ]


@case("classic.overexpression_of_common_neighbors.neuron_properties_are_ignored")
def _():
    adj = regular_digraph(24, 4, seed=24)
    return C.overexpression_of_common_neighbors(
        adj, neuron_properties=node_table(24), direction="afferent"
    )


@case("classic.overexpression_of_common_neighbors.fractional_mean_degree_gives_nan")
def _():
    adj = hub_graph(40, seed=14)
    assert adj.nnz % 40 != 0
    return {d: C.overexpression_of_common_neighbors(adj, direction=d) for d in DIRECTIONS}


@case("classic.overexpression_of_common_neighbors.input_formats")
def _():
    base = regular_digraph(20, 3, seed=25)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float)]
    return [C.overexpression_of_common_neighbors(m, direction="efferent") for m in inputs]


@case("classic.overexpression_of_common_neighbors.complete_graph")
def _():
    return {d: C.overexpression_of_common_neighbors(complete(6), direction=d) for d in DIRECTIONS}


@case("classic.overexpression_of_common_neighbors.no_edges")
def _():
    return {
        d: C.overexpression_of_common_neighbors(sp.csr_matrix((5, 5), dtype=bool), direction=d)
        for d in DIRECTIONS
    }


@case("classic.overexpression_of_common_neighbors.bad_direction")
def _():
    return C.overexpression_of_common_neighbors(
        regular_digraph(20, 3, seed=16), direction="sideways"
    )


# ----------------------------------------------------------------------------- common_neighbor_weight_bias


@case("classic.common_neighbor_weight_bias.directions")
def _():
    adj = hub_graph(40, seed=31, mean_degree=6.0, reciprocal=0.4)
    return {
        d: C.common_neighbor_weight_bias(common_neighbour_weights(adj, d, seed=32), direction=d)
        for d in DIRECTIONS
    }


@case("classic.common_neighbor_weight_bias.weights_unrelated_to_common_neighbours")
def _():
    adj = weighted(40, 0.2, seed=33)
    return {d: C.common_neighbor_weight_bias(adj, direction=d) for d in DIRECTIONS}


@case("classic.common_neighbor_weight_bias.real_weights")
def _():
    adj = weighted(40, 0.2, seed=34, integers=False)
    return {d: C.common_neighbor_weight_bias(adj, direction=d) for d in DIRECTIONS}


@case("classic.common_neighbor_weight_bias.default_is_efferent")
def _():
    adj = weighted(40, 0.2, seed=33)
    return [
        C.common_neighbor_weight_bias(adj),
        C.common_neighbor_weight_bias(adj, None, "efferent"),
    ]


@case("classic.common_neighbor_weight_bias.neuron_properties_are_ignored")
def _():
    adj = weighted(40, 0.2, seed=33)
    return C.common_neighbor_weight_bias(
        adj, neuron_properties=node_table(40), direction="afferent"
    )


@case("classic.common_neighbor_weight_bias.binary_matrix_has_no_variance")
def _():
    adj = hub_graph(40, seed=35, mean_degree=6.0)
    return {d: C.common_neighbor_weight_bias(adj, direction=d) for d in DIRECTIONS}


@case("classic.common_neighbor_weight_bias.csc_and_int_input")
def _():
    adj = weighted(40, 0.2, seed=33)
    return [
        C.common_neighbor_weight_bias(adj.tocsc(), direction="efferent"),
        C.common_neighbor_weight_bias(adj.astype(int), direction="afferent"),
        C.common_neighbor_weight_bias(adj.tocsc().astype(int), direction="afferent"),
    ]


@case("classic.common_neighbor_weight_bias.explicit_zero_weights")
def _():
    adj = with_explicit_zeros(weighted(40, 0.2, seed=33), every=5)
    return {d: C.common_neighbor_weight_bias(adj, direction=d) for d in DIRECTIONS}


@case("classic.common_neighbor_weight_bias.negative_weights")
def _():
    # edges with weight <= 0 are not edges for this function
    adj = weighted(40, 0.2, seed=33)
    adj.data[::4] *= -1.0
    return {d: C.common_neighbor_weight_bias(adj, direction=d) for d in DIRECTIONS}


@case("classic.common_neighbor_weight_bias.bad_direction", tags=("fixed",))
def _():
    # the original fails with an UnboundLocalError instead of reporting the direction
    return C.common_neighbor_weight_bias(weighted(40, 0.2, seed=33), direction="sideways")


@case("classic.common_neighbor_weight_bias.dense_input", tags=("fixed",))
def _():
    return C.common_neighbor_weight_bias(weighted(40, 0.2, seed=33).toarray())


@case("classic.common_neighbor_weight_bias.coo_input", tags=("fixed",))
def _():
    return C.common_neighbor_weight_bias(weighted(40, 0.2, seed=33).tocoo())


@case("classic.common_neighbor_weight_bias.lil_input", tags=("fixed",))
def _():
    return C.common_neighbor_weight_bias(weighted(40, 0.2, seed=33).tolil())


@case("classic.common_neighbor_weight_bias.no_edges", tags=("fixed",))
def _():
    return C.common_neighbor_weight_bias(sp.csr_matrix((5, 5)))


# ----------------------------------------------------------------------------- common_neighbor_connectivity_bias
# statsmodels < 0.15 is needed; 0.15 refuses the patsy ModelDesc the function passes to smf.ols


def _bias_inputs(n=30, seed=41, p=0.12):
    return digraph(n, p, seed=seed, reciprocal=0.3), node_table(n, seed=seed)


@case("classic.common_neighbor_connectivity_bias.directions", rtol=1e-6)
def _():
    adj, _ = _bias_inputs()
    return {d: C.common_neighbor_connectivity_bias(adj, direction=d) for d in DIRECTIONS}


@case("classic.common_neighbor_connectivity_bias.default_is_efferent", rtol=1e-6)
def _():
    adj, nodes = _bias_inputs()
    return [
        C.common_neighbor_connectivity_bias(adj),
        C.common_neighbor_connectivity_bias(adj, nodes, "efferent"),
    ]


@case("classic.common_neighbor_connectivity_bias.with_location", rtol=1e-6)
def _():
    adj, nodes = _bias_inputs()
    return {
        d: C.common_neighbor_connectivity_bias(
            adj, nodes, direction=d, cols_location=["x", "y", "z"]
        )
        for d in DIRECTIONS
    }


@case("classic.common_neighbor_connectivity_bias.with_location_in_the_plane", rtol=1e-6)
def _():
    adj, nodes = _bias_inputs(n=35, seed=42)
    return {
        d: C.common_neighbor_connectivity_bias(adj, nodes, direction=d, cols_location=["x", "y"])
        for d in DIRECTIONS
    }


@case("classic.common_neighbor_connectivity_bias.fit_log", rtol=1e-6)
def _():
    adj, _ = _bias_inputs()
    return {
        f"{d}_{shift}": C.common_neighbor_connectivity_bias(adj, direction=d, fit_log=shift)
        for d in DIRECTIONS
        for shift in (0.1, 0.5, 1)
    }


@case("classic.common_neighbor_connectivity_bias.with_location_and_fit_log", rtol=1e-6)
def _():
    adj, nodes = _bias_inputs()
    return {
        d: C.common_neighbor_connectivity_bias(
            adj, nodes, direction=d, cols_location=["x", "y", "z"], fit_log=0.1
        )
        for d in DIRECTIONS
    }


@case("classic.common_neighbor_connectivity_bias.falsy_fit_log_means_no_log", rtol=1e-6)
def _():
    adj, _ = _bias_inputs()
    return [
        C.common_neighbor_connectivity_bias(adj, fit_log=0.0),
        C.common_neighbor_connectivity_bias(adj),
    ]


@case("classic.common_neighbor_connectivity_bias.hub_graph", rtol=1e-6)
def _():
    adj = hub_graph(50, seed=43, mean_degree=4.0)
    nodes = node_table(50, seed=43)
    return [
        C.common_neighbor_connectivity_bias(
            adj, nodes, direction=d, cols_location=["x", "y", "z"], fit_log=0.1
        )
        for d in DIRECTIONS
    ]


@case("classic.common_neighbor_connectivity_bias.weighted_matrix", rtol=1e-6)
def _():
    adj = weighted(30, 0.15, seed=44)
    nodes = node_table(30, seed=44)
    return {
        d: C.common_neighbor_connectivity_bias(
            adj, nodes, direction=d, cols_location=["x", "y"], fit_log=0.1
        )
        for d in DIRECTIONS
    }


@case("classic.common_neighbor_connectivity_bias.input_formats", rtol=1e-6)
def _():
    adj, _ = _bias_inputs()
    inputs = [adj.tocsc(), adj.tocoo(), adj.tolil(), adj.astype(int), adj.astype(float)]
    return [C.common_neighbor_connectivity_bias(m, direction="afferent") for m in inputs]


@case("classic.common_neighbor_connectivity_bias.two_and_three_nodes", rtol=1e-6)
def _():
    return [
        C.common_neighbor_connectivity_bias(from_edges(2, [(0, 1)])),
        C.common_neighbor_connectivity_bias(from_edges(3, [(0, 1), (1, 2), (0, 2)])),
    ]


@case("classic.common_neighbor_connectivity_bias.complete_graph", rtol=1e-6)
def _():
    # every pair is connected, so the connection indicator is collinear with the intercept
    return C.common_neighbor_connectivity_bias(complete(6))


@case("classic.common_neighbor_connectivity_bias.no_edges", rtol=1e-6)
def _():
    return [
        C.common_neighbor_connectivity_bias(sp.csr_matrix((6, 6), dtype=bool)),
        C.common_neighbor_connectivity_bias(sp.csr_matrix((6, 6), dtype=bool), fit_log=0.5),
    ]


@case("classic.common_neighbor_connectivity_bias.bad_direction", tags=("fixed",))
def _():
    # the original fails with an UnboundLocalError instead of reporting the direction
    adj, _ = _bias_inputs()
    return C.common_neighbor_connectivity_bias(adj, direction="sideways")


@case("classic.common_neighbor_connectivity_bias.dense_input", tags=("fixed",))
def _():
    adj, _ = _bias_inputs()
    return C.common_neighbor_connectivity_bias(adj.toarray())


@case("classic.common_neighbor_connectivity_bias.location_without_node_properties")
def _():
    adj, _ = _bias_inputs()
    return C.common_neighbor_connectivity_bias(adj, None, cols_location=["x", "y"])


@case("classic.common_neighbor_connectivity_bias.unknown_location_column")
def _():
    adj, nodes = _bias_inputs()
    return C.common_neighbor_connectivity_bias(adj, nodes, cols_location=["x", "nope"])
