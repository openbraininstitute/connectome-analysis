"""The ``classic`` analyses on subsets of the real connectome in ``tutorials/`` (the first ``n`` neurons)."""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import tutorial_connectome

from connalysis.network import classic as C

TYPES = ("directed", "undirected", "reciprocal")
DIRECTIONS = ("TOTAL", "IN", "OUT")
NULL_DIRECTIONS = ("efferent", "afferent")


def _thin(adj, keep_fraction, seed):
    """Keep each edge with probability ``keep_fraction`` (frozen RandomState stream)."""
    coo = adj.tocoo()
    keep = np.random.RandomState(seed).random_sample(coo.nnz) < keep_fraction
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=adj.shape)


@case("classic.real.rich_club_curve")
def _():
    adj, _nodes = tutorial_connectome(500)
    return {d: C.rich_club_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.real.rich_club_curve_weighted_by_synapse_like_counts")
def _():
    # the same graph with integer weights, which sends rich_club_curve down its binned path
    adj, _nodes = tutorial_connectome(300)
    weights = adj.astype(float).tocsr()
    weights.data = 1.0 + (np.arange(weights.nnz) % 4)
    return {d: C.rich_club_curve(weights, direction=d) for d in DIRECTIONS}


@case("classic.real.efficient_rich_club_curve")
def _():
    adj, _nodes = tutorial_connectome(700)
    return {
        f"{d}_{flag}": C.efficient_rich_club_curve(adj, direction=d, sparse_bin_set=flag)
        for d in DIRECTIONS
        for flag in (False, True)
    }


@case("classic.real.efficient_rich_club_curve_with_depth_filtration")
def _():
    adj, nodes = tutorial_connectome(500)
    depth = nodes["depth"]
    return [
        C.efficient_rich_club_curve(adj, pre_calculated_filtration=depth, sparse_bin_set=flag)
        for flag in (False, True)
    ]


@case("classic.real.efficient_rich_club_curve_with_layer_filtration")
def _():
    adj, nodes = tutorial_connectome(500)
    return C.efficient_rich_club_curve(adj, pre_calculated_filtration=nodes["layer"].astype(int))


@case("classic.real.analytical_expected_rich_club_curve")
def _():
    adj, _nodes = tutorial_connectome(300)
    return {d: C._analytical_expected_rich_club_curve(adj, direction=d) for d in NULL_DIRECTIONS}


@case("classic.real.generate_degree_based_control")
def _():
    adj, _nodes = tutorial_connectome(300)
    np.random.seed(21)
    shuffled = C.generate_degree_based_control(adj.copy(), direction="efferent")
    np.random.seed(21)
    shuffled_in = C.generate_degree_based_control(adj.tocsc(), direction="afferent")
    return shuffled, shuffled_in


@case("classic.real.randomized_control_rich_club_curve")
def _():
    adj, _nodes = tutorial_connectome(300)
    np.random.seed(22)
    return C._randomized_control_rich_club_curve(adj.copy(), direction="efferent", n=4)


@case("classic.real.connection_probability_within")
def _():
    adj, nodes = tutorial_connectome(700)
    return {
        t: C.connection_probability_within(adj, nodes, cols=["x", "y"], max_dist=60, type=t)
        for t in TYPES
    }


@case("classic.real.connection_probability_within_columns_and_distances")
def _():
    adj, nodes = tutorial_connectome(500)
    out = {
        f"xyz_{d}": C.connection_probability_within(adj, nodes, cols=["x", "y", "z"], max_dist=d)
        for d in (25, 50, 100, 200)
    }
    out["depth"] = C.connection_probability_within(adj, nodes, cols=["depth"], max_dist=30)
    out["default"] = C.connection_probability_within(adj, nodes)
    return out


@case("classic.real.connection_probability_within_pre_and_post_populations")
def _():
    adj, nodes = tutorial_connectome(600)
    pre, post = nodes.iloc[:350], nodes.iloc[350:]
    return C.connection_probability_within(
        adj[:350, 350:].tocsr(), (pre, post), cols=["x", "y", "z"], max_dist=80
    )


@case("classic.real.get_pairs_within")
def _():
    adj, nodes = tutorial_connectome(400)
    return C.get_pairs_within(adj, nodes, cols=["x", "y", "z"], max_dist=30)


@case("classic.real.distribution_number_of_common_neighbors")
def _():
    adj, _nodes = tutorial_connectome(700)
    return {d: C.distribution_number_of_common_neighbors(adj, direction=d) for d in NULL_DIRECTIONS}


@case("classic.real.normalized_distribution_and_overexpression_of_common_neighbors")
def _():
    # the null model needs an integer mean degree: take the first n neurons for the largest n where nnz is
    # divisible by n (525 at the time of writing)
    full, _nodes = tutorial_connectome(700)
    n = next(n for n in range(700, 400, -1) if full[:n, :n].nnz % n == 0)
    adj = full[:n, :n].tocsr()
    return {
        d: (
            C.normalized_distribution_of_common_neighbors(adj, direction=d),
            C.overexpression_of_common_neighbors(adj, direction=d),
        )
        for d in NULL_DIRECTIONS
    }


@case("classic.real.common_neighbor_overexpression_with_fractional_mean_degree")
def _():
    adj, _nodes = tutorial_connectome(300)
    return {d: C.overexpression_of_common_neighbors(adj, direction=d) for d in NULL_DIRECTIONS}


@case("classic.real.common_neighbor_connectivity_bias", rtol=1e-6)
def _():
    adj, nodes = tutorial_connectome(300)
    return {
        d: C.common_neighbor_connectivity_bias(
            adj, nodes, direction=d, cols_location=["x", "y", "z"], fit_log=0.1
        )
        for d in NULL_DIRECTIONS
    }


@case("classic.real.common_neighbor_connectivity_bias_without_location", rtol=1e-6)
def _():
    adj, nodes = tutorial_connectome(300)
    return {
        d: C.common_neighbor_connectivity_bias(adj, nodes, direction=d) for d in NULL_DIRECTIONS
    }


@case("classic.real.common_neighbor_weight_bias")
def _():
    # synapse-count-like weights that grow with the number of common neighbours
    adj, _nodes = tutorial_connectome(300)
    b = adj.astype(int)
    out = {}
    for d in NULL_DIRECTIONS:
        cn = b @ b.T if d == "efferent" else b.T @ b
        weights = (adj.astype(float) + cn.multiply(adj)).tocsr()
        weights.data = weights.data + (np.arange(weights.nnz) % 3)
        out[d] = C.common_neighbor_weight_bias(weights, direction=d)
    return out


@case("classic.real.ccc")
def _():
    adj, _nodes = tutorial_connectome(500)
    return [C.ccc(adj), C.ccc(adj.toarray())]


@case("classic.real.core_number")
def _():
    adj, _nodes = tutorial_connectome(500)
    return C.core_number(adj)


@case("classic.real.connected_components")
def _():
    adj, _nodes = tutorial_connectome(700)
    return [
        C.connected_components(adj, connection=c, return_labels=True) for c in ("weak", "strong")
    ]


@case("classic.real.connected_components_of_a_thinned_graph")
def _():
    adj, _nodes = tutorial_connectome(700)
    thinned = _thin(adj, 0.12, seed=3)
    return [
        C.connected_components(thinned, connection=c, return_labels=True)
        for c in ("weak", "strong")
    ]


@case("classic.real.closeness_connected_components")
def _():
    adj, _nodes = tutorial_connectome(300)
    return [C.closeness_connected_components(adj, directed=d) for d in (False, True)]


@case("classic.real.density")
def _():
    adj, _nodes = tutorial_connectome(700)
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.real.gini")
def _():
    adj, nodes = tutorial_connectome(500)
    return {
        d: (
            C.gini_curve(adj, nodes, direction=d),
            C.gini_coefficient(adj, nodes, direction=d),
            C.normalized_gini_coefficient(adj, nodes, direction=d),
        )
        for d in NULL_DIRECTIONS
    }


@case("classic.real.analytical_expected_gini_curve")
def _():
    adj, _nodes = tutorial_connectome(500)
    return {d: C._analytical_expected_gini_curve(adj, direction=d) for d in NULL_DIRECTIONS}


@case("classic.real.largest_strongly_connected_component")
def _():
    adj, _nodes = tutorial_connectome(200)
    result = C.largest_strongly_connected_component(adj)
    return result.dtype.name, result.shape, sp.csr_matrix(result)


@case("classic.real.largest_strongly_connected_component_of_a_thinned_graph")
def _():
    adj, _nodes = tutorial_connectome(300)
    thinned = _thin(adj, 0.35, seed=5)
    result = C.largest_strongly_connected_component(thinned)
    return result.dtype.name, result.shape, sp.csr_matrix(result)


@case("classic.real.networkx_round_trip")
def _():
    adj, _nodes = tutorial_connectome(100)
    graph = C.np_to_nx(adj.toarray())
    return graph, C.nx_to_np(graph)
