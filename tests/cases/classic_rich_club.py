"""Rich-club curves, their null models and the degree-preserving shuffle in ``connalysis.network.classic``."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, planted, rng, weighted

from cases.classic_inputs import complete, from_edges, hub_graph, with_explicit_zeros
from connalysis.network import classic as C

DIRECTIONS = ("TOTAL", "IN", "OUT")
NULL_DIRECTIONS = ("efferent", "afferent")


def _hubs(n=60, seed=51, **kwargs):
    return hub_graph(n, seed=seed, mean_degree=kwargs.pop("mean_degree", 6.0), **kwargs)


# ----------------------------------------------------------------------------- _bin_degrees


@case("classic.bin_degrees.integer_series")
def _():
    return C._bin_degrees(pd.Series([1, 2, 2, 3, 5, 8, 8, 9, 12, 20]))


@case("classic.bin_degrees.ndarray")
def _():
    return C._bin_degrees(np.array([1, 2, 2, 3, 5, 8, 8, 9, 12, 20]))


@case("classic.bin_degrees.fewer_nodes_than_thirty")
def _():
    return C._bin_degrees(pd.Series([4, 4, 7, 1, 0, 9, 2]))


@case("classic.bin_degrees.forty_nodes")
def _():
    return C._bin_degrees(pd.Series(np.arange(40) % 13))


@case("classic.bin_degrees.four_hundred_nodes_use_ten_percent_bins")
def _():
    return C._bin_degrees(np.arange(400) % 37)


@case("classic.bin_degrees.float_values")
def _():
    return C._bin_degrees(pd.Series(np.linspace(0.0, 1.0, 50) ** 2))


@case("classic.bin_degrees.constant_values")
def _():
    return C._bin_degrees(pd.Series(np.full(5, 3)))


@case("classic.bin_degrees.zeros")
def _():
    return C._bin_degrees(np.zeros(8, dtype=int))


@case("classic.bin_degrees.single_value")
def _():
    return C._bin_degrees(np.array([7]))


@case("classic.bin_degrees.negative_values")
def _():
    return C._bin_degrees(np.array([-3.5, -1.0, 0.0, 2.0, 2.5, 9.0]))


@case("classic.bin_degrees.nan_values_are_skipped_for_the_range")
def _():
    return C._bin_degrees(np.array([1, 2, np.nan, 3, 5, 8, 8, 9, 12, 20]))


@case("classic.bin_degrees.hub_graph_degrees")
def _():
    adj = _hubs(100, seed=52)
    return C._bin_degrees(pd.Series(np.asarray((adj + adj.T).astype(int).sum(axis=0)).ravel()))


# ----------------------------------------------------------------------------- rich_club_curve


@case("classic.rich_club_curve.binary_matrix_all_directions")
def _():
    adj = _hubs()
    return {d: C.rich_club_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.rich_club_curve.default_is_total")
def _():
    adj = planted(50, 0.06, seed=53, sizes=(7, 5, 4))
    return [
        C.rich_club_curve(adj),
        C.rich_club_curve(adj, "TOTAL"),
        C.rich_club_curve(adj, direction=None),
    ]


@case("classic.rich_club_curve.weighted_matrix_all_directions")
def _():
    # a matrix that is not boolean has its degrees binned and counts weights instead of edges
    adj = weighted(60, 0.1, seed=54)
    return {d: C.rich_club_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.rich_club_curve.real_weights")
def _():
    adj = weighted(60, 0.1, seed=55, integers=False)
    return {d: C.rich_club_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.rich_club_curve.integer_and_float_dtypes_take_the_binned_path")
def _():
    adj = _hubs()
    return [
        C.rich_club_curve(adj.astype(int)),
        C.rich_club_curve(adj.astype(float), direction="OUT"),
        C.rich_club_curve(adj.astype(np.int8), direction="IN"),
    ]


@case("classic.rich_club_curve.dense_input")
def _():
    adj = _hubs()
    return [
        C.rich_club_curve(adj.toarray(), direction="OUT"),
        C.rich_club_curve(adj.toarray().astype(float), direction="OUT"),
        C.rich_club_curve(weighted(40, 0.1, seed=56).toarray(), direction="IN"),
    ]


@case("classic.rich_club_curve.input_formats")
def _():
    adj = _hubs(40, seed=57)
    return [C.rich_club_curve(m) for m in (adj.tocsc(), adj.tocoo(), adj.tolil())]


@case("classic.rich_club_curve.sparse_array_input")
def _():
    adj = sp.csr_array(_hubs(40, seed=57))
    return [C.rich_club_curve(adj, direction=d) for d in DIRECTIONS]


@case("classic.rich_club_curve.explicit_zeros")
def _():
    adj = with_explicit_zeros(_hubs(40, seed=57), every=3)
    wadj = with_explicit_zeros(weighted(40, 0.15, seed=58), every=3)
    return [C.rich_club_curve(adj), C.rich_club_curve(wadj)]


@case("classic.rich_club_curve.self_loops")
def _():
    adj = _hubs(30, seed=59).tolil()
    for i in (0, 5, 9):
        adj[i, i] = True
    return C.rich_club_curve(adj.tocsr(), direction="OUT")


@case("classic.rich_club_curve.complete_graph")
def _():
    return {d: C.rich_club_curve(complete(6), direction=d) for d in DIRECTIONS}


@case("classic.rich_club_curve.no_edges")
def _():
    return C.rich_club_curve(sp.csr_matrix((5, 5), dtype=bool))


@case("classic.rich_club_curve.two_nodes")
def _():
    return [
        C.rich_club_curve(from_edges(2, e), direction=d)
        for e in ([(0, 1)], [(0, 1), (1, 0)])
        for d in DIRECTIONS
    ]


@case("classic.rich_club_curve.bad_direction")
def _():
    return C.rich_club_curve(_hubs(20, seed=60), direction="sideways")


# ----------------------------------------------------------------------------- efficient_rich_club_curve


@case("classic.efficient_rich_club_curve.directions")
def _():
    adj = _hubs()
    return {d: C.efficient_rich_club_curve(adj, direction=d) for d in DIRECTIONS}


@case("classic.efficient_rich_club_curve.default_is_total")
def _():
    adj = planted(50, 0.06, seed=53, sizes=(7, 5, 4))
    return [
        C.efficient_rich_club_curve(adj),
        C.efficient_rich_club_curve(adj, "TOTAL"),
        C.efficient_rich_club_curve(adj, direction=None),
    ]


@case("classic.efficient_rich_club_curve.sparse_bin_set")
def _():
    adj = _hubs()
    return {
        f"{d}_{flag}": C.efficient_rich_club_curve(adj, direction=d, sparse_bin_set=flag)
        for d in DIRECTIONS
        for flag in (False, True)
    }


@case("classic.efficient_rich_club_curve.agrees_with_rich_club_curve_on_binary_matrices")
def _():
    adj = planted(50, 0.06, seed=53, sizes=(7, 5, 4))
    return [
        (C.efficient_rich_club_curve(adj, direction=d), C.rich_club_curve(adj, direction=d))
        for d in DIRECTIONS
    ]


@case("classic.efficient_rich_club_curve.integer_filtration")
def _():
    adj = _hubs()
    filtration = pd.Series(rng(61).randint(0, 6, 60))
    return [
        C.efficient_rich_club_curve(adj, pre_calculated_filtration=filtration, sparse_bin_set=flag)
        for flag in (False, True)
    ]


@case("classic.efficient_rich_club_curve.filtration_with_gaps")
def _():
    adj = _hubs()
    filtration = pd.Series(rng(62).choice([0, 3, 7, 12], 60))
    return [
        C.efficient_rich_club_curve(adj, pre_calculated_filtration=filtration, sparse_bin_set=flag)
        for flag in (False, True)
    ]


@case("classic.efficient_rich_club_curve.float_filtration")
def _():
    adj = _hubs()
    filtration = pd.Series(rng(63).random_sample(60) * 10)
    return [
        C.efficient_rich_club_curve(adj, pre_calculated_filtration=filtration, sparse_bin_set=flag)
        for flag in (False, True)
    ]


@case("classic.efficient_rich_club_curve.filtration_overrides_direction")
def _():
    adj = _hubs()
    filtration = pd.Series(rng(61).randint(0, 6, 60))
    return [
        C.efficient_rich_club_curve(adj, direction=d, pre_calculated_filtration=filtration)
        for d in DIRECTIONS
    ]


@case("classic.efficient_rich_club_curve.filtration_given_for_some_nodes_only")
def _():
    adj = _hubs()
    filtration = pd.Series(rng(64).randint(0, 6, 30), index=np.arange(0, 60, 2))
    return [
        C.efficient_rich_club_curve(adj, pre_calculated_filtration=filtration, sparse_bin_set=flag)
        for flag in (False, True)
    ]


@case("classic.efficient_rich_club_curve.filtration_with_permuted_index")
def _():
    # the filtration value belongs to the node named by the index, not to its position
    adj = _hubs()
    perm = rng(66).permutation(60)
    filtration = pd.Series(rng(61).randint(0, 6, 60), index=perm)
    return C.efficient_rich_club_curve(adj, pre_calculated_filtration=filtration)


@case("classic.efficient_rich_club_curve.sparse_array_input")
def _():
    adj = sp.csr_array(_hubs(40, seed=57))
    return [C.efficient_rich_club_curve(adj, direction=d) for d in DIRECTIONS]


@case("classic.efficient_rich_club_curve.weights_are_ignored")
def _():
    adj = weighted(60, 0.1, seed=65)
    return [C.efficient_rich_club_curve(adj, direction=d) for d in DIRECTIONS]


@case("classic.efficient_rich_club_curve.input_formats")
def _():
    adj = _hubs(40, seed=57)
    inputs = [adj.tocsc(), adj.tocoo(), adj.tolil(), adj.astype(int), adj.astype(float)]
    return [C.efficient_rich_club_curve(m, direction="OUT") for m in inputs]


@case("classic.efficient_rich_club_curve.stored_zeros_count_as_edges")
def _():
    adj = with_explicit_zeros(_hubs(40, seed=57), every=3)
    return [C.efficient_rich_club_curve(adj, direction=d) for d in DIRECTIONS]


@case("classic.efficient_rich_club_curve.self_loops")
def _():
    adj = _hubs(30, seed=59).tolil()
    for i in (0, 5, 9):
        adj[i, i] = True
    return C.efficient_rich_club_curve(adj.tocsr(), direction="OUT")


@case("classic.efficient_rich_club_curve.complete_graph")
def _():
    return [
        C.efficient_rich_club_curve(complete(6), direction=d, sparse_bin_set=flag)
        for d in DIRECTIONS
        for flag in (False, True)
    ]


@case("classic.efficient_rich_club_curve.no_edges")
def _():
    return [
        C.efficient_rich_club_curve(sp.csr_matrix((6, 6), dtype=bool), sparse_bin_set=flag)
        for flag in (False, True)
    ]


@case("classic.efficient_rich_club_curve.single_node")
def _():
    return C.efficient_rich_club_curve(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.efficient_rich_club_curve.two_nodes")
def _():
    return [
        C.efficient_rich_club_curve(from_edges(2, e), direction=d)
        for e in ([(0, 1)], [(0, 1), (1, 0)])
        for d in DIRECTIONS
    ]


@case("classic.efficient_rich_club_curve.bad_direction")
def _():
    return C.efficient_rich_club_curve(_hubs(20, seed=60), direction="sideways")


@case("classic.efficient_rich_club_curve.null_model_direction_names_are_rejected")
def _():
    return C.efficient_rich_club_curve(_hubs(20, seed=60), direction="efferent")


@case("classic.efficient_rich_club_curve.dense_input", tags=("fixed",))
def _():
    return C.efficient_rich_club_curve(_hubs(20, seed=60).toarray())


# ----------------------------------------------------------------------------- _analytical_expected_rich_club_curve


@case("classic.analytical_expected_rich_club_curve.directions")
def _():
    adj = _hubs()
    return {d: C._analytical_expected_rich_club_curve(adj, direction=d) for d in NULL_DIRECTIONS}


@case("classic.analytical_expected_rich_club_curve.default_is_efferent")
def _():
    adj = planted(40, 0.07, seed=66, sizes=(6, 5))
    return [
        C._analytical_expected_rich_club_curve(adj),
        C._analytical_expected_rich_club_curve(adj, "efferent"),
    ]


@case("classic.analytical_expected_rich_club_curve.input_formats")
def _():
    adj = _hubs(30, seed=67)
    return [
        C._analytical_expected_rich_club_curve(m, direction="afferent")
        for m in (adj.tocsc(), adj.tocoo())
    ]


@case("classic.analytical_expected_rich_club_curve.reciprocal_graph")
def _():
    adj = digraph(40, 0.1, seed=68, reciprocal=0.8)
    return {d: C._analytical_expected_rich_club_curve(adj, direction=d) for d in NULL_DIRECTIONS}


@case("classic.analytical_expected_rich_club_curve.complete_graph")
def _():
    return {
        d: C._analytical_expected_rich_club_curve(complete(6), direction=d) for d in NULL_DIRECTIONS
    }


@case("classic.analytical_expected_rich_club_curve.no_edges")
def _():
    return C._analytical_expected_rich_club_curve(sp.csr_matrix((5, 5), dtype=bool))


@case("classic.analytical_expected_rich_club_curve.two_nodes")
def _():
    return [
        C._analytical_expected_rich_club_curve(from_edges(2, e), direction=d)
        for e in ([(0, 1)], [(0, 1), (1, 0)])
        for d in NULL_DIRECTIONS
    ]


@case("classic.analytical_expected_rich_club_curve.bad_direction")
def _():
    return C._analytical_expected_rich_club_curve(_hubs(20, seed=60), direction="OUT")


@case("classic.analytical_expected_rich_club_curve.weighted_matrix_is_rejected")
def _():
    return C._analytical_expected_rich_club_curve(weighted(20, 0.2, seed=69))


@case("classic.analytical_expected_rich_club_curve.dense_input", tags=("fixed",))
def _():
    return C._analytical_expected_rich_club_curve(_hubs(20, seed=60).toarray())


# ----------------------------------------------------------------------------- generate_degree_based_control


def _shuffle(adj, direction, seed):
    """Result, the input after the call (it is shuffled in place when no format conversion is needed), and
    whether the result is the input object."""
    np.random.seed(seed)
    out = C.generate_degree_based_control(adj, direction=direction)
    return out, adj, out is adj


@case("classic.generate_degree_based_control.efferent_csr_input_is_modified_in_place")
def _():
    return _shuffle(digraph(30, 0.15, seed=71), "efferent", 5)


@case("classic.generate_degree_based_control.afferent_csc_input_is_modified_in_place")
def _():
    return _shuffle(digraph(30, 0.15, seed=71, fmt="csc"), "afferent", 5)


@case("classic.generate_degree_based_control.efferent_csc_input_is_converted")
def _():
    return _shuffle(digraph(30, 0.15, seed=71, fmt="csc"), "efferent", 5)


@case("classic.generate_degree_based_control.afferent_csr_input_is_converted")
def _():
    return _shuffle(digraph(30, 0.15, seed=71), "afferent", 5)


@case("classic.generate_degree_based_control.coo_and_lil_input")
def _():
    adj = digraph(30, 0.15, seed=71)
    return [_shuffle(adj.tocoo(), "efferent", 6), _shuffle(adj.tolil(), "afferent", 6)]


@case("classic.generate_degree_based_control.default_is_efferent")
def _():
    np.random.seed(7)
    a = C.generate_degree_based_control(digraph(30, 0.15, seed=72))
    np.random.seed(7)
    b = C.generate_degree_based_control(digraph(30, 0.15, seed=72), "efferent")
    return a, b


@case("classic.generate_degree_based_control.different_seeds_give_different_matrices")
def _():
    return [_shuffle(digraph(30, 0.15, seed=71), "efferent", s)[0] for s in (1, 2, 3)]


@case("classic.generate_degree_based_control.hub_graph")
def _():
    return [_shuffle(_hubs(50, seed=73), d, 8)[0] for d in NULL_DIRECTIONS]


@case("classic.generate_degree_based_control.weighted_matrix_keeps_weights_in_place")
def _():
    # the stored weights keep their position in the data array; only the column indices are redrawn
    return [_shuffle(weighted(30, 0.15, seed=74), d, 9) for d in NULL_DIRECTIONS]


@case("classic.generate_degree_based_control.explicit_zeros")
def _():
    return _shuffle(with_explicit_zeros(digraph(30, 0.15, seed=71), every=3), "efferent", 5)


@case("classic.generate_degree_based_control.complete_graph")
def _():
    return [_shuffle(complete(6), d, 4)[0] for d in NULL_DIRECTIONS]


@case("classic.generate_degree_based_control.two_nodes_reciprocal")
def _():
    return _shuffle(from_edges(2, [(0, 1), (1, 0)]), "efferent", 3)


@case("classic.generate_degree_based_control.node_without_edges")
def _():
    # node 7 has no edges at all, node 3 has no out-edges and node 5 no in-edges
    edges = [
        (i, j)
        for i in range(8)
        for j in range(8)
        if i != j and i not in (3, 7) and j not in (5, 7) and (i + 2 * j) % 3 == 0
    ]
    return [_shuffle(from_edges(8, edges), d, 2)[0] for d in NULL_DIRECTIONS]


@case("classic.generate_degree_based_control.random_stream_consumption")
def _():
    np.random.seed(10)
    C.generate_degree_based_control(digraph(30, 0.15, seed=71), "efferent")
    first = np.random.rand(3)
    np.random.seed(10)
    C.generate_degree_based_control(digraph(30, 0.15, seed=71, fmt="csc"), "afferent")
    return first, np.random.rand(3)


@case("classic.generate_degree_based_control.random_stream_consumption_with_many_collisions")
def _():
    # a heavy-tailed target distribution makes the sampling without replacement redraw often
    np.random.seed(10)
    C.generate_degree_based_control(_hubs(40, seed=85), "efferent")
    first = np.random.rand(3)
    np.random.seed(10)
    C.generate_degree_based_control(_hubs(40, seed=85, fmt="csc"), "afferent")
    return first, np.random.rand(3)


@case("classic.generate_degree_based_control.bad_direction")
def _():
    return C.generate_degree_based_control(digraph(10, 0.3, seed=75), direction="sideways")


@case("classic.generate_degree_based_control.no_edges", tags=("fixed",))
def _():
    np.random.seed(1)
    return C.generate_degree_based_control(sp.csr_matrix((5, 5), dtype=bool))


@case("classic.generate_degree_based_control.dense_input", tags=("fixed",))
def _():
    np.random.seed(1)
    return C.generate_degree_based_control(digraph(10, 0.3, seed=75).toarray())


# ----------------------------------------------------------------------------- _randomized_control_rich_club_curve


@case("classic.randomized_control_rich_club_curve.efferent")
def _():
    adj = planted(40, 0.08, seed=81, sizes=(6, 5))
    np.random.seed(11)
    result = C._randomized_control_rich_club_curve(adj, direction="efferent", n=4)
    return result, adj


@case("classic.randomized_control_rich_club_curve.afferent")
def _():
    adj = planted(40, 0.08, seed=81, sizes=(6, 5), fmt="csc")
    np.random.seed(11)
    result = C._randomized_control_rich_club_curve(adj, direction="afferent", n=4)
    return result, adj


@case("classic.randomized_control_rich_club_curve.csr_input_is_converted_for_afferent")
def _():
    adj = planted(40, 0.08, seed=81, sizes=(6, 5))
    np.random.seed(11)
    result = C._randomized_control_rich_club_curve(adj, direction="afferent", n=3)
    return result, adj


@case("classic.randomized_control_rich_club_curve.csc_input_is_converted_for_efferent")
def _():
    adj = planted(40, 0.08, seed=81, sizes=(6, 5), fmt="csc")
    np.random.seed(11)
    result = C._randomized_control_rich_club_curve(adj, direction="efferent", n=3)
    return result, adj


@case("classic.randomized_control_rich_club_curve.default_arguments")
def _():
    adj = _hubs(40, seed=82)
    np.random.seed(12)
    return C._randomized_control_rich_club_curve(adj)


@case("classic.randomized_control_rich_club_curve.single_shuffle_has_zero_spread")
def _():
    adj = _hubs(40, seed=82)
    np.random.seed(13)
    return C._randomized_control_rich_club_curve(adj, n=1)


@case("classic.randomized_control_rich_club_curve.hub_graph_afferent")
def _():
    adj = _hubs(60, seed=83, fmt="csc")
    np.random.seed(14)
    return C._randomized_control_rich_club_curve(adj, direction="afferent", n=5)


@case("classic.randomized_control_rich_club_curve.random_stream_consumption")
def _():
    adj = planted(40, 0.08, seed=81, sizes=(6, 5))
    np.random.seed(15)
    C._randomized_control_rich_club_curve(adj, n=3)
    return np.random.rand(3)


@case("classic.randomized_control_rich_club_curve.bad_direction")
def _():
    np.random.seed(16)
    return C._randomized_control_rich_club_curve(_hubs(20, seed=60), direction="OUT", n=2)


# ----------------------------------------------------------------------------- normalized_rich_club_curve / rich_club_coefficient
# In the original both always fail: the default direction 'efferent' is not a direction of
# efficient_rich_club_curve, and 'IN' / 'OUT' / 'TOTAL' are not directions of the two null models.


@case("classic.normalized_rich_club_curve.default_arguments", tags=("fixed",))
def _():
    np.random.seed(17)
    return C.normalized_rich_club_curve(_hubs(40, seed=84))


@case("classic.normalized_rich_club_curve.analytical_null_model", tags=("fixed",))
def _():
    return C.normalized_rich_club_curve(_hubs(40, seed=84), normalize_with="analytical")


@case("classic.normalized_rich_club_curve.afferent", tags=("fixed",))
def _():
    np.random.seed(17)
    return C.normalized_rich_club_curve(_hubs(40, seed=84), direction="afferent")


@case("classic.normalized_rich_club_curve.out_direction_with_shuffled_null_model", tags=("fixed",))
def _():
    np.random.seed(17)
    return C.normalized_rich_club_curve(_hubs(40, seed=84), direction="OUT")


@case(
    "classic.normalized_rich_club_curve.out_direction_with_analytical_null_model", tags=("fixed",)
)
def _():
    return C.normalized_rich_club_curve(
        _hubs(40, seed=84), direction="OUT", normalize_with="analytical"
    )


@case("classic.normalized_rich_club_curve.mean_normalisation", tags=("fixed",))
def _():
    np.random.seed(17)
    return C.normalized_rich_club_curve(_hubs(40, seed=84), normalize="mean")


@case("classic.normalized_rich_club_curve.weighted_matrix_is_rejected")
def _():
    return C.normalized_rich_club_curve(weighted(20, 0.2, seed=69))


@case("classic.rich_club_coefficient.default_arguments", tags=("fixed",))
def _():
    np.random.seed(18)
    return C.rich_club_coefficient(_hubs(40, seed=84))


@case("classic.rich_club_coefficient.analytical_null_model", tags=("fixed",))
def _():
    return C.rich_club_coefficient(_hubs(40, seed=84), normalize_with="analytical")


@case("classic.rich_club_coefficient.weighted_matrix_is_rejected")
def _():
    return C.rich_club_coefficient(weighted(20, 0.2, seed=69))
