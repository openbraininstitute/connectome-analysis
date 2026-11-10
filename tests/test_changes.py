"""The behaviour that differs from the original on purpose (see ``tests/support/intentional.py``).

Either the original failed because of a defect and the function works now, or it refused an input for no good reason.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from support.data import digraph, node_table, planted, positions

from connalysis.modelling import modelling as M
from connalysis.network import classic as C
from connalysis.network import local as L
from connalysis.network import topology as T
from connalysis.randomization import add_rc_connections_skeleta


def assert_matrices_equal(left, right):
    assert left.shape == right.shape and (left != right).nnz == 0


# ------------------------------------------------------------------------------- inputs that are accepted now


@pytest.mark.parametrize("make", [lambda a: a.tocoo(), lambda a: a.tolil(), lambda a: a.tocsc()])
def test_the_model_extraction_accepts_any_sparse_format(make):
    adj = digraph(60, 0.15, seed=1)
    distance = np.sqrt(((positions(60)[:, None] - positions(60)[None]) ** 2).sum(-1))
    np.fill_diagonal(distance, np.nan)
    bins = np.arange(0, 500, 50)
    expected = M._extract_dependent_p_conn(adj, [distance], [bins])
    for got, want in zip(M._extract_dependent_p_conn(make(adj), [distance], [bins]), expected):
        np.testing.assert_array_equal(got, want)


def test_model_building_with_a_coo_adjacency_matrix():
    adj, nodes = digraph(60, 0.15, seed=2), node_table(60, seed=2)
    pd.testing.assert_frame_equal(
        M.conn_prob_2nd_order_model(adj.tocoo(), nodes), M.conn_prob_2nd_order_model(adj, nodes)
    )


def test_dense_arrays_where_the_original_asked_for_sparse_matrices():
    adj = planted(30, 0.1, seed=2, sizes=(4,))
    simplices = T.list_simplices_by_dimension(adj)
    for got, want in zip(T.bedge_counts(adj.toarray(), simplices), T.bedge_counts(adj, simplices)):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(
        T.count_triads_fully_connected(adj.toarray()), T.count_triads_fully_connected(adj)
    )
    pd.testing.assert_series_equal(
        C.distribution_number_of_common_neighbors(adj.toarray()),
        C.distribution_number_of_common_neighbors(adj),
    )
    pd.testing.assert_series_equal(
        T.node_degree(np.matrix(adj.toarray())), T.node_degree(adj.toarray())
    )


def test_connection_probability_without_any_pair_within_reach_is_nan():
    adj, nodes = digraph(40, 0.2, seed=26), node_table(40, seed=26)
    assert np.isnan(
        C.connection_probability_within(adj, nodes, cols=["x", "y", "z"], max_dist=1e-3)
    )


def test_dimensions_given_as_an_array():
    adj = planted(40, 0.1, seed=3, sizes=(5, 4))
    for position in ("all", "spine"):
        as_list = T.get_k_skeleta_graph(adj, dimensions=[1, 2], position=position)
        as_array = T.get_k_skeleta_graph(adj, dimensions=np.array([1, 2]), position=position)
        assert list(as_list) == list(as_array)
        for key in as_list:
            assert_matrices_equal(as_list[key], as_array[key])
    by_list = add_rc_connections_skeleta(adj, 2, dimensions=[1, 2], seed=1)
    assert_matrices_equal(
        add_rc_connections_skeleta(adj, 2, dimensions=np.array([1, 2]), seed=1), by_list
    )


def test_betti_numbers_are_not_computed_with_an_array_of_approximations():
    with pytest.raises(AssertionError):
        T.betti_counts(digraph(20, 0.2, seed=1), approximation=np.array([-1, -1]))


def test_the_distance_dependent_random_graph_fits_its_parameters_itself():
    from connalysis.randomization import run_DD2_model

    adj, nodes = digraph(80, 0.15, seed=1), node_table(80, seed=1)
    graph, parameters = run_DD2_model(adj, nodes, return_params=True)
    assert graph.shape == (80, 80)
    pd.testing.assert_frame_equal(parameters, M.conn_prob_2nd_order_model(adj, nodes))


# ------------------------------------------------------------------------------- local


def test_neighborhood_of_a_set_includes_the_set_by_default():
    adj = digraph(30, 0.1, seed=4)
    centers = np.array([3, 7])
    around = L.neighborhood_of_set_indices(adj, centers)
    nodes = np.unique(np.concatenate([centers, around]))
    got, got_nodes = L.neighborhood_of_set(adj, centers, return_neighbors=True)
    np.testing.assert_array_equal(got_nodes, nodes)
    np.testing.assert_array_equal(got.toarray(), adj.toarray()[np.ix_(nodes, nodes)])
    without = L.neighborhood_of_set(adj, centers, include_centers=False)
    np.testing.assert_array_equal(without.toarray(), adj.toarray()[np.ix_(around, around)])


def test_duplicate_centers_are_an_error():
    adj = digraph(20, 0.2, seed=5)
    with pytest.raises(ValueError, match="unique"):
        L.neighborhood_indices(adj, all_nodes=False, centers=np.array([2, 2, 0]))
    with pytest.raises(ValueError, match="unique"):
        L.neighborhood_of_set_indices(adj, np.array([4, 4]))


# ------------------------------------------------------------------------------- classic


@pytest.mark.parametrize(
    "function",
    [
        lambda adj: C.common_neighbor_weight_bias(adj, direction="sideways"),
        lambda adj: C.common_neighbor_connectivity_bias(adj, direction="sideways"),
        lambda adj: C._analytical_expected_gini_curve(adj, direction="sideways"),
    ],
)
def test_an_unknown_direction_is_a_value_error(function):
    with pytest.raises(ValueError, match="sideways"):
        function(digraph(15, 0.2, seed=1))


def test_closeness_only_follows_connections_that_are_not_zero():
    """A stored zero joins the nodes of a strongly connected component, but is no path (scikit-network raised)."""
    cycle = sp.csr_matrix((np.array([1.0, 1.0, 0.0]), ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
    assert cycle.nnz == 3
    np.testing.assert_allclose(
        C.closeness_connected_components(cycle, directed=True), [2 / 3, 0, 0]
    )


def test_closeness_of_an_undirected_path_with_a_stored_zero():
    """The same defect without direction: scikit-network raised that the graph is not connected."""
    path = sp.csr_matrix((np.array([1.0, 0.0]), ([0, 1], [1, 2])), shape=(3, 3))
    np.testing.assert_allclose(C.closeness_connected_components(path), [0, 0, 0])


@pytest.mark.parametrize("direction", ["efferent", "afferent"])
def test_common_neighbour_weight_bias_of_a_sparse_array(direction):
    """For a sparse array ``*`` is elementwise, which made the original count the wrong common neighbours."""
    adj = digraph(40, 0.2, seed=7)
    adj.data = np.random.RandomState(7).rand(adj.nnz) + 0.1
    np.testing.assert_allclose(
        C.common_neighbor_weight_bias(sp.csr_array(adj), direction=direction),
        C.common_neighbor_weight_bias(adj, direction=direction),
    )


@pytest.mark.parametrize("direction", ["efferent", "afferent"])
def test_normalised_rich_club_curves(direction):
    adj = digraph(40, 0.2, seed=6, reciprocal=0.3)
    degree = {"efferent": "OUT", "afferent": "IN"}[direction]
    data = C.efficient_rich_club_curve(adj, direction=degree)
    analytical = C._analytical_expected_rich_club_curve(adj, direction=direction)
    for normalize, expected in (
        ("mean", lambda b, m, s: b / m),
        ("std", lambda b, m, s: (b - m) / s),
    ):
        got = C.normalized_rich_club_curve(
            adj, direction=direction, normalize=normalize, normalize_with="analytical"
        )
        n = len(analytical)
        values = expected(
            data.to_numpy()[:n], analytical["mean"].to_numpy(), analytical["std"].to_numpy()
        )
        np.testing.assert_allclose(got.to_numpy(), values, equal_nan=True)
        np.testing.assert_array_equal(got.index.to_numpy(), data.index.to_numpy()[:n])
    np.random.seed(1)
    shuffled = C.normalized_rich_club_curve(
        adj.copy(), direction=direction, normalize_with="shuffled"
    )
    np.random.seed(1)
    again = C.normalized_rich_club_curve(adj.copy(), direction=direction, normalize_with="shuffled")
    pd.testing.assert_series_equal(shuffled, again)
    coefficient = C.rich_club_coefficient(adj, direction=direction, normalize_with="analytical")
    np.testing.assert_allclose(
        coefficient,
        np.nanmean(
            C.normalized_rich_club_curve(adj, direction=direction, normalize_with="analytical")
        ),
    )
    with pytest.raises(Exception, match="Unknown normalization"):
        C.normalized_rich_club_curve(
            adj, direction=direction, normalize="median", normalize_with="analytical"
        )


# ------------------------------------------------------------------------------- topology


@pytest.mark.parametrize("seed", range(4))
def test_euler_characteristic_is_the_alternating_sum_of_the_simplex_counts(seed):
    adj = planted(40, 0.08, seed=seed, sizes=(5, 4, 3))
    counts = T.simplex_counts(adj).to_numpy()
    assert T.euler_characteristic(adj) == sum((-1) ** k * c for k, c in enumerate(counts))
    assert T.euler_characteristic(adj.toarray()) == T.euler_characteristic(adj)


def transitive_triangles_through(dense, center):
    """Orderings (v0, v1, v2) of three nodes with edges v0->v1, v0->v2 and v1->v2 that contain ``center``."""
    n = dense.shape[0]
    return sum(
        1
        for a in range(n)
        for b in range(n)
        for c in range(n)
        if len({a, b, c}) == 3
        and center in (a, b, c)
        and dense[a, b]
        and dense[a, c]
        and dense[b, c]
    )


@pytest.mark.parametrize(
    "edges, expected",
    [
        (
            [(0, 1), (0, 2), (1, 2)],
            0.5,
        ),  # 0 -> 1, 0 -> 2 and 1 -> 2: one triangle out of the 2 pairs of out-neighbours
        ([(0, 1), (1, 0), (0, 2), (1, 2)], 2 / 3),
    ],
)
def test_transitive_clustering_coefficient_of_small_neighbourhoods(edges, expected):
    dense = np.zeros((3, 3), dtype=int)
    for source, target in edges:
        dense[source, target] = 1
    assert T.tcc(dense) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(4))
def test_transitive_clustering_coefficient_of_a_node_in_a_larger_graph(seed):
    dense = digraph(25, 0.2, seed=seed, reciprocal=0.4).toarray().astype(int)
    for center in (0, 5, 11):
        around = T.neighbourhood(center, dense)
        assert T.tcc(dense, center=center) == pytest.approx(T.tcc(around, center=0))
    center = 3
    nodes = T.neighbourhood(center, dense)
    out_degree, in_degree = np.count_nonzero(dense[center]), np.count_nonzero(dense[:, center])
    reciprocal = C.reciprocal_connections(dense, chief_only=True, center=center)
    total = in_degree + out_degree
    denominator = total * (total - 1) - (in_degree * out_degree + reciprocal)
    neighbours = np.flatnonzero(dense[center] | dense[:, center])
    sub = dense[np.ix_([center, *neighbours], [center, *neighbours])]
    expected = transitive_triangles_through(sub, 0) / denominator if denominator else 0
    assert T.tcc(dense, center=center) == pytest.approx(expected)
    assert nodes.shape[0] == len(neighbours) + 1


def test_reciprocal_connections():
    dense = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    assert C.reciprocal_connections(dense) == 2
    assert C.reciprocal_connections(sp.csr_matrix(dense)) == 2
    assert C.reciprocal_connections(dense, chief_only=True) == 2
    assert C.reciprocal_connections(dense, chief_only=True, center=1) == 1


@pytest.mark.parametrize("maximal", [False, True])
def test_simplicial_rich_club_curve(maximal):
    adj = planted(50, 0.08, seed=1, sizes=(6, 5, 4))
    curve = T.simplicial_rich_club_curve(adj, maximal=maximal)
    assert isinstance(curve, pd.DataFrame) and list(curve.columns) == list(range(curve.shape[1]))
    values = curve.to_numpy()[~np.isnan(curve.to_numpy())]
    assert values.size and (values >= 0).all() and (values <= 1).all()
    if not maximal:  # the number of edges through a node is its degree
        degree_curve = C.efficient_rich_club_curve(adj).dropna()
        np.testing.assert_allclose(curve[1].dropna().to_numpy(), degree_curve.to_numpy())
        np.testing.assert_array_equal(
            curve[1].dropna().index.to_numpy(), degree_curve.index.to_numpy()
        )
    np.testing.assert_allclose(
        T.simplicial_rich_club_curve(adj.toarray(), maximal=maximal).to_numpy(),
        curve.to_numpy(),
        equal_nan=True,
    )
