"""The fast implementations agree with plain, slow definitions of the same quantities on many random inputs.

These complement the golden files: the golden cases pin what the original returned for a fixed set of inputs,
here each rewritten algorithm is compared with a direct, obviously correct version of it on random ones.
"""

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import hypergeom
from scipy.stats import t as student_t
from support.data import digraph, planted, positions, rng

from connalysis.modelling import modelling as M
from connalysis.network import classic as C
from connalysis.network import topology as T
from connalysis.network._components import strong_components

SEEDS = range(12)


def random_graphs(seeds=SEEDS, sizes=(6, 12, 25, 40), densities=(0.05, 0.15, 0.35), dtype=bool):
    for seed in seeds:
        r = rng(seed)
        n, p = int(r.choice(sizes)), float(r.choice(densities))
        yield digraph(n, p, seed=seed, reciprocal=float(r.choice([0.0, 0.3, 0.8])), dtype=dtype)


# ------------------------------------------------------------------------------------------------ topology


def brute_force_triads(adj):
    dense = adj.toarray()
    undirected = (dense + dense.T) != 0
    counts = np.zeros(13, dtype=int)
    for trio in itertools.combinations(range(dense.shape[0]), 3):
        a, b, c = trio
        if (
            int(undirected[a, b]) + int(undirected[a, c]) + int(undirected[b, c]) < 2
        ):  # not connected
            continue
        sub = dense[np.ix_(trio, trio)]
        order = np.argsort(-sub.sum(axis=0) - 10 * sub.sum(axis=1), kind="stable")
        counts[T.triad_dict[tuple(np.nonzero(sub[:, order][order].flatten())[0])]] += 1
    return counts


@pytest.mark.parametrize(
    "adj", list(random_graphs(sizes=(5, 9, 14, 20))), ids=lambda a: f"n{a.shape[0]}nnz{a.nnz}"
)
def test_triad_counts_match_a_brute_force_over_all_triples(adj):
    expected = brute_force_triads(adj)
    if expected.sum() == 0:  # the original divides by the number of triads
        with pytest.raises(ZeroDivisionError):
            T.count_triads_fully_connected(adj)
    else:
        np.testing.assert_array_equal(T.count_triads_fully_connected(adj), expected)


def sampled_triads_as_the_original_drew_them(adj, max_num_sampled):
    """The original algorithm: connected triples collected in a Python set, a sample drawn from ``list(set)``."""
    dense = adj.toarray()
    undirected = (dense + dense.T) != 0
    reach = np.triu(undirected.astype(int) @ undirected.astype(int), 1)
    triads = set()
    for x, y in zip(*np.nonzero(reach)):
        for z in np.nonzero(undirected[x] & undirected[y])[0]:
            triads.add(tuple(sorted([x, y, z])))
    triads = list(triads)
    counts = np.zeros(13)
    sample = np.random.choice(len(triads), np.minimum(max_num_sampled, len(triads)), replace=False)
    for index in sample:
        sub = dense[np.ix_(triads[index], triads[index])]
        order = np.argsort(-sub.sum(axis=0) - 10 * sub.sum(axis=1), kind="stable")
        counts[T.triad_dict[tuple(np.nonzero(sub[:, order][order].flatten())[0])]] += 1
    return ((len(triads) / len(sample)) * counts).astype(int)


@pytest.mark.parametrize("seed", range(8))
def test_sampled_triad_counts_are_those_of_the_original_sample_and_leave_the_same_random_state(
    seed,
):
    adj = digraph([20, 30, 40, 25][seed % 4], 0.15, seed=seed, reciprocal=0.4)
    limit = [10, 50, 200][seed % 3]
    np.random.seed(seed)
    expected = sampled_triads_as_the_original_drew_them(adj, limit)
    stream_after = np.random.rand(3)
    np.random.seed(seed)
    got = T.count_triads_fully_connected(adj, max_num_sampled=limit)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.random.rand(3), stream_after)


def test_bedge_counts_match_the_definition():
    for seed in range(6):
        adj = planted(40, 0.1, seed=seed, sizes=(5, 4, 4))
        simplices = T.list_simplices_by_dimension(adj)
        dense = adj.toarray().astype(int)
        result = T.bedge_counts(adj, simplices)
        assert np.isnan(result.iloc[0])
        for dim in range(1, len(simplices)):
            expected = sum(dense[np.ix_(s, s)].T for s in simplices.iloc[dim])
            np.testing.assert_array_equal(result.iloc[dim], expected)


def test_skeleton_matrix_of_a_simplex_list_has_each_edge_once():
    for seed in range(6):
        r = rng(seed)
        simplices = r.randint(0, 30, size=(int(r.randint(1, 200)), int(r.randint(2, 6))))
        for position in ("spine", "all"):
            got = T.extract_submatrix_from_simplex_list(simplices, 30, position=position)
            edges = np.unique(
                np.concatenate(
                    [
                        simplices[:, e]
                        for e in T._generate_abstract_edges_in_simplices(
                            simplices.shape[1] - 1, position
                        )
                    ]
                ),
                axis=0,
            )
            expected = sp.coo_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(30, 30)
            ).astype(bool)
            assert got.shape == (30, 30) and (got != expected).nnz == 0
            assert np.array_equal(got.row, expected.row) and np.array_equal(got.col, expected.col)


def test_all_faces_of_maximal_simplices_in_the_order_of_the_list_based_search():
    def reference(max_simplices):
        simplices = [list(level) for level in max_simplices]
        for k in range(len(max_simplices) - 1, 0, -1):
            for simplex in simplices[k]:
                for s in range(k, -1, -1):
                    face = tuple(simplex[:s] + simplex[s + 1 :])
                    if face not in simplices[k - 1]:
                        simplices[k - 1].append(face)
        return simplices

    for seed in range(8):
        r = rng(seed)
        levels = [[(int(v),) for v in range(3)]] + [
            [
                tuple(sorted(r.choice(12, k + 1, replace=False).tolist()))
                for _ in range(int(r.randint(1, 8)))
            ]
            for k in range(1, 5)
        ]
        assert T.get_all_simplices_from_max([list(level) for level in levels]) == reference(levels)


def test_degrees_of_sparse_matrices_with_repeated_entries_count_each_connection_once():
    for seed in range(6):
        adj = digraph(20, 0.2, seed=seed)
        # every entry stored twice in a CSR matrix, which is not in canonical format
        indices = np.concatenate(
            [np.tile(adj.indices[a:b], 2) for a, b in zip(adj.indptr[:-1], adj.indptr[1:])]
        )
        doubled = sp.csr_matrix(
            (np.ones(len(indices), dtype=bool), indices, 2 * adj.indptr), shape=adj.shape
        )
        assert not doubled.has_canonical_format
        for direction in (None, "IN", "OUT"):
            pd.testing.assert_series_equal(
                T.node_degree(doubled, direction=direction), T.node_degree(adj, direction=direction)
            )
        assert (
            doubled.nnz == 2 * adj.nnz and not doubled.has_canonical_format
        )  # the input is untouched


# ------------------------------------------------------------------------------------------------ classic


@pytest.mark.parametrize("seed", range(20))
def test_core_numbers_match_networkx(seed):
    r = rng(seed)
    n = int(r.choice([1, 2, 5, 12, 40, 90]))
    adj = digraph(
        n,
        float(r.choice([0.0, 0.02, 0.08, 0.3])),
        seed=seed,
        reciprocal=float(r.choice([0.0, 0.5])),
        dtype=float,
    )
    if seed % 3 == 0 and adj.nnz:
        adj.data[::4] = 0.0  # stored zeros are edges for networkx, so for core_number too
    expected = nx.core_number(nx.from_scipy_sparse_array(adj))
    got = C.core_number(adj)
    assert got == expected and list(got) == list(expected)
    assert all(type(v) is int for v in got.values())


def test_core_numbers_of_a_long_path_and_a_clique_with_tails():
    path = sp.diags([np.ones(299), np.ones(299)], [1, -1], format="csr")
    assert set(C.core_number(path).values()) == {1}
    clique = np.ones((6, 6), dtype=bool) ^ np.eye(6, dtype=bool)
    graph = sp.lil_matrix((12, 12), dtype=bool)
    graph[:6, :6] = clique
    for tail, node in zip(range(6, 12), (0, 0, 1, 2, 6, 7)):
        graph[tail, node] = True
    assert C.core_number(graph.tocsr()) == nx.core_number(nx.from_scipy_sparse_array(graph.tocsr()))


def reference_closeness(adj, directed):
    """The closeness per component, from networkx: (n - 1) / the sum of the distances to the other nodes."""
    n = adj.shape[0]
    total = np.zeros(n)
    if directed:
        graph = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
        components = list(nx.strongly_connected_components(graph))
    else:
        graph = nx.from_scipy_sparse_array(adj + adj.T)
        components = list(nx.connected_components(graph))
    for component in components:
        if len(component) < 2:
            continue
        sub = graph.subgraph(component)
        for node, value in nx.closeness_centrality(sub.reverse() if directed else sub).items():
            total[node] = value
    return total


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize(
    "adj",
    list(random_graphs(sizes=(8, 20, 60), densities=(0.03, 0.08, 0.2))),
    ids=lambda a: f"n{a.shape[0]}nnz{a.nnz}",
)
def test_closeness_matches_networkx(adj, directed):
    np.testing.assert_allclose(
        C.closeness_connected_components(adj, directed=directed),
        reference_closeness(adj, directed),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "adj", list(random_graphs(sizes=(15, 40, 90))), ids=lambda a: f"n{a.shape[0]}nnz{a.nnz}"
)
def test_rich_club_curve_matches_the_definition_with_submatrices(adj):
    for direction in ("TOTAL", "IN", "OUT"):
        for matrix in (adj, adj.astype(float) * 2):
            m = sp.csc_matrix(matrix)
            degrees = T.node_degree(m, direction=None if direction == "TOTAL" else direction)
            if m.dtype == bool:
                levels, positions_ = degrees, np.arange(1, degrees.max() + 1)
                labels = positions_
            else:
                labels, positions_, levels = C._bin_degrees(degrees)
            expected = [
                m[np.ix_(levels >= i, levels >= i)].sum()
                / ((levels >= i).sum() * ((levels >= i).sum() - 1))
                for i in positions_
            ]
            with np.errstate(divide="ignore", invalid="ignore"):
                got = C.rich_club_curve(matrix, direction=direction)
            np.testing.assert_allclose(
                got.to_numpy(), np.array(expected, dtype=float), rtol=1e-12, equal_nan=True
            )
            np.testing.assert_array_equal(got.index.to_numpy(), np.asarray(labels))


@pytest.mark.parametrize(
    "adj", list(random_graphs(sizes=(15, 30, 60)))[:6], ids=lambda a: f"n{a.shape[0]}"
)
def test_expected_rich_club_curve_matches_scalar_calls_of_scipy(adj):
    for direction in ("efferent", "afferent"):
        indegree, outdegree = np.array(adj.sum(axis=0))[0], np.array(adj.sum(axis=1))[:, 0]
        degrees = indegree if direction == "afferent" else outdegree
        expected_mean, expected_std = [], []
        for d in np.arange(1, degrees.max() + 1):
            valid = np.nonzero(degrees >= d)[0]
            i_v, o_v = indegree[valid], outdegree[valid]
            stats_ = np.array(
                [
                    hypergeom.stats(a, b, c)
                    for a, b, c in zip(indegree.sum() - i_v, i_v.sum() - i_v, o_v)
                ]
            )
            pairs = (degrees >= d).sum() * ((degrees >= d).sum() - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                expected_mean.append(stats_[:, 0].sum() / pairs)
                expected_std.append(np.sqrt(stats_[:, 1].sum()) / pairs)
        with np.errstate(divide="ignore", invalid="ignore"):
            got = C._analytical_expected_rich_club_curve(adj, direction=direction)
        np.testing.assert_allclose(got["mean"], expected_mean, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(got["std"], expected_std, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "adj", list(random_graphs(sizes=(8, 30, 70))), ids=lambda a: f"n{a.shape[0]}nnz{a.nnz}"
)
def test_common_neighbour_distribution_matches_the_dense_histogram(adj):
    for direction in ("efferent", "afferent"):
        a = adj.tocsc().astype(int)
        cn = (a @ a.T if direction == "efferent" else a.T @ a).toarray()
        values = cn[np.triu_indices_from(cn, 1)]
        bins = np.arange(0, values.max() + 2)
        expected = pd.Series(np.histogram(values, bins=bins)[0], index=bins[:-1])
        pd.testing.assert_series_equal(
            C.distribution_number_of_common_neighbors(adj, direction=direction), expected
        )


@pytest.mark.parametrize(
    "adj", list(random_graphs(sizes=(8, 30, 70))), ids=lambda a: f"n{a.shape[0]}nnz{a.nnz}"
)
def test_clustering_coefficient_matches_the_matrix_power_definition(adj):
    binary = adj.astype(int)
    symmetric = (binary + binary.T).toarray()
    numerator = 0.5 * np.diag(np.linalg.matrix_power(symmetric, 3))
    degree = T.node_degree(adj)
    denominator = degree * (degree - 1) - 2 * T.rc_submatrix(adj).toarray().sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = numerator / denominator
        for form in (adj, adj.toarray()):
            np.testing.assert_allclose(
                np.asarray(C.ccc(form)), np.asarray(expected), rtol=1e-12, equal_nan=True
            )


@pytest.mark.parametrize("seed", range(6))
def test_strongly_connected_component_labels_follow_the_numbering_of_scipy_up_to_1_17(seed):
    from scipy.sparse import csgraph

    adj = digraph([12, 30, 60, 100, 150, 200][seed], 0.03, seed=seed)
    n_components, labels = strong_components(adj)
    scipy_count, scipy_labels = csgraph.connected_components(
        adj, directed=True, connection="strong"
    )
    assert n_components == scipy_count and labels.dtype == scipy_labels.dtype
    # the same partition (whatever the numbering), and a numbering where every component completes before the
    # components it can reach, which is what the iterative Tarjan algorithm produces
    assert len(set(zip(labels.tolist(), scipy_labels.tolist()))) == n_components
    coo = adj.tocoo()
    assert np.all(labels[coo.row] >= labels[coo.col])


@pytest.mark.parametrize("seed", range(5))
def test_connectivity_bias_matches_a_least_squares_fit_of_the_full_design(seed):
    r = rng(seed)
    n = int(r.choice([12, 20, 30]))
    adj = digraph(n, 0.2, seed=seed, reciprocal=0.4)
    nodes = pd.DataFrame(positions(n, seed=seed, dims=3), columns=list("xyz"))
    a = adj.astype(int)
    cn = (a @ a.T).toarray().ravel().astype(float)
    connected = adj.toarray().ravel().astype(float)
    from scipy.spatial import distance

    dist = distance.squareform(distance.pdist(nodes.to_numpy())).ravel()
    for with_distance in (False, True):
        columns = [np.ones(n * n), connected] + ([dist] if with_distance else [])
        design = np.column_stack(columns)
        coef = np.linalg.lstsq(design, cn, rcond=None)[0]
        residuals = cn - design @ coef
        dof = n * n - design.shape[1]
        cov = np.linalg.inv(design.T @ design) * (residuals @ residuals) / dof
        t_stat = coef[1] / np.sqrt(cov[1, 1])
        expected = (
            2 * student_t.sf(abs(t_stat), dof),
            coef[1] / coef[0],
            100 * (coef[2] if with_distance else 0) / coef[0],
        )
        got = C.common_neighbor_connectivity_bias(
            adj, nodes, cols_location=list("xyz") if with_distance else None
        )
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-12)


# ------------------------------------------------------------------------------------------------ modelling


def random_bins_problem(seed):
    r = rng(seed)
    n, m = int(r.randint(3, 40)), int(r.randint(3, 40))
    dependency = np.round(r.random_sample((n, m)) * 100) / 4  # many values on bin edges
    dependency[r.random_sample((n, m)) < 0.05] = np.nan
    adj = sp.csr_matrix(r.random_sample((n, m)) < 0.2)
    edges = np.sort(r.choice(np.arange(0, 30, 2.5), size=int(r.randint(2, 9)), replace=False))
    return adj, dependency, edges


@pytest.mark.parametrize("seed", range(25))
def test_counting_pairs_in_bins_matches_one_pass_per_bin(seed):
    adj, dependency, edges = random_bins_problem(seed)
    second = (
        np.round(rng(seed + 100).random_sample(adj.shape) * 4) - 2
    )  # a bipolar-like second dependency
    for dependencies, bins in (
        ([dependency], [edges]),
        ([dependency, second], [edges, np.array([-2, 0, 2])]),
    ):
        for weights in (None, "ints"):
            matrix = adj
            if weights:
                matrix = sp.csr_matrix(
                    adj.astype(float).multiply(rng(seed).randint(1, 5, size=adj.shape))
                )
            fast = M._count_pairs_in_bins(matrix, dependencies, bins)
            assert fast is not None
            slow = M._count_pairs_in_bins_by_looping(matrix, dependencies, bins)
            for got, expected in zip(fast, slow):
                np.testing.assert_array_equal(got, expected)
                assert got.dtype == expected.dtype


def test_counting_pairs_falls_back_to_the_original_loop_when_sums_would_depend_on_the_order():
    adj, dependency, edges = random_bins_problem(3)
    fractional = sp.csr_matrix(adj.astype(float) * 0.1)
    assert M._count_pairs_in_bins(fractional, [dependency], [edges]) is None
    assert M._count_pairs_in_bins(adj, [dependency], [edges[::-1]]) is None  # edges that decrease
    counts = M._extract_dependent_p_conn(fractional, [dependency], [edges])
    expected = M._count_pairs_in_bins_by_looping(fractional, [dependency], [edges])
    np.testing.assert_array_equal(counts[1], expected[1])
    np.testing.assert_array_equal(counts[2], expected[0])


@pytest.mark.parametrize("seed", range(6))
def test_group_bias_matches_one_submatrix_per_pair_of_groups(seed):
    r = rng(seed)
    n = 60
    adj = sp.csr_matrix(r.random_sample((n, n)) < 0.1)
    table = pd.DataFrame({"g": np.array(list("abcd"))[r.randint(0, 4, size=n)]})
    codes = pd.Categorical(table["g"])
    counts = np.array(
        [[adj[np.ix_(codes.codes == i, codes.codes == j)].nnz for j in range(4)] for i in range(4)]
    )
    sizes = table["g"].value_counts()[codes.categories].to_numpy()
    prob = counts / (sizes[:, None] * sizes[None, :] + 1e-9)
    prob = prob.mean() * prob**1.5 / (prob**1.5).mean()
    for axis in (0, 1):
        weights = pd.Series(
            np.nanmean(prob, axis=axis) / np.sqrt(np.nanmean(prob)), index=codes.categories
        )
        np.testing.assert_allclose(
            M.per_group_in_out_bias_model(adj, table, "g", axis),
            weights[table["g"]].to_numpy(),
            rtol=1e-12,
        )


def test_strong_components_of_a_matrix_that_is_not_square_is_an_error_as_in_scipy():
    with pytest.raises(ValueError, match="shape"):
        strong_components(sp.csr_matrix(np.ones((3, 4))))
    with pytest.raises(ValueError, match="shape"):
        C.connected_components(sp.csr_matrix(np.ones((3, 4))), directed=True, connection="strong")


def test_betti_numbers_in_dimensions_without_approximation_are_exact():
    """flagser's approximation is a heuristic whose numbers differ between machines, so the golden cases of it
    only pin the shape of the result; here the dimensions that are not approximated must be exact."""
    adj = planted(40, 0.1, seed=3)
    exact = T.betti_counts(adj, max_dim=4)
    for approximation in ([-1, 3, -1, -1, -1], [-1, -1, 2, 2, -1], [-1, -1, -1, -1, 4], [-1]):
        padded = approximation + [-1] * (5 - len(approximation))
        got = T.betti_counts(adj, approximation=approximation, max_dim=4)
        pd.testing.assert_index_equal(got.index, exact.index)
        for dim, value in enumerate(padded):
            if value == -1:
                assert got[dim] == exact[dim]
    pd.testing.assert_series_equal(
        T.betti_counts(adj, approximation=[100000] * 5, max_dim=4), exact
    )
