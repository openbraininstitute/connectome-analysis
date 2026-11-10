"""``_extract_dependent_p_conn``: counts of connected and of all pairs of neurons in bins of D dependency matrices.

This is the heavy loop of the model building (one pass over the full matrices per bin combination). Distances are
chosen exact (integer-valued floats) wherever a case is about bin edges, so the comparisons ``>= lower`` and
``< upper`` (``<= upper`` for the last bin) are decided without rounding.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, positions, rng

from connalysis.modelling import modelling as M

from .modelling_common import pairwise_distances, planted_dd

EXTRACT = M._extract_dependent_p_conn


def geometry(n=60, seed=0, scale=300.0, step=None):
    """Distance matrix (NaN diagonal) of random positions, optionally rounded to multiples of ``step``."""
    dist = pairwise_distances(positions(n, seed=seed, scale=scale))
    if step:
        dist = np.round(dist / step) * step
    np.fill_diagonal(dist, np.nan)
    return dist


def depth_signs(n=60, seed=0):
    """Bipolar matrix as the 3rd-order model builds it (rounded depths, so zeros occur off the diagonal)."""
    depth = np.round(positions(n, seed=seed + 1, dims=1, scale=300.0)[:, 0] / 15.0)
    return np.sign(depth[:, None] - depth[None, :])


def chain():
    """Ten neurons on a line 10 apart (distances 10..90 are exact) and a fixed connection pattern."""
    x = np.arange(10) * 10.0
    dist = np.abs(x[:, None] - x[None, :])
    np.fill_diagonal(dist, np.nan)
    i, j = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    dense = ((3 * i + 5 * j) % 7 < 3) & (i != j)
    dense[0, 9] = dense[1, 9] = (
        True  # connections at the two largest distances (80 and 90), the pattern has none
    )
    return sp.csr_matrix(dense), dist


# ----------------------------------------------------------------------------- one dependency


@case("modelling.extract_dependent_p_conn.one_dimension_bool_csr")
def _():
    adj = digraph(60, 0.15, seed=1)
    return EXTRACT(adj, [geometry(60, seed=1)], [np.arange(0, 525, 25)])


@case("modelling.extract_dependent_p_conn.one_dimension_values_on_bin_edges")
def _():
    adj = digraph(60, 0.15, seed=2)
    return EXTRACT(adj, [geometry(60, seed=2, step=5.0)], [np.arange(0, 525, 25)])


@case("modelling.extract_dependent_p_conn.edges_are_lower_inclusive_upper_exclusive_except_last")
def _():
    adj, dist = chain()
    exact = EXTRACT(
        adj, [dist], [np.arange(0, 100, 10)]
    )  # distance 10 is in [10, 20); distance 90 only in the closed last bin
    uneven = EXTRACT(adj, [dist], [np.array([0, 20, 40, 60, 80, 90])])
    short = EXTRACT(
        adj, [dist], [np.array([0, 20, 40, 60, 80])]
    )  # the last bin [60, 80] now holds distance 80 as well
    return exact, uneven, short


@case("modelling.extract_dependent_p_conn.zero_distance_off_the_diagonal_is_counted")
def _():
    adj, dist = chain()
    dist = dist.copy()
    dist[2, 5] = dist[5, 2] = 0.0
    return EXTRACT(adj, [dist], [np.array([0, 10, 50, 90])])


@case("modelling.extract_dependent_p_conn.nan_and_infinite_values_belong_to_no_bin")
def _():
    adj = digraph(40, 0.25, seed=3)
    dist = geometry(40, seed=3, step=5.0)
    r = rng(30)
    flat = dist.reshape(-1)
    flat[r.randint(0, flat.size, 60)] = np.nan
    flat[r.randint(0, flat.size, 20)] = np.inf
    flat[r.randint(0, flat.size, 20)] = -np.inf
    return EXTRACT(adj, [dist], [np.arange(0, 525, 50)])


@case("modelling.extract_dependent_p_conn.values_outside_all_bins")
def _():
    adj = digraph(40, 0.25, seed=4)
    dist = geometry(40, seed=4, step=5.0)
    return EXTRACT(adj, [dist], [np.array([100.0, 150.0, 300.0])])  # data spans about 5..500


@case("modelling.extract_dependent_p_conn.negative_bin_edges")
def _():
    adj = digraph(30, 0.3, seed=5)
    values = (rng(50).randint(-3, 4, size=(30, 30))).astype(float)
    return EXTRACT(adj, [values], [np.array([-3, -1, 0, 2, 3])])


@case("modelling.extract_dependent_p_conn.single_bin")
def _():
    adj = digraph(30, 0.3, seed=6)
    return EXTRACT(adj, [geometry(30, seed=6)], [np.array([0.0, 1000.0])])


@case("modelling.extract_dependent_p_conn.single_bin_value_on_its_upper_edge")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([0.0, 90.0])])


@case("modelling.extract_dependent_p_conn.zero_width_last_bin_holds_the_edge_value")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([40.0, 40.0])]), EXTRACT(
        adj, [dist], [np.array([0.0, 40.0, 40.0])]
    )


@case("modelling.extract_dependent_p_conn.repeated_inner_edge")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([0.0, 30.0, 30.0, 60.0, 90.0])])


@case("modelling.extract_dependent_p_conn.decreasing_edges_select_nothing")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([90.0, 60.0, 30.0, 0.0])])


@case("modelling.extract_dependent_p_conn.single_edge_means_no_bins")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([0.0])])


@case("modelling.extract_dependent_p_conn.no_edges_at_all")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([])])


@case("modelling.extract_dependent_p_conn.bins_without_pairs_give_nan")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.array([0, 5, 8, 10, 15, 100, 200, 300])])


@case("modelling.extract_dependent_p_conn.many_narrow_bins")
def _():
    adj = digraph(50, 0.2, seed=7)
    return EXTRACT(adj, [geometry(50, seed=7)], [np.linspace(0, 520, 261)])


@case("modelling.extract_dependent_p_conn.bins_given_as_python_lists")
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [[0, 20, 40, 90]])


@case("modelling.extract_dependent_p_conn.integer_valued_dependency")
def _():
    adj = digraph(30, 0.3, seed=8)
    values = rng(80).randint(0, 12, size=(30, 30))
    return EXTRACT(adj, [values], [np.array([0, 3, 6, 12])])


@case("modelling.extract_dependent_p_conn.float32_dependency")
def _():
    adj = digraph(30, 0.3, seed=9)
    return EXTRACT(
        adj, [geometry(30, seed=9, step=5.0).astype(np.float32)], [np.arange(0, 525, 75)]
    )


# ----------------------------------------------------------------------------- several dependencies


@case("modelling.extract_dependent_p_conn.two_dimensions_distance_and_bipolar")
def _():
    adj = digraph(60, 0.15, seed=10)
    return EXTRACT(
        adj,
        [geometry(60, seed=10), depth_signs(60, seed=10)],
        [np.arange(0, 525, 25), [-1.0, 0, 1.0]],
    )


@case("modelling.extract_dependent_p_conn.two_dimensions_with_bipolar_edges_as_integers")
def _():
    adj = digraph(60, 0.15, seed=11)
    return EXTRACT(
        adj, [geometry(60, seed=11), depth_signs(60, seed=11)], [np.arange(0, 525, 50), [-1, 0, 1]]
    )


@case("modelling.extract_dependent_p_conn.two_dimensions_all_values_on_edges")
def _():
    adj = digraph(30, 0.3, seed=12)
    first = rng(120).randint(0, 7, size=(30, 30)).astype(float)
    second = rng(121).randint(0, 5, size=(30, 30)).astype(float)
    return EXTRACT(adj, [first, second], [np.array([0, 2, 4, 6]), np.array([0, 2, 4])])


@case("modelling.extract_dependent_p_conn.two_dimensions_nan_in_only_one_of_them")
def _():
    adj = digraph(40, 0.25, seed=13)
    bip = depth_signs(40, seed=13)
    bip[3, :] = np.nan
    bip[:, 7] = np.nan
    return EXTRACT(adj, [geometry(40, seed=13), bip], [np.arange(0, 525, 75), [-1, 0, 1]])


@case("modelling.extract_dependent_p_conn.two_dimensions_single_bin_in_the_second")
def _():
    adj = digraph(30, 0.3, seed=14)
    return EXTRACT(
        adj, [geometry(30, seed=14), depth_signs(30, seed=14)], [np.arange(0, 525, 100), [-1, 1]]
    )


@case("modelling.extract_dependent_p_conn.two_dimensions_float_and_integer_dtypes")
def _():
    adj = digraph(40, 0.25, seed=31)
    levels = rng(310).randint(0, 4, size=(40, 40))  # int64 dependency next to a float one
    return EXTRACT(
        adj,
        [geometry(40, seed=31, step=5.0), levels],
        [np.arange(0, 525, 100), np.array([0, 2, 3])],
    )


@case("modelling.extract_dependent_p_conn.boolean_dependency")
def _():
    adj = digraph(40, 0.25, seed=32)
    same_group = (
        rng(320).randint(0, 3, size=40)[:, None] == rng(320).randint(0, 3, size=40)[None, :]
    )
    return EXTRACT(adj, [same_group], [np.array([0, 1, 2])])


@case("modelling.extract_dependent_p_conn.dependency_in_transposed_and_fortran_layout")
def _():
    adj = digraph(30, 0.3, seed=33)
    values = (
        rng(330).randint(0, 12, size=(30, 30)).astype(float)
    )  # not symmetric: layout mistakes would show
    bins = [np.array([0, 3, 6, 12])]
    plain = EXTRACT(adj, [np.ascontiguousarray(values.T)], bins)
    return plain, EXTRACT(adj, [values.T], bins), EXTRACT(adj, [np.asfortranarray(values.T)], bins)


@case("modelling.extract_dependent_p_conn.three_dimensions")
def _():
    adj = digraph(40, 0.25, seed=15)
    dz = np.abs(rng(150).randint(-4, 5, size=(40, 40))).astype(float)
    return EXTRACT(
        adj,
        [geometry(40, seed=15), depth_signs(40, seed=15), dz],
        [np.arange(0, 525, 100), [-1, 0, 1], [0, 2, 4]],
    )


@case("modelling.extract_dependent_p_conn.no_dependency_at_all")
def _():
    adj = digraph(20, 0.3, seed=16)
    return EXTRACT(adj, [], [])


@case("modelling.extract_dependent_p_conn.planted_distance_law_with_bipolar_dependency")
def _():
    adj, nodes = planted_dd(120, seed=3, a=0.5, b=0.008, bipolar=(0.3, 0.012))
    xyz = nodes[["x", "y", "z"]].to_numpy()
    dist = pairwise_distances(xyz)
    np.fill_diagonal(dist, np.nan)
    depth = nodes["depth"].to_numpy()
    bip = np.sign(depth[:, None] - depth[None, :])
    return EXTRACT(adj, [dist, bip], [np.arange(0, 700, 50), [-1, 0, 1]])


# ----------------------------------------------------------------------------- adjacency content


@case("modelling.extract_dependent_p_conn.non_square_pathway_matrix")
def _():
    r = rng(17)
    adj = sp.csr_matrix(r.random_sample((13, 21)) < 0.3)
    dist = pairwise_distances(
        positions(13, seed=17, scale=300.0), positions(21, seed=18, scale=300.0)
    )
    return EXTRACT(adj, [dist], [np.arange(0, 525, 75)])


@case("modelling.extract_dependent_p_conn.no_connections")
def _():
    _, dist = chain()
    return EXTRACT(sp.csr_matrix((10, 10), dtype=bool), [dist], [np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.all_pairs_connected")
def _():
    _, dist = chain()
    adj = sp.csr_matrix(np.ones((10, 10), dtype=bool))
    return EXTRACT(adj, [dist], [np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.self_connections_are_dropped_by_the_nan_diagonal")
def _():
    adj, dist = chain()
    with_self = adj.tolil()
    with_self.setdiag(True)
    return EXTRACT(with_self.tocsr(), [dist], [np.arange(0, 100, 10)]), EXTRACT(
        adj, [dist], [np.arange(0, 100, 10)]
    )


@case("modelling.extract_dependent_p_conn.self_connections_count_without_nan_diagonal")
def _():
    adj, dist = chain()
    dist = np.where(np.isnan(dist), 0.0, dist)
    with_self = adj.tolil()
    with_self.setdiag(True)
    return EXTRACT(with_self.tocsr(), [dist], [np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.explicit_zeros_stored_in_a_float_matrix")
def _():
    adj, dist = chain()
    weights = sp.csr_matrix(adj.astype(float))
    weights.data[::3] = 0.0
    return weights.nnz, EXTRACT(weights, [dist], [np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.explicit_false_stored_in_a_bool_matrix")
def _():
    adj, dist = chain()
    flags = sp.csr_matrix(adj)
    flags.data[::4] = False
    return flags.nnz, EXTRACT(flags, [dist], [np.arange(0, 100, 10)])


# ----------------------------------------------------------------------------- adjacency formats


def _format_inputs(seed):
    adj = digraph(30, 0.25, seed=seed)
    return adj, geometry(30, seed=seed), [np.arange(0, 525, 75)]


def _format_case(name, convert):
    @case(f"modelling.extract_dependent_p_conn.format_{name}")
    def _():
        adj, dist, bins = _format_inputs(20)
        return EXTRACT(convert(adj), [dist], bins)


_format_case("csr", lambda a: a.tocsr())
_format_case("csc", lambda a: a.tocsc())
_format_case("lil", lambda a: a.tolil())
_format_case("dense_ndarray", lambda a: a.toarray())
_format_case("csr_array", lambda a: sp.csr_array(a))
_format_case("csc_array", lambda a: sp.csc_array(a))
_format_case("lil_array", lambda a: sp.lil_array(a))


@case("modelling.extract_dependent_p_conn.format_coo")
def _():
    adj, dist, bins = _format_inputs(20)
    return EXTRACT(adj.tocoo(), [dist], bins)  # coo matrices cannot be indexed


@case("modelling.extract_dependent_p_conn.format_dense_matches_sparse")
def _():
    adj, dist, bins = _format_inputs(21)
    return EXTRACT(adj, [dist], bins), EXTRACT(adj.toarray(), [dist], bins)


# ----------------------------------------------------------------------------- dtypes and weights


@case("modelling.extract_dependent_p_conn.dtypes_of_the_adjacency")
def _():
    adj, dist, bins = _format_inputs(22)
    return {
        np.dtype(dt).name: EXTRACT(adj.astype(dt), [dist], bins)
        for dt in (bool, np.int8, np.uint8, np.int32, np.int64, np.float32, np.float64)
    }


@case("modelling.extract_dependent_p_conn.dense_dtypes_of_the_adjacency")
def _():
    adj, dist, bins = _format_inputs(23)
    return {
        np.dtype(dt).name: EXTRACT(adj.toarray().astype(dt), [dist], bins)
        for dt in (bool, np.int16, np.int64, np.float32, np.float64)
    }


def _weighted_adj(seed, values):
    adj = digraph(30, 0.3, seed=seed)
    data = np.asarray(values)[rng(seed + 900).randint(0, len(values), size=adj.nnz)]
    return sp.csr_matrix((data, adj.indices.copy(), adj.indptr.copy()), shape=adj.shape)


@case("modelling.extract_dependent_p_conn.weights_integer_valued_floats_are_summed")
def _():
    adj = _weighted_adj(24, [1.0, 2.0, 3.0, 4.0, 5.0])
    return EXTRACT(adj, [geometry(30, seed=24)], [np.arange(0, 525, 75)])


@case("modelling.extract_dependent_p_conn.weights_fractional_sums_are_truncated")
def _():
    # binary fractions, so the sums are exact in floating point whatever the summation order
    adj = _weighted_adj(25, [0.25, 0.5, 1.5, 2.75])
    dist = geometry(30, seed=25)
    bins = [np.arange(0, 525, 75)]
    return EXTRACT(adj, [dist], bins), adj.sum()


@case("modelling.extract_dependent_p_conn.weights_negative_sums_truncate_toward_zero")
def _():
    adj = _weighted_adj(26, [-0.25, -0.5, -1.5, -2.75])
    return EXTRACT(adj, [geometry(30, seed=26)], [np.arange(0, 525, 75)])


@case("modelling.extract_dependent_p_conn.weights_mixed_signs_with_exact_cancellation")
def _():
    adj = _weighted_adj(27, [-1.0, -0.5, 0.5, 1.0, 2.0])
    return EXTRACT(adj, [geometry(30, seed=27)], [np.arange(0, 525, 75)])


@case("modelling.extract_dependent_p_conn.weights_on_pairs_without_a_usable_dependency_are_dropped")
def _():
    # heavy weights on the diagonal (NaN distance), on a pair with infinite distance and on pairs beyond the last edge (300)
    adj = _weighted_adj(34, [0.5, 1.5, 2.0]).tolil()
    adj.setdiag(100.0)
    dist = geometry(30, seed=34, step=5.0)
    dist[2, 7] = dist[7, 2] = np.inf
    adj[2, 7] = adj[7, 2] = 50.0
    far = np.argwhere(np.nan_to_num(dist, nan=0.0, posinf=0.0) > 300)[:3]
    for i, j in far:
        adj[i, j] = 70.0
    return EXTRACT(adj.tocsr(), [dist], [np.arange(0, 301, 100)]), len(far)


@case("modelling.extract_dependent_p_conn.weights_random_reals_truncated")
def _():
    adj = digraph(40, 0.3, seed=28)
    data = 1.0 - rng(280).random_sample(adj.nnz)
    adj = sp.csr_matrix((data * 3.0, adj.indices.copy(), adj.indptr.copy()), shape=adj.shape)
    return EXTRACT(adj, [geometry(40, seed=28)], [np.arange(0, 525, 75)])


@case("modelling.extract_dependent_p_conn.weights_sparse_vs_dense_agree")
def _():
    adj = _weighted_adj(29, [0.5, 1.0, 1.5])
    dist, bins = geometry(30, seed=29), [np.arange(0, 525, 75)]
    return EXTRACT(adj, [dist], bins), EXTRACT(adj.toarray(), [dist], bins)


# ----------------------------------------------------------------------------- invalid input


@case("modelling.extract_dependent_p_conn.more_bins_than_dependencies", check_message=True)
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist], [np.arange(0, 100, 10), np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.more_dependencies_than_bins", check_message=True)
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist, dist], [np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.dependency_of_wrong_shape", check_message=True)
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist[:9, :9]], [np.arange(0, 100, 10)])


@case("modelling.extract_dependent_p_conn.second_dependency_of_wrong_shape", check_message=True)
def _():
    adj, dist = chain()
    return EXTRACT(adj, [dist, dist[:, :9]], [np.arange(0, 100, 10), np.arange(0, 100, 10)])


# ----------------------------------------------------------------------------- inputs are left alone


@case("modelling.extract_dependent_p_conn.inputs_are_not_modified")
def _():
    adj = digraph(40, 0.25, seed=30).astype(float)
    adj = sp.csr_matrix(adj)
    adj.data[::5] = 0.0  # explicit zeros stay stored
    dist = geometry(40, seed=30, step=5.0)
    dist[3, 4] = np.nan
    bip = depth_signs(40, seed=30)
    before = (adj.data.copy(), adj.indices.copy(), adj.indptr.copy(), dist.copy(), bip.copy())
    result = EXTRACT(adj, [dist, bip], [np.arange(0, 525, 100), [-1, 0, 1]])
    after = (adj.data, adj.indices, adj.indptr, dist, bip)
    return (
        result,
        [bool(np.array_equal(b, a, equal_nan=True)) for b, a in zip(before, after)],
        adj.nnz,
    )


@case("modelling.extract_dependent_p_conn.bins_are_not_modified")
def _():
    adj, dist = chain()
    edges = np.array([0.0, 20.0, 40.0, 90.0])
    EXTRACT(adj, [dist], [edges])
    return edges


@case("modelling.extract_dependent_p_conn.unsorted_column_indices_and_duplicates_in_csr")
def _():
    dist = np.array(
        [[np.nan, 10, 20, 30], [10, np.nan, 40, 5], [20, 40, np.nan, 25], [30, 5, 25, np.nan]]
    )
    # entries (0,2) and (2,1) are stored twice, and the columns of rows 0 and 1 are stored out of order
    data = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    indices = np.array([2, 1, 2, 3, 0, 1, 0, 1, 2, 3])
    indptr = np.array([0, 3, 5, 8, 10])
    adj = sp.csr_matrix((data, indices, indptr), shape=(4, 4))
    return adj.has_canonical_format, EXTRACT(adj, [dist], [np.array([0, 10, 20, 30, 40])])
