"""``density``, ``get_pairs_within`` and ``connection_probability_within`` in ``connalysis.network.classic``."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, node_table, weighted

from cases.classic_inputs import complete, from_edges, spatial_digraph, with_explicit_zeros
from connalysis.network import classic as C

TYPES = ("directed", "undirected", "reciprocal")

# ----------------------------------------------------------------------------- density


@case("classic.density.default")
def _():
    return C.density(digraph(30, 0.15, seed=1, reciprocal=0.5))


@case("classic.density.types_on_asymmetric_matrix")
def _():
    adj = digraph(30, 0.15, seed=1, reciprocal=0.5)
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.density.types_on_symmetric_matrix")
def _():
    adj = digraph(30, 0.1, seed=2)
    adj = (adj + adj.T).astype(bool)
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.density.undirected_skip_symmetry_check")
def _():
    # skipping the check means the asymmetric matrix is used as it is, i.e. its directed density
    adj = digraph(30, 0.15, seed=1, reciprocal=0.5)
    return [C.density(adj, type="undirected", skip_symmetry_check=skip) for skip in (False, True)]


@case("classic.density.skip_symmetry_check_is_ignored_for_other_types")
def _():
    adj = digraph(30, 0.15, seed=1, reciprocal=0.5)
    return [C.density(adj, type=t, skip_symmetry_check=True) for t in ("directed", "reciprocal")]


@case("classic.density.unknown_type_is_the_directed_density")
def _():
    adj = digraph(30, 0.15, seed=1, reciprocal=0.5)
    return [C.density(adj, type="mutual"), C.density(adj)]


@case("classic.density.dense_input")
def _():
    dense = digraph(20, 0.2, seed=3, reciprocal=0.4).toarray()
    return {t: C.density(dense, type=t) for t in TYPES}


@case("classic.density.weighted_input_is_binarised")
def _():
    adj = weighted(25, 0.2, seed=4)
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.density.input_formats")
def _():
    base = digraph(25, 0.2, seed=5, reciprocal=0.4)
    inputs = [base.tocsc(), base.tocoo(), base.tolil(), base.astype(int), base.astype(float)]
    return [[C.density(m, type=t) for t in TYPES] for m in inputs]


@case("classic.density.sparse_array_input")
def _():
    adj = sp.csr_array(digraph(25, 0.2, seed=5, reciprocal=0.4))
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.density.no_edges")
def _():
    return {t: C.density(sp.csr_matrix((5, 5), dtype=bool), type=t) for t in TYPES}


@case("classic.density.complete_graph")
def _():
    return {t: C.density(complete(6), type=t) for t in TYPES}


@case("classic.density.two_nodes")
def _():
    one_way, both_ways = from_edges(2, [(0, 1)]), from_edges(2, [(0, 1), (1, 0)])
    return {
        f"{name}_{t}": C.density(m, type=t)
        for name, m in (("one_way", one_way), ("both", both_ways))
        for t in TYPES
    }


@case("classic.density.single_node")
def _():
    return C.density(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.density.diagonal_is_counted")
def _():
    adj = digraph(20, 0.2, seed=6, reciprocal=0.3).tolil()
    adj[3, 3] = True
    adj[7, 7] = True
    adj = adj.tocsr()
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.density.explicit_zeros")
def _():
    adj = with_explicit_zeros(digraph(20, 0.25, seed=7, reciprocal=0.5), every=3)
    return {t: C.density(adj, type=t) for t in TYPES}


@case("classic.density.non_square")
def _():
    return C.density(sp.csr_matrix(np.ones((3, 4), dtype=bool)))


# ----------------------------------------------------------------------------- get_pairs_within


def _nodes(n=40, seed=3):
    return node_table(n, seed=seed)


@case("classic.get_pairs_within.default_columns_and_distance")
def _():
    nodes = _nodes(60, seed=11)
    return C.get_pairs_within(digraph(60, 0.1, seed=11), nodes)


@case("classic.get_pairs_within.single_frame_xy")
def _():
    nodes = _nodes()
    return C.get_pairs_within(digraph(40, 0.2, seed=3), nodes, cols=["x", "y"], max_dist=150)


@case("classic.get_pairs_within.single_frame_xyz")
def _():
    nodes = _nodes()
    return C.get_pairs_within(digraph(40, 0.2, seed=3), nodes, cols=["x", "y", "z"], max_dist=250)


@case("classic.get_pairs_within.single_column")
def _():
    nodes = _nodes()
    return C.get_pairs_within(digraph(40, 0.2, seed=3), nodes, cols=["depth"], max_dist=15)


@case("classic.get_pairs_within.pre_and_post_tuple")
def _():
    nodes = _nodes()
    pre, post = nodes.iloc[:25], nodes.iloc[25:]
    adj = digraph(40, 0.2, seed=3)[:25, 25:].tocsr()
    return C.get_pairs_within(adj, (pre, post), max_dist=200)


@case("classic.get_pairs_within.pre_and_post_tuple_more_post_than_pre")
def _():
    nodes = _nodes(50, seed=12)
    pre, post = nodes.iloc[:10], nodes.iloc[10:]
    adj = digraph(50, 0.2, seed=12)[:10, 10:].tocsr()
    return C.get_pairs_within(adj, (pre, post), cols=["x", "y", "z"], max_dist=300)


@case("classic.get_pairs_within.tuple_of_the_same_square_population_keeps_the_diagonal")
def _():
    nodes = _nodes()
    adj = digraph(40, 0.2, seed=3)
    return [
        C.get_pairs_within(adj, (nodes, nodes), max_dist=200),
        C.get_pairs_within(adj, nodes, max_dist=200),
    ]


@case("classic.get_pairs_within.pre_and_post_of_the_same_size_are_not_interchangeable")
def _():
    nodes = _nodes(40, seed=14)
    first, second = nodes.iloc[:20], nodes.iloc[20:]
    adj = digraph(40, 0.2, seed=14)[:20, 20:].tocsr()
    return [
        C.get_pairs_within(adj, (first, second), max_dist=200),
        C.get_pairs_within(adj, (second, first), max_dist=200),
    ]


@case("classic.get_pairs_within.sparse_array_input")
def _():
    nodes = _nodes()
    return C.get_pairs_within(sp.csr_array(digraph(40, 0.2, seed=3)), nodes, max_dist=150)


@case("classic.get_pairs_within.no_pair_within_reach_returns_nan")
def _():
    nodes = _nodes()
    pre, post = nodes.iloc[:25], nodes.iloc[25:]
    adj = digraph(40, 0.2, seed=3)[:25, 25:].tocsr()
    return C.get_pairs_within(adj, (pre, post), max_dist=1e-3)


@case("classic.get_pairs_within.single_frame_with_tiny_distance_has_no_off_diagonal_pair")
def _():
    nodes = _nodes()
    return C.get_pairs_within(digraph(40, 0.2, seed=3), nodes, max_dist=1e-3)


@case("classic.get_pairs_within.coincident_nodes")
def _():
    nodes = pd.DataFrame({"x": [0.0, 0.0, 5.0, 5.0, 9.0], "y": [0.0, 0.0, 1.0, 1.0, 9.0]})
    return C.get_pairs_within(sp.csr_matrix((5, 5), dtype=bool), nodes, max_dist=1e-6)


@case("classic.get_pairs_within.distance_equal_to_max_dist_is_included")
def _():
    nodes = pd.DataFrame({"x": [0.0, 3.0, 0.0, 6.0, 3.0], "y": [0.0, 4.0, 4.0, 8.0, 0.0]})
    return [
        C.get_pairs_within(sp.csr_matrix((5, 5), dtype=bool), nodes, max_dist=d)
        for d in (5.0, 4.0, 3.0)
    ]


@case("classic.get_pairs_within.frame_with_custom_index_and_extra_columns")
def _():
    nodes = _nodes(30, seed=13)
    nodes.index = [f"n{i}" for i in range(30)]
    return C.get_pairs_within(digraph(30, 0.2, seed=13), nodes, cols=["z", "x"], max_dist=180)


@case("classic.get_pairs_within.matrix_content_is_not_used")
def _():
    nodes = _nodes()
    return [
        C.get_pairs_within(m, nodes, max_dist=150)
        for m in (digraph(40, 0.2, seed=3), sp.csr_matrix((40, 40), dtype=bool), np.zeros((40, 40)))
    ]


@case("classic.get_pairs_within.zero_max_dist_leaves_only_the_removed_diagonal")
def _():
    return C.get_pairs_within(digraph(40, 0.2, seed=3), _nodes(), max_dist=0)


@case("classic.get_pairs_within.zero_max_dist_between_distinct_populations")
def _():
    nodes = _nodes()
    pre, post = nodes.iloc[:25], nodes.iloc[25:]
    adj = digraph(40, 0.2, seed=3)[:25, 25:].tocsr()
    return C.get_pairs_within(adj, (pre, post), max_dist=0)


@case("classic.get_pairs_within.unknown_column")
def _():
    return C.get_pairs_within(digraph(40, 0.2, seed=3), _nodes(), cols=["x", "nope"])


@case("classic.get_pairs_within.non_square_matrix_with_a_single_frame")
def _():
    # the matrix has fewer columns than the frame has rows
    nodes = _nodes()
    return C.get_pairs_within(sp.csr_matrix(np.ones((25, 15), dtype=bool)), nodes, max_dist=200)


# ----------------------------------------------------------------------------- connection_probability_within


def _spatial(n=60, seed=21, cols=("x", "y")):
    nodes = node_table(n, seed=seed)
    return spatial_digraph(nodes, seed, cols=cols), nodes


@case("classic.connection_probability_within.default")
def _():
    adj, nodes = _spatial()
    return C.connection_probability_within(adj, nodes)


@case("classic.connection_probability_within.types")
def _():
    adj, nodes = _spatial()
    return {t: C.connection_probability_within(adj, nodes, max_dist=150, type=t) for t in TYPES}


@case("classic.connection_probability_within.types_skip_symmetry_check")
def _():
    adj, nodes = _spatial()
    return {
        t: C.connection_probability_within(
            adj, nodes, max_dist=150, type=t, skip_symmetry_check=True
        )
        for t in TYPES
    }


@case("classic.connection_probability_within.decreases_with_distance")
def _():
    adj, nodes = _spatial(n=80, seed=22)
    return [
        C.connection_probability_within(adj, nodes, max_dist=d) for d in (30, 60, 120, 240, 480)
    ]


@case("classic.connection_probability_within.columns")
def _():
    adj, nodes = _spatial(cols=("x", "y", "z"))
    return {
        "xyz": C.connection_probability_within(adj, nodes, cols=["x", "y", "z"], max_dist=200),
        "xy": C.connection_probability_within(adj, nodes, cols=["x", "y"], max_dist=200),
        "depth": C.connection_probability_within(adj, nodes, cols=["depth"], max_dist=20),
    }


@case("classic.connection_probability_within.pre_and_post_tuple")
def _():
    adj, nodes = _spatial(n=60, seed=23)
    pre, post = nodes.iloc[:40], nodes.iloc[40:]
    return C.connection_probability_within(adj[:40, 40:].tocsr(), (pre, post), max_dist=200)


@case("classic.connection_probability_within.pre_and_post_tuple_with_columns_and_wide_reach")
def _():
    adj, nodes = _spatial(n=60, seed=23, cols=("x", "y", "z"))
    pre, post = nodes.iloc[20:], nodes.iloc[:20]
    return C.connection_probability_within(
        adj[20:, :20].tocsr(), (pre, post), cols=["x", "y", "z"], max_dist=400
    )


@case(
    "classic.connection_probability_within.tuple_of_the_same_population_counts_the_diagonal_pairs"
)
def _():
    # with (nodes, nodes) the pairs (i, i) are not removed, so the mean also covers the empty diagonal of the matrix
    adj, nodes = _spatial()
    return [
        C.connection_probability_within(adj, nodes, max_dist=150),
        C.connection_probability_within(adj, (nodes, nodes), max_dist=150),
    ]


@case("classic.connection_probability_within.pre_and_post_of_the_same_size_are_not_interchangeable")
def _():
    adj, nodes = _spatial(n=60, seed=27)
    first, second = nodes.iloc[:30], nodes.iloc[30:]
    block = adj[:30, 30:].tocsr()
    return [
        C.connection_probability_within(block, (first, second), max_dist=200),
        C.connection_probability_within(block, (second, first), max_dist=200),
    ]


@case("classic.connection_probability_within.sparse_array_input")
def _():
    adj, nodes = _spatial()
    arr = sp.csr_array(adj)
    return {t: C.connection_probability_within(arr, nodes, max_dist=150, type=t) for t in TYPES}


@case("classic.connection_probability_within.weighted_input_is_binarised")
def _():
    adj, nodes = _spatial()
    wadj = adj.astype(float).tocsr()
    wadj.data = np.random.RandomState(24).randint(1, 6, size=wadj.nnz).astype(float)
    return {t: C.connection_probability_within(wadj, nodes, max_dist=150, type=t) for t in TYPES}


@case("classic.connection_probability_within.input_formats")
def _():
    adj, nodes = _spatial()
    inputs = [
        adj.tocsc(),
        adj.tocoo(),
        adj.tolil(),
        adj.toarray(),
        adj.astype(int),
        adj.astype(float),
    ]
    return [C.connection_probability_within(m, nodes, max_dist=150) for m in inputs]


@case("classic.connection_probability_within.reciprocal_input_formats")
def _():
    adj, nodes = _spatial()
    inputs = [adj.tocsc(), adj.tocoo(), adj.toarray()]
    return [
        C.connection_probability_within(m, nodes, max_dist=150, type="reciprocal") for m in inputs
    ]


@case("classic.connection_probability_within.undirected_input_formats")
def _():
    adj, nodes = _spatial()
    inputs = [adj.tocsc(), adj.tocoo(), adj.tolil(), adj.astype(int), adj.astype(float)]
    return [
        C.connection_probability_within(m, nodes, max_dist=150, type="undirected") for m in inputs
    ]


@case("classic.connection_probability_within.undirected_dense_input", tags=("fixed",))
def _():
    # the symmetry check calls .nnz on the comparison of two dense arrays
    adj, nodes = _spatial()
    return C.connection_probability_within(adj.toarray(), nodes, max_dist=150, type="undirected")


@case("classic.connection_probability_within.undirected_dense_input_without_symmetry_check")
def _():
    adj, nodes = _spatial()
    return C.connection_probability_within(
        adj.toarray(), nodes, max_dist=150, type="undirected", skip_symmetry_check=True
    )


@case("classic.connection_probability_within.complete_and_empty_graphs")
def _():
    nodes = node_table(20, seed=25)
    return [
        C.connection_probability_within(complete(20), nodes, max_dist=300),
        C.connection_probability_within(sp.csr_matrix((20, 20), dtype=bool), nodes, max_dist=300),
    ]


@case("classic.connection_probability_within.explicit_zeros")
def _():
    adj, nodes = _spatial()
    adj = with_explicit_zeros(adj, every=4)
    return {t: C.connection_probability_within(adj, nodes, max_dist=150, type=t) for t in TYPES}


@case("classic.connection_probability_within.no_pair_returns_nan")
def _():
    adj, nodes = _spatial(n=40, seed=26)
    pre, post = nodes.iloc[:25], nodes.iloc[25:]
    return C.connection_probability_within(adj[:25, 25:].tocsr(), (pre, post), max_dist=1e-3)


@case("classic.connection_probability_within.type_is_not_checked_when_no_pair_exists")
def _():
    adj, nodes = _spatial(n=40, seed=26)
    pre, post = nodes.iloc[:25], nodes.iloc[25:]
    return C.connection_probability_within(
        adj[:25, 25:].tocsr(), (pre, post), max_dist=1e-3, type="mutual"
    )


@case("classic.connection_probability_within.single_frame_with_tiny_distance", tags=("fixed",))
def _():
    # only self-pairs are within reach, so the selection is empty; the original ends in a ZeroDivisionError
    adj, nodes = _spatial(n=40, seed=26)
    return C.connection_probability_within(adj, nodes, max_dist=1e-3)


@case("classic.connection_probability_within.bad_type")
def _():
    adj, nodes = _spatial()
    return C.connection_probability_within(adj, nodes, max_dist=150, type="mutual")


@case("classic.connection_probability_within.undirected_needs_a_square_matrix")
def _():
    adj, nodes = _spatial(n=60, seed=23)
    pre, post = nodes.iloc[:40], nodes.iloc[40:]
    return C.connection_probability_within(
        adj[:40, 40:].tocsr(), (pre, post), max_dist=200, type="undirected"
    )


@case("classic.connection_probability_within.reciprocal_needs_a_square_matrix")
def _():
    adj, nodes = _spatial(n=60, seed=23)
    pre, post = nodes.iloc[:40], nodes.iloc[40:]
    return C.connection_probability_within(
        adj[:40, 40:].tocsr(), (pre, post), max_dist=200, type="reciprocal"
    )


@case("classic.connection_probability_within.bad_type_with_a_non_square_matrix")
def _():
    adj, nodes = _spatial(n=60, seed=23)
    pre, post = nodes.iloc[:40], nodes.iloc[40:]
    return C.connection_probability_within(
        adj[:40, 40:].tocsr(), (pre, post), max_dist=200, type="mutual"
    )
