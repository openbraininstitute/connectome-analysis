"""Betti numbers of the flag complexes (``flagserpy.flagser_unweighted``): ``betti_counts`` and ``nbc``."""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import node_table, weighted

from cases.topology_helpers import (
    complete,
    cycle,
    explicit_zeros,
    from_edges,
    octahedron,
    projective_plane,
    rich,
    structured,
    transitive,
    with_loops,
)
from connalysis.network import topology as T

FORMATS = ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")
TYPES = ("directed", "undirected", "reciprocal")


def _graph():
    """30 nodes, directed counts (30, 140, 189, 107, 22, 2): Betti numbers (1, 21, 23, 11, 0, 0)."""
    return rich(30, 0.08, seed=13, sizes=(6, 5, 4), rc_sizes=(4, 3))


def _degenerate_graphs():
    return {
        "empty": from_edges(5, []),
        "single_node": from_edges(1, []),
        "no_nodes": from_edges(0, []),
        "one_way_pair": from_edges(2, [(0, 1)]),
        "reciprocal_pair": from_edges(2, [(0, 1), (1, 0)]),
        "directed_3_cycle": cycle(3),
        "directed_4_cycle": cycle(4),
        "complete_5": complete(5),
        "transitive_5": transitive(5),
        "octahedron": octahedron(),
    }


# ----------------------------------------------------------------------------- betti_counts


@case("topology.betti_counts.default")
def _():
    return T.betti_counts(_graph())


@case("topology.betti_counts.structured_graph_all_types")
def _():
    # eight components, a 4-cycle, a reciprocal triangle and an octahedron (a 2-sphere)
    adj = structured()
    return {t: T.betti_counts(adj, simplex_type=t) for t in TYPES}


@case("topology.betti_counts.simplex_types")
def _():
    adj = _graph()
    return {t: T.betti_counts(adj, simplex_type=t) for t in TYPES}


@case("topology.betti_counts.unknown_simplex_type_is_directed")
def _():
    adj = _graph()
    return [T.betti_counts(adj, simplex_type="foo"), T.betti_counts(adj, simplex_type=None)]


@case("topology.betti_counts.max_dim_values")
def _():
    adj = _graph()
    return {
        repr(md): T.betti_counts(adj, max_dim=md)
        for md in (0, 1, 2, 3, 4, 5, 6, 10, [], np.inf, -1)
    }


@case("topology.betti_counts.min_dim_values")
def _():
    adj = _graph()
    return {md: T.betti_counts(adj, min_dim=md) for md in (0, 1, 2, 3, 4, 5, 6, 8)}


@case("topology.betti_counts.min_dim_and_max_dim")
def _():
    adj = _graph()
    return {
        (lo, hi): T.betti_counts(adj, min_dim=lo, max_dim=hi)
        for lo, hi in ((1, 3), (2, 2), (3, 3), (2, 1), (4, 9))
    }


@case("topology.betti_counts.min_dim_with_types")
def _():
    adj = _graph()
    return {
        (t, lo): T.betti_counts(adj, min_dim=lo, simplex_type=t)
        for t in ("undirected", "reciprocal")
        for lo in (1, 2)
    }


@case("topology.betti_counts.negative_min_dim")
def _():
    return T.betti_counts(_graph(), min_dim=-1)


@case("topology.betti_counts.non_integer_max_dim")
def _():
    return T.betti_counts(_graph(), max_dim=2.0)


# ---- approximation


@case("topology.betti_counts.approximation_values")
def _():
    adj = _graph()
    return {
        repr(a): T.betti_counts(adj, approximation=a)
        for a in (
            [-1],
            [-1, -1, -1],
            [1],
            [1, 1, 1],
            [100, 100, 100],
            [100000],
            [2, 2, 2, 2],
            [5] * 8,
        )
    }


@case("topology.betti_counts.approximation_mixed_with_exact_dimensions")
def _():
    adj = _graph()
    return {
        repr(a): T.betti_counts(adj, approximation=a)
        for a in (
            [-1, -1, -1, 1, 1, 1],
            [1, -1, 1],
            [3, -1, 3, -1, 3, -1, 3, -1],
            [-1, 1],
            [1, -1],
            [0],
        )
    }


@case("topology.betti_counts.approximation_list_shorter_than_max_dim")
def _():
    # padded on the right with -1 (exact computation)
    adj = _graph()
    return {
        (tuple(a), md): T.betti_counts(adj, approximation=a, max_dim=md)
        for a, md in (([1, 1], 4), ([1], 3), ([1, 1, 1], 5), ([2, 2], 6), ([-1], 2))
    }


@case("topology.betti_counts.approximation_list_equal_to_max_dim_plus_one")
def _():
    adj = _graph()
    return {
        (tuple(a), md): T.betti_counts(adj, approximation=a, max_dim=md)
        for a, md in (([1, 1, 1], 2), ([1, -1, 1, 1], 3), ([2, 2], 1), ([1], 0))
    }


@case("topology.betti_counts.approximation_list_longer_than_max_dim")
def _():
    # sliced to max_dim + 1 entries
    adj = _graph()
    return {
        (tuple(a), md): T.betti_counts(adj, approximation=a, max_dim=md)
        for a, md in (
            ([2, 2, 2, 2, 2, 2], 2),
            ([1, 1, 1, 1], 1),
            ([-1, -1, 1, 1], 2),
            ([3, 3, 3], 0),
        )
    }


@case("topology.betti_counts.approximation_with_unbounded_max_dim_continues_exactly")
def _():
    # max_dim=[] : the dimensions beyond the list are computed without approximation
    adj = _graph()
    return {
        (tuple(a)): T.betti_counts(adj, approximation=a, max_dim=[])
        for a in ([1, 1, 1], [1, 1], [-1, 1], [2, -1, 2, -1])
    }


@case("topology.betti_counts.approximation_with_other_simplex_types")
def _():
    adj = _graph()
    return {
        (t, tuple(a)): T.betti_counts(adj, approximation=a, simplex_type=t)
        for t in ("undirected", "reciprocal")
        for a in ([1], [1, 1, 1], [2, -1])
    }


@case("topology.betti_counts.approximation_as_tuple")
def _():
    return T.betti_counts(_graph(), approximation=(1, 1, 1))


@case("topology.betti_counts.approximation_on_degenerate_graphs")
def _():
    return {
        name: T.betti_counts(adj, approximation=[1, 1])
        for name, adj in _degenerate_graphs().items()
        if name != "no_nodes"
    }


@case("topology.betti_counts.approximation_requires_min_dim_zero")
def _():
    return T.betti_counts(_graph(), approximation=[1, 1], min_dim=1)


@case("topology.betti_counts.approximation_requires_min_dim_zero_even_if_empty")
def _():
    return T.betti_counts(_graph(), approximation=[], min_dim=2)


@case("topology.betti_counts.approximation_none_allows_min_dim")
def _():
    return T.betti_counts(_graph(), approximation=None, min_dim=2)


@case("topology.betti_counts.approximation_float_items")
def _():
    return T.betti_counts(_graph(), approximation=[1.0, 1.0])


@case("topology.betti_counts.approximation_numpy_integer_items")
def _():
    return T.betti_counts(_graph(), approximation=[np.int64(1), np.int64(1)])


@case("topology.betti_counts.approximation_numpy_array")
def _():
    return T.betti_counts(_graph(), approximation=np.array([1, 1]))


@case("topology.betti_counts.approximation_empty_list")
def _():
    return T.betti_counts(_graph(), approximation=[])


@case("topology.betti_counts.approximation_empty_list_with_max_dim")
def _():
    return T.betti_counts(_graph(), approximation=[], max_dim=3)


@case("topology.betti_counts.approximation_scalar")
def _():
    return T.betti_counts(_graph(), approximation=5)


@case("topology.betti_counts.approximation_strings")
def _():
    return T.betti_counts(_graph(), approximation=["a"])


# ---- inputs


@case("topology.betti_counts.input_formats")
def _():
    adj = _graph()
    out = {fmt: T.betti_counts(adj.asformat(fmt)) for fmt in FORMATS}
    out["dense_bool"] = T.betti_counts(adj.toarray())
    out["dense_int"] = T.betti_counts(adj.toarray().astype(int))
    out["dense_float"] = T.betti_counts(adj.toarray().astype(float))
    out["np.matrix"] = T.betti_counts(np.matrix(adj.toarray()))
    out["csr_array"] = T.betti_counts(sp.csr_array(adj))
    return out


@case("topology.betti_counts.weights_are_ignored")
def _():
    return {
        "integer_weights": T.betti_counts(weighted(20, 0.2, seed=80)),
        "real_weights": T.betti_counts(weighted(20, 0.2, seed=81, integers=False)),
        "explicit_zeros": T.betti_counts(explicit_zeros(_graph(), 20)),
    }


@case("topology.betti_counts.homology_over_the_field_with_two_elements")
def _():
    # the projective plane has 2-torsion: (1, 1, 1) is only obtained with coefficients in F_2 (flagser is called with coeff=2)
    adj = projective_plane()
    return {t: T.betti_counts(adj, simplex_type=t) for t in ("directed", "undirected")}


@case("topology.betti_counts.projective_plane_with_approximation_and_dimension_ranges")
def _():
    adj = projective_plane()
    return {
        "approximation": T.betti_counts(adj, approximation=[1, -1, -1]),
        "min_dim_1": T.betti_counts(adj, min_dim=1),
        "max_dim_1": T.betti_counts(adj, max_dim=1),
    }


@case("topology.betti_counts.degenerate_graphs")
def _():
    return {name: T.betti_counts(adj) for name, adj in _degenerate_graphs().items()}


@case("topology.betti_counts.degenerate_graphs_undirected")
def _():
    return {
        name: T.betti_counts(adj, simplex_type="undirected")
        for name, adj in _degenerate_graphs().items()
    }


@case("topology.betti_counts.degenerate_graphs_reciprocal")
def _():
    return {
        name: T.betti_counts(adj, simplex_type="reciprocal")
        for name, adj in _degenerate_graphs().items()
    }


@case("topology.betti_counts.degenerate_graphs_min_dim_one")
def _():
    return {name: T.betti_counts(adj, min_dim=1) for name, adj in _degenerate_graphs().items()}


@case("topology.betti_counts.diagonal_rejected")
def _():
    return T.betti_counts(with_loops(_graph(), (2,)))


@case("topology.betti_counts.non_square")
def _():
    return T.betti_counts(sp.csr_matrix((3, 4), dtype=bool))


@case("topology.betti_counts.extra_arguments_are_ignored")
def _():
    return T.betti_counts(_graph(), node_properties=node_table(30), unknown_option=1)


# ----------------------------------------------------------------------------- nbc


@case("topology.nbc.rich_graph")
def _():
    return T.nbc(_graph())


@case("topology.nbc.structured_graph")
def _():
    return T.nbc(structured())


@case("topology.nbc.input_types")
def _():
    adj = _graph()
    return {
        "csr": T.nbc(adj),
        "csc": T.nbc(adj.tocsc()),
        "coo": T.nbc(adj.tocoo()),
        "lil": T.nbc(adj.tolil()),
        "dense_bool": T.nbc(adj.toarray()),
        "dense_int": T.nbc(adj.toarray().astype(int)),
        "csr_float": T.nbc(adj.astype(float)),
        "csr_int": T.nbc(adj.astype(int)),
        "np.matrix": T.nbc(np.matrix(adj.toarray())),
    }


@case("topology.nbc.weights_diagonal_and_non_square_are_not_checked")
def _():
    return {
        "weighted": T.nbc(weighted(20, 0.2, seed=82, integers=False)),
        "diagonal": T.nbc(with_loops(_graph(), (2, 3))),
        "non_square_without_edges": T.nbc(sp.csr_matrix((3, 4))),
    }


@case("topology.nbc.projective_plane")
def _():
    return T.nbc(projective_plane())


@case("topology.nbc.small_graphs")
def _():
    return {name: T.nbc(adj) for name, adj in _degenerate_graphs().items() if name != "no_nodes"}


@case("topology.nbc.no_nodes")
def _():
    return T.nbc(from_edges(0, []))


@case("topology.nbc.nested_list_input")
def _():
    return T.nbc(_graph().toarray().tolist())


# ----------------------------------------------------------------------------- euler_characteristic


@case("topology.euler_characteristic.undefined_name_in_the_function_body", tags=("fixed",))
def _():
    # the body refers to ``M`` instead of ``matrix``: NameError for every input
    return T.euler_characteristic(_graph())
