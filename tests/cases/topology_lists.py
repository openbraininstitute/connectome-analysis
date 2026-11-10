"""Simplex lists and the generalised degrees computed from them: ``list_simplices_by_dimension``, ``node_k_degree``,
``in_degree_from_pop``, ``cross_col_k_in_degree`` and ``get_all_simplices_from_max``.

flagser lists simplices in an order that depends on its thread count (``list_simplices_by_dimension`` uses one thread
unless ``threads``/``n_threads`` is given), so the raw lists are compared here as they come.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.data import node_table, rng, weighted

from cases.topology_helpers import (
    complete,
    cycle,
    explicit_zeros,
    from_edges,
    rich,
    structured,
    transitive,
    with_loops,
)
from cases.topology_helpers import (
    contained_case as case,
)  # runs each case with a throw-away temp directory
from connalysis.network import topology as T

FORMATS = ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")
TYPES = ("directed", "undirected", "reciprocal")


def _graph():
    """14 nodes, counts (14, 55, 67, 36, 8): small enough to return every simplex list."""
    return rich(14, 0.12, seed=10, sizes=(5, 4), rc_sizes=(3,))


def _small():
    """25 nodes, counts (25, 84, 65, 15, 2)."""
    return rich(25, 0.08, seed=3, sizes=(5, 4), rc_sizes=(3,))


def _population():
    """30 nodes with some higher simplices, for in-degrees from a source population."""
    return rich(30, 0.1, seed=11, sizes=(5, 4, 3), rc_sizes=(3,))


def _degenerate_graphs():
    return {
        "empty": from_edges(5, []),
        "single_node": from_edges(1, []),
        "no_nodes": from_edges(0, []),
        "one_way_pair": from_edges(2, [(0, 1)]),
        "reciprocal_pair": from_edges(2, [(0, 1), (1, 0)]),
        "directed_3_cycle": cycle(3),
        "complete_4": complete(4),
        "transitive_4": transitive(4),
        "structured": structured(),
    }


# ----------------------------------------------------------------------------- list_simplices_by_dimension


@case("topology.list_simplices_by_dimension.default")
def _():
    return T.list_simplices_by_dimension(_graph())


@case("topology.list_simplices_by_dimension.max_simplices")
def _():
    return T.list_simplices_by_dimension(_graph(), max_simplices=True)


@case("topology.list_simplices_by_dimension.types_and_maximality")
def _():
    adj = _graph()
    return {
        (t, ms): T.list_simplices_by_dimension(adj, simplex_type=t, max_simplices=ms)
        for t in TYPES
        for ms in (False, True)
    }


@case("topology.list_simplices_by_dimension.unknown_simplex_type_is_directed")
def _():
    adj = _graph()
    return [
        T.list_simplices_by_dimension(adj, simplex_type="foo"),
        T.list_simplices_by_dimension(adj, simplex_type="foo", max_simplices=True),
    ]


@case("topology.list_simplices_by_dimension.max_dim_values")
def _():
    adj = _graph()
    return {
        (md, ms): T.list_simplices_by_dimension(adj, max_dim=md, max_simplices=ms)
        for md in (-2, -1, 0, 1, 2, 3, 10)
        for ms in (False, True)
    }


@case("topology.list_simplices_by_dimension.max_dim_with_types")
def _():
    adj = _graph()
    return {
        (t, md): T.list_simplices_by_dimension(adj, simplex_type=t, max_dim=md)
        for t in TYPES
        for md in (1, 2)
    }


@case("topology.list_simplices_by_dimension.nodes_subset")
def _():
    adj = _graph()
    return {
        ms: T.list_simplices_by_dimension(adj, nodes=np.array([2, 5, 0, 9]), max_simplices=ms)
        for ms in (False, True)
    }


@case("topology.list_simplices_by_dimension.nodes_with_undirected_and_reciprocal_types")
def _():
    # unlike simplex_counts, listing accepts a node selection for every simplex type
    adj = _graph()
    return {
        t: T.list_simplices_by_dimension(adj, nodes=np.array([0, 3, 7]), simplex_type=t)
        for t in ("undirected", "reciprocal")
    }


@case("topology.list_simplices_by_dimension.nodes_with_max_dim")
def _():
    adj = _graph()
    return {
        ms: T.list_simplices_by_dimension(
            adj, nodes=np.arange(0, 14, 2), max_dim=2, max_simplices=ms
        )
        for ms in (False, True)
    }


@case("topology.list_simplices_by_dimension.nodes_duplicated")
def _():
    adj = _graph()
    return {
        ms: T.list_simplices_by_dimension(adj, nodes=np.array([2, 2, 5]), max_simplices=ms)
        for ms in (False, True)
    }


@case("topology.list_simplices_by_dimension.nodes_empty")
def _():
    adj = _graph()
    return {
        ms: T.list_simplices_by_dimension(adj, nodes=np.array([], dtype=int), max_simplices=ms)
        for ms in (False, True)
    }


@case("topology.list_simplices_by_dimension.nodes_all_and_single")
def _():
    adj = _graph()
    return {
        "all": T.list_simplices_by_dimension(adj, nodes=np.arange(14)),
        "single": T.list_simplices_by_dimension(adj, nodes=np.array([6])),
        "single_max": T.list_simplices_by_dimension(adj, nodes=np.array([6]), max_simplices=True),
    }


@case("topology.list_simplices_by_dimension.nodes_dtype_is_kept_in_dimension_zero")
def _():
    adj = _graph()
    return [
        T.list_simplices_by_dimension(adj, nodes=np.array([1, 4], dtype=np.int32))[0],
        T.list_simplices_by_dimension(adj, nodes=np.array([1, 4], dtype=np.int64))[0],
        T.list_simplices_by_dimension(adj)[0],
    ]


@case("topology.list_simplices_by_dimension.nodes_as_list_with_max_simplices")
def _():
    return T.list_simplices_by_dimension(_graph(), nodes=[2, 5, 0], max_simplices=True)


@case("topology.list_simplices_by_dimension.nodes_as_list_without_max_simplices")
def _():
    # the node list is later reshaped through ``nodes.size``, which a python list does not have
    return T.list_simplices_by_dimension(_graph(), nodes=[2, 5, 0])


@case("topology.list_simplices_by_dimension.nodes_as_pandas_objects")
def _():
    adj = _graph()
    return {
        "index": T.list_simplices_by_dimension(adj, nodes=pd.Index([2, 5, 0])),
        "series": T.list_simplices_by_dimension(adj, nodes=pd.Series([2, 5, 0])),
        "index_max_simplices": T.list_simplices_by_dimension(
            adj, nodes=pd.Index([2, 5, 0]), max_simplices=True
        ),
    }


@case("topology.list_simplices_by_dimension.nodes_outside_the_matrix")
def _():
    return T.list_simplices_by_dimension(_graph(), nodes=np.array([2, 20]))


@case("topology.list_simplices_by_dimension.nodes_negative")
def _():
    return T.list_simplices_by_dimension(_graph(), nodes=np.array([-1, 2]), max_simplices=True)


@case("topology.list_simplices_by_dimension.thread_arguments")
def _():
    adj = _small()
    return {
        "default": T.list_simplices_by_dimension(adj),
        "threads_2": T.list_simplices_by_dimension(adj, threads=2),
        "n_threads_2": T.list_simplices_by_dimension(adj, n_threads=2),
        "threads_wins_over_n_threads": T.list_simplices_by_dimension(adj, threads=2, n_threads=4),
        "n_threads_4": T.list_simplices_by_dimension(adj, n_threads=4),
        "threads_3_max_simplices": T.list_simplices_by_dimension(
            adj, threads=3, max_simplices=True
        ),
    }


@case("topology.list_simplices_by_dimension.threads_with_nodes")
def _():
    adj = _small()
    nodes = np.array([1, 4, 7, 10, 22])
    return {th: T.list_simplices_by_dimension(adj, nodes=nodes, threads=th) for th in (1, 2, 4)}


@case("topology.list_simplices_by_dimension.input_formats")
def _():
    adj = _graph()
    out = {fmt: T.list_simplices_by_dimension(adj.asformat(fmt)) for fmt in FORMATS}
    out["dense_bool"] = T.list_simplices_by_dimension(adj.toarray())
    out["dense_float"] = T.list_simplices_by_dimension(adj.toarray().astype(float))
    out["np.matrix"] = T.list_simplices_by_dimension(np.matrix(adj.toarray()))
    out["csr_array"] = T.list_simplices_by_dimension(sp.csr_array(adj))
    out["coo_array_reciprocal"] = T.list_simplices_by_dimension(
        sp.coo_array(adj), simplex_type="reciprocal"
    )
    return out


@case("topology.list_simplices_by_dimension.input_formats_other_types")
def _():
    adj = _graph()
    return {
        (fmt, t): T.list_simplices_by_dimension(adj.asformat(fmt), simplex_type=t)
        for fmt in ("csc", "lil")
        for t in ("undirected", "reciprocal")
    }


@case("topology.list_simplices_by_dimension.weights_and_explicit_zeros")
def _():
    return {
        "integer_weights": T.list_simplices_by_dimension(weighted(15, 0.25, seed=70)),
        "explicit_zeros": T.list_simplices_by_dimension(explicit_zeros(_graph(), 8)),
    }


@case("topology.list_simplices_by_dimension.degenerate_graphs")
def _():
    return {name: T.list_simplices_by_dimension(adj) for name, adj in _degenerate_graphs().items()}


@case("topology.list_simplices_by_dimension.degenerate_graphs_max_simplices")
def _():
    return {
        name: T.list_simplices_by_dimension(adj, max_simplices=True)
        for name, adj in _degenerate_graphs().items()
    }


@case("topology.list_simplices_by_dimension.degenerate_graphs_reciprocal")
def _():
    return {
        name: T.list_simplices_by_dimension(adj, simplex_type="reciprocal")
        for name, adj in _degenerate_graphs().items()
    }


@case("topology.list_simplices_by_dimension.diagonal_rejected")
def _():
    return T.list_simplices_by_dimension(with_loops(_graph(), (3,)))


@case("topology.list_simplices_by_dimension.non_square")
def _():
    return T.list_simplices_by_dimension(
        sp.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=bool))
    )


@case("topology.list_simplices_by_dimension.extra_arguments_are_ignored")
def _():
    return T.list_simplices_by_dimension(
        _graph(), node_properties=node_table(14), verbose=True, unknown_option=3
    )


@case("topology.list_simplices_by_dimension.undirected_and_reciprocal_are_listed_upper_triangular")
def _():
    # the symmetrised matrix is reduced to its upper triangle first, so every listed simplex is increasing
    adj = _graph()
    lists = {
        t: T.list_simplices_by_dimension(adj, simplex_type=t) for t in ("undirected", "reciprocal")
    }
    return {
        t: [bool((np.diff(a, axis=1) > 0).all()) for a in series] for t, series in lists.items()
    }


# ----------------------------------------------------------------------------- node_k_degree


@case("topology.node_k_degree.default")
def _():
    return T.node_k_degree(_small())


@case("topology.node_k_degree.in_only")
def _():
    return T.node_k_degree(_small(), direction="IN")


@case("topology.node_k_degree.out_only")
def _():
    return T.node_k_degree(_small(), direction="OUT")


@case("topology.node_k_degree.in_and_out_list")
def _():
    return T.node_k_degree(_small(), direction=["IN", "OUT"])


@case("topology.node_k_degree.max_dim_values")
def _():
    adj = _small()
    return {md: T.node_k_degree(adj, max_dim=md) for md in (-1, 2, 3, 4, 5, 10)}


@case("topology.node_k_degree.max_dim_with_each_direction")
def _():
    adj = _small()
    return {
        (d, md): T.node_k_degree(adj, direction=d, max_dim=md)
        for d in ("IN", "OUT")
        for md in (2, 3)
    }


@case("topology.node_k_degree.out_in_order_is_rejected")
def _():
    return T.node_k_degree(_small(), direction=("OUT", "IN"))


@case("topology.node_k_degree.invalid_direction")
def _():
    return T.node_k_degree(_small(), direction="BOTH")


@case("topology.node_k_degree.single_element_direction")
def _():
    return T.node_k_degree(_small(), direction=("IN",))


@case("topology.node_k_degree.direction_none")
def _():
    return T.node_k_degree(_small(), direction=None)


@case("topology.node_k_degree.max_dim_zero")
def _():
    return T.node_k_degree(_small(), max_dim=0)


@case("topology.node_k_degree.max_dim_one")
def _():
    return T.node_k_degree(_small(), max_dim=1)


@case("topology.node_k_degree.max_dim_negative")
def _():
    return T.node_k_degree(_small(), max_dim=-2)


@case("topology.node_k_degree.no_simplices_of_dimension_two_returns_none")
def _():
    # every one of these prints "There are no simplices of dimension 2 or higher" and returns None
    return {
        "empty": T.node_k_degree(from_edges(5, [])),
        "single_node": T.node_k_degree(from_edges(1, [])),
        "path": T.node_k_degree(from_edges(3, [(0, 1), (1, 2)])),
        "directed_3_cycle": T.node_k_degree(cycle(3)),
        "reciprocal_pair": T.node_k_degree(from_edges(2, [(0, 1), (1, 0)])),
        "star": T.node_k_degree(from_edges(6, [(0, k) for k in range(1, 6)])),
    }


@case("topology.node_k_degree.max_dim_given_but_no_triangles_returns_none")
def _():
    # max_dim is clamped to the largest dimension of the complex (1 here) and that is too low: None again
    return {
        "path": T.node_k_degree(from_edges(4, [(0, 1), (1, 2), (2, 3)]), max_dim=3),
        "cycle": T.node_k_degree(cycle(3), max_dim=2),
    }


@case("topology.node_k_degree.max_dim_beyond_the_complex_with_few_simplices")
def _():
    # max_dim larger than what exists is clamped (with a warning); a triangle only reaches dimension 2
    return T.node_k_degree(from_edges(3, [(0, 1), (0, 2), (1, 2)]), max_dim=6)


@case("topology.node_k_degree.complete_graph_has_integer_degrees")
def _():
    # when every node takes part in every dimension the columns stay integer; otherwise missing nodes become NaN -> float
    return {
        "complete_5": T.node_k_degree(complete(5)),
        "transitive_5": T.node_k_degree(transitive(5)),
        "complete_4_out": T.node_k_degree(complete(4), direction="OUT"),
    }


@case("topology.node_k_degree.structured_graph")
def _():
    return T.node_k_degree(structured())


@case("topology.node_k_degree.input_formats")
def _():
    adj = _small()
    out = {fmt: T.node_k_degree(adj.asformat(fmt)) for fmt in ("csc", "coo", "lil", "dok")}
    out["dense_bool"] = T.node_k_degree(adj.toarray())
    out["dense_float"] = T.node_k_degree(adj.toarray().astype(float))
    out["np.matrix"] = T.node_k_degree(np.matrix(adj.toarray()))
    out["csr_array"] = T.node_k_degree(sp.csr_array(adj))
    return out


@case("topology.node_k_degree.weights_and_explicit_zeros")
def _():
    return {
        "integer_weights": T.node_k_degree(weighted(25, 0.2, seed=71)),
        "explicit_zeros": T.node_k_degree(explicit_zeros(_small(), 10)),
    }


@case("topology.node_k_degree.diagonal_loops_are_counted_as_edges")
def _():
    # a warning says loops are ignored, but flagser counts them (there is no assertion in this function)
    adj = _small()
    return {
        "without_loops": T.node_k_degree(adj),
        "with_loops": T.node_k_degree(with_loops(adj, (1, 5, 8))),
    }


@case("topology.node_k_degree.extra_arguments_are_ignored")
def _():
    return T.node_k_degree(_small(), node_properties=node_table(25), threads=3, unknown_option=1)


# ----------------------------------------------------------------------------- in_degree_from_pop


@case("topology.in_degree_from_pop.default")
def _():
    return T.in_degree_from_pop(_population(), np.arange(0, 30, 3))


@case("topology.in_degree_from_pop.max_simplices")
def _():
    return T.in_degree_from_pop(_population(), np.arange(0, 30, 3), max_simplices=True)


@case("topology.in_degree_from_pop.max_dim_values")
def _():
    adj, src = _population(), np.arange(0, 30, 3)
    return {md: T.in_degree_from_pop(adj, src, max_dim=md) for md in (-2, -1, 1, 2, 3, 100)}


@case("topology.in_degree_from_pop.unsorted_and_list_source")
def _():
    adj = _population()
    return [
        T.in_degree_from_pop(adj, np.array([9, 3, 21, 0, 12, 6, 27, 24])),
        T.in_degree_from_pop(adj, [9, 3, 21, 0, 12, 6, 27, 24]),
        T.in_degree_from_pop(adj, np.array([0, 3, 6, 9, 12, 21, 24, 27])),
    ]


@case("topology.in_degree_from_pop.source_as_pandas_index")
def _():
    return T.in_degree_from_pop(_population(), pd.Index([9, 3, 21, 0, 12, 6, 27, 24]))


@case("topology.in_degree_from_pop.dense_source_block")
def _():
    # a large source population containing most of the higher simplices
    adj = _population()
    return T.in_degree_from_pop(adj, np.arange(0, 24))


@case("topology.in_degree_from_pop.threads")
def _():
    adj, src = _population(), np.arange(0, 30, 3)
    return {th: T.in_degree_from_pop(adj, src, threads=th) for th in (1, 2, 8)}


@case("topology.in_degree_from_pop.source_single_node")
def _():
    return T.in_degree_from_pop(_population(), np.array([4]))


@case("topology.in_degree_from_pop.source_empty")
def _():
    return T.in_degree_from_pop(_population(), np.array([], dtype=int))


@case("topology.in_degree_from_pop.source_is_everything")
def _():
    return T.in_degree_from_pop(_population(), np.arange(30))


@case("topology.in_degree_from_pop.empty_graph")
def _():
    return T.in_degree_from_pop(from_edges(8, []), np.array([1, 2, 3]))


@case("topology.in_degree_from_pop.formats")
def _():
    adj, src = _population(), np.arange(0, 30, 3)
    return {fmt: T.in_degree_from_pop(adj.asformat(fmt), src) for fmt in ("csc", "coo", "lil")}


@case("topology.in_degree_from_pop.weighted_graph")
def _():
    return T.in_degree_from_pop(weighted(30, 0.25, seed=72), np.arange(0, 30, 3))


@case("topology.in_degree_from_pop.dense_graph_is_not_accepted")
def _():
    # the docstring says arrays are accepted, but adj.tocsr() is called on the input
    return T.in_degree_from_pop(_population().toarray(), np.arange(0, 30, 3))


@case("topology.in_degree_from_pop.diagonal_in_source_rejected")
def _():
    return T.in_degree_from_pop(with_loops(_population(), (3,)), np.arange(0, 30, 3))


@case("topology.in_degree_from_pop.diagonal_outside_source_is_ignored")
def _():
    return T.in_degree_from_pop(with_loops(_population(), (4, 5)), np.arange(0, 30, 3))


@case("topology.in_degree_from_pop.source_outside_the_matrix")
def _():
    return T.in_degree_from_pop(_population(), np.array([3, 40]))


@case("topology.in_degree_from_pop.verbose_keyword_collides")
def _():
    # **kwargs is forwarded together with verbose=False to list_simplices_by_dimension
    return T.in_degree_from_pop(_population(), np.arange(0, 30, 3), verbose=False)


@case("topology.in_degree_from_pop.other_extra_arguments_are_forwarded_and_ignored")
def _():
    return T.in_degree_from_pop(_population(), np.arange(0, 30, 3), unknown_option=1)


# ----------------------------------------------------------------------------- cross_col_k_in_degree


def _cross_inputs():
    """A 9-node source population with simplices and a 9 x 6 matrix of edges onto six target nodes."""
    source = rich(9, 0.2, seed=5, sizes=(5, 4), rc_sizes=(3,))
    cross = sp.csr_matrix(rng(3).random_sample((9, 6)) < 0.4)
    return cross, source


@case("topology.cross_col_k_in_degree.default")
def _():
    cross, source = _cross_inputs()
    return T.cross_col_k_in_degree(cross, source)


@case("topology.cross_col_k_in_degree.max_simplices")
def _():
    cross, source = _cross_inputs()
    return T.cross_col_k_in_degree(cross, source, max_simplices=True)


@case("topology.cross_col_k_in_degree.max_dim_values")
def _():
    cross, source = _cross_inputs()
    return {md: T.cross_col_k_in_degree(cross, source, max_dim=md) for md in (-1, 1, 2, 3, 10)}


@case("topology.cross_col_k_in_degree.dense_and_weighted_inputs")
def _():
    cross, source = _cross_inputs()
    return {
        "dense": T.cross_col_k_in_degree(cross.toarray(), source.toarray()),
        "weighted_cross": T.cross_col_k_in_degree(cross.astype(float) * 2.5, source),
        "csc": T.cross_col_k_in_degree(cross.tocsc(), source.tocsc()),
        "int_cross": T.cross_col_k_in_degree(cross.astype(int), source.astype(int)),
    }


@case("topology.cross_col_k_in_degree.threads")
def _():
    cross, source = _cross_inputs()
    return {th: T.cross_col_k_in_degree(cross, source, threads=th) for th in (1, 2, 8)}


@case("topology.cross_col_k_in_degree.empty_cross")
def _():
    _, source = _cross_inputs()
    return T.cross_col_k_in_degree(from_edges(9, []).tocsr()[:, :6], source)


@case("topology.cross_col_k_in_degree.empty_source")
def _():
    cross, _ = _cross_inputs()
    return T.cross_col_k_in_degree(cross, from_edges(9, []))


@case("topology.cross_col_k_in_degree.complete_source_and_full_cross")
def _():
    # every target receives from every source node: the in-degree is the number of simplices of the source graph
    return T.cross_col_k_in_degree(sp.csr_matrix(np.ones((5, 3), dtype=bool)), complete(5))


@case("topology.cross_col_k_in_degree.no_targets")
def _():
    _, source = _cross_inputs()
    return T.cross_col_k_in_degree(sp.csr_matrix((9, 0), dtype=bool), source)


@case("topology.cross_col_k_in_degree.no_sources")
def _():
    return T.cross_col_k_in_degree(
        sp.csr_matrix((0, 6), dtype=bool), sp.csr_matrix((0, 0), dtype=bool)
    )


@case("topology.cross_col_k_in_degree.source_not_square")
def _():
    cross, _ = _cross_inputs()
    return T.cross_col_k_in_degree(cross, sp.csr_matrix((9, 10), dtype=bool))


@case("topology.cross_col_k_in_degree.row_mismatch")
def _():
    cross, source = _cross_inputs()
    return T.cross_col_k_in_degree(cross[:5], source)


@case("topology.cross_col_k_in_degree.diagonal_in_source_rejected")
def _():
    cross, source = _cross_inputs()
    return T.cross_col_k_in_degree(cross, with_loops(source, (1,)))


@case("topology.cross_col_k_in_degree.verbose_keyword_collides")
def _():
    cross, source = _cross_inputs()
    return T.cross_col_k_in_degree(cross, source, verbose=True)


@case("topology.cross_col_k_in_degree.other_extra_arguments_are_forwarded_and_ignored")
def _():
    cross, source = _cross_inputs()
    return T.cross_col_k_in_degree(cross, source, unknown_option=1)


# ----------------------------------------------------------------------------- get_all_simplices_from_max


@case("topology.get_all_simplices_from_max.tuples_result_and_mutated_input")
def _():
    given = [[], [(5, 6)], [(0, 1, 2), (1, 2, 3)], [(0, 1, 2, 3)]]
    result = T.get_all_simplices_from_max(given)
    # the inner lists of the input are extended in place and shared with the result
    return result, given, [a is b for a, b in zip(result, given)], result is given


@case("topology.get_all_simplices_from_max.vertices_edges_triangle")
def _():
    given = [[(0,), (1,)], [(0, 1)], [(2, 3, 4)]]
    return T.get_all_simplices_from_max(given), given


@case("topology.get_all_simplices_from_max.from_flagser_maximal_simplices")
def _():
    maximal = T.list_simplices_by_dimension(_graph(), max_simplices=True)
    given = [
        [tuple(int(v) for v in simplex) for simplex in dim_list] if dim_list.size else []
        for dim_list in maximal
    ]
    result = T.get_all_simplices_from_max(given)
    return result, given


@case(
    "topology.get_all_simplices_from_max.faces_agree_with_all_simplices_of_the_undirected_complex"
)
def _():
    # compare as sets of vertex sets with the undirected simplices flagser lists directly
    adj = _graph()
    maximal = T.list_simplices_by_dimension(adj, max_simplices=True, simplex_type="undirected")
    given = [
        [tuple(int(v) for v in simplex) for simplex in dim_list] if dim_list.size else []
        for dim_list in maximal
    ]
    result = T.get_all_simplices_from_max(given)
    every = T.list_simplices_by_dimension(adj, simplex_type="undirected")
    return {
        dim: (
            sorted(tuple(sorted(s)) for s in result[dim]),
            sorted(tuple(sorted(map(int, s))) for s in every[dim]),
        )
        for dim in range(1, len(result))
    }


@case("topology.get_all_simplices_from_max.lists_as_simplices")
def _():
    # a face of a list simplex is built with + on lists and then turned into a tuple; the originals stay lists
    given = [[], [[5, 6]], [[0, 1, 2]]]
    return T.get_all_simplices_from_max(given), given


@case("topology.get_all_simplices_from_max.numpy_rows_fail")
def _():
    return T.get_all_simplices_from_max([[], [np.array([5, 6])], [np.array([0, 1, 2])]])


@case("topology.get_all_simplices_from_max.unsorted_vertices_are_kept_in_order")
def _():
    given = [[], [(9, 3)], [(7, 2, 5)]]
    return T.get_all_simplices_from_max(given), given


@case("topology.get_all_simplices_from_max.empty_middle_dimensions")
def _():
    given = [[], [], [(0, 1, 2)]]
    return T.get_all_simplices_from_max(given), given


@case("topology.get_all_simplices_from_max.degenerate_inputs")
def _():
    only_vertices = [[(0,), (1,)]]
    return {
        "empty": T.get_all_simplices_from_max([]),
        "only_vertices": T.get_all_simplices_from_max(only_vertices),
        "only_vertices_after": only_vertices,
    }


@case("topology.get_all_simplices_from_max.tuple_of_lists_and_series_input")
def _():
    import pandas as pd

    return [
        T.get_all_simplices_from_max(([], [(5, 6)], [(0, 1, 2)])),
        T.get_all_simplices_from_max(pd.Series([[], [(5, 6)], [(0, 1, 2)]])),
    ]


@case("topology.get_all_simplices_from_max.iterator_input")
def _():
    return T.get_all_simplices_from_max(iter([[], [(5, 6)], [(0, 1, 2)]]))
