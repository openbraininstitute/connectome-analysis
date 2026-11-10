"""k-skeleta and edge positions inside simplices: ``_generate_abstract_edges_in_simplices``,
``extract_submatrix_from_simplex_list``, ``extract_submatrices_from_edge_participation``, ``get_k_skeleta_graph``,
``count_rc_edges_skeleta`` and ``bedge_counts``.

Precomputed simplex lists / edge participation are built by brute force (``topology_helpers``) so that these cases do not
depend on ``list_simplices_by_dimension`` and ``edge_participation``; a few cases feed in the library's own output as well.
``count_rc_edges_skeleta`` fills its table by chained assignment (``df[col].loc[idx] = value``), which copy-on-write pandas
(3.x) no longer lets reach the table.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cases.topology_helpers import (
    complete,
    edge_participation_frame,
    from_edges,
    rich,
    simplex_series,
    structured,
    transitive,
    with_loops,
)
from cases.topology_helpers import (
    contained_case as case,
)  # runs each case with a throw-away temp directory
from connalysis.network import topology as T

POSITIONS = ("all", "spine")
TYPES = ("directed", "undirected", "reciprocal")


def _graph():
    """14 nodes, counts (14, 55, 67, 36, 8), with reciprocal pairs."""
    return rich(14, 0.12, seed=10, sizes=(5, 4), rc_sizes=(3,))


def _mixed():
    """A transitive 4-clique, an isolated edge and a reciprocal pair: the maximal 2-simplices do not exist."""
    return from_edges(
        8, [(i, j) for i in range(4) for j in range(i + 1, 4)] + [(4, 5), (6, 7), (7, 6)]
    )


# ----------------------------------------------------------------------------- _generate_abstract_edges_in_simplices


@case("topology._generate_abstract_edges_in_simplices.all_dimensions_and_positions")
def _():
    return {
        (dim, pos): T._generate_abstract_edges_in_simplices(dim, position=pos)
        for dim in range(-1, 7)
        for pos in POSITIONS
    }


@case("topology._generate_abstract_edges_in_simplices.default_position_is_all")
def _():
    return [
        T._generate_abstract_edges_in_simplices(3),
        T._generate_abstract_edges_in_simplices(2, "spine"),
    ]


@case("topology._generate_abstract_edges_in_simplices.unknown_position")
def _():
    return T._generate_abstract_edges_in_simplices(2, position="middle")


# ----------------------------------------------------------------------------- extract_submatrix_from_simplex_list


@case("topology.extract_submatrix_from_simplex_list.spine_and_all_for_each_dimension")
def _():
    lists = simplex_series(_graph())
    return {
        (dim, pos): T.extract_submatrix_from_simplex_list(lists[dim], 14, position=pos)
        for dim in range(1, len(lists))
        for pos in POSITIONS
    }


@case("topology.extract_submatrix_from_simplex_list.default_position_is_spine")
def _():
    lists = simplex_series(_graph())
    return [
        T.extract_submatrix_from_simplex_list(lists[3], 14),
        T.extract_submatrix_from_simplex_list(lists[3], 14, "all"),
    ]


@case("topology.extract_submatrix_from_simplex_list.maximal_simplex_lists")
def _():
    lists = simplex_series(_graph(), max_simplices=True)
    return {
        (dim, pos): T.extract_submatrix_from_simplex_list(lists[dim], 14, position=pos)
        for dim in range(1, len(lists))
        for pos in POSITIONS
    }


@case("topology.extract_submatrix_from_simplex_list.larger_matrix_than_needed")
def _():
    lists = simplex_series(_graph())
    return T.extract_submatrix_from_simplex_list(lists[2], 30)


@case("topology.extract_submatrix_from_simplex_list.empty_lists_give_empty_csr_matrix")
def _():
    # note the class: csr_matrix here, coo_matrix when there are simplices
    return {
        (shape, pos): T.extract_submatrix_from_simplex_list(
            np.empty(shape, dtype=np.int64), 9, position=pos
        )
        for shape in ((0, 3), (0, 1), (0,))
        for pos in POSITIONS
    }


@case("topology.extract_submatrix_from_simplex_list.empty_list_and_unusual_sizes")
def _():
    return [
        T.extract_submatrix_from_simplex_list(np.empty((0, 3)), 0),
        T.extract_submatrix_from_simplex_list(np.array([]), 1),
    ]


@case("topology.extract_submatrix_from_simplex_list.empty_list_negative_size")
def _():
    return T.extract_submatrix_from_simplex_list(np.empty((0, 3)), -1)


@case("topology.extract_submatrix_from_simplex_list.vertices_have_no_edges")
def _():
    return T.extract_submatrix_from_simplex_list(simplex_series(_graph())[0], 14)


@case("topology.extract_submatrix_from_simplex_list.vertices_have_no_edges_all_position")
def _():
    return T.extract_submatrix_from_simplex_list(simplex_series(_graph())[0], 14, position="all")


@case("topology.extract_submatrix_from_simplex_list.single_simplex_and_duplicates")
def _():
    return {
        "single_spine": T.extract_submatrix_from_simplex_list(np.array([[3, 1, 2]]), 5),
        "single_all": T.extract_submatrix_from_simplex_list(
            np.array([[3, 1, 2]]), 5, position="all"
        ),
        "duplicates": T.extract_submatrix_from_simplex_list(
            np.array([[3, 1, 2], [3, 1, 2], [3, 1, 4]]), 5
        ),
        "one_edge": T.extract_submatrix_from_simplex_list(np.array([[2, 4]]), 5, position="all"),
    }


@case("topology.extract_submatrix_from_simplex_list.repeated_vertices_give_loops")
def _():
    return T.extract_submatrix_from_simplex_list(np.array([[1, 1, 2]]), 5)


@case("topology.extract_submatrix_from_simplex_list.float_valued_simplices")
def _():
    return T.extract_submatrix_from_simplex_list(simplex_series(_graph())[2].astype(float), 14)


@case("topology.extract_submatrix_from_simplex_list.size_exactly_largest_vertex_plus_one")
def _():
    lists = simplex_series(_graph())
    return T.extract_submatrix_from_simplex_list(lists[2], int(lists[2].max()) + 1)


@case("topology.extract_submatrix_from_simplex_list.size_too_small")
def _():
    return T.extract_submatrix_from_simplex_list(simplex_series(_graph())[2], 5)


@case("topology.extract_submatrix_from_simplex_list.negative_vertex")
def _():
    return T.extract_submatrix_from_simplex_list(np.array([[-1, 1, 2]]), 5)


@case("topology.extract_submatrix_from_simplex_list.python_list_input")
def _():
    return T.extract_submatrix_from_simplex_list([[3, 1, 2]], 5)


@case("topology.extract_submatrix_from_simplex_list.unknown_position")
def _():
    return T.extract_submatrix_from_simplex_list(simplex_series(_graph())[2], 14, position="middle")


@case("topology.extract_submatrix_from_simplex_list.input_is_not_modified")
def _():
    given = simplex_series(_graph())[3]
    before = given.copy()
    return T.extract_submatrix_from_simplex_list(given, 14, position="all"), given, before


# ----------------------------------------------------------------------------- extract_submatrices_from_edge_participation


def _edge_par():
    return edge_participation_frame(_graph())


@case("topology.extract_submatrices_from_edge_participation.default")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par(), 14)


@case("topology.extract_submatrices_from_edge_participation.dims_subsets")
def _():
    edge_par = _edge_par()
    subsets = {
        "list": [1, 3],
        "single": [2],
        "tuple": (2, 4),
        "array": np.array([4, 2]),
        "index": pd.Index([2, 4], name="dim"),
        "empty": [],
        "repeated": [3, 3],
    }
    return {
        name: T.extract_submatrices_from_edge_participation(edge_par, 14, dims=dims)
        for name, dims in subsets.items()
    }


@case("topology.extract_submatrices_from_edge_participation.thresholds")
def _():
    edge_par = _edge_par()
    return {
        thresh: T.extract_submatrices_from_edge_participation(edge_par, 14, thresh=thresh)
        for thresh in (0, 1, 2, 3, 5, 100, 1.5)
    }


@case("topology.extract_submatrices_from_edge_participation.thresholds_with_dims")
def _():
    edge_par = _edge_par()
    return {
        (thresh, dim): T.extract_submatrices_from_edge_participation(
            edge_par, 14, dims=[dim], thresh=thresh
        )
        for thresh in (2, 4)
        for dim in (2, 3)
    }


@case("topology.extract_submatrices_from_edge_participation.maximal_simplices_participation")
def _():
    return T.extract_submatrices_from_edge_participation(
        edge_participation_frame(_graph(), max_simplices=True), 14
    )


@case("topology.extract_submatrices_from_edge_participation.size_larger_than_needed")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par(), 30, dims=[1, 2])


@case("topology.extract_submatrices_from_edge_participation.size_too_small")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par(), 5)


@case("topology.extract_submatrices_from_edge_participation.dims_not_columns")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par(), 14, dims=[2, 9])


@case("topology.extract_submatrices_from_edge_participation.scalar_dims")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par(), 14, dims=2)


@case("topology.extract_submatrices_from_edge_participation.size_none")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par(), None)


@case("topology.extract_submatrices_from_edge_participation.hand_made_tables")
def _():
    edges = [(0, 1), (2, 1), (1, 0), (3, 2)]
    integer_columns = pd.DataFrame(
        {1: [1, 1, 1, 1], 2: [0, 2, 3, 1]}, index=pd.Index(edges, tupleize_cols=False)
    )
    string_columns = pd.DataFrame(
        {"a": [1, 1, 1, 1], "b": [0, 2, 3, 1]}, index=pd.Index(edges, tupleize_cols=False)
    )
    return {
        "integer_columns": T.extract_submatrices_from_edge_participation(integer_columns, 4),
        "integer_columns_thresh_2": T.extract_submatrices_from_edge_participation(
            integer_columns, 4, thresh=2
        ),
        "integer_columns_dims": T.extract_submatrices_from_edge_participation(
            integer_columns, 4, dims=[2]
        ),
        "string_columns": T.extract_submatrices_from_edge_participation(string_columns, 4),
        "string_columns_dims": T.extract_submatrices_from_edge_participation(
            string_columns, 4, dims=["b"]
        ),
    }


@case("topology.extract_submatrices_from_edge_participation.multiindex_input")
def _():
    edge_par = _edge_par()
    return T.extract_submatrices_from_edge_participation(
        edge_par.set_index(pd.MultiIndex.from_tuples(edge_par.index)), 14
    )


@case("topology.extract_submatrices_from_edge_participation.unordered_edges")
def _():
    edge_par = _edge_par()
    return T.extract_submatrices_from_edge_participation(edge_par.iloc[::-1], 14, dims=[2, 3])


@case("topology.extract_submatrices_from_edge_participation.empty_table")
def _():
    return T.extract_submatrices_from_edge_participation(pd.DataFrame(), 5)


@case("topology.extract_submatrices_from_edge_participation.table_without_rows")
def _():
    return T.extract_submatrices_from_edge_participation(_edge_par().iloc[:0], 14)


@case("topology.extract_submatrices_from_edge_participation.library_edge_participation_as_input")
def _():
    return T.extract_submatrices_from_edge_participation(
        T.edge_participation(_graph()), 14, dims=[1, 2, 4]
    )


@case("topology.extract_submatrices_from_edge_participation.input_is_not_modified")
def _():
    edge_par = _edge_par()
    before = edge_par.copy()
    return T.extract_submatrices_from_edge_participation(edge_par, 14), edge_par.equals(before)


# ----------------------------------------------------------------------------- get_k_skeleta_graph


@case("topology.get_k_skeleta_graph.computed_from_adjacency")
def _():
    adj = _graph()
    return {pos: T.get_k_skeleta_graph(adj, position=pos) for pos in POSITIONS}


@case("topology.get_k_skeleta_graph.maximal_simplices")
def _():
    adj = _graph()
    return {pos: T.get_k_skeleta_graph(adj, position=pos, max_simplices=True) for pos in POSITIONS}


@case("topology.get_k_skeleta_graph.dimensions_subsets")
def _():
    adj = _graph()
    return {
        (pos, repr(dims)): T.get_k_skeleta_graph(adj, position=pos, dimensions=dims)
        for pos in POSITIONS
        for dims in ([1, 3], 2, [1, 4, 6], (1, 3), [0], [4], [2, 2])
    }


@case("topology.get_k_skeleta_graph.dimensions_empty_list")
def _():
    return T.get_k_skeleta_graph(_graph(), dimensions=[])


@case("topology.get_k_skeleta_graph.dimensions_numpy_array")
def _():
    return T.get_k_skeleta_graph(_graph(), dimensions=np.array([1, 3]))


@case("topology.get_k_skeleta_graph.dimensions_numpy_array_spine")
def _():
    return T.get_k_skeleta_graph(_graph(), dimensions=np.array([1, 3]), position="spine")


@case("topology.get_k_skeleta_graph.simplex_types")
def _():
    # only position 'all' honours the type: the spine is always computed on directed simplices
    adj = _graph()
    return {
        (pos, t): T.get_k_skeleta_graph(adj, position=pos, simplex_type=t)
        for pos in POSITIONS
        for t in TYPES
    }


@case("topology.get_k_skeleta_graph.simplex_types_maximal")
def _():
    adj = _graph()
    return {t: T.get_k_skeleta_graph(adj, max_simplices=True, simplex_type=t) for t in TYPES}


@case("topology.get_k_skeleta_graph.threads_arguments")
def _():
    adj = _graph()
    return {
        "all_threads_1": T.get_k_skeleta_graph(adj, threads=1),
        "all_threads_3": T.get_k_skeleta_graph(adj, threads=3),
        "spine_threads_ignored": T.get_k_skeleta_graph(adj, position="spine", threads=3),
        "spine_n_threads_forwarded": T.get_k_skeleta_graph(adj, position="spine", n_threads=3),
    }


@case("topology.get_k_skeleta_graph.adjacency_formats")
def _():
    adj = _graph()
    return {
        (pos, name): T.get_k_skeleta_graph(m, position=pos)
        for pos in POSITIONS
        for name, m in (
            ("dense", adj.toarray()),
            ("csc", adj.tocsc()),
            ("coo", adj.tocoo()),
            ("lil", adj.tolil()),
            ("csr_array", sp.csr_array(adj)),
        )
    }


@case("topology.get_k_skeleta_graph.size_argument_is_replaced_by_the_matrix_size")
def _():
    return T.get_k_skeleta_graph(_graph(), N=5)


@case("topology.get_k_skeleta_graph.unknown_position_with_adjacency")
def _():
    return T.get_k_skeleta_graph(_graph(), position="middle")


@case("topology.get_k_skeleta_graph.unknown_position_with_dimensions")
def _():
    return T.get_k_skeleta_graph(_graph(), position="middle", dimensions=[2])


@case("topology.get_k_skeleta_graph.nothing_to_compute_from")
def _():
    return T.get_k_skeleta_graph()


@case("topology.get_k_skeleta_graph.extra_arguments_spine")
def _():
    adj = _graph()
    return T.get_k_skeleta_graph(adj, position="spine", unknown_option=2)


@case("topology.get_k_skeleta_graph.extra_arguments_all")
def _():
    adj = _graph()
    return T.get_k_skeleta_graph(adj, position="all", unknown_option=2, verbose=True)


@case("topology.get_k_skeleta_graph.spine_verbose_keyword_collides")
def _():
    return T.get_k_skeleta_graph(_graph(), position="spine", verbose=True)


@case("topology.get_k_skeleta_graph.spine_nodes_keyword_collides")
def _():
    return T.get_k_skeleta_graph(_graph(), position="spine", nodes=np.arange(3))


@case("topology.get_k_skeleta_graph.diagonal_is_rejected")
def _():
    return T.get_k_skeleta_graph(with_loops(_graph(), (2,)), position="spine")


@case("topology.get_k_skeleta_graph.diagonal_is_rejected_for_position_all")
def _():
    return T.get_k_skeleta_graph(with_loops(_graph(), (2,)), position="all")


# ---- precomputed simplex lists (position 'spine')


@case("topology.get_k_skeleta_graph.precomputed_simplex_list")
def _():
    lists = simplex_series(_graph())
    return {
        "all_dimensions": T.get_k_skeleta_graph(simplex_list=lists, N=14, position="spine"),
        "dimensions": T.get_k_skeleta_graph(
            simplex_list=lists, N=14, position="spine", dimensions=[2, 3]
        ),
        "dimension_scalar": T.get_k_skeleta_graph(
            simplex_list=lists, N=14, position="spine", dimensions=4
        ),
    }


@case("topology.get_k_skeleta_graph.precomputed_maximal_simplex_list")
def _():
    return T.get_k_skeleta_graph(
        simplex_list=simplex_series(_graph(), max_simplices=True), N=14, position="spine"
    )


@case("topology.get_k_skeleta_graph.precomputed_simplex_list_with_empty_dimensions")
def _():
    # dimensions without simplices are reported as None
    mixed_lists = simplex_series(_mixed(), max_simplices=True)
    return {
        "mixed": T.get_k_skeleta_graph(simplex_list=mixed_lists, N=8, position="spine"),
        "complete_4": T.get_k_skeleta_graph(
            simplex_list=simplex_series(complete(4), max_simplices=True), N=4, position="spine"
        ),
    }


@case("topology.get_k_skeleta_graph.precomputed_sliced_simplex_list")
def _():
    lists = simplex_series(_graph())
    return {
        "loc_2_2": T.get_k_skeleta_graph(simplex_list=lists.loc[2:2], N=14, position="spine"),
        "loc_2_3": T.get_k_skeleta_graph(simplex_list=lists.loc[2:3], N=14, position="spine"),
    }


@case("topology.get_k_skeleta_graph.precomputed_simplex_list_ignores_adjacency")
def _():
    lists = simplex_series(_graph())
    return T.get_k_skeleta_graph(
        adj=sp.csr_matrix((3, 3)), simplex_list=lists, N=14, position="spine"
    )


@case("topology.get_k_skeleta_graph.precomputed_simplex_list_library_output")
def _():
    return T.get_k_skeleta_graph(
        simplex_list=T.list_simplices_by_dimension(_graph()), N=14, position="spine"
    )


@case("topology.get_k_skeleta_graph.precomputed_simplex_list_size_too_small")
def _():
    return T.get_k_skeleta_graph(simplex_list=simplex_series(_graph()), N=10, position="spine")


@case("topology.get_k_skeleta_graph.precomputed_simplex_list_size_equal_to_largest_vertex")
def _():
    lists = simplex_series(_graph())
    return T.get_k_skeleta_graph(
        simplex_list=lists, N=int(max(a.max() for a in lists if a.size)), position="spine"
    )


@case("topology.get_k_skeleta_graph.precomputed_simplex_list_size_missing")
def _():
    return T.get_k_skeleta_graph(simplex_list=simplex_series(_graph()), position="spine")


@case("topology.get_k_skeleta_graph.simplex_list_only_is_not_enough_for_position_all")
def _():
    # the tutorial calls it this way: simplex_list + N with the default position 'all'
    return T.get_k_skeleta_graph(
        simplex_list=simplex_series(_graph()).loc[2:2], N=14, position="all"
    )


@case("topology.get_k_skeleta_graph.edge_participation_only_is_not_enough_for_position_spine")
def _():
    return T.get_k_skeleta_graph(
        edge_par=edge_participation_frame(_graph()), N=14, position="spine"
    )


@case("topology.get_k_skeleta_graph.size_only")
def _():
    return T.get_k_skeleta_graph(position="spine", N=14)


# ---- precomputed edge participation (position 'all')


@case("topology.get_k_skeleta_graph.precomputed_edge_participation")
def _():
    edge_par = edge_participation_frame(_graph())
    return {
        "all_dimensions": T.get_k_skeleta_graph(edge_par=edge_par, N=14, position="all"),
        "dimensions": T.get_k_skeleta_graph(
            edge_par=edge_par, N=14, position="all", dimensions=[2, 4]
        ),
        "dimension_scalar": T.get_k_skeleta_graph(edge_par=edge_par, N=14, dimensions=3),
        "larger_matrix": T.get_k_skeleta_graph(edge_par=edge_par, N=20, position="all"),
    }


@case("topology.get_k_skeleta_graph.precomputed_maximal_edge_participation")
def _():
    return T.get_k_skeleta_graph(
        edge_par=edge_participation_frame(_graph(), max_simplices=True), N=14
    )


@case("topology.get_k_skeleta_graph.precomputed_edge_participation_library_output")
def _():
    return T.get_k_skeleta_graph(edge_par=T.edge_participation(_graph()), N=14)


@case(
    "topology.get_k_skeleta_graph.precomputed_edge_participation_ignores_adjacency_and_simplex_list"
)
def _():
    edge_par = edge_participation_frame(_graph())
    return T.get_k_skeleta_graph(
        adj=sp.csr_matrix((3, 3)),
        simplex_list=simplex_series(_graph()),
        edge_par=edge_par,
        N=14,
        position="all",
    )


@case("topology.get_k_skeleta_graph.precomputed_edge_participation_size_too_small")
def _():
    return T.get_k_skeleta_graph(edge_par=edge_participation_frame(_graph()), N=5)


@case("topology.get_k_skeleta_graph.precomputed_edge_participation_size_missing")
def _():
    return T.get_k_skeleta_graph(edge_par=edge_participation_frame(_graph()))


# ---- special graphs


@case("topology.get_k_skeleta_graph.special_graphs_spine")
def _():
    return {
        name: T.get_k_skeleta_graph(adj, position="spine")
        for name, adj in (
            ("mixed", _mixed()),
            ("complete_4", complete(4)),
            ("transitive_5", transitive(5)),
            ("one_edge", from_edges(2, [(0, 1)])),
            ("reciprocal_pair", from_edges(2, [(0, 1), (1, 0)])),
            ("structured", structured()),
        )
    }


@case("topology.get_k_skeleta_graph.special_graphs_all")
def _():
    return {
        name: T.get_k_skeleta_graph(adj, position="all")
        for name, adj in (
            ("mixed", _mixed()),
            ("complete_4", complete(4)),
            ("transitive_5", transitive(5)),
            ("one_edge", from_edges(2, [(0, 1)])),
            ("reciprocal_pair", from_edges(2, [(0, 1), (1, 0)])),
            ("structured", structured()),
        )
    }


@case("topology.get_k_skeleta_graph.maximal_simplices_of_special_graphs")
def _():
    return {
        (name, pos): T.get_k_skeleta_graph(adj, position=pos, max_simplices=True)
        for name, adj in (
            ("mixed", _mixed()),
            ("complete_4", complete(4)),
            ("transitive_5", transitive(5)),
        )
        for pos in POSITIONS
    }


@case("topology.get_k_skeleta_graph.graph_without_edges_spine")
def _():
    return T.get_k_skeleta_graph(from_edges(5, []), position="spine")


@case("topology.get_k_skeleta_graph.graph_without_edges_all")
def _():
    return T.get_k_skeleta_graph(from_edges(5, []), position="all")


# ----------------------------------------------------------------------------- count_rc_edges_skeleta


@case("topology.count_rc_edges_skeleta.computed_from_adjacency")
def _():
    adj = _graph()
    return {pos: T.count_rc_edges_skeleta(adj, position=pos) for pos in POSITIONS}


@case("topology.count_rc_edges_skeleta.maximal_simplices")
def _():
    adj = _graph()
    return {
        pos: T.count_rc_edges_skeleta(adj, position=pos, max_simplices=True) for pos in POSITIONS
    }


@case("topology.count_rc_edges_skeleta.max_dim_values")
def _():
    adj = _graph()
    return {
        (pos, md): T.count_rc_edges_skeleta(adj, position=pos, max_dim=md)
        for pos in POSITIONS
        for md in (-1, 1, 2, 3, 10)
    }


@case("topology.count_rc_edges_skeleta.return_matrices")
def _():
    adj = _graph()
    return {
        pos: T.count_rc_edges_skeleta(adj, position=pos, return_mats=True, max_dim=3)
        for pos in POSITIONS
    }


@case("topology.count_rc_edges_skeleta.simplex_types_position_all")
def _():
    # reciprocal edges of the skeleta of the undirected/reciprocal complexes: their matrices are upper triangular, so none
    adj = _graph()
    return {t: T.count_rc_edges_skeleta(adj, position="all", simplex_type=t) for t in TYPES}


@case("topology.count_rc_edges_skeleta.simplex_type_is_ignored_for_the_spine")
def _():
    adj = _graph()
    return [T.count_rc_edges_skeleta(adj, position="spine", simplex_type=t) for t in TYPES]


@case("topology.count_rc_edges_skeleta.threads_arguments")
def _():
    adj = _graph()
    return {
        "all_threads_1": T.count_rc_edges_skeleta(adj, threads=1),
        "spine_threads_ignored": T.count_rc_edges_skeleta(adj, position="spine", threads=3),
        "spine_n_threads_forwarded": T.count_rc_edges_skeleta(adj, position="spine", n_threads=3),
    }


@case("topology.count_rc_edges_skeleta.dense_adjacency")
def _():
    adj = _graph().toarray()
    return {pos: T.count_rc_edges_skeleta(adj, position=pos) for pos in POSITIONS}


@case("topology.count_rc_edges_skeleta.precomputed_simplex_list")
def _():
    lists = simplex_series(_graph())
    return {
        "all": T.count_rc_edges_skeleta(simplex_list=lists, N=14, position="spine"),
        "max_dim_is_ignored": T.count_rc_edges_skeleta(
            simplex_list=lists, N=14, position="spine", max_dim=2
        ),
        "sliced": T.count_rc_edges_skeleta(
            simplex_list=lists.loc[2:3], N=14, position="spine", return_mats=True
        ),
        "maximal": T.count_rc_edges_skeleta(
            simplex_list=simplex_series(_graph(), max_simplices=True), N=14, position="spine"
        ),
    }


@case("topology.count_rc_edges_skeleta.precomputed_edge_participation")
def _():
    edge_par = edge_participation_frame(_graph())
    return {
        "all": T.count_rc_edges_skeleta(edge_par=edge_par, N=14, position="all"),
        "max_dim_is_ignored": T.count_rc_edges_skeleta(edge_par=edge_par, N=14, max_dim=2),
        "return_mats": T.count_rc_edges_skeleta(edge_par=edge_par, N=14, return_mats=True),
        "maximal": T.count_rc_edges_skeleta(
            edge_par=edge_participation_frame(_graph(), max_simplices=True), N=14, return_mats=True
        ),
    }


@case("topology.count_rc_edges_skeleta.precomputed_library_output")
def _():
    adj = _graph()
    return {
        "simplex_list": T.count_rc_edges_skeleta(
            simplex_list=T.list_simplices_by_dimension(adj), N=14, position="spine"
        ),
        "edge_par": T.count_rc_edges_skeleta(edge_par=T.edge_participation(adj), N=14),
    }


@case("topology.count_rc_edges_skeleta.precomputed_simplex_list_size_too_small")
def _():
    return T.count_rc_edges_skeleta(simplex_list=simplex_series(_graph()), N=9, position="spine")


@case("topology.count_rc_edges_skeleta.precomputed_simplex_list_size_missing")
def _():
    return T.count_rc_edges_skeleta(simplex_list=simplex_series(_graph()), position="spine")


@case("topology.count_rc_edges_skeleta.precomputed_edge_participation_size_too_small")
def _():
    return T.count_rc_edges_skeleta(edge_par=edge_participation_frame(_graph()), N=5)


@case("topology.count_rc_edges_skeleta.precomputed_edge_participation_size_missing")
def _():
    return T.count_rc_edges_skeleta(edge_par=edge_participation_frame(_graph()))


@case("topology.count_rc_edges_skeleta.nothing_to_compute_from")
def _():
    return T.count_rc_edges_skeleta()


@case("topology.count_rc_edges_skeleta.simplex_list_only_is_not_enough_for_position_all")
def _():
    return T.count_rc_edges_skeleta(simplex_list=simplex_series(_graph()), N=14, position="all")


@case("topology.count_rc_edges_skeleta.edge_participation_only_is_not_enough_for_position_spine")
def _():
    return T.count_rc_edges_skeleta(
        edge_par=edge_participation_frame(_graph()), N=14, position="spine"
    )


@case("topology.count_rc_edges_skeleta.unknown_position")
def _():
    return T.count_rc_edges_skeleta(_graph(), position="middle")


@case("topology.count_rc_edges_skeleta.special_graphs")
def _():
    return {
        (name, pos): T.count_rc_edges_skeleta(adj, position=pos)
        for name, adj in (
            ("complete_4", complete(4)),
            ("transitive_5", transitive(5)),
            ("one_edge", from_edges(2, [(0, 1)])),
            ("reciprocal_pair", from_edges(2, [(0, 1), (1, 0)])),
            ("structured", structured()),
        )
        for pos in POSITIONS
    }


@case("topology.count_rc_edges_skeleta.maximal_simplices_with_empty_dimensions")
def _():
    # dimensions without maximal simplices: edge count 0, and NaN reciprocal columns in the spine version
    return {
        (name, pos): T.count_rc_edges_skeleta(adj, position=pos, max_simplices=True)
        for name, adj in (
            ("mixed", _mixed()),
            ("complete_4", complete(4)),
            ("transitive_5", transitive(5)),
        )
        for pos in POSITIONS
    }


@case("topology.count_rc_edges_skeleta.maximal_simplices_with_empty_dimensions_and_matrices_all")
def _():
    return T.count_rc_edges_skeleta(_mixed(), position="all", max_simplices=True, return_mats=True)


@case("topology.count_rc_edges_skeleta.spine_matrices_of_empty_dimension_reuse_the_previous_one")
def _():
    # dimension 2 has no maximal simplices; the matrix stored for it is the one left over from dimension 1
    return T.count_rc_edges_skeleta(
        _mixed(), position="spine", max_simplices=True, return_mats=True
    )


@case("topology.count_rc_edges_skeleta.spine_matrices_when_the_first_dimension_is_empty")
def _():
    return T.count_rc_edges_skeleta(
        complete(4), position="spine", max_simplices=True, return_mats=True
    )


@case("topology.count_rc_edges_skeleta.graph_without_edges_spine")
def _():
    return T.count_rc_edges_skeleta(from_edges(5, []), position="spine")


@case("topology.count_rc_edges_skeleta.graph_without_edges_all")
def _():
    return T.count_rc_edges_skeleta(from_edges(5, []), position="all")


@case("topology.count_rc_edges_skeleta.spine_verbose_keyword_collides")
def _():
    return T.count_rc_edges_skeleta(_graph(), position="spine", verbose=True)


@case("topology.count_rc_edges_skeleta.extra_arguments_all")
def _():
    return T.count_rc_edges_skeleta(_graph(), position="all", unknown_option=1, verbose=True)


@case("topology.count_rc_edges_skeleta.diagonal_is_rejected")
def _():
    return T.count_rc_edges_skeleta(with_loops(_graph(), (2,)))


# ----------------------------------------------------------------------------- bedge_counts


@case("topology.bedge_counts.default")
def _():
    return T.bedge_counts(_graph())


@case("topology.bedge_counts.precomputed_simplices_equal_the_computed_ones")
def _():
    adj = _graph()
    return [T.bedge_counts(adj, simplices=simplex_series(adj)), T.bedge_counts(adj)]


@case("topology.bedge_counts.library_simplex_lists_as_input")
def _():
    adj = _graph()
    return T.bedge_counts(adj, simplices=T.list_simplices_by_dimension(adj))


@case("topology.bedge_counts.max_dim_values")
def _():
    adj = _graph()
    return {md: T.bedge_counts(adj, max_dim=md) for md in (-1, 0, 1, 2, 3)}


@case("topology.bedge_counts.simplex_types")
def _():
    adj = _graph()
    return {t: T.bedge_counts(adj, simplex_type=t) for t in TYPES}


@case("topology.bedge_counts.sliced_simplex_list")
def _():
    adj = _graph()
    lists = simplex_series(adj)
    return {
        "loc_2_3": T.bedge_counts(adj, simplices=lists.loc[2:3]),
        "loc_1_1": T.bedge_counts(adj, simplices=lists.loc[1:1]),
        "without_dimension_zero": T.bedge_counts(adj, simplices=lists.iloc[1:]),
    }


@case("topology.bedge_counts.simplices_of_another_set_of_nodes")
def _():
    # the simplices need not come from the adjacency matrix: the counts are taken in `adjacency` on their nodes
    adj = _graph()
    other = simplex_series(rich(14, 0.1, seed=4, sizes=(4, 3), rc_sizes=(3,)))
    return T.bedge_counts(adj, simplices=other)


@case("topology.bedge_counts.maximal_simplices", tags=("fixed",))
def _():
    # the dimension 0 list of maximal simplices is an empty 1-d array and has no second axis
    return T.bedge_counts(_graph(), max_simplices=True)


@case("topology.bedge_counts.maximal_simplices_given_explicitly", tags=("fixed",))
def _():
    adj = _graph()
    return T.bedge_counts(adj, simplices=simplex_series(adj, max_simplices=True))


@case("topology.bedge_counts.maximal_simplices_without_dimension_zero")
def _():
    adj = _graph()
    return T.bedge_counts(adj, simplices=simplex_series(adj, max_simplices=True).iloc[1:])


@case("topology.bedge_counts.orientation_of_the_matrices")
def _():
    # a transitive tournament has all its edges forward; the matrices count them in the lower triangle
    return T.bedge_counts(transitive(4))


@case("topology.bedge_counts.special_graphs")
def _():
    return {
        name: T.bedge_counts(adj)
        for name, adj in (
            ("complete_4", complete(4)),
            ("transitive_5", transitive(5)),
            ("one_edge", from_edges(2, [(0, 1)])),
            ("reciprocal_pair", from_edges(2, [(0, 1), (1, 0)])),
            ("structured", structured()),
            ("mixed", _mixed()),
        )
    }


@case("topology.bedge_counts.graph_without_edges")
def _():
    # a dimension without simplices gives a Series of column sums instead of a matrix
    return T.bedge_counts(from_edges(5, []))


@case("topology.bedge_counts.weighted_adjacency_counts_weights")
def _():
    adj = _graph().astype(int)
    adj.data[:] = 2
    return T.bedge_counts(adj)


@case("topology.bedge_counts.keyword_arguments_reach_the_listing")
def _():
    adj = _graph()
    return {
        "threads": T.bedge_counts(adj, threads=2),
        "n_threads": T.bedge_counts(adj, n_threads=2),
        "nodes": T.bedge_counts(adj, nodes=np.array([0, 3, 7])),
        "unknown": T.bedge_counts(adj, unknown_option=1),
    }


@case("topology.bedge_counts.dense_adjacency")
def _():
    return T.bedge_counts(_graph().toarray())


@case("topology.bedge_counts.python_list_of_simplices")
def _():
    return T.bedge_counts(_graph(), simplices=[np.array([[0, 1]])])


@case("topology.bedge_counts.csc_adjacency")
def _():
    adj = _graph()
    return T.bedge_counts(adj.tocsc(), simplices=simplex_series(adj).loc[1:3])
