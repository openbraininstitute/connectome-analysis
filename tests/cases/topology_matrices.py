"""More variants of ``rc_submatrix``, ``underlying_undirected_matrix`` and ``node_degree`` (the basic ones are in
``topology_basic``): input formats, dtypes, weights, explicit zeros, diagonals, degenerate graphs and the odd corners."""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, node_table, weighted

from cases.topology_helpers import (
    complete,
    explicit_zeros,
    from_edges,
    rich,
    structured,
    with_loops,
)
from connalysis.network import topology as T

FORMATS = ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")


def _with_duplicates():
    """Non-canonical CSR matrix that stores some entries twice: (0,1) = 2 + 3, (1,0) = 1 + 1, (2,0) = 4 + 1 and a cancelling
    pair at (3,2) = 1 - 1 with a plain reverse edge (2,3)."""
    rows = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    cols = [1, 1, 2, 0, 0, 0, 0, 3, 2, 2]
    data = [2.0, 3.0, 1.0, 1.0, 1.0, 4.0, 1.0, 5.0, 1.0, -1.0]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=4))])
    return sp.csr_matrix((np.array(data), np.array(cols), indptr), shape=(4, 4))


# ----------------------------------------------------------------------------- rc_submatrix


@case("topology.rc_submatrix.formats")
def _():
    adj = digraph(30, 0.15, seed=21, reciprocal=0.5)
    out = {fmt: T.rc_submatrix(adj.asformat(fmt)) for fmt in FORMATS}
    out["csr_array"] = T.rc_submatrix(sp.csr_array(adj))
    out["coo_array"] = T.rc_submatrix(sp.coo_array(adj))
    out["np.matrix"] = T.rc_submatrix(np.matrix(adj.toarray()))
    out["nested_list"] = T.rc_submatrix(adj.toarray().tolist())
    return out


@case("topology.rc_submatrix.dtypes")
def _():
    adj = digraph(25, 0.2, seed=22, reciprocal=0.5)
    out = {}
    for dtype in (bool, np.int8, np.uint8, np.int64, np.float32, np.float64):
        typed = adj.astype(dtype)
        if dtype is not bool:
            typed.data[:] = 3
        out[np.dtype(dtype).name] = T.rc_submatrix(typed)
    out["dense_float"] = T.rc_submatrix(adj.toarray().astype(float) * 2.5)
    out["dense_int"] = T.rc_submatrix(adj.toarray().astype(int))
    return out


@case("topology.rc_submatrix.weighted_pairs_keep_both_weights")
def _():
    # Reciprocal pairs keep their own weight in each direction (the weights of a pair differ here).
    adj = (
        weighted(25, 0.12, seed=23, integers=False) + weighted(25, 0.12, seed=24, integers=False).T
    )
    return T.rc_submatrix(adj)


@case("topology.rc_submatrix.signed_weights")
def _():
    adj = digraph(20, 0.3, seed=25, reciprocal=0.6).astype(float).tocsr()
    adj.data[::2] = -2.5
    return T.rc_submatrix(adj)


@case("topology.rc_submatrix.explicit_zeros")
def _():
    # Explicitly stored zeros are not edges: they must not make a pair reciprocal.
    adj = digraph(20, 0.3, seed=26, reciprocal=0.6)
    return T.rc_submatrix(explicit_zeros(adj, 6))


@case("topology.rc_submatrix.diagonal_is_kept")
def _():
    # Only a warning: a loop is its own reverse, so it survives (and so do the weights on it).
    adj = with_loops(digraph(15, 0.25, seed=27, reciprocal=0.5), (2, 7, 11))
    return [T.rc_submatrix(adj), T.rc_submatrix(adj.astype(float) * 4.0)]


@case("topology.rc_submatrix.unsorted_indices")
def _():
    adj = digraph(20, 0.3, seed=28, reciprocal=0.5).astype(int)
    indices, data = adj.indices.copy(), adj.data.copy()
    for i in range(adj.shape[0]):  # reverse the column order inside every row
        lo, hi = adj.indptr[i], adj.indptr[i + 1]
        indices[lo:hi], data[lo:hi] = indices[lo:hi][::-1].copy(), data[lo:hi][::-1].copy()
    unsorted = sp.csr_matrix((data, indices, adj.indptr.copy()), shape=adj.shape)
    return T.rc_submatrix(unsorted)


@case("topology.rc_submatrix.duplicate_entries")
def _():
    # stored duplicates are summed in the weights, but reciprocity is read from the stored flags (also for a cancelled pair)
    adj = _with_duplicates()
    return [T.rc_submatrix(adj), T.rc_submatrix(adj.astype(bool))]


@case("topology.rc_submatrix.degenerate_graphs")
def _():
    two_way = from_edges(2, [(0, 1), (1, 0)])
    return {
        "empty": T.rc_submatrix(from_edges(6, [])),
        "single_node": T.rc_submatrix(from_edges(1, [])),
        "no_nodes": T.rc_submatrix(from_edges(0, [])),
        "one_way_pair": T.rc_submatrix(from_edges(2, [(0, 1)])),
        "reciprocal_pair": T.rc_submatrix(two_way),
        "complete": T.rc_submatrix(complete(5)),
        "symmetric_input": T.rc_submatrix(
            (digraph(12, 0.2, seed=29) + digraph(12, 0.2, seed=29).T).astype(bool)
        ),
        "all_one_way": T.rc_submatrix(
            structured()[:5, :5]
        ),  # transitive tournament: nothing reciprocal
    }


@case("topology.rc_submatrix.structured_graph")
def _():
    return T.rc_submatrix(structured())


@case("topology.rc_submatrix.non_square")
def _():
    return T.rc_submatrix(sp.csr_matrix(np.ones((3, 4), dtype=bool)))


# ----------------------------------------------------------------------------- underlying_undirected_matrix


@case("topology.underlying_undirected_matrix.formats")
def _():
    adj = digraph(30, 0.1, seed=31, reciprocal=0.3)
    out = {fmt: T.underlying_undirected_matrix(adj.asformat(fmt)) for fmt in FORMATS}
    out["csr_array"] = T.underlying_undirected_matrix(sp.csr_array(adj))
    out["np.matrix"] = T.underlying_undirected_matrix(np.matrix(adj.toarray()))
    out["dense_bool"] = T.underlying_undirected_matrix(adj.toarray())
    out["dense_float"] = T.underlying_undirected_matrix(adj.toarray().astype(float))
    out["nested_list"] = T.underlying_undirected_matrix(adj.toarray().tolist())
    return out


@case("topology.underlying_undirected_matrix.weighted")
def _():
    return [
        T.underlying_undirected_matrix(weighted(25, 0.2, seed=32)),
        T.underlying_undirected_matrix(weighted(25, 0.2, seed=33, integers=False)),
    ]


@case("topology.underlying_undirected_matrix.cancelling_weights")
def _():
    # a(i,j) + a(j,i) == 0 drops the edge: the matrix is added to its transpose and only then cast to bool
    adj = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0],
                [-1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, -3.0],
                [0.0, 3.0, 3.0, 0.0],
            ]
        )
    )
    return T.underlying_undirected_matrix(adj)


@case("topology.underlying_undirected_matrix.small_integer_overflow")
def _():
    adj = sp.csr_matrix(
        np.array([[0, 255, 3], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    )  # 255 + 1 wraps to 0
    adj8 = sp.csr_matrix(
        np.array([[0, 127, 3], [127, 0, 0], [0, 0, 0]], dtype=np.int8)
    )  # 127 + 127 wraps to -2
    return [T.underlying_undirected_matrix(adj), T.underlying_undirected_matrix(adj8)]


@case("topology.underlying_undirected_matrix.duplicate_entries")
def _():
    return T.underlying_undirected_matrix(_with_duplicates())


@case("topology.underlying_undirected_matrix.explicit_zeros")
def _():
    return T.underlying_undirected_matrix(
        explicit_zeros(digraph(20, 0.2, seed=34, reciprocal=0.3), 5)
    )


@case("topology.underlying_undirected_matrix.diagonal_is_kept")
def _():
    return T.underlying_undirected_matrix(with_loops(digraph(15, 0.2, seed=35), (0, 6)))


@case("topology.underlying_undirected_matrix.degenerate_graphs")
def _():
    return {
        "empty": T.underlying_undirected_matrix(from_edges(5, [])),
        "single_node": T.underlying_undirected_matrix(from_edges(1, [])),
        "no_nodes": T.underlying_undirected_matrix(from_edges(0, [])),
        "one_way_pair": T.underlying_undirected_matrix(from_edges(2, [(0, 1)])),
        "complete": T.underlying_undirected_matrix(complete(4)),
        "symmetric_input": T.underlying_undirected_matrix(
            rich(15, 0.1, seed=36) + rich(15, 0.1, seed=36).T
        ),
        "structured": T.underlying_undirected_matrix(structured()),
    }


@case("topology.underlying_undirected_matrix.non_square")
def _():
    return T.underlying_undirected_matrix(sp.csr_matrix((3, 4), dtype=bool))


# ----------------------------------------------------------------------------- node_degree


@case("topology.node_degree.in")
def _():
    return T.node_degree(structured(), direction="IN")


@case("topology.node_degree.out")
def _():
    return T.node_degree(structured(), direction="OUT")


@case("topology.node_degree.in_and_out_tuple")
def _():
    return T.node_degree(structured(), direction=("IN", "OUT"))


@case("topology.node_degree.in_and_out_list")
def _():
    return T.node_degree(structured(), direction=["IN", "OUT"])


@case("topology.node_degree.out_in_order_is_rejected")
def _():
    # The ("OUT", "IN") branch of the function is unreachable: the assertion at the top refuses that order.
    return T.node_degree(structured(), direction=("OUT", "IN"))


@case("topology.node_degree.out_in_list_is_rejected")
def _():
    return T.node_degree(structured(), direction=["OUT", "IN"])


@case("topology.node_degree.empty_direction_means_total")
def _():
    adj = digraph(20, 0.15, seed=41, reciprocal=0.4)
    return {repr(d): T.node_degree(adj, direction=d) for d in (None, (), [], "", 0)}


@case("topology.node_degree.lowercase_direction")
def _():
    return T.node_degree(structured(), direction="in")


@case("topology.node_degree.single_element_direction")
def _():
    return T.node_degree(structured(), direction=("IN",))


@case("topology.node_degree.three_element_direction")
def _():
    return T.node_degree(structured(), direction=("IN", "OUT", "IN"))


@case("topology.node_degree.array_direction")
def _():
    return T.node_degree(structured(), direction=np.array(["IN", "OUT"]))


@case("topology.node_degree.weighted_total")
def _():
    return T.node_degree(weighted(25, 0.2, seed=42), weighted=True)


@case("topology.node_degree.weighted_real_in_and_out")
def _():
    adj = weighted(25, 0.2, seed=43, integers=False)
    return [
        T.node_degree(adj, direction="OUT", weighted=True),
        T.node_degree(adj, direction=("IN", "OUT"), weighted=True),
    ]


@case("topology.node_degree.unweighted_ignores_weights")
def _():
    adj = weighted(25, 0.2, seed=43, integers=False)
    return [T.node_degree(adj, direction=("IN", "OUT")), T.node_degree(adj)]


@case("topology.node_degree.signed_weights")
def _():
    adj = digraph(20, 0.25, seed=44, reciprocal=0.4).astype(float).tocsr()
    adj.data[::3] = -1.5
    return [
        T.node_degree(adj, weighted=True, direction=("IN", "OUT")),
        T.node_degree(adj, direction=("IN", "OUT")),
    ]


@case("topology.node_degree.weighted_dense")
def _():
    dense = weighted(15, 0.3, seed=45, integers=False).toarray()
    return [
        T.node_degree(dense, weighted=True, direction="IN"),
        T.node_degree(dense, direction="IN"),
    ]


@case("topology.node_degree.dense_dtypes")
def _():
    dense = digraph(18, 0.25, seed=46, reciprocal=0.4).toarray()
    return {
        np.dtype(d).name: T.node_degree(dense.astype(d), direction=("IN", "OUT"), weighted=True)
        for d in (bool, np.int8, np.int64, np.float32, np.float64)
    }


@case("topology.node_degree.formats")
def _():
    adj = digraph(22, 0.15, seed=47, reciprocal=0.3)
    return {fmt: T.node_degree(adj.asformat(fmt), direction=("IN", "OUT")) for fmt in FORMATS}


@case("topology.node_degree.sparse_arrays")
def _():
    adj = digraph(22, 0.15, seed=47, reciprocal=0.3)
    return [T.node_degree(sp.csr_array(adj), direction="IN"), T.node_degree(sp.coo_array(adj))]


@case("topology.node_degree.explicit_zeros")
def _():
    adj = explicit_zeros(digraph(20, 0.25, seed=48, reciprocal=0.4), 8)
    return [
        T.node_degree(adj, direction=("IN", "OUT"), weighted=True),
        T.node_degree(adj, direction=("IN", "OUT")),
    ]


@case("topology.node_degree.duplicate_entries")
def _():
    adj = _with_duplicates()
    return [
        T.node_degree(adj, direction=("IN", "OUT")),
        T.node_degree(adj, direction=("IN", "OUT"), weighted=True),
    ]


@case("topology.node_degree.diagonal_counts_as_in_and_out")
def _():
    # Only a warning: a loop adds one to the in-degree and one to the out-degree (two to the total).
    adj = with_loops(digraph(15, 0.2, seed=49, reciprocal=0.3), (1, 4, 4, 9))
    return [
        T.node_degree(adj),
        T.node_degree(adj, direction=("IN", "OUT")),
        T.node_degree(adj.astype(float) * 2.0, direction="OUT", weighted=True),
    ]


@case("topology.node_degree.degenerate_graphs")
def _():
    return {
        "empty": T.node_degree(from_edges(5, []), direction=("IN", "OUT")),
        "empty_total": T.node_degree(from_edges(5, [])),
        "single_node": T.node_degree(from_edges(1, []), direction="IN"),
        "no_nodes": T.node_degree(from_edges(0, []), direction=("IN", "OUT")),
        "no_nodes_total": T.node_degree(from_edges(0, [])),
        "one_way_pair": T.node_degree(from_edges(2, [(0, 1)]), direction=("IN", "OUT")),
        "complete": T.node_degree(complete(5), direction=("IN", "OUT")),
        "star_hub": T.node_degree(
            from_edges(8, [(0, k) for k in range(1, 8)]), direction=("IN", "OUT")
        ),
    }


@case("topology.node_degree.extra_arguments_are_ignored")
def _():
    adj = digraph(15, 0.2, seed=50)
    return T.node_degree(
        adj, node_properties=node_table(15), direction="IN", threads=3, unknown_option=True
    )


@case("topology.node_degree.non_square_out_dense")
def _():
    # Rows are summed per row, so a dense non-square matrix gives an out-degree per row.
    return T.node_degree(np.arange(15).reshape(3, 5) % 2, direction="OUT")


@case("topology.node_degree.non_square_in_dense")
def _():
    return T.node_degree(np.ones((3, 5)), direction="IN")


@case("topology.node_degree.np_matrix_input")
def _():
    return T.node_degree(np.matrix(digraph(10, 0.3, seed=51).toarray()), direction="IN")


@case("topology.node_degree.input_is_not_modified")
def _():
    dense = weighted(12, 0.3, seed=52).toarray()
    before = dense.copy()
    result = T.node_degree(dense, weighted=False, direction=("IN", "OUT"))
    return result, dense, before
