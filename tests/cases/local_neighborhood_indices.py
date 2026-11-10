"""``neighborhood_indices`` in ``connalysis.network.local``: neighbours of every node, or of chosen centers.

The result is a Series (index ``center``, name ``neighbors``, dtype object) whose items are int64 arrays.  It is built
with pandas ``combine``/``union1d``, so the order of the centers in the result index is *not* always the order given.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, tutorial_connectome, weighted

from connalysis.network import local as L


def _mixed(fmt="csr", dtype=bool):
    """16 nodes with reciprocal pairs, two isolated nodes (5, 11), a pure source (12) and a pure sink (13)."""
    dense = digraph(16, 0.14, seed=3, reciprocal=0.3).toarray()
    dense[[5, 11], :] = False
    dense[:, [5, 11]] = False
    dense[:, 12] = False
    dense[12, [1, 2, 7]] = True
    dense[13, :] = False
    dense[[0, 4, 9], 13] = True
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


COMBOS = {
    "both": dict(pre=True, post=True),
    "pre": dict(pre=True, post=False),
    "post": dict(pre=False, post=True),
}
UNSORTED = np.array(
    [13, 5, 2, 12, 9, 0]
)  # isolated 5, pure source 12, pure sink 13, in no particular order


@case("local.neighborhood_indices.all_nodes_default")
def _():
    return L.neighborhood_indices(_mixed())


@case("local.neighborhood_indices.pre_post_combinations")
def _():
    adj = _mixed()
    return {name: L.neighborhood_indices(adj, **kw) for name, kw in COMBOS.items()}


@case("local.neighborhood_indices.centers_sorted")
def _():
    adj = _mixed()
    centers = np.array([0, 2, 5, 12, 13])
    return {
        name: L.neighborhood_indices(adj, all_nodes=False, centers=centers, **kw)
        for name, kw in COMBOS.items()
    }


@case("local.neighborhood_indices.centers_unsorted_order_of_index")
def _():
    # the index of the result comes out sorted although the centers were given in another order
    adj = _mixed()
    return {
        name: L.neighborhood_indices(adj, all_nodes=False, centers=UNSORTED, **kw)
        for name, kw in COMBOS.items()
    }


@case("local.neighborhood_indices.centers_unsorted_all_without_neighbours")
def _():
    # when no center has a neighbour in the requested direction the given order of the centers is kept
    adj = _mixed()
    return {
        "isolated_both": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([11, 5])),
        "source_and_isolated_pre": L.neighborhood_indices(
            adj, pre=True, post=False, all_nodes=False, centers=np.array([12, 5])
        ),
        "sink_and_isolated_post": L.neighborhood_indices(
            adj, pre=False, post=True, all_nodes=False, centers=np.array([13, 5, 11])
        ),
        "isolated_one_center": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([5])),
    }


@case("local.neighborhood_indices.centers_one_direction_missing")
def _():
    # centers that have neighbours in one direction only: the other direction contributes an empty array
    adj = _mixed()
    centers = np.array([12, 1, 13])
    return {
        "pre": L.neighborhood_indices(adj, pre=True, post=False, all_nodes=False, centers=centers),
        "post": L.neighborhood_indices(adj, pre=False, post=True, all_nodes=False, centers=centers),
    }


@case("local.neighborhood_indices.one_and_no_center")
def _():
    adj = _mixed()
    return {
        "one": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([2])),
        "none": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([], dtype=int)),
    }


@case("local.neighborhood_indices.centers_outside_the_matrix")
def _():
    # nodes that are not in the matrix get an empty neighbourhood instead of an error
    adj = _mixed()
    return {
        "above": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([40, 3])),
        "negative": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([-1, 3])),
        "only_above": L.neighborhood_indices(adj, all_nodes=False, centers=np.array([99])),
    }


@case("local.neighborhood_indices.centers_ignored_when_all_nodes")
def _():
    return L.neighborhood_indices(_mixed(), all_nodes=True, centers=np.array([3, 1]))


@case("local.neighborhood_indices.centers_types")
def _():
    adj = _mixed()
    return {
        "int32": L.neighborhood_indices(
            adj, all_nodes=False, centers=np.array([9, 1, 4], dtype=np.int32)
        ),
        "index": L.neighborhood_indices(adj, all_nodes=False, centers=pd.Index([9, 1, 4])),
        "series": L.neighborhood_indices(adj, all_nodes=False, centers=pd.Series([9, 1, 4])),
        "arange": L.neighborhood_indices(adj, all_nodes=False, centers=np.arange(3, 8)),
    }


@case("local.neighborhood_indices.formats")
def _():
    adj = _mixed()
    dense = adj.toarray()
    inputs = {fmt: adj.asformat(fmt) for fmt in ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")}
    inputs.update(ndarray=dense, matrix=np.matrix(dense))
    out = {}
    for name, m in inputs.items():
        out[name] = {
            "all": L.neighborhood_indices(m),
            "centers": L.neighborhood_indices(
                m, pre=True, post=False, all_nodes=False, centers=UNSORTED
            ),
        }
    return out


@case("local.neighborhood_indices.sparse_arrays")
def _():
    adj = _mixed()
    return {
        "csr_array": L.neighborhood_indices(sp.csr_array(adj)),
        "coo_array": L.neighborhood_indices(sp.coo_array(adj), all_nodes=False, centers=UNSORTED),
    }


@case("local.neighborhood_indices.keyword_arguments")
def _():
    return L.neighborhood_indices(
        M=_mixed(), pre=True, post=False, all_nodes=False, centers=np.array([3, 1])
    )


@case("local.neighborhood_indices.value_dtypes")
def _():
    # only the position of the non-zero entries matters, not the values
    adj = _mixed()
    wts = weighted(16, 0.2, seed=4, integers=False)
    out = {
        str(np.dtype(dt)): L.neighborhood_indices(adj.astype(dt))
        for dt in (bool, np.int8, np.int64, np.float32, np.float64)
    }
    out["negative"] = L.neighborhood_indices(-2.5 * adj.astype(float))
    out["weighted_real"] = L.neighborhood_indices(wts)
    out["weighted_real_centers"] = L.neighborhood_indices(
        wts, all_nodes=False, centers=np.array([8, 1, 15])
    )
    return out


@case("local.neighborhood_indices.explicit_zeros")
def _():
    # a stored zero is an edge for a sparse matrix (it is in coo.row/col) but not for the dense equivalent
    rows, cols = np.array([0, 1, 2, 3, 3]), np.array([1, 2, 0, 4, 5])
    data = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    sparse = sp.csr_matrix((data, (rows, cols)), shape=(6, 6))
    assert sparse.nnz == 5
    return {
        "sparse": L.neighborhood_indices(sparse),
        "sparse_centers": L.neighborhood_indices(sparse, all_nodes=False, centers=np.array([4, 1])),
        "dense": L.neighborhood_indices(sparse.toarray()),
    }


@case("local.neighborhood_indices.coo_with_duplicate_entries")
def _():
    # repeated coordinates stay repeated in a coo matrix and (0, 1) sums to zero: still an entry of the sparse
    # matrices, absent from the dense one
    row = np.array([0, 0, 0, 1, 2, 2, 3])
    col = np.array([1, 1, 1, 2, 3, 3, 0])
    coo = sp.coo_matrix((np.array([1.0, 2.0, -3.0, 1.0, 1.0, 1.0, 5.0]), (row, col)), shape=(5, 5))
    return {
        "coo": L.neighborhood_indices(coo),
        "csr": L.neighborhood_indices(coo.tocsr()),
        "dense": L.neighborhood_indices(coo.toarray()),
    }


def _raw(m):
    """The stored structure of a sparse matrix, exactly as it is (duplicates, explicit zeros, index order)."""
    if m.format == "coo":
        return [m.nnz, m.row.tolist(), m.col.tolist(), m.data.tolist()]
    return [m.nnz, m.indptr.tolist(), m.indices.tolist(), m.data.tolist()]


@case("local.neighborhood_indices.input_not_modified")
def _():
    # a coo matrix with a repeated coordinate and a stored zero, a csr matrix with unsorted column indices
    coo = sp.coo_matrix(
        (
            np.array([1.0, 2.0, 0.0, 4.0, 5.0]),
            (np.array([2, 2, 0, 1, 3]), np.array([1, 1, 3, 2, 0])),
        ),
        shape=(4, 4),
    )
    csr = sp.csr_matrix(
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([3, 1, 2, 0]), np.array([0, 2, 3, 4, 4])),
        shape=(4, 4),
    )
    assert not csr.has_sorted_indices
    before = [_raw(coo), _raw(csr)]
    result = [
        L.neighborhood_indices(coo),
        L.neighborhood_indices(csr, all_nodes=False, centers=np.array([2, 0])),
    ]
    return result, before, [_raw(coo), _raw(csr)], csr.has_sorted_indices, csr.has_canonical_format


@case("local.neighborhood_indices.self_loops")
def _():
    # a node with a loop is its own neighbour
    dense = _mixed().toarray()
    dense[3, 3] = dense[5, 5] = dense[13, 13] = True
    adj = sp.csr_matrix(dense)
    return {name: L.neighborhood_indices(adj, **kw) for name, kw in COMBOS.items()}


@case("local.neighborhood_indices.edgeless")
def _():
    return {
        "sparse": L.neighborhood_indices(sp.csr_matrix((5, 5), dtype=bool)),
        "dense": L.neighborhood_indices(np.zeros((4, 4), dtype=int)),
        "sparse_pre": L.neighborhood_indices(
            sp.csr_matrix((3, 3), dtype=bool), pre=True, post=False
        ),
    }


@case("local.neighborhood_indices.tiny_graphs")
def _():
    loop = sp.csr_matrix(np.array([[1]], dtype=bool))
    one_edge = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=bool))
    both_ways = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool))
    return {
        "single_node": L.neighborhood_indices(sp.csr_matrix((1, 1), dtype=bool)),
        "single_node_loop": L.neighborhood_indices(loop),
        "two_nodes_one_edge": {
            name: L.neighborhood_indices(one_edge, **kw) for name, kw in COMBOS.items()
        },
        "two_nodes_reciprocal": L.neighborhood_indices(both_ways),
        "zero_nodes": L.neighborhood_indices(sp.csr_matrix((0, 0), dtype=bool)),
    }


@case("local.neighborhood_indices.complete_graph")
def _():
    dense = np.ones((6, 6), dtype=bool)
    np.fill_diagonal(dense, False)
    adj = sp.csr_matrix(dense)
    return {
        "all": L.neighborhood_indices(adj),
        "pre_centers": L.neighborhood_indices(
            adj, pre=True, post=False, all_nodes=False, centers=np.array([5, 0])
        ),
    }


@case("local.neighborhood_indices.tutorial_all_nodes")
def _():
    adj, _nodes = tutorial_connectome(400)
    return L.neighborhood_indices(adj)


@case("local.neighborhood_indices.tutorial_centers")
def _():
    adj, _nodes = tutorial_connectome(600)
    centers = np.random.RandomState(7).permutation(600)[:80]
    return {
        name: L.neighborhood_indices(adj, all_nodes=False, centers=centers, **kw)
        for name, kw in COMBOS.items()
    }


# ----------------------------------------------------------------------------------------------- failures


@case("local.neighborhood_indices.not_square")
def _():
    return L.neighborhood_indices(sp.csr_matrix(np.ones((3, 4))))


@case("local.neighborhood_indices.pre_and_post_false")
def _():
    return L.neighborhood_indices(_mixed(), pre=False, post=False)


@case("local.neighborhood_indices.centers_required")
def _():
    return L.neighborhood_indices(_mixed(), all_nodes=False)


@case("local.neighborhood_indices.centers_as_list")
def _():
    # centers.shape is used, so a plain list fails although only "1d-array" is documented
    return L.neighborhood_indices(_mixed(), all_nodes=False, centers=[3, 1])


@case("local.neighborhood_indices.duplicate_centers")
def _():
    # defect: combine() then looks up a repeated label twice and np.union1d fails
    return L.neighborhood_indices(_mixed(), all_nodes=False, centers=np.array([2, 2, 0]))


@case("local.neighborhood_indices.duplicate_isolated_centers_both")
def _():
    return L.neighborhood_indices(_mixed(), all_nodes=False, centers=np.array([5, 5]))


@case("local.neighborhood_indices.duplicate_isolated_centers_pre_only")
def _():
    return L.neighborhood_indices(
        _mixed(), pre=True, post=False, all_nodes=False, centers=np.array([5, 5])
    )


@case("local.neighborhood_indices.edgeless_with_centers")
def _():
    # defect: base_df.apply on an empty frame fails when only some centers are requested
    return L.neighborhood_indices(
        sp.csr_matrix((5, 5), dtype=bool), all_nodes=False, centers=np.array([3, 1])
    )
