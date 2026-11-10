"""Sub-matrices of ``connalysis.network.local``: ``submat_at_ind``, ``neighborhood``,
``neighborhood_of_set_indices`` and ``neighborhood_of_set``.

``neighborhood_of_set`` is broken for its default ``include_centers=True`` (it uses an undefined name) and, with
``include_centers=False``, only handles dense arrays, CSR, CSC and COO input.
"""

import numpy as np
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


SPARSE_FORMATS = ("csr", "csc", "coo", "lil", "dok", "bsr")
COMBOS = {
    "both": dict(pre=True, post=True),
    "pre": dict(pre=True, post=False),
    "post": dict(pre=False, post=True),
}


# =============================================================================================== submat_at_ind


@case("local.submat_at_ind.csr_weighted")
def _():
    # the order of ind is kept, so this is not the sorted sub-matrix
    return L.submat_at_ind(weighted(20, 0.25, seed=21, integers=False), np.array([14, 3, 9, 0, 11]))


@case("local.submat_at_ind.sparse_formats")
def _():
    # every sparse format comes out as csr_matrix
    adj = digraph(20, 0.3, seed=22, reciprocal=0.3)
    ind = np.array([7, 2, 19, 4, 11])
    return {fmt: L.submat_at_ind(adj.asformat(fmt), ind) for fmt in SPARSE_FORMATS}


@case("local.submat_at_ind.sparse_arrays")
def _():
    adj = digraph(20, 0.3, seed=22, reciprocal=0.3)
    ind = np.array([7, 2, 19, 4, 11])
    return {
        "csr_array": L.submat_at_ind(sp.csr_array(adj), ind),
        "csc_array": L.submat_at_ind(sp.csc_array(adj), ind),
        "coo_array": L.submat_at_ind(sp.coo_array(adj), ind),
    }


@case("local.submat_at_ind.keyword_arguments")
def _():
    return L.submat_at_ind(M=digraph(12, 0.4, seed=25, reciprocal=0.3), ind=np.array([4, 1, 9]))


@case("local.submat_at_ind.dense")
def _():
    dense = weighted(20, 0.3, seed=23).toarray()
    ind = np.array([7, 2, 19, 4, 11])
    return {
        "ndarray": L.submat_at_ind(dense, ind),
        "matrix": L.submat_at_ind(np.matrix(dense), ind),
    }


@case("local.submat_at_ind.dtypes")
def _():
    adj = weighted(20, 0.3, seed=24)
    ind = np.array([5, 17, 3, 8])
    out = {}
    for dt in (bool, np.int8, np.int64, np.float32, np.float64):
        name = np.dtype(dt).name
        out[f"sparse_{name}"] = L.submat_at_ind(adj.astype(dt), ind)
        out[f"dense_{name}"] = L.submat_at_ind(adj.toarray().astype(dt), ind)
    return out


@case("local.submat_at_ind.index_types")
def _():
    adj = digraph(12, 0.4, seed=25, reciprocal=0.3)
    dense = adj.toarray()
    mask = np.arange(12) % 3 == 0
    out = {}
    for name, ind in {
        "list": [4, 1, 9],
        "mask": mask,
        "range": np.arange(2, 7),
        "negative": np.array([-1, 0, -3]),
        "int32": np.array([4, 1, 9], dtype=np.int32),
        "uint8": np.array([4, 1, 9], dtype=np.uint8),
    }.items():
        out[f"sparse_{name}"] = L.submat_at_ind(adj, ind)
        out[f"dense_{name}"] = L.submat_at_ind(dense, ind)
    return out


@case("local.submat_at_ind.repeated_and_few_indices")
def _():
    adj = digraph(12, 0.4, seed=25, reciprocal=0.3)
    dense = adj.toarray()
    out = {}
    for name, ind in {
        "repeated": np.array([4, 4, 1, 4]),
        "single": np.array([4]),
        "empty": np.array([], dtype=int),
        "all_reversed": np.arange(12)[::-1].copy(),
    }.items():
        out[f"sparse_{name}"] = L.submat_at_ind(adj, ind)
        out[f"dense_{name}"] = L.submat_at_ind(dense, ind)
    return out


@case("local.submat_at_ind.non_square")
def _():
    values = np.arange(1, 13).reshape(3, 4)
    ind = np.array([2, 0])
    return {
        "sparse": L.submat_at_ind(sp.csr_matrix(values), ind),
        "dense": L.submat_at_ind(values, ind),
    }


@case("local.submat_at_ind.result_is_a_copy")
def _():
    # changing the sub-matrix must not change the matrix it was cut from (even for a contiguous index)
    sparse = digraph(12, 0.4, seed=25, reciprocal=0.3)
    dense = sparse.toarray()
    ind = np.arange(2, 8)
    sub_sparse, sub_dense = L.submat_at_ind(sparse, ind), L.submat_at_ind(dense, ind)
    sub_sparse.data[:] = False
    sub_dense[...] = True
    return sub_sparse, sub_dense, sparse, dense


@case("local.submat_at_ind.tutorial")
def _():
    adj, _nodes = tutorial_connectome(400)
    ind = np.random.RandomState(29).permutation(400)[:70]
    return {
        "sparse": L.submat_at_ind(adj, ind),
        "dense": L.submat_at_ind(adj[:150, :150].toarray(), ind[ind < 150]),
    }


@case("local.submat_at_ind.out_of_range_sparse")
def _():
    return L.submat_at_ind(digraph(10, 0.3, seed=26), np.array([0, 10]))


@case("local.submat_at_ind.out_of_range_dense")
def _():
    return L.submat_at_ind(digraph(10, 0.3, seed=26).toarray(), np.array([0, 10]))


# =============================================================================================== neighborhood


@case("local.neighborhood.default")
def _():
    return L.neighborhood(_mixed(), 2)


@case("local.neighborhood.options")
def _():
    adj = _mixed()
    out = {}
    for name, kw in COMBOS.items():
        for center in (True, False):
            for nbrs in (True, False):
                out[f"{name}_center{int(center)}_nbrs{int(nbrs)}"] = L.neighborhood(
                    adj, 7, include_center=center, return_neighbors=nbrs, **kw
                )
    return out


@case("local.neighborhood.keyword_arguments")
def _():
    return L.neighborhood(
        adj=_mixed(), v=7, pre=False, post=True, include_center=False, return_neighbors=True
    )


@case("local.neighborhood.formats")
def _():
    adj = _mixed()
    inputs = {fmt: adj.asformat(fmt) for fmt in ("csr", "csc", "coo", "lil", "dok")}
    inputs.update(ndarray=adj.toarray(), matrix=np.matrix(adj.toarray()))
    return {name: L.neighborhood(m, 7, return_neighbors=True) for name, m in inputs.items()}


@case("local.neighborhood.weighted")
def _():
    adj = weighted(16, 0.25, seed=27, integers=False)
    return {
        "sparse": L.neighborhood(adj, 6, return_neighbors=True),
        "dense": L.neighborhood(adj.toarray(), 6, return_neighbors=True),
        "pre_no_center": L.neighborhood(
            adj, 6, pre=True, post=False, include_center=False, return_neighbors=True
        ),
    }


@case("local.neighborhood.isolated_center")
def _():
    # without the center nothing is left: a 0 x 0 matrix
    adj = _mixed()
    return {
        "with_center": L.neighborhood(adj, 5, return_neighbors=True),
        "without_center": L.neighborhood(adj, 5, include_center=False, return_neighbors=True),
        "dense_with_center": L.neighborhood(adj.toarray(), 5, return_neighbors=True),
        "dense_without_center": L.neighborhood(
            adj.toarray(), 5, include_center=False, return_neighbors=True
        ),
    }


@case("local.neighborhood.source_and_sink")
def _():
    adj = _mixed()
    return {
        "source_pre": L.neighborhood(adj, 12, pre=True, post=False, return_neighbors=True),
        "source_post": L.neighborhood(adj, 12, pre=False, post=True, return_neighbors=True),
        "sink_pre": L.neighborhood(adj, 13, pre=True, post=False, return_neighbors=True),
        "sink_post": L.neighborhood(adj, 13, pre=False, post=True, return_neighbors=True),
    }


@case("local.neighborhood.self_loop")
def _():
    # the center is its own neighbour: it shows up twice in the neighbourhood when the center is included
    dense = _mixed().toarray()
    dense[3, 3] = True
    adj = sp.csr_matrix(dense)
    return {
        "with_center": L.neighborhood(adj, 3, return_neighbors=True),
        "without_center": L.neighborhood(adj, 3, include_center=False, return_neighbors=True),
    }


@case("local.neighborhood.center_types")
def _():
    adj = _mixed()
    return {
        "int32": L.neighborhood(adj, np.int32(2), return_neighbors=True),
        "int64": L.neighborhood(adj, np.int64(2), return_neighbors=True),
        "first_node": L.neighborhood(adj, 0, return_neighbors=True),
        "last_node": L.neighborhood(adj, 15, return_neighbors=True),
    }


@case("local.neighborhood.tutorial_hubs")
def _():
    adj, _nodes = tutorial_connectome(500)
    degree = np.asarray(adj.sum(axis=0)).ravel() + np.asarray(adj.sum(axis=1)).ravel()
    return {
        int(v): L.neighborhood(adj, int(v), return_neighbors=True)
        for v in np.argsort(-degree, kind="stable")[:4]
    }


@case("local.neighborhood.center_outside_matrix")
def _():
    return L.neighborhood(_mixed(), 100)


@case("local.neighborhood.edgeless_graph")
def _():
    # defect: neighborhood_indices(all_nodes=False) fails on a matrix without edges
    return L.neighborhood(sp.csr_matrix((6, 6), dtype=bool), 2)


@case("local.neighborhood.not_square")
def _():
    return L.neighborhood(sp.csr_matrix(np.ones((3, 4))), 1)


@case("local.neighborhood.pre_and_post_false")
def _():
    return L.neighborhood(_mixed(), 2, pre=False, post=False)


# =============================================================================================== neighborhood_of_set_indices


@case("local.neighborhood_of_set_indices.default")
def _():
    return L.neighborhood_of_set_indices(_mixed(), np.array([4, 7, 0]))


@case("local.neighborhood_of_set_indices.pre_post")
def _():
    adj = _mixed()
    node_set = np.array([4, 7, 0])
    return {name: L.neighborhood_of_set_indices(adj, node_set, **kw) for name, kw in COMBOS.items()}


@case("local.neighborhood_of_set_indices.keyword_arguments")
def _():
    return L.neighborhood_of_set_indices(
        M=_mixed(), node_set=np.array([4, 7, 0]), pre=False, post=True
    )


@case("local.neighborhood_of_set_indices.positional_pre_post")
def _():
    return L.neighborhood_of_set_indices(_mixed(), np.array([4, 7, 0]), False, True)


@case("local.neighborhood_of_set_indices.formats")
def _():
    adj = _mixed()
    inputs = {fmt: adj.asformat(fmt) for fmt in ("csr", "csc", "coo", "lil")}
    inputs.update(ndarray=adj.toarray(), matrix=np.matrix(adj.toarray()))
    return {
        name: L.neighborhood_of_set_indices(m, np.array([9, 1, 6])) for name, m in inputs.items()
    }


@case("local.neighborhood_of_set_indices.sets")
def _():
    # unsorted set, members that are neighbours of each other (they are part of the answer), the whole graph, a source
    # and a sink, and sets with nodes that have no neighbour at all
    adj = _mixed()
    return {
        "unsorted": L.neighborhood_of_set_indices(adj, np.array([14, 3, 10, 1])),
        "adjacent_members": L.neighborhood_of_set_indices(adj, np.array([0, 6])),
        "single_node": L.neighborhood_of_set_indices(adj, np.array([2])),
        "whole_graph": L.neighborhood_of_set_indices(adj, np.arange(16)),
        "source_pre": L.neighborhood_of_set_indices(adj, np.array([12]), pre=True, post=False),
        "isolated_only": L.neighborhood_of_set_indices(adj, np.array([5, 11])),
        "isolated_and_connected": L.neighborhood_of_set_indices(
            adj, np.array([5, 12, 11]), pre=False, post=True
        ),
        "outside_matrix": L.neighborhood_of_set_indices(adj, np.array([30, 3])),
    }


def _two_components():
    """Two disconnected blocks of 8 and 7 nodes (reciprocal pairs in both)."""
    return sp.block_diag(
        [digraph(8, 0.3, seed=61, reciprocal=0.4), digraph(7, 0.35, seed=62, reciprocal=0.4)],
        format="csr",
    )


@case("local.neighborhood_of_set_indices.two_components")
def _():
    adj = _two_components()
    return {
        "first": L.neighborhood_of_set_indices(adj, np.array([1, 3])),
        "second": L.neighborhood_of_set_indices(adj, np.array([9, 13])),
        "across": L.neighborhood_of_set_indices(adj, np.array([2, 11])),
        "across_pre": L.neighborhood_of_set_indices(adj, np.array([2, 11]), pre=True, post=False),
    }


@case("local.neighborhood_of_set_indices.tutorial")
def _():
    adj, _nodes = tutorial_connectome(500)
    node_set = np.random.RandomState(31).permutation(500)[:25]
    return {name: L.neighborhood_of_set_indices(adj, node_set, **kw) for name, kw in COMBOS.items()}


@case("local.neighborhood_of_set_indices.pre_and_post_false")
def _():
    return L.neighborhood_of_set_indices(_mixed(), np.array([1, 2]), pre=False, post=False)


@case("local.neighborhood_of_set_indices.empty_set")
def _():
    return L.neighborhood_of_set_indices(_mixed(), np.array([], dtype=int))


@case("local.neighborhood_of_set_indices.repeated_nodes")
def _():
    # inherits the duplicate-centers failure of neighborhood_indices
    return L.neighborhood_of_set_indices(_mixed(), np.array([7, 4, 4]))


@case("local.neighborhood_of_set_indices.set_as_list")
def _():
    return L.neighborhood_of_set_indices(_mixed(), [4, 7])


@case("local.neighborhood_of_set_indices.not_square")
def _():
    return L.neighborhood_of_set_indices(sp.csr_matrix(np.ones((3, 4))), np.array([0]))


# =============================================================================================== neighborhood_of_set


@case("local.neighborhood_of_set.without_centers_formats")
def _():
    # csr input comes back as csc and csc as csr; coo is converted to csr first; dense stays dense
    adj = _mixed()
    node_set = np.array([4, 7, 0])
    inputs = {fmt: adj.asformat(fmt) for fmt in ("csr", "csc", "coo")}
    inputs.update(ndarray=adj.toarray(), matrix=np.matrix(adj.toarray()))
    return {
        name: L.neighborhood_of_set(m, node_set, include_centers=False, return_neighbors=True)
        for name, m in inputs.items()
    }


@case("local.neighborhood_of_set.keyword_arguments")
def _():
    return L.neighborhood_of_set(
        M=_mixed(),
        node_set=np.array([4, 7, 0]),
        pre=True,
        post=False,
        include_centers=False,
        return_neighbors=True,
    )


@case("local.neighborhood_of_set.without_centers_pre_post")
def _():
    adj = _mixed()
    node_set = np.array([9, 1, 6])
    out = {}
    for name, kw in COMBOS.items():
        out[f"csr_{name}"] = L.neighborhood_of_set(adj, node_set, include_centers=False, **kw)
        out[f"csc_{name}"] = L.neighborhood_of_set(
            adj.tocsc(), node_set, include_centers=False, **kw
        )
        out[f"dense_{name}"] = L.neighborhood_of_set(
            adj.toarray(), node_set, include_centers=False, **kw
        )
    return out


@case("local.neighborhood_of_set.without_centers_weighted")
def _():
    adj = weighted(16, 0.25, seed=28, integers=False)
    return L.neighborhood_of_set(
        adj, np.array([2, 11]), include_centers=False, return_neighbors=True
    )


@case("local.neighborhood_of_set.without_centers_isolated_set")
def _():
    adj = _mixed()
    return {
        "csr": L.neighborhood_of_set(
            adj, np.array([5, 11]), include_centers=False, return_neighbors=True
        ),
        "dense": L.neighborhood_of_set(
            adj.toarray(), np.array([5, 11]), include_centers=False, return_neighbors=True
        ),
    }


@case("local.neighborhood_of_set.without_centers_tutorial")
def _():
    adj, _nodes = tutorial_connectome(500)
    node_set = np.random.RandomState(32).permutation(500)[:12]
    return L.neighborhood_of_set(adj, node_set, include_centers=False, return_neighbors=True)


@case("local.neighborhood_of_set.two_components")
def _():
    adj = _two_components()
    return L.neighborhood_of_set(
        adj, np.array([2, 11]), include_centers=False, return_neighbors=True
    )


@case("local.neighborhood_of_set.default_include_centers")
def _():
    # defect: include_centers=True (the default) uses the undefined name ``nodesA``
    return L.neighborhood_of_set(_mixed(), np.array([4, 7, 0]))


@case("local.neighborhood_of_set.include_centers_return_neighbors")
def _():
    return L.neighborhood_of_set(
        _mixed().toarray(),
        np.array([4, 7, 0]),
        pre=True,
        post=False,
        include_centers=True,
        return_neighbors=True,
    )


@case("local.neighborhood_of_set.assertion_before_name_error")
def _():
    # the neighbours are computed (and checked) before the broken include_centers branch
    return L.neighborhood_of_set(_mixed(), np.array([4, 7, 0]), pre=False, post=False)


@case("local.neighborhood_of_set.pre_and_post_false")
def _():
    return L.neighborhood_of_set(
        _mixed(), np.array([4, 7, 0]), pre=False, post=False, include_centers=False
    )


@case("local.neighborhood_of_set.lil_input")
def _():
    # defect: formats other than ndarray/csr/csc/coo leave ``nbd`` unassigned
    return L.neighborhood_of_set(_mixed("lil"), np.array([4, 7, 0]), include_centers=False)


@case("local.neighborhood_of_set.csr_array_input")
def _():
    # sparse arrays are not csr_matrix instances
    return L.neighborhood_of_set(sp.csr_array(_mixed()), np.array([4, 7, 0]), include_centers=False)


@case("local.neighborhood_of_set.empty_set")
def _():
    return L.neighborhood_of_set(_mixed(), np.array([], dtype=int), include_centers=False)
