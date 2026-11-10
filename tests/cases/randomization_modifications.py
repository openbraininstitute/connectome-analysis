"""Graph modifications: add_rc_connections_skeleta, add_rc_connections, add_connections.

All three draw from ``np.random.default_rng(seed)`` (add_connections, sparse mode, also through
``scipy.sparse.random(random_state=generator)``), so their seeded cases are tagged ``generator-stream``.
add_rc_connections_skeleta computes the skeleta with flagser through ``connalysis.network.topology.get_k_skeleta_graph``
and prints how many connections it adds per dimension (the cases do not capture that output).
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import tutorial_connectome, weighted

from cases.randomization_common import (
    GEN,
    R,
    digraph,
    is_superset,
    reciprocal_entries,
    shape_info,
    skeleta_graph,
)


def _skeleta(adj, **kw):
    from connalysis.network import topology

    return topology.get_k_skeleta_graph(adj, max_simplices=True, threads=1, **kw)


def _skel(adj, factors=2, **kw):
    kw.setdefault("threads", 1)
    kw.setdefault("seed", 1)
    return R.add_rc_connections_skeleta(adj, factors, **kw)


# ----------------------------------------------------------------------------- add_rc_connections_skeleta


@case("randomization.add_rc_connections_skeleta.factor_int", tags=GEN)
def _():
    # every dimension: the number of reciprocal pairs of its skeleton is multiplied by 2
    return _skel(skeleta_graph(), 2)


@case("randomization.add_rc_connections_skeleta.factors_dict", tags=GEN)
def _():
    return _skel(skeleta_graph(), {1: 2, 2: 1, 3: 2, 4: 1})


@case("randomization.add_rc_connections_skeleta.factors_dict_with_extra_keys", tags=GEN)
def _():
    # keys for dimensions without simplices are ignored
    return _skel(skeleta_graph(), {1: 2, 2: 2, 3: 2, 4: 2, 5: 7, 9: 3})


@case("randomization.add_rc_connections_skeleta.dimensions_as_list", tags=GEN)
def _():
    return _skel(skeleta_graph(), 2, dimensions=[2, 3])


@case("randomization.add_rc_connections_skeleta.dimensions_not_in_the_graph", tags=GEN)
def _():
    # dimension 9 has no simplices and is dropped (intersection with the dimensions of the skeleta)
    return _skel(skeleta_graph(), 2, dimensions=[2, 3, 9])


@case("randomization.add_rc_connections_skeleta.dimensions_as_array_fails_while_computing_skeleta")
def _():
    # docstring: "dimensions: array"; get_k_skeleta_graph compares the array with None (`dimensions == None`)
    return _skel(skeleta_graph(), 2, dimensions=np.array([2, 3]))


@case(
    "randomization.add_rc_connections_skeleta.dimensions_as_array_with_precomputed_skeleta",
    tags=GEN,
)
def _():
    adj = skeleta_graph()
    return _skel(adj, 2, dimensions=np.array([2, 3]), skeleta=_skeleta(adj))


@case("randomization.add_rc_connections_skeleta.factor_one_changes_nothing")
def _():
    adj = skeleta_graph()
    out = _skel(adj, 1)
    return {"result": out, "unchanged": bool((out != adj).nnz == 0)}


@case("randomization.add_rc_connections_skeleta.precomputed_skeleta_give_the_same_result", tags=GEN)
def _():
    adj = skeleta_graph()
    skeleta = _skeleta(adj)
    return {"precomputed": _skel(adj, 2, skeleta=skeleta), "computed": _skel(adj, 2)}


@case("randomization.add_rc_connections_skeleta.restricted_precomputed_skeleta", tags=GEN)
def _():
    # only the skeleta that are passed are used, whatever the graph contains
    adj = skeleta_graph()
    skeleta = _skeleta(adj)
    return _skel(
        adj,
        2,
        skeleta={"dimension_2": skeleta["dimension_2"], "dimension_3": skeleta["dimension_3"]},
    )


@case("randomization.add_rc_connections_skeleta.return_skeleta", tags=GEN)
def _():
    adj = skeleta_graph()
    out, skeleta = _skel(adj, 2, return_skeleta=True)
    return {"graph": out, "skeleta": skeleta, "keys": list(skeleta)}


@case("randomization.add_rc_connections_skeleta.return_skeleta_restricted_dimensions", tags=GEN)
def _():
    # the skeleta are computed for the requested dimensions only
    out, skeleta = _skel(skeleta_graph(), 2, dimensions=[1, 3], return_skeleta=True)
    return {"graph": out, "keys": list(skeleta)}


@case("randomization.add_rc_connections_skeleta.keeps_old_edges_and_adds_reverses", tags=GEN)
def _():
    adj = skeleta_graph()
    out = _skel(adj, 2)
    return {
        "edges_added": int(out.nnz - adj.nnz),
        "superset": is_superset(out, adj),
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
        "cls": type(out).__name__,
        "dtype": str(out.dtype),
    }


@case("randomization.add_rc_connections_skeleta.seed_values", tags=GEN)
def _():
    adj = skeleta_graph()
    return {s: _skel(adj, 2, seed=s) for s in (0, 1, 2)}


@case("randomization.add_rc_connections_skeleta.default_seed_is_zero", tags=GEN)
def _():
    adj = skeleta_graph()
    return {
        "default": R.add_rc_connections_skeleta(adj, 2, threads=1),
        "zero": _skel(adj, 2, seed=0),
    }


@case("randomization.add_rc_connections_skeleta.default_threads", tags=GEN)
def _():
    return R.add_rc_connections_skeleta(skeleta_graph(), 2, seed=1)


@case("randomization.add_rc_connections_skeleta.integer_weights_add_up", tags=GEN)
def _():
    # a reverse edge that is added in several dimensions (or already exists) adds to the weight
    adj = skeleta_graph().astype(int)
    out = _skel(adj, 2)
    return {"result": out, "weights": np.unique(out.data, return_counts=True)}


@case("randomization.add_rc_connections_skeleta.float_weights", tags=GEN)
def _():
    return _skel(skeleta_graph().astype(float), 2)


@case("randomization.add_rc_connections_skeleta.input_formats", tags=GEN)
def _():
    adj = skeleta_graph()
    return {fmt: _skel(adj.asformat(fmt), 2) for fmt in ("csr", "csc", "coo", "lil")}


@case("randomization.add_rc_connections_skeleta.dense_input_rejected")
def _():
    return _skel(skeleta_graph().toarray(), 2)


@case("randomization.add_rc_connections_skeleta.factor_too_large")
def _():
    # more reciprocal pairs requested than there are one-way edges in a skeleton
    return _skel(skeleta_graph(), 4)


@case("randomization.add_rc_connections_skeleta.factor_zero")
def _():
    return _skel(skeleta_graph(), 0)


@case("randomization.add_rc_connections_skeleta.factor_float_rejected")
def _():
    return _skel(skeleta_graph(), 2.0)


@case("randomization.add_rc_connections_skeleta.factor_numpy_int_rejected")
def _():
    return _skel(skeleta_graph(), np.int64(2))


@case("randomization.add_rc_connections_skeleta.factor_none_rejected")
def _():
    return _skel(skeleta_graph(), None)


@case("randomization.add_rc_connections_skeleta.factors_dict_missing_dimension")
def _():
    return _skel(skeleta_graph(), {1: 2, 2: 2})


@case("randomization.add_rc_connections_skeleta.graph_without_reciprocal_pairs_in_skeleta")
def _():
    # nothing to multiply: the graph comes back unchanged
    adj = digraph(30, 0.06, seed=3)
    out = _skel(adj, 3)
    return {"result": out, "unchanged": bool((out != adj).nnz == 0)}


@case("randomization.add_rc_connections_skeleta.single_edge_and_path")
def _():
    one = sp.csr_matrix(([True], ([0], [1])), shape=(4, 4))
    path = sp.csr_matrix(([True, True], ([0, 1], [1, 2])), shape=(4, 4))
    return {"single_edge": _skel(one, 2), "path": _skel(path, 2)}


@case("randomization.add_rc_connections_skeleta.empty_graph")
def _():
    # flagser finds no simplices at all and edge_participation fails on the empty list
    return _skel(sp.csr_matrix((6, 6), dtype=bool), 2)


@case("randomization.add_rc_connections_skeleta.unseeded_properties")
def _():
    adj = skeleta_graph()
    out = _skel(adj, 2, seed=None)
    return {
        **shape_info(out),
        "superset": is_superset(out, adj),
        "at_least_as_many_edges": bool(out.nnz >= adj.nnz),
    }


@case("randomization.add_rc_connections_skeleta.real_data", tags=GEN)
def _():
    adj, _ = tutorial_connectome(300)
    return _skel(adj, 2)


@case("randomization.add_rc_connections_skeleta.real_data_factors_and_dimensions", tags=GEN)
def _():
    adj, _ = tutorial_connectome(300)
    out, skeleta = _skel(adj, {1: 2, 2: 2, 3: 1, 4: 3}, dimensions=[2, 3, 4], return_skeleta=True)
    return {"graph": out, "keys": list(skeleta)}


# ----------------------------------------------------------------------------- add_rc_connections


def _rc_input():
    return digraph(20, 0.2, seed=1)


@case("randomization.add_rc_connections.basic", tags=GEN)
def _():
    return R.add_rc_connections(_rc_input(), 5, seed=1)


@case("randomization.add_rc_connections.adds_exactly_n_reciprocal_pairs")
def _():
    adj = _rc_input()
    out = R.add_rc_connections(adj, 5, seed=1)
    return {
        "edges_added": int(out.nnz - adj.nnz),
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
        "superset": is_superset(out, adj),
        "cls": type(out).__name__,
        "dtype": str(out.dtype),
    }


@case("randomization.add_rc_connections.default_seed_is_zero", tags=GEN)
def _():
    adj = _rc_input()
    return {
        "default": R.add_rc_connections(adj, 5),
        "zero": R.add_rc_connections(adj, 5, seed=0),
        "one": R.add_rc_connections(adj, 5, seed=1),
    }


@case("randomization.add_rc_connections.zero_connections")
def _():
    adj = _rc_input()
    out = R.add_rc_connections(adj, 0, seed=1)
    return {"result": out, "unchanged": bool((out != adj).nnz == 0)}


@case("randomization.add_rc_connections.input_formats", tags=GEN)
def _():
    adj = _rc_input()
    return {
        fmt: R.add_rc_connections(adj.asformat(fmt), 5, seed=1)
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.add_rc_connections.weighted_input_gives_bool_output", tags=GEN)
def _():
    return R.add_rc_connections(weighted(20, 0.2, seed=1), 5, seed=1)


@case("randomization.add_rc_connections.all_one_way_edges_become_reciprocal", tags=GEN)
def _():
    # a transitive tournament on 5 nodes has 10 one-way edges
    adj = sp.csr_matrix(np.triu(np.ones((5, 5), dtype=bool), 1))
    return R.add_rc_connections(adj, 10, seed=1)


@case("randomization.add_rc_connections.more_than_one_way_edges")
def _():
    return R.add_rc_connections(_rc_input(), 500, seed=1)


@case("randomization.add_rc_connections.negative_number")
def _():
    return R.add_rc_connections(_rc_input(), -1, seed=1)


@case("randomization.add_rc_connections.dense_bool_input")
def _():
    return R.add_rc_connections(_rc_input().toarray(), 5, seed=1)


@case("randomization.add_rc_connections.dense_float_input")
def _():
    return R.add_rc_connections(_rc_input().toarray().astype(float), 5, seed=1)


@case("randomization.add_rc_connections.empty_graph")
def _():
    return R.add_rc_connections(sp.csr_matrix((6, 6), dtype=bool), 0, seed=1)


@case("randomization.add_rc_connections.complete_graph")
def _():
    full = sp.csr_matrix(np.ones((4, 4)) - np.eye(4))
    return R.add_rc_connections(full, 0, seed=1)


@case("randomization.add_rc_connections.non_square")
def _():
    return R.add_rc_connections(sp.csr_matrix(np.ones((3, 5))), 1, seed=1)


@case("randomization.add_rc_connections.diagonal_entries_are_kept")
def _():
    # rc_submatrix warns about the diagonal; the loops stay and are never made reciprocal
    adj = sp.csr_matrix(np.eye(4) + np.diag(np.ones(3), 1))
    return R.add_rc_connections(adj, 1, seed=1)


@case("randomization.add_rc_connections.unseeded_properties")
def _():
    adj = _rc_input()
    out = R.add_rc_connections(adj, 5, seed=None)
    return {
        **shape_info(out),
        "edges_added": int(out.nnz - adj.nnz),
        "superset": is_superset(out, adj),
        "reciprocal_entries_added": reciprocal_entries(out) - reciprocal_entries(adj),
    }


@case("randomization.add_rc_connections.real_data", tags=GEN)
def _():
    adj, _ = tutorial_connectome(300)
    return R.add_rc_connections(adj, 25, seed=2)


# ----------------------------------------------------------------------------- add_connections


def _ac_input():
    return digraph(20, 0.1, seed=1)


@case("randomization.add_connections.sparse_mode", tags=GEN)
def _():
    return R.add_connections(_ac_input(), 10, seed=0)


@case("randomization.add_connections.sparse_mode_is_the_default", tags=GEN)
def _():
    adj = _ac_input()
    return {
        "default": R.add_connections(adj, 10, seed=0),
        "explicit": R.add_connections(adj, 10, seed=0, sparse_mode=True),
    }


@case("randomization.add_connections.default_seed_is_zero", tags=GEN)
def _():
    adj = _ac_input()
    return {
        "default": R.add_connections(adj, 10),
        "zero": R.add_connections(adj, 10, seed=0),
        "one": R.add_connections(adj, 10, seed=1),
    }


@case("randomization.add_connections.dense_mode", tags=GEN)
def _():
    return R.add_connections(_ac_input(), 10, seed=0, sparse_mode=False)


@case("randomization.add_connections.adds_exactly_nc_edges", tags=GEN)
def _():
    adj = _ac_input()
    out = {mode: R.add_connections(adj, 25, seed=3, sparse_mode=mode) for mode in (True, False)}
    return {
        str(mode): {
            "cls": type(m).__name__,
            "dtype": str(m.dtype),
            "edges": int(m.sum()),
            "diagonal": int(m.diagonal().sum()),
            "superset": is_superset(m, adj),
        }
        for mode, m in out.items()
    }


@case("randomization.add_connections.sparse_mode_seed_values", tags=GEN)
def _():
    adj = _ac_input()
    return {s: R.add_connections(adj, 40, seed=s) for s in (0, 1, 2)}


@case("randomization.add_connections.dense_mode_seed_values", tags=GEN)
def _():
    adj = _ac_input()
    return {s: R.add_connections(adj, 40, seed=s, sparse_mode=False) for s in (0, 1, 2)}


@case("randomization.add_connections.zero_connections")
def _():
    adj = _ac_input()
    return {
        "sparse": R.add_connections(adj, 0, seed=0),
        "dense": R.add_connections(adj, 0, seed=0, sparse_mode=False),
    }


@case("randomization.add_connections.sparse_mode_stops_after_max_iter", tags=GEN)
def _():
    # 300 more edges on 20 nodes: the iterations keep drawing the same positions (the generator is recreated with the
    # same seed in every iteration) and stop after max_iter + 1 passes with fewer edges than asked for
    adj = _ac_input()
    out = R.add_connections(adj, 300, seed=0)
    return {"result": out, "edges_asked_for": int(adj.sum()) + 300, "edges_got": int(out.sum())}


@case("randomization.add_connections.max_iter_values", tags=GEN)
def _():
    adj = _ac_input()
    return {k: R.add_connections(adj, 300, seed=0, max_iter=k) for k in (0, 1, 2, 5)}


@case("randomization.add_connections.max_iter_limits_the_passes_even_for_a_small_request", tags=GEN)
def _():
    # max_iter=0 allows two passes (the check is `k > max_iter` before each pass): 48 of the 51 edges, 51 with max_iter=30
    adj = _ac_input()
    return {k: R.add_connections(adj, 10, seed=0, max_iter=k) for k in (0, 1, 30)}


@case("randomization.add_connections.full_graph_cannot_grow_in_sparse_mode")
def _():
    full = sp.csr_matrix(np.ones((6, 6)) - np.eye(6))
    out = R.add_connections(full, 3, seed=0)
    return {"result": out, "unchanged": bool((out != full.astype(bool)).nnz == 0)}


@case("randomization.add_connections.too_many_connections_sparse_mode")
def _():
    # the density of the random matrix would exceed 1
    return R.add_connections(_ac_input(), 1000, seed=0)


@case("randomization.add_connections.too_many_connections_dense_mode")
def _():
    return R.add_connections(_ac_input(), 1000, seed=0, sparse_mode=False)


@case("randomization.add_connections.input_formats", tags=GEN)
def _():
    adj = _ac_input()
    return {
        fmt: R.add_connections(adj.asformat(fmt), 10, seed=0)
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.add_connections.input_formats_dense_mode", tags=GEN)
def _():
    adj = _ac_input()
    return {
        fmt: R.add_connections(adj.asformat(fmt), 10, seed=0, sparse_mode=False)
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.add_connections.weighted_input_becomes_bool", tags=GEN)
def _():
    adj = weighted(20, 0.1, seed=1)
    return {
        "sparse": R.add_connections(adj, 10, seed=0),
        "dense": R.add_connections(adj, 10, seed=0, sparse_mode=False),
    }


@case("randomization.add_connections.dense_array_input_sparse_mode")
def _():
    return R.add_connections(_ac_input().toarray(), 10, seed=0)


@case("randomization.add_connections.dense_array_input_dense_mode", tags=GEN)
def _():
    return R.add_connections(_ac_input().toarray(), 10, seed=0, sparse_mode=False)


@case("randomization.add_connections.dense_array_is_not_modified", tags=GEN)
def _():
    arr = _ac_input().toarray()
    before = arr.copy()
    out = R.add_connections(arr, 10, seed=0, sparse_mode=False)
    return {"result": out, "input_unchanged": bool((arr == before).all())}


@case("randomization.add_connections.diagonal_in_input", tags=GEN)
def _():
    # existing loops stay; new connections never go on the diagonal
    adj = sp.csr_matrix(np.eye(5))
    return {
        "sparse": R.add_connections(adj, 3, seed=0),
        "dense": R.add_connections(adj, 3, seed=0, sparse_mode=False),
    }


@case("randomization.add_connections.empty_graph", tags=GEN)
def _():
    adj = sp.csr_matrix((10, 10), dtype=bool)
    return {
        "sparse": R.add_connections(adj, 5, seed=0),
        "dense": R.add_connections(adj, 5, seed=0, sparse_mode=False),
    }


@case("randomization.add_connections.non_square", tags=GEN)
def _():
    # N is the number of rows but the random matrix has the shape of adj
    adj = sp.csr_matrix(np.zeros((4, 7), dtype=bool))
    return {
        "sparse": R.add_connections(adj, 3, seed=0),
        "dense": R.add_connections(adj, 3, seed=0, sparse_mode=False),
    }


@case("randomization.add_connections.unseeded_properties")
def _():
    adj = _ac_input()
    out_sparse = R.add_connections(adj, 10, seed=None)
    out_dense = R.add_connections(adj, 10, seed=None, sparse_mode=False)
    return {
        name: {
            "cls": type(m).__name__,
            "dtype": str(m.dtype),
            "edges": int(m.sum()),
            "diagonal": int(m.diagonal().sum()),
            "superset": is_superset(m, adj),
        }
        for name, m in [("sparse", out_sparse), ("dense", out_dense)]
    }


@case("randomization.add_connections.real_data", tags=GEN)
def _():
    adj, _ = tutorial_connectome(120)
    return {
        "sparse": R.add_connections(adj, 200, seed=1),
        "dense": R.add_connections(adj, 200, seed=1, sparse_mode=False),
    }
