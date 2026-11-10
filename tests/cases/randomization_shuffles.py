"""Shuffle models: ER_shuffle (legacy global numpy RNG), configuration_model, adjusted_ER, underlying_model, bishuffled_model.

ER_shuffle seeds and draws from the legacy ``np.random`` state (frozen stream).  The other four draw from
``np.random.default_rng(seed)``; their seeded cases are tagged ``generator-stream``.  adjusted_ER also calls
ER_shuffle with the same seed, so it re-seeds the global state as a side effect.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import tutorial_connectome, weighted

from cases.randomization_common import (
    GEN,
    R,
    digraph,
    edge_info,
    reciprocal_entries,
    reciprocal_graph,
    undirected_edges,
)


def _er_shuffle(adj, **kw):
    return R.ER_shuffle(adj, **kw)


def _full(n):
    return sp.csr_matrix(np.ones((n, n)) - np.eye(n))


# ----------------------------------------------------------------------------- ER_shuffle, sparse


@case("randomization.ER_shuffle.sparse_bool")
def _():
    return _er_shuffle(digraph(40, 0.1, seed=5), seed=3)


@case("randomization.ER_shuffle.sparse_integer_weights")
def _():
    # the weights are reassigned to the new positions in stored order
    return _er_shuffle(weighted(12, 0.3, seed=2), seed=5)


@case("randomization.ER_shuffle.sparse_real_weights")
def _():
    return _er_shuffle(weighted(12, 0.3, seed=2, integers=False), seed=5)


@case("randomization.ER_shuffle.sparse_input_formats")
def _():
    # csc and coo keep their own stored order, so the weights land on different positions than for csr
    adj = weighted(12, 0.3, seed=2)
    return {fmt: _er_shuffle(adj.asformat(fmt), seed=5) for fmt in ("csr", "csc", "coo")}


@case("randomization.ER_shuffle.sparse_int_and_float_dtypes")
def _():
    adj = digraph(15, 0.3, seed=6)
    return {
        name: _er_shuffle(adj.astype(dt), seed=2)
        for name, dt in [
            ("bool", bool),
            ("int8", np.int8),
            ("int64", np.int64),
            ("float32", np.float32),
            ("float64", float),
        ]
    }


@case("randomization.ER_shuffle.sparse_lil_rejected")
def _():
    return _er_shuffle(weighted(12, 0.3, seed=2, fmt="lil"), seed=5)


@case("randomization.ER_shuffle.sparse_dok_rejected")
def _():
    return _er_shuffle(weighted(12, 0.3, seed=2, fmt="dok"), seed=5)


@case("randomization.ER_shuffle.sparse_stored_zero_is_shuffled_too")
def _():
    # nnz counts the stored zero: it takes a position like any edge (and then vanishes from the snapshot's nonzeros)
    adj = weighted(12, 0.3, seed=2)
    adj.data[0] = 0.0
    out = _er_shuffle(adj, seed=5)
    return {"result": out, "stored_entries": out.nnz, "input_stored_entries": adj.nnz}


@case("randomization.ER_shuffle.sparse_diagonal_entries_move_off_the_diagonal")
def _():
    # K counts the diagonal entries too: they come back as ordinary off-diagonal edges
    adj = weighted(12, 0.3, seed=2).tolil()
    adj[3, 3] = 4.0
    adj[7, 7] = 2.0
    out = _er_shuffle(adj.tocsr(), seed=5)
    return {"result": out, "diagonal": out.diagonal().tolist()}


@case("randomization.ER_shuffle.sparse_complete_graph")
def _():
    # all n (n - 1) positions are taken: any seed gives the complete digraph, with the weights permuted
    w = np.arange(1.0, 17.0).reshape(4, 4)
    np.fill_diagonal(w, 0.0)
    return {
        "bool": _er_shuffle(_full(5), seed=1),
        "distinct_weights": _er_shuffle(sp.csr_matrix(w), seed=1),
    }


@case("randomization.ER_shuffle.sparse_tiny_matrices")
def _():
    return {
        "empty": _er_shuffle(sp.csr_matrix((5, 5)), seed=1),
        "one_node": _er_shuffle(sp.csr_matrix((1, 1)), seed=1),
        "two_nodes_one_edge": _er_shuffle(sp.csr_matrix(np.array([[0, 1.0], [0, 0]])), seed=1),
        "two_nodes_two_edges": _er_shuffle(sp.csr_matrix(np.array([[0, 1.0], [2.0, 0]])), seed=1),
        "three_nodes": _er_shuffle(_full(3), seed=1),
    }


@case("randomization.ER_shuffle.sparse_more_entries_than_off_diagonal_pairs")
def _():
    return _er_shuffle(sp.csr_matrix(np.ones((4, 4))), seed=1)


@case("randomization.ER_shuffle.sparse_non_square")
def _():
    return _er_shuffle(sp.csr_matrix(np.ones((3, 5))), seed=1)


@case("randomization.ER_shuffle.sparse_rejects_a_dense_array")
def _():
    return _er_shuffle(weighted(12, 0.3, seed=2).toarray(), seed=5)


@case("randomization.ER_shuffle.seed_zero_is_a_seed")
def _():
    # only None means "no seed"
    adj = digraph(20, 0.2, seed=7)
    return {"zero": _er_shuffle(adj, seed=0), "one": _er_shuffle(adj, seed=1)}


@case("randomization.ER_shuffle.seed_out_of_range_for_the_legacy_generator")
def _():
    return _er_shuffle(digraph(20, 0.2, seed=7), seed=2**40)


@case("randomization.ER_shuffle.positional_arguments")
def _():
    # the wrapper is (adj, neuron_properties=[], seed=None, **kwargs)
    adj = digraph(20, 0.2, seed=7)
    return {
        "seed_positional": R.ER_shuffle(adj, [], 5),
        "seed_keyword": R.ER_shuffle(adj, seed=5),
        "properties_ignored": R.ER_shuffle(adj, "anything", 5),
    }


@case("randomization.ER_shuffle.shuffle_type_positional_rejected")
def _():
    return R.ER_shuffle(digraph(20, 0.2, seed=7), [], 5, "dense")


@case("randomization.ER_shuffle.unknown_shuffle_type_returns_none")
def _():
    return _er_shuffle(digraph(20, 0.2, seed=7), seed=5, shuffle_type="rows")


@case("randomization.ER_shuffle.unknown_keyword_rejected")
def _():
    return _er_shuffle(digraph(20, 0.2, seed=7), seed=5, mode="dense")


# ----------------------------------------------------------------------------- ER_shuffle, dense


@case("randomization.ER_shuffle.dense_from_sparse_input")
def _():
    # a sparse input is converted: it is not modified
    adj = weighted(12, 0.3, seed=2)
    before = adj.copy()
    out = _er_shuffle(adj, seed=5, shuffle_type="dense")
    return {"result": out, "input_unchanged": bool((adj != before).nnz == 0)}


@case("randomization.ER_shuffle.dense_array_is_modified_in_place")
def _():
    # a numpy array is shuffled in place and also wrapped in the returned coo matrix
    adj = weighted(12, 0.3, seed=2).toarray()
    original = adj.copy()
    out = _er_shuffle(adj, seed=5, shuffle_type="dense")
    return {
        "result": out,
        "array_after": adj,
        "array_changed": bool((adj != original).any()),
        "result_equals_array": bool((out.toarray() == adj).all()),
    }


@case("randomization.ER_shuffle.dense_keeps_the_diagonal_in_place")
def _():
    adj = weighted(12, 0.3, seed=2).toarray()
    adj[np.arange(12), np.arange(12)] = np.arange(1, 13)
    out = _er_shuffle(adj, seed=5, shuffle_type="dense")
    return {"result": out, "diagonal": out.diagonal()}


@case("randomization.ER_shuffle.dense_dtypes")
def _():
    base = weighted(10, 0.4, seed=3).toarray()
    return {
        name: _er_shuffle(base.astype(dt), seed=4, shuffle_type="dense")
        for name, dt in [
            ("bool", bool),
            ("int64", np.int64),
            ("float32", np.float32),
            ("float64", float),
        ]
    }


@case("randomization.ER_shuffle.dense_from_other_sparse_formats")
def _():
    adj = weighted(12, 0.3, seed=2)
    return {
        fmt: _er_shuffle(adj.asformat(fmt), seed=5, shuffle_type="dense")
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.ER_shuffle.dense_tiny_matrices")
def _():
    return {
        "one_node": _er_shuffle(np.array([[3.0]]), seed=1, shuffle_type="dense"),
        "two_nodes": _er_shuffle(np.array([[0.0, 1.0], [2.0, 0.0]]), seed=1, shuffle_type="dense"),
        "empty": _er_shuffle(np.zeros((4, 4)), seed=1, shuffle_type="dense"),
    }


@case("randomization.ER_shuffle.dense_non_square")
def _():
    return _er_shuffle(np.arange(15.0).reshape(3, 5), seed=1, shuffle_type="dense")


# ----------------------------------------------------------------------------- ER_shuffle, the global random state


@case("randomization.ER_shuffle.no_seed_uses_the_global_state")
def _():
    # without a seed the permutation continues the global stream (the harness seeds it; here explicitly)
    adj = digraph(30, 0.15, seed=8)
    np.random.seed(7)
    first = _er_shuffle(adj)
    np.random.seed(7)
    second = R.ER_shuffle(adj, seed=None)
    np.random.seed(8)
    other = _er_shuffle(adj)
    return {"omitted": first, "none": second, "other_state": other}


@case("randomization.ER_shuffle.no_seed_dense_uses_the_global_state")
def _():
    adj = weighted(12, 0.3, seed=2)
    np.random.seed(7)
    return _er_shuffle(adj, shuffle_type="dense")


@case("randomization.ER_shuffle.stream_after_seeded_sparse_call")
def _():
    # a seeded call re-seeds the global state, then draws a permutation of n (n - 1) positions
    np.random.seed(99)
    _er_shuffle(digraph(30, 0.15, seed=8), seed=3)
    return np.random.rand(3)


@case("randomization.ER_shuffle.stream_after_seeded_dense_call")
def _():
    np.random.seed(99)
    _er_shuffle(weighted(12, 0.3, seed=2), seed=3, shuffle_type="dense")
    return np.random.rand(3)


@case("randomization.ER_shuffle.stream_after_unseeded_sparse_call")
def _():
    np.random.seed(99)
    _er_shuffle(digraph(30, 0.15, seed=8))
    return np.random.rand(3)


@case("randomization.ER_shuffle.stream_after_unseeded_dense_call")
def _():
    np.random.seed(99)
    _er_shuffle(weighted(12, 0.3, seed=2), shuffle_type="dense")
    return np.random.rand(3)


@case("randomization.ER_shuffle.real_data")
def _():
    adj, _ = tutorial_connectome(300)
    return _er_shuffle(adj, seed=1)


@case("randomization.ER_shuffle.real_data_dense")
def _():
    adj, _ = tutorial_connectome(150)
    return _er_shuffle(adj.toarray(), seed=1, shuffle_type="dense")


# ----------------------------------------------------------------------------- configuration_model


def _cm_input():
    return digraph(30, 0.2, seed=4)


@case("randomization.configuration_model.seeded", tags=GEN)
def _():
    return R.configuration_model(_cm_input(), seed=1)


@case("randomization.configuration_model.seed_values", tags=GEN)
def _():
    adj = _cm_input()
    return {
        "0": R.configuration_model(adj, seed=0),
        "2": R.configuration_model(adj, seed=2),
        "tuple": R.configuration_model(adj, seed=(1, 2)),
        "numpy_int": R.configuration_model(adj, seed=np.int64(1)),
        "generator": R.configuration_model(adj, seed=np.random.default_rng(5)),
    }


@case("randomization.configuration_model.input_formats", tags=GEN)
def _():
    # csr, coo and lil give the same graph (same row-major order of the stored edges); csc is column-major
    adj = _cm_input()
    return {
        fmt: R.configuration_model(adj.asformat(fmt), seed=1)
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.configuration_model.weights_and_dtypes_are_ignored", tags=GEN)
def _():
    adj = _cm_input()
    inputs = {
        "bool": adj,
        "int": adj.astype(int),
        "float": adj.astype(float),
        "weighted": weighted(30, 0.2, seed=4),
    }
    return {name: R.configuration_model(m, seed=1) for name, m in inputs.items()}


@case("randomization.configuration_model.multi_edges_become_weights_above_one", tags=GEN)
def _():
    # the docstring says multiple edges are removed; they are summed (the data are integer counts)
    out = R.configuration_model(_full(7), seed=3)
    values, counts = np.unique(out.data, return_counts=True)
    return {
        "result": out,
        "weights": values,
        "number_of_edges_per_weight": counts,
        "total_weight": int(out.sum()),
    }


@case("randomization.configuration_model.loops_are_removed", tags=GEN)
def _():
    # a matrix of loops only: the shuffled edges are mostly off the diagonal, the remaining loops are dropped
    return R.configuration_model(sp.csr_matrix(np.eye(6)), seed=2)


@case("randomization.configuration_model.result_dtype_and_class", tags=GEN)
def _():
    out = R.configuration_model(_cm_input(), seed=1)
    return {
        "cls": type(out).__name__,
        "dtype": str(out.dtype),
        "diagonal_sum": int(out.diagonal().sum()),
        "min_weight": int(out.data.min()),
    }


@case("randomization.configuration_model.empty_input_has_float_dtype")
def _():
    # [1] * 0 is an empty list: the data array is float64 instead of int64
    return {
        "empty": R.configuration_model(sp.csr_matrix((6, 6)), seed=1),
        "single_edge": R.configuration_model(
            sp.csr_matrix(([1], ([0], [1])), shape=(3, 3)), seed=1
        ),
    }


@case("randomization.configuration_model.input_is_not_modified", tags=GEN)
def _():
    adj = _cm_input()
    coo = adj.tocoo()
    before = (coo.row.copy(), coo.col.copy())
    out_csr = R.configuration_model(adj, seed=1)
    out_coo = R.configuration_model(coo, seed=1)
    return {
        "csr": out_csr,
        "coo": out_coo,
        "csr_unchanged": bool((adj != _cm_input()).nnz == 0),
        "coo_row_unchanged": bool((coo.row == before[0]).all()),
        "coo_col_unchanged": bool((coo.col == before[1]).all()),
    }


@case("randomization.configuration_model.coo_with_duplicate_entries", tags=GEN)
def _():
    # every stored entry (also a repeated one) is one stub
    coo = sp.coo_matrix(([1, 1, 1, 1], ([0, 0, 1, 2], [1, 1, 2, 3])), shape=(4, 4))
    return R.configuration_model(coo, seed=1)


@case("randomization.configuration_model.non_square", tags=GEN)
def _():
    return R.configuration_model(sp.csr_matrix(np.ones((3, 5))), seed=1)


@case("randomization.configuration_model.unseeded_properties")
def _():
    # the multi-edge weights vary from run to run, so the set of weights is not pinned
    adj = _cm_input()
    out = R.configuration_model(adj)
    return {
        "cls": type(out).__name__,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "diagonal_sum": int(out.diagonal().sum()),
        "total_weight_at_most_edges": bool(out.sum() <= adj.nnz),
        "min_weight": int(out.data.min()),
    }


@case("randomization.configuration_model.dense_input_rejected")
def _():
    return R.configuration_model(_cm_input().toarray(), seed=1)


@case("randomization.configuration_model.real_data", tags=GEN)
def _():
    # as in the randomization tutorial: configuration_model(A.tocoo(), 1)
    adj, _ = tutorial_connectome(300)
    return R.configuration_model(adj.tocoo(), 1)


# ----------------------------------------------------------------------------- adjusted_ER


@case("randomization.adjusted_ER.seeded", tags=GEN)
def _():
    return R.adjusted_ER(reciprocal_graph(), seed=3)


@case("randomization.adjusted_ER.seed_values", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {s: R.adjusted_ER(adj, seed=s) for s in (0, 1, 4)}


@case("randomization.adjusted_ER.keeps_edge_and_reciprocal_counts", tags=GEN)
def _():
    adj = reciprocal_graph()
    out = R.adjusted_ER(adj, seed=3)
    return {
        "result": out,
        "edges": [int(adj.nnz), int(out.nnz)],
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
    }


@case("randomization.adjusted_ER.input_formats", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {fmt: R.adjusted_ER(adj.asformat(fmt), seed=3) for fmt in ("csr", "csc", "coo")}


@case("randomization.adjusted_ER.weighted_input", tags=GEN)
def _():
    # weights are shuffled with the edges and then 1 is subtracted from removed one-way edges
    return R.adjusted_ER(weighted(30, 0.15, seed=4), seed=3)


@case("randomization.adjusted_ER.integer_and_float_input", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {
        "int": R.adjusted_ER(adj.astype(int), seed=3),
        "float": R.adjusted_ER(adj.astype(float), seed=3),
    }


@case("randomization.adjusted_ER.dense_one_way_graph", tags=GEN)
def _():
    # a dense random graph has many reciprocal pairs already: the control is adjusted up to that number
    adj = digraph(30, 0.5, seed=4)
    out = R.adjusted_ER(adj, seed=3)
    return {"result": out, "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)]}


@case("randomization.adjusted_ER.shuffle_already_has_more_reciprocal_pairs", tags=GEN)
def _():
    # a transitive tournament has no reciprocal pair, so the plain shuffle has more: it is returned as it is
    adj = sp.csr_matrix(np.triu(np.ones((8, 8)), 1))
    out = R.adjusted_ER(adj, seed=3)
    shuffled = R.ER_shuffle(adj, seed=3)
    return {
        "result": out,
        "equals_plain_shuffle": bool((out != shuffled.tocsc()).nnz == 0),
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
    }


@case("randomization.adjusted_ER.complete_graph")
def _():
    # every edge is reciprocal already: nothing to adjust (zero bidirectional edges to add)
    return R.adjusted_ER(_full(6), seed=3)


@case("randomization.adjusted_ER.empty_graph")
def _():
    return R.adjusted_ER(sp.csr_matrix((8, 8), dtype=bool), seed=3)


@case("randomization.adjusted_ER.lil_rejected")
def _():
    return R.adjusted_ER(reciprocal_graph(fmt="lil"), seed=3)


@case("randomization.adjusted_ER.dense_rejected")
def _():
    return R.adjusted_ER(reciprocal_graph().toarray(), seed=3)


@case("randomization.adjusted_ER.stream_after_call")
def _():
    # ER_shuffle(adj, seed=seed) leaves the global state seeded and consumed
    np.random.seed(99)
    R.adjusted_ER(reciprocal_graph(), seed=3)
    return np.random.rand(3)


@case("randomization.adjusted_ER.unseeded_properties")
def _():
    # seed=None: the shuffle continues the global state (seeded here), the adjustment uses fresh entropy
    adj = reciprocal_graph()
    np.random.seed(5)
    out = R.adjusted_ER(adj)
    return {
        **edge_info(out),
        "same_edge_count_as_input": bool(out.nnz == adj.nnz),
        "same_reciprocal_count_as_input": bool(reciprocal_entries(out) == reciprocal_entries(adj)),
    }


@case("randomization.adjusted_ER.real_data", tags=GEN)
def _():
    adj, _ = tutorial_connectome(300)
    return R.adjusted_ER(adj, seed=1)


# ----------------------------------------------------------------------------- underlying_model


@case("randomization.underlying_model.seeded", tags=GEN)
def _():
    return R.underlying_model(reciprocal_graph(), seed=3)


@case("randomization.underlying_model.seed_values", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {s: R.underlying_model(adj, seed=s) for s in (0, 1, 4)}


@case("randomization.underlying_model.keeps_the_underlying_graph_and_reciprocal_count", tags=GEN)
def _():
    adj = reciprocal_graph()
    out = R.underlying_model(adj, seed=3)
    return {
        "result": out,
        "undirected_edges": [undirected_edges(adj), undirected_edges(out)],
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
    }


@case("randomization.underlying_model.input_formats", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {
        name: R.underlying_model(m, seed=3)
        for name, m in [
            ("csr", adj),
            ("csc", adj.tocsc()),
            ("coo", adj.tocoo()),
            ("dense", adj.toarray()),
        ]
    }


@case("randomization.underlying_model.integer_input_gives_weight_two_on_reciprocal_pairs", tags=GEN)
def _():
    # adj + adj.T is 2 on reciprocal pairs for a numeric dtype (and True for bool)
    adj = reciprocal_graph()
    out_int = R.underlying_model(adj.astype(int), seed=3)
    return {
        "int": out_int,
        "weights": np.unique(out_int.data),
        "bool": R.underlying_model(adj, seed=3),
    }


@case("randomization.underlying_model.weighted_input", tags=GEN)
def _():
    return R.underlying_model(weighted(30, 0.15, seed=4), seed=3)


@case("randomization.underlying_model.no_reciprocal_connections", tags=GEN)
def _():
    return R.underlying_model(digraph(20, 0.1, seed=4), seed=3)


@case("randomization.underlying_model.complete_graph")
def _():
    return R.underlying_model(_full(6), seed=3)


@case("randomization.underlying_model.empty_graph")
def _():
    return R.underlying_model(sp.csr_matrix((8, 8), dtype=bool), seed=3)


@case("randomization.underlying_model.non_square")
def _():
    return R.underlying_model(sp.csr_matrix(np.ones((3, 5))), seed=3)


@case("randomization.underlying_model.unseeded_properties")
def _():
    adj = reciprocal_graph()
    out = R.underlying_model(adj)
    return {
        **edge_info(out),
        "same_undirected_edges_as_input": bool(undirected_edges(out) == undirected_edges(adj)),
        "same_reciprocal_count_as_input": bool(reciprocal_entries(out) == reciprocal_entries(adj)),
    }


@case("randomization.underlying_model.real_data", tags=GEN)
def _():
    adj, _ = tutorial_connectome(300)
    return R.underlying_model(adj, seed=1)


# ----------------------------------------------------------------------------- bishuffled_model


@case("randomization.bishuffled_model.seeded", tags=GEN)
def _():
    return R.bishuffled_model(reciprocal_graph(), seed=3)


@case("randomization.bishuffled_model.seed_values", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {s: R.bishuffled_model(adj, seed=s) for s in (0, 1, 4)}


@case("randomization.bishuffled_model.keeps_the_edge_and_reciprocal_counts", tags=GEN)
def _():
    adj = reciprocal_graph()
    out = R.bishuffled_model(adj, seed=3)
    return {
        "result": out,
        "edges": [int(adj.nnz), int(out.nnz)],
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
    }


@case("randomization.bishuffled_model.input_formats", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {fmt: R.bishuffled_model(adj.asformat(fmt), seed=3) for fmt in ("csr", "csc", "coo")}


@case("randomization.bishuffled_model.integer_and_float_input", tags=GEN)
def _():
    adj = reciprocal_graph()
    return {
        "int": R.bishuffled_model(adj.astype(int), seed=3),
        "float": R.bishuffled_model(adj.astype(float), seed=3),
    }


@case("randomization.bishuffled_model.weighted_input", tags=GEN)
def _():
    return R.bishuffled_model(weighted(30, 0.15, seed=4), seed=3)


@case("randomization.bishuffled_model.no_reciprocal_connections", tags=GEN)
def _():
    return R.bishuffled_model(digraph(20, 0.1, seed=4), seed=3)


@case("randomization.bishuffled_model.complete_graph")
def _():
    return R.bishuffled_model(_full(6), seed=3)


@case("randomization.bishuffled_model.empty_graph")
def _():
    return R.bishuffled_model(sp.csr_matrix((8, 8), dtype=bool), seed=3)


@case("randomization.bishuffled_model.dense_rejected")
def _():
    return R.bishuffled_model(reciprocal_graph().toarray(), seed=3)


@case("randomization.bishuffled_model.unseeded_properties")
def _():
    adj = reciprocal_graph()
    out = R.bishuffled_model(adj)
    return {
        **edge_info(out),
        "same_edge_count_as_input": bool(out.nnz == adj.nnz),
        "same_reciprocal_count_as_input": bool(reciprocal_entries(out) == reciprocal_entries(adj)),
    }


@case("randomization.bishuffled_model.real_data", tags=GEN)
def _():
    adj, _ = tutorial_connectome(300)
    return R.bishuffled_model(adj, seed=1)
