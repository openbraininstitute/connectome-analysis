"""Helpers in ``connalysis.randomization.rand_utils``.

The subsampling helpers take a ``generator`` and only call ``generator.choice(n, replace=False, size=k)`` or
``generator.permutation(n)``.  They are covered with ``np.random.default_rng`` (tagged ``generator-stream``) and with
``np.random.RandomState``, whose streams are frozen across numpy releases.  ``evaluate_probs*`` use the legacy global
state and are seeded inside each case.  The weight helpers return pandas objects with a (neuron, i) MultiIndex.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, positions, weighted

from cases.randomization_common import GEN, U, reciprocal_entries


def _gen(seed=0):
    return np.random.default_rng(seed)


def _rs(seed=0):
    return np.random.RandomState(seed)


def _coo(n=10, p=0.3, seed=1):
    return digraph(n, p, seed=seed).tocoo()


def _bi_input(fmt="csc"):
    """20 nodes, 112 edges of which 78 entries are reciprocal and 34 are one-way."""
    return digraph(20, 0.2, seed=1, reciprocal=0.3, fmt=fmt)


def _supports_disjoint(mats):
    return all(
        (mats[i].multiply(mats[j])).nnz == 0
        for i in range(len(mats))
        for j in range(i + 1, len(mats))
    )


# ----------------------------------------------------------------------------- subset_matrix


@case("randomization.subset_matrix.basic")
def _():
    return U.subset_matrix(_coo(), np.array([0, 2, 4]))


@case("randomization.subset_matrix.selected_edges_become_ones")
def _():
    # weights and dtype of the input are dropped: float64 ones
    w = weighted(10, 0.3, seed=1).tocoo()
    return {
        "weighted": U.subset_matrix(w, np.array([0, 2, 4])),
        "bool": U.subset_matrix(_coo(), np.array([0, 2, 4])),
    }


@case("randomization.subset_matrix.selection_types")
def _():
    coo = _coo()
    return {
        "list": U.subset_matrix(coo, [0, 1, 5]),
        "int32": U.subset_matrix(coo, np.array([0, 1, 5], dtype=np.int32)),
        "unsorted": U.subset_matrix(coo, np.array([7, 3, 5])),
        "negative": U.subset_matrix(coo, np.array([-1, -2])),
        "empty": U.subset_matrix(coo, np.array([], dtype=int)),
        "range": U.subset_matrix(coo, range(0, 6, 2)),
    }


@case("randomization.subset_matrix.repeated_index_gives_duplicate_entries")
def _():
    out = U.subset_matrix(_coo(), np.array([1, 1, 1, 4]))
    return {"matrix": out, "stored_entries": out.nnz, "csr": out.tocsr()}


@case("randomization.subset_matrix.non_square_shape_is_kept")
def _():
    coo = sp.coo_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0]]))
    return U.subset_matrix(coo, np.array([0, 3]))


@case("randomization.subset_matrix.all_edges")
def _():
    coo = _coo()
    return U.subset_matrix(coo, np.arange(coo.nnz))


@case("randomization.subset_matrix.csr_input_rejected")
def _():
    return U.subset_matrix(digraph(10, 0.3, seed=1), np.array([0, 2]))


@case("randomization.subset_matrix.index_out_of_range")
def _():
    return U.subset_matrix(_coo(), np.array([1000]))


# ----------------------------------------------------------------------------- subsampled_matrix


@case("randomization.subsampled_matrix.generator", tags=GEN)
def _():
    return U.subsampled_matrix(_coo(), 5, _gen(0))


@case("randomization.subsampled_matrix.random_state")
def _():
    return U.subsampled_matrix(_coo(), 5, _rs(0))


@case("randomization.subsampled_matrix.sizes_and_seeds", tags=GEN)
def _():
    coo = _coo()
    return {(n, s): U.subsampled_matrix(coo, n, _gen(s)) for n in (1, 10) for s in (0, 1)}


@case("randomization.subsampled_matrix.sizes_and_seeds_random_state")
def _():
    coo = _coo()
    return {(n, s): U.subsampled_matrix(coo, n, _rs(s)) for n in (1, 10) for s in (0, 1)}


@case("randomization.subsampled_matrix.all_edges_or_none", tags=GEN)
def _():
    coo = _coo()
    return {
        "none": U.subsampled_matrix(coo, 0, _gen(0)),
        "all": U.subsampled_matrix(coo, coo.nnz, _gen(0)),
    }


@case("randomization.subsampled_matrix.weights_are_dropped")
def _():
    return U.subsampled_matrix(weighted(10, 0.3, seed=1).tocoo(), 6, _rs(2))


@case("randomization.subsampled_matrix.draws_one_choice_from_the_generator", tags=GEN)
def _():
    g = _gen(3)
    out = U.subsampled_matrix(_coo(), 5, g)
    return {"result": out, "next_draw": g.random(3)}


@case("randomization.subsampled_matrix.draws_one_choice_from_the_random_state")
def _():
    g = _rs(3)
    out = U.subsampled_matrix(_coo(), 5, g)
    return {"result": out, "next_draw": g.random_sample(3)}


@case("randomization.subsampled_matrix.more_than_the_edges")
def _():
    coo = _coo()
    return U.subsampled_matrix(coo, coo.nnz + 1, _gen(0))


# ----------------------------------------------------------------------------- non_overlapping_subsampled_matrices


@case("randomization.non_overlapping_subsampled_matrices.generator", tags=GEN)
def _():
    mats = U.non_overlapping_subsampled_matrices(_coo(), [3, 4, 2], _gen(0))
    return {
        "matrices": mats,
        "disjoint": _supports_disjoint(mats),
        "sizes": [m.nnz for m in mats],
        "type": type(mats).__name__,
    }


@case("randomization.non_overlapping_subsampled_matrices.random_state")
def _():
    mats = U.non_overlapping_subsampled_matrices(_coo(), [3, 4, 2], _rs(0))
    return {"matrices": mats, "disjoint": _supports_disjoint(mats)}


@case("randomization.non_overlapping_subsampled_matrices.size_containers")
def _():
    coo = _coo()
    return {
        "tuple": U.non_overlapping_subsampled_matrices(coo, (2, 3), _rs(1)),
        "array": U.non_overlapping_subsampled_matrices(coo, np.array([2, 3]), _rs(1)),
        "list": U.non_overlapping_subsampled_matrices(coo, [2, 3], _rs(1)),
    }


@case("randomization.non_overlapping_subsampled_matrices.zero_sizes_and_exact_total")
def _():
    coo = _coo()
    return {
        "zeros": U.non_overlapping_subsampled_matrices(coo, [0, 3, 0], _rs(1)),
        "single": U.non_overlapping_subsampled_matrices(coo, [4], _rs(1)),
        "partition": U.non_overlapping_subsampled_matrices(coo, [10, 20], _rs(1)),
        "no_sizes": U.non_overlapping_subsampled_matrices(coo, [], _rs(1)),
    }


@case("randomization.non_overlapping_subsampled_matrices.consumes_one_choice_from_the_random_state")
def _():
    g = _rs(5)
    mats = U.non_overlapping_subsampled_matrices(_coo(), [3, 4], g)
    return {"matrices": mats, "next_draw": g.random_sample(3)}


@case("randomization.non_overlapping_subsampled_matrices.total_larger_than_the_edges")
def _():
    coo = _coo()
    return U.non_overlapping_subsampled_matrices(coo, [coo.nnz, 1], _gen(0))


# ----------------------------------------------------------------------------- half_matrix


@case("randomization.half_matrix.basic", tags=GEN)
def _():
    halves = U.half_matrix(_coo(), _gen(0))
    return {
        "halves": halves,
        "type": type(halves).__name__,
        "sizes": [h.nnz for h in halves],
        "disjoint": _supports_disjoint(list(halves)),
        "union_nnz": int(sum(h.nnz for h in halves)),
    }


@case("randomization.half_matrix.random_state")
def _():
    return U.half_matrix(_coo(), _rs(0))


@case("randomization.half_matrix.odd_number_of_edges_gives_the_larger_half_second")
def _():
    odd = sp.coo_matrix(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]))
    return {
        "three": U.half_matrix(odd, _rs(0)),
        "one": U.half_matrix(sp.coo_matrix(np.array([[0, 1], [0, 0]])), _rs(0)),
    }


@case("randomization.half_matrix.even_number_of_edges")
def _():
    return U.half_matrix(sp.coo_matrix(np.ones((3, 3)) - np.eye(3)), _rs(0))


@case("randomization.half_matrix.no_edges")
def _():
    return U.half_matrix(sp.coo_matrix((3, 3)), _rs(0))


@case("randomization.half_matrix.weights_are_dropped")
def _():
    return U.half_matrix(weighted(10, 0.3, seed=1).tocoo(), _rs(0))


@case("randomization.half_matrix.seeds_change_the_split", tags=GEN)
def _():
    coo = _coo()
    return {s: U.half_matrix(coo, _gen(s)) for s in (0, 1, 2)}


@case("randomization.half_matrix.consumes_one_permutation_from_the_random_state")
def _():
    g = _rs(7)
    halves = U.half_matrix(_coo(), g)
    return {"halves": halves, "next_draw": g.random_sample(3)}


@case("randomization.half_matrix.csr_input_rejected")
def _():
    return U.half_matrix(digraph(10, 0.3, seed=1), _gen(0))


# ----------------------------------------------------------------------------- adjust_bidirectional_connections


@case("randomization.adjust_bidirectional_connections.generator", tags=GEN)
def _():
    return U.adjust_bidirectional_connections(_bi_input(), 3, _gen(0))


@case("randomization.adjust_bidirectional_connections.random_state")
def _():
    return U.adjust_bidirectional_connections(_bi_input(), 3, _rs(0))


@case("randomization.adjust_bidirectional_connections.counts_after_adjusting")
def _():
    # k one-way edges get their reverse, k other one-way edges are removed: same edge count, 2k more reciprocal entries
    adj = _bi_input()
    out = U.adjust_bidirectional_connections(adj, 3, _rs(0))
    return {
        "result": out,
        "edges": [int(adj.nnz), int(out.nnz)],
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
        "cls": type(out).__name__,
        "dtype": str(out.dtype),
    }


@case("randomization.adjust_bidirectional_connections.input_formats")
def _():
    return {
        fmt: U.adjust_bidirectional_connections(_bi_input(fmt), 3, _rs(0))
        for fmt in ("csc", "csr", "coo", "lil")
    }


@case("randomization.adjust_bidirectional_connections.dtypes")
def _():
    adj = _bi_input()
    return {
        name: U.adjust_bidirectional_connections(adj.astype(dt), 2, _rs(0))
        for name, dt in [
            ("bool", bool),
            ("int8", np.int8),
            ("int64", np.int64),
            ("float32", np.float32),
            ("float64", float),
        ]
    }


@case("randomization.adjust_bidirectional_connections.weighted_edges_lose_one_each_time")
def _():
    # the removed edges are subtracted as ones: weights go down by 1 (zeros disappear), the new reverse edges have weight 1
    return U.adjust_bidirectional_connections(weighted(20, 0.2, seed=1), 3, _rs(0))


@case("randomization.adjust_bidirectional_connections.zero_edges")
def _():
    adj = _bi_input()
    out = U.adjust_bidirectional_connections(adj, 0, _rs(0))
    return {"result": out, "unchanged": bool((out != adj).nnz == 0)}


@case("randomization.adjust_bidirectional_connections.more_than_half_of_the_one_way_edges")
def _():
    # needs 2 k one-way edges (k to reverse, k to remove)
    return U.adjust_bidirectional_connections(_bi_input(), 18, _rs(0))


@case("randomization.adjust_bidirectional_connections.exactly_half_of_the_one_way_edges")
def _():
    return U.adjust_bidirectional_connections(_bi_input(), 17, _rs(0))


@case("randomization.adjust_bidirectional_connections.negative_number")
def _():
    return U.adjust_bidirectional_connections(_bi_input(), -1, _rs(0))


@case("randomization.adjust_bidirectional_connections.draws_from_the_generator", tags=GEN)
def _():
    g = _gen(4)
    out = U.adjust_bidirectional_connections(_bi_input(), 3, g)
    return {"result": out, "next_draw": g.random(3)}


# ----------------------------------------------------------------------------- add_bidirectional_connections


@case("randomization.add_bidirectional_connections.generator", tags=GEN)
def _():
    return U.add_bidirectional_connections(_bi_input(), 3, _gen(0))


@case("randomization.add_bidirectional_connections.random_state")
def _():
    return U.add_bidirectional_connections(_bi_input(), 3, _rs(0))


@case("randomization.add_bidirectional_connections.counts_after_adding")
def _():
    adj = _bi_input()
    out = U.add_bidirectional_connections(adj, 3, _rs(0))
    return {
        "result": out,
        "edges": [int(adj.nnz), int(out.nnz)],
        "reciprocal_entries": [reciprocal_entries(adj), reciprocal_entries(out)],
        "cls": type(out).__name__,
        "dtype": str(out.dtype),
    }


@case("randomization.add_bidirectional_connections.input_formats")
def _():
    return {
        fmt: U.add_bidirectional_connections(_bi_input(fmt), 3, _rs(0))
        for fmt in ("csc", "csr", "coo", "lil")
    }


@case("randomization.add_bidirectional_connections.dtypes")
def _():
    adj = _bi_input()
    return {
        name: U.add_bidirectional_connections(adj.astype(dt), 4, _rs(0))
        for name, dt in [
            ("bool", bool),
            ("int8", np.int8),
            ("int64", np.int64),
            ("float32", np.float32),
            ("float64", float),
        ]
    }


@case("randomization.add_bidirectional_connections.weighted_edges")
def _():
    # the reverse edges have weight 1 (ones from subsampled_matrix), whatever the weight of the original edge
    return U.add_bidirectional_connections(weighted(20, 0.2, seed=1), 4, _rs(0))


@case("randomization.add_bidirectional_connections.zero_edges")
def _():
    adj = _bi_input()
    out = U.add_bidirectional_connections(adj, 0, _rs(0))
    return {"result": out, "unchanged": bool((out != adj).nnz == 0)}


@case("randomization.add_bidirectional_connections.all_one_way_edges")
def _():
    return U.add_bidirectional_connections(_bi_input(), 17, _rs(0))


@case("randomization.add_bidirectional_connections.more_than_the_one_way_edges")
def _():
    return U.add_bidirectional_connections(_bi_input(), 18, _rs(0))


@case("randomization.add_bidirectional_connections.negative_number")
def _():
    return U.add_bidirectional_connections(_bi_input(), -1, _rs(0))


@case("randomization.add_bidirectional_connections.dense_input_rejected")
def _():
    return U.add_bidirectional_connections(_bi_input().toarray(), 2, _rs(0))


@case("randomization.add_bidirectional_connections.draws_from_the_generator", tags=GEN)
def _():
    g = _gen(4)
    out = U.add_bidirectional_connections(_bi_input(), 3, g)
    return {"result": out, "next_draw": g.random(3)}


@case("randomization.add_bidirectional_connections.upper_triangular_input", tags=GEN)
def _():
    # how underlying_model uses it: no reciprocal pair at all, k reverse edges added
    return U.add_bidirectional_connections(sp.triu(_bi_input().astype(float)), 10, _gen(1))


# ----------------------------------------------------------------------------- evaluate_probs_less_random and evaluate_probs


def _state(seed=0, n=8):
    """Candidate probabilities of a spread step: random positive entries, many rows with a sum above 1."""
    r = np.random.RandomState(seed)
    return sp.csr_matrix(np.where(r.rand(n, n) < 0.5, r.rand(n, n), 0.0))


def _evaluate(fn, p, seed=1, **kw):
    np.random.seed(seed)
    return fn(p, **kw)


def _stream_after(fn, p, seed=1, **kw):
    np.random.seed(seed)
    fn(p, **kw)
    return np.random.rand(3)


ADJUST = np.linspace(0.5, 3.0, 8)


@case("randomization.evaluate_probs.default")
def _():
    # every stored entry is compared with one uniform draw, in the stored order of the coo matrix
    return _evaluate(U.evaluate_probs, _state())


@case("randomization.evaluate_probs.adjust")
def _():
    return _evaluate(U.evaluate_probs, _state(), adjust=ADJUST)


@case("randomization.evaluate_probs.less_random")
def _():
    return _evaluate(U.evaluate_probs, _state(), less_random=True)


@case("randomization.evaluate_probs.less_random_with_adjust")
def _():
    return _evaluate(U.evaluate_probs, _state(), adjust=ADJUST, less_random=True)


@case("randomization.evaluate_probs.result_types")
def _():
    # float ones in a coo matrix for the plain step, bool for the less random one
    p = _state()
    plain, less = _evaluate(U.evaluate_probs, p), _evaluate(U.evaluate_probs, p, less_random=True)
    return {
        "plain": [type(plain).__name__, str(plain.dtype)],
        "less_random": [type(less).__name__, str(less.dtype)],
    }


@case("randomization.evaluate_probs.input_formats")
def _():
    # the draws follow the stored order, so csc input selects other entries than csr
    p = _state()
    return {
        fmt: _evaluate(U.evaluate_probs, p.asformat(fmt)) for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.evaluate_probs.input_formats_less_random")
def _():
    p = _state()
    return {
        fmt: _evaluate(U.evaluate_probs, p.asformat(fmt), less_random=True)
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.evaluate_probs.dtypes")
def _():
    p = _state()
    return {
        name: _evaluate(U.evaluate_probs, p.astype(dt))
        for name, dt in [("float32", np.float32), ("bool", bool)]
    }


@case("randomization.evaluate_probs.dtypes_less_random")
def _():
    p = _state()
    return {
        name: _evaluate(U.evaluate_probs, p.astype(dt), less_random=True)
        for name, dt in [("float32", np.float32), ("bool", bool)]
    }


@case("randomization.evaluate_probs.negative_entries_are_never_selected")
def _():
    p = _state().tolil()
    p[0, 0] = -1.0
    p[1, 1] = -2.0
    return _evaluate(U.evaluate_probs, p.tocsr())


@case("randomization.evaluate_probs.negative_entries_break_the_less_random_step")
def _():
    # docstring: negative entries mark excluded columns; the normalised row then has negative probabilities
    p = _state().tolil()
    p[0, 0] = -1.0
    p[1, 1] = -2.0
    return _evaluate(U.evaluate_probs, p.tocsr(), less_random=True)


@case("randomization.evaluate_probs.stored_zeros")
def _():
    p = _state()
    p.data[:3] = 0.0
    return {
        "plain": _evaluate(U.evaluate_probs, p),
        "less_random": _evaluate(U.evaluate_probs, p, less_random=True),
    }


@case("randomization.evaluate_probs.rows_without_entries")
def _():
    p = sp.csr_matrix(np.array([[0, 0.5, 0.5, 0.5], [0, 0, 0, 0], [0.9, 0, 0.9, 0], [0, 0, 0, 0]]))
    return {
        "plain": _evaluate(U.evaluate_probs, p),
        "less_random": _evaluate(U.evaluate_probs, p, less_random=True),
        "less_random_adjust": _evaluate(U.evaluate_probs, p, adjust=np.ones(4), less_random=True),
    }


@case("randomization.evaluate_probs.probabilities_above_one")
def _():
    p = sp.csr_matrix(np.full((4, 4), 2.0))
    return {
        "plain": _evaluate(U.evaluate_probs, p),
        "less_random": _evaluate(U.evaluate_probs, p, less_random=True),
    }


@case("randomization.evaluate_probs.non_square")
def _():
    r = np.random.RandomState(1)
    p = sp.csr_matrix(np.where(r.rand(3, 6) < 0.6, r.rand(3, 6), 0.0))
    return {
        "plain": _evaluate(U.evaluate_probs, p),
        "less_random": _evaluate(U.evaluate_probs, p, less_random=True),
    }


@case("randomization.evaluate_probs.empty_matrix")
def _():
    return _evaluate(U.evaluate_probs, sp.csr_matrix((5, 5)))


@case("randomization.evaluate_probs.empty_matrix_less_random")
def _():
    # no row picks anything: np.hstack of an empty list fails
    return _evaluate(U.evaluate_probs, sp.csr_matrix((5, 5)), less_random=True)


@case("randomization.evaluate_probs.every_expected_count_rounds_to_zero")
def _():
    return _evaluate(U.evaluate_probs, sp.csr_matrix(np.full((4, 4), 0.01)), less_random=True)


@case("randomization.evaluate_probs.rounded_counts_with_adjust")
def _():
    return _evaluate(
        U.evaluate_probs,
        sp.csr_matrix(np.full((4, 4), 0.01)),
        adjust=np.full(4, 100.0),
        less_random=True,
    )


@case("randomization.evaluate_probs.adjust_is_per_row")
def _():
    # row i is scaled by adjust[i]: zero removes the row, a large factor takes everything
    p = sp.csr_matrix(np.full((4, 5), 0.3))
    return _evaluate(U.evaluate_probs, p, adjust=np.array([0.0, 1.0, 2.0, 10.0]))


@case("randomization.evaluate_probs.negative_adjust")
def _():
    return {"plain": _evaluate(U.evaluate_probs, _state(), adjust=-np.ones(8))}


@case("randomization.evaluate_probs.negative_adjust_less_random")
def _():
    return _evaluate(U.evaluate_probs, _state(), adjust=-np.ones(8), less_random=True)


@case("randomization.evaluate_probs.scalar_adjust")
def _():
    # a scalar works for the less random step (n_pick * adjust) but not for the plain one (adjust[rows])
    return {"less_random": _evaluate(U.evaluate_probs, _state(), adjust=2.0, less_random=True)}


@case("randomization.evaluate_probs.scalar_adjust_plain_step")
def _():
    return _evaluate(U.evaluate_probs, _state(), adjust=2.0)


@case("randomization.evaluate_probs.list_adjust_less_random")
def _():
    return _evaluate(U.evaluate_probs, _state(), adjust=list(ADJUST), less_random=True)


@case("randomization.evaluate_probs.dense_input_rejected")
def _():
    return _evaluate(U.evaluate_probs, _state().toarray())


@case("randomization.evaluate_probs.dense_input_rejected_less_random")
def _():
    return _evaluate(U.evaluate_probs, _state().toarray(), less_random=True)


@case("randomization.evaluate_probs.stream_after_plain_step")
def _():
    # one uniform draw per stored entry
    return _stream_after(U.evaluate_probs, _state())


@case("randomization.evaluate_probs.stream_after_less_random_step")
def _():
    # one np.random.choice(..., p=..., replace=False) per row with a pick
    return _stream_after(U.evaluate_probs, _state(), less_random=True)


@case("randomization.evaluate_probs.stream_after_less_random_step_with_adjust")
def _():
    return _stream_after(U.evaluate_probs, _state(), adjust=ADJUST, less_random=True)


@case("randomization.evaluate_probs.seeds_change_the_step")
def _():
    p = _state()
    return {s: _evaluate(U.evaluate_probs, p, seed=s) for s in (0, 1, 2)}


@case("randomization.evaluate_probs_less_random.default")
def _():
    return _evaluate(U.evaluate_probs_less_random, _state())


@case("randomization.evaluate_probs_less_random.adjust")
def _():
    return _evaluate(U.evaluate_probs_less_random, _state(), adjust=ADJUST)


@case("randomization.evaluate_probs_less_random.counts_per_row")
def _():
    # a row with probabilities summing to s gets round(s * adjust) targets (at most its number of positive entries)
    p = _state()
    out = _evaluate(U.evaluate_probs_less_random, p)
    sums = np.asarray(p.sum(axis=1))[:, 0]
    return {
        "targets_per_row": np.asarray(out.sum(axis=1))[:, 0],
        "expected": np.minimum(np.round(sums), np.diff(p.indptr)),
        "type": [type(out).__name__, str(out.dtype)],
    }


@case("randomization.evaluate_probs_less_random.targets_come_from_the_positive_entries")
def _():
    p = _state()
    out = _evaluate(U.evaluate_probs_less_random, p, adjust=ADJUST)
    return {
        "all_targets_are_candidates": bool(((out.astype(float).multiply(p > 0)).nnz == out.nnz))
    }


@case("randomization.evaluate_probs_less_random.input_formats")
def _():
    p = _state()
    return {
        fmt: _evaluate(U.evaluate_probs_less_random, p.asformat(fmt))
        for fmt in ("csr", "csc", "coo", "lil")
    }


@case("randomization.evaluate_probs_less_random.empty_matrix")
def _():
    return _evaluate(U.evaluate_probs_less_random, sp.csr_matrix((5, 5)))


@case("randomization.evaluate_probs_less_random.negative_entries")
def _():
    p = _state().tolil()
    p[2, 3] = -0.5
    return _evaluate(U.evaluate_probs_less_random, p.tocsr())


@case("randomization.evaluate_probs_less_random.single_row_with_one_target")
def _():
    p = sp.csr_matrix(np.array([[0.0, 0.4, 0.0, 0.6]]))
    return {s: _evaluate(U.evaluate_probs_less_random, p, seed=s) for s in (0, 1, 2, 3)}


@case("randomization.evaluate_probs_less_random.stream_after")
def _():
    return _stream_after(U.evaluate_probs_less_random, _state())


# ----------------------------------------------------------------------------- connection_df_to_csc_matrix


def _edge_series(
    values=(1.5, 2.0, 3.0, 4.0),
    pairs=((0, 1), (0, 2), (1, 0), (3, 2)),
    dtype=float,
    names=("neuron", "i"),
):
    return pd.Series(
        np.array(values, dtype=dtype),
        index=pd.MultiIndex.from_tuples(list(pairs), names=list(names)),
        name="weight",
    )


@case("randomization.connection_df_to_csc_matrix.no_mirror")
def _():
    return U.connection_df_to_csc_matrix(_edge_series(), mirror=False, shape=(4, 4))


@case("randomization.connection_df_to_csc_matrix.mirror_sums_with_the_reverse")
def _():
    # a pair listed in both directions is added: (0, 1) and (1, 0) have weights 1.5 and 3.0, so both become 4.5
    return U.connection_df_to_csc_matrix(_edge_series(), mirror=True, shape=(4, 4))


@case("randomization.connection_df_to_csc_matrix.mirror_is_the_default")
def _():
    s = _edge_series()
    return {
        "default": U.connection_df_to_csc_matrix(s, shape=(4, 4)),
        "explicit": U.connection_df_to_csc_matrix(s, mirror=True, shape=(4, 4)),
    }


@case("randomization.connection_df_to_csc_matrix.mirror_doubles_loops")
def _():
    s = _edge_series(values=(1.0, 2.0), pairs=((0, 0), (1, 2)))
    return U.connection_df_to_csc_matrix(s, mirror=True, shape=(3, 3))


@case("randomization.connection_df_to_csc_matrix.rectangular_shape")
def _():
    s = _edge_series(pairs=((0, 1), (0, 4), (1, 2), (3, 5)))
    return U.connection_df_to_csc_matrix(s, mirror=False, shape=(4, 6))


@case("randomization.connection_df_to_csc_matrix.rectangular_shape_with_mirror")
def _():
    # the mirrored entries are (i, neuron); they fit in a (4, 6) matrix as long as every i is below 4
    s = _edge_series(pairs=((0, 1), (0, 2), (1, 3), (3, 2)))
    return U.connection_df_to_csc_matrix(s, mirror=True, shape=(4, 6))


@case("randomization.connection_df_to_csc_matrix.mirror_outside_a_rectangular_shape")
def _():
    # target 4 mirrors to row 4, which a (4, 6) matrix does not have
    s = _edge_series(pairs=((0, 1), (0, 4), (1, 2), (3, 5)))
    return U.connection_df_to_csc_matrix(s, mirror=True, shape=(4, 6))


@case("randomization.connection_df_to_csc_matrix.value_dtypes")
def _():
    pairs = ((0, 1), (1, 2), (2, 0))
    return {
        "int": U.connection_df_to_csc_matrix(
            _edge_series((1, 2, 3), pairs, dtype=np.int64), mirror=True, shape=(3, 3)
        ),
        "bool": U.connection_df_to_csc_matrix(
            _edge_series((True, True, False), pairs, dtype=bool), mirror=False, shape=(3, 3)
        ),
        "float32": U.connection_df_to_csc_matrix(
            _edge_series((1, 2, 3), pairs, dtype=np.float32), mirror=False, shape=(3, 3)
        ),
    }


@case("randomization.connection_df_to_csc_matrix.repeated_index_entries_are_summed")
def _():
    s = _edge_series(values=(1.0, 2.0, 5.0), pairs=((0, 1), (0, 1), (2, 1)))
    return U.connection_df_to_csc_matrix(s, mirror=False, shape=(3, 3))


@case("randomization.connection_df_to_csc_matrix.no_connections")
def _():
    return U.connection_df_to_csc_matrix(_edge_series().iloc[:0], mirror=False, shape=(5, 5))


@case("randomization.connection_df_to_csc_matrix.from_per_node_weights")
def _():
    # the pipeline of random_geometric_model: weights of the candidate pairs, then the matrix
    idx = [np.array([1, 2]), np.array([0]), np.array([3]), np.array([2, 1])]
    w = U.evaluate_per_node_weights(np.arange(1, 5, dtype=float), np.arange(1, 5, dtype=float), idx)
    return {
        "no_mirror": U.connection_df_to_csc_matrix(w, mirror=False, shape=(4, 4)),
        "mirror": U.connection_df_to_csc_matrix(w, mirror=True, shape=(4, 4)),
    }


@case("randomization.connection_df_to_csc_matrix.shape_is_required")
def _():
    # the docstring says the shape is guessed when it is not given
    return U.connection_df_to_csc_matrix(_edge_series(), mirror=False)


@case("randomization.connection_df_to_csc_matrix.index_levels_must_be_named_neuron_and_i")
def _():
    return U.connection_df_to_csc_matrix(_edge_series(names=("a", "b")), mirror=False, shape=(4, 4))


# ----------------------------------------------------------------------------- direction_or_distance_dependent_w

PTS6 = positions(6, seed=1, scale=10.0)
IDX6 = [
    np.array([1, 2]),
    np.array([0]),
    np.array([4]),
    np.array([4, 5, 0]),
    np.array([3]),
    np.array([2, 3]),
]


@case("randomization.direction_or_distance_dependent_w.default_is_one_for_every_pair")
def _():
    return U.direction_or_distance_dependent_w(PTS6, PTS6, IDX6)


@case("randomization.direction_or_distance_dependent_w.distance_func")
def _():
    return {
        "square": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, distance_func=lambda d: d**2
        ),
        "decay": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, distance_func=lambda d: np.exp(-d / 5.0)
        ),
        "identity": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, distance_func=lambda d: d
        ),
    }


@case("randomization.direction_or_distance_dependent_w.directionality")
def _():
    # weight = (cos(angle between source - target and the axis) * fac + 1) * distance_func(distance)
    return {
        "x_positive": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, directionality_axis=[1, 0, 0], directionality_fac=0.5
        ),
        "y_negative": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, directionality_axis=[0, 1, 0], directionality_fac=-0.8
        ),
        "column_vector": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, directionality_axis=np.array([[0], [1], [0]]), directionality_fac=-1
        ),
        "factor_zero": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, directionality_axis=[1, 0, 0], directionality_fac=0.0
        ),
        "factor_outside_the_unit_interval": U.direction_or_distance_dependent_w(
            PTS6, PTS6, IDX6, directionality_axis=[1, 0, 0], directionality_fac=3
        ),
    }


@case("randomization.direction_or_distance_dependent_w.factor_without_axis_is_ignored")
def _():
    return U.direction_or_distance_dependent_w(PTS6, PTS6, IDX6, directionality_fac=0.7)


@case("randomization.direction_or_distance_dependent_w.axis_and_distance_func_multiply")
def _():
    # only random_geometric_model forbids the combination
    return U.direction_or_distance_dependent_w(
        PTS6,
        PTS6,
        IDX6,
        directionality_axis=[1, 0, 0],
        directionality_fac=0.5,
        distance_func=lambda d: d,
    )


@case("randomization.direction_or_distance_dependent_w.different_source_and_target_points")
def _():
    src = positions(3, seed=2, scale=10.0)
    return U.direction_or_distance_dependent_w(
        src,
        PTS6,
        [np.array([0, 1]), np.array([5]), np.array([2, 3, 4])],
        directionality_axis=[0, 0, 1],
        directionality_fac=0.5,
    )


@case("randomization.direction_or_distance_dependent_w.index_and_name_structure")
def _():
    out = U.direction_or_distance_dependent_w(PTS6, PTS6, IDX6)
    return {
        "index_names": list(out.index.names),
        "name": out.name,
        "dtype": str(out.dtype),
        "length": len(out),
        "index": out.index.tolist(),
    }


@case("randomization.direction_or_distance_dependent_w.sources_without_targets_are_missing")
def _():
    idx = [
        np.array([1, 2]),
        np.array([0]),
        np.array([], dtype=int),
        np.array([4, 5, 0]),
        np.array([3]),
        np.array([2, 3]),
    ]
    out = U.direction_or_distance_dependent_w(PTS6, PTS6, idx)
    return {"weights": out, "sources": sorted(set(out.index.get_level_values("neuron").tolist()))}


@case("randomization.direction_or_distance_dependent_w.target_lists_in_any_order_with_repeats")
def _():
    idx = [
        np.array([2, 1, 1]),
        np.array([0]),
        np.array([3, 0]),
        np.array([2, 1]),
        np.array([5]),
        np.array([0]),
    ]
    return U.direction_or_distance_dependent_w(PTS6, PTS6, idx, distance_func=lambda d: d)


@case("randomization.direction_or_distance_dependent_w.python_lists_as_target_lists")
def _():
    return U.direction_or_distance_dependent_w(
        PTS6, PTS6, [[1, 2], [0], [4], [4, 5, 0], [3], [2, 3]]
    )


@case("randomization.direction_or_distance_dependent_w.zero_distance_gives_nan_with_a_direction")
def _():
    # pairs of identical points: 0 / 0 in the cosine
    idx = [np.array([i]) for i in range(6)]
    return {
        "direction": U.direction_or_distance_dependent_w(
            PTS6, PTS6, idx, directionality_axis=[1, 0, 0], directionality_fac=0.3
        ),
        "no_direction": U.direction_or_distance_dependent_w(PTS6, PTS6, idx),
    }


@case("randomization.direction_or_distance_dependent_w.only_the_first_three_coordinates_count")
def _():
    p4 = positions(6, seed=1, dims=4, scale=10.0)
    return {
        "four_dimensions": U.direction_or_distance_dependent_w(
            p4, p4, IDX6, distance_func=lambda d: d
        ),
        "first_three": U.direction_or_distance_dependent_w(
            p4[:, :3], p4[:, :3], IDX6, distance_func=lambda d: d
        ),
    }


@case("randomization.direction_or_distance_dependent_w.two_dimensional_points")
def _():
    return U.direction_or_distance_dependent_w(PTS6[:, :2], PTS6[:, :2], IDX6)


@case("randomization.direction_or_distance_dependent_w.points_as_lists")
def _():
    return U.direction_or_distance_dependent_w(PTS6.tolist(), PTS6.tolist(), IDX6)


@case("randomization.direction_or_distance_dependent_w.no_pairs_at_all")
def _():
    return U.direction_or_distance_dependent_w(PTS6, PTS6, [])


# ----------------------------------------------------------------------------- evaluate_per_node_weights

IDX4 = [np.array([1, 2]), np.array([0]), np.array([3]), np.array([2, 1])]


@case("randomization.evaluate_per_node_weights.basic")
def _():
    return U.evaluate_per_node_weights(np.arange(1, 5, dtype=float), np.linspace(1, 2, 4), IDX4)


@case("randomization.evaluate_per_node_weights.structure_of_the_result")
def _():
    out = U.evaluate_per_node_weights(np.arange(1, 5, dtype=float), np.linspace(1, 2, 4), IDX4)
    return {
        "index_names": list(out.index.names),
        "name": out.name,
        "dtype": str(out.dtype),
        "index": out.index.tolist(),
    }


@case("randomization.evaluate_per_node_weights.integers_and_lists")
def _():
    return {
        "lists": U.evaluate_per_node_weights([1, 2, 3, 4], [4, 3, 2, 1], IDX4),
        "int_arrays": U.evaluate_per_node_weights(np.arange(1, 5), np.arange(1, 5), IDX4),
        "python_idx_lists": U.evaluate_per_node_weights(
            np.arange(1.0, 5.0), np.arange(1.0, 5.0), [[1, 2], [0], [3], [2, 1]]
        ),
    }


@case("randomization.evaluate_per_node_weights.sources_without_targets_are_missing")
def _():
    idx = [np.array([1, 2]), np.array([], dtype=int), np.array([3]), np.array([2, 1])]
    out = U.evaluate_per_node_weights(np.arange(1, 5, dtype=float), np.linspace(1, 2, 4), idx)
    return {"weights": out, "sources": sorted(set(out.index.get_level_values("neuron").tolist()))}


@case("randomization.evaluate_per_node_weights.empty_python_list_of_targets")
def _():
    return U.evaluate_per_node_weights(
        np.arange(1, 5, dtype=float), np.linspace(1, 2, 4), [[1, 2], [], [3], [2, 1]]
    )


@case("randomization.evaluate_per_node_weights.nan_and_zero_weights")
def _():
    return U.evaluate_per_node_weights(
        np.array([1.0, np.nan, 0.0, 2.0]), np.array([1.0, 0.0, 2.0, 3.0]), IDX4
    )


@case("randomization.evaluate_per_node_weights.more_sources_than_target_lists")
def _():
    # zip stops at the shorter of w_out and idx
    return U.evaluate_per_node_weights(np.arange(1, 7, dtype=float), np.linspace(1, 2, 4), IDX4)


@case("randomization.evaluate_per_node_weights.no_target_lists")
def _():
    return U.evaluate_per_node_weights(np.arange(1, 5, dtype=float), np.linspace(1, 2, 4), [])


@case("randomization.evaluate_per_node_weights.negative_target_index_wraps_around")
def _():
    return U.evaluate_per_node_weights(
        np.arange(1, 5, dtype=float), np.linspace(1, 2, 4), [np.array([-1])] * 4
    )
