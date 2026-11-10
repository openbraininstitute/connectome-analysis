"""stochastic_spread_model (uses ``evaluate_probs`` / ``evaluate_probs_less_random``).

Every step draws from the legacy global ``np.random`` state, so each case seeds it right before the call.
The output is a boolean csr matrix; ``return_history`` adds the list of the mean number of new targets per step.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, tutorial_connectome, weighted

from cases.randomization_common import R

M = digraph(30, 0.15, seed=2)
W = weighted(30, 0.15, seed=2, integers=False)  # probabilities in (0, 1]


def _ssm(m=None, seed=3, **kw):
    np.random.seed(seed)
    return R.stochastic_spread_model(M if m is None else m, **kw)


def _stream_after(m=None, seed=3, **kw):
    np.random.seed(seed)
    R.stochastic_spread_model(M if m is None else m, **kw)
    return np.random.rand(3)


def _complete(n):
    return sp.csr_matrix(np.ones((n, n), dtype=bool) ^ np.eye(n, dtype=bool))


# ----------------------------------------------------------------------------- defaults and the scaling of probabilities


@case("randomization.stochastic_spread_model.defaults")
def _():
    # 100 steps, q = 10, individual targets: the spread dies out long before the last step
    return _ssm()


@case("randomization.stochastic_spread_model.defaults_with_history")
def _():
    return _ssm(return_history=True)


@case("randomization.stochastic_spread_model.bool_matrix_with_q")
def _():
    return _ssm(n_steps=10, q=3.0)


@case("randomization.stochastic_spread_model.tgt_level_mean")
def _():
    return {
        "mean": _ssm(n_steps=10, q=3.0, tgt_level="mean"),
        "individual": _ssm(n_steps=10, q=3.0, tgt_level="individual"),
    }


@case("randomization.stochastic_spread_model.decay")
def _():
    return {d: _ssm(n_steps=6, q=3.0, decay=d) for d in (1.0, 0.8, 0.5, 0.0)}


@case("randomization.stochastic_spread_model.q_values")
def _():
    return {
        "huge": _ssm(n_steps=3, q=1000.0),
        "small": _ssm(n_steps=6, q=0.5),
        "zero": _ssm(n_steps=3, q=0.0),
        "negative": _ssm(n_steps=3, q=-1.0),
    }


@case("randomization.stochastic_spread_model.float_matrix_without_q")
def _():
    # the weights are the probabilities of crossing an edge
    return _ssm(W, n_steps=10, q=None)


@case("randomization.stochastic_spread_model.float_matrix_with_q")
def _():
    return {
        "individual": _ssm(W, n_steps=10, q=2.0),
        "mean": _ssm(W, n_steps=10, q=2.0, tgt_level="mean"),
    }


@case(
    "randomization.stochastic_spread_model.probabilities_one_make_the_spread_deterministic_per_step"
)
def _():
    return _ssm(M.astype(float), n_steps=5, q=None)


@case("randomization.stochastic_spread_model.n_steps_values")
def _():
    return {k: _ssm(n_steps=k, q=3.0) for k in (0, 1, 2, 5)}


@case("randomization.stochastic_spread_model.history_per_step")
def _():
    out, history = _ssm(n_steps=12, q=3.0, return_history=True)
    return {
        "graph": out,
        "history": history,
        "length": len(history),
        "type": type(history).__name__,
    }


@case("randomization.stochastic_spread_model.history_without_steps")
def _():
    return _ssm(n_steps=0, q=3.0, return_history=True)


@case("randomization.stochastic_spread_model.history_with_float_probabilities")
def _():
    return _ssm(W, n_steps=10, q=None, return_history=True)


# ----------------------------------------------------------------------------- protection, exclusion, who can spread


@case("randomization.stochastic_spread_model.n_protected")
def _():
    # the first steps use the "less random" rule: exactly the expected number of targets
    return {k: _ssm(n_steps=6, q=3.0, n_protected=k) for k in (0, 1, 3)}


@case("randomization.stochastic_spread_model.n_protected_with_mean_level")
def _():
    return _ssm(n_steps=6, q=3.0, n_protected=3, tgt_level="mean")


@case("randomization.stochastic_spread_model.n_protected_equals_n_steps")
def _():
    return _ssm(n_steps=4, q=3.0, n_protected=4)


@case("randomization.stochastic_spread_model.n_protected_larger_than_n_steps")
def _():
    return _ssm(n_steps=4, q=3.0, n_protected=100)


@case("randomization.stochastic_spread_model.n_protected_with_float_probabilities")
def _():
    return {
        "q_none": _ssm(W, n_steps=6, q=None, n_protected=2),
        "q": _ssm(W, n_steps=6, q=2.0, n_protected=2),
    }


@case("randomization.stochastic_spread_model.n_protected_on_a_complete_graph")
def _():
    return {
        "plain": _ssm(_complete(8), n_steps=5, q=2.0),
        "protected": _ssm(_complete(8), n_steps=5, q=2.0, n_protected=2),
    }


@case("randomization.stochastic_spread_model.sum_exclusion")
def _():
    # True: a node that was a candidate once is never a target later; False: only the previous step's sources are excluded
    return {
        "true": _ssm(n_steps=10, q=3.0, sum_exclusion=True),
        "false": _ssm(n_steps=10, q=3.0, sum_exclusion=False),
    }


@case("randomization.stochastic_spread_model.sum_exclusion_truthy_values")
def _():
    return {
        "zero": _ssm(n_steps=5, q=3.0, sum_exclusion=0),
        "one": _ssm(n_steps=5, q=3.0, sum_exclusion=1),
        "none": _ssm(n_steps=5, q=3.0, sum_exclusion=None),
    }


@case("randomization.stochastic_spread_model.node_can_spread")
def _():
    even = np.arange(30) % 2 == 0
    return {
        "numpy_bool": _ssm(n_steps=5, q=3.0, node_can_spread=even),
        "list_of_bool": _ssm(n_steps=5, q=3.0, node_can_spread=[bool(i % 2) for i in range(30)]),
        "tuple": _ssm(n_steps=5, q=3.0, node_can_spread=tuple([True] * 15 + [False] * 15)),
        "integers": _ssm(n_steps=5, q=3.0, node_can_spread=(np.arange(30) % 3).tolist()),
        "all_true": _ssm(n_steps=5, q=3.0, node_can_spread=np.ones(30, dtype=bool)),
    }


@case("randomization.stochastic_spread_model.node_can_spread_nobody")
def _():
    # no source can spread: empty graph
    return _ssm(n_steps=3, q=3.0, node_can_spread=[False] * 30)


@case("randomization.stochastic_spread_model.node_can_spread_nobody_with_protection")
def _():
    return _ssm(n_steps=3, q=3.0, node_can_spread=[False] * 30, n_protected=1)


@case("randomization.stochastic_spread_model.node_can_spread_with_history")
def _():
    return _ssm(n_steps=6, q=3.0, node_can_spread=np.arange(30) % 2 == 0, return_history=True)


# ----------------------------------------------------------------------------- inputs


@case("randomization.stochastic_spread_model.input_formats")
def _():
    # the formats give the same stored order of entries after the transpose only for some of them; the draws follow it
    return {fmt: _ssm(M.asformat(fmt), n_steps=4, q=3.0) for fmt in ("csr", "csc", "coo", "lil")}


@case("randomization.stochastic_spread_model.input_formats_float")
def _():
    return {fmt: _ssm(W.asformat(fmt), n_steps=4, q=None) for fmt in ("csr", "csc", "coo")}


@case("randomization.stochastic_spread_model.dtypes")
def _():
    return {
        "uint8": _ssm(M.astype(np.uint8), n_steps=4, q=2.0),
        "int64": _ssm(M.astype(np.int64), n_steps=4, q=2.0),
        "float32_q": _ssm(W.astype(np.float32), n_steps=4, q=2.0),
        "int_without_q": _ssm(M.astype(int), n_steps=4, q=None),
    }


@case("randomization.stochastic_spread_model.explicit_zero_weights")
def _():
    w = W.copy()
    w.data[:3] = 0.0
    return _ssm(w, n_steps=6, q=None)


@case("randomization.stochastic_spread_model.loops_in_the_matrix_are_never_followed")
def _():
    loops = sp.csr_matrix(np.eye(6, dtype=bool) | np.diag(np.ones(5, dtype=bool), 1))
    return _ssm(loops, n_steps=4, q=2.0)


@case("randomization.stochastic_spread_model.tiny_graphs")
def _():
    return {
        "one_node": _ssm(sp.csr_matrix(([True], ([0], [0])), shape=(1, 1)), n_steps=3, q=2.0),
        "two_nodes": _ssm(sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool)), n_steps=3, q=2.0),
        "two_nodes_mean": _ssm(
            sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool)),
            n_steps=3,
            q=2.0,
            tgt_level="mean",
        ),
        "complete_8": _ssm(_complete(8), n_steps=5, q=2.0),
    }


@case("randomization.stochastic_spread_model.empty_graph")
def _():
    empty = sp.csr_matrix((10, 10), dtype=bool)
    return {
        "individual": _ssm(empty, n_steps=5, q=3.0),
        "mean": _ssm(empty, n_steps=5, q=3.0, tgt_level="mean"),
        "float": _ssm(sp.csr_matrix((10, 10)), n_steps=5, q=None),
    }


@case("randomization.stochastic_spread_model.empty_graph_with_protection")
def _():
    # the "less random" step has nothing to pick from: np.hstack([]) fails
    return _ssm(sp.csr_matrix((10, 10), dtype=bool), n_steps=5, q=3.0, n_protected=2)


@case("randomization.stochastic_spread_model.empty_graph_mean_level_with_protection")
def _():
    return _ssm(
        sp.csr_matrix((10, 10), dtype=bool), n_steps=3, q=3.0, tgt_level="mean", n_protected=1
    )


@case("randomization.stochastic_spread_model.protection_with_a_tiny_q")
def _():
    # every expected count rounds to 0 targets: np.hstack([]) fails here too
    return _ssm(n_steps=5, q=0.01, n_protected=3)


@case("randomization.stochastic_spread_model.dense_input_rejected")
def _():
    return _ssm(M.toarray(), n_steps=3, q=3.0)


# ----------------------------------------------------------------------------- random stream


@case("randomization.stochastic_spread_model.stream_after_default")
def _():
    return _stream_after(n_steps=10, q=3.0)


@case("randomization.stochastic_spread_model.stream_after_mean_level")
def _():
    return _stream_after(n_steps=10, q=3.0, tgt_level="mean")


@case("randomization.stochastic_spread_model.stream_after_n_protected")
def _():
    return _stream_after(n_steps=6, q=3.0, n_protected=3)


@case("randomization.stochastic_spread_model.stream_after_float_probabilities")
def _():
    return _stream_after(W, n_steps=10, q=None)


@case("randomization.stochastic_spread_model.seeds_change_the_graph")
def _():
    return {s: _ssm(seed=s, n_steps=6, q=3.0) for s in (0, 1, 2)}


# ----------------------------------------------------------------------------- invalid arguments


@case("randomization.stochastic_spread_model.float_weights_above_one")
def _():
    return _ssm(weighted(30, 0.15, seed=2), n_steps=3, q=None)


@case("randomization.stochastic_spread_model.int_weights_above_one_with_q")
def _():
    return _ssm(weighted(30, 0.15, seed=2).astype(int), n_steps=3, q=3.0)


@case("randomization.stochastic_spread_model.bool_matrix_needs_q")
def _():
    return _ssm(n_steps=3, q=None)


@case("randomization.stochastic_spread_model.unknown_tgt_level")
def _():
    return _ssm(n_steps=3, q=3.0, tgt_level="Mean")


@case("randomization.stochastic_spread_model.decay_above_one")
def _():
    return _ssm(n_steps=3, q=3.0, decay=1.1)


@case("randomization.stochastic_spread_model.decay_below_zero")
def _():
    return _ssm(n_steps=3, q=3.0, decay=-0.1)


@case("randomization.stochastic_spread_model.node_can_spread_of_wrong_length")
def _():
    return _ssm(n_steps=3, q=3.0, node_can_spread=[True] * 5)


# ----------------------------------------------------------------------------- real data


@case("randomization.stochastic_spread_model.real_data")
def _():
    adj, _ = tutorial_connectome(300)
    return _ssm(adj, n_steps=8, q=3.0)


@case("randomization.stochastic_spread_model.real_data_protected_mean_level_with_decay")
def _():
    adj, _ = tutorial_connectome(300)
    return _ssm(
        adj, n_steps=8, q=3.0, tgt_level="mean", decay=0.9, n_protected=2, return_history=True
    )


@case("randomization.stochastic_spread_model.real_data_float_probabilities")
def _():
    adj, _ = tutorial_connectome(200)
    probs = sp.csr_matrix(
        (
            np.random.RandomState(0).random_sample(adj.nnz) * 0.3 + 0.02,
            adj.indices.copy(),
            adj.indptr.copy(),
        ),
        shape=adj.shape,
    )
    return _ssm(probs, n_steps=8, q=None)
