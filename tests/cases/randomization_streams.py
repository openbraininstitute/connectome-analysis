"""Which random state each randomization function uses.

The functions that take a ``seed`` and build ``np.random.default_rng(seed)`` must leave the legacy global state alone;
``ER_shuffle`` (and so ``adjusted_ER``) re-seed it; ``random_geometric_model``, ``stochastic_spread_model`` and
``evaluate_probs*`` consume it.  The per-function ``stream_after`` cases pin the consumption of the second group;
this module pins that the first group does not touch it.
"""

import numpy as np
from support.cases import case
from support.data import digraph

from cases.randomization_common import R, U, reciprocal_graph, skeleta_graph


def _next_draws_after(call):
    np.random.seed(5)
    call()
    return np.random.rand(3)


@case("randomization.global_state.generator_based_functions_do_not_touch_it")
def _():
    adj = reciprocal_graph()
    coo = adj.tocoo()
    calls = {
        "reference": lambda: None,
        "configuration_model": lambda: R.configuration_model(adj, seed=1),
        "configuration_model_unseeded": lambda: R.configuration_model(adj),
        "underlying_model": lambda: R.underlying_model(adj, seed=1),
        "bishuffled_model": lambda: R.bishuffled_model(adj, seed=1),
        "add_rc_connections": lambda: R.add_rc_connections(adj, 3, seed=1),
        "add_rc_connections_skeleta": lambda: R.add_rc_connections_skeleta(
            skeleta_graph(), 2, threads=1, seed=1
        ),
        "add_connections_sparse": lambda: R.add_connections(digraph(20, 0.1, seed=1), 10, seed=1),
        "add_connections_dense": lambda: R.add_connections(
            digraph(20, 0.1, seed=1), 10, seed=1, sparse_mode=False
        ),
        "subsampled_matrix": lambda: U.subsampled_matrix(coo, 5, np.random.default_rng(1)),
        "non_overlapping_subsampled_matrices": lambda: U.non_overlapping_subsampled_matrices(
            coo, [3, 4], np.random.default_rng(1)
        ),
        "half_matrix": lambda: U.half_matrix(coo, np.random.default_rng(1)),
        "adjust_bidirectional_connections": lambda: U.adjust_bidirectional_connections(
            adj.tocsc(), 3, np.random.default_rng(1)
        ),
        "add_bidirectional_connections": lambda: U.add_bidirectional_connections(
            adj.tocsc(), 3, np.random.default_rng(1)
        ),
    }
    return {name: _next_draws_after(call) for name, call in calls.items()}


@case("randomization.global_state.cpp_generators_do_not_touch_it")
def _():
    xyz = np.random.RandomState(0).random_sample((20, 3)) * 50
    calls = {
        "reference": lambda: None,
        "run_ER": lambda: R.run_ER(20, 0.3, threads=2, seed=(1, 2)),
        "run_ER_unseeded": lambda: R.run_ER(20, 0.3, threads=2),
        "run_DD2": lambda: R.run_DD2(20, 0.5, 0.02, xyz, threads=2, seed=(1, 2)),
        "ER_model": lambda: R.ER_model(digraph(20, 0.2, seed=1), threads=2, seed=3),
    }
    return {name: _next_draws_after(call) for name, call in calls.items()}


@case("randomization.global_state.er_shuffle_reseeds_it")
def _():
    # a seeded ER_shuffle makes the global state a function of the seed alone (whatever it was before)
    adj = digraph(20, 0.2, seed=1)

    def after(before_seed):
        np.random.seed(before_seed)
        R.ER_shuffle(adj, seed=9)
        return np.random.rand(3)

    return {"state_before_1": after(1), "state_before_2": after(2)}
