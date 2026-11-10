"""Triad motifs: ``count_triads_fully_connected`` and the class tables it relies on.

The function draws its sample from the legacy global generator (``np.random.choice``), also when everything is classified,
so every case seeds ``np.random`` right before the call and some return ``np.random.rand(3)`` afterwards to pin how much of
the stream was consumed.
"""

import itertools

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import tutorial_connectome, weighted

from cases.topology_helpers import complete, cycle, from_edges, triad_graph
from connalysis.network import topology as T


def _count(adj, seed=1, **kwargs):
    np.random.seed(seed)
    return T.count_triads_fully_connected(adj, **kwargs)


# ----------------------------------------------------------------------------- class tables


@case("topology.triad_tables.dictionaries_and_arrays")
def _():
    return {
        "label_edges": T.label_edges,
        "connected_triads": T.connected_triads,
        "triad_dict": T.triad_dict,
        "triad_combinations": T.triad_combinations,
    }


@case("topology.triad_tables.triad_id_of_each_class")
def _():
    return {i: T._get_triad_id(edges) for i, edges in T.connected_triads.items()}


# ----------------------------------------------------------------------------- count_triads_fully_connected


@case("topology.count_triads_fully_connected.all_thirteen_classes")
def _():
    # 578 connected triads of a random graph with reciprocal pairs and planted 3-cycles of both orientations
    return _count(triad_graph())


@case("topology.count_triads_fully_connected.normalized")
def _():
    return _count(triad_graph(), return_normalized=True)


@case("topology.count_triads_fully_connected.random_stream_after_unsampled_call")
def _():
    result = _count(triad_graph(), seed=7)
    return result, np.random.rand(3)


@case("topology.count_triads_fully_connected.sampled")
def _():
    adj = triad_graph()
    return {
        (seed, size): _count(adj, seed=seed, max_num_sampled=size)
        for seed in (1, 2, 3)
        for size in (50, 200)
    }


@case("topology.count_triads_fully_connected.sampled_normalized")
def _():
    adj = triad_graph()
    return {
        seed: _count(adj, seed=seed, max_num_sampled=120, return_normalized=True) for seed in (4, 5)
    }


@case("topology.count_triads_fully_connected.random_stream_after_sampled_call")
def _():
    result = _count(triad_graph(), seed=7, max_num_sampled=60)
    return result, np.random.rand(3)


@case("topology.count_triads_fully_connected.different_seeds_give_different_samples")
def _():
    adj = triad_graph()
    return [_count(adj, seed=seed, max_num_sampled=40) for seed in (11, 12, 13, 14)]


@case("topology.count_triads_fully_connected.sample_larger_than_the_number_of_triads")
def _():
    adj = triad_graph()
    return [
        _count(adj, max_num_sampled=10**9),
        _count(adj, max_num_sampled=578),
        _count(adj, max_num_sampled=577),
    ]


@case("topology.count_triads_fully_connected.sample_of_one_triad")
def _():
    adj = triad_graph()
    return [_count(adj, seed=seed, max_num_sampled=1) for seed in (1, 2, 3)]


@case("topology.count_triads_fully_connected.sample_of_zero_triads")
def _():
    return _count(triad_graph(), max_num_sampled=0)


@case("topology.count_triads_fully_connected.global_seed_is_what_matters")
def _():
    # same global seed -> same sample, whatever happened to the generator before
    adj = triad_graph()
    np.random.seed(5)
    np.random.rand(100)
    np.random.seed(9)
    first = T.count_triads_fully_connected(adj, max_num_sampled=70)
    np.random.seed(9)
    second = T.count_triads_fully_connected(adj, max_num_sampled=70)
    return first, second


@case("topology.count_triads_fully_connected.every_class_under_every_labelling")
def _():
    # a single triad of class i, with its nodes numbered in each of the 6 possible ways, is counted once in class i
    out = {}
    np.random.seed(3)
    for i, edges in T.connected_triads.items():
        for perm in itertools.permutations(range(3)):
            dense = np.zeros((3, 3), dtype=bool)
            for a, b in edges:
                dense[perm[a], perm[b]] = True
            out[(i, perm)] = T.count_triads_fully_connected(sp.csr_matrix(dense))
    return out


@case("topology.count_triads_fully_connected.four_nodes_with_overlapping_triads")
def _():
    # a reciprocal triangle plus a pendant node: triads {0,1,2} (class 12), {0,1,3}, {0,2,3}, {1,2,3}
    adj = from_edges(4, [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (2, 3)])
    return [_count(adj), _count(adj, return_normalized=True)]


@case("topology.count_triads_fully_connected.complete_and_cycle_graphs")
def _():
    return {
        "complete_5": _count(complete(5)),
        "complete_5_normalized": _count(complete(5), return_normalized=True),
        "cycle_5": _count(cycle(5)),
        "cycle_3": _count(cycle(3)),
        "cycle_3_normalized": _count(cycle(3), return_normalized=True),
    }


@case("topology.count_triads_fully_connected.sparse_formats")
def _():
    adj = triad_graph()
    return {
        fmt: _count(adj.asformat(fmt), max_num_sampled=100)
        for fmt in ("csr", "csc", "coo", "lil", "dok")
    }


@case("topology.count_triads_fully_connected.sparse_array")
def _():
    return _count(sp.csr_array(triad_graph()))


@case("topology.count_triads_fully_connected.integer_and_float_matrices")
def _():
    adj = triad_graph()
    return {"int": _count(adj.astype(int)), "float": _count(adj.astype(float)), "bool": _count(adj)}


@case("topology.count_triads_fully_connected.dense_array_is_not_accepted")
def _():
    return _count(triad_graph().toarray())


@case("topology.count_triads_fully_connected.no_connected_triad")
def _():
    # two disjoint reciprocal pairs have no triad: dividing by the empty sample fails
    return _count(from_edges(4, [(0, 1), (1, 0), (2, 3), (3, 2)]))


@case("topology.count_triads_fully_connected.empty_graph")
def _():
    return _count(from_edges(6, []))


@case("topology.count_triads_fully_connected.weights_break_the_classification")
def _():
    return _count(weighted(20, 0.15, seed=4))


@case("topology.count_triads_fully_connected.diagonal_breaks_the_classification")
def _():
    adj = triad_graph().tolil()
    adj[0, 0] = True
    return _count(adj.tocsr())


@case("topology.count_triads_fully_connected.input_is_not_modified")
def _():
    adj = triad_graph()
    before = adj.copy()
    result = _count(adj)
    return result, (adj != before).nnz


@case("topology.count_triads_fully_connected.real_connectome")
def _():
    adj, _ = tutorial_connectome(120)
    return {
        "unsampled": _count(adj),
        "sampled": _count(adj, seed=2, max_num_sampled=500),
        "sampled_normalized": _count(adj, seed=2, max_num_sampled=500, return_normalized=True),
    }
