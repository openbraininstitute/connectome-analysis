"""Random graph generators backed by the bigrandomgraphs C++ library: run_ER ... run_DD2_block, ER_model, run_DD2_model.

The C++ library (bigrandomgraphs 0.1.1) seeds one PCG32 stream per thread (seed1 + 2354 i, seed2 + 5223 i) and deals the
rows round-robin to the threads, so a seeded result depends on ``threads`` (not on the machine).  For SBM/DD2/DD3/
DD2_block(_pre) the last thread, the one that runs on the caller, takes both of its seeds from seed1: with
``threads=1`` the second seed is ignored (ER is not affected).  Seeded cases therefore use ``threads=2``, where both
seeds matter.  Cases that must not depend on the stream use probabilities that are 0 or >= 1 (a step in the distance
dependence), which makes the graph deterministic even without a seed.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import tutorial_connectome, weighted

from cases.randomization_common import (
    CPP,
    RR,
    STEP_A,
    STEP_B,
    R,
    clusters,
    depth_values,
    digraph,
    moderate_points,
    shape_info,
)

# ----------------------------------------------------------------------------- run_ER


@case("randomization.run_ER.seeded", tags=CPP)
def _():
    return R.run_ER(30, 0.2, threads=2, seed=(1, 2))


@case("randomization.run_ER.single_thread", tags=CPP)
def _():
    return R.run_ER(30, 0.2, threads=1, seed=(1, 2))


@case("randomization.run_ER.default_threads", tags=CPP)
def _():
    return R.run_ER(40, 0.15, seed=(5, 6))


@case("randomization.run_ER.thread_count_changes_the_graph", tags=CPP)
def _():
    # rows are dealt to the threads round-robin and every thread has its own stream: same seed, different graph
    return {t: R.run_ER(30, 0.2, threads=t, seed=(1, 2)) for t in (1, 2, 3, 8)}


@case("randomization.run_ER.seed_values", tags=CPP)
def _():
    # 0 is a valid seed (only None means "unseeded")
    return {s: R.run_ER(20, 0.3, threads=2, seed=s) for s in [(0, 0), (0, 1), (1, 0), (2**40, 7)]}


@case("randomization.run_ER.seed_as_array", tags=CPP)
def _():
    return R.run_ER(20, 0.3, threads=2, seed=np.array([4, 5]))


@case("randomization.run_ER.seed_as_list", tags=CPP)
def _():
    return R.run_ER(20, 0.3, threads=2, seed=[4, 5])


@case("randomization.run_ER.p_zero")
def _():
    return R.run_ER(8, 0, threads=3, seed=(1, 2))


@case("randomization.run_ER.p_one")
def _():
    return R.run_ER(6, 1.0, threads=3, seed=(1, 2))


@case("randomization.run_ER.docstring_example")
def _():
    # run_ER(3, 1): the complete digraph on 3 vertices (the docstring shows a dict, the function returns a coo matrix)
    return R.run_ER(3, 1)


@case("randomization.run_ER.p_zero_unseeded_default_threads")
def _():
    return R.run_ER(8, 0)


@case("randomization.run_ER.tiny_n", tags=CPP)
def _():
    return {n: R.run_ER(n, 0.5, threads=2, seed=(1, 2)) for n in (0, 1, 2, 3)}


@case("randomization.run_ER.more_threads_than_vertices", tags=CPP)
def _():
    # extra threads (here 8 for 4 vertices) have no rows; the result is the same for any larger thread count
    return {t: R.run_ER(4, 0.5, threads=t, seed=(1, 2)) for t in (1, 4, 8, 64)}


@case("randomization.run_ER.unseeded_properties")
def _():
    # without a (complete) seed the graph is random: only shape, dtype and the absence of loops can be pinned
    calls = {
        "omitted": lambda: R.run_ER(30, 0.3, threads=2),
        "none_pair": lambda: R.run_ER(30, 0.3, threads=2, seed=(None, None)),
        "first_only": lambda: R.run_ER(30, 0.3, threads=2, seed=(3, None)),
        "second_only": lambda: R.run_ER(30, 0.3, threads=2, seed=(None, 3)),
    }
    out = {}
    for name, call in calls.items():
        m = call()
        out[name] = {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 30 * 29)}
    return out


@case("randomization.run_ER.p_negative")
def _():
    return R.run_ER(5, -0.1, threads=1, seed=(1, 2))


@case("randomization.run_ER.p_above_one")
def _():
    return R.run_ER(5, 1.1, threads=1, seed=(1, 2))


@case("randomization.run_ER.int_seed_rejected")
def _():
    # the seed is a pair; run_ER (unlike ER_model) does not turn an int into one
    return R.run_ER(5, 0.5, threads=1, seed=3)


@case("randomization.run_ER.zero_threads")
def _():
    return R.run_ER(5, 0.5, threads=0, seed=(1, 2))


# ----------------------------------------------------------------------------- ER_model


def _er_input(**kw):
    return digraph(30, 0.15, seed=4, **kw)


@case("randomization.ER_model.int_seed", tags=CPP)
def _():
    return R.ER_model(_er_input(), threads=2, seed=5)


@case("randomization.ER_model.tuple_seed", tags=CPP)
def _():
    # an int seed s means the pair (s, s)
    adj = _er_input()
    return {
        "int": R.ER_model(adj, threads=2, seed=5),
        "same_pair": R.ER_model(adj, threads=2, seed=(5, 5)),
        "other_pair": R.ER_model(adj, threads=2, seed=(5, 6)),
    }


@case("randomization.ER_model.default_threads", tags=CPP)
def _():
    return R.ER_model(_er_input(), seed=5)


@case("randomization.ER_model.input_formats", tags=CPP)
def _():
    adj = _er_input()
    return {
        name: R.ER_model(m, threads=2, seed=(1, 2))
        for name, m in [
            ("csr", adj),
            ("csc", adj.tocsc()),
            ("coo", adj.tocoo()),
            ("lil", adj.tolil()),
            ("dense", adj.toarray()),
            ("int", adj.astype(int)),
            ("float", adj.astype(float)),
        ]
    }


@case("randomization.ER_model.weights_do_not_matter", tags=CPP)
def _():
    # p is the fraction of non-zero entries among the n (n - 1) off-diagonal pairs
    return R.ER_model(weighted(30, 0.15, seed=4), threads=2, seed=5)


@case("randomization.ER_model.complete_and_empty")
def _():
    full = sp.csr_matrix(np.ones((6, 6)) - np.eye(6))
    return {
        "complete": R.ER_model(full, threads=2, seed=3),
        "empty": R.ER_model(sp.csr_matrix((6, 6)), threads=2, seed=3),
    }


@case("randomization.ER_model.two_nodes", tags=CPP)
def _():
    return R.ER_model(sp.csr_matrix(np.array([[0, 1], [0, 0]])), threads=1, seed=(1, 2))


@case("randomization.ER_model.unseeded_properties")
def _():
    m = R.ER_model(_er_input(), threads=2)
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 30 * 29)}


@case("randomization.ER_model.not_square")
def _():
    return R.ER_model(sp.csr_matrix((4, 5)), threads=1, seed=1)


@case("randomization.ER_model.diagonal_pushes_p_above_one")
def _():
    # p = nnz / (n (n - 1)) counts diagonal entries: a full matrix gives p > 1
    return R.ER_model(sp.csr_matrix(np.ones((6, 6))), threads=1, seed=1)


@case("randomization.ER_model.single_node")
def _():
    # p = 0 / 0 is nan, which fails the 0 <= p <= 1 assertion of run_ER
    return R.ER_model(sp.csr_matrix((1, 1)), threads=1, seed=1)


@case("randomization.ER_model.numpy_int_seed_rejected")
def _():
    # only a python int is turned into a pair; a numpy integer is indexed as if it were one
    return R.ER_model(_er_input(), threads=1, seed=np.int64(5))


# ----------------------------------------------------------------------------- run_SBM

_PROBS = np.array([[0.1, 0.8, 0.3], [0.5, 0.2, 0.9], [0.0, 0.4, 0.6]])


def _blocks(n=24, k=3, seed=0):
    return np.random.RandomState(seed).randint(0, k, size=n)


@case("randomization.run_SBM.seeded", tags=CPP)
def _():
    return R.run_SBM(24, _PROBS, _blocks(), threads=2, seed=(3, 4))


@case("randomization.run_SBM.default_threads", tags=CPP)
def _():
    return R.run_SBM(24, _PROBS, _blocks(), seed=(3, 4))


@case("randomization.run_SBM.thread_count_changes_the_graph", tags=CPP)
def _():
    return {t: R.run_SBM(24, _PROBS, _blocks(), threads=t, seed=(3, 4)) for t in (1, 2, 3, 8)}


@case("randomization.run_SBM.second_seed", tags=CPP)
def _():
    # with one thread the second seed is ignored (C++ passes seed1 twice); with two threads it matters
    return {
        "one_thread_seed2=4": R.run_SBM(24, _PROBS, _blocks(), threads=1, seed=(3, 4)),
        "one_thread_seed2=5": R.run_SBM(24, _PROBS, _blocks(), threads=1, seed=(3, 5)),
        "two_threads_seed2=4": R.run_SBM(24, _PROBS, _blocks(), threads=2, seed=(3, 4)),
        "two_threads_seed2=5": R.run_SBM(24, _PROBS, _blocks(), threads=2, seed=(3, 5)),
    }


@case("randomization.run_SBM.block_dtypes", tags=CPP)
def _():
    b = _blocks()
    return {
        name: R.run_SBM(24, _PROBS, blk, threads=2, seed=(3, 4))
        for name, blk in [
            ("int64", b),
            ("uint8", b.astype(np.uint8)),
            ("int32", b.astype(np.int32)),
            ("list", b.tolist()),
            ("tuple", tuple(b.tolist())),
        ]
    }


@case("randomization.run_SBM.probs_as_list", tags=CPP)
def _():
    return R.run_SBM(24, _PROBS.tolist(), _blocks(), threads=2, seed=(3, 4))


@case("randomization.run_SBM.deterministic_blocks")
def _():
    # probabilities 0 and 1: complete inside the blocks, nothing between them, whatever the seed
    blocks = np.array([0, 1, 0, 1, 2, 2, 0, 1])
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    return {
        "seeded": R.run_SBM(8, probs, blocks, threads=2, seed=(1, 2)),
        "unseeded": R.run_SBM(8, probs, blocks),
    }


@case("randomization.run_SBM.bipartite_pattern")
def _():
    # the docstring example: even to odd (and back) with high probability; here with probability 1 and 0 so it is exact
    return R.run_SBM(4, np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1, 0, 1]))


@case("randomization.run_SBM.probabilities_outside_unit_interval")
def _():
    # the C++ code just compares a uniform draw with the entry: <= 0 never, >= 1 always
    probs = np.array([[-0.5, 3.0], [3.0, -0.5]])
    return R.run_SBM(6, probs, np.array([0, 1, 0, 1, 0, 1]), threads=2, seed=(1, 2))


@case("randomization.run_SBM.unseeded_properties")
def _():
    m = R.run_SBM(24, _PROBS, _blocks(), threads=2)
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 24 * 23)}


@case("randomization.run_SBM.float_blocks_rejected")
def _():
    return R.run_SBM(24, _PROBS, _blocks().astype(float), threads=1, seed=(3, 4))


@case("randomization.run_SBM.more_labels_than_vertices", tags=CPP)
def _():
    # only the first n labels are used
    return R.run_SBM(12, _PROBS, _blocks(24), threads=2, seed=(3, 4))


@case("randomization.run_SBM.int_seed_rejected")
def _():
    return R.run_SBM(24, _PROBS, _blocks(), threads=1, seed=3)


# ----------------------------------------------------------------------------- run_DD2


@case("randomization.run_DD2.seeded", tags=CPP)
def _():
    return R.run_DD2(30, 0.5, 0.02, moderate_points(30), threads=2, seed=(7, 8))


@case("randomization.run_DD2.default_threads", tags=CPP)
def _():
    return R.run_DD2(30, 0.5, 0.02, moderate_points(30), seed=(7, 8))


@case("randomization.run_DD2.thread_count_changes_the_graph", tags=CPP)
def _():
    xyz = moderate_points(30)
    return {t: R.run_DD2(30, 0.5, 0.02, xyz, threads=t, seed=(7, 8)) for t in (1, 2, 3, 8)}


@case("randomization.run_DD2.second_seed", tags=CPP)
def _():
    xyz = moderate_points(30)
    return {
        "one_thread_seed2=8": R.run_DD2(30, 0.5, 0.02, xyz, threads=1, seed=(7, 8)),
        "one_thread_seed2=9": R.run_DD2(30, 0.5, 0.02, xyz, threads=1, seed=(7, 9)),
        "two_threads_seed2=8": R.run_DD2(30, 0.5, 0.02, xyz, threads=2, seed=(7, 8)),
        "two_threads_seed2=9": R.run_DD2(30, 0.5, 0.02, xyz, threads=2, seed=(7, 9)),
    }


@case("randomization.run_DD2.exponent_signs", tags=CPP)
def _():
    # b < 0 makes the probability grow with distance (the tutorial notebook uses a negative b)
    xyz = moderate_points(30)
    return {
        "decaying": R.run_DD2(30, 0.9, 0.01, xyz, threads=2, seed=(7, 8)),
        "growing": R.run_DD2(30, 0.01, -0.02, xyz, threads=2, seed=(7, 8)),
        "flat": R.run_DD2(30, 0.2, 0.0, xyz, threads=2, seed=(7, 8)),
    }


@case("randomization.run_DD2.coordinate_types", tags=CPP)
def _():
    xyz = moderate_points(30)
    return {
        name: R.run_DD2(30, 0.5, 0.02, c, threads=2, seed=(7, 8))
        for name, c in [
            ("float64", xyz),
            ("float32", xyz.astype(np.float32)),
            ("list", xyz.tolist()),
            ("int", np.round(xyz).astype(int)),
            ("fortran_order", np.asfortranarray(xyz)),
        ]
    }


@case("randomization.run_DD2.step_in_distance")
def _():
    # probability 1 within a cluster and 0 between clusters: deterministic, with or without seeds
    xyz = clusters()
    return {
        "unseeded": R.run_DD2(12, STEP_A, STEP_B, xyz),
        "seeded": R.run_DD2(12, STEP_A, STEP_B, xyz, threads=3, seed=(1, 2)),
    }


@case("randomization.run_DD2.zero_scale")
def _():
    return R.run_DD2(12, 0.0, STEP_B, clusters(), threads=2, seed=(1, 2))


@case("randomization.run_DD2.distance_is_euclidean")
def _():
    # p(d) = a exp(-b d) with a = exp(300 * 1.5), b = 300: >= 1 for d <= 1.5 and ~1e-30 at d = 1.73.
    # Pair (0, 1) is offset by (1, 1, 0): Euclidean 1.41 (connected; its L1 distance 2 or squared distance 2 would not be);
    # pair (2, 3) by (1, 1, 1): Euclidean 1.73 (not connected; its max-norm distance 1 would be).
    xyz = np.array([[0.0, 0, 0], [1.0, 1, 0], [100.0, 0, 0], [101.0, 1, 1]])
    return R.run_DD2(4, np.exp(450.0), 300.0, xyz, threads=1, seed=(1, 2))


@case("randomization.run_DD2.unseeded_properties")
def _():
    m = R.run_DD2(30, 0.5, 0.02, moderate_points(30), threads=2)
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 30 * 29)}


@case("randomization.run_DD2.int_seed_rejected")
def _():
    return R.run_DD2(30, 0.5, 0.02, moderate_points(30), threads=1, seed=3)


# ----------------------------------------------------------------------------- run_DD2_model


def _cluster_table():
    """10 nodes: A (4) at the origin, B (3) 1000 above A in z, C (3) 1000 away from A in x."""
    xyz = np.vstack(
        [
            clusters((4,), seed=1),
            clusters((3,), seed=2) + [0, 0, 1000.0],
            clusters((3,), seed=3) + [1000.0, 0, 0],
        ]
    )
    return pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "w": np.arange(10.0)})


_STEP_PARAMS = pd.DataFrame({"exp_model_scale": [STEP_A], "exp_model_exponent": [STEP_B]})


def _empty_adj(n=10):
    return sp.csr_matrix((n, n), dtype=bool)


@case("randomization.run_DD2_model.precomputed_params")
def _():
    return R.run_DD2_model(_empty_adj(), _cluster_table(), model_params_dd2=_STEP_PARAMS, threads=2)


@case("randomization.run_DD2_model.default_threads")
def _():
    return R.run_DD2_model(_empty_adj(), _cluster_table(), model_params_dd2=_STEP_PARAMS)


@case("randomization.run_DD2_model.return_params")
def _():
    # the parameter table comes back unchanged, after the graph
    params = pd.DataFrame(
        {"exp_model_scale": [STEP_A], "exp_model_exponent": [STEP_B]},
        index=pd.Index([7], name="seed"),
    )
    return R.run_DD2_model(
        _empty_adj(), _cluster_table(), model_params_dd2=params, threads=2, return_params=True
    )


@case("randomization.run_DD2_model.return_params_false_is_default")
def _():
    return R.run_DD2_model(
        _empty_adj(),
        _cluster_table(),
        model_params_dd2=_STEP_PARAMS,
        threads=2,
        return_params=False,
    )


@case("randomization.run_DD2_model.params_are_averaged_over_rows")
def _():
    # rows are fits for different subsamples: the generator uses their mean (scale 0 and 2e6 -> 1e6, exponent 1)
    two = pd.DataFrame({"exp_model_scale": [0.0, 2 * STEP_A], "exp_model_exponent": [0.5, 1.5]})
    return R.run_DD2_model(_empty_adj(), _cluster_table(), model_params_dd2=two, threads=2)


@case("randomization.run_DD2_model.coordinate_subsets")
def _():
    # the columns named in coord_names decide the distance; fewer than three are padded with zeros
    table, adj = _cluster_table(), _empty_adj()
    return {
        names: R.run_DD2_model(
            adj, table, model_params_dd2=_STEP_PARAMS, coord_names=list(names), threads=2
        )
        for names in ["xyz", "zxy", "xy", "yz", "z", "x", "y"]
    }


@case("randomization.run_DD2_model.extra_column_ignored")
def _():
    # 'w' is a column of the table but not a coordinate
    return R.run_DD2_model(
        _empty_adj(),
        _cluster_table(),
        model_params_dd2=_STEP_PARAMS,
        coord_names=["x", "y", "z"],
        threads=2,
    )


@case("randomization.run_DD2_model.config_arguments_ignored_with_params")
def _():
    # **config_dict only reaches the fit; with precomputed parameters (and for the seed always) it is swallowed
    return R.run_DD2_model(
        _empty_adj(),
        _cluster_table(),
        model_params_dd2=_STEP_PARAMS,
        threads=2,
        seed=(1, 2),
        bin_size_um=7,
        unknown_option="x",
    )


@case("randomization.run_DD2_model.zero_and_nan_parameters")
def _():
    zero = pd.DataFrame({"exp_model_scale": [0.0], "exp_model_exponent": [STEP_B]})
    nan = pd.DataFrame({"exp_model_scale": [np.nan], "exp_model_exponent": [np.nan]})
    return {
        "zero": R.run_DD2_model(_empty_adj(), _cluster_table(), model_params_dd2=zero, threads=2),
        "nan": R.run_DD2_model(_empty_adj(), _cluster_table(), model_params_dd2=nan, threads=2),
    }


@case("randomization.run_DD2_model.adjacency_only_gives_the_size")
def _():
    # adj is only used for its number of rows: an empty sparse matrix, a dense zero matrix and a full one give the same
    table, n = _cluster_table(), 10
    return {
        name: R.run_DD2_model(m, table, model_params_dd2=_STEP_PARAMS, threads=2)
        for name, m in [
            ("sparse_empty", _empty_adj()),
            ("dense_zero", np.zeros((n, n))),
            ("full", sp.csr_matrix(np.ones((n, n)))),
        ]
    }


@case("randomization.run_DD2_model.adjacency_smaller_than_table")
def _():
    # n = adj.shape[0]: the C++ generator reads only the first n rows of the table
    return R.run_DD2_model(
        _empty_adj(6), _cluster_table(), model_params_dd2=_STEP_PARAMS, threads=2
    )


@case("randomization.run_DD2_model.moderate_probability_properties")
def _():
    # random (the seed is not used): only shape, dtype and the absence of loops are fixed
    params = pd.DataFrame({"exp_model_scale": [0.5], "exp_model_exponent": [0.001]})
    m = R.run_DD2_model(
        _empty_adj(), _cluster_table(), model_params_dd2=params, threads=2, seed=(1, 2)
    )
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 90)}


@case("randomization.run_DD2_model.missing_coordinate_column")
def _():
    return R.run_DD2_model(
        _empty_adj(),
        _cluster_table(),
        model_params_dd2=_STEP_PARAMS,
        coord_names=["x", "y", "nope"],
        threads=1,
    )


@case("randomization.run_DD2_model.four_coordinates_rejected")
def _():
    return R.run_DD2_model(
        _empty_adj(),
        _cluster_table(),
        model_params_dd2=_STEP_PARAMS,
        coord_names=["x", "y", "z", "w"],
        threads=1,
    )


@case("randomization.run_DD2_model.missing_parameter_column")
def _():
    return R.run_DD2_model(
        _empty_adj(),
        _cluster_table(),
        model_params_dd2=pd.DataFrame({"exp_model_scale": [1.0]}),
        threads=1,
    )


@case("randomization.run_DD2_model.zero_threads")
def _():
    return R.run_DD2_model(_empty_adj(), _cluster_table(), model_params_dd2=_STEP_PARAMS, threads=0)


def _distance_dependent_adjacency(n=80, seed=3):
    """Adjacency drawn with p(d) = 0.4 exp(-0.01 d) on random points (numpy only, so that it is the same everywhere)."""
    r = np.random.RandomState(seed)
    xyz = r.random_sample((n, 3)) * 300.0
    dist = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    adj = r.random_sample((n, n)) < 0.4 * np.exp(-0.01 * dist)
    np.fill_diagonal(adj, False)
    return sp.csr_matrix(adj), pd.DataFrame(xyz, columns=list("xyz"))


@case("randomization.run_DD2_model.fit_requires_a_broken_import")
def _():
    # model_params_dd2=None fits the parameters with ``from . import modelling``, which looks for
    # connalysis.randomization.modelling (not connalysis.modelling) and fails with ImportError.
    adj, table = _distance_dependent_adjacency()
    return R.run_DD2_model(adj, table, threads=1)


@case("randomization.run_DD2_model.fit_with_modelling_attached", rtol=1e-6)
def _():
    # The fitting branch itself, reached by making ``from . import modelling`` work: connalysis.modelling is attached to
    # the package for the duration of the call.  Returns the fitted table and what is deterministic about the graph.
    import connalysis.modelling
    import connalysis.randomization as pkg

    adj, table = _distance_dependent_adjacency()
    had = hasattr(pkg, "modelling")
    pkg.modelling = connalysis.modelling
    try:
        graph, params = R.run_DD2_model(adj, table, threads=2, return_params=True)
        graph_bin, params_bin = R.run_DD2_model(
            adj, table, threads=2, return_params=True, bin_size_um=50, max_range_um=300
        )
        graph_xy, params_xy = R.run_DD2_model(
            adj, table, coord_names=["x", "y"], threads=2, return_params=True
        )
    finally:
        if not had:
            del pkg.modelling
    return {
        "params": params,
        "shape": shape_info(graph),
        "params_binned": params_bin,
        "shape_binned": shape_info(graph_bin),
        "params_xy": params_xy,
        "shape_xy": shape_info(graph_xy),
    }


# ----------------------------------------------------------------------------- run_DD3


def _depths_with_ties(n=30):
    return np.round(depth_values(n) / 20.0)  # a few distinct values, so many pairs have equal depth


@case("randomization.run_DD3.seeded", tags=CPP)
def _():
    return R.run_DD3(
        30, 0.5, 0.02, 0.3, 0.01, moderate_points(30), depth_values(30), threads=2, seed=(7, 8)
    )


@case("randomization.run_DD3.default_threads", tags=CPP)
def _():
    return R.run_DD3(30, 0.5, 0.02, 0.3, 0.01, moderate_points(30), depth_values(30), seed=(7, 8))


@case("randomization.run_DD3.thread_count_changes_the_graph", tags=CPP)
def _():
    xyz, depth = moderate_points(30), depth_values(30)
    return {
        t: R.run_DD3(30, 0.5, 0.02, 0.3, 0.01, xyz, depth, threads=t, seed=(7, 8))
        for t in (1, 2, 3, 8)
    }


@case("randomization.run_DD3.second_seed", tags=CPP)
def _():
    xyz, depth = moderate_points(30), depth_values(30)
    return {
        "one_thread_seed2=8": R.run_DD3(
            30, 0.5, 0.02, 0.3, 0.01, xyz, depth, threads=1, seed=(7, 8)
        ),
        "one_thread_seed2=9": R.run_DD3(
            30, 0.5, 0.02, 0.3, 0.01, xyz, depth, threads=1, seed=(7, 9)
        ),
        "two_threads_seed2=8": R.run_DD3(
            30, 0.5, 0.02, 0.3, 0.01, xyz, depth, threads=2, seed=(7, 8)
        ),
        "two_threads_seed2=9": R.run_DD3(
            30, 0.5, 0.02, 0.3, 0.01, xyz, depth, threads=2, seed=(7, 9)
        ),
    }


@case("randomization.run_DD3.equal_depths_use_the_product_formula", tags=CPP)
def _():
    # for dz == 0 the C++ code uses a1 a2 exp(-(b1 + b2) d) / 2 instead of a1 or a2
    return R.run_DD3(
        30, 0.9, 0.02, 0.8, 0.01, moderate_points(30), _depths_with_ties(30), threads=2, seed=(7, 8)
    )


@case("randomization.run_DD3.depth_direction")
def _():
    # a1 acts when depth[source] < depth[target], a2 when it is larger (and neither when they are equal and a1 a2 = 0)
    xyz = clusters((4, 5, 3))
    depth = np.array([0.0, 1, 2, 3, 10, 11, 12, 13, 14, 20, 21, 22])
    return {
        "a1_only": R.run_DD3(12, STEP_A, STEP_B, 0.0, STEP_B, xyz, depth, threads=1),
        "a2_only": R.run_DD3(12, 0.0, STEP_B, STEP_A, STEP_B, xyz, depth, threads=1),
        "both": R.run_DD3(12, STEP_A, STEP_B, STEP_A, STEP_B, xyz, depth, threads=2, seed=(1, 2)),
        "depth_reversed": R.run_DD3(
            12, STEP_A, STEP_B, 0.0, STEP_B, xyz, depth[::-1].copy(), threads=1
        ),
    }


@case("randomization.run_DD3.unseeded_properties")
def _():
    m = R.run_DD3(30, 0.5, 0.02, 0.3, 0.01, moderate_points(30), depth_values(30), threads=2)
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 30 * 29)}


# ----------------------------------------------------------------------------- run_DD2_block_pre

_PRE_PROBS = np.array([[0.5, 0.02], [0.3, 0.01], [0.8, 0.05]])


@case("randomization.run_DD2_block_pre.seeded", tags=CPP)
def _():
    return R.run_DD2_block_pre(
        30, _PRE_PROBS, _blocks(30), moderate_points(30), threads=2, seed=(7, 8)
    )


@case("randomization.run_DD2_block_pre.default_threads", tags=CPP)
def _():
    return R.run_DD2_block_pre(30, _PRE_PROBS, _blocks(30), moderate_points(30), seed=(7, 8))


@case("randomization.run_DD2_block_pre.thread_count_changes_the_graph", tags=CPP)
def _():
    xyz, blocks = moderate_points(30), _blocks(30)
    return {
        t: R.run_DD2_block_pre(30, _PRE_PROBS, blocks, xyz, threads=t, seed=(7, 8))
        for t in (1, 2, 3, 8)
    }


@case("randomization.run_DD2_block_pre.second_seed", tags=CPP)
def _():
    xyz, blocks = moderate_points(30), _blocks(30)
    return {
        "one_thread_seed2=8": R.run_DD2_block_pre(
            30, _PRE_PROBS, blocks, xyz, threads=1, seed=(7, 8)
        ),
        "one_thread_seed2=9": R.run_DD2_block_pre(
            30, _PRE_PROBS, blocks, xyz, threads=1, seed=(7, 9)
        ),
        "two_threads_seed2=8": R.run_DD2_block_pre(
            30, _PRE_PROBS, blocks, xyz, threads=2, seed=(7, 8)
        ),
        "two_threads_seed2=9": R.run_DD2_block_pre(
            30, _PRE_PROBS, blocks, xyz, threads=2, seed=(7, 9)
        ),
    }


@case("randomization.run_DD2_block_pre.block_of_the_source_decides")
def _():
    # source block 0 and 2 connect inside their cluster, source block 1 has no outgoing edges
    blocks = np.array([0] * 4 + [1] * 5 + [2] * 3)
    probs = np.array([[STEP_A, STEP_B], [0.0, STEP_B], [STEP_A, STEP_B]])
    return R.run_DD2_block_pre(12, probs, blocks, clusters())


@case("randomization.run_DD2_block_pre.sources_in_one_cluster_with_different_blocks")
def _():
    # one cluster, alternating blocks: only the sources of block 0 send edges, to everybody in the cluster
    blocks = np.array([0, 1, 0, 1, 0, 1])
    probs = np.array([[STEP_A, STEP_B], [0.0, STEP_B]])
    return R.run_DD2_block_pre(6, probs, blocks, clusters((6,)), threads=2, seed=(1, 2))


@case("randomization.run_DD2_block_pre.unseeded_properties")
def _():
    m = R.run_DD2_block_pre(30, _PRE_PROBS, _blocks(30), moderate_points(30), threads=2)
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 30 * 29)}


@case("randomization.run_DD2_block_pre.float_blocks_rejected")
def _():
    return R.run_DD2_block_pre(
        30, _PRE_PROBS, _blocks(30).astype(float), moderate_points(30), threads=1, seed=(1, 2)
    )


@case("randomization.run_DD2_block_pre.probs_as_list_of_pairs", tags=CPP)
def _():
    return R.run_DD2_block_pre(
        30,
        [tuple(p) for p in _PRE_PROBS.tolist()],
        _blocks(30),
        moderate_points(30),
        threads=2,
        seed=(7, 8),
    )


# ----------------------------------------------------------------------------- run_DD2_block


def _block_probs(seed=4):
    return np.random.RandomState(seed).rand(3, 3, 2) * [0.9, 0.05]


@case("randomization.run_DD2_block.seeded", tags=CPP)
def _():
    return R.run_DD2_block(30, _block_probs(), _blocks(30), moderate_points(30), 2, seed=(7, 8))


@case("randomization.run_DD2_block.thread_count_changes_the_graph", tags=CPP)
def _():
    xyz, blocks, probs = moderate_points(30), _blocks(30), _block_probs()
    return {t: R.run_DD2_block(30, probs, blocks, xyz, t, seed=(7, 8)) for t in (1, 2, 3, 8)}


@case("randomization.run_DD2_block.second_seed", tags=CPP)
def _():
    xyz, blocks, probs = moderate_points(30), _blocks(30), _block_probs()
    return {
        "one_thread_seed2=8": R.run_DD2_block(30, probs, blocks, xyz, 1, seed=(7, 8)),
        "one_thread_seed2=9": R.run_DD2_block(30, probs, blocks, xyz, 1, seed=(7, 9)),
        "two_threads_seed2=8": R.run_DD2_block(30, probs, blocks, xyz, 2, seed=(7, 8)),
        "two_threads_seed2=9": R.run_DD2_block(30, probs, blocks, xyz, 2, seed=(7, 9)),
    }


@case("randomization.run_DD2_block.source_and_target_blocks_decide")
def _():
    # block 0 <-> block 1 connect inside a cluster, block 2 (own cluster) connects to itself, nothing else
    both = np.zeros((3, 3, 2))
    both[:, :, 1] = STEP_B
    both[0, 1, 0] = both[1, 0, 0] = both[2, 2, 0] = STEP_A
    # only the pathway from block 0 to block 1
    one_way = np.zeros((3, 3, 2))
    one_way[:, :, 1] = STEP_B
    one_way[0, 1, 0] = STEP_A
    blocks = np.array([0, 1, 0, 1] + [0, 1, 0, 1, 0] + [2, 2, 2])
    return {
        "both_ways_unseeded": R.run_DD2_block(12, both, blocks, clusters(), 1),
        "one_way": R.run_DD2_block(12, one_way, blocks, clusters(), 2, seed=(1, 2)),
    }


@case("randomization.run_DD2_block.threads_is_required")
def _():
    # unlike the other generators, run_DD2_block has no default for threads
    return R.run_DD2_block(12, _block_probs(), _blocks(12), clusters())


@case("randomization.run_DD2_block.unseeded_properties")
def _():
    m = R.run_DD2_block(30, _block_probs(), _blocks(30), moderate_points(30), 2)
    return {**shape_info(m), "n_edges_at_most_full": bool(m.nnz <= 30 * 29)}


@case("randomization.run_DD2_block.float_blocks_rejected")
def _():
    return R.run_DD2_block(
        30, _block_probs(), _blocks(30).astype(float), moderate_points(30), 1, seed=(1, 2)
    )


# ----------------------------------------------------------------------------- real data (first 300 neurons of the tutorial connectome)


def _real_inputs():
    """Adjacency, coordinates, depth and the two synapse classes (294 EXC, 6 INH) of the first 300 neurons."""
    adj, nodes = tutorial_connectome(300)
    cls = nodes["synapse_class"].cat.codes.to_numpy()
    dense = adj.toarray()
    probs = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            mi, mj = cls == i, cls == j
            probs[i, j] = dense[np.ix_(mi, mj)].sum() / (mi.sum() * mj.sum() - (mi & mj).sum())
    return adj, nodes[["x", "y", "z"]].to_numpy(), nodes["depth"].to_numpy(), cls, probs


@case("randomization.ER_model.real_data", tags=CPP)
def _():
    adj, _, _, _, _ = _real_inputs()
    return R.ER_model(adj, threads=2, seed=1)


@case("randomization.run_SBM.real_data", tags=CPP)
def _():
    _, _, _, cls, probs = _real_inputs()
    return R.run_SBM(300, probs, cls, threads=2, seed=(1, 2))


@case("randomization.run_DD2.real_data", tags=CPP)
def _():
    _, xyz, _, _, _ = _real_inputs()
    return R.run_DD2(300, 0.15, 0.0087, xyz, threads=2, seed=(1, 2))


@case("randomization.run_DD3.real_data", tags=CPP)
def _():
    # the depths are whole numbers: many pairs have equal depth
    _, xyz, depth, _, _ = _real_inputs()
    return R.run_DD3(300, 0.18, 0.0097, 0.14, 0.0080, xyz, depth, threads=2, seed=(1, 2))


@case("randomization.run_DD2_block_pre.real_data", tags=CPP)
def _():
    _, xyz, _, cls, _ = _real_inputs()
    return R.run_DD2_block_pre(
        300, np.array([[0.15, 0.0087], [0.2, 0.01]]), cls, xyz, threads=2, seed=(1, 2)
    )


@case("randomization.run_DD2_block.real_data", tags=CPP)
def _():
    _, xyz, _, cls, _ = _real_inputs()
    probs = np.array([[[0.15, 0.0087], [0.05, 0.004]], [[0.2, 0.01], [0.1, 0.006]]])
    return R.run_DD2_block(300, probs, cls, xyz, 2, seed=(1, 2))


@case("randomization.run_DD2_model.real_data_fit_with_modelling_attached", rtol=1e-6)
def _():
    # see fit_with_modelling_attached: the fit branch, with the real adjacency and coordinates
    import connalysis.modelling
    import connalysis.randomization as pkg

    adj, nodes = tutorial_connectome(300)
    had = hasattr(pkg, "modelling")
    pkg.modelling = connalysis.modelling
    try:
        graph, params = R.run_DD2_model(adj, nodes, threads=2, return_params=True)
    finally:
        if not had:
            del pkg.modelling
    return {"params": params, "shape": shape_info(graph)}


@case("randomization.run_DD2_model.missing_coordinates_are_padded_with_zeros")
def _():
    # p(d) ~ exp(300 (1.5 - d)): connected within distance 1.5.  A pair at (0.9, 0.9) is 1.27 apart when the third
    # coordinate is zero but 1.56 apart if it were padded with a copy of the first one; a pair 1.2 apart in z only
    # is connected with zero padding but not with a repeated coordinate.
    table = pd.DataFrame(
        {
            "x": [0.0, 0.9, 100.0, 100.0, 200.0, 200.0],
            "y": [0.0, 0.9, 100.0, 100.0, 200.0, 200.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 1.2],
        }
    )
    params = pd.DataFrame({"exp_model_scale": [np.exp(450.0)], "exp_model_exponent": [300.0]})
    adj = sp.csr_matrix((6, 6), dtype=bool)
    return {
        "xy": R.run_DD2_model(
            adj, table, model_params_dd2=params, coord_names=["x", "y"], threads=1
        ),
        "z": R.run_DD2_model(adj, table, model_params_dd2=params, coord_names=["z"], threads=1),
        "xyz": R.run_DD2_model(
            adj, table, model_params_dd2=params, coord_names=["x", "y", "z"], threads=1
        ),
    }


# ----------------------------------------------------------------------------- _dict_to_coo


def _dict_to_coo(adj, n):
    return RR._dict_to_coo(adj, n)


@case("randomization._dict_to_coo.basic")
def _():
    return _dict_to_coo({"row": [0, 0, 1, 1, 2, 2], "col": [1, 2, 0, 2, 0, 1]}, 3)


@case("randomization._dict_to_coo.empty")
def _():
    return _dict_to_coo({"row": [], "col": []}, 4)


@case("randomization._dict_to_coo.zero_vertices")
def _():
    return _dict_to_coo({"row": [], "col": []}, 0)


@case("randomization._dict_to_coo.duplicates_are_kept_in_the_coo_matrix")
def _():
    m = _dict_to_coo({"row": [0, 0, 1], "col": [1, 1, 2]}, 3)
    return {"matrix": m, "stored_entries": m.nnz, "csr": m.tocsr()}


@case("randomization._dict_to_coo.array_and_tuple_input")
def _():
    return {
        "arrays": _dict_to_coo({"row": np.array([0, 1]), "col": np.array([1, 0])}, 2),
        "tuples": _dict_to_coo({"row": (0, 1), "col": (1, 0)}, 2),
        "extra_keys": _dict_to_coo({"row": [0], "col": [1], "ignored": [5]}, 2),
    }


@case("randomization._dict_to_coo.larger_than_the_indices_need")
def _():
    return _dict_to_coo({"row": [0], "col": [1]}, 6)


@case("randomization._dict_to_coo.index_out_of_range")
def _():
    return _dict_to_coo({"row": [0, 5], "col": [1, 0]}, 3)


@case("randomization._dict_to_coo.missing_key")
def _():
    return _dict_to_coo({"row": [0]}, 3)


@case("randomization._dict_to_coo.lengths_differ")
def _():
    return _dict_to_coo({"row": [0, 1], "col": [1]}, 3)


@case("randomization._dict_to_coo.dtype_of_the_generators_output")
def _():
    # what the generators return is built by this function: float64 ones
    m = R.run_ER(6, 1.0, threads=1, seed=(1, 2))
    return {"dtype": str(m.dtype), "all_ones": bool((m.data == 1).all()), "nnz": m.nnz}
