"""``node_stats_per_position_single`` and ``node_stats_per_position`` in ``connalysis.network.stats``.

Mean, standard deviation and standard error of a node property over the nodes of a simplex list, per position of the
node in the simplex and over all positions (row ``"all"``, so the index is an object index ``[0, 1, ..., "all"]``).
The simplex lists are computed with ``topology.list_simplices_by_dimension`` on graphs with planted simplices.
"""

import numpy as np
import pandas as pd
from support.cases import case
from support.data import planted, tutorial_connectome

from connalysis.network import stats as S
from connalysis.network import topology as T


def _lists(n=40, seed=10, **kw):
    """Simplex lists (a Series indexed by dimension) of a graph with a 5-clique and a 4-clique planted."""
    return T.list_simplices_by_dimension(planted(n, 0.08, seed=seed, sizes=(5, 4)), **kw)


def _values(n=40, seed=0, nans=(), index=None, dtype=float):
    v = np.random.RandomState(seed).random_sample(n) * 10
    v[list(nans)] = np.nan
    return pd.Series(
        v.astype(dtype), index=pd.RangeIndex(n, name="node") if index is None else index, name="v"
    )


NANS = (3, 9, 10, 17, 22, 30)


# ============================================================================================== single


@case("stats.node_stats_per_position_single.with_multiplicity")
def _():
    # all dimensions of the lists, including the vertices (dim 0, one position) and the edges (dim 1)
    sl, vals = _lists(), _values()
    return {int(dim): S.node_stats_per_position_single(sl[dim], vals) for dim in sl.index}


@case("stats.node_stats_per_position_single.without_multiplicity")
def _():
    sl, vals = _lists(), _values()
    return {
        int(dim): S.node_stats_per_position_single(sl[dim], vals, with_multiplicity=False)
        for dim in sl.index
    }


@case("stats.node_stats_per_position_single.multiplicity_flag_positional")
def _():
    sl, vals = _lists(), _values()
    return S.node_stats_per_position_single(sl[2], vals, False)


@case("stats.node_stats_per_position_single.keyword_arguments")
def _():
    return S.node_stats_per_position_single(
        simplex_list=_lists()[2], values=_values(), with_multiplicity=False
    )


@case("stats.node_stats_per_position_single.nan_values")
def _():
    # NaN values are left out of every statistic
    sl, vals = _lists(), _values(nans=NANS)
    return {
        (int(dim), with_mult): S.node_stats_per_position_single(
            sl[dim], vals, with_multiplicity=with_mult
        )
        for dim in (1, 2, 3, 4)
        for with_mult in (True, False)
    }


@case("stats.node_stats_per_position_single.all_values_nan")
def _():
    sl = _lists()
    vals = _values(nans=range(40))
    return {
        "with": S.node_stats_per_position_single(sl[2], vals),
        "without": S.node_stats_per_position_single(sl[2], vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.single_simplex")
def _():
    # one simplex: the standard deviation is 0 and the standard error undefined
    simplex = np.array([[4, 17, 2, 31]])
    vals = _values()
    return {
        "with": S.node_stats_per_position_single(simplex, vals),
        "without": S.node_stats_per_position_single(simplex, vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.repeated_nodes_in_a_position")
def _():
    # the same node in the same position of several simplices counts several times, unless with_multiplicity=False
    simplices = np.array([[0, 5, 9], [0, 5, 7], [0, 6, 9], [1, 6, 9], [1, 5, 9], [0, 5, 9]])
    vals = _values()
    return {
        "with": S.node_stats_per_position_single(simplices, vals),
        "without": S.node_stats_per_position_single(simplices, vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.no_simplices")
def _():
    empty = np.zeros((0, 3), dtype=int)
    vals = _values()
    return {
        "with": S.node_stats_per_position_single(empty, vals),
        "without": S.node_stats_per_position_single(empty, vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.value_dtypes")
def _():
    sl = _lists()
    base = _values(seed=3)
    return {
        "int64": S.node_stats_per_position_single(sl[2], (base * 3).astype(np.int64)),
        "int64_without": S.node_stats_per_position_single(
            sl[2], (base * 3).astype(np.int64), with_multiplicity=False
        ),
        "float32": S.node_stats_per_position_single(sl[2], base.astype(np.float32)),
        "float32_without": S.node_stats_per_position_single(
            sl[2], base.astype(np.float32), with_multiplicity=False
        ),
    }


@case("stats.node_stats_per_position_single.bool_values")
def _():
    # fractions of True: fine with multiplicity ...
    return S.node_stats_per_position_single(_lists()[2], _values(seed=4) > 5)


@case("stats.node_stats_per_position_single.bool_values_without_multiplicity")
def _():
    # ... but the concat of the per-position values turns into an object array that np.nanstd cannot handle
    return S.node_stats_per_position_single(
        _lists()[2], _values(seed=4) > 5, with_multiplicity=False
    )


@case("stats.node_stats_per_position_single.values_with_permuted_index")
def _():
    # the nodes of the simplices are looked up by label, not by position
    sl = _lists()
    base = _values()
    perm = np.random.RandomState(5).permutation(40)
    vals = pd.Series(base.to_numpy(), index=pd.Index(perm, name="node"))
    return {
        "with": S.node_stats_per_position_single(sl[2], vals),
        "without": S.node_stats_per_position_single(sl[2], vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.values_labelled_by_gid")
def _():
    # a Series whose labels are ids (1000 + node), simplices given in those ids
    sl = _lists()
    base = _values()
    vals = pd.Series(base.to_numpy(), index=pd.Index(np.arange(40) + 1000, name="gid"))
    return {
        "with": S.node_stats_per_position_single(sl[3] + 1000, vals),
        "without": S.node_stats_per_position_single(sl[3] + 1000, vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.values_for_more_nodes_than_listed")
def _():
    # values may be defined for many more nodes than occur in the simplices
    sl = _lists()
    vals = _values(n=120, seed=6)
    return S.node_stats_per_position_single(sl[2], vals)


@case("stats.node_stats_per_position_single.max_simplices")
def _():
    sl = _lists(max_simplices=True)
    vals = _values()
    return {int(dim): S.node_stats_per_position_single(sl[dim], vals) for dim in (1, 2, 3, 4)}


@case("stats.node_stats_per_position_single.undirected_and_reciprocal_lists")
def _():
    vals = _values()
    sl_u = _lists(simplex_type="undirected")
    sl_r = _lists(simplex_type="reciprocal")
    return {
        "undirected": {d: S.node_stats_per_position_single(sl_u[d], vals) for d in (1, 2, 3)},
        "reciprocal": S.node_stats_per_position_single(sl_r[1], vals, with_multiplicity=False),
    }


@case("stats.node_stats_per_position_single.tutorial")
def _():
    adj, nodes = tutorial_connectome(500)
    sl = T.list_simplices_by_dimension(adj)
    depth = nodes["depth"]
    return {
        (int(dim), with_mult): S.node_stats_per_position_single(
            sl[dim], depth, with_multiplicity=with_mult
        )
        for dim in (1, 2, 3, 4)
        for with_mult in (True, False)
    }


@case("stats.node_stats_per_position_single.inputs_are_not_modified")
def _():
    sl, vals = _lists()[2].copy(), _values(nans=NANS)
    before = (sl.copy(), vals.copy())
    result = [
        S.node_stats_per_position_single(sl, vals),
        S.node_stats_per_position_single(sl, vals, False),
    ]
    return result, sl, vals, np.array_equal(sl, before[0]), vals.equals(before[1])


@case("stats.node_stats_per_position_single.node_not_in_values")
def _():
    return S.node_stats_per_position_single(np.array([[0, 1], [2, 99]]), _values())


@case("stats.node_stats_per_position_single.node_not_in_values_without_multiplicity")
def _():
    return S.node_stats_per_position_single(
        np.array([[0, 1], [2, 99]]), _values(), with_multiplicity=False
    )


@case("stats.node_stats_per_position_single.one_dimensional_list")
def _():
    return S.node_stats_per_position_single(np.array([1, 2, 3]), _values())


# ============================================================================================== all dimensions


@case("stats.node_stats_per_position.all_dimensions")
def _():
    return S.node_stats_per_position(_lists(), _values())


@case("stats.node_stats_per_position.without_multiplicity")
def _():
    return S.node_stats_per_position(_lists(), _values(), with_multiplicity=False)


@case("stats.node_stats_per_position.keyword_arguments")
def _():
    return S.node_stats_per_position(
        simplex_lists=_lists(), values=_values(), dims=[2, 3], with_multiplicity=False
    )


@case("stats.node_stats_per_position.dims_list_keeps_given_order")
def _():
    return S.node_stats_per_position(_lists(), _values(), dims=[3, 1])


@case("stats.node_stats_per_position.dims_array_and_range")
def _():
    sl, vals = _lists(), _values()
    return {
        "array": S.node_stats_per_position(sl, vals, dims=np.array([2, 3])),
        "range": S.node_stats_per_position(sl, vals, dims=range(1, 3)),
        "tuple_without_multiplicity": S.node_stats_per_position(
            sl, vals, dims=(4, 2), with_multiplicity=False
        ),
    }


@case("stats.node_stats_per_position.dims_positional")
def _():
    return S.node_stats_per_position(_lists(), _values(), [2], False)


@case("stats.node_stats_per_position.series_of_some_dimensions")
def _():
    # a Series that only holds dimensions 2 and 4: the default dims are its index
    sl = _lists()
    return S.node_stats_per_position(sl.loc[[4, 2]], _values())


@case("stats.node_stats_per_position.no_dims")
def _():
    return S.node_stats_per_position(_lists(), _values(), dims=[])


@case("stats.node_stats_per_position.nan_values")
def _():
    sl, vals = _lists(), _values(nans=NANS)
    return {
        "with": S.node_stats_per_position(sl, vals, dims=[1, 2, 3, 4]),
        "without": S.node_stats_per_position(sl, vals, dims=[1, 2, 3, 4], with_multiplicity=False),
    }


@case("stats.node_stats_per_position.max_simplices_lists")
def _():
    # the dimension 0 entry of max-simplex lists is an empty 1d array, so it has to be left out with dims
    return S.node_stats_per_position(_lists(max_simplices=True), _values(), dims=[1, 2, 3, 4])


@case("stats.node_stats_per_position.tutorial")
def _():
    adj, nodes = tutorial_connectome(500)
    sl = T.list_simplices_by_dimension(adj)
    return {
        "y": S.node_stats_per_position(sl, nodes["y"]),
        "depth_without_multiplicity": S.node_stats_per_position(
            sl, nodes["depth"], dims=[2, 3, 4], with_multiplicity=False
        ),
    }


@case("stats.node_stats_per_position.max_simplices_lists_all_dims")
def _():
    # defect: the empty dimension 0 entry breaks the default dims
    return S.node_stats_per_position(_lists(max_simplices=True), _values())


@case("stats.node_stats_per_position.unknown_dimension")
def _():
    return S.node_stats_per_position(_lists(), _values(), dims=[9])


@case("stats.node_stats_per_position.simplex_lists_as_dict")
def _():
    # documented as a dict, but a Series is required (.index / .loc)
    return S.node_stats_per_position(dict(_lists()), _values(), dims=[2])
