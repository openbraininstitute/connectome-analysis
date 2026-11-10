"""``node_stats_neighborhood`` in ``connalysis.network.stats``: sum, mean, std and sem of a node property over the
neighbourhood of each center.

The neighbourhoods come from ``local.neighborhood_indices`` (computed here, or passed in with ``precomputed=True``).
The values are read with ``values[nbd_indices]``, i.e. ``Series.__getitem__`` with an integer array: label based for an
integer index, with a positional fallback (pandas 2: FutureWarning) for a non-integer one.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, tutorial_connectome

from connalysis.network import local as L
from connalysis.network import stats as S


def _mixed(fmt="csr"):
    """16 nodes with reciprocal pairs, two isolated nodes (5, 11), a pure source (12) and a pure sink (13)."""
    dense = digraph(16, 0.14, seed=3, reciprocal=0.3).toarray()
    dense[[5, 11], :] = False
    dense[:, [5, 11]] = False
    dense[:, 12] = False
    dense[12, [1, 2, 7]] = True
    dense[13, :] = False
    dense[[0, 4, 9], 13] = True
    return sp.csr_matrix(dense).asformat(fmt)


def _values(n=16, seed=0, nans=(), dtype=float):
    v = np.random.RandomState(seed).random_sample(n) * 10
    v[list(nans)] = np.nan
    return pd.Series(v.astype(dtype), index=pd.RangeIndex(n), name="v")


COMBOS = {
    "both": dict(pre=True, post=True),
    "pre": dict(pre=True, post=False),
    "post": dict(pre=False, post=True),
}
CENTERS = np.array([13, 5, 2, 12, 9, 0])  # unsorted; isolated 5, pure source 12, pure sink 13


@case("stats.node_stats_neighborhood.all_nodes_default")
def _():
    return S.node_stats_neighborhood(_values(), _mixed())


@case("stats.node_stats_neighborhood.without_center")
def _():
    # an isolated node without its center has an empty neighbourhood: sum 0, the rest undefined
    return S.node_stats_neighborhood(_values(), _mixed(), include_center=False)


@case("stats.node_stats_neighborhood.pre_post")
def _():
    vals, adj = _values(), _mixed()
    return {
        f"{name}_center{int(center)}": S.node_stats_neighborhood(
            vals, adj, include_center=center, **kw
        )
        for name, kw in COMBOS.items()
        for center in (True, False)
    }


@case("stats.node_stats_neighborhood.centers_unsorted")
def _():
    # the index of the result is the sorted one of neighborhood_indices
    vals, adj = _values(), _mixed()
    return {
        name: S.node_stats_neighborhood(vals, adj, all_nodes=False, centers=CENTERS, **kw)
        for name, kw in COMBOS.items()
    }


@case("stats.node_stats_neighborhood.centers_int32")
def _():
    # the index of the result keeps the dtype of the centers
    return S.node_stats_neighborhood(
        _values(), _mixed(), all_nodes=False, centers=np.array([9, 2, 12], dtype=np.int32)
    )


@case("stats.node_stats_neighborhood.centers_without_center")
def _():
    return S.node_stats_neighborhood(
        _values(), _mixed(), all_nodes=False, centers=CENTERS, include_center=False
    )


@case("stats.node_stats_neighborhood.centers_ignored_when_all_nodes")
def _():
    return S.node_stats_neighborhood(_values(), _mixed(), centers=np.array([9, 2]))


@case("stats.node_stats_neighborhood.positional_arguments")
def _():
    # (values, adj, pre, post, all_nodes, centers, include_center, precomputed, neighborhoods)
    return S.node_stats_neighborhood(
        _values(), _mixed(), True, False, False, np.array([7, 3]), False
    )


@case("stats.node_stats_neighborhood.keyword_arguments")
def _():
    return S.node_stats_neighborhood(
        values=_values(),
        adj=_mixed(),
        pre=True,
        post=False,
        all_nodes=False,
        centers=np.array([7, 3]),
        include_center=False,
        precomputed=False,
        neighborhoods=None,
    )


@case("stats.node_stats_neighborhood.adjacency_formats")
def _():
    vals, adj = _values(), _mixed()
    inputs = {fmt: adj.asformat(fmt) for fmt in ("csr", "csc", "coo", "lil")}
    inputs.update(ndarray=adj.toarray(), matrix=np.matrix(adj.toarray()))
    return {
        name: S.node_stats_neighborhood(vals, m, all_nodes=False, centers=CENTERS)
        for name, m in inputs.items()
    }


@case("stats.node_stats_neighborhood.precomputed")
def _():
    vals, adj = _values(), _mixed()
    nbd = L.neighborhood_indices(adj)
    nbd_centers = L.neighborhood_indices(
        adj, pre=True, post=False, all_nodes=False, centers=CENTERS
    )
    return {
        "with_center": S.node_stats_neighborhood(vals, precomputed=True, neighborhoods=nbd),
        "without_center": S.node_stats_neighborhood(
            vals, precomputed=True, neighborhoods=nbd, include_center=False
        ),
        "centers": S.node_stats_neighborhood(vals, precomputed=True, neighborhoods=nbd_centers),
        "computed_for_comparison": S.node_stats_neighborhood(vals, adj),
    }


@case("stats.node_stats_neighborhood.precomputed_ignores_the_other_arguments")
def _():
    # adj, pre, post, all_nodes and centers play no role once the neighbourhoods are given
    vals, adj = _values(), _mixed()
    nbd = L.neighborhood_indices(adj, pre=False, post=True)
    return S.node_stats_neighborhood(
        vals,
        adj=np.eye(3),
        pre=False,
        post=False,
        all_nodes=False,
        centers=None,
        precomputed=True,
        neighborhoods=nbd,
    )


@case("stats.node_stats_neighborhood.inputs_are_not_modified")
def _():
    vals, adj = _values(nans=(2, 7)), _mixed()
    nbd = L.neighborhood_indices(adj)
    nbd_before = pd.Series([a.copy() for a in nbd], index=nbd.index.copy(), name=nbd.name)
    result = [
        S.node_stats_neighborhood(vals, precomputed=True, neighborhoods=nbd),
        S.node_stats_neighborhood(vals, precomputed=True, neighborhoods=nbd, include_center=False),
        S.node_stats_neighborhood(vals, adj),
    ]
    same = all(np.array_equal(a, b) for a, b in zip(nbd, nbd_before)) and nbd.index.equals(
        nbd_before.index
    )
    return result, nbd, vals, adj, same


@case("stats.node_stats_neighborhood.precomputed_by_hand")
def _():
    # a hand-made Series: arrays (one empty), unsorted centers keep their order, a node as its own neighbour
    nbd = pd.Series(
        [np.array([1, 2]), np.array([], dtype=int), np.array([0, 5, 7, 9]), np.array([6])],
        index=pd.Index([9, 4, 3, 6], name="center"),
        name="neighbors",
    )
    vals = _values()
    return {
        "with_center": S.node_stats_neighborhood(vals, precomputed=True, neighborhoods=nbd),
        "without_center": S.node_stats_neighborhood(
            vals, precomputed=True, neighborhoods=nbd, include_center=False
        ),
    }


@case("stats.node_stats_neighborhood.precomputed_lists_and_equal_lengths")
def _():
    vals = _values()
    as_lists = pd.Series(
        [[1, 2], [], [0, 5, 7]], index=pd.Index([4, 6, 9], name="center"), name="neighbors"
    )
    same_length = pd.Series(
        [np.array([1, 2]), np.array([3, 4]), np.array([0, 5])],
        index=pd.Index([4, 6, 9], name="center"),
        name="neighbors",
    )
    return {
        "lists": S.node_stats_neighborhood(vals, precomputed=True, neighborhoods=as_lists),
        "lists_without_center": S.node_stats_neighborhood(
            vals, precomputed=True, neighborhoods=as_lists, include_center=False
        ),
        "equal_lengths": S.node_stats_neighborhood(
            vals, precomputed=True, neighborhoods=same_length
        ),
        "equal_lengths_without_center": S.node_stats_neighborhood(
            vals, precomputed=True, neighborhoods=same_length, include_center=False
        ),
    }


@case("stats.node_stats_neighborhood.nan_values")
def _():
    vals = _values(nans=(1, 4, 6, 9, 13))
    adj = _mixed()
    return {
        "with_center": S.node_stats_neighborhood(vals, adj),
        "without_center": S.node_stats_neighborhood(vals, adj, include_center=False),
    }


@case("stats.node_stats_neighborhood.all_values_nan")
def _():
    return S.node_stats_neighborhood(_values(nans=range(16)), _mixed())


@case("stats.node_stats_neighborhood.integer_values")
def _():
    vals = (_values(seed=2) * 3).astype(np.int64)
    return S.node_stats_neighborhood(vals, _mixed())


@case("stats.node_stats_neighborhood.values_with_permuted_integer_index")
def _():
    # integer labels are looked up by label: the value of node k is the one labelled k, wherever it sits
    base = _values()
    perm = np.random.RandomState(8).permutation(16)
    vals = pd.Series(base.to_numpy(), index=perm, name="v")
    return S.node_stats_neighborhood(vals, _mixed())


@case("stats.node_stats_neighborhood.values_with_string_index")
def _():
    # not an integer index: v[array of ints] falls back to positions (FutureWarning in pandas 2, removed in 3)
    base = _values()
    vals = pd.Series(base.to_numpy(), index=[f"cell{i}" for i in range(16)], name="v")
    return S.node_stats_neighborhood(vals, _mixed())


@case("stats.node_stats_neighborhood.values_as_array")
def _():
    # a plain array is indexed by position
    return S.node_stats_neighborhood(_values().to_numpy(), _mixed())


@case("stats.node_stats_neighborhood.values_as_array_with_nan")
def _():
    # for an array (unlike a Series) a plain sum would return NaN: the NaN-skipping functions are needed
    return {
        "with_center": S.node_stats_neighborhood(
            _values(nans=(1, 4, 6, 9, 13)).to_numpy(), _mixed()
        ),
        "without_center": S.node_stats_neighborhood(
            _values(nans=(1, 4, 6, 9, 13)).to_numpy(), _mixed(), include_center=False
        ),
    }


@case("stats.node_stats_neighborhood.values_for_more_nodes")
def _():
    return S.node_stats_neighborhood(
        _values(n=50, seed=9), _mixed(), all_nodes=False, centers=np.array([14, 2])
    )


@case("stats.node_stats_neighborhood.no_centers")
def _():
    return {
        "with_center": S.node_stats_neighborhood(
            _values(), _mixed(), all_nodes=False, centers=np.array([], dtype=int)
        ),
        "without_center": S.node_stats_neighborhood(
            _values(),
            _mixed(),
            all_nodes=False,
            centers=np.array([], dtype=int),
            include_center=False,
        ),
    }


@case("stats.node_stats_neighborhood.tiny_graphs")
def _():
    one_edge = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=bool))
    return {
        "single_node": S.node_stats_neighborhood(
            pd.Series([2.5]), sp.csr_matrix((1, 1), dtype=bool)
        ),
        "two_nodes": S.node_stats_neighborhood(pd.Series([2.5, 4.0]), one_edge),
        "two_nodes_without_center": S.node_stats_neighborhood(
            pd.Series([2.5, 4.0]), one_edge, include_center=False
        ),
    }


@case("stats.node_stats_neighborhood.tutorial_all_nodes")
def _():
    adj, nodes = tutorial_connectome(500)
    return S.node_stats_neighborhood(nodes["depth"], adj)


@case("stats.node_stats_neighborhood.tutorial_centers")
def _():
    adj, nodes = tutorial_connectome(500)
    centers = np.random.RandomState(10).permutation(500)[:60]
    return {
        name: S.node_stats_neighborhood(
            nodes["y"], adj, all_nodes=False, centers=centers, include_center=False, **kw
        )
        for name, kw in COMBOS.items()
    }


@case("stats.node_stats_neighborhood.tutorial_precomputed")
def _():
    adj, nodes = tutorial_connectome(500)
    nbd = L.neighborhood_indices(adj, pre=True, post=False)
    return {
        "depth": S.node_stats_neighborhood(nodes["depth"], precomputed=True, neighborhoods=nbd),
        "z": S.node_stats_neighborhood(
            nodes["z"], precomputed=True, neighborhoods=nbd, include_center=False
        ),
    }


# ----------------------------------------------------------------------------------------------- failures


@case("stats.node_stats_neighborhood.precomputed_without_neighborhoods")
def _():
    return S.node_stats_neighborhood(_values(), _mixed(), precomputed=True)


@case("stats.node_stats_neighborhood.precomputed_with_a_frame")
def _():
    nbd = L.neighborhood_indices(_mixed())
    return S.node_stats_neighborhood(_values(), precomputed=True, neighborhoods=nbd.to_frame())


@case("stats.node_stats_neighborhood.adjacency_missing")
def _():
    return S.node_stats_neighborhood(_values())


@case("stats.node_stats_neighborhood.neighborhoods_given_but_not_precomputed")
def _():
    return S.node_stats_neighborhood(_values(), neighborhoods=L.neighborhood_indices(_mixed()))


@case("stats.node_stats_neighborhood.centers_required")
def _():
    return S.node_stats_neighborhood(_values(), _mixed(), all_nodes=False)


@case("stats.node_stats_neighborhood.pre_and_post_false")
def _():
    return S.node_stats_neighborhood(_values(), _mixed(), pre=False, post=False)


@case("stats.node_stats_neighborhood.values_with_shifted_integer_index")
def _():
    # labels 1000.. do not contain the node numbers 0..15
    base = _values()
    vals = pd.Series(base.to_numpy(), index=np.arange(16) + 1000)
    return S.node_stats_neighborhood(vals, _mixed())


@case("stats.node_stats_neighborhood.precomputed_index_without_name")
def _():
    # reset_index() then calls the column "index", not "center"
    nbd = L.neighborhood_indices(_mixed())
    nbd.index.name = None
    return S.node_stats_neighborhood(_values(), precomputed=True, neighborhoods=nbd)


@case("stats.node_stats_neighborhood.precomputed_series_without_name")
def _():
    # reset_index() then calls the column 0, not "neighbors"
    nbd = L.neighborhood_indices(_mixed())
    nbd.name = None
    return S.node_stats_neighborhood(_values(), precomputed=True, neighborhoods=nbd)


@case("stats.node_stats_neighborhood.precomputed_unnamed_but_no_center_wanted")
def _():
    # without the center the names do not matter
    nbd = L.neighborhood_indices(_mixed())
    nbd.index.name = None
    nbd.name = None
    return S.node_stats_neighborhood(
        _values(), precomputed=True, neighborhoods=nbd, include_center=False
    )


@case("stats.node_stats_neighborhood.values_as_list")
def _():
    return S.node_stats_neighborhood(list(_values()), _mixed())
