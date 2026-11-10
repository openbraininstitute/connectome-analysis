"""``node_stats_participation`` and ``edge_stats_participation`` in ``connalysis.network.stats``.

Statistics of a node (edge) property over the nodes (edges) grouped by the maximal dimension of the simplices they
take part in.  The participation tables are computed with ``topology.node_participation`` and
``topology.edge_participation`` on graphs with planted simplices.

Both functions call ``np.product``, which numpy 2 removed: the golden files were written with ``np.product`` restored
as an alias of ``np.prod`` (see ``regenerate_golden.py``), so every case that reaches that call fails with an
AttributeError under numpy 2 -- on the original code and on a rewrite that keeps the call.  Cases that raise (or
return) before it, or that run no dimension at all, pass.
"""

import operator

import numpy as np
import pandas as pd
from support.cases import case
from support.data import planted, tutorial_connectome

from connalysis.network import stats as S
from connalysis.network import topology as T


def _graph(n=40, seed=10):
    return planted(n, 0.08, seed=seed, sizes=(5, 4))


def _node_values(n=40, seed=0, nans=(), dtype=float):
    v = np.random.RandomState(seed).random_sample(n) * 10
    v[list(nans)] = np.nan
    return pd.Series(v.astype(dtype), index=pd.RangeIndex(n, name="node"), name="v")


def _edge_values(index, nans=(), scale=0.37):
    """A value per edge that depends only on the edge itself (never on the row order of the participation table)."""
    v = np.array([((7 * int(a) + 13 * int(b)) % 17) * scale for a, b in index], dtype=float)
    v[list(nans)] = np.nan
    return pd.Series(v, index=index, name="edge_value")


def _hierarchy():
    """14 nodes: a feedforward 4-clique (0-3) with a path hanging off it (3-4-5-6), a reciprocal pair (7, 8), a
    feedforward triangle (9, 10, 11) and two isolated nodes (12, 13)."""
    dense = np.zeros((14, 14), dtype=bool)
    for i in range(4):
        dense[i, i + 1 : 4] = True
    dense[3, 4] = dense[4, 5] = dense[5, 6] = True
    dense[7, 8] = dense[8, 7] = True
    dense[9, 10] = dense[10, 11] = dense[9, 11] = True
    return dense


NANS = (3, 9, 10, 17, 22, 30)


# ============================================================================================== nodes


@case("stats.node_stats_participation.maximal_dimension_equals")
def _():
    return S.node_stats_participation(T.node_participation(_graph()), _node_values())


@case("stats.node_stats_participation.maximal_dimension_at_least")
def _():
    return S.node_stats_participation(
        T.node_participation(_graph()), _node_values(), condition=operator.ge
    )


@case("stats.node_stats_participation.condition_positional")
def _():
    return S.node_stats_participation(
        T.node_participation(_graph()), _node_values(), operator.ge, [1, 2]
    )


@case("stats.node_stats_participation.keyword_arguments")
def _():
    return S.node_stats_participation(
        participation=T.node_participation(_graph()),
        vals=_node_values(),
        condition=operator.ge,
        dims=[1, 2],
    )


@case("stats.node_stats_participation.dims_subset")
def _():
    par, vals = T.node_participation(_graph()), _node_values()
    return {
        "list_eq": S.node_stats_participation(par, vals, dims=[3, 1]),
        "array_ge": S.node_stats_participation(
            par, vals, condition=operator.ge, dims=np.array([2, 4])
        ),
        "range_eq": S.node_stats_participation(par, vals, dims=range(1, 3)),
        "single_ge": S.node_stats_participation(par, vals, condition=operator.ge, dims=[3]),
    }


@case("stats.node_stats_participation.no_dims")
def _():
    # no dimension to analyse: an empty table (np.product is never reached)
    return S.node_stats_participation(T.node_participation(_graph()), _node_values(), dims=[])


@case("stats.node_stats_participation.nan_values")
def _():
    # NaN values are skipped by the mean/std/sem; the weighted mean divides by the weights of *all* the nodes
    par, vals = T.node_participation(_graph()), _node_values(nans=NANS)
    return {
        "eq": S.node_stats_participation(par, vals),
        "ge": S.node_stats_participation(par, vals, operator.ge),
    }


@case("stats.node_stats_participation.integer_values")
def _():
    par = T.node_participation(_graph())
    vals = (_node_values(seed=2) * 3).astype(np.int64)
    return {
        "eq": S.node_stats_participation(par, vals),
        "ge": S.node_stats_participation(par, vals, operator.ge),
    }


@case("stats.node_stats_participation.other_comparisons")
def _():
    # any binary comparison works as the condition, not only eq and ge
    par, vals = T.node_participation(_graph()), _node_values()
    return {
        op.__name__: S.node_stats_participation(par, vals, condition=op, dims=[1, 2, 3])
        for op in (operator.gt, operator.le, operator.lt, operator.ne)
    }


@case("stats.node_stats_participation.maximal_simplices_table")
def _():
    par = T.node_participation(_graph(), max_simplices=True)
    vals = _node_values()
    return {
        "eq": S.node_stats_participation(par, vals),
        "ge": S.node_stats_participation(par, vals, operator.ge),
    }


@case("stats.node_stats_participation.truncated_participation")
def _():
    # participation computed up to max_dim=2: nodes in higher simplices count as having maximal dimension 2
    par = T.node_participation(_graph(), max_dim=2)
    vals = _node_values()
    return {
        "columns": list(par.columns),
        "eq": S.node_stats_participation(par, vals),
        "ge": S.node_stats_participation(par, vals, operator.ge),
    }


@case("stats.node_stats_participation.undirected_and_reciprocal_tables")
def _():
    adj, vals = _graph(), _node_values()
    undirected = T.node_participation(adj, simplex_type="undirected")
    reciprocal = T.node_participation(adj, simplex_type="reciprocal")
    return {
        "undirected": S.node_stats_participation(undirected, vals),
        "undirected_ge": S.node_stats_participation(undirected, vals, operator.ge),
        "reciprocal": S.node_stats_participation(reciprocal, vals),
        "reciprocal_ge": S.node_stats_participation(reciprocal, vals, operator.ge),
    }


@case("stats.node_stats_participation.table_of_some_nodes")
def _():
    # participation known for part of the nodes only, values for all of them (looked up by label)
    par = T.node_participation(_graph())
    subset = par.iloc[::3]
    return {
        "eq": S.node_stats_participation(subset, _node_values(n=60, seed=7)),
        "ge": S.node_stats_participation(subset, _node_values(n=60, seed=7), operator.ge),
    }


@case("stats.node_stats_participation.values_labelled_by_gid")
def _():
    par = T.node_participation(_graph())
    base = _node_values()
    par.index = pd.Index(np.arange(40) + 5000, name="gid")
    vals = pd.Series(base.to_numpy(), index=par.index)
    return S.node_stats_participation(par, vals, operator.ge)


@case("stats.node_stats_participation.inputs_are_not_modified")
def _():
    par, vals = T.node_participation(_graph()), _node_values()
    before = (par.copy(), vals.copy())
    result = S.node_stats_participation(par, vals, operator.ge)
    return (
        result,
        par,
        vals,
        par.equals(before[0]),
        vals.equals(before[1]),
        vals.name,
        list(par.columns),
    )


@case("stats.node_stats_participation.hierarchy")
def _():
    # every maximal dimension 0..3 occurs, so no row of the table is empty (dimension 0 = the isolated nodes)
    par = T.node_participation(_hierarchy())
    vals = pd.Series(np.arange(14, dtype=float) ** 1.5, index=pd.RangeIndex(14))
    return {
        "eq": S.node_stats_participation(par, vals),
        "ge": S.node_stats_participation(par, vals, operator.ge),
    }


@case("stats.node_stats_participation.tutorial")
def _():
    adj, nodes = tutorial_connectome(500)
    par = T.node_participation(adj)
    return {
        "eq_depth": S.node_stats_participation(par, nodes["depth"]),
        "ge_y": S.node_stats_participation(par, nodes["y"], operator.ge),
    }


@case("stats.node_stats_participation.unknown_dimension")
def _():
    # dimension 7 is not a column of the table: raised before np.product is reached
    return S.node_stats_participation(T.node_participation(_graph()), _node_values(), dims=[7])


@case("stats.node_stats_participation.node_without_value")
def _():
    # nodes 0..39 take part in simplices but values only exist for 0..9 (also raised before np.product)
    return S.node_stats_participation(T.node_participation(_graph()), _node_values(n=10), dims=[2])


# ============================================================================================== edges


def _edge_table(adj=None, **kw):
    adj = _graph() if adj is None else adj
    par = T.edge_participation(adj, **kw)
    return par, _edge_values(par.index)


@case("stats.edge_stats_participation.maximal_dimension_equals")
def _():
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals)


@case("stats.edge_stats_participation.maximal_dimension_at_least")
def _():
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals, condition=operator.ge)


@case("stats.edge_stats_participation.condition_positional")
def _():
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals, operator.ge, [0, 2, 3])


@case("stats.edge_stats_participation.keyword_arguments")
def _():
    par, vals = _edge_table()
    return S.edge_stats_participation(
        participation=par, vals=vals, condition=operator.ge, dims=[0, 1, 2]
    )


@case("stats.edge_stats_participation.dims_with_zero")
def _():
    # dimension 0 has to be among dims because the result always drops its row
    par, vals = _edge_table()
    return {
        "list_eq": S.edge_stats_participation(par, vals, dims=[0, 3, 1]),
        "array_ge": S.edge_stats_participation(
            par, vals, condition=operator.ge, dims=np.array([0, 2, 4])
        ),
        "range_eq": S.edge_stats_participation(par, vals, dims=range(0, 3)),
    }


@case("stats.edge_stats_participation.only_dimension_zero")
def _():
    # no edge has maximal dimension 0: the dimension 0 row is all NaN and is dropped, an empty table remains
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals, dims=[0])


@case("stats.edge_stats_participation.nan_values")
def _():
    par = T.edge_participation(_graph())
    vals = _edge_values(par.index, nans=range(0, len(par), 7))
    return {
        "eq": S.edge_stats_participation(par, vals),
        "ge": S.edge_stats_participation(par, vals, operator.ge),
    }


@case("stats.edge_stats_participation.other_comparisons")
def _():
    par, vals = _edge_table()
    return {
        op.__name__: S.edge_stats_participation(par, vals, condition=op, dims=[0, 1, 2, 3])
        for op in (operator.gt, operator.le, operator.lt, operator.ne)
    }


@case("stats.edge_stats_participation.maximal_simplices_table")
def _():
    par, vals = _edge_table(max_simplices=True)
    return {
        "eq": S.edge_stats_participation(par, vals),
        "ge": S.edge_stats_participation(par, vals, operator.ge),
    }


@case("stats.edge_stats_participation.undirected_and_reciprocal_tables")
def _():
    undirected, uvals = _edge_table(simplex_type="undirected")
    reciprocal, rvals = _edge_table(simplex_type="reciprocal")
    return {
        "undirected": S.edge_stats_participation(undirected, uvals),
        "undirected_ge": S.edge_stats_participation(undirected, uvals, operator.ge),
        "reciprocal": S.edge_stats_participation(reciprocal, rvals),
    }


@case("stats.edge_stats_participation.truncated_participation")
def _():
    par, vals = _edge_table(max_dim=2)
    return {
        "columns": list(par.columns),
        "eq": S.edge_stats_participation(par, vals),
        "ge": S.edge_stats_participation(par, vals, operator.ge),
    }


@case("stats.edge_stats_participation.multiindex_edges")
def _():
    # edges labelled by a two-level index instead of tuples in an object index
    par, vals = _edge_table()
    mi = pd.MultiIndex.from_tuples(list(par.index), names=["pre", "post"])
    par_mi, vals_mi = par.copy(), vals.copy()
    par_mi.index, vals_mi.index = mi, mi
    return {
        "both": S.edge_stats_participation(par_mi, vals_mi, operator.ge),
        "values_only": S.edge_stats_participation(par, pd.Series(vals.to_numpy(), index=mi)),
    }


@case("stats.edge_stats_participation.values_for_more_edges_than_listed")
def _():
    # values known for more edges than the table lists
    par, _ = _edge_table()
    listed = set(par.index)
    extra = pd.Index(
        list(par.index) + [e for e in [(0, 39), (39, 0), (1, 38), (38, 1)] if e not in listed],
        tupleize_cols=False,
    )
    assert len(extra) > len(par)
    return S.edge_stats_participation(par, _edge_values(extra), operator.ge)


@case("stats.edge_stats_participation.inputs_are_not_modified")
def _():
    par, vals = _edge_table()
    before = (par.copy(), vals.copy())
    result = S.edge_stats_participation(par, vals, operator.ge)
    return (
        result,
        par,
        vals,
        par.equals(before[0]),
        vals.equals(before[1]),
        vals.name,
        list(par.columns),
    )


@case("stats.edge_stats_participation.hierarchy")
def _():
    par, vals = _edge_table(_hierarchy())
    return {
        "eq": S.edge_stats_participation(par, vals),
        "ge": S.edge_stats_participation(par, vals, operator.ge),
    }


@case("stats.edge_stats_participation.tutorial")
def _():
    adj, nodes = tutorial_connectome(500)
    par = T.edge_participation(adj)
    depth = nodes["depth"].to_numpy()
    vals = pd.Series(
        [abs(depth[a] - depth[b]) for a, b in par.index], index=par.index, name="depth_difference"
    )
    return {
        "eq": S.edge_stats_participation(par, vals),
        "ge": S.edge_stats_participation(par, vals, operator.ge),
    }


@case("stats.edge_stats_participation.dims_without_zero")
def _():
    # defect: the final .drop(0) fails when dimension 0 is not among dims
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals, dims=[1, 3])


@case("stats.edge_stats_participation.no_dims")
def _():
    # defect: nothing is analysed, then .drop(0) fails on the empty table
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals, dims=[])


@case("stats.edge_stats_participation.unknown_dimension")
def _():
    par, vals = _edge_table()
    return S.edge_stats_participation(par, vals, dims=[6])
