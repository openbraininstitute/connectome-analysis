"""Simplex counting built on ``pyflagsercount.flagser_count``: ``_flagser_counts``, ``simplex_counts`` (more variants than
``topology_basic``), ``normalized_simplex_counts``, ``normalised_simplex_count``, ``node_participation`` and
``edge_participation``.

The order of the simplex *lists* flagser returns depends on its thread count, so the cases that return lists run with a
fixed one (an explicit ``threads`` or the function's default).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.data import node_table, weighted

from cases.topology_helpers import (
    complete,
    cycle,
    explicit_zeros,
    from_edges,
    rich,
    structured,
    transitive,
    with_loops,
)
from cases.topology_helpers import (
    contained_case as case,
)  # runs each case with a throw-away temp directory
from connalysis.network import topology as T

FORMATS = ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")
TYPES = ("directed", "undirected", "reciprocal")


def _small():
    """25 nodes, counts (25, 84, 65, 15, 2): a feedforward 5-clique and 4-clique, a reciprocal triangle, random edges."""
    return rich(25, 0.08, seed=3, sizes=(5, 4), rc_sizes=(3,))


def _medium():
    """40 nodes, directed counts (40, 184, 154, 92, 21, 4), undirected (40, 128, 68, 27, 7, 1), reciprocal (40, 56, 8, 1)."""
    return rich(40, 0.06, seed=10)


def _nodes_file(nodes):
    """``.npy`` file of node ids (inside the case's throw-away temp directory) for the ``vertices_todo`` argument."""
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "nodes.npy")
    np.save(path, np.asarray(nodes), allow_pickle=False)
    return path


def _degenerate_graphs():
    return {
        "empty": from_edges(6, []),
        "single_node": from_edges(1, []),
        "no_nodes": from_edges(0, []),
        "one_way_pair": from_edges(2, [(0, 1)]),
        "reciprocal_pair": from_edges(2, [(0, 1), (1, 0)]),
        "directed_3_cycle": cycle(3),
        "complete_5": complete(5),
        "transitive_5": transitive(5),
        "structured": structured(),
    }


# ----------------------------------------------------------------------------- _flagser_counts


@case("topology._flagser_counts.default")
def _():
    return T._flagser_counts(_small())


@case("topology._flagser_counts.max_simplices")
def _():
    return T._flagser_counts(_small(), max_simplices=True)


@case("topology._flagser_counts.count_node_participation")
def _():
    return T._flagser_counts(_small(), count_node_participation=True)


@case("topology._flagser_counts.count_node_participation_max_simplices")
def _():
    return T._flagser_counts(_small(), count_node_participation=True, max_simplices=True)


@case("topology._flagser_counts.list_simplices")
def _():
    return T._flagser_counts(_small(), list_simplices=True, threads=1)


@case("topology._flagser_counts.list_simplices_default_threads")
def _():
    return T._flagser_counts(_small(), list_simplices=True)


@case("topology._flagser_counts.list_simplices_order_per_thread_count")
def _():
    return {
        threads: T._flagser_counts(_small(), list_simplices=True, threads=threads)["simplices"]
        for threads in (1, 2, 3, 4)
    }


@case("topology._flagser_counts.list_max_simplices")
def _():
    return T._flagser_counts(_small(), list_simplices=True, max_simplices=True, threads=1)


@case("topology._flagser_counts.edge_containment")
def _():
    return T._flagser_counts(_small(), edge_containment=True)


@case("topology._flagser_counts.edge_containment_max_simplices")
def _():
    return T._flagser_counts(_small(), edge_containment=True, max_simplices=True)


@case("topology._flagser_counts.all_flags")
def _():
    return T._flagser_counts(
        _small(),
        max_simplices=True,
        count_node_participation=True,
        list_simplices=True,
        edge_containment=True,
        threads=1,
        max_dim=3,
    )


@case("topology._flagser_counts.max_dim_values")
def _():
    adj = _small()
    return {
        md: T._flagser_counts(adj, count_node_participation=True, max_dim=md)
        for md in (-1, 0, 1, 2, 3, 4, 10, -2)
    }


@case("topology._flagser_counts.max_dim_with_lists_and_max_simplices")
def _():
    adj = _small()
    return {
        md: T._flagser_counts(adj, list_simplices=True, max_simplices=True, threads=1, max_dim=md)
        for md in (0, 1, 2, 3)
    }


@case("topology._flagser_counts.threads_do_not_change_counts")
def _():
    adj = _small()
    return {
        threads: T._flagser_counts(
            adj,
            count_node_participation=True,
            edge_containment=True,
            max_simplices=True,
            threads=threads,
        )
        for threads in (1, 2, 4, 8, 16)
    }


@case("topology._flagser_counts.vertices_todo")
def _():
    adj = _small()
    path = _nodes_file([1, 4, 7, 10])
    return {
        "counts": T._flagser_counts(adj, vertices_todo=path),
        "max_simplices": T._flagser_counts(adj, vertices_todo=path, max_simplices=True),
        "participation": T._flagser_counts(adj, vertices_todo=path, count_node_participation=True),
        "lists": T._flagser_counts(adj, vertices_todo=path, list_simplices=True, threads=1),
        "edges": T._flagser_counts(adj, vertices_todo=path, edge_containment=True),
    }


@case("topology._flagser_counts.vertices_todo_missing_file")
def _():
    return T._flagser_counts(_small(), vertices_todo="/nonexistent-directory/missing.npy")


@case("topology._flagser_counts.input_formats")
def _():
    adj = _small()
    out = {fmt: T._flagser_counts(adj.asformat(fmt))["simplex_counts"] for fmt in FORMATS}
    out["dense_bool"] = T._flagser_counts(adj.toarray())
    out["dense_int"] = T._flagser_counts(adj.toarray().astype(int))
    out["dense_float"] = T._flagser_counts(adj.toarray().astype(float) * 0.5)
    out["np.matrix"] = T._flagser_counts(np.matrix(adj.toarray()))["simplex_counts"]
    out["csr_array"] = T._flagser_counts(sp.csr_array(adj))["simplex_counts"]
    return out


@case("topology._flagser_counts.nested_list_is_not_accepted")
def _():
    # unlike simplex_counts, _flagser_counts calls adjacency.astype(...) before converting to a sparse matrix
    return T._flagser_counts([[0, 1], [1, 0]])


@case("topology._flagser_counts.weights_and_explicit_zeros_are_cast_to_edges")
def _():
    return {
        "integer_weights": T._flagser_counts(weighted(25, 0.2, seed=60)),
        "real_weights": T._flagser_counts(weighted(25, 0.2, seed=61, integers=False)),
        "explicit_zeros": T._flagser_counts(explicit_zeros(_small(), 7)),
    }


@case("topology._flagser_counts.diagonal_is_counted_as_loops")
def _():
    # The log message says the diagonal is ignored, but flagser takes the loops as edges (counts differ from the same
    # graph without them).
    adj = _small()
    return {
        "without": T._flagser_counts(adj)["simplex_counts"],
        "with": T._flagser_counts(with_loops(adj, (0, 5, 11, 12))),
    }


@case("topology._flagser_counts.degenerate_graphs")
def _():
    return {
        name: T._flagser_counts(
            adj,
            count_node_participation=True,
            list_simplices=True,
            edge_containment=True,
            threads=1,
        )
        for name, adj in _degenerate_graphs().items()
    }


@case("topology._flagser_counts.degenerate_graphs_max_simplices")
def _():
    return {
        name: T._flagser_counts(
            adj, max_simplices=True, count_node_participation=True, list_simplices=True, threads=1
        )
        for name, adj in _degenerate_graphs().items()
    }


# ----------------------------------------------------------------------------- simplex_counts


@case("topology.simplex_counts.types_and_maximality")
def _():
    adj = _medium()
    return {
        (t, ms): T.simplex_counts(adj, simplex_type=t, max_simplices=ms)
        for t in TYPES
        for ms in (False, True)
    }


@case("topology.simplex_counts.unknown_simplex_type_is_directed")
def _():
    adj = _medium()
    return [
        T.simplex_counts(adj, simplex_type="foo"),
        T.simplex_counts(adj, simplex_type=None),
        T.simplex_counts(adj, simplex_type="foo", max_simplices=True),
    ]


@case("topology.simplex_counts.max_dim_values")
def _():
    adj = _medium()
    return {md: T.simplex_counts(adj, max_dim=md) for md in (-2, -1, 0, 1, 2, 3, 4, 5, 6, 100)}


@case("topology.simplex_counts.max_dim_with_max_simplices_and_types")
def _():
    adj = _medium()
    return {
        (t, md): T.simplex_counts(adj, simplex_type=t, max_simplices=True, max_dim=md)
        for t in TYPES
        for md in (0, 1, 2, 3, 6)
    }


@case("topology.simplex_counts.threads_do_not_change_counts")
def _():
    adj = _medium()
    return {threads: T.simplex_counts(adj, threads=threads) for threads in (1, 2, 4, 8, 16)}


@case("topology.simplex_counts.input_formats")
def _():
    adj = _medium()
    out = {fmt: T.simplex_counts(adj.asformat(fmt)) for fmt in FORMATS}
    out["csr_array"] = T.simplex_counts(sp.csr_array(adj))
    out["dense_bool"] = T.simplex_counts(adj.toarray())
    out["dense_int"] = T.simplex_counts(adj.toarray().astype(int))
    out["dense_float"] = T.simplex_counts(adj.toarray().astype(float))
    out["np.matrix"] = T.simplex_counts(np.matrix(adj.toarray()))
    out["nested_list"] = T.simplex_counts(adj.toarray().tolist())
    return out


@case("topology.simplex_counts.input_formats_other_types")
def _():
    adj = _medium()
    return {
        (fmt, t): T.simplex_counts(adj.asformat(fmt), simplex_type=t)
        for fmt in ("csc", "coo", "lil")
        for t in ("undirected", "reciprocal")
    }


@case("topology.simplex_counts.weights_are_ignored")
def _():
    return [
        T.simplex_counts(weighted(30, 0.2, seed=62)),
        T.simplex_counts(weighted(30, 0.2, seed=63, integers=False)),
    ]


@case("topology.simplex_counts.signed_weights_are_edges")
def _():
    adj = _medium().astype(float).tocsr()
    adj.data[::2] = -1.0
    return [T.simplex_counts(adj), T.simplex_counts(adj, simplex_type="reciprocal")]


@case("topology.simplex_counts.explicit_zeros_are_not_edges")
def _():
    adj = explicit_zeros(_medium(), 30)
    return {t: T.simplex_counts(adj, simplex_type=t) for t in TYPES}


@case("topology.simplex_counts.stored_zero_on_diagonal_is_accepted")
def _():
    coo = _medium().astype(float).tocoo()
    adj = sp.csr_matrix(
        (np.append(coo.data, 0.0), (np.append(coo.row, 3), np.append(coo.col, 3))), shape=coo.shape
    )
    return T.simplex_counts(adj)


@case("topology.simplex_counts.nodes_as_list")
def _():
    return T.simplex_counts(_medium(), nodes=[0, 1, 2, 3, 4, 5])


@case("topology.simplex_counts.nodes_unsorted")
def _():
    adj = _medium()
    return [
        T.simplex_counts(adj, nodes=np.array([17, 3, 25, 0, 9])),
        T.simplex_counts(adj, nodes=np.array([0, 3, 9, 17, 25])),
    ]


@case("topology.simplex_counts.nodes_duplicated_are_counted_twice")
def _():
    adj = _medium()
    return [
        T.simplex_counts(adj, nodes=np.array([3, 3, 4])),
        T.simplex_counts(adj, nodes=np.array([3, 4])),
    ]


@case("topology.simplex_counts.nodes_empty")
def _():
    adj = _medium()
    return [T.simplex_counts(adj, nodes=np.array([], dtype=int)), T.simplex_counts(adj, nodes=[])]


@case("topology.simplex_counts.nodes_all_equals_no_nodes")
def _():
    adj = _medium()
    return [T.simplex_counts(adj, nodes=np.arange(40)), T.simplex_counts(adj)]


@case("topology.simplex_counts.nodes_single")
def _():
    adj = _medium()
    return {int(v): T.simplex_counts(adj, nodes=np.array([v])) for v in (0, 5, 22, 39)}


@case("topology.simplex_counts.nodes_range_and_int32")
def _():
    adj = _medium()
    return [
        T.simplex_counts(adj, nodes=range(8)),
        T.simplex_counts(adj, nodes=np.arange(8, dtype=np.int32)),
        T.simplex_counts(adj, nodes=np.arange(8, dtype=np.int64)),
    ]


@case("topology.simplex_counts.nodes_as_pandas_objects")
def _():
    adj = _medium()
    return [
        T.simplex_counts(adj, nodes=pd.Index([3, 17, 25, 0, 9])),
        T.simplex_counts(adj, nodes=pd.Series([3, 17, 25, 0, 9])),
        T.simplex_counts(adj, nodes=np.array([3, 17, 25, 0, 9])),
    ]


@case("topology.simplex_counts.nodes_with_max_simplices_max_dim_and_threads")
def _():
    adj = _medium()
    nodes = np.arange(0, 40, 4)
    return {
        "max_simplices": T.simplex_counts(adj, nodes=nodes, max_simplices=True),
        "max_dim": T.simplex_counts(adj, nodes=nodes, max_dim=2),
        "threads": T.simplex_counts(adj, nodes=nodes, threads=3),
        "max_simplices_max_dim": T.simplex_counts(adj, nodes=nodes, max_simplices=True, max_dim=3),
    }


@case("topology.simplex_counts.nodes_rejected_for_undirected")
def _():
    return T.simplex_counts(_medium(), nodes=np.arange(5), simplex_type="undirected")


@case("topology.simplex_counts.nodes_rejected_for_reciprocal")
def _():
    return T.simplex_counts(_medium(), nodes=np.arange(5), simplex_type="reciprocal")


@case("topology.simplex_counts.nodes_boolean_mask")
def _():
    return T.simplex_counts(_medium(), nodes=np.arange(40) % 2 == 0)


@case("topology.simplex_counts.degenerate_graphs")
def _():
    return {
        (name, t): T.simplex_counts(adj, simplex_type=t)
        for name, adj in _degenerate_graphs().items()
        for t in TYPES
    }


@case("topology.simplex_counts.degenerate_graphs_max_simplices")
def _():
    return {
        (name, t): T.simplex_counts(adj, simplex_type=t, max_simplices=True)
        for name, adj in _degenerate_graphs().items()
        for t in TYPES
    }


@case("topology.simplex_counts.non_square")
def _():
    # zero diagonal, so the squareness assertion is the one that fires
    return T.simplex_counts(
        sp.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=bool))
    )


@case("topology.simplex_counts.diagonal_rejected_for_undirected")
def _():
    adj = _medium().tolil()
    adj[7, 7] = True
    return T.simplex_counts(adj.tocsr(), simplex_type="undirected")


@case("topology.simplex_counts.extra_arguments_are_ignored")
def _():
    adj = _medium()
    return T.simplex_counts(adj, node_properties=node_table(40), unknown_option=5)


@case("topology.simplex_counts.input_is_not_modified")
def _():
    adj = _medium().toarray()
    before = adj.copy()
    return T.simplex_counts(adj, simplex_type="undirected"), adj, before


# ----------------------------------------------------------------------------- normalized_simplex_counts


@case("topology.normalized_simplex_counts.default")
def _():
    return T.normalized_simplex_counts(_medium())


@case("topology.normalized_simplex_counts.max_simplices")
def _():
    return T.normalized_simplex_counts(_medium(), max_simplices=True)


@case("topology.normalized_simplex_counts.max_dim_values")
def _():
    adj = _medium()
    return {md: T.normalized_simplex_counts(adj, max_dim=md) for md in (-2, -1, 0, 1, 2, 3, 4, 10)}


@case("topology.normalized_simplex_counts.max_simplices_with_max_dim")
def _():
    adj = _medium()
    return {
        md: T.normalized_simplex_counts(adj, max_simplices=True, max_dim=md) for md in (1, 2, 3)
    }


@case("topology.normalized_simplex_counts.threads")
def _():
    adj = _medium()
    return [
        T.normalized_simplex_counts(adj, threads=1),
        T.normalized_simplex_counts(adj, threads=5),
    ]


@case("topology.normalized_simplex_counts.input_formats")
def _():
    adj = _medium()
    return {
        "dense": T.normalized_simplex_counts(adj.toarray()),
        "csc": T.normalized_simplex_counts(adj.tocsc()),
        "weighted": T.normalized_simplex_counts(weighted(30, 0.2, seed=64)),
    }


@case("topology.normalized_simplex_counts.complete_and_transitive")
def _():
    # complete graph: every undirected simplex carries all (k+1)! directed ones; the tournament has one per undirected
    return {
        "complete": T.normalized_simplex_counts(complete(6)),
        "transitive": T.normalized_simplex_counts(transitive(6)),
        "complete_max": T.normalized_simplex_counts(complete(6), max_simplices=True),
        "transitive_max": T.normalized_simplex_counts(transitive(6), max_simplices=True),
    }


@case("topology.normalized_simplex_counts.degenerate_graphs")
def _():
    return {name: T.normalized_simplex_counts(adj) for name, adj in _degenerate_graphs().items()}


@case("topology.normalized_simplex_counts.structured_max_simplices")
def _():
    return T.normalized_simplex_counts(structured(), max_simplices=True)


@case("topology.normalized_simplex_counts.simplex_type_argument_collides")
def _():
    # simplex_type travels in **kwargs and meets the explicit simplex_type='undirected' of the denominator call
    return T.normalized_simplex_counts(_medium(), simplex_type="directed")


@case("topology.normalized_simplex_counts.misspelt_simple_type_collides")
def _():
    # the numerator call passes the misspelt keyword ``simple_type='directed'``; a caller's one then collides with it
    return T.normalized_simplex_counts(_medium(), simple_type="undirected")


@case("topology.normalized_simplex_counts.simple_type_is_otherwise_swallowed")
def _():
    # without the collision the misspelt keyword is absorbed by simplex_counts(**kwargs) and the numerator stays directed
    adj = _medium()
    return [T.normalized_simplex_counts(adj), T.normalized_simplex_counts(adj, unknown_option=1)]


@case("topology.normalized_simplex_counts.nodes_rejected")
def _():
    # nodes reaches the undirected denominator call, which only accepts it for directed simplices
    return T.normalized_simplex_counts(_medium(), nodes=np.arange(5))


@case("topology.normalized_simplex_counts.diagonal_rejected")
def _():
    return T.normalized_simplex_counts((_medium() + sp.eye(40)).tocsr())


# ----------------------------------------------------------------------------- normalised_simplex_count (old name)


@case("topology.normalised_simplex_count.default_dim")
def _():
    return T.normalised_simplex_count(_medium())


@case("topology.normalised_simplex_count.dims")
def _():
    adj = _medium()
    return {dim: T.normalised_simplex_count(adj, dim=dim) for dim in (1, 2, 3, 4, 5)}


@case("topology.normalised_simplex_count.dense_and_transitive")
def _():
    return [
        T.normalised_simplex_count(_medium().toarray(), dim=2),
        T.normalised_simplex_count(transitive(6), dim=2),
    ]


@case("topology.normalised_simplex_count.dim_beyond_the_complex")
def _():
    return T.normalised_simplex_count(_medium(), dim=9)


@case("topology.normalised_simplex_count.dim_zero")
def _():
    return T.normalised_simplex_count(_medium(), dim=0)


# ----------------------------------------------------------------------------- node_participation


@case("topology.node_participation.types_and_maximality")
def _():
    adj = _small()
    return {
        (t, ms): T.node_participation(adj, simplex_type=t, max_simplices=ms)
        for t in TYPES
        for ms in (False, True)
    }


@case("topology.node_participation.unknown_simplex_type_is_directed")
def _():
    adj = _small()
    return [
        T.node_participation(adj, simplex_type="foo"),
        T.node_participation(adj, simplex_type="foo", max_simplices=True),
    ]


@case("topology.node_participation.max_dim_values")
def _():
    adj = _small()
    return {md: T.node_participation(adj, max_dim=md) for md in (-2, -1, 0, 1, 2, 3, 4, 10)}


@case("topology.node_participation.max_dim_with_max_simplices_and_types")
def _():
    adj = _small()
    return {
        (t, md): T.node_participation(adj, simplex_type=t, max_simplices=True, max_dim=md)
        for t in TYPES
        for md in (1, 2, 3)
    }


@case("topology.node_participation.threads")
def _():
    adj = _small()
    return {threads: T.node_participation(adj, threads=threads) for threads in (1, 2, 8, 16)}


@case("topology.node_participation.input_formats")
def _():
    adj = _small()
    out = {fmt: T.node_participation(adj.asformat(fmt)) for fmt in FORMATS}
    out["dense_bool"] = T.node_participation(adj.toarray())
    out["dense_float"] = T.node_participation(adj.toarray().astype(float))
    out["np.matrix"] = T.node_participation(np.matrix(adj.toarray()))
    out["csr_array"] = T.node_participation(sp.csr_array(adj))
    out["coo_array"] = T.node_participation(sp.coo_array(adj), simplex_type="reciprocal")
    return out


@case("topology.node_participation.weights_and_explicit_zeros")
def _():
    return {
        "integer_weights": T.node_participation(weighted(20, 0.25, seed=65)),
        "real_weights": T.node_participation(weighted(20, 0.25, seed=66, integers=False)),
        "explicit_zeros": T.node_participation(explicit_zeros(_small(), 9)),
    }


@case("topology.node_participation.degenerate_graphs")
def _():
    return {
        (name, t): T.node_participation(adj, simplex_type=t)
        for name, adj in _degenerate_graphs().items()
        for t in ("directed", "reciprocal")
    }


@case("topology.node_participation.degenerate_graphs_max_simplices")
def _():
    return {
        name: T.node_participation(adj, max_simplices=True, simplex_type=t)
        for name, adj in _degenerate_graphs().items()
        for t in ("directed", "undirected")
    }


@case("topology.node_participation.hub_graph")
def _():
    # a hub sourcing a fan and a triangle plus an isolated node; the participation is very uneven
    adj = from_edges(
        12,
        [(0, k) for k in range(1, 8)]
        + [(1, 2), (2, 3), (1, 3), (9, 0), (9, 1), (1, 0), (3, 2), (5, 0)],
    )
    return {t: T.node_participation(adj, simplex_type=t) for t in TYPES}


@case("topology.node_participation.diagonal_rejected")
def _():
    return T.node_participation((_small() + sp.eye(25)).tocsr())


@case("topology.node_participation.non_square")
def _():
    return T.node_participation(
        sp.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=bool))
    )


@case("topology.node_participation.extra_arguments_are_ignored")
def _():
    return T.node_participation(_small(), node_properties=node_table(25), unknown_option=1)


# ----------------------------------------------------------------------------- edge_participation


@case("topology.edge_participation.default")
def _():
    return T.edge_participation(_small())


@case("topology.edge_participation.max_simplices")
def _():
    return T.edge_participation(_small(), max_simplices=True)


@case("topology.edge_participation.types_and_maximality")
def _():
    adj = _small()
    return {
        (t, ms): T.edge_participation(adj, simplex_type=t, max_simplices=ms)
        for t in TYPES
        for ms in (False, True)
    }


@case("topology.edge_participation.unknown_simplex_type_is_directed")
def _():
    adj = _small()
    return [
        T.edge_participation(adj, simplex_type="foo"),
        T.edge_participation(adj, simplex_type=None),
    ]


@case("topology.edge_participation.max_dim_values")
def _():
    adj = _small()
    return {md: T.edge_participation(adj, max_dim=md) for md in (-2, -1, 0, 1, 2, 3, 4, 10)}


@case("topology.edge_participation.return_simplex_counts")
def _():
    adj = _small()
    return {
        (t, ms): T.edge_participation(
            adj, simplex_type=t, max_simplices=ms, return_simplex_counts=True
        )
        for t in TYPES
        for ms in (False, True)
    }


@case("topology.edge_participation.return_simplex_counts_with_max_dim")
def _():
    return [
        T.edge_participation(_small(), max_dim=md, return_simplex_counts=True) for md in (0, 1, 2)
    ]


@case("topology.edge_participation.threads")
def _():
    adj = _small()
    return {threads: T.edge_participation(adj, threads=threads) for threads in (1, 2, 8)}


@case("topology.edge_participation.input_formats")
def _():
    adj = _small()
    out = {fmt: T.edge_participation(adj.asformat(fmt)) for fmt in ("csc", "coo", "lil", "dok")}
    out["dense_bool"] = T.edge_participation(adj.toarray())
    out["dense_int"] = T.edge_participation(adj.toarray().astype(int))
    out["np.matrix"] = T.edge_participation(np.matrix(adj.toarray()))
    out["csr_array"] = T.edge_participation(sp.csr_array(adj))
    out["coo_array_undirected"] = T.edge_participation(sp.coo_array(adj), simplex_type="undirected")
    return out


@case("topology.edge_participation.weights_and_explicit_zeros")
def _():
    return {
        "integer_weights": T.edge_participation(weighted(20, 0.25, seed=67)),
        "real_weights": T.edge_participation(weighted(20, 0.25, seed=68, integers=False)),
        "explicit_zeros": T.edge_participation(explicit_zeros(_small(), 9)),
    }


@case("topology.edge_participation.edges_follow_csr_order_of_the_matrix")
def _():
    # edges are listed by source then target, whatever the storage order of the input
    adj = _small()
    coo = adj.tocoo()
    shuffled = sp.coo_matrix((coo.data[::-1], (coo.row[::-1], coo.col[::-1])), shape=adj.shape)
    return [T.edge_participation(shuffled), T.edge_participation(adj)]


@case("topology.edge_participation.degenerate_graphs")
def _():
    return {
        (name, t): T.edge_participation(adj, simplex_type=t, return_simplex_counts=True)
        for name, adj in _degenerate_graphs().items()
        for t in ("directed", "reciprocal")
    }


@case("topology.edge_participation.degenerate_graphs_max_simplices")
def _():
    return {
        name: T.edge_participation(adj, max_simplices=True)
        for name, adj in _degenerate_graphs().items()
    }


@case("topology.edge_participation.diagonal_rejected")
def _():
    return T.edge_participation((_small() + sp.eye(25)).tocsr())


@case("topology.edge_participation.non_square")
def _():
    return T.edge_participation(
        sp.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=bool))
    )


@case("topology.edge_participation.extra_arguments_are_ignored")
def _():
    return T.edge_participation(
        _small(), node_properties=node_table(25), verbose=True, unknown_option=1
    )
