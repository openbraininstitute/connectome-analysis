"""The main analyses of ``topology`` on the real connectome of ``tutorials/`` (6719 neurons ordered by layer).

The first neurons are a dense, spatially local column of layer 3; the windows further down mix excitatory and inhibitory
neurons of other layers and have different densities.  Sizes are chosen so that every case runs in well under a second.
"""

import numpy as np
from support.data import tutorial_connectome

from cases.topology_helpers import (
    contained_case as case,
)  # runs each case with a throw-away temp directory
from connalysis.network import topology as T

TYPES = ("directed", "undirected", "reciprocal")


def _first(n):
    return tutorial_connectome(n)


def _window(start, n):
    """Neurons ``start .. start + n`` of the connectome with their properties."""
    adj, nodes = tutorial_connectome()
    return adj[start : start + n, start : start + n].tocsr(), nodes.iloc[
        start : start + n
    ].reset_index(drop=True)


# ----------------------------------------------------------------------------- matrices and degrees


@case("topology.realdata.rc_submatrix_and_underlying_undirected_matrix")
def _():
    adj, _ = _first(300)
    return T.rc_submatrix(adj), T.underlying_undirected_matrix(adj)


@case("topology.realdata.node_degree")
def _():
    adj, _ = _first(700)
    return {
        "total": T.node_degree(adj),
        "in_out": T.node_degree(adj, direction=("IN", "OUT")),
        "dense_in": T.node_degree(adj.toarray(), direction="IN"),
    }


@case("topology.realdata.node_degree_window")
def _():
    adj, _ = _window(1500, 400)
    return T.node_degree(adj, direction=("IN", "OUT"))


# ----------------------------------------------------------------------------- counts


@case("topology.realdata.simplex_counts_first_neurons")
def _():
    adj, _ = _first(700)
    out = {
        (t, ms): T.simplex_counts(adj, simplex_type=t, max_simplices=ms)
        for t in TYPES
        for ms in (False, True)
    }
    out["max_dim_3"] = T.simplex_counts(adj, max_dim=3)
    return out


@case("topology.realdata.simplex_counts_windows")
def _():
    out = {}
    for start in (1500, 3000, 4500, 6000):
        adj, _ = _window(start, 400)
        out[start] = {t: T.simplex_counts(adj, simplex_type=t) for t in TYPES}
    return out


@case("topology.realdata.simplex_counts_dense_window")
def _():
    # a denser part of the connectome (layer 2): simplices up to dimension 6
    adj, _ = _window(4500, 700)
    return [
        T.simplex_counts(adj),
        T.simplex_counts(adj, max_simplices=True),
        T.simplex_counts(adj, simplex_type="reciprocal"),
        T.simplex_counts(adj, simplex_type="undirected"),
    ]


@case("topology.realdata.simplex_counts_from_inhibitory_sources")
def _():
    adj, nodes = _window(1500, 400)
    inhibitory = np.flatnonzero((nodes["synapse_class"] == "INH").to_numpy())
    return {
        "inhibitory": T.simplex_counts(adj, nodes=inhibitory),
        "inhibitory_max": T.simplex_counts(adj, nodes=inhibitory, max_simplices=True),
        "everything": T.simplex_counts(adj),
    }


@case("topology.realdata.normalized_simplex_counts")
def _():
    adj, _ = _first(700)
    return {
        "all": T.normalized_simplex_counts(adj),
        "maximal": T.normalized_simplex_counts(adj, max_simplices=True),
        "max_dim_3": T.normalized_simplex_counts(adj, max_dim=3),
        "dim_2": T.normalised_simplex_count(adj, dim=2),
    }


@case("topology.realdata.normalized_simplex_counts_window")
def _():
    adj, _ = _window(4500, 400)
    return T.normalized_simplex_counts(adj)


# ----------------------------------------------------------------------------- participation


@case("topology.realdata.node_participation")
def _():
    adj, _ = _first(400)
    return {
        "directed": T.node_participation(adj),
        "reciprocal": T.node_participation(adj, simplex_type="reciprocal"),
        "undirected_maximal": T.node_participation(
            adj, simplex_type="undirected", max_simplices=True
        ),
    }


@case("topology.realdata.node_participation_window")
def _():
    adj, _ = _window(3000, 400)
    return T.node_participation(adj, max_dim=3)


@case("topology.realdata.edge_participation")
def _():
    adj, _ = _first(300)
    return T.edge_participation(adj)


@case("topology.realdata.edge_participation_variants")
def _():
    adj, _ = _first(200)
    return {
        "maximal": T.edge_participation(adj, max_simplices=True),
        "reciprocal_with_counts": T.edge_participation(
            adj, simplex_type="reciprocal", return_simplex_counts=True
        ),
        "undirected": T.edge_participation(adj, simplex_type="undirected"),
        "max_dim_2": T.edge_participation(adj, max_dim=2),
    }


@case("topology.realdata.node_k_degree")
def _():
    adj, _ = _first(300)
    return {
        "both": T.node_k_degree(adj),
        "in": T.node_k_degree(adj, direction="IN", max_dim=3),
        "out": T.node_k_degree(adj, direction="OUT", max_dim=3),
    }


@case("topology.realdata.in_degree_from_pop")
def _():
    adj, _ = _first(300)
    source = np.arange(0, 300, 4)
    return {
        "all": T.in_degree_from_pop(adj, source),
        "maximal": T.in_degree_from_pop(adj, source, max_simplices=True),
        "max_dim_3": T.in_degree_from_pop(adj, source, max_dim=3),
    }


@case("topology.realdata.in_degree_from_inhibitory_population")
def _():
    adj, nodes = _window(1500, 400)
    source = np.flatnonzero((nodes["synapse_class"] == "INH").to_numpy())
    return T.in_degree_from_pop(adj, source)


# ----------------------------------------------------------------------------- lists


@case("topology.realdata.list_simplices_by_dimension")
def _():
    adj, _ = _first(300)
    return T.list_simplices_by_dimension(adj)


@case("topology.realdata.list_simplices_by_dimension_variants")
def _():
    adj, _ = _first(200)
    return {
        "maximal": T.list_simplices_by_dimension(adj, max_simplices=True),
        "reciprocal": T.list_simplices_by_dimension(adj, simplex_type="reciprocal"),
        "nodes": T.list_simplices_by_dimension(adj, nodes=np.arange(0, 200, 10)),
        "max_dim_2_threads_4": T.list_simplices_by_dimension(adj, max_dim=2, threads=4),
    }


# ----------------------------------------------------------------------------- Betti numbers


@case("topology.realdata.betti_counts")
def _():
    adj, _ = _first(500)
    return {t: T.betti_counts(adj, simplex_type=t) for t in TYPES}


@case("topology.realdata.betti_counts_variants")
def _():
    adj, _ = _first(300)
    return {
        "min_dim_1": T.betti_counts(adj, min_dim=1),
        "max_dim_1": T.betti_counts(adj, max_dim=1),
        "approximation_1": T.betti_counts(adj, approximation=[1, 1]),
        "approximation_100": T.betti_counts(adj, approximation=[100, 100, 100]),
        "nbc": T.nbc(adj),
    }


@case("topology.realdata.betti_counts_dense_window")
def _():
    adj, _ = _window(4500, 300)
    return {
        "directed": T.betti_counts(adj),
        "undirected": T.betti_counts(adj, simplex_type="undirected"),
        "nbc": T.nbc(adj),
    }


# ----------------------------------------------------------------------------- skeleta


@case("topology.realdata.get_k_skeleta_graph")
def _():
    adj, _ = _first(300)
    return {
        pos: T.get_k_skeleta_graph(adj, position=pos, dimensions=[1, 2, 3])
        for pos in ("all", "spine")
    }


@case("topology.realdata.count_rc_edges_skeleta")
def _():
    adj, _ = _first(300)
    return {pos: T.count_rc_edges_skeleta(adj, position=pos) for pos in ("all", "spine")}


@case("topology.realdata.count_rc_edges_skeleta_maximal_window")
def _():
    adj, _ = _window(4500, 300)
    return {
        pos: T.count_rc_edges_skeleta(adj, position=pos, max_simplices=True)
        for pos in ("all", "spine")
    }


@case("topology.realdata.extract_submatrices_from_edge_participation")
def _():
    adj, _ = _first(300)
    edge_par = T.edge_participation(adj)
    return T.extract_submatrices_from_edge_participation(edge_par, 300, dims=[2, 3], thresh=2)


@case("topology.realdata.bedge_counts")
def _():
    adj, _ = _first(200)
    return T.bedge_counts(adj)
