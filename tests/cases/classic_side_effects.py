"""The ``classic`` functions leave their arguments untouched (``generate_degree_based_control`` is the one that shuffles
its input in place and is covered in ``classic_rich_club``).

Each case feeds a matrix that stores explicit zeros and unsorted indices - exactly what ``eliminate_zeros()`` or
``sort_indices()`` would change in place - and returns whether the stored arrays came back identical.
"""

import numpy as np
import pandas as pd
from support.cases import case
from support.data import node_table

from cases.classic_inputs import (
    hub_graph,
    same_state,
    stored_state,
    with_explicit_zeros,
    with_unsorted_indices,
)
from connalysis.network import classic as C


def _awkward(fmt, n=30, seed=1, dtype=bool):
    """Hub graph with stored zeros and descending indices."""
    adj = with_unsorted_indices(
        with_explicit_zeros(hub_graph(n, seed=seed, mean_degree=6.0, dtype=dtype), every=4)
    )
    return adj.asformat(fmt) if adj.format != fmt else adj


def _untouched(calls, formats=("csr", "csc"), **kwargs):
    out = {}
    for fmt in formats:
        for name, call in calls.items():
            adj = _awkward(fmt, **kwargs)
            before = stored_state(adj)
            call(adj)
            out[f"{name}_{fmt}"] = same_state(before, adj)
    return out


@case("classic.inputs_untouched.components_and_degrees")
def _():
    nodes = node_table(30, seed=1)
    return _untouched(
        {
            "closeness": lambda a: C.closeness_connected_components(a),
            "closeness_directed": lambda a: C.closeness_connected_components(a, directed=True),
            "connected_components": lambda a: C.connected_components(
                a, connection="strong", return_labels=True
            ),
            "core_number": lambda a: C.core_number(a),
            "density": lambda a: C.density(a),
            "density_undirected": lambda a: C.density(a, type="undirected"),
            "density_reciprocal": lambda a: C.density(a, type="reciprocal"),
            "get_pairs_within": lambda a: C.get_pairs_within(a, nodes, max_dist=200),
            "cpw": lambda a: C.connection_probability_within(a, nodes, max_dist=200),
            "cpw_reciprocal": lambda a: C.connection_probability_within(
                a, nodes, max_dist=200, type="reciprocal"
            ),
            "cpw_undirected": lambda a: C.connection_probability_within(
                a, nodes, max_dist=200, type="undirected"
            ),
            "largest_scc": lambda a: C.largest_strongly_connected_component(a),
            "ccc": lambda a: C.ccc(a),
        }
    )


@case("classic.inputs_untouched.common_neighbours")
def _():
    return _untouched(
        {
            "distribution": lambda a: C.distribution_number_of_common_neighbors(a),
            "distribution_afferent": lambda a: C.distribution_number_of_common_neighbors(
                a, direction="afferent"
            ),
            "normalized": lambda a: C.normalized_distribution_of_common_neighbors(a),
            "overexpression": lambda a: C.overexpression_of_common_neighbors(
                a, direction="afferent"
            ),
            "weight_bias": lambda a: C.common_neighbor_weight_bias(a.astype(float)),
        }
    )


@case("classic.inputs_untouched.common_neighbor_connectivity_bias")
def _():
    nodes = node_table(30, seed=1)
    out = _untouched(
        {
            "connectivity_bias": lambda a: C.common_neighbor_connectivity_bias(
                a, nodes, cols_location=["x", "y"]
            ),
            "connectivity_bias_afferent": lambda a: C.common_neighbor_connectivity_bias(
                a, direction="afferent"
            ),
        }
    )
    frame = nodes.copy()
    C.common_neighbor_connectivity_bias(
        hub_graph(30, seed=1, mean_degree=6.0), frame, cols_location=["x", "y", "z"], fit_log=0.1
    )
    out["node_table"] = bool(frame.equals(nodes))
    return out


@case("classic.inputs_untouched.gini_and_rich_club")
def _():
    return _untouched(
        {
            "gini_curve": lambda a: C.gini_curve(a, None),
            "gini_curve_afferent": lambda a: C.gini_curve(a, None, direction="afferent"),
            "gini_coefficient": lambda a: C.gini_coefficient(a, None),
            "analytical_gini": lambda a: C._analytical_expected_gini_curve(a),
            "normalized_gini": lambda a: C.normalized_gini_coefficient(
                a, None, direction="afferent"
            ),
            "rich_club_curve": lambda a: C.rich_club_curve(a),
            "rich_club_curve_weighted": lambda a: C.rich_club_curve(
                a.astype(float), direction="OUT"
            ),
            "efficient": lambda a: C.efficient_rich_club_curve(a),
            "efficient_sparse_bins": lambda a: C.efficient_rich_club_curve(
                a, direction="IN", sparse_bin_set=True
            ),
            "analytical_rich_club": lambda a: C._analytical_expected_rich_club_curve(
                a, direction="afferent"
            ),
        }
    )


@case("classic.inputs_untouched.dense_arrays")
def _():
    dense = hub_graph(30, seed=1, mean_degree=6.0).toarray()
    calls = {
        "closeness": lambda a: C.closeness_connected_components(a),
        "connected_components": lambda a: C.connected_components(a, return_labels=True),
        "core_number": lambda a: C.core_number(a),
        "density": lambda a: C.density(a, type="reciprocal"),
        "largest_scc": lambda a: C.largest_strongly_connected_component(a),
        "ccc": lambda a: C.ccc(a),
        "rich_club_curve": lambda a: C.rich_club_curve(a),
        "np_to_nx": lambda a: C.np_to_nx(a),
    }
    out = {}
    for name, call in calls.items():
        arr = dense.copy()
        call(arr)
        out[name] = bool(np.array_equal(arr, dense)) and arr.dtype == dense.dtype
    return out


@case("classic.inputs_untouched.frames_and_series")
def _():
    nodes = node_table(30, seed=1)
    adj = hub_graph(30, seed=1, mean_degree=6.0)
    filtration = pd.Series(np.random.RandomState(2).randint(0, 6, 30))
    pre, post = nodes.iloc[:15].copy(), nodes.iloc[15:].copy()
    checks = {}
    frame = nodes.copy()
    C.get_pairs_within(adj, frame, cols=["x", "z"], max_dist=200)
    checks["get_pairs_within"] = bool(frame.equals(nodes))
    frame = nodes.copy()
    C.connection_probability_within(adj, frame, max_dist=200)
    checks["connection_probability_within"] = bool(frame.equals(nodes))
    pre_before, post_before = pre.copy(), post.copy()
    C.get_pairs_within(adj[:15, 15:].tocsr(), (pre, post), max_dist=200)
    checks["pre_post_tuple"] = bool(pre.equals(pre_before) and post.equals(post_before))
    series = filtration.copy()
    C.efficient_rich_club_curve(adj, pre_calculated_filtration=series)
    C.efficient_rich_club_curve(adj, pre_calculated_filtration=series, sparse_bin_set=True)
    checks["filtration"] = bool(series.equals(filtration))
    return checks
