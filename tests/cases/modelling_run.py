"""``run_model_building`` and ``run_pathway_model_building``: data extraction (counts of connected and all pairs per
distance bin), model fit, optional pickles in ``data_dir``/``model_dir`` (also for data splits to be merged later) and
optional figures in ``plot_dir``.

Both return ``(data_dict, model_dict)``. Fitted parameters come from ``scipy.optimize.curve_fit``, hence
``rtol=1e-6`` where a fit is involved. Figures are checked by their text and plotted data (not pixels) and the file names.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, node_table, rng

from connalysis.modelling import modelling as M

from .modelling_common import (
    figures,
    logged,
    pairwise_distances,
    pickles,
    planted_dd,
    planted_pathway,
    real,
    real_pathway,
    real_spread,
    real_spread_pathway,
    tree,
    use_agg,
    without_path,
)

FIT = {"rtol": 1e-6}
RANGE = {
    "bin_size_um": 50,
    "max_range_um": 450,
}  # 9 bins on the 500-neuron real data (distances up to 483)
SUBSET = {"sample_size": 200, "sample_seed": 7}


def run(order, adj=None, nodes=None, name="mdl", **kwargs):
    if adj is None:
        adj, nodes = real()
    return M.run_model_building(adj, nodes, name, order, **kwargs)


def run_pathway(order, populations=None, name="pw", **kwargs):
    adj, src, tgt = real_pathway() if populations is None else populations
    return M.run_pathway_model_building(adj, src, tgt, name, order, **kwargs)


# ============================================================================= order 2: extraction and fit


@case("modelling.run_model_building.order2_default_arguments", **FIT)
def _():
    return run(2)


@case("modelling.run_model_building.order2_bin_size_and_range", **FIT)
def _():
    return run(2, **RANGE)


@case("modelling.run_model_building.order2_range_not_a_multiple_of_the_bin_size", **FIT)
def _():
    return run(2, bin_size_um=45, max_range_um=300)


@case("modelling.run_model_building.order2_range_shorter_than_the_data", **FIT)
def _():
    return run(2, bin_size_um=25, max_range_um=150)


@case("modelling.run_model_building.order2_fractional_bin_size", **FIT)
def _():
    return run(2, bin_size_um=37.5, max_range_um=412.5)


@case("modelling.run_model_building.order2_coordinates_in_two_dimensions", **FIT)
def _():
    return run(2, coord_names=["x", "z"], bin_size_um=50, max_range_um=400)


@case("modelling.run_model_building.order2_coordinates_in_one_dimension", **FIT)
def _():
    return run(2, coord_names=["z"], bin_size_um=25, max_range_um=275)


@case("modelling.run_model_building.order2_planted_law_is_recovered", **FIT)
def _():
    adj, nodes = planted_dd(120, seed=3, a=0.5, b=0.008)
    return run(2, adj, nodes, bin_size_um=50)


@case("modelling.run_model_building.order2_extra_keywords_are_ignored", **FIT)
def _():
    return run(2, foo="bar", depth_name="missing", meta_seed=3, **RANGE), run(2, **RANGE)


@case("modelling.run_model_building.order2_explicit_none_options_mean_the_defaults", **FIT)
def _():
    return (
        run(
            2,
            max_range_um=None,
            coord_names=None,
            sample_size=None,
            sample_seed=None,
            N_split=None,
            part_idx=None,
            data_dir=None,
            model_dir=None,
            plot_dir=None,
            do_plot=False,
        ),
        run(2),
    )


@case("modelling.run_model_building.order3_explicit_none_options_mean_the_defaults", **FIT)
def _():
    return (
        run(
            3,
            max_range_um=None,
            coord_names=None,
            depth_name=None,
            sample_size=None,
            N_split=None,
            part_idx=None,
        ),
        run(3),
    )


@case("modelling.run_model_building.order2_float_model_order", **FIT)
def _():
    return run(2.0, **RANGE)


@case("modelling.run_model_building.order2_without_model_name_and_without_files", **FIT)
def _():
    return run(2, name=None, **RANGE)


@case("modelling.run_model_building.order2_neurons_spread_over_the_whole_tissue", **FIT)
def _():
    adj, nodes = real_spread()
    return run(
        2, adj, nodes, bin_size_um=100, max_range_um=1500
    )  # 15 bins, the last two without any pair


@case("modelling.run_model_building.order2_neurons_spread_over_the_whole_tissue_full_range", **FIT)
def _():
    adj, nodes = real_spread()
    return run(2, adj, nodes, bin_size_um=100)


@case("modelling.run_model_building.order2_spread_neurons_in_a_data_split", **FIT)
def _():
    adj, nodes = real_spread()
    return run(2, adj, nodes, N_split=4, bin_size_um=100, max_range_um=1500), run(
        2, adj, nodes, bin_size_um=100, max_range_um=1500
    )


@case("modelling.run_model_building.order2_spread_neurons_random_subset", **FIT)
def _():
    adj, nodes = real_spread()
    return run(2, adj, nodes, sample_size=300, sample_seed=5, bin_size_um=100, max_range_um=1500)


@case("modelling.run_model_building.order2_a_single_bin_cannot_be_fitted", **FIT)
def _():
    with logged() as records:
        result = run(2, bin_size_um=1000)
    return result, [(level, text.split("(")[0]) for level, text in records]


@case("modelling.run_model_building.order3_a_single_bin_cannot_be_fitted", **FIT)
def _():
    with logged() as records:
        result = run(3, bin_size_um=1000)
    return result, [(level, text.split("(")[0]) for level, text in records]


@case("modelling.run_model_building.order2_two_bins_are_fitted_exactly", **FIT)
def _():
    return run(2, bin_size_um=250)


@case("modelling.run_model_building.order3_two_bins_are_fitted_exactly", **FIT)
def _():
    return run(3, bin_size_um=250)


def lattice(side=6, spacing=50):
    """Neurons on a square grid (integer coordinates, exact distances) and a connection pattern that depends on them."""
    ix, iy = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    nodes = pd.DataFrame(
        {"x": ix.ravel() * spacing, "y": iy.ravel() * spacing, "z": 0, "depth": iy.ravel()}
    )
    dense = rng(77).random_sample((side * side, side * side)) < 0.35 * np.exp(
        -pairwise_distances(nodes[["x", "y", "z"]].to_numpy()) / 120.0
    )
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense), nodes


@case("modelling.run_model_building.order2_lattice_distances_fall_exactly_on_bin_edges", **FIT)
def _():
    adj, nodes = lattice()
    return run(2, adj, nodes, bin_size_um=50, max_range_um=350), run(
        2, adj, nodes, bin_size_um=50, max_range_um=350, N_split=4
    )


@case("modelling.run_model_building.order3_lattice_with_integer_depth_ties", **FIT)
def _():
    adj, nodes = lattice()
    return run(3, adj, nodes, bin_size_um=50, max_range_um=350), run(
        3, adj, nodes, bin_size_um=50, max_range_um=350, N_split=5
    )


@case(
    "modelling.run_pathway_model_building.order2_lattice_distances_fall_exactly_on_bin_edges", **FIT
)
def _():
    adj, nodes = lattice()
    return run_pathway(2, (adj[:20], nodes.iloc[:20], nodes), bin_size_um=50, max_range_um=350)


@case("modelling.run_model_building.order2_float_division_of_range_by_bin_size_adds_a_bin", **FIT)
def _():
    # 2.1 / 0.7 is 3.0000000000000004, so 4 bins are made instead of 3 and the last one reaches 2.8
    adj, nodes = planted_dd(60, seed=8, a=0.5, b=0.008)
    nodes = nodes.assign(**{c: nodes[c] / 250.0 for c in ("x", "y", "z")})
    return run(2, adj, nodes, bin_size_um=0.7, max_range_um=2.1)


@case("modelling.run_model_building.order2_sample_size_as_float")
def _():
    return run(2, sample_size=200.0, sample_seed=7, **RANGE)


@case("modelling.run_model_building.order2_sample_size_as_numpy_integer", **FIT)
def _():
    return run(2, sample_size=np.int64(200), sample_seed=np.int64(7), **RANGE), run(
        2, **RANGE, **SUBSET
    )


# ---- subsampling (global numpy random state)


@case("modelling.run_model_building.order2_random_subset", **FIT)
def _():
    return run(2, **RANGE, **SUBSET)


@case("modelling.run_model_building.order2_random_subset_other_seed", **FIT)
def _():
    return run(2, sample_size=200, sample_seed=8, **RANGE)


@case("modelling.run_model_building.order2_random_subset_is_reproducible", **FIT)
def _():
    return run(2, **RANGE, **SUBSET), run(2, **RANGE, **SUBSET)


@case("modelling.run_model_building.order2_small_random_subset", **FIT)
def _():
    return run(2, sample_size=60, sample_seed=1, bin_size_um=50)


@case("modelling.run_model_building.order2_sample_size_all_neurons_ignores_the_seed", **FIT)
def _():
    return run(2, sample_size=500, sample_seed=7, **RANGE), run(2, **RANGE)


@case("modelling.run_model_building.order2_sample_size_zero_or_negative_means_all", **FIT)
def _():
    return (
        run(2, sample_size=0, sample_seed=7, **RANGE),
        run(2, sample_size=-3, sample_seed=7, **RANGE),
        run(2, sample_size=None, sample_seed=7, **RANGE),
    )


@case("modelling.run_model_building.order2_sample_seed_without_sample_size_is_unused", **FIT)
def _():
    return run(2, sample_seed=7, **RANGE)


@case("modelling.run_model_building.order2_random_subset_with_data_split", **FIT)
def _():
    return run(2, N_split=3, **RANGE, **SUBSET), run(2, **RANGE, **SUBSET)


@case("modelling.run_model_building.order2_global_stream_after_subsampling")
def _():
    run(2, **RANGE, **SUBSET)
    return np.random.rand(3)


@case("modelling.run_model_building.order2_global_stream_after_subsampling_other_seed")
def _():
    run(2, sample_size=200, sample_seed=8, **RANGE)
    return np.random.rand(3)


@case("modelling.run_model_building.order2_global_stream_untouched_without_subsampling")
def _():
    run(2, **RANGE)
    return np.random.rand(3)


@case("modelling.run_model_building.order2_global_stream_untouched_when_sample_size_covers_all")
def _():
    run(2, sample_size=500, sample_seed=7, **RANGE)
    return np.random.rand(3)


# ---- data splits kept in memory


@case("modelling.run_model_building.order2_data_split_in_one_call", **FIT)
def _():
    return run(2, N_split=3, **RANGE), run(2, **RANGE)


@case("modelling.run_model_building.order2_data_split_with_many_parts", **FIT)
def _():
    return run(2, N_split=11, **RANGE)


@case("modelling.run_model_building.order2_data_split_with_csc_lil_and_dense_adjacency", **FIT)
def _():
    adj, nodes = real()
    inputs = {
        "csc": adj.tocsc(),
        "lil": adj.tolil(),
        "dense": adj.toarray(),
        "csr_array": sp.csr_array(adj),
    }
    return {name: run(2, a, nodes, N_split=3, **RANGE) for name, a in inputs.items()}


@case("modelling.run_model_building.order3_data_split_with_csc_and_lil_adjacency", **FIT)
def _():
    adj, nodes = real()
    return {
        name: run(3, a, nodes, N_split=3, **RANGE)
        for name, a in {"csc": adj.tocsc(), "lil": adj.tolil()}.items()
    }


@case("modelling.run_model_building.order2_data_split_with_empty_trailing_parts", **FIT)
def _():
    adj, nodes = planted_dd(10, seed=5, a=0.6, b=0.004)
    return run(2, adj, nodes, N_split=7, bin_size_um=100, max_range_um=800), run(
        2, adj, nodes, bin_size_um=100, max_range_um=800
    )


@case("modelling.run_model_building.order2_data_split_needs_a_maximum_range", check_message=True)
def _():
    return run(2, N_split=3)


@case("modelling.run_model_building.order2_data_split_into_one_part", check_message=True)
def _():
    return run(2, N_split=1, **RANGE)


@case("modelling.run_model_building.order2_data_split_into_zero_parts", check_message=True)
def _():
    return run(2, N_split=0, **RANGE)


@case("modelling.run_model_building.order2_part_index_without_data_split", check_message=True)
def _():
    return run(2, part_idx=0, **RANGE)


@case("modelling.run_model_building.order2_part_index_beyond_the_parts", check_message=True)
def _():
    return run(2, N_split=3, part_idx=3, **RANGE)


@case("modelling.run_model_building.order2_part_index_below_minus_one", check_message=True)
def _():
    return run(2, N_split=3, part_idx=-2, **RANGE)


@case("modelling.run_model_building.order2_merge_without_data_split", check_message=True)
def _():
    return run(2, part_idx=-1, **RANGE)


@case("modelling.run_model_building.order2_merge_without_data_dir")
def _():
    return run(2, N_split=3, part_idx=-1, **RANGE)


@case("modelling.run_model_building.order2_merge_with_missing_parts", check_message=True)
def _():
    with tempfile.TemporaryDirectory() as tmp:
        run(2, N_split=3, part_idx=0, data_dir=tmp, **RANGE)
        with without_path(tmp):
            return run(2, N_split=3, part_idx=-1, data_dir=tmp, **RANGE)


# ---- invalid input


@case("modelling.run_model_building.model_order_one", check_message=True)
def _():
    return run(1)


@case("modelling.run_model_building.model_order_four", check_message=True)
def _():
    return run(4)


@case("modelling.run_model_building.model_order_as_text", check_message=True)
def _():
    return run("3")


@case("modelling.run_model_building.model_order_none", check_message=True)
def _():
    return run(None)


@case(
    "modelling.run_model_building.unsupported_order_is_found_after_subsampling", check_message=True
)
def _():
    return run(7, **SUBSET)


@case("modelling.run_model_building.neurons_and_adjacency_of_different_size", check_message=True)
def _():
    adj, nodes = real()
    return run(2, adj, nodes.iloc[:400])


@case("modelling.run_model_building.non_square_adjacency", check_message=True)
def _():
    adj, nodes = real()
    return run(2, adj[:, :400], nodes)


@case("modelling.run_model_building.unknown_coordinate_column")
def _():
    return run(2, coord_names=["x", "y", "w"])


@case("modelling.run_model_building.unknown_depth_column")
def _():
    return run(3, depth_name="no_such_column")


@case("modelling.run_model_building.maximum_range_zero")
def _():
    return run(2, max_range_um=0)


# ---- degenerate networks


@case("modelling.run_model_building.order2_two_neurons_fit_fails_with_nan", **FIT)
def _():
    nodes = node_table(2, seed=1)
    adj = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=bool))
    with logged() as records:
        result = run(2, adj, nodes, bin_size_um=50)
    return result, [(level, text.split("(")[0]) for level, text in records]


@case("modelling.run_model_building.order3_two_neurons_fit_fails_with_nan", **FIT)
def _():
    nodes = node_table(2, seed=1)
    adj = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool))
    with logged() as records:
        result = run(3, adj, nodes, bin_size_um=50)
    return result, [(level, text.split("(")[0]) for level, text in records]


@case("modelling.run_model_building.order2_single_neuron_with_a_range", **FIT)
def _():
    return run(2, sp.csr_matrix((1, 1), dtype=bool), node_table(1, seed=1), max_range_um=300)


@case("modelling.run_model_building.order3_single_neuron_with_a_range", **FIT)
def _():
    return run(3, sp.csr_matrix((1, 1), dtype=bool), node_table(1, seed=1), max_range_um=300)


@case("modelling.run_model_building.order2_three_neurons", **FIT)
def _():
    return run(2, digraph(3, 0.6, seed=1), node_table(3, seed=1), bin_size_um=50)


@case("modelling.run_model_building.order2_no_connections_fits_zero", **FIT)
def _():
    return run(2, sp.csr_matrix((30, 30), dtype=bool), node_table(30, seed=2), bin_size_um=100)


@case("modelling.run_model_building.order3_no_connections_fits_zero", **FIT)
def _():
    return run(3, sp.csr_matrix((30, 30), dtype=bool), node_table(30, seed=2), bin_size_um=100)


@case("modelling.run_model_building.order2_all_pairs_connected_fits_a_constant", **FIT)
def _():
    adj = sp.csr_matrix(np.ones((30, 30), dtype=bool) & ~np.eye(30, dtype=bool))
    return run(2, adj, node_table(30, seed=2), bin_size_um=100)


@case("modelling.run_model_building.order3_all_pairs_connected_fits_a_constant", **FIT)
def _():
    adj = sp.csr_matrix(np.ones((30, 30), dtype=bool) & ~np.eye(30, dtype=bool))
    return run(3, adj, node_table(30, seed=2), bin_size_um=100)


@case("modelling.run_model_building.order2_self_connections_are_not_counted", **FIT)
def _():
    adj, nodes = real()
    with_self = adj.tolil()
    with_self.setdiag(True)
    return run(2, with_self.tocsr(), nodes, **RANGE), run(2, adj, nodes, **RANGE)


@case("modelling.run_model_building.order2_neuron_without_coordinates", **FIT)
def _():
    adj, nodes = planted_dd(30, seed=2)
    nodes.loc[3, "x"] = np.nan
    return run(2, adj, nodes, bin_size_um=100)


@case("modelling.run_model_building.order2_two_neurons_at_the_same_place", **FIT)
def _():
    adj, nodes = planted_dd(30, seed=2)
    nodes.loc[5, ["x", "y", "z"]] = nodes.loc[4, ["x", "y", "z"]].to_numpy()
    return run(2, adj, nodes, bin_size_um=100)


# ---- adjacency formats


@case("modelling.run_model_building.order2_adjacency_formats_and_dtypes", **FIT)
def _():
    adj, nodes = real()
    inputs = {
        "csc": adj.tocsc(),
        "lil": adj.tolil(),
        "csr_array": sp.csr_array(adj),
        "dense": adj.toarray(),
        "int8": adj.astype(np.int8),
        "float32": adj.astype(np.float32),
        "float64": adj.astype(float),
    }
    return {name: run(2, a, nodes, **RANGE) for name, a in inputs.items()}


@case("modelling.run_model_building.order3_adjacency_formats_and_dtypes", **FIT)
def _():
    adj, nodes = real()
    inputs = {
        "csc": adj.tocsc(),
        "lil": adj.tolil(),
        "csr_array": sp.csr_array(adj),
        "dense": adj.toarray(),
        "int8": adj.astype(np.int8),
        "float64": adj.astype(float),
    }
    return {name: run(3, a, nodes, **RANGE) for name, a in inputs.items()}


@case("modelling.run_model_building.order2_weighted_adjacency_counts_are_truncated", **FIT)
def _():
    adj, nodes = real()
    return run(2, adj.astype(float) * 2.5, nodes, **RANGE)


@case("modelling.run_model_building.order2_coo_adjacency_without_subsampling")
def _():
    adj, nodes = real()
    return run(2, adj.tocoo(), nodes, **RANGE)


@case("modelling.run_model_building.order2_coo_adjacency_with_subsampling_is_converted", **FIT)
def _():
    adj, nodes = real()
    return run(2, adj.tocoo(), nodes, **RANGE, **SUBSET), run(2, adj, nodes, **RANGE, **SUBSET)


@case("modelling.run_model_building.order2_dense_adjacency_with_subsampling")
def _():
    adj, nodes = real()
    return run(2, adj.toarray(), nodes, **RANGE, **SUBSET)


@case("modelling.run_model_building.order2_adjacency_is_not_modified", **FIT)
def _():
    adj, nodes = real()
    before = (adj.copy(), nodes.copy())
    result = run(2, adj, nodes, N_split=3, **RANGE, **SUBSET)
    return result, (adj != before[0]).nnz, bool(nodes.equals(before[1]))


# ============================================================================= order 3: extraction and fit


@case("modelling.run_model_building.order3_default_arguments", **FIT)
def _():
    return run(3)


@case("modelling.run_model_building.order3_bin_size_and_range", **FIT)
def _():
    return run(3, **RANGE)


@case("modelling.run_model_building.order3_range_not_a_multiple_of_the_bin_size", **FIT)
def _():
    return run(3, bin_size_um=45, max_range_um=300)


@case("modelling.run_model_building.order3_fractional_bin_size", **FIT)
def _():
    return run(3, bin_size_um=37.5, max_range_um=412.5)


@case("modelling.run_model_building.order3_other_depth_column", **FIT)
def _():
    return run(3, depth_name="y", **RANGE)


@case("modelling.run_model_building.order3_coordinates_in_two_dimensions", **FIT)
def _():
    return run(3, coord_names=["x", "z"], bin_size_um=50, max_range_um=400)


@case("modelling.run_model_building.order3_planted_bipolar_law_is_recovered", **FIT)
def _():
    adj, nodes = planted_dd(120, seed=4, a=0.5, b=0.008, bipolar=(0.3, 0.012))
    return run(3, adj, nodes, bin_size_um=50)


@case("modelling.run_model_building.order3_neurons_spread_over_the_whole_tissue", **FIT)
def _():
    adj, nodes = real_spread()
    return run(3, adj, nodes, bin_size_um=100, max_range_um=1500)


@case("modelling.run_model_building.order3_neurons_spread_over_the_whole_tissue_full_range", **FIT)
def _():
    adj, nodes = real_spread()
    return run(3, adj, nodes, bin_size_um=100)


@case("modelling.run_model_building.order3_spread_neurons_in_a_data_split", **FIT)
def _():
    adj, nodes = real_spread()
    return run(3, adj, nodes, N_split=3, bin_size_um=100, max_range_um=1500), run(
        3, adj, nodes, bin_size_um=100, max_range_um=1500
    )


@case("modelling.run_model_building.order3_random_subset", **FIT)
def _():
    return run(3, **RANGE, **SUBSET)


@case("modelling.run_model_building.order3_global_stream_after_subsampling")
def _():
    run(3, **RANGE, **SUBSET)
    return np.random.rand(3)


@case("modelling.run_model_building.order3_data_split_in_one_call", **FIT)
def _():
    return run(3, N_split=3, **RANGE), run(3, **RANGE)


@case("modelling.run_model_building.order3_data_split_with_empty_trailing_parts", **FIT)
def _():
    adj, nodes = planted_dd(10, seed=6, a=0.6, b=0.004, bipolar=(0.4, 0.006))
    return run(3, adj, nodes, N_split=7, bin_size_um=100, max_range_um=800), run(
        3, adj, nodes, bin_size_um=100, max_range_um=800
    )


@case("modelling.run_model_building.order3_data_split_needs_a_maximum_range", check_message=True)
def _():
    return run(3, N_split=2)


@case("modelling.run_model_building.order3_constant_depth_leaves_one_bipolar_side_empty", **FIT)
def _():
    adj, nodes = planted_dd(30, seed=2)
    nodes["depth"] = 5.0
    with logged() as records:
        result = run(3, adj, nodes, bin_size_um=100)
    return result, [(level, text.split("(")[0]) for level, text in records]


@case("modelling.run_model_building.order3_neuron_without_depth", **FIT)
def _():
    adj, nodes = planted_dd(30, seed=2)
    nodes.loc[3, "depth"] = np.nan
    return run(3, adj, nodes, bin_size_um=100)


@case("modelling.run_model_building.order3_integer_depths", **FIT)
def _():
    adj, nodes = planted_dd(60, seed=7, bipolar=(0.3, 0.01))
    nodes["depth"] = nodes["depth"].astype(int)
    return run(3, adj, nodes, bin_size_um=100)


# ============================================================================= files written by the model building


def files_case(order, name, directories, **kwargs):
    """Run with the given {'data_dir': parts, 'model_dir': parts} (relative to a temporary directory)."""
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {key: os.path.join(tmp, *parts) for key, parts in directories.items()}
        result = M.run_model_building(adj, nodes, name, order, **dirs, **kwargs)
        return {"result": result, "files": tree(tmp), "pickles": pickles(tmp)}


@case("modelling.run_model_building.order2_data_and_model_pickles", **FIT)
def _():
    return files_case(2, "mdl", {"data_dir": ["data"], "model_dir": ["model"]}, **RANGE)


@case("modelling.run_model_building.order3_data_and_model_pickles", **FIT)
def _():
    return files_case(3, "mdl", {"data_dir": ["data"], "model_dir": ["model"]}, **RANGE)


@case(
    "modelling.run_model_building.order2_pickles_in_the_same_directory_with_nested_new_folders",
    **FIT,
)
def _():
    return files_case(2, "net", {"data_dir": ["a", "b"], "model_dir": ["a", "b"]}, **RANGE)


@case("modelling.run_model_building.order2_only_model_pickle", **FIT)
def _():
    return files_case(2, "mdl", {"model_dir": ["m"]}, **RANGE)


@case("modelling.run_model_building.order2_only_data_pickle", **FIT)
def _():
    return files_case(2, "mdl", {"data_dir": ["d"]}, **RANGE)


@case("modelling.run_model_building.order2_no_name_is_written_as_none", **FIT)
def _():
    return files_case(2, None, {"data_dir": ["d"], "model_dir": ["m"]}, **RANGE)


@case("modelling.run_model_building.order2_empty_directory_names_write_nothing", **FIT)
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(adj, nodes, "mdl", 2, data_dir="", model_dir=None, **RANGE)
        return result, tree(tmp)


@case("modelling.run_model_building.order2_random_subset_pickles", **FIT)
def _():
    return files_case(2, "sub", {"data_dir": ["d"], "model_dir": ["m"]}, **RANGE, **SUBSET)


def split_flow(order, n_parts, **kwargs):
    """Extract every part into data_dir, then merge them (part_idx=-1) and compare with the direct run."""
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        common = dict(
            data_dir=os.path.join(tmp, "data"),
            model_dir=os.path.join(tmp, "model"),
            N_split=n_parts,
            **kwargs,
        )
        parts = [
            M.run_model_building(adj, nodes, "mdl", order, part_idx=p, **common)
            for p in range(n_parts)
        ]
        after_parts = tree(tmp)
        merged = M.run_model_building(adj, nodes, "mdl", order, part_idx=-1, **common)
        direct = M.run_model_building(adj, nodes, "mdl", order, **{k: v for k, v in kwargs.items()})
        return {
            "parts": parts,
            "after_parts": after_parts,
            "merged": merged,
            "direct": direct,
            "files": tree(tmp),
            "pickles": pickles(tmp),
        }


@case("modelling.run_model_building.order2_parts_extracted_separately_then_merged", **FIT)
def _():
    return split_flow(2, 3, **RANGE)


@case("modelling.run_model_building.order3_parts_extracted_separately_then_merged", **FIT)
def _():
    return split_flow(3, 3, **RANGE)


@case("modelling.run_model_building.order2_parts_of_a_random_subset_then_merged", **FIT)
def _():
    return split_flow(2, 2, **RANGE, **SUBSET)


@case("modelling.run_model_building.order3_parts_of_a_random_subset_then_merged", **FIT)
def _():
    return split_flow(3, 4, **RANGE, **SUBSET)


@case("modelling.run_model_building.order2_part_files_are_numbered_with_padding", **FIT)
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            M.run_model_building(
                adj, nodes, "mdl", 2, N_split=12, part_idx=p, data_dir=tmp, **RANGE
            )
            for p in (0, 3, 11)
        ]
        return results, tree(tmp)


@case("modelling.run_model_building.order2_a_part_run_returns_data_and_an_empty_model", **FIT)
def _():
    adj, nodes = real()
    data, model = M.run_model_building(adj, nodes, "mdl", 2, N_split=4, part_idx=2, **RANGE)
    return data, model


@case("modelling.run_model_building.order2_part_run_with_plotting_requested_still_plots_nothing")
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "mdl", 2, N_split=2, part_idx=0, do_plot=True, plot_dir=tmp, **RANGE
        )
        return result[1], tree(tmp), figures()


@case("modelling.run_model_building.order2_merge_uses_the_parts_not_the_matrix", **FIT)
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        for p in range(2):
            M.run_model_building(adj, nodes, "mdl", 2, N_split=2, part_idx=p, data_dir=tmp, **RANGE)
        empty = sp.csr_matrix(adj.shape, dtype=bool)  # connections are not looked at when merging
        return M.run_model_building(
            empty, nodes, "mdl", 2, N_split=2, part_idx=-1, data_dir=tmp, **RANGE
        )


@case("modelling.run_model_building.order2_merge_of_parts_with_other_bins", check_message=True)
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        M.run_model_building(adj, nodes, "mdl", 2, N_split=2, part_idx=0, data_dir=tmp, **RANGE)
        M.run_model_building(
            adj,
            nodes,
            "mdl",
            2,
            N_split=2,
            part_idx=1,
            data_dir=tmp,
            bin_size_um=50,
            max_range_um=400,
        )
        return M.run_model_building(
            adj, nodes, "mdl", 2, N_split=2, part_idx=-1, data_dir=tmp, **RANGE
        )


# ============================================================================= figures


@case("modelling.run_model_building.order2_plots_are_saved", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "mdl", 2, do_plot=True, plot_dir=os.path.join(tmp, "figs"), **RANGE
        )
        return result, tree(tmp), figures()


@case("modelling.run_model_building.order3_plots_are_saved", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "mdl", 3, do_plot=True, plot_dir=os.path.join(tmp, "figs"), **RANGE
        )
        return result, tree(tmp), figures()


@case("modelling.run_model_building.order2_plot_without_a_directory_only_draws", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    result = M.run_model_building(
        adj, nodes, "mdl", 2, do_plot=True, bin_size_um=100, max_range_um=400
    )
    return result, figures()


@case("modelling.run_model_building.order3_plot_without_a_directory_only_draws", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    result = M.run_model_building(
        adj, nodes, "mdl", 3, do_plot=True, bin_size_um=100, max_range_um=400
    )
    return result, figures()


@case("modelling.run_model_building.order2_plot_directory_without_plotting_stays_empty", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "mdl", 2, do_plot=False, plot_dir=os.path.join(tmp, "figs"), **RANGE
        )
        return result, tree(tmp), figures()


@case("modelling.run_model_building.order2_plots_and_pickles_together", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj,
            nodes,
            "both",
            2,
            do_plot=True,
            plot_dir=os.path.join(tmp, "p"),
            data_dir=os.path.join(tmp, "d"),
            model_dir=os.path.join(tmp, "m"),
            **RANGE,
        )
        return result, tree(tmp), len(figures())


@case("modelling.run_model_building.order2_plots_of_a_random_subset_after_a_data_split", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "split", 2, N_split=2, do_plot=True, plot_dir=tmp, **RANGE, **SUBSET
        )
        return result, tree(tmp), figures()


@case("modelling.run_model_building.order2_plot_in_an_existing_directory", **FIT)
def _():
    use_agg()
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "mdl", 2, do_plot=True, plot_dir=tmp, bin_size_um=100, max_range_um=400
        )
        again = M.run_model_building(
            adj, nodes, "mdl", 2, do_plot=True, plot_dir=tmp, bin_size_um=100, max_range_um=400
        )
        return result, again[1], tree(tmp), len(figures())


@case("modelling.run_model_building.order2_plots_with_empty_bins_at_the_far_end", **FIT)
def _():
    use_agg()
    adj, nodes = real_spread()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "far", 2, do_plot=True, plot_dir=tmp, bin_size_um=100, max_range_um=1500
        )
        return result, tree(tmp), figures()


@case("modelling.run_model_building.order3_plots_with_empty_bins_at_the_far_end", **FIT)
def _():
    use_agg()
    adj, nodes = real_spread()
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_model_building(
            adj, nodes, "far", 3, do_plot=True, plot_dir=tmp, bin_size_um=100, max_range_um=1500
        )
        return result, tree(tmp), figures()


@case("modelling.run_model_building.order2_plot_directory_needs_a_model_name")
def _():
    use_agg()
    adj, nodes = real()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return M.run_model_building(adj, nodes, None, 2, do_plot=True, plot_dir=tmp, **RANGE)
    finally:
        figures()


@case("modelling.run_model_building.order2_plot_of_a_failed_fit", **FIT)
def _():
    use_agg()
    nodes = node_table(2, seed=1)
    adj = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=bool))
    result = M.run_model_building(adj, nodes, "nan", 2, do_plot=True, bin_size_um=50)
    return result[1], len(figures())


# ============================================================================= pathways


@case("modelling.run_pathway_model_building.order2_default_arguments", **FIT)
def _():
    return run_pathway(2)


@case("modelling.run_pathway_model_building.order2_bin_size_and_range", **FIT)
def _():
    return run_pathway(2, **RANGE)


@case("modelling.run_pathway_model_building.order2_populations_of_different_size", **FIT)
def _():
    return run_pathway(2, real_pathway(300, 200), **RANGE)


@case("modelling.run_pathway_model_building.order2_range_not_a_multiple_of_the_bin_size", **FIT)
def _():
    return run_pathway(2, bin_size_um=45, max_range_um=300)


@case("modelling.run_pathway_model_building.order2_coordinates_in_two_dimensions", **FIT)
def _():
    return run_pathway(2, coord_names=["x", "z"], bin_size_um=50, max_range_um=400)


@case("modelling.run_pathway_model_building.order2_planted_law_is_recovered", **FIT)
def _():
    return run_pathway(2, planted_pathway(90, 70, seed=5, a=0.5, b=0.008), bin_size_um=50)


@case(
    "modelling.run_pathway_model_building.order2_overlapping_populations_drop_the_same_neuron",
    **FIT,
)
def _():
    adj, nodes = real()
    return run_pathway(2, (adj[:300, 200:].tocsr(), nodes.iloc[:300], nodes.iloc[200:]), **RANGE)


@case("modelling.run_pathway_model_building.order2_same_population_as_the_symmetric_version", **FIT)
def _():
    adj, nodes = real()
    return run_pathway(2, (adj, nodes, nodes), **RANGE), run(2, adj, nodes, **RANGE)


@case("modelling.run_pathway_model_building.order2_subsets_of_both_populations", **FIT)
def _():
    return run_pathway(2, sample_size=100, sample_seed=3, **RANGE)


@case("modelling.run_pathway_model_building.order2_subset_of_the_larger_population_only", **FIT)
def _():
    return run_pathway(2, real_pathway(300, 200), sample_size=250, sample_seed=3, **RANGE)


@case(
    "modelling.run_pathway_model_building.order2_subset_of_the_larger_target_population_only", **FIT
)
def _():
    return run_pathway(2, real_pathway(200, 300), sample_size=250, sample_seed=3, **RANGE)


@case(
    "modelling.run_pathway_model_building.order2_global_stream_after_subset_of_the_target_population"
)
def _():
    run_pathway(2, real_pathway(200, 300), sample_size=250, sample_seed=3, **RANGE)
    return np.random.rand(3)


@case("modelling.run_pathway_model_building.order2_more_targets_than_sources", **FIT)
def _():
    return run_pathway(2, real_pathway(200, 300), **RANGE)


@case("modelling.run_pathway_model_building.order3_more_targets_than_sources", **FIT)
def _():
    return run_pathway(3, real_pathway(200, 300), **RANGE)


@case(
    "modelling.run_pathway_model_building.order3_subset_of_the_larger_target_population_only", **FIT
)
def _():
    return run_pathway(3, real_pathway(200, 300), sample_size=250, sample_seed=3, **RANGE)


@case(
    "modelling.run_pathway_model_building.order2_sample_size_just_below_the_larger_population",
    **FIT,
)
def _():
    return run_pathway(2, real_pathway(300, 200), sample_size=299, sample_seed=3, **RANGE)


@case("modelling.run_pathway_model_building.order2_sample_size_covering_both_populations", **FIT)
def _():
    return run_pathway(
        2, real_pathway(300, 200), sample_size=300, sample_seed=3, **RANGE
    ), run_pathway(2, real_pathway(300, 200), **RANGE)


@case("modelling.run_pathway_model_building.order2_sample_size_zero_or_negative", **FIT)
def _():
    return run_pathway(2, sample_size=0, sample_seed=3, **RANGE), run_pathway(
        2, sample_size=-1, sample_seed=3, **RANGE
    )


@case("modelling.run_pathway_model_building.order2_global_stream_after_subsets_of_both_populations")
def _():
    run_pathway(2, sample_size=100, sample_seed=3, **RANGE)
    return np.random.rand(3)


@case("modelling.run_pathway_model_building.order2_global_stream_after_subset_of_one_population")
def _():
    run_pathway(2, real_pathway(300, 200), sample_size=250, sample_seed=3, **RANGE)
    return np.random.rand(3)


@case("modelling.run_pathway_model_building.order2_global_stream_when_nothing_is_subsampled")
def _():
    run_pathway(2, real_pathway(300, 200), sample_size=300, sample_seed=3, **RANGE)
    return np.random.rand(3)


@case("modelling.run_pathway_model_building.order3_default_arguments", **FIT)
def _():
    return run_pathway(3)


@case("modelling.run_pathway_model_building.order3_bin_size_and_range", **FIT)
def _():
    return run_pathway(3, **RANGE)


@case("modelling.run_pathway_model_building.order3_populations_of_different_size", **FIT)
def _():
    return run_pathway(3, real_pathway(300, 200), **RANGE)


@case("modelling.run_pathway_model_building.order3_other_depth_column", **FIT)
def _():
    return run_pathway(3, depth_name="y", **RANGE)


@case("modelling.run_pathway_model_building.order3_planted_bipolar_law_is_recovered", **FIT)
def _():
    return run_pathway(
        3, planted_pathway(100, 80, seed=6, a=0.5, b=0.008, bipolar=(0.3, 0.012)), bin_size_um=50
    )


@case("modelling.run_pathway_model_building.order3_subsets_of_both_populations", **FIT)
def _():
    return run_pathway(3, sample_size=100, sample_seed=3, **RANGE)


@case("modelling.run_pathway_model_building.order3_subset_of_the_larger_population_only", **FIT)
def _():
    return run_pathway(3, real_pathway(300, 200), sample_size=250, sample_seed=3, **RANGE)


@case("modelling.run_pathway_model_building.order2_neurons_spread_over_the_whole_tissue", **FIT)
def _():
    return run_pathway(2, real_spread_pathway(), bin_size_um=100, max_range_um=1500)


@case("modelling.run_pathway_model_building.order3_neurons_spread_over_the_whole_tissue", **FIT)
def _():
    return run_pathway(3, real_spread_pathway(), bin_size_um=100, max_range_um=1500)


@case("modelling.run_pathway_model_building.order3_spread_neurons_random_subset", **FIT)
def _():
    return run_pathway(3, real_spread_pathway(), sample_size=200, sample_seed=2, bin_size_um=100)


@case("modelling.run_pathway_model_building.data_and_model_pickles", **FIT)
def _():
    adj, src, tgt = real_pathway(300, 200)
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_pathway_model_building(
            adj,
            src,
            tgt,
            "pw",
            3,
            data_dir=os.path.join(tmp, "d"),
            model_dir=os.path.join(tmp, "m"),
            **RANGE,
        )
        return {"result": result, "files": tree(tmp), "pickles": pickles(tmp)}


@case("modelling.run_pathway_model_building.order2_plots_are_saved", **FIT)
def _():
    use_agg()
    adj, src, tgt = real_pathway(300, 200)
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_pathway_model_building(
            adj, src, tgt, "pw", 2, do_plot=True, plot_dir=os.path.join(tmp, "figs"), **RANGE
        )
        return result, tree(tmp), figures()


@case("modelling.run_pathway_model_building.order3_plots_are_saved", **FIT)
def _():
    use_agg()
    adj, src, tgt = real_pathway(300, 200)
    with tempfile.TemporaryDirectory() as tmp:
        result = M.run_pathway_model_building(
            adj, src, tgt, "pw", 3, do_plot=True, plot_dir=os.path.join(tmp, "figs"), **RANGE
        )
        return result, tree(tmp), figures()


@case("modelling.run_pathway_model_building.order2_plot_without_a_directory_only_draws", **FIT)
def _():
    use_agg()
    adj, src, tgt = real_pathway(300, 200)
    result = M.run_pathway_model_building(
        adj, src, tgt, "pw", 2, do_plot=True, bin_size_um=100, max_range_um=400
    )
    return result, len(figures())


@case("modelling.run_pathway_model_building.data_split_is_not_supported", check_message=True)
def _():
    return run_pathway(2, N_split=2, **RANGE)


@case("modelling.run_pathway_model_building.part_index_is_not_supported", check_message=True)
def _():
    return run_pathway(2, part_idx=0, **RANGE)


@case("modelling.run_pathway_model_building.merge_is_not_supported", check_message=True)
def _():
    return run_pathway(3, part_idx=-1, **RANGE)


@case("modelling.run_pathway_model_building.empty_split_keywords_are_accepted", **FIT)
def _():
    return run_pathway(2, N_split=None, part_idx=None, **RANGE)


@case("modelling.run_pathway_model_building.model_order_not_supported", check_message=True)
def _():
    return run_pathway(5)


@case(
    "modelling.run_pathway_model_building.unsupported_order_is_found_after_subsampling",
    check_message=True,
)
def _():
    return run_pathway(6, sample_size=100, sample_seed=3)


@case("modelling.run_pathway_model_building.sources_and_rows_of_different_size", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return run_pathway(2, (adj, src.iloc[:100], tgt))


@case(
    "modelling.run_pathway_model_building.targets_and_columns_of_different_size", check_message=True
)
def _():
    adj, src, tgt = real_pathway()
    return run_pathway(2, (adj, src, tgt.iloc[:100]))


@case("modelling.run_pathway_model_building.unknown_coordinate_column")
def _():
    adj, src, tgt = real_pathway()
    return run_pathway(2, (adj, src, tgt), coord_names=["x", "y", "w"])


@case("modelling.run_pathway_model_building.order2_single_source_neuron", **FIT)
def _():
    adj, src, tgt = real_pathway(250, 250)
    return run_pathway(2, (adj[:1], src.iloc[:1], tgt), **RANGE)


@case("modelling.run_pathway_model_building.order2_adjacency_formats_and_dtypes", **FIT)
def _():
    adj, src, tgt = real_pathway()
    inputs = {
        "csc": adj.tocsc(),
        "lil": adj.tolil(),
        "dense": adj.toarray(),
        "csr_array": sp.csr_array(adj),
        "int8": adj.astype(np.int8),
        "float64": adj.astype(float),
    }
    return {name: run_pathway(2, (a, src, tgt), **RANGE) for name, a in inputs.items()}


@case("modelling.run_pathway_model_building.order2_coo_adjacency_without_subsampling")
def _():
    adj, src, tgt = real_pathway()
    return run_pathway(2, (adj.tocoo(), src, tgt), **RANGE)
