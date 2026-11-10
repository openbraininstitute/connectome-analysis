"""The six pipeline wrappers of ``connalysis.modelling``: ``conn_prob_{2nd_order,3rd_order,}_model`` and the pathway
versions. They return a DataFrame of fitted model parameters with one row per random subset of neurons (index
``seed``), and they refuse the arguments they set internally.

Fitted parameters come from ``scipy.optimize.curve_fit``, hence ``rtol=1e-6`` in every case that fits.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case

from connalysis.modelling import modelling as M

from .modelling_common import (
    logged,
    planted_dd,
    planted_pathway,
    real,
    real_pathway,
    real_spread,
    real_spread_pathway,
)

FIT = {"rtol": 1e-6}
RANGE = {
    "bin_size_um": 50,
    "max_range_um": 450,
}  # 9 bins on the 500-neuron real data (distances up to 483)
SUBSET = {"sample_size": 200, "bin_size_um": 50, "max_range_um": 450}


# ============================================================================= conn_prob_2nd_order_model


@case("modelling.conn_prob_2nd_order_model.default_arguments", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes)


@case("modelling.conn_prob_2nd_order_model.bin_size_and_range", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, **RANGE)


@case("modelling.conn_prob_2nd_order_model.range_not_a_multiple_of_the_bin_size", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(
        adj, nodes, bin_size_um=45, max_range_um=300
    )  # 7 bins, the last one ends at 315


@case("modelling.conn_prob_2nd_order_model.range_shorter_than_the_data", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, bin_size_um=25, max_range_um=150)


@case("modelling.conn_prob_2nd_order_model.fractional_bin_size", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, bin_size_um=37.5, max_range_um=412.5)


@case("modelling.conn_prob_2nd_order_model.coordinates_in_two_dimensions", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(
        adj, nodes, coord_names=["x", "z"], bin_size_um=50, max_range_um=400
    )


@case("modelling.conn_prob_2nd_order_model.coordinates_in_one_dimension", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(
        adj, nodes, coord_names=["z"], bin_size_um=25, max_range_um=275
    )


@case("modelling.conn_prob_2nd_order_model.unknown_coordinate_column")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, coord_names=["x", "y", "w"])


@case("modelling.conn_prob_2nd_order_model.planted_law_is_recovered", **FIT)
def _():
    adj, nodes = planted_dd(120, seed=3, a=0.5, b=0.008)
    return M.conn_prob_2nd_order_model(adj, nodes, bin_size_um=50)


@case("modelling.conn_prob_2nd_order_model.neurons_spread_over_the_whole_tissue", **FIT)
def _():
    adj, nodes = real_spread()
    return M.conn_prob_2nd_order_model(adj, nodes, bin_size_um=100)


@case(
    "modelling.conn_prob_2nd_order_model.neurons_spread_over_the_whole_tissue_random_subsets", **FIT
)
def _():
    adj, nodes = real_spread()
    return M.conn_prob_2nd_order_model(
        adj, nodes, bin_size_um=100, max_range_um=1200, sample_size=300, sample_seeds=3, meta_seed=2
    )


# ---- random subsets


@case("modelling.conn_prob_2nd_order_model.subsets_from_generated_seeds", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=3, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.subsets_from_generated_seeds_with_meta_seed", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=3, meta_seed=11, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.subsets_from_a_seed_list", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[30, 5, 17], **SUBSET)


@case(
    "modelling.conn_prob_2nd_order_model.seed_list_with_duplicates_warns_and_keeps_unique_sorted",
    **FIT,
)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[11, 5, 11, 7, 5], **SUBSET)
    return frame, records


@case("modelling.conn_prob_2nd_order_model.single_seed_in_a_list", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[123456], **SUBSET)


@case("modelling.conn_prob_2nd_order_model.sample_size_alone_draws_one_generated_seed", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.meta_seed_is_not_used_with_a_seed_list", **FIT)
def _():
    adj, nodes = real()
    plain = M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[8, 9], **SUBSET)
    with_meta = M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[8, 9], meta_seed=99, **SUBSET)
    return plain, with_meta


@case(
    "modelling.conn_prob_2nd_order_model.sample_size_equal_to_the_number_of_neurons_ignores_seeds",
    **FIT,
)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(adj, nodes, sample_size=500, sample_seeds=3, **RANGE)
    return frame, records


@case(
    "modelling.conn_prob_2nd_order_model.sample_size_larger_than_the_number_of_neurons_ignores_seeds",
    **FIT,
)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(
            adj, nodes, sample_size=900, sample_seeds=[1, 2], **RANGE
        )
    return frame, records


@case("modelling.conn_prob_2nd_order_model.sample_size_zero_ignores_seeds", **FIT)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(adj, nodes, sample_size=0, sample_seeds=2, **RANGE)
    return frame, records


@case("modelling.conn_prob_2nd_order_model.negative_sample_size_ignores_seeds", **FIT)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(adj, nodes, sample_size=-5, sample_seeds=2, **RANGE)
    return frame, records


@case("modelling.conn_prob_2nd_order_model.sample_seeds_without_sample_size_are_ignored", **FIT)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=4, **RANGE)
    return frame, records


@case("modelling.conn_prob_2nd_order_model.no_subsampling_gives_one_row_without_seed", **FIT)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_2nd_order_model(adj, nodes, meta_seed=5, **RANGE)
    return frame, frame.index.tolist(), records


@case("modelling.conn_prob_2nd_order_model.seed_count_zero")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=0, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.seed_count_negative")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=-2, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.seed_count_float")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=2.0, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.seed_count_numpy_integer")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=np.int64(2), **SUBSET)


@case("modelling.conn_prob_2nd_order_model.seeds_as_tuple")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=(1, 2), **SUBSET)


@case("modelling.conn_prob_2nd_order_model.seeds_as_array")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=np.array([1, 2]), **SUBSET)


@case("modelling.conn_prob_2nd_order_model.empty_seed_list_gives_an_empty_frame")
def _():
    adj, nodes = real()
    frame = M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[], **SUBSET)
    return frame, frame.shape, frame.index, frame.columns


@case("modelling.conn_prob_2nd_order_model.global_random_stream_after_subsets", **FIT)
def _():
    adj, nodes = real()
    frame = M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=2, **SUBSET)
    return frame, np.random.rand(3)


@case("modelling.conn_prob_2nd_order_model.global_random_stream_untouched_without_subsets")
def _():
    adj, nodes = real()
    M.conn_prob_2nd_order_model(adj, nodes, **RANGE)
    return np.random.rand(3)


# ---- data splits


@case("modelling.conn_prob_2nd_order_model.data_split_gives_the_same_model", **FIT)
def _():
    adj, nodes = real()
    return (
        M.conn_prob_2nd_order_model(adj, nodes, N_split=3, **RANGE),
        M.conn_prob_2nd_order_model(adj, nodes, **RANGE),
    )


@case("modelling.conn_prob_2nd_order_model.data_split_of_random_subsets", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, N_split=2, sample_seeds=2, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.data_split_needs_a_maximum_range")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, N_split=3)


@case("modelling.conn_prob_2nd_order_model.data_split_into_one_part")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, N_split=1, **RANGE)


# ---- unusual arguments


@case("modelling.conn_prob_2nd_order_model.unknown_keyword_is_silently_ignored", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(
        adj, nodes, bin_size=999, colour="red", **RANGE
    ), M.conn_prob_2nd_order_model(adj, nodes, **RANGE)


@case("modelling.conn_prob_2nd_order_model.depth_name_is_not_used", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, depth_name="no_such_column", **RANGE)


@case("modelling.conn_prob_2nd_order_model.model_order_keyword_is_rejected", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, model_order=2)


@case(
    "modelling.conn_prob_2nd_order_model.neurons_and_adjacency_of_different_size",
    check_message=True,
)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes.iloc[:499])


@case("modelling.conn_prob_2nd_order_model.non_square_adjacency", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj[:, :499], nodes)


# ---- adjacency formats and dtypes


@case("modelling.conn_prob_2nd_order_model.adjacency_formats_and_dtypes", **FIT)
def _():
    adj, nodes = real()
    inputs = {
        "csc": adj.tocsc(),
        "lil": adj.tolil(),
        "csr_array": sp.csr_array(adj),
        "dense": adj.toarray(),
        "int": adj.astype(np.int32),
        "float": adj.astype(float),
        "csc_float32": adj.tocsc().astype(np.float32),
    }
    return {name: M.conn_prob_2nd_order_model(a, nodes, **RANGE) for name, a in inputs.items()}


@case("modelling.conn_prob_2nd_order_model.weighted_adjacency_counts_are_truncated", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj.astype(float) * 2.5, nodes, **RANGE)


@case("modelling.conn_prob_2nd_order_model.subsets_of_csc_lil_and_csr_array_inputs", **FIT)
def _():
    adj, nodes = real()
    return {
        name: M.conn_prob_2nd_order_model(a, nodes, sample_seeds=[3], **SUBSET)
        for name, a in {
            "csc": adj.tocsc(),
            "lil": adj.tolil(),
            "csr_array": sp.csr_array(adj),
            "coo": adj.tocoo(),
        }.items()
    }


@case("modelling.conn_prob_2nd_order_model.dense_adjacency_cannot_be_subsampled")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj.toarray(), nodes, sample_seeds=2, **SUBSET)


@case("modelling.conn_prob_2nd_order_model.coo_adjacency_cannot_be_indexed_without_subsampling")
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj.tocoo(), nodes, **RANGE)


@case("modelling.conn_prob_2nd_order_model.node_table_with_other_index", **FIT)
def _():
    adj, nodes = real()
    nodes.index = nodes.index + 1000
    return M.conn_prob_2nd_order_model(adj, nodes, sample_seeds=[4], **SUBSET)


# ============================================================================= conn_prob_3rd_order_model


@case("modelling.conn_prob_3rd_order_model.default_arguments", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes)


@case("modelling.conn_prob_3rd_order_model.bin_size_and_range", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, **RANGE)


@case("modelling.conn_prob_3rd_order_model.range_shorter_than_the_data", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, bin_size_um=25, max_range_um=200)


@case("modelling.conn_prob_3rd_order_model.fractional_bin_size", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, bin_size_um=37.5, max_range_um=412.5)


@case("modelling.conn_prob_3rd_order_model.other_depth_column", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, depth_name="y", **RANGE)


@case("modelling.conn_prob_3rd_order_model.coordinates_in_two_dimensions", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(
        adj, nodes, coord_names=["x", "z"], bin_size_um=50, max_range_um=400
    )


@case("modelling.conn_prob_3rd_order_model.unknown_coordinate_column")
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, coord_names=["x", "y", "w"])


@case("modelling.conn_prob_3rd_order_model.unknown_depth_column")
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, depth_name="no_such_column")


@case("modelling.conn_prob_3rd_order_model.planted_bipolar_law_is_recovered", **FIT)
def _():
    adj, nodes = planted_dd(120, seed=4, a=0.5, b=0.008, bipolar=(0.3, 0.012))
    return M.conn_prob_3rd_order_model(adj, nodes, bin_size_um=50)


@case("modelling.conn_prob_3rd_order_model.neurons_spread_over_the_whole_tissue", **FIT)
def _():
    adj, nodes = real_spread()
    return M.conn_prob_3rd_order_model(adj, nodes, bin_size_um=100)


@case(
    "modelling.conn_prob_3rd_order_model.neurons_spread_over_the_whole_tissue_random_subsets", **FIT
)
def _():
    adj, nodes = real_spread()
    return M.conn_prob_3rd_order_model(
        adj, nodes, bin_size_um=100, max_range_um=1200, sample_size=300, sample_seeds=[4, 9]
    )


@case("modelling.conn_prob_3rd_order_model.subsets_from_generated_seeds", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, sample_seeds=3, **SUBSET)


@case("modelling.conn_prob_3rd_order_model.subsets_from_a_seed_list_with_duplicates", **FIT)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_3rd_order_model(adj, nodes, sample_seeds=[21, 9, 21], **SUBSET)
    return frame, records


@case("modelling.conn_prob_3rd_order_model.sample_size_covering_all_neurons_ignores_seeds", **FIT)
def _():
    adj, nodes = real()
    with logged() as records:
        frame = M.conn_prob_3rd_order_model(adj, nodes, sample_size=500, sample_seeds=2, **RANGE)
    return frame, records


@case("modelling.conn_prob_3rd_order_model.data_split_gives_the_same_model", **FIT)
def _():
    adj, nodes = real()
    return (
        M.conn_prob_3rd_order_model(adj, nodes, N_split=4, **RANGE),
        M.conn_prob_3rd_order_model(adj, nodes, **RANGE),
    )


@case("modelling.conn_prob_3rd_order_model.data_split_needs_a_maximum_range")
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, N_split=2)


@case("modelling.conn_prob_3rd_order_model.global_random_stream_after_subsets", **FIT)
def _():
    adj, nodes = real()
    frame = M.conn_prob_3rd_order_model(adj, nodes, sample_seeds=2, **SUBSET)
    return frame, np.random.rand(3)


@case("modelling.conn_prob_3rd_order_model.adjacency_formats", **FIT)
def _():
    adj, nodes = real()
    return {
        name: M.conn_prob_3rd_order_model(a, nodes, **RANGE)
        for name, a in {
            "csc": adj.tocsc(),
            "lil": adj.tolil(),
            "dense": adj.toarray(),
            "float": adj.astype(float),
        }.items()
    }


@case("modelling.conn_prob_3rd_order_model.model_order_keyword_is_rejected", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, model_order=3)


@case(
    "modelling.conn_prob_3rd_order_model.neurons_and_adjacency_of_different_size",
    check_message=True,
)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes.iloc[:10])


# ============================================================================= conn_prob_model (generic)


@case("modelling.conn_prob_model.second_order", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=2, **RANGE)


@case("modelling.conn_prob_model.third_order", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=3, **RANGE)


@case("modelling.conn_prob_model.second_order_subsets", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=2, sample_seeds=[2, 8], **SUBSET)


@case("modelling.conn_prob_model.third_order_subsets_and_split", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=3, sample_seeds=[2, 8], N_split=2, **SUBSET)


@case("modelling.conn_prob_model.same_as_the_order_specific_wrappers", **FIT)
def _():
    adj, nodes = real()
    return [
        M.conn_prob_model(adj, nodes, model_order=2, **RANGE),
        M.conn_prob_2nd_order_model(adj, nodes, **RANGE),
        M.conn_prob_model(adj, nodes, model_order=3, **RANGE),
        M.conn_prob_3rd_order_model(adj, nodes, **RANGE),
    ]


@case("modelling.conn_prob_model.model_order_missing")
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, **RANGE)


@case("modelling.conn_prob_model.model_order_not_supported", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=4, **RANGE)


@case("modelling.conn_prob_model.model_order_one_is_not_supported", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=1)


@case("modelling.conn_prob_model.model_order_as_text", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order="2")


@case("modelling.conn_prob_model.model_order_as_float", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=2.0, **RANGE)


@case("modelling.conn_prob_model.unsupported_order_with_subsets_still_raises", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_order=5, sample_seeds=1, **SUBSET)


@case("modelling.conn_prob_model.size_is_checked_before_the_arguments", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes.iloc[:10], model_name="x")


@case(
    "modelling.conn_prob_model.invalid_argument_is_checked_before_the_model_order",
    check_message=True,
)
def _():
    adj, nodes = real()
    return M.conn_prob_model(adj, nodes, model_name="x")


@case(
    "modelling.conn_prob_model.several_refused_arguments_name_the_first_of_the_internal_list",
    check_message=True,
)
def _():
    adj, nodes = real()
    return M.conn_prob_model(
        adj, nodes, model_order=2, part_idx=0, do_plot=True, data_dir="out", model_dir="out"
    )


@case(
    "modelling.conn_prob_pathway_model.several_refused_arguments_name_the_first_of_the_internal_list",
    check_message=True,
)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(
        adj, src, tgt, model_order=2, do_plot=True, plot_dir="out", sample_seed=1
    )


@case("modelling.conn_prob_2nd_order_model.explicit_none_options_mean_the_defaults", **FIT)
def _():
    adj, nodes = real()
    return (
        M.conn_prob_2nd_order_model(
            adj, nodes, max_range_um=None, coord_names=None, sample_size=None, N_split=None
        ),
        M.conn_prob_2nd_order_model(adj, nodes),
    )


@case("modelling.conn_prob_3rd_order_model.explicit_none_options_mean_the_defaults", **FIT)
def _():
    adj, nodes = real()
    return (
        M.conn_prob_3rd_order_model(
            adj,
            nodes,
            max_range_um=None,
            coord_names=None,
            depth_name=None,
            sample_size=None,
            N_split=None,
        ),
        M.conn_prob_3rd_order_model(adj, nodes),
    )


def _internal_argument_cases():
    values = {
        "model_name": "name",
        "sample_seed": 3,
        "model_dir": "out",
        "data_dir": "out",
        "plot_dir": "out",
        "do_plot": True,
        "part_idx": 0,
    }
    for arg, value in values.items():

        def register(arg=arg, value=value):
            @case(f"modelling.conn_prob_model.refuses_{arg}", check_message=True)
            def _():
                adj, nodes = real()
                return M.conn_prob_model(adj, nodes, model_order=2, **{arg: value}, **RANGE)

            @case(f"modelling.conn_prob_pathway_model.refuses_{arg}", check_message=True)
            def _():
                adj, src, tgt = real_pathway()
                return M.conn_prob_pathway_model(
                    adj, src, tgt, model_order=2, **{arg: value}, **RANGE
                )

        register()


_internal_argument_cases()


# the order-specific wrappers pass their keywords on, so one internal argument each shows the refusal arrives
@case("modelling.conn_prob_2nd_order_model.refuses_do_plot", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_2nd_order_model(adj, nodes, do_plot=True, **RANGE)


@case("modelling.conn_prob_3rd_order_model.refuses_model_dir", check_message=True)
def _():
    adj, nodes = real()
    return M.conn_prob_3rd_order_model(adj, nodes, model_dir="out", **RANGE)


@case("modelling.conn_prob_2nd_order_pathway_model.refuses_plot_dir", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, plot_dir="out", **RANGE)


@case("modelling.conn_prob_3rd_order_pathway_model.refuses_data_dir", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, data_dir="out", **RANGE)


# ============================================================================= pathway wrappers


@case("modelling.conn_prob_2nd_order_pathway_model.default_arguments", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt)


@case("modelling.conn_prob_2nd_order_pathway_model.bin_size_and_range", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, **RANGE)


@case("modelling.conn_prob_2nd_order_pathway_model.populations_of_different_size", **FIT)
def _():
    adj, src, tgt = real_pathway(300, 200)
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, **RANGE)


@case("modelling.conn_prob_2nd_order_pathway_model.coordinates_in_two_dimensions", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, coord_names=["x", "z"], bin_size_um=50, max_range_um=400
    )


@case("modelling.conn_prob_2nd_order_pathway_model.planted_law_is_recovered", **FIT)
def _():
    adj, src, tgt = planted_pathway(90, 70, seed=5, a=0.5, b=0.008)
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, bin_size_um=50)


@case(
    "modelling.conn_prob_2nd_order_pathway_model.same_neuron_in_both_populations_is_excluded", **FIT
)
def _():
    # populations overlap in neurons 200..299: their pair has distance 0 and is dropped like an autapse
    adj, nodes = real(500)
    return M.conn_prob_2nd_order_pathway_model(
        adj[:300, 200:].tocsr(), nodes.iloc[:300], nodes.iloc[200:], **RANGE
    )


@case("modelling.conn_prob_2nd_order_pathway_model.subsets_of_both_populations", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=100, sample_seeds=[2, 6], **RANGE
    )


@case("modelling.conn_prob_2nd_order_pathway_model.subset_of_the_larger_population_only", **FIT)
def _():
    adj, src, tgt = real_pathway(300, 200)
    return M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=250, sample_seeds=[2, 6], **RANGE
    )


@case(
    "modelling.conn_prob_2nd_order_pathway_model.subset_of_the_larger_target_population_only", **FIT
)
def _():
    adj, src, tgt = real_pathway(200, 300)
    return M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=250, sample_seeds=[2, 6], **RANGE
    )


@case("modelling.conn_prob_2nd_order_pathway_model.generated_seeds", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=100, sample_seeds=3, meta_seed=4, **RANGE
    )


@case("modelling.conn_prob_2nd_order_pathway_model.seed_list_with_duplicates", **FIT)
def _():
    adj, src, tgt = real_pathway()
    with logged() as records:
        frame = M.conn_prob_2nd_order_pathway_model(
            adj, src, tgt, sample_size=100, sample_seeds=[7, 7, 3], **RANGE
        )
    return frame, records


@case(
    "modelling.conn_prob_2nd_order_pathway_model.sample_size_not_below_the_larger_population", **FIT
)
def _():
    adj, src, tgt = real_pathway(300, 200)
    with logged() as records:
        frame = M.conn_prob_2nd_order_pathway_model(
            adj, src, tgt, sample_size=300, sample_seeds=2, **RANGE
        )
    return frame, records


@case(
    "modelling.conn_prob_2nd_order_pathway_model.sample_size_below_the_larger_but_above_the_smaller",
    **FIT,
)
def _():
    adj, src, tgt = real_pathway(300, 200)
    return M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=299, sample_seeds=[5], **RANGE
    )


@case("modelling.conn_prob_2nd_order_pathway_model.global_random_stream_after_subsets", **FIT)
def _():
    adj, src, tgt = real_pathway()
    frame = M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=100, sample_seeds=2, **RANGE
    )
    return frame, np.random.rand(3)


@case(
    "modelling.conn_prob_2nd_order_pathway_model.global_random_stream_after_subset_of_one_population",
    **FIT,
)
def _():
    adj, src, tgt = real_pathway(300, 200)
    frame = M.conn_prob_2nd_order_pathway_model(
        adj, src, tgt, sample_size=250, sample_seeds=[5], **RANGE
    )
    return frame, np.random.rand(3)


@case("modelling.conn_prob_2nd_order_pathway_model.data_split_is_not_supported", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, N_split=2, **RANGE)


@case(
    "modelling.conn_prob_2nd_order_pathway_model.model_order_keyword_is_rejected",
    check_message=True,
)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, model_order=2)


@case(
    "modelling.conn_prob_2nd_order_pathway_model.sources_and_rows_of_different_size",
    check_message=True,
)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src.iloc[:100], tgt)


@case(
    "modelling.conn_prob_2nd_order_pathway_model.targets_and_columns_of_different_size",
    check_message=True,
)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt.iloc[:100])


@case("modelling.conn_prob_2nd_order_pathway_model.unknown_coordinate_column")
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt.rename(columns={"z": "zz"}))


@case("modelling.conn_prob_2nd_order_pathway_model.adjacency_formats", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return {
        name: M.conn_prob_2nd_order_pathway_model(a, src, tgt, **RANGE)
        for name, a in {
            "csc": adj.tocsc(),
            "lil": adj.tolil(),
            "dense": adj.toarray(),
            "float": adj.astype(float),
        }.items()
    }


@case("modelling.conn_prob_2nd_order_pathway_model.dense_adjacency_cannot_be_subsampled")
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_2nd_order_pathway_model(
        adj.toarray(), src, tgt, sample_size=100, sample_seeds=[1], **RANGE
    )


@case("modelling.conn_prob_3rd_order_pathway_model.default_arguments", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt)


@case("modelling.conn_prob_3rd_order_pathway_model.bin_size_and_range", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, **RANGE)


@case("modelling.conn_prob_3rd_order_pathway_model.populations_of_different_size", **FIT)
def _():
    adj, src, tgt = real_pathway(300, 200)
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, **RANGE)


@case("modelling.conn_prob_3rd_order_pathway_model.other_depth_column", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, depth_name="y", **RANGE)


@case("modelling.conn_prob_3rd_order_pathway_model.planted_bipolar_law_is_recovered", **FIT)
def _():
    adj, src, tgt = planted_pathway(100, 80, seed=6, a=0.5, b=0.008, bipolar=(0.3, 0.012))
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, bin_size_um=50)


@case("modelling.conn_prob_3rd_order_pathway_model.subsets_of_both_populations", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(
        adj, src, tgt, sample_size=150, sample_seeds=3, **RANGE
    )


@case("modelling.conn_prob_3rd_order_pathway_model.subset_of_the_larger_population_only", **FIT)
def _():
    adj, src, tgt = real_pathway(300, 200)
    return M.conn_prob_3rd_order_pathway_model(
        adj, src, tgt, sample_size=250, sample_seeds=[5, 6], **RANGE
    )


@case(
    "modelling.conn_prob_3rd_order_pathway_model.subset_of_the_larger_target_population_only", **FIT
)
def _():
    adj, src, tgt = real_pathway(200, 300)
    return M.conn_prob_3rd_order_pathway_model(
        adj, src, tgt, sample_size=250, sample_seeds=[5, 6], **RANGE
    )


@case("modelling.conn_prob_3rd_order_pathway_model.data_split_is_not_supported", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, N_split=3, **RANGE)


@case(
    "modelling.conn_prob_3rd_order_pathway_model.model_order_keyword_is_rejected",
    check_message=True,
)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, model_order=3)


@case("modelling.conn_prob_3rd_order_pathway_model.unknown_depth_column")
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, depth_name="no_such_column")


@case("modelling.conn_prob_2nd_order_pathway_model.neurons_spread_over_the_whole_tissue", **FIT)
def _():
    adj, src, tgt = real_spread_pathway()
    return M.conn_prob_2nd_order_pathway_model(adj, src, tgt, bin_size_um=100, max_range_um=1500)


@case("modelling.conn_prob_3rd_order_pathway_model.neurons_spread_over_the_whole_tissue", **FIT)
def _():
    adj, src, tgt = real_spread_pathway()
    return M.conn_prob_3rd_order_pathway_model(adj, src, tgt, bin_size_um=100, max_range_um=1500)


@case(
    "modelling.conn_prob_3rd_order_pathway_model.neurons_spread_over_the_whole_tissue_random_subsets",
    **FIT,
)
def _():
    adj, src, tgt = real_spread_pathway()
    return M.conn_prob_3rd_order_pathway_model(
        adj, src, tgt, bin_size_um=100, sample_size=200, sample_seeds=2, meta_seed=3
    )


@case("modelling.conn_prob_pathway_model.second_order", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt, model_order=2, **RANGE)


@case("modelling.conn_prob_pathway_model.third_order", **FIT)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt, model_order=3, **RANGE)


@case("modelling.conn_prob_pathway_model.same_as_the_order_specific_wrappers", **FIT)
def _():
    adj, src, tgt = real_pathway(300, 200)
    return [
        M.conn_prob_pathway_model(adj, src, tgt, model_order=2, **RANGE),
        M.conn_prob_2nd_order_pathway_model(adj, src, tgt, **RANGE),
        M.conn_prob_pathway_model(adj, src, tgt, model_order=3, **RANGE),
        M.conn_prob_3rd_order_pathway_model(adj, src, tgt, **RANGE),
    ]


@case("modelling.conn_prob_pathway_model.same_population_as_the_symmetric_version", **FIT)
def _():
    adj, nodes = real()
    return M.conn_prob_pathway_model(adj, nodes, nodes, model_order=2, **RANGE), M.conn_prob_model(
        adj, nodes, model_order=2, **RANGE
    )


@case("modelling.conn_prob_pathway_model.model_order_missing")
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt, **RANGE)


@case("modelling.conn_prob_pathway_model.model_order_not_supported", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt, model_order=4, **RANGE)


@case("modelling.conn_prob_pathway_model.data_split_is_not_supported", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt, model_order=2, N_split=2, **RANGE)


@case("modelling.conn_prob_pathway_model.sources_and_rows_of_different_size", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src.iloc[:100], tgt, model_order=2)


@case("modelling.conn_prob_pathway_model.targets_and_columns_of_different_size", check_message=True)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt.iloc[:100], model_order=2)


@case(
    "modelling.conn_prob_pathway_model.invalid_argument_is_checked_before_the_model_order",
    check_message=True,
)
def _():
    adj, src, tgt = real_pathway()
    return M.conn_prob_pathway_model(adj, src, tgt, do_plot=True)
