"""``per_group_in_out_bias_model``: per-neuron weights from the number of connections between groups of neurons.

Only the number of stored matrix entries per pair of groups is used (weights are not), and the group names come from
a node-table column (strings, integers, or a categorical column that may have unused categories, as in the tutorial
connectome).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, node_table, weighted

from connalysis.modelling import modelling as M

from .modelling_common import real, real_spread

BIAS = M.per_group_in_out_bias_model


def small(seed=3, n=40, p=0.2, n_groups=4):
    return digraph(n, p, seed=seed), node_table(n, seed=seed, n_mtypes=n_groups)


# ----------------------------------------------------------------------------- typical use


@case("modelling.per_group_in_out_bias_model.axis0_string_groups")
def _():
    adj, nodes = small()
    return BIAS(adj, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.axis1_string_groups")
def _():
    adj, nodes = small()
    return BIAS(adj, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.axes_differ_for_a_directed_network")
def _():
    adj, nodes = small(seed=4, n=60, p=0.12, n_groups=5)
    return BIAS(adj, nodes, "mtype", 0), BIAS(adj, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.integer_groups")
def _():
    adj, nodes = small(seed=5)
    return BIAS(adj, nodes, "layer", 0), BIAS(adj, nodes, "layer", 1)


@case("modelling.per_group_in_out_bias_model.float_groups")
def _():
    adj, nodes = small(seed=6)
    nodes["bucket"] = np.round(nodes["x"] / 125.0)
    return BIAS(adj, nodes, "bucket", 1)


@case("modelling.per_group_in_out_bias_model.categorical_groups")
def _():
    adj, nodes = small(seed=7)
    nodes["mtype"] = pd.Categorical(nodes["mtype"])
    return BIAS(adj, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.categorical_groups_with_unused_categories")
def _():
    adj, nodes = small(seed=7)
    nodes["mtype"] = pd.Categorical(
        nodes["mtype"], categories=["m0", "m1", "m2", "m3", "m9", "m10"]
    )
    return BIAS(adj, nodes, "mtype", 0), BIAS(adj, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.many_groups")
def _():
    adj, nodes = small(seed=8, n=80, p=0.1, n_groups=12)
    return BIAS(adj, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.groups_do_not_depend_on_the_table_index")
def _():
    adj, nodes = small(seed=9)
    shifted = nodes.copy()
    shifted.index = [f"cell{i}" for i in range(len(nodes))]
    return BIAS(adj, nodes, "mtype", 0), BIAS(adj, shifted, "mtype", 0)


# ----------------------------------------------------------------------------- tutorial connectome


@case("modelling.per_group_in_out_bias_model.real_data_mtype_as_target_bias")
def _():
    adj, nodes = real()
    return BIAS(adj, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.real_data_mtype_as_source_bias")
def _():
    adj, nodes = real()
    return BIAS(adj, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.real_data_layer")
def _():
    adj, nodes = real(700)
    return BIAS(adj, nodes, "layer", 0), BIAS(adj, nodes, "layer", 1)


@case("modelling.per_group_in_out_bias_model.real_data_mtype_as_plain_strings")
def _():
    adj, nodes = real()
    nodes["mtype"] = nodes["mtype"].astype(str)
    return BIAS(adj, nodes, "mtype", 1), BIAS(adj, real()[1], "mtype", 1)


@case("modelling.per_group_in_out_bias_model.real_data_spread_over_all_layers_and_50_mtypes")
def _():
    adj, nodes = real_spread()
    return BIAS(adj, nodes, "mtype", 0), BIAS(adj, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.real_data_spread_grouped_by_layer")
def _():
    adj, nodes = real_spread()
    return BIAS(adj, nodes, "layer", 0), BIAS(adj, nodes, "layer", 1)


# ----------------------------------------------------------------------------- degenerate networks


@case("modelling.per_group_in_out_bias_model.group_without_any_connection_as_target_bias")
def _():
    adj, nodes = small(seed=10)
    silent = (nodes["mtype"] == "m0").to_numpy()
    dense = adj.toarray()
    dense[silent, :] = False
    dense[:, silent] = False
    return BIAS(sp.csr_matrix(dense), nodes, "mtype", 0), nodes["mtype"].to_numpy()


@case("modelling.per_group_in_out_bias_model.group_without_any_connection_as_source_bias")
def _():
    adj, nodes = small(seed=10)
    silent = (nodes["mtype"] == "m0").to_numpy()
    dense = adj.toarray()
    dense[silent, :] = False
    dense[:, silent] = False
    return BIAS(sp.csr_matrix(dense), nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.group_that_only_receives")
def _():
    adj, nodes = small(seed=11)
    dense = adj.toarray()
    mine = (nodes["mtype"] == "m1").to_numpy()
    dense[mine, :] = False  # no outgoing connections of this group; incoming ones stay
    mat = sp.csr_matrix(dense)
    return BIAS(mat, nodes, "mtype", 0), BIAS(mat, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.no_connections_at_all")
def _():
    _, nodes = small(seed=12)
    return BIAS(sp.csr_matrix((40, 40), dtype=bool), nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.single_group")
def _():
    adj, nodes = small(seed=13)
    nodes["all_same"] = "a"
    return BIAS(adj, nodes, "all_same", 0), BIAS(adj, nodes, "all_same", 1)


@case("modelling.per_group_in_out_bias_model.complete_network")
def _():
    nodes = node_table(20, seed=14, n_mtypes=3)
    mat = sp.csr_matrix(np.ones((20, 20), dtype=bool) & ~np.eye(20, dtype=bool))
    return BIAS(mat, nodes, "mtype", 0), BIAS(mat, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.group_of_one_neuron")
def _():
    adj, nodes = small(seed=15, n=30, p=0.25, n_groups=3)
    nodes["who"] = np.where(
        np.arange(30) == 7, "alone", np.where(np.arange(30) % 2 == 0, "even", "odd")
    )
    return BIAS(adj, nodes, "who", 0), BIAS(adj, nodes, "who", 1)


@case("modelling.per_group_in_out_bias_model.self_connections_are_counted")
def _():
    adj, nodes = small(seed=16)
    with_self = adj.tolil()
    with_self.setdiag(True)
    return BIAS(with_self.tocsr(), nodes, "mtype", 0), BIAS(adj, nodes, "mtype", 0)


# ----------------------------------------------------------------------------- what counts as a connection


@case("modelling.per_group_in_out_bias_model.weights_are_ignored_only_the_number_of_entries_counts")
def _():
    adj = weighted(40, 0.2, seed=3, integers=True)
    nodes = node_table(40, seed=3)
    plain = digraph(40, 0.2, seed=3)
    return BIAS(adj, nodes, "mtype", 0), BIAS(plain, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.explicit_zero_entries_are_counted")
def _():
    adj = weighted(40, 0.2, seed=3, integers=True)
    nodes = node_table(40, seed=3)
    with_zeros = adj.copy()
    with_zeros.data[::2] = 0.0
    cleaned = with_zeros.copy()
    cleaned.eliminate_zeros()
    return (
        with_zeros.nnz,
        cleaned.nnz,
        BIAS(with_zeros, nodes, "mtype", 1),
        BIAS(cleaned, nodes, "mtype", 1),
    )


@case("modelling.per_group_in_out_bias_model.negative_weights_count_like_any_other_entry")
def _():
    adj = weighted(40, 0.2, seed=3, integers=True)
    adj.data[::3] *= -1.0
    return BIAS(adj, node_table(40, seed=3), "mtype", 0)


@case("modelling.per_group_in_out_bias_model.matrix_formats_give_the_same_weights")
def _():
    adj, nodes = small(seed=17)
    return {
        name: BIAS(m, nodes, "mtype", 1)
        for name, m in {
            "csr": adj,
            "csc": adj.tocsc(),
            "lil": adj.tolil(),
            "coo": adj.tocoo(),
            "csr_array": sp.csr_array(adj),
            "coo_array": sp.coo_array(adj),
        }.items()
    }


@case("modelling.per_group_in_out_bias_model.dtypes_give_the_same_weights")
def _():
    adj, nodes = small(seed=18)
    return {
        np.dtype(dt).name: BIAS(adj.astype(dt), nodes, "mtype", 0)
        for dt in (bool, np.int8, np.int64, np.float32, np.float64)
    }


@case("modelling.per_group_in_out_bias_model.input_matrix_is_left_unchanged")
def _():
    adj, nodes = small(seed=19)
    lil = adj.tolil()
    before = lil.toarray().copy()
    result = BIAS(lil, nodes, "mtype", 1)
    return result, bool((lil.toarray() == before).all()), type(lil).__name__


@case("modelling.per_group_in_out_bias_model.dense_matrix")
def _():
    adj, nodes = small(seed=20)
    return BIAS(adj.toarray(), nodes, "mtype", 0)


# ----------------------------------------------------------------------------- unusual axis values and tables


@case("modelling.per_group_in_out_bias_model.axis_minus_one_is_accepted_like_axis_one")
def _():
    adj, nodes = small(seed=21)
    return BIAS(adj, nodes, "mtype", -1), BIAS(adj, nodes, "mtype", 1)


@case("modelling.per_group_in_out_bias_model.axis_minus_two_is_accepted_like_axis_zero")
def _():
    adj, nodes = small(seed=21)
    return BIAS(adj, nodes, "mtype", -2), BIAS(adj, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.axis_none_gives_every_neuron_the_same_weight")
def _():
    adj, nodes = small(seed=22)
    return BIAS(adj, nodes, "mtype", None)


@case("modelling.per_group_in_out_bias_model.axis_two")
def _():
    adj, nodes = small(seed=23)
    return BIAS(adj, nodes, "mtype", 2)


@case("modelling.per_group_in_out_bias_model.axis_as_float")
def _():
    adj, nodes = small(seed=23)
    return BIAS(adj, nodes, "mtype", 1.0)


@case("modelling.per_group_in_out_bias_model.axis_as_text")
def _():
    adj, nodes = small(seed=23)
    return BIAS(adj, nodes, "mtype", "0")


@case("modelling.per_group_in_out_bias_model.unknown_property")
def _():
    adj, nodes = small(seed=24)
    return BIAS(adj, nodes, "no_such_column", 0)


@case("modelling.per_group_in_out_bias_model.table_shorter_than_the_matrix_is_not_checked")
def _():
    adj, nodes = small(seed=25)
    return BIAS(adj, nodes.iloc[:30], "mtype", 0)


@case("modelling.per_group_in_out_bias_model.table_longer_than_the_matrix")
def _():
    adj, nodes = small(seed=26)
    return BIAS(adj, node_table(50, seed=26), "mtype", 0)


@case("modelling.per_group_in_out_bias_model.missing_group_name")
def _():
    adj, nodes = small(seed=27)
    nodes.loc[3, "mtype"] = np.nan
    return BIAS(adj, nodes, "mtype", 0)


@case("modelling.per_group_in_out_bias_model.boolean_groups")
def _():
    adj, nodes = small(seed=28)
    nodes["flag"] = np.arange(40) % 3 == 0
    return BIAS(adj, nodes, "flag", 0)


@case("modelling.per_group_in_out_bias_model.non_square_matrix")
def _():
    adj, nodes = small(seed=29)
    return BIAS(adj[:, :35], nodes, "mtype", 0)
