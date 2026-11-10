"""``run_batch_model_building``: model building from files (adjacency ``.npz``, node table ``.h5`` or ``.feather``,
config ``.json``) as used by the SLURM batch script of the tutorial. It returns nothing; what it does is visible in the
pickles written to the directories named in the config.

All files live in a temporary directory created inside each case; the config names its output folders with ``@name``,
replaced by that directory.
"""

import json
import os
import tempfile

import numpy as np
import scipy.sparse as sp
from support.cases import case

from connalysis.modelling import modelling as M

from .modelling_common import figures, pickles, planted_dd, real, tree, use_agg, without_path

FIT = {"rtol": 1e-6}
BASE = {
    "model_name": "net",
    "model_order": 2,
    "bin_size_um": 50,
    "max_range_um": 450,
    "model_dir": "@models",
    "data_dir": "@data",
}


def write_inputs(tmp, adj, nodes, ext, config, adj_name="adj.npz", cfg_name="config.json"):
    adj_file = os.path.join(tmp, adj_name)
    sp.save_npz(adj_file, adj)
    nrn_file = os.path.join(tmp, "nodes" + ext)
    if ext == ".h5":
        nodes.to_hdf(nrn_file, key="nodes", mode="w", format="table")
    elif ext == ".feather":
        nodes.reset_index(drop=True).to_feather(nrn_file)
    else:
        nodes.to_csv(nrn_file)
    resolved = {
        k: (os.path.join(tmp, v[1:]) if isinstance(v, str) and v.startswith("@") else v)
        for k, v in config.items()
    }
    cfg_file = os.path.join(tmp, cfg_name)
    with open(cfg_file, "w") as fh:
        json.dump(resolved, fh)
    return adj_file, nrn_file, cfg_file


def batch(config, ext=".h5", calls=({},), adj=None, nodes=None, **write):
    """Write the inputs, call ``run_batch_model_building`` once per entry of ``calls`` and report what it left behind."""
    if adj is None:
        adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        adj_file, nrn_file, cfg_file = write_inputs(tmp, adj, nodes, ext, config, **write)
        with without_path(tmp):
            returned = [
                M.run_batch_model_building(adj_file, nrn_file, cfg_file, **call) for call in calls
            ]
        return {"returned": returned, "files": tree(tmp), "pickles": pickles(tmp)}


# ----------------------------------------------------------------------------- whole runs


@case("modelling.run_batch_model_building.second_order_with_hdf5_node_table", **FIT)
def _():
    return batch(BASE, ".h5")


@case("modelling.run_batch_model_building.third_order_with_feather_node_table", **FIT)
def _():
    return batch({**BASE, "model_order": 3}, ".feather")


@case("modelling.run_batch_model_building.second_order_with_feather_node_table", **FIT)
def _():
    return batch(BASE, ".feather")


@case("modelling.run_batch_model_building.third_order_with_hdf5_node_table", **FIT)
def _():
    return batch({**BASE, "model_order": 3}, ".h5")


@case("modelling.run_batch_model_building.config_as_in_the_tutorial_file", **FIT)
def _():
    config = {
        "model_name": "ConnProb2ndOrder",
        "model_order": 2,
        "bin_size_um": 100,
        "max_range_um": 400,
        "sample_size": None,
        "sample_seed": None,
        "model_dir": "@modelling/models",
        "data_dir": "@modelling/data",
        "do_plot": False,
        "plot_dir": None,
    }
    return batch(config, ".h5")


@case("modelling.run_batch_model_building.every_option_in_the_config", **FIT)
def _():
    config = {
        **BASE,
        "model_order": 3,
        "sample_size": 200,
        "sample_seed": 7,
        "coord_names": ["x", "z"],
        "depth_name": "y",
        "max_range_um": 400,
        "do_plot": False,
        "plot_dir": "@figs",
        "N_split": None,
        "part_idx": None,
    }
    return batch(config, ".feather")


@case("modelling.run_batch_model_building.random_subset_from_the_config", **FIT)
def _():
    return batch({**BASE, "sample_size": 150, "sample_seed": 11}, ".h5")


@case("modelling.run_batch_model_building.floats_in_the_config", **FIT)
def _():
    return batch({**BASE, "bin_size_um": 50.0, "max_range_um": 450.0}, ".h5")


@case("modelling.run_batch_model_building.only_data_is_saved", **FIT)
def _():
    config = {k: v for k, v in BASE.items() if k != "model_dir"}
    return batch(config, ".h5")


@case("modelling.run_batch_model_building.only_the_model_is_saved", **FIT)
def _():
    config = {k: v for k, v in BASE.items() if k != "data_dir"}
    return batch(config, ".h5")


@case("modelling.run_batch_model_building.nothing_is_saved", **FIT)
def _():
    config = {k: v for k, v in BASE.items() if k not in ("model_dir", "data_dir")}
    return batch(config, ".h5")


@case("modelling.run_batch_model_building.unknown_config_keys_are_ignored", **FIT)
def _():
    return batch({**BASE, "comment": "not used", "threads": 4}, ".h5")


@case("modelling.run_batch_model_building.plots_from_the_config", **FIT)
def _():
    use_agg()
    result = batch({**BASE, "do_plot": True, "plot_dir": "@figs"}, ".h5")
    return result, figures()


@case("modelling.run_batch_model_building.planted_law_with_bipolar_dependency", **FIT)
def _():
    adj, nodes = planted_dd(120, seed=9, a=0.5, b=0.008, bipolar=(0.3, 0.012))
    return batch({**BASE, "model_order": 3, "max_range_um": 600}, ".feather", adj=adj, nodes=nodes)


# ----------------------------------------------------------------------------- data splits


@case("modelling.run_batch_model_building.parts_chosen_by_arguments_then_merged", **FIT)
def _():
    calls = [
        {"N_split": 3, "part_idx": 0},
        {"N_split": 3, "part_idx": 1},
        {"N_split": 3, "part_idx": 2},
        {"N_split": 3, "part_idx": -1},
    ]
    return batch(BASE, ".h5", calls=calls)


@case("modelling.run_batch_model_building.third_order_parts_then_merged", **FIT)
def _():
    calls = [
        {"N_split": 2, "part_idx": 0},
        {"N_split": 2, "part_idx": 1},
        {"N_split": 2, "part_idx": -1},
    ]
    return batch({**BASE, "model_order": 3}, ".feather", calls=calls)


@case("modelling.run_batch_model_building.split_in_the_config_part_by_argument", **FIT)
def _():
    calls = [{"part_idx": 0}, {"part_idx": 1}, {"part_idx": -1}]
    return batch({**BASE, "N_split": 2}, ".h5", calls=calls)


@case("modelling.run_batch_model_building.split_and_part_in_the_config", **FIT)
def _():
    return batch({**BASE, "N_split": 4, "part_idx": 2}, ".h5")


@case("modelling.run_batch_model_building.split_arguments_replace_the_config_values", **FIT)
def _():
    return batch(
        {**BASE, "N_split": 5, "part_idx": 4}, ".h5", calls=[{"N_split": 2, "part_idx": 0}]
    )


@case("modelling.run_batch_model_building.split_without_part_runs_all_parts_in_one_call", **FIT)
def _():
    return batch(BASE, ".h5", calls=[{"N_split": 3}])


@case("modelling.run_batch_model_building.split_arguments_given_as_text_are_converted", **FIT)
def _():
    return batch(BASE, ".h5", calls=[{"N_split": "3", "part_idx": "1"}])


@case("modelling.run_batch_model_building.split_arguments_as_numpy_integers", **FIT)
def _():
    return batch(BASE, ".h5", calls=[{"N_split": np.int64(2), "part_idx": np.int64(1)}])


@case("modelling.run_batch_model_building.random_subset_in_parts_then_merged", **FIT)
def _():
    calls = [
        {"N_split": 2, "part_idx": 0},
        {"N_split": 2, "part_idx": 1},
        {"N_split": 2, "part_idx": -1},
    ]
    return batch({**BASE, "sample_size": 200, "sample_seed": 5}, ".h5", calls=calls)


@case("modelling.run_batch_model_building.part_index_zero_is_applied", **FIT)
def _():
    return batch(BASE, ".h5", calls=[{"N_split": 2, "part_idx": 0}])


@case("modelling.run_batch_model_building.part_index_without_a_split", check_message=True)
def _():
    return batch(BASE, ".h5", calls=[{"part_idx": 0}])


@case("modelling.run_batch_model_building.merge_before_the_parts_exist", check_message=True)
def _():
    return batch(BASE, ".h5", calls=[{"N_split": 3, "part_idx": -1}])


@case("modelling.run_batch_model_building.split_without_a_maximum_range", check_message=True)
def _():
    return batch(
        {k: v for k, v in BASE.items() if k != "max_range_um"},
        ".h5",
        calls=[{"N_split": 3, "part_idx": 0}],
    )


# ----------------------------------------------------------------------------- adjacency files


@case("modelling.run_batch_model_building.adjacency_saved_as_csc", **FIT)
def _():
    adj, nodes = real()
    return batch(BASE, ".h5", adj=adj.tocsc(), nodes=nodes)


@case("modelling.run_batch_model_building.weighted_adjacency_saved_as_float", **FIT)
def _():
    adj, nodes = real()
    return batch(BASE, ".h5", adj=adj.astype(np.float32) * 1.5, nodes=nodes)


@case("modelling.run_batch_model_building.adjacency_saved_as_coo_without_subsampling")
def _():
    adj, nodes = real()
    return batch(BASE, ".h5", adj=adj.tocoo(), nodes=nodes)


@case("modelling.run_batch_model_building.adjacency_saved_as_coo_with_subsampling", **FIT)
def _():
    adj, nodes = real()
    return batch(
        {**BASE, "sample_size": 200, "sample_seed": 3}, ".h5", adj=adj.tocoo(), nodes=nodes
    )


# ----------------------------------------------------------------------------- invalid input


@case("modelling.run_batch_model_building.node_table_in_csv_format", check_message=True)
def _():
    return batch(BASE, ".csv")


@case(
    "modelling.run_batch_model_building.node_table_extension_in_capital_letters", check_message=True
)
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        adj_file, nrn_file, cfg_file = write_inputs(tmp, adj, nodes, ".h5", BASE)
        upper = os.path.join(tmp, "nodes.H5")
        os.rename(nrn_file, upper)
        return M.run_batch_model_building(adj_file, upper, cfg_file)


@case(
    "modelling.run_batch_model_building.node_table_with_other_length_than_the_adjacency",
    check_message=True,
)
def _():
    adj, nodes = real()
    return batch(BASE, ".h5", adj=adj, nodes=nodes.iloc[:450])


@case("modelling.run_batch_model_building.non_square_adjacency", check_message=True)
def _():
    adj, nodes = real()
    return batch(BASE, ".feather", adj=adj[:, :450], nodes=nodes)


@case("modelling.run_batch_model_building.config_without_model_name")
def _():
    return batch({k: v for k, v in BASE.items() if k != "model_name"}, ".h5")


@case("modelling.run_batch_model_building.config_without_model_order")
def _():
    return batch({k: v for k, v in BASE.items() if k != "model_order"}, ".h5")


@case("modelling.run_batch_model_building.config_with_unsupported_model_order", check_message=True)
def _():
    return batch({**BASE, "model_order": 4}, ".h5")


@case("modelling.run_batch_model_building.config_with_unknown_coordinate_column")
def _():
    return batch({**BASE, "coord_names": ["x", "y", "w"]}, ".h5")


@case("modelling.run_batch_model_building.missing_adjacency_file")
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        _, nrn_file, cfg_file = write_inputs(tmp, adj, nodes, ".h5", BASE)
        with without_path(tmp):
            return M.run_batch_model_building(os.path.join(tmp, "absent.npz"), nrn_file, cfg_file)


@case("modelling.run_batch_model_building.missing_config_file")
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        adj_file, nrn_file, _ = write_inputs(tmp, adj, nodes, ".h5", BASE)
        with without_path(tmp):
            return M.run_batch_model_building(adj_file, nrn_file, os.path.join(tmp, "absent.json"))


@case("modelling.run_batch_model_building.missing_node_table_file")
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        adj_file, _, cfg_file = write_inputs(tmp, adj, nodes, ".h5", BASE)
        with without_path(tmp):
            return M.run_batch_model_building(adj_file, os.path.join(tmp, "absent.h5"), cfg_file)


@case(
    "modelling.run_batch_model_building.unsupported_table_format_is_found_before_the_config_is_read",
    check_message=True,
)
def _():
    adj, nodes = real()
    with tempfile.TemporaryDirectory() as tmp:
        adj_file, _, _ = write_inputs(tmp, adj, nodes, ".h5", BASE)
        return M.run_batch_model_building(
            adj_file, os.path.join(tmp, "nodes.csv"), os.path.join(tmp, "absent.json")
        )
