"""Small helpers of ``connalysis.modelling.modelling``: distance and bipolar matrices, model functions from their
string form (``eval``), seed generation, data part names, and the pickle files written/merged for data splits."""

import os
import pickle
import tempfile

import numpy as np
from support.cases import case
from support.data import positions

from connalysis.modelling import modelling as M

from .modelling_common import pickles, tree, without_path

# Model strings exactly as ``run_model_building`` stores them in the model dictionary (and in old pickle files).
MODEL_2 = "exp_model_scale * np.exp(-exp_model_exponent * np.array(d))"
MODEL_3 = (
    "np.select([np.array(dz) < 0, np.array(dz) > 0, np.array(dz) == 0], "
    "[bip_neg_exp_model_scale * np.exp(-bip_neg_exp_model_exponent * np.array(d)), "
    "bip_pos_exp_model_scale * np.exp(-bip_pos_exp_model_exponent * np.array(d)), "
    "0.5 * (bip_neg_exp_model_scale * np.exp(-bip_neg_exp_model_exponent * np.array(d)) + "
    "bip_pos_exp_model_scale * np.exp(-bip_pos_exp_model_exponent * np.array(d)))])"
)
PARAMS_2 = {"exp_model_scale": 0.35, "exp_model_exponent": 0.012}
PARAMS_3 = {
    "bip_neg_exp_model_scale": 0.4,
    "bip_neg_exp_model_exponent": 0.01,
    "bip_pos_exp_model_scale": 0.25,
    "bip_pos_exp_model_exponent": 0.02,
}
D_GRID = np.array([0.0, 1.0, 50.0, 100.0, 250.5, 400.0, 1000.0, 5000.0])


def model_2():
    return M._get_model_function(MODEL_2, ["d"], dict(PARAMS_2))


def model_3():
    return M._get_model_function(MODEL_3, ["d", "dz"], dict(PARAMS_3))


# ----------------------------------------------------------------------------- _compute_dist_matrix


@case("modelling.compute_dist_matrix.rectangular")
def _():
    return M._compute_dist_matrix(
        positions(7, seed=1, scale=300.0), positions(11, seed=2, scale=300.0)
    )


@case("modelling.compute_dist_matrix.same_points_diagonal_nan")
def _():
    pos = positions(8, seed=3, scale=200.0)
    return M._compute_dist_matrix(pos, pos)


@case("modelling.compute_dist_matrix.zero_distance_between_distinct_neurons")
def _():
    # rows 0 and 2 sit at the same place: their (non-autaptic) pair is excluded as well
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [6.0, 8.0, 0.0]])
    return M._compute_dist_matrix(pos, pos)


@case("modelling.compute_dist_matrix.one_two_three_coordinates")
def _():
    pos = positions(6, seed=4, scale=100.0)
    return {k: M._compute_dist_matrix(pos[:4, :k], pos[:, :k]) for k in (1, 2, 3)}


@case("modelling.compute_dist_matrix.integer_positions")
def _():
    pos = np.array([[0, 0, 0], [3, 4, 0], [6, 8, 0], [0, 12, 5]])
    return M._compute_dist_matrix(pos, pos[::-1])


@case("modelling.compute_dist_matrix.empty_source")
def _():
    return M._compute_dist_matrix(np.empty((0, 3)), positions(5, seed=5))


@case("modelling.compute_dist_matrix.nan_coordinate")
def _():
    pos = positions(5, seed=6, scale=100.0)
    pos[2, 1] = np.nan
    return M._compute_dist_matrix(pos, pos)


# ----------------------------------------------------------------------------- _compute_dist_matrix_symmetric


@case("modelling.compute_dist_matrix_symmetric.random")
def _():
    return M._compute_dist_matrix_symmetric(positions(9, seed=7, scale=300.0))


@case("modelling.compute_dist_matrix_symmetric.zero_distance_between_distinct_neurons")
def _():
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [6.0, 8.0, 0.0]])
    return M._compute_dist_matrix_symmetric(pos)


@case("modelling.compute_dist_matrix_symmetric.one_two_three_coordinates")
def _():
    pos = positions(6, seed=8, scale=100.0)
    return {k: M._compute_dist_matrix_symmetric(pos[:, :k]) for k in (1, 2, 3)}


@case("modelling.compute_dist_matrix_symmetric.single_neuron")
def _():
    return M._compute_dist_matrix_symmetric(np.array([[1.0, 2.0, 3.0]]))


@case("modelling.compute_dist_matrix_symmetric.two_neurons")
def _():
    return M._compute_dist_matrix_symmetric(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]))


@case("modelling.compute_dist_matrix_symmetric.integer_positions")
def _():
    return M._compute_dist_matrix_symmetric(np.array([[0, 0, 0], [3, 4, 0], [6, 8, 0], [0, 12, 5]]))


@case("modelling.compute_dist_matrix_symmetric.nan_coordinate")
def _():
    pos = positions(5, seed=9, scale=100.0)
    pos[1, 0] = np.nan
    return M._compute_dist_matrix_symmetric(pos)


@case("modelling.compute_dist_matrix_symmetric.same_as_general_version")
def _():
    pos = positions(12, seed=10, scale=400.0)
    return M._compute_dist_matrix_symmetric(pos), M._compute_dist_matrix(pos, pos)


# ----------------------------------------------------------------------------- _compute_bip_matrix


@case("modelling.compute_bip_matrix.float_depths_with_ties")
def _():
    depth = np.array([36.0, 110.0, 36.0, 68.0, 68.0, 405.0, 377.0])
    return M._compute_bip_matrix(depth, depth)


@case("modelling.compute_bip_matrix.integer_depths")
def _():
    depth = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    return M._compute_bip_matrix(depth, depth)


@case("modelling.compute_bip_matrix.rectangular")
def _():
    return M._compute_bip_matrix(
        np.array([10.0, 20.0, 30.0]), np.array([5.0, 20.0, 35.0, 10.0, 40.0])
    )


@case("modelling.compute_bip_matrix.nan_depth")
def _():
    depth = np.array([1.0, np.nan, 3.0, 2.0])
    return M._compute_bip_matrix(depth, depth)


@case("modelling.compute_bip_matrix.constant_depth")
def _():
    depth = np.full(4, 7.0)
    return M._compute_bip_matrix(depth, depth)


@case("modelling.compute_bip_matrix.single_neuron")
def _():
    return M._compute_bip_matrix(np.array([2.0]), np.array([2.0]))


@case("modelling.compute_bip_matrix.rounded_depths_of_a_node_table")
def _():
    # depth values as found in a node table: rounded, many ties, pre-synaptic depth 0 included
    depth = np.round(positions(15, seed=11, dims=1, scale=300.0)[:, 0] / 10.0)
    depth[0] = 0.0
    return M._compute_bip_matrix(depth, depth[::-1])


@case("modelling.compute_matrices.inputs_are_not_modified")
def _():
    pos = positions(6, seed=12, scale=100.0)
    depth = np.array([3.0, 1.0, np.nan, 2.0, 2.0, 5.0])
    before = (pos.copy(), depth.copy())
    results = (
        M._compute_dist_matrix(pos, pos),
        M._compute_dist_matrix_symmetric(pos),
        M._compute_bip_matrix(depth, depth),
    )
    return (
        results,
        bool(np.array_equal(pos, before[0])),
        bool(np.array_equal(depth, before[1], equal_nan=True)),
    )


# ----------------------------------------------------------------------------- _get_model_function (eval of the model string)


@case("modelling.get_model_function.second_order_arrays")
def _():
    f = model_2()
    return {
        "1d": f(D_GRID),
        "list": f(D_GRID.tolist()),
        "2d": f(D_GRID.reshape(2, 4)),
        "scalar": f(100.0),
        "int": f(100),
        "nan_inf_negative": f(np.array([np.nan, np.inf, -50.0, -np.inf])),
        "empty": f(np.empty(0)),
    }


@case("modelling.get_model_function.second_order_scalar_type")
def _():
    f = model_2()
    return [f(0.0), f(np.float32(10.0)), f(np.array(10.0))]


@case("modelling.get_model_function.third_order_sign_branches")
def _():
    f = model_3()
    return {
        "neg": f(D_GRID, -1),
        "pos": f(D_GRID, 1),
        "zero": f(D_GRID, 0),
        "other_values": f(D_GRID[:4], np.array([-7.5, 0.1, 2, -0.0])),
        "per_element": f(D_GRID, np.array([-1, 1, 0, -1, 1, 0, -1, 1])),
        "nan_sign_gives_zero": f(D_GRID[:3], np.array([np.nan, 1, np.nan])),
    }


@case("modelling.get_model_function.third_order_shapes")
def _():
    f = model_3()
    d = D_GRID.reshape(2, 4)
    dz = np.array([[-1, 1, 0, 1], [0, -1, -1, 1]])
    return {
        "matrix": f(d, dz),
        "lists": f(D_GRID[:3].tolist(), [1, -1, 0]),
        "scalars": f(10.0, -1),
        "scalar_distance_array_sign": f(10.0, np.array([-1, 0, 1])),
        "int_inputs": f(3, 1),
    }


@case("modelling.get_model_function.model_params_keyword_overrides_bound_parameters")
def _():
    f = model_2()
    return f(D_GRID), f(D_GRID, model_params={"exp_model_scale": 1.0, "exp_model_exponent": 0.5})


@case("modelling.get_model_function.model_params_keyword_third_order")
def _():
    f = model_3()
    other = {**PARAMS_3, "bip_pos_exp_model_scale": 0.9}
    return f(D_GRID[:4], 1), f(D_GRID[:4], 1, model_params=other)


@case("modelling.get_model_function.custom_two_inputs")
def _():
    f = M._get_model_function(
        "a * np.sqrt(x) + b * y ** 2 - c", ["x", "y"], {"a": 2.0, "b": 0.5, "c": 1.0}
    )
    return f(np.array([1.0, 4.0, 9.0]), np.array([0.0, 1.0, 2.0])), f(16.0, 2.0)


@case("modelling.get_model_function.custom_without_parameters")
def _():
    f = M._get_model_function("np.log1p(d) * 2", ["d"], {})
    return f(np.array([0.0, 1.0, np.e - 1.0]))


@case("modelling.get_model_function.custom_without_inputs")
def _():
    f = M._get_model_function("a + b", [], {"a": 1.5, "b": 2.0})
    return f()


@case("modelling.get_model_function.parameter_order_follows_dictionary")
def _():
    # model_params are passed by keyword, so their order only matters for the generated lambda signature
    f = M._get_model_function("a - b", ["d"], {"b": 1.0, "a": 10.0})
    return f(0.0)


@case("modelling.get_model_function.model_from_a_stored_pickle")
def _():
    # the tutorial loads {'model', 'model_inputs', 'model_params'} from a pickle and calls _get_model_function(**dict)
    stored = pickle.loads(
        pickle.dumps({"model": MODEL_2, "model_inputs": ["d"], "model_params": dict(PARAMS_2)})
    )
    return M._get_model_function(**stored)(np.arange(0.0, 501.0, 100.0))


@case("modelling.get_model_function.syntax_error")
def _():
    return M._get_model_function("a *", ["d"], {"a": 1.0})


@case("modelling.get_model_function.undefined_name_when_called")
def _():
    return M._get_model_function("a * undefined_name", ["d"], {"a": 1.0})(1.0)


@case("modelling.get_model_function.missing_input_argument")
def _():
    return model_3()(D_GRID)


# ----------------------------------------------------------------------------- _generate_seeds (global numpy RNG)


@case("modelling.generate_seeds.default_digits")
def _():
    return {n: M._generate_seeds(n) for n in (1, 2, 3, 5, 12)}


@case("modelling.generate_seeds.meta_seeds")
def _():
    return {m: M._generate_seeds(4, meta_seed=m) for m in (0, 1, 7, 12345, 2**31)}


@case("modelling.generate_seeds.number_of_digits")
def _():
    return {k: M._generate_seeds(5, num_digits=k, meta_seed=3) for k in (2, 3, 4, 6)}


@case("modelling.generate_seeds.many_seeds_unique_and_sorted")
def _():
    seeds = M._generate_seeds(200, meta_seed=5)
    return seeds, len(set(seeds)), seeds == sorted(seeds), min(seeds), max(seeds)


@case("modelling.generate_seeds.whole_population")
def _():
    return M._generate_seeds(9, num_digits=1, meta_seed=2)


@case("modelling.generate_seeds.element_types")
def _():
    return [type(s).__name__ for s in M._generate_seeds(3)]


@case("modelling.generate_seeds.global_stream_afterwards")
def _():
    seeds = M._generate_seeds(3, meta_seed=9)
    return seeds, np.random.rand(3)


@case("modelling.generate_seeds.reseeds_the_global_stream")
def _():
    np.random.rand(5)
    M._generate_seeds(1, meta_seed=4)
    first = np.random.rand(2)
    np.random.rand(5)
    M._generate_seeds(1, meta_seed=4)
    return first, np.random.rand(2)


@case("modelling.generate_seeds.zero", check_message=True)
def _():
    return M._generate_seeds(0)


@case("modelling.generate_seeds.negative", check_message=True)
def _():
    return M._generate_seeds(-3)


@case("modelling.generate_seeds.float_count", check_message=True)
def _():
    return M._generate_seeds(2.0)


@case("modelling.generate_seeds.numpy_integer_count", check_message=True)
def _():
    return M._generate_seeds(np.int64(3))


@case("modelling.generate_seeds.list_count", check_message=True)
def _():
    return M._generate_seeds([1, 2, 3])


@case("modelling.generate_seeds.zero_digits", check_message=True)
def _():
    return M._generate_seeds(3, num_digits=0)


@case("modelling.generate_seeds.more_seeds_than_numbers")
def _():
    return M._generate_seeds(10, num_digits=1)


# ----------------------------------------------------------------------------- _get_data_part_name


@case("modelling.get_data_part_name.padding")
def _():
    return [
        M._get_data_part_name(n, p)
        for n, p in [
            (2, 0),
            (2, 1),
            (5, 4),
            (9, 8),
            (10, 0),
            (10, 9),
            (12, 3),
            (12, 11),
            (100, 7),
            (100, 99),
            (1000, 25),
        ]
    ]


@case("modelling.get_data_part_name.merge_marker_and_numpy_integers")
def _():
    return [
        M._get_data_part_name(7, -1),
        M._get_data_part_name(10, -1),
        M._get_data_part_name(np.int64(12), np.int64(5)),
        M._get_data_part_name(1, 0),
    ]


# ----------------------------------------------------------------------------- _save_data


def _save_and_read(save_dir_name, model_name, spec, content=None):
    content = (
        {"count": np.arange(4), "nested": {"a": [1.5, 2.5], "b": "text"}}
        if content is None
        else content
    )
    with tempfile.TemporaryDirectory() as tmp:
        target = None if save_dir_name is None else os.path.join(tmp, *save_dir_name)
        result = M._save_data(content, target, model_name, spec)
        return {"returned": result, "files": tree(tmp), "content": pickles(tmp)}


@case("modelling.save_data.with_specification")
def _():
    return _save_and_read(["out"], "mymodel", "model")


@case("modelling.save_data.without_specification")
def _():
    return _save_and_read(["out"], "mymodel", None)


@case("modelling.save_data.creates_nested_directories")
def _():
    return _save_and_read(["a", "b", "c"], "m", "data__part-3-1")


@case("modelling.save_data.model_name_none_is_written_as_text")
def _():
    return _save_and_read(["out"], None, "model")


@case("modelling.save_data.no_directory_writes_nothing")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            M._save_data({"x": 1}, None, "m", "model"),
            M._save_data({"x": 1}, "", "m", "model"),
        ]
        return results, tree(tmp)


@case("modelling.save_data.overwrites_existing_file")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        M._save_data({"v": 1}, tmp, "m", "model")
        M._save_data({"v": 2, "extra": np.ones(2)}, tmp, "m", "model")
        return tree(tmp), pickles(tmp)


@case("modelling.save_data.empty_dictionary_is_still_written")
def _():
    return _save_and_read(["out"], "m", "model", content={})


# ----------------------------------------------------------------------------- _merge_data


def _write_parts(tmp, name, spec, parts, N):
    for idx, part in enumerate(parts):
        with open(
            os.path.join(tmp, f"{name}__{spec}{M._get_data_part_name(N, idx)}.pickle"), "wb"
        ) as fh:
            pickle.dump(part, fh)
    return [M._get_data_part_name(N, p) for p in range(N)]


def _parts_2nd(counts_conn, counts_all, bins=(0, 100, 200, 300)):
    return [
        {
            "p_conn_dist": np.array(c, dtype=float) / np.maximum(a, 1),
            "count_conn": np.array(c),
            "count_all": np.array(a),
            "dist_bins": np.array(bins),
        }
        for c, a in zip(counts_conn, counts_all)
    ]


@case("modelling.merge_data.second_order_three_parts")
def _():
    parts = _parts_2nd([[1, 2, 0], [3, 1, 0], [0, 4, 0]], [[10, 20, 0], [30, 10, 0], [5, 40, 0]])
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "mdl", "data", parts, 3)
        return M._merge_data(tmp, "mdl", "data", names)


@case("modelling.merge_data.third_order_two_parts")
def _():
    parts = [
        {
            "p_conn_dist_bip": np.zeros((3, 2)),
            "count_conn": np.array([[1, 2], [3, 4], [0, 0]]),
            "count_all": np.array([[10, 12], [20, 30], [0, 0]]),
            "dist_bins": np.array([0, 50, 100, 150]),
            "bip_bins": [-1.0, 0, 1.0],
        },
        {
            "p_conn_dist_bip": np.ones((3, 2)),
            "count_conn": np.array([[0, 1], [2, 2], [1, 0]]),
            "count_all": np.array([[5, 6], [10, 11], [2, 0]]),
            "dist_bins": np.array([0, 50, 100, 150]),
            "bip_bins": [-1.0, 0, 1.0],
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m3", "data", parts, 2)
        return M._merge_data(tmp, "m3", "data", names)


@case("modelling.merge_data.key_order_is_probability_counts_then_other_keys_sorted")
def _():
    parts = [
        {
            "zeta": "last",
            "count_all": np.array([2, 2]),
            "dist_bins": np.array([0, 1, 2]),
            "p_conn_dist": np.zeros(2),
            "alpha": [1, 2],
            "count_conn": np.array([1, 0]),
        }
        for _ in range(2)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 2)
        merged = M._merge_data(tmp, "m", "data", names)
        return list(merged.keys()), merged


@case("modelling.merge_data.parts_are_added_in_any_order_and_float_counts")
def _():
    parts = [
        {
            "p_conn_dist": np.zeros(2),
            "count_conn": np.array([0.5, 1.0]),
            "count_all": np.array([2.0, 4.0]),
            "dist_bins": np.array([0.0, 50.0, 100.0]),
        },
        {
            "p_conn_dist": np.zeros(2),
            "count_conn": np.array([1.0, 0.0]),
            "count_all": np.array([2.0, 0.0]),
            "dist_bins": np.array([0.0, 50.0, 100.0]),
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 2)
        return M._merge_data(tmp, "m", "data", names[::-1])


@case("modelling.merge_data.a_single_part")
def _():
    parts = _parts_2nd([[1, 2, 3]], [[4, 5, 6]])
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 1)
        return M._merge_data(tmp, "m", "data", names)


@case("modelling.merge_data.bins_without_any_pair_give_nan")
def _():
    parts = _parts_2nd([[0, 1, 0], [0, 0, 0]], [[0, 2, 0], [0, 2, 0]])
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 2)
        return M._merge_data(tmp, "m", "data", names)


@case("modelling.merge_data.missing_part_file", check_message=True)
def _():
    parts = _parts_2nd([[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]])
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 2)
        with without_path(tmp):
            return M._merge_data(tmp, "m", "data", names + [M._get_data_part_name(3, 2)])


@case("modelling.merge_data.two_probability_entries")
def _():
    parts = [{**p, "p_extra": np.zeros(3)} for p in _parts_2nd([[1, 1, 1]], [[2, 2, 2]])]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 1)
        return M._merge_data(tmp, "m", "data", names)


@case("modelling.merge_data.no_probability_entry")
def _():
    parts = [
        {"count_conn": np.array([1]), "count_all": np.array([2]), "dist_bins": np.array([0, 1])}
    ]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 1)
        return M._merge_data(tmp, "m", "data", names)


@case("modelling.merge_data.bins_differ_between_parts", check_message=True)
def _():
    parts = _parts_2nd([[1, 1, 1]], [[2, 2, 2]]) + _parts_2nd(
        [[1, 1, 1]], [[2, 2, 2]], bins=(0, 100, 200, 400)
    )
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 2)
        return M._merge_data(
            tmp, "m", "data", names
        )  # the assertion message names the key and part, not the path


@case("modelling.merge_data.count_shapes_differ_between_parts", check_message=True)
def _():
    parts = [
        {
            "p_conn_dist": np.zeros(3),
            "count_conn": np.array([1, 1, 1]),
            "count_all": np.array([2, 2, 2]),
            "dist_bins": np.array([0, 1, 2, 3]),
        },
        {
            "p_conn_dist": np.zeros(3),
            "count_conn": np.array([1, 1]),
            "count_all": np.array([2, 2]),
            "dist_bins": np.array([0, 1, 2, 3]),
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 2)
        return M._merge_data(tmp, "m", "data", names)


@case("modelling.merge_data.first_part_lacks_count_key")
def _():
    parts = [
        {
            "p_conn_dist": np.zeros(2),
            "count_all": np.array([2, 2]),
            "dist_bins": np.array([0, 1, 2]),
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", parts, 1)
        return M._merge_data(tmp, "m", "data", names)


@case("modelling.merge_data.later_part_lacks_count_key", check_message=True)
def _():
    good = {
        "p_conn_dist": np.zeros(2),
        "count_conn": np.array([1, 1]),
        "count_all": np.array([2, 2]),
        "dist_bins": np.array([0, 1, 2]),
    }
    bad = {k: v for k, v in good.items() if k != "count_all"}
    with tempfile.TemporaryDirectory() as tmp:
        names = _write_parts(tmp, "m", "data", [good, bad], 2)
        return M._merge_data(tmp, "m", "data", names)
