"""The exponential fits inside the model building, called directly on prepared connection probabilities.

``_build_2nd_order`` fits ``a * exp(-b * d)`` to the finite probabilities of the distance bins (bin centres as ``d``);
``_build_3rd_order`` fits the two columns (depth difference negative / non-negative) separately, on the bins where
*both* columns are finite. Both start ``scipy.optimize.curve_fit`` from ``p0=[0, 0]`` and answer NaN parameters (and an
error log record) when the fit raises. These are private functions: if the rewrite renames them, this file is the one to
adapt, the same fits are reached through ``run_model_building`` in ``modelling_run.py``.

Fitted values are compared with ``rtol=1e-6``.
"""

import numpy as np
from support.cases import case
from support.data import rng

from connalysis.modelling import modelling as M

from .modelling_common import logged

FIT = {"rtol": 1e-6}
BINS = np.arange(0, 600, 100)  # six bins, centres 50 .. 550
CENTRES = BINS[:-1] + 50.0


def law(a, b, centres=CENTRES):
    return a * np.exp(-b * centres)


def fit2(p, bins=BINS):
    with logged() as records:
        model = M._build_2nd_order(np.asarray(p, dtype=float), bins)
    return model, [(level, text.split("(")[0]) for level, text in records]


def fit3(p, bins=BINS):
    with logged() as records:
        model = M._build_3rd_order(np.asarray(p, dtype=float), bins)
    return model, [(level, text.split("(")[0]) for level, text in records]


# ----------------------------------------------------------------------------- second order


@case("modelling.build_2nd_order.exact_exponential_data", **FIT)
def _():
    return fit2(law(0.4, 0.01))


@case("modelling.build_2nd_order.exact_exponential_data_with_fine_bins", **FIT)
def _():
    bins = np.arange(0, 1001, 25)
    return fit2(law(0.25, 0.004, bins[:-1] + 12.5), bins)


@case("modelling.build_2nd_order.noisy_data", **FIT)
def _():
    noise = 1.0 + 0.2 * (rng(1).random_sample(CENTRES.size) - 0.5)
    return fit2(law(0.3, 0.006) * noise)


@case("modelling.build_2nd_order.bins_without_data_are_skipped", **FIT)
def _():
    p = law(0.4, 0.01)
    p[[1, 4]] = np.nan
    return fit2(p)


@case("modelling.build_2nd_order.infinite_values_are_skipped", **FIT)
def _():
    p = law(0.4, 0.01)
    p[2] = np.inf
    p[3] = -np.inf
    return fit2(p)


@case("modelling.build_2nd_order.constant_probability", **FIT)
def _():
    return fit2(np.full(CENTRES.size, 0.3))


@case("modelling.build_2nd_order.all_zero_probabilities", **FIT)
def _():
    return fit2(np.zeros(CENTRES.size))


@case("modelling.build_2nd_order.all_ones", **FIT)
def _():
    return fit2(np.ones(CENTRES.size))


@case("modelling.build_2nd_order.probability_increasing_with_distance", **FIT)
def _():
    return fit2(law(0.05, -0.003))


@case("modelling.build_2nd_order.probabilities_above_one_as_from_weights", **FIT)
def _():
    return fit2(law(3.0, 0.004))


@case("modelling.build_2nd_order.negative_probabilities_as_from_negative_weights", **FIT)
def _():
    return fit2(-law(0.4, 0.01))


@case("modelling.build_2nd_order.last_bins_without_pairs", **FIT)
def _():
    p = law(0.4, 0.01)
    p[-2:] = np.nan
    return fit2(p)


@case("modelling.build_2nd_order.two_points", **FIT)
def _():
    p = np.full(CENTRES.size, np.nan)
    p[[0, 3]] = [0.3, 0.1]
    return fit2(p)


@case("modelling.build_2nd_order.single_point_fails_and_logs_an_error", **FIT)
def _():
    p = np.full(CENTRES.size, np.nan)
    p[2] = 0.2
    return fit2(p)


@case("modelling.build_2nd_order.no_finite_point_fails_and_logs_an_error", **FIT)
def _():
    return fit2(np.full(CENTRES.size, np.nan))


@case("modelling.build_2nd_order.one_bin_fails_and_logs_an_error", **FIT)
def _():
    return fit2([0.2], np.array([0, 100]))


@case("modelling.build_2nd_order.bin_width_sets_the_distances_of_the_points", **FIT)
def _():
    bins = np.arange(0, 301, 75)
    p = law(0.4, 0.01, bins[:-1] + 37.5)
    return fit2(p, bins), fit2(p, np.arange(0, 201, 50))


@case("modelling.build_2nd_order.fractional_bins", **FIT)
def _():
    bins = np.arange(0, 7) * 37.5
    return fit2(law(0.35, 0.008, bins[:-1] + 18.75), bins)


@case("modelling.build_2nd_order.model_dictionary_structure")
def _():
    model, _ = fit2(law(0.4, 0.01))
    return (
        list(model.keys()),
        model["model"],
        model["model_inputs"],
        list(model["model_params"].keys()),
        [type(v).__name__ for v in model["model_params"].values()],
    )


# ----------------------------------------------------------------------------- third order


def sides(a_neg, b_neg, a_pos, b_pos):
    return np.column_stack([law(a_neg, b_neg), law(a_pos, b_pos)])


@case("modelling.build_3rd_order.exact_exponential_data_on_both_sides", **FIT)
def _():
    return fit3(sides(0.3, 0.012, 0.5, 0.008))


@case("modelling.build_3rd_order.noisy_data_on_both_sides", **FIT)
def _():
    noise = 1.0 + 0.2 * (rng(2).random_sample((CENTRES.size, 2)) - 0.5)
    return fit3(sides(0.3, 0.012, 0.5, 0.008) * noise)


@case("modelling.build_3rd_order.a_bin_empty_on_one_side_is_dropped_on_both", **FIT)
def _():
    p = sides(0.3, 0.012, 0.5, 0.008)
    p[3, 0] = np.nan
    return fit3(p)


@case("modelling.build_3rd_order.infinite_value_drops_the_bin_on_both_sides", **FIT)
def _():
    p = sides(0.3, 0.012, 0.5, 0.008)
    p[1, 1] = np.inf
    return fit3(p)


@case("modelling.build_3rd_order.bins_without_pairs_on_both_sides", **FIT)
def _():
    p = sides(0.3, 0.012, 0.5, 0.008)
    p[-2:, :] = np.nan
    return fit3(p)


@case("modelling.build_3rd_order.one_side_without_any_data_fails_both", **FIT)
def _():
    p = sides(0.3, 0.012, 0.5, 0.008)
    p[:, 0] = np.nan
    return fit3(p)


@case("modelling.build_3rd_order.sides_with_different_shapes", **FIT)
def _():
    return fit3(sides(0.1, 0.0, 0.5, 0.01))


@case("modelling.build_3rd_order.zeros_on_one_side", **FIT)
def _():
    p = sides(0.3, 0.012, 0.5, 0.008)
    p[:, 0] = 0.0
    return fit3(p)


@case("modelling.build_3rd_order.two_bins", **FIT)
def _():
    return fit3(sides(0.3, 0.012, 0.5, 0.008)[:2], np.array([0, 100, 200]))


@case("modelling.build_3rd_order.one_bin_fails_and_logs_an_error", **FIT)
def _():
    return fit3([[0.2, 0.3]], np.array([0, 100]))


@case("modelling.build_3rd_order.negative_exponents", **FIT)
def _():
    return fit3(sides(0.05, -0.003, 0.04, -0.002))


@case("modelling.build_3rd_order.model_dictionary_structure")
def _():
    model, _ = fit3(sides(0.3, 0.012, 0.5, 0.008))
    return (
        list(model.keys()),
        model["model"],
        model["model_inputs"],
        list(model["model_params"].keys()),
        [type(v).__name__ for v in model["model_params"].values()],
    )


@case(
    "modelling.build_3rd_order.fitted_model_function_follows_the_sign_of_the_depth_difference",
    **FIT,
)
def _():
    model, _ = fit3(sides(0.3, 0.012, 0.5, 0.008))
    f = M._get_model_function(**model)
    d = np.array([0.0, 100.0, 300.0])
    return f(d, -1), f(d, 1), f(d, 0)
