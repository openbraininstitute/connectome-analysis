"""random_geometric_model.

The picks draw from the legacy global ``np.random`` state (one ``np.random.rand``/``np.random.choice`` call per source node,
in the order of the source index), so every case seeds it right before the call.  The result holds the relative
weights that were used for the picks as matrix values (all 1.0 without any weighting).  Unless stated, the points are
30 random points in a cube of side 100.
"""

import numpy as np
from support.cases import case
from support.data import positions, tutorial_connectome

from cases.randomization_common import R, shape_info

PTS = positions(30, seed=1, scale=100.0)
PTS_X = positions(8, seed=2, scale=100.0)


def _rgm(pts=None, seed=11, **kw):
    np.random.seed(seed)
    return R.random_geometric_model(PTS if pts is None else pts, **kw)


def _stream_after(pts=None, seed=11, **kw):
    np.random.seed(seed)
    R.random_geometric_model(PTS if pts is None else pts, **kw)
    return np.random.rand(3)


# ----------------------------------------------------------------------------- candidates and picks


@case("randomization.random_geometric_model.n_neighbors_and_n_pick")
def _():
    return _rgm(n_neighbors=6, n_pick=3)


@case("randomization.random_geometric_model.n_neighbors_and_p_pick")
def _():
    return _rgm(n_neighbors=6, p_pick=0.5)


@case("randomization.random_geometric_model.dist_neighbors_and_n_pick")
def _():
    return _rgm(dist_neighbors=40.0, n_pick=3)


@case("randomization.random_geometric_model.dist_neighbors_and_p_pick")
def _():
    return _rgm(dist_neighbors=40.0, p_pick=0.5)


@case("randomization.random_geometric_model.seeds_change_the_picks")
def _():
    return {s: _rgm(seed=s, n_neighbors=6, n_pick=3) for s in (0, 1, 2)}


@case("randomization.random_geometric_model.p_pick_one_takes_every_candidate")
def _():
    return {
        "n_neighbors": _rgm(n_neighbors=5, p_pick=1.0),
        "dist_neighbors": _rgm(dist_neighbors=35.0, p_pick=1.0),
    }


@case("randomization.random_geometric_model.all_candidates_when_n_pick_is_larger")
def _():
    # n_pick > number of candidates: all candidates, no warning
    return _rgm(n_neighbors=3, n_pick=10)


@case("randomization.random_geometric_model.n_neighbors_covering_all_other_points")
def _():
    return _rgm(n_neighbors=29, n_pick=2)


@case("randomization.random_geometric_model.n_neighbors_one")
def _():
    return {"n_pick": _rgm(n_neighbors=1, n_pick=1), "p_pick": _rgm(n_neighbors=1, p_pick=0.5)}


@case("randomization.random_geometric_model.very_large_distance_takes_all_points_as_candidates")
def _():
    return {
        "n_pick": _rgm(dist_neighbors=1e6, n_pick=3),
        "p_pick": _rgm(dist_neighbors=1e6, p_pick=0.05),
    }


@case("randomization.random_geometric_model.stream_after_n_pick")
def _():
    return _stream_after(n_neighbors=6, n_pick=3)


@case("randomization.random_geometric_model.stream_after_p_pick")
def _():
    return _stream_after(n_neighbors=6, p_pick=0.5)


@case("randomization.random_geometric_model.stream_after_dist_neighbors")
def _():
    return _stream_after(dist_neighbors=40.0, n_pick=3)


@case("randomization.random_geometric_model.stream_after_empty_result")
def _():
    # groups that pick nothing still draw (p_pick) or not (n_pick = 0 draws a choice of size 0)
    return {
        "p_pick_zero": _stream_after(n_neighbors=6, p_pick=0.0),
        "n_pick_zero": _stream_after(n_neighbors=6, n_pick=0),
    }


# ----------------------------------------------------------------------------- loops and symmetric output


@case("randomization.random_geometric_model.no_diag_false")
def _():
    # the nearest neighbour of every point is the point itself: loops can be picked
    return {
        "n_neighbors": _rgm(n_neighbors=5, p_pick=1.0, no_diag=False),
        "dist_neighbors": _rgm(dist_neighbors=30.0, p_pick=1.0, no_diag=False),
        "n_pick": _rgm(n_neighbors=5, n_pick=2, no_diag=False),
    }


@case("randomization.random_geometric_model.mirror_adds_the_reverse_edges_and_sums_weights")
def _():
    # docstring: a reverse edge is added "if it does not already exist"; the code sums, so pairs picked both ways get 2
    return {
        "n_pick": _rgm(n_neighbors=5, n_pick=2, mirror=True),
        "dist_neighbors": _rgm(dist_neighbors=40.0, p_pick=0.4, mirror=True),
    }


@case("randomization.random_geometric_model.mirror_with_loops_doubles_them")
def _():
    return _rgm(n_neighbors=4, p_pick=1.0, mirror=True, no_diag=False)


@case("randomization.random_geometric_model.mirror_accepts_truthy_values")
def _():
    return {
        "one": _rgm(n_neighbors=4, n_pick=2, mirror=1),
        "zero": _rgm(n_neighbors=4, n_pick=2, mirror=0),
        "false": _rgm(n_neighbors=4, n_pick=2, mirror=False),
    }


# ----------------------------------------------------------------------------- second set of points


@case("randomization.random_geometric_model.pts_x_n_neighbors")
def _():
    # rows are the points of pts_x (the sources), columns the points of pts; the nearest candidate is skipped (no_diag)
    return _rgm(pts_x=PTS_X, n_neighbors=5, n_pick=2)


@case("randomization.random_geometric_model.pts_x_dist_neighbors")
def _():
    return _rgm(pts_x=PTS_X, dist_neighbors=40.0, p_pick=0.5)


@case("randomization.random_geometric_model.pts_x_no_diag_false")
def _():
    return {
        "n_neighbors": _rgm(pts_x=PTS_X, n_neighbors=5, n_pick=2, no_diag=False),
        "dist_neighbors": _rgm(pts_x=PTS_X, dist_neighbors=50.0, p_pick=0.5, no_diag=False),
    }


@case("randomization.random_geometric_model.pts_x_ignores_mirror")
def _():
    out = _rgm(pts_x=PTS_X, n_neighbors=5, n_pick=2, mirror=True)
    return {
        "result": out,
        "same_as_without": bool((out != _rgm(pts_x=PTS_X, n_neighbors=5, n_pick=2)).nnz == 0),
    }


@case("randomization.random_geometric_model.pts_x_with_scale_axes_and_direction")
def _():
    return {
        "scaled": _rgm(pts_x=PTS_X, n_neighbors=5, n_pick=2, scale_axes=np.array([1.0, 2.0, 3.0])),
        "directional": _rgm(
            pts_x=PTS_X,
            n_neighbors=5,
            n_pick=2,
            directionality_axis=[0, 0, 1],
            directionality_fac=0.7,
        ),
    }


@case("randomization.random_geometric_model.pts_x_same_points")
def _():
    # pts_x equal to pts behaves like the default, except that mirror is switched off
    return {
        "explicit": _rgm(pts_x=PTS, n_neighbors=5, n_pick=2),
        "default": _rgm(n_neighbors=5, n_pick=2),
    }


@case("randomization.random_geometric_model.pts_x_with_custom_weights")
def _():
    return _rgm(
        pts_x=PTS_X,
        n_neighbors=5,
        n_pick=2,
        custom_w_in=np.linspace(0.5, 2.0, 30),
        custom_w_out=np.linspace(2.0, 0.5, 8),
    )


@case("randomization.random_geometric_model.pts_x_dist_neighbors_without_candidates")
def _():
    # a point of pts_x without any point of pts in range: setdiff1d of an empty list gives a float array, which cannot index
    return _rgm(pts_x=PTS_X, dist_neighbors=1e-6, p_pick=0.5)


# ----------------------------------------------------------------------------- weighting


@case("randomization.random_geometric_model.scale_axes")
def _():
    return {
        "list": _rgm(n_neighbors=5, n_pick=2, scale_axes=[1.0, 1.0, 10.0]),
        "array": _rgm(n_neighbors=5, n_pick=2, scale_axes=np.array([1.0, 2.0, 3.0])),
        "dist_neighbors": _rgm(dist_neighbors=2.0, p_pick=0.5, scale_axes=[10.0, 10.0, 10.0]),
    }


@case("randomization.random_geometric_model.directionality")
def _():
    # the weights (1 + fac * cos) are the values of the matrix; the cosine uses (source - target)
    return {
        "positive_x": _rgm(
            n_neighbors=8, n_pick=3, directionality_axis=[1.0, 0, 0], directionality_fac=0.9
        ),
        "negative_x": _rgm(
            n_neighbors=8, n_pick=3, directionality_axis=[1.0, 0, 0], directionality_fac=-0.9
        ),
        "y_p_pick": _rgm(
            n_neighbors=8, p_pick=0.4, directionality_axis=[0, 1.0, 0], directionality_fac=0.5
        ),
        "diagonal_axis": _rgm(
            n_neighbors=8,
            n_pick=3,
            directionality_axis=np.array([0.0, 0.6, 0.8]),
            directionality_fac=1.0,
        ),
        "column_axis": _rgm(
            n_neighbors=8,
            n_pick=3,
            directionality_axis=np.array([[0.0], [1.0], [0.0]]),
            directionality_fac=0.5,
        ),
        "dist_neighbors": _rgm(
            dist_neighbors=40.0, n_pick=3, directionality_axis=[0, 0, 1.0], directionality_fac=0.3
        ),
    }


@case("randomization.random_geometric_model.directionality_factor_boundaries")
def _():
    return {
        "plus_one": _rgm(
            n_neighbors=8, p_pick=0.5, directionality_axis=[1.0, 0, 0], directionality_fac=1.0
        ),
        "minus_one": _rgm(
            n_neighbors=8, p_pick=0.5, directionality_axis=[1.0, 0, 0], directionality_fac=-1.0
        ),
        "zero": _rgm(
            n_neighbors=8, p_pick=0.5, directionality_axis=[1.0, 0, 0], directionality_fac=0.0
        ),
    }


@case("randomization.random_geometric_model.directionality_factor_without_axis_is_ignored")
def _():
    return {
        "with_factor": _rgm(n_neighbors=6, n_pick=3, directionality_fac=0.5),
        "without": _rgm(n_neighbors=6, n_pick=3),
    }


@case("randomization.random_geometric_model.directionality_axis_longer_than_one_with_p_pick")
def _():
    # |axis| = 2 makes some weights negative: they are simply never picked
    return _rgm(n_neighbors=8, p_pick=0.5, directionality_axis=[2.0, 0, 0], directionality_fac=1.0)


@case("randomization.random_geometric_model.directionality_axis_longer_than_one_with_n_pick")
def _():
    return _rgm(n_neighbors=8, n_pick=3, directionality_axis=[2.0, 0, 0], directionality_fac=1.0)


@case("randomization.random_geometric_model.distance_func")
def _():
    return {
        "decay_n_pick": _rgm(n_neighbors=8, n_pick=3, distance_func=lambda d: np.exp(-d / 20.0)),
        "inverse_p_pick": _rgm(n_neighbors=8, p_pick=0.5, distance_func=lambda d: 1.0 / (1.0 + d)),
        "growing": _rgm(dist_neighbors=40.0, n_pick=3, distance_func=lambda d: d),
    }


@case("randomization.random_geometric_model.distance_func_with_zero_weights_and_p_pick")
def _():
    # mean weight 0: p_pick * 0 / 0 is nan, nothing is picked and the empty matrix comes back
    return _rgm(n_neighbors=8, p_pick=0.5, distance_func=lambda d: 0.0)


@case("randomization.random_geometric_model.distance_func_with_zero_weights_and_n_pick")
def _():
    return _rgm(n_neighbors=8, n_pick=3, distance_func=lambda d: 0.0)


@case("randomization.random_geometric_model.custom_weights")
def _():
    w_in, w_out = np.linspace(0.5, 2.0, 30), np.linspace(2.0, 0.5, 30)
    return {
        "both": _rgm(n_neighbors=8, n_pick=3, custom_w_in=w_in, custom_w_out=w_out),
        "lists": _rgm(
            n_neighbors=8, n_pick=3, custom_w_in=[1.0] * 29 + [3.0], custom_w_out=[1.0] * 30
        ),
        "with_p_pick": _rgm(n_neighbors=8, p_pick=0.4, custom_w_in=w_in, custom_w_out=w_out),
        "with_zero_weights": _rgm(
            n_neighbors=8,
            n_pick=3,
            custom_w_in=np.r_[np.zeros(10), np.ones(20)],
            custom_w_out=np.ones(30),
        ),
    }


@case("randomization.random_geometric_model.custom_weights_need_both_arrays")
def _():
    # with only one of custom_w_in / custom_w_out the weights are silently ignored
    w = np.linspace(0.5, 2.0, 30)
    return {
        "only_in": _rgm(n_neighbors=8, n_pick=3, custom_w_in=w),
        "only_out": _rgm(n_neighbors=8, n_pick=3, custom_w_out=w),
        "none": _rgm(n_neighbors=8, n_pick=3),
    }


@case("randomization.random_geometric_model.custom_weights_combined_with_other_weights")
def _():
    w_in, w_out = np.linspace(0.5, 2.0, 30), np.linspace(2.0, 0.5, 30)
    return {
        "distance_func": _rgm(
            n_neighbors=8,
            n_pick=3,
            custom_w_in=w_in,
            custom_w_out=w_out,
            distance_func=lambda d: np.exp(-d / 30.0),
        ),
        "direction": _rgm(
            n_neighbors=8,
            n_pick=3,
            custom_w_in=w_in,
            custom_w_out=w_out,
            directionality_axis=[1, 0, 0],
            directionality_fac=0.5,
        ),
    }


# ----------------------------------------------------------------------------- inputs


@case("randomization.random_geometric_model.more_than_three_dimensions")
def _():
    # candidates use all 4 dimensions, the weights only the first three
    p4 = positions(30, seed=1, dims=4, scale=100.0)
    return {
        "plain": _rgm(p4, n_neighbors=5, n_pick=2),
        "weighted": _rgm(p4, n_neighbors=5, n_pick=2, distance_func=lambda d: np.exp(-d / 20.0)),
    }


@case("randomization.random_geometric_model.two_dimensions_fail_in_the_weights")
def _():
    return _rgm(positions(30, seed=1, dims=2, scale=100.0), n_neighbors=5, n_pick=2)


@case("randomization.random_geometric_model.integer_points")
def _():
    return _rgm(np.round(PTS).astype(int), n_neighbors=5, n_pick=2)


@case("randomization.random_geometric_model.points_as_list_rejected")
def _():
    return _rgm(PTS.tolist(), n_neighbors=5, n_pick=2)


@case("randomization.random_geometric_model.two_points")
def _():
    two = positions(2, seed=1, scale=100.0)
    return {
        "n_pick": _rgm(two, n_neighbors=1, n_pick=1),
        "p_pick": _rgm(two, n_neighbors=1, p_pick=1.0),
    }


@case("randomization.random_geometric_model.duplicate_points")
def _():
    # distance 0 between the copies: fine without a directionality bias
    return _rgm(np.vstack([PTS[:5], PTS[:5]]), n_neighbors=2, n_pick=1)


@case("randomization.random_geometric_model.duplicate_points_with_directionality")
def _():
    # cos = 0 / 0 for the copies: nan weights break the probabilities of np.random.choice
    return _rgm(
        np.vstack([PTS[:5], PTS[:5]]),
        n_neighbors=2,
        n_pick=1,
        directionality_axis=[1.0, 0, 0],
        directionality_fac=0.5,
    )


@case("randomization.random_geometric_model.loops_with_directionality_and_n_pick")
def _():
    return _rgm(
        n_neighbors=3,
        n_pick=1,
        directionality_axis=[1.0, 0, 0],
        directionality_fac=0.5,
        no_diag=False,
    )


@case("randomization.random_geometric_model.loops_with_directionality_and_p_pick")
def _():
    # the nan weight of the loop is never picked
    return _rgm(
        n_neighbors=3,
        p_pick=0.5,
        directionality_axis=[1.0, 0, 0],
        directionality_fac=0.5,
        no_diag=False,
    )


# ----------------------------------------------------------------------------- the empty-matrix fallback


@case("randomization.random_geometric_model.empty_when_p_pick_is_zero")
def _():
    out = _rgm(n_neighbors=5, p_pick=0.0)
    return {"result": out, **shape_info(out)}


@case("randomization.random_geometric_model.empty_when_n_pick_is_zero")
def _():
    return _rgm(n_neighbors=5, n_pick=0)


@case("randomization.random_geometric_model.empty_when_nobody_is_in_range")
def _():
    return {
        "no_diag": _rgm(dist_neighbors=1e-6, p_pick=0.5),
        "zero_distance": _rgm(dist_neighbors=0.0, p_pick=1.0),
    }


@case("randomization.random_geometric_model.only_loops_in_range")
def _():
    # every point is its own only candidate: the result is the identity (as a float matrix)
    return {
        "loops": _rgm(dist_neighbors=1e-6, p_pick=1.0, no_diag=False),
        "no_loops_allowed": _rgm(dist_neighbors=1e-6, p_pick=1.0),
    }


@case("randomization.random_geometric_model.empty_fallback_ignores_pts_x")
def _():
    # the fallback has shape (len(pts), len(pts)) even when pts_x has another length
    return _rgm(pts_x=PTS_X, n_neighbors=5, p_pick=0.0)


# ----------------------------------------------------------------------------- invalid arguments


@case("randomization.random_geometric_model.both_n_neighbors_and_dist_neighbors")
def _():
    return _rgm(n_neighbors=3, dist_neighbors=3.0, n_pick=1)


@case("randomization.random_geometric_model.neither_n_neighbors_nor_dist_neighbors")
def _():
    return _rgm(n_pick=1)


@case("randomization.random_geometric_model.both_n_pick_and_p_pick")
def _():
    return _rgm(n_neighbors=3, n_pick=1, p_pick=0.5)


@case("randomization.random_geometric_model.neither_n_pick_nor_p_pick")
def _():
    return _rgm(n_neighbors=3)


@case("randomization.random_geometric_model.p_pick_below_zero")
def _():
    return _rgm(n_neighbors=3, p_pick=-0.1)


@case("randomization.random_geometric_model.p_pick_above_one")
def _():
    return _rgm(n_neighbors=3, p_pick=1.1)


@case("randomization.random_geometric_model.directionality_fac_above_one")
def _():
    return _rgm(n_neighbors=3, n_pick=1, directionality_fac=1.5)


@case("randomization.random_geometric_model.directionality_fac_below_minus_one")
def _():
    return _rgm(n_neighbors=3, n_pick=1, directionality_fac=-1.5)


@case("randomization.random_geometric_model.directionality_axis_and_distance_func")
def _():
    return _rgm(n_neighbors=3, n_pick=1, directionality_axis=[1, 0, 0], distance_func=lambda d: 1.0)


@case("randomization.random_geometric_model.more_neighbors_than_points")
def _():
    return _rgm(n_neighbors=30, n_pick=2)


@case("randomization.random_geometric_model.single_point")
def _():
    return _rgm(positions(1, seed=1), n_neighbors=1, n_pick=1)


# ----------------------------------------------------------------------------- real data


@case("randomization.random_geometric_model.real_data_n_neighbors")
def _():
    _, nodes = tutorial_connectome(300)
    return _rgm(nodes[["x", "y", "z"]].to_numpy(), n_neighbors=10, n_pick=3)


@case("randomization.random_geometric_model.real_data_dist_neighbors_and_scaled_depth")
def _():
    _, nodes = tutorial_connectome(300)
    xyz = nodes[["x", "y", "z"]].to_numpy()
    return {
        "dist": _rgm(xyz, dist_neighbors=80.0, p_pick=0.2),
        "scaled_mirrored": _rgm(
            xyz, n_neighbors=10, p_pick=0.3, scale_axes=[1.0, 0.25, 1.0], mirror=True
        ),
    }


@case("randomization.random_geometric_model.real_data_directional")
def _():
    _, nodes = tutorial_connectome(300)
    return _rgm(
        nodes[["x", "y", "z"]].to_numpy(),
        n_neighbors=10,
        n_pick=3,
        directionality_axis=[0, -1.0, 0],
        directionality_fac=0.8,
    )
