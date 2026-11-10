# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Functions to average (functional or structural) metrics across simplices or neighborhoods.
Author(s): Daniela Egas Santander,
Last update: 11.2023
"""

import operator

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .local import neighborhood_indices


def _take(values, indices):
    """``values[indices]`` for integer ``indices``: by label for an index of numbers, by position for any other
    (pandas 3 no longer falls back to positions as pandas 2 did, with a warning)."""
    if isinstance(values, pd.Series) and values.index.inferred_type not in (
        "integer", "mixed-integer", "floating", "complex",
    ):  # fmt: skip
        return values.iloc[indices]
    return values[indices]


def node_stats_per_position_single(simplex_list, values, with_multiplicity=True):
    """Get mean, standard deviation and standard error of the mean averaged across simplex lists and filtered per position
    Parameters
    ----------
    simplex_list : 2d-array
        Array of dimension (no. of simplices, dimension) listing simplices to be considered.
        Each row corresponds to a list of nodes on a simplex indexed by the order of the nodes in an NxN matrix.
        All entries must be an index in values
    values : Series
        pandas Series with index the nodes of the NxN matrix of which the simplices are listed,
        and values the values on that node to be averaged.
    with_multiplicity : bool
        if ``True`` the values are averaged with multiplicity i.e., they are weighted by the number of times a node
        participates in a simplex in a given position
        if ``False`` repetitions of a node in a given position are ignored.

    Returns
    -------
    DataFrame
        with index, the possible positions of o node in a ``k``-simplex and columns the mean, standard deviation and
        standard error of the mean for that position
    """
    # Filter values
    if with_multiplicity:
        vals_sl = values.loc[simplex_list.flatten()].to_numpy().reshape(simplex_list.shape)
    else:
        vals_sl = pd.concat(
            [values.loc[np.unique(simplex_list[:, pos])] for pos in range(simplex_list.shape[1])],
            axis=1,
            keys=range(simplex_list.shape[1]),
        )
    # Compute stats
    stats_vals = pd.DataFrame(index=pd.Index(range(simplex_list.shape[1]), name="position"))
    # Stats per position
    stats_vals["mean"] = np.nanmean(vals_sl, axis=0)
    stats_vals["std"] = np.nanstd(vals_sl, axis=0)
    stats_vals["sem"] = stats.sem(vals_sl, axis=0, nan_policy="omit")
    # Stats in any position
    stats_vals.loc["all", "mean"] = np.nanmean(vals_sl)
    stats_vals.loc["all", "std"] = np.nanstd(vals_sl)
    stats_vals.loc["all", "sem"] = stats.sem(vals_sl, axis=None, nan_policy="omit")
    return stats_vals


def node_stats_per_position(simplex_lists, values, dims=None, with_multiplicity=True):
    """Get across dimensions mean, standard deviation and standard error of the mean averaged across simplex lists
    and filtered per position
    Parameters
    ----------
    simplex_lists : dict
        keys : are int values representing dimensions
        values : for key ``k`` array of dimension (no. of simplices, ``k``) listing simplices to be considered.
        Each row corresponds to a list of nodes on a simplex indexed by the order of the nodes in an NxN matrix.
        All entries must be an index in values
    values : Series
        pandas Series with index the nodes of the NxN matrix of which the simplices are listed,
        and values the values on that node to be averaged.
    with_multiplicity : bool
        if ``True`` the values are averaged with multiplicity i.e., they are weighted by the number of times a node
        participates in a simplex in a given position
        if ``False`` repetitions of a node in a given position are ignored.
    dims : iterable
        dimensions for which to run the analysis, if ``None`` all the keys of simplex lists will be analyzed

    Returns
    -------
    dict
        keys the dimensions anlayzed and values for key ``k`` a DataFrame
        with index, the possible positions of o node in a ``k``-simplex and columns the mean, standard deviation and
        standard error of the mean for that position.
    """
    if dims is None:
        dims = simplex_lists.index
    stats_dict = {}
    for dim in tqdm(dims):
        sl = simplex_lists.loc[dim]
        stats_dict[dim] = node_stats_per_position_single(
            sl, values, with_multiplicity=with_multiplicity
        )
    return stats_dict


def node_stats_participation(participation, vals, condition=operator.eq, dims=None):
    """Get statistics of the values in vals across nodes filtered using node participation
    Parameters
    ----------
    participation : DataFrame
        DataFrame of node participation with index the nodes in nodes of an NxN matrix to consider,
        columns are dimensions and values are node participation computed with
        :func:`connalysis.network.topology.node_participation`.
    values : Series
        pandas Series with index the nodes of the NxN matrix of where node participation has been computed
        and vals the values on that node to be averaged.
    condition : operator
        operator with which to filter the nodes. The default ``operator.eq`` filters nodes such that their maximal
        dimension of node participation is a given value.
        Alternatively, ``operator.ge`` filters nodes such that their maximal dimension of node participation is at least a given
        value.
    dims : iterable
        dimensions for which to run the analysis, if ``None`` all the columns of participation will be analyzed

    Returns
    -------
    DataFrame
        with index, the dimensions for which the analysis have been run and columns the statistics of the values in vals
        where the nodes have been grouped according to the condition given.

    See Also
    --------
    :func:`connalysis.network.stats.node_stats_per_position_single`:
    A similar function where the position of the nodes in the simplex are taken into account.  Note in particular that
    if condition = ``operator.ge`` the weighted_mean of this analyisis is equivalent than the value given by this function for position ``all``.
    However the computation using
    :func:`connalysis.network.topology.node_participation`
    is more efficient.
    """
    par_df = participation.copy()
    if dims is None:
        dims = par_df.columns
    vals = vals.rename("values")
    par_df["max_dim"] = (par_df > 0).sum(axis=1) - 1  # maximal dimension a node is part of
    stats_vals = {}
    for dim in dims:
        mask = condition(par_df.max_dim, dim)
        c = pd.DataFrame(vals.loc[par_df[mask].index], columns=["values"])
        c["weight"] = par_df[mask][dim]
        w_mean = (c["values"].fillna(1) * c["weight"]).sum() / c[
            "weight"
        ].sum()  # NaN values count as 1
        stats_vals[dim] = (
            c.shape[0],  # Number of nodes fulfilling the condition
            np.nanmean(c["values"]),
            np.nanstd(c["values"]),
            stats.sem(c["values"], nan_policy="omit"),
            w_mean,  # mean weighted by participation
        )
    stats_vals = pd.DataFrame.from_dict(
        stats_vals, orient="index", columns=["counts", "mean", "std", "sem", "weighted_mean"]
    )
    stats_vals.index.name = "dim"
    return stats_vals


def node_stats_neighborhood(
    values,
    adj=None,
    pre=True,
    post=True,
    all_nodes=True,
    centers=None,
    include_center=True,
    precomputed=False,
    neighborhoods=None,
):
    """Get basic statistics of the property values on the neighbhood of the nodes in centers in the
    graph described by adj.
    Parameters
    ----------
    values : Series
        pandas Series with index the nodes of the NxN matrix of which the simplices are listed,
        and values the values on that node to be averaged.
    adj : sparse matrix or 2d array
        The adjacency matrix of the graph
    pre : bool
        If ``True`` compute the nodes mapping to the nodes in centers (the in-neighbors of the centers)
    post : bool
        If ``True`` compute the nodes that the centers map to (the out-neighbors of the centers)
    all_nodes : bool
        If ``True`` compute the neighbors of all nodes in adj, if ``False`` compute only the neighbors of the nodes
        listed in centers
    centers : 1d-array
        The indices of the nodes for which the neighbors need to be computed.  This entry is ignored if
        all_nodes is ``True`` and required if all_nodes is ``False``
    include_center : bool
        If ``True`` it includes the center in the computation otherwise it ignores it
    precomputed : bool
        If ``False`` it precomputes the neighbhorhoods in adj,
        if ``False`` it skips the computation and reads it fromt the input
    neighborhoods : DataFrame
        DataFrame of neighbhoord indices. Required if precomputed is ``True``

    Returns
    -------
    DataFrame
        with index, centers to be considered and columns the sum, mean, standard deviation and
        standard error of the mean of the values in that neighborhood.

    See Also
    --------
    :func:`connalysis.network.local.neighborhood_indices`:
    Function to precompute the neighborhood_indices that can be used if precomputed is set ``True``.
    Precomputing the neighborhoods would increase efficiency if multiple properties are averaged across neighborhoods.
    """

    # Single value functions for DataFrames
    def append_center(x):
        # To include center in the computation
        return np.append(x["center"], x["neighbors"])

    def mean_nbd(nbd_indices, v):
        df = _take(v, nbd_indices)
        return [np.nansum(df), np.nanmean(df), np.nanstd(df), stats.sem(df, nan_policy="omit")]

    # Get neighborhoods
    if precomputed:
        assert isinstance(neighborhoods, pd.Series), (
            "If precomputed a Series of neighbhoords indexed by their center must be provided"
        )
    else:
        assert adj is not None, "If not precomputed and adjancecy matrix must be provided"
        neighborhoods = neighborhood_indices(
            adj, pre=pre, post=post, all_nodes=all_nodes, centers=centers
        )
    centers = neighborhoods.index
    if include_center:
        neighborhoods = neighborhoods.reset_index().apply(append_center, axis=1)
    else:
        neighborhoods = neighborhoods.reset_index(drop=True)
    stat_vals = pd.DataFrame.from_records(
        neighborhoods.map(lambda x: mean_nbd(x, values)), columns=["sum", "mean", "std", "sem"]
    )
    stat_vals["center"] = centers
    return stat_vals.set_index("center")


def edge_stats_participation(participation, vals, condition=operator.eq, dims=None):
    """Get statistics of the values in vals across edges filtered using edge participation

    Parameters
    ----------
    participation : DataFrame
        DataFrame of edge participation with index the edges of an NxN matrix to consider,
        columns are dimensions and values are edge participation.
    values : Series
        pandas Series with index the edges of the NxN matrix of which edge participation has been computed
        and vals the values on that edge to be averaged.
    condition : operator
        operator with which to filter the nodes. The default ``operator.eq`` filters nodes such that their maximal
        dimension of edge participation is a given value.
        Alternatively, ``operator.ge`` filters edges such that their maximal dimension of node participation is at least a given
        value.
    dims : iterable
        dimensions for which to run the analysis, if ``None`` all the columns of participation will be analyzed

    Returns
    -------
    DataFrame
        with index, the dimensions for which the analysis have been run and columns the statistics of the values in vals
        where the nodes have been grouped according to the condition given.
    """
    par_df = participation.copy()
    if dims is None:
        dims = par_df.columns
    vals = vals.rename("values")
    par_df["max_dim"] = (par_df > 0).sum(
        axis=1
    )  # maximal dimension an edge is part of. Note that edge participation in dimension 0 is 0
    stats_vals = {}
    for dim in dims:
        mask = condition(par_df.max_dim, dim)
        c = pd.DataFrame(vals.loc[par_df[mask].index], columns=["values"])
        c["weight"] = par_df[mask][dim]
        w_mean = (c["values"].fillna(1) * c["weight"]).sum() / c[
            "weight"
        ].sum()  # NaN values count as 1
        stats_vals[dim] = (
            c.shape[0],  # Number of nodes fulfilling the condition
            np.nanmean(c["values"]),
            np.nanstd(c["values"]),
            stats.sem(c["values"], nan_policy="omit"),
            w_mean,  # mean weighted by participation
        )
    stats_vals = pd.DataFrame.from_dict(
        stats_vals, orient="index", columns=["counts", "mean", "std", "sem", "weighted_mean"]
    )
    stats_vals.index.name = "dim"
    return stats_vals.drop(0)
