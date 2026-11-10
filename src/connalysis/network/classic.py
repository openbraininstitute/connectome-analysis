# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.spatial import KDTree
from scipy.stats import binom, hypergeom

from ._components import strong_components
from .topology import node_degree, rc_submatrix, underlying_undirected_matrix

LOG = logging.getLogger(__name__)


def _closeness(adjacency):
    """Closeness of every node of a connected graph: (n - 1) over the sum of its hop distances to the others.

    Runs breadth first searches from 64 * words sources at once: bit j of the word of a node says that the search
    from source j has just reached the node, so a level of all the searches is an OR over the in-neighbours.
    """
    graph = sp.csr_matrix(adjacency, dtype=bool)
    n = graph.shape[0]
    incoming = graph.T.tocsr()  # row v lists the nodes with an edge to v
    sources_in, starts = incoming.indices, incoming.indptr[:-1]
    no_incoming = np.diff(incoming.indptr) == 0
    words = max(1, min(8, 6_000_000 // max(len(sources_in), 1)))
    scores = np.empty(n)
    gathered = np.zeros(
        (len(sources_in) + 1, words), dtype=np.uint64
    )  # the extra zero row closes the last segment
    for first in range(0, n, 64 * words):
        sources = np.arange(first, min(first + 64 * words, n))
        within = np.arange(len(sources))
        frontier = np.zeros((n, words), dtype=np.uint64)
        frontier[sources, within // 64] = np.uint64(1) << (within % 64).astype(np.uint64)
        visited = frontier.copy()
        total = np.zeros(64 * words)  # sum of the distances from each source
        reached = np.ones(64 * words)  # nodes reached, the source included
        level = 0
        while True:
            np.take(frontier, sources_in, axis=0, out=gathered[:-1])
            new = np.bitwise_or.reduceat(gathered, starts, axis=0)
            new[no_incoming] = 0
            new &= ~visited
            fresh = np.flatnonzero(new.any(axis=1))
            if fresh.size == 0:
                break
            level += 1
            visited |= new
            bits = np.unpackbits(
                new[fresh].astype("<u8", copy=False).view(np.uint8), axis=1, bitorder="little"
            )
            counts = bits.sum(axis=0)
            total += level * counts
            reached += counts
            frontier = new
        total, reached = total[: len(sources)], reached[: len(sources)]
        with np.errstate(divide="ignore", invalid="ignore"):
            block_scores = (n - 1) / n / (total / n)
        block_scores[reached < n] = 0.0  # a node that cannot reach every other one
        scores[sources] = block_scores
    return scores


def closeness_connected_components(adj, directed=False, return_sum=True):
    """Compute the closeness of each connected component of more than 1 vertex

    Parameters
    ----------
    adj : array_like
        Adjacency matrix of the graph
    directed : bool
        If `True`, will be computed using strongly connected components and directed closeness.
    return_sum : bool
        If `True`, only one list will be returned, by summing over all the connected components.


    Returns
    -------
    array_like
        A single array( if `return_sum=True`) or a list of arrays of shape `n`, containting closeness of vertices in that component, or 0 if the vertex is not in the component. Closeness cannot be zero otherwise.

    """
    matrix = sp.csr_matrix(adj)
    if directed:
        n_comp, comp = strong_components(matrix)
    else:
        n_comp, comp = csgraph.connected_components(matrix, directed=False)
        matrix = matrix + matrix.T  # we need to make the matrix symmetric

    n = matrix.shape[0]
    by_component = np.argsort(
        comp, kind="stable"
    )  # the nodes of each component, in increasing order
    ends = np.cumsum(np.bincount(comp, minlength=n_comp))
    all_c = []  # one array per component, only kept when they are returned separately
    total = np.zeros(n)
    found = False
    for i in range(n_comp):
        idx = by_component[ends[i - 1] if i else 0 : ends[i]]
        sub_mat = matrix[np.ix_(idx, idx)]
        if sub_mat.getnnz() > 0:
            found = True
            sub_mat.eliminate_zeros()
            if return_sum:
                total[idx] = _closeness(sub_mat)
            else:
                c = np.zeros(n)
                c[idx] = _closeness(sub_mat)
                all_c.append(c)
    if return_sum:
        # With no component that has an edge, the sum over no arrays is the scalar 0.0
        return total if found else np.sum(np.array(all_c), axis=0)
    return all_c


def connected_components(adj, directed=True, connection="weak", return_labels=False):
    """Returns a list of the size of the connected components of the graph

    Parameters
    ----------
    adj : array_like or sparse matrix
        Adjacency matrix of the graph
    directed : bool
        If `True`, will be compute connected components of the directed graph
    connection : str {'weak', 'strong'}
        If `weak`, it will compute the connected components of the underlying undirected graph.
        If `strong`, it will compute strongly connected components of the directed graph.
    return_labels : bool
        If `True`, will return the labels of the connected components

    Returns
    -------
    array_like
        A list of the size of the connected components of the graph.
        If return_labels == True, it also returns the list of labels of the connected components.
    """

    matrix = sp.csr_matrix(adj)

    if directed and connection == "strong":
        comps = strong_components(
            matrix
        )  # numbered as before scipy 1.18, which numbers them differently
    else:
        comps = csgraph.connected_components(
            matrix, directed=directed, connection=connection, return_labels=True
        )
    comps_size = np.unique(comps[1], return_counts=True)[1]
    if return_labels:
        return comps_size, comps[1]
    else:
        return comps_size


def core_number(adj):
    """Returns the core number for each node.

    Parameters
    ----------
    adj : array_like or sparse matrix
        Adjacency matrix of the graph
    directed : bool
        If `True`, will be compute connected components of the directed graph
    connection : str {'weak', 'strong'}
        If `weak`, it will compute the connected components of the underlying undirected graph.
        If `strong`, it will compute strongly connected components of the directed graph.
    return_labels : bool
        If `True`, will return the labels of the connected components

    Returns
    -------
    dict
        A dictionary with keys the indices of the nodes of adj and values their corresponding core number.

    Notes
    -----
    The k-core of a graph is the maximal subgraph that contains nodes of degree k or more in the induced subgraph.
    The core number of a node is the largest value k of a k-core containing that node.
    For directed graphs the total node degree is use, i.e., the sum of in-degree + out-degree.
    """
    # TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
    # TODO: Filtered simplex counts with different weights on vertices (coreness, intersection)
    #  or on edges (strength of connection).
    adj = sp.csr_matrix(adj)
    n = adj.shape[0]
    if adj.shape[0] != adj.shape[1]:
        raise nx.NetworkXError("Adjacency matrix not square: nx,ny=%s" % (adj.shape,))
    coo = adj.tocoo()  # every stored entry is an edge, as when the graph was built with networkx
    if np.any(coo.row == coo.col):
        raise nx.NetworkXNotImplemented(
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
    # The simple undirected graph: each pair of nodes joined by one or two directed edges is one edge
    pairs = np.unique(
        np.minimum(coo.row, coo.col).astype(np.int64) * n + np.maximum(coo.row, coo.col)
    )
    low, high = np.divmod(pairs, n)
    source, target = np.concatenate([low, high]), np.concatenate([high, low])
    order = np.argsort(source, kind="stable")
    neighbours = target[order]
    degree = np.bincount(source, minlength=n)
    start = np.concatenate([[0], np.cumsum(degree)])

    # Remove, level by level, every node of degree <= k (and the nodes that this leaves with degree <= k)
    core = np.zeros(n, dtype=np.int64)
    remaining = degree.copy()
    alive = np.ones(n, dtype=bool)
    k = 0
    while alive.any():
        frontier = np.flatnonzero(alive & (remaining <= k))
        if frontier.size == 0:
            k = remaining[alive].min()
            continue
        while frontier.size:
            core[frontier] = k
            alive[frontier] = False
            counts = degree[frontier]
            first = np.repeat(start[frontier] - np.cumsum(counts) + counts, counts)
            around = neighbours[first + np.arange(counts.sum())]
            nodes, hits = np.unique(around[alive[around]], return_counts=True)
            remaining[nodes] -= hits
            frontier = nodes[remaining[nodes] <= k]
    return dict(enumerate(core.tolist()))


def density(adj, type="directed", skip_symmetry_check=False):
    """Returns the density of a matrix.

    Parameters
    ----------
    adj : array_like or sparse matrix
        Adjacency matrix of the graph
    type : str {'directed', 'undirected', 'reciprocal'}
        The type of the graph considered for the computation.
        If 'directed', the density as a directed graph is computed.
        If 'undirected', the density of the underlying undirected graph is computed.
        If 'reciprocal', the density of the underlying reciprocal graph is computed.
    skip_symmetry_check : bool
        If `True`, it will skip the check for symmetry of the matrix.

    Returns
    -------
    float
        The density of the graph.
    """
    assert adj.shape[0] == adj.shape[1], "The matrix must be square!"
    n = adj.shape[0]
    adj = sp.csr_matrix(adj).astype("bool")
    if type == "undirected":
        if not skip_symmetry_check:
            if (adj != adj.T).nnz > 0:
                LOG.warning("The graph is directed. Taking the underlying undirected graph")
                adj = underlying_undirected_matrix(adj)
    if type == "reciprocal":
        adj = rc_submatrix(adj)
    return adj.sum() / (n * (n - 1))


def get_pairs_within(m, v, cols=["x", "y"], max_dist=100):  # noqa: B006
    """Returns a matrix of the paris of nodes within distance `max_dist` of each other.

    Parameters
    ----------
    m : array or sparse matrix
        Adjacency matrix of the graph
    v : DataFrame or tuple
        DataFrame with the coordinates of the nodes if m is square and has the same source and target nodes.
        If a tuple is passed, it should contain two DataFrames, one for the source nodes and one for the target nodes.
    cols : list
        Columns of the DataFrame containing the coordinates of the nodes.
    max_dist : float
        Maximum distance between nodes to be considered connected.

    Returns
    -------
    sparse matrix
        Boolean matrix with 1 indicating the pairs of nodes within `max_dist` of each other,
        excluding the diagonal for square matrices.
    """
    if isinstance(v, tuple):
        vpre, vpost = v
    else:
        vpre = v
        vpost = v
    tree = KDTree(vpre[cols])
    pairs = tree.query_ball_point(vpost[cols], max_dist)
    indptr = np.cumsum([0] + [len(_x) for _x in pairs])
    if indptr[-1] == 0:
        return np.nan
    pairs_mat = sp.csc_matrix(
        (np.ones(indptr[-1], dtype=bool), np.hstack(pairs), indptr), shape=m.shape
    )
    if not isinstance(v, tuple):
        pairs_mat.setdiag(
            0
        )  # Don't consider edges from i to i when pre and post subpopulations are the same
    return pairs_mat.astype(bool)


def connection_probability_within(
    m,
    v,
    cols=["x", "y"],  # noqa: B006
    max_dist=100,
    type="directed",
    skip_symmetry_check=False,
):
    """Returns the average density of submatrices of nodes within distance `max_dist` of each node in `m`.

    Parameters
    ----------
    m : array or sparse matrix
        Adjacency matrix of the graph
    v : DataFrame or tuple
        DataFrame with the coordinates of the nodes if m is square and has the same pre and post nodes.
        If a tuple is passed, it should contain two DataFrames, one for the source nodes and one for the target nodes.
    cols : list
        Columns of the DataFrame containing the coordinates of the nodes.
    max_dist : float
        Maximum distance between nodes to be considered connected.
    type : str {'directed', 'undirected', 'reciprocal'}
        The type of the graph considered for the computation.
        If 'directed', the density as a directed graph is computed.
        If 'undirected', the density of the underlying undirected graph is computed. Only possible if the matrix is square.
        If 'reciprocal', the density of the underlying reciprocal graph is computed. Only possible if the matrix is square.
    skip_symmetry_check : bool
        If `True`, it will skip the check for symmetry of the matrix.

    Returns
    -------
    float
        The average density of the submatrices.
    """
    # Get coordinates of nodes and their nearest neighbors
    pairs_mat = get_pairs_within(m, v, cols=cols, max_dist=max_dist)
    if pairs_mat is np.nan:
        return np.nan
    else:
        if type == "reciprocal":
            assert m.shape[0] == m.shape[1], (
                "The matrix must be square to compute the reciprocal connectivity"
            )
            m = rc_submatrix(m).tocsc()
        elif type == "undirected":
            assert m.shape[0] == m.shape[1], (
                "The matrix must be square to compute the undirected connectivity"
            )
            if not skip_symmetry_check:
                if (m != m.T).nnz > 0:
                    LOG.warning("The graph is directed. Taking the underlying undirected graph")
                    m = underlying_undirected_matrix(m)
            m = sp.csc_matrix(m)
        elif type == "directed":
            m = sp.csc_matrix(m)
        else:
            raise ValueError(f"Type must be 'directed', 'undirected' or 'reciprocal', not {type!r}")
        connected = m[pairs_mat].astype(bool)
        if (
            np.prod(connected.shape) == 0
        ):  # no pair of neurons within reach: the probability is undefined
            return np.nan
        return connected.mean()


###### TODO: Code in this block needs docstrings ######


def __make_expected_distribution_model_first_order__(adj, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    if direction == "efferent":
        N = adj.sum(axis=1).mean()
        M = adj.shape[1]
    elif direction == "afferent":
        N = adj.sum(axis=0).mean()
        M = adj.shape[0]
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    expected = hypergeom(M, N, N)
    return expected


def distribution_number_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    adj = sp.csc_matrix(adj).astype(int)
    if direction == "efferent":
        cn = adj @ adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() @ adj
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    n_pairs = cn.shape[0] * (cn.shape[0] - 1) // 2
    if n_pairs == 0:
        raise ValueError("zero-size array to reduction operation maximum which has no identity")
    # Only the non-zero entries above the diagonal are stored; all other pairs have no common neighbour
    stored = sp.triu(cn, k=1).data
    stored = stored[stored != 0]
    counts = np.bincount(stored[stored > 0], minlength=int(stored.max(initial=0)) + 1)
    counts[0] += n_pairs - stored.size
    return pd.Series(counts, index=np.arange(len(counts)))


def normalized_distribution_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    expected = __make_expected_distribution_model_first_order__(adj, direction=direction)
    expected = expected.pmf(data.index) * data.sum()
    expected = pd.Series(expected, index=data.index)
    return (data - expected) / (data + expected)


def overexpression_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    data_mean = (data.index.values * data.values).sum() / data.values.sum()
    ctrl = __make_expected_distribution_model_first_order__(adj, direction=direction)
    ctrl_mean = ctrl.mean()
    return (data_mean - ctrl_mean) / (data_mean + ctrl_mean)


def common_neighbor_weight_bias(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    # A sparse matrix, as `*` is elementwise for a sparse array
    adj_bin = sp.csr_matrix((adj.tocsc() > 0).astype(int))
    if direction == "efferent":
        cn = adj_bin * adj_bin.transpose()
    elif direction == "afferent":
        cn = adj_bin.transpose() * adj_bin
    else:
        raise ValueError("Unknown direction: {0}".format(direction))

    return np.corrcoef(cn[adj > 0], adj[adj > 0])[0, 1]


def common_neighbor_connectivity_bias(
    adj, neuron_properties=None, direction="efferent", cols_location=None, fit_log=False
):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    from scipy import stats
    from scipy.spatial import distance

    if adj.dtype == bool:
        adj = adj.astype(int)

    if direction == "efferent":
        cn = adj * adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() * adj
    else:
        raise ValueError("Unknown direction: {0}".format(direction))

    # Least squares fit of  CN ~ 1 + Connected (+ Distance)  over all pairs of neurons, the diagonal included
    response = cn.toarray().ravel().astype(float)
    if fit_log:
        response = np.log10(response + fit_log)
    connected = adj.astype(bool).toarray().ravel()
    columns = [np.ones(len(response))]
    names = ["Intercept"]
    columns.append(connected.astype(float))
    names.append("Connected")
    if cols_location is not None:
        dist = distance.squareform(distance.pdist(neuron_properties[cols_location].values)).ravel()
        centre, scale = (
            dist.mean(),
            dist.std() or 1.0,
        )  # fitted on standardised distances for precision
        columns.append((dist - centre) / scale)
        names.append("Distance")
    design = np.column_stack(columns)
    gram = design.T @ design
    gram_inverse = np.linalg.pinv(gram)
    coef = gram_inverse @ (design.T @ response)
    residuals = response - design @ coef
    dof = len(response) - np.linalg.matrix_rank(gram)
    std_error = np.sqrt(np.diag(gram_inverse) * (residuals @ residuals) / dof)
    pvalues = 2 * stats.t.sf(np.abs(coef / std_error), dof)

    pval = pvalues[names.index("Connected")]
    mdl_added = coef[names.index("Connected")]
    mdl_distance, mdl_intercept = 0.0, coef[0]
    if "Distance" in names:  # back to the unit of the distances
        mdl_distance = coef[names.index("Distance")] / scale
        mdl_intercept = coef[0] - coef[names.index("Distance")] * centre / scale
    return pval, mdl_added / mdl_intercept, 100 * mdl_distance / mdl_intercept


# TODO ADD CODE FROM CLUSTER OR WITH CONNECTOM UTILITIES CODE TO COMPUTE PROBABILITY OF CONNECTION PER PATHWAY OR ANY OTHER PROPERTIE ON THE NEURON_PROPERTY.


def gini_curve(m, nrn, direction="efferent"):
    m = m.tocsc()
    if direction == "afferent":
        degrees = np.array(m.sum(axis=0)).flatten()
    elif direction == "efferent":
        degrees = np.array(m.sum(axis=1).flatten())
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)
    total = np.sum(degrees)
    if degrees.ndim == 1:
        degrees = np.sort(degrees)[::-1]
    # NOTE: the degrees of a sparse *matrix* in the efferent direction are a 2d (1, n) array.  They were passed
    # to sorted(), which sorts the rows, so these degrees have never been sorted.
    cs = np.cumsum(degrees).astype(float) / total
    return pd.Series(cs, index=np.linspace(0, 1, len(cs)))


def gini_coefficient(m, nrn, direction="efferent"):
    gc = gini_curve(m, nrn, direction=direction)
    A = gc.index.values
    B = gc.values
    return np.sum(np.diff(A) * (B[:-1] + B[1:]) / 2.0)


def _analytical_expected_gini_curve(m, direction="efferent"):
    if direction == "afferent":
        N = m.shape[0] - 1
        C = m.shape[1] * N
    elif direction == "efferent":
        N = m.shape[1] - 1
        C = m.shape[0] * N
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    P = m.nnz / C
    # Only using degrees, not distribution of weigthts. TODO: Fix that
    x = np.arange(N, -1, -1)
    p = binom.pmf(x, N, P)
    A = np.cumsum(p) / p.sum()
    B = np.cumsum(p * x) / np.sum(p * x)
    return pd.Series(B, index=A)


def normalized_gini_coefficient(m, nrn, direction="efferent"):
    gc = gini_coefficient(m, nrn, direction=direction)
    ctrl = _analytical_expected_gini_curve(m, direction=direction)
    A = ctrl.index.values
    B = ctrl.values
    return 2 * (gc - np.sum(np.diff(A) * (B[:-1] + B[1:]) / 2.0))


def _bin_degrees(degrees):
    nbins = np.maximum(int(len(degrees) * 0.1), np.minimum(len(degrees), 30))
    mx = np.nanmax(degrees)
    mn = np.nanmin(degrees)
    bins = np.linspace(mn, mx + 1e-6 * (mx - mn), nbins + 1)
    degrees = np.digitize(degrees, bins=bins) - 1
    udegrees = np.arange(nbins)
    ret_x = 0.5 * (bins[:-1] + bins[1:])
    return ret_x, udegrees, degrees


###### TODO: End of block that needs docstrings ######


def _at_least(levels, thresholds, weights=None):
    """For each threshold, the number of ``levels`` that are at least as large (or the sum of their ``weights``)."""
    order = np.argsort(levels, kind="stable")
    first = np.searchsorted(levels[order], thresholds, side="left")
    if weights is None:
        return len(levels) - first
    # sums over the tail of the sorted weights, so that small tails keep their precision
    tails = np.concatenate(
        [np.cumsum(weights[order][::-1])[::-1], np.zeros(1, dtype=np.cumsum(weights[:1]).dtype)]
    )
    return tails[first]


def rich_club_curve(m, direction="TOTAL"):
    """Compute the rich-club curve of the matrix m.

    Parameters
    ----------
    m : array_like or sparse matrix
        Adjacency matrix of the graph.
    direction : str {'OUT', 'IN', 'TOTAL'}
        'OUT' : Compute the rich-club curve for the out-degree.

        'IN' : Compute the rich-club curve for the in-degree.

        'TOTAL' : Compute the rich-club curve for the total degree i.e., in-degree + out-degree.

    Returns
    -------
    Pandas series
        With index the center of the binned degrees and values the rich club coefficient at that degree.
    Notes
    -----
    The rich-club coefficient is a measure of the tendency of high-degree nodes to form tightly interconnected communities.
    """
    m = sp.csc_matrix(m)
    if direction == "TOTAL":
        direction = None
    degrees = node_degree(m, direction=direction)
    if m.dtype == bool:
        udegrees = np.arange(1, degrees.max() + 1)
        ret_x = udegrees
    else:
        ret_x, udegrees, degrees = _bin_degrees(degrees)

    levels = np.asarray(degrees)
    coo = m.tocoo()
    # An edge is inside the nodes of level >= i when both of its ends are
    edge_levels = np.minimum(levels[coo.row], levels[coo.col])
    n_nodes = _at_least(levels, udegrees)  # nodes of level >= i
    n_possible = n_nodes * (n_nodes - 1)  # number of pot. edges
    weight = _at_least(edge_levels, udegrees, coo.data)  # number of actual edges
    ret = weight.astype(float) / n_possible
    return pd.Series(ret, index=pd.Index(ret_x, name="degree"))


def efficient_rich_club_curve(
    M, direction="TOTAL", pre_calculated_filtration=None, sparse_bin_set=False
):
    """Fast computation of the rich-club curve of the matrix M with respect to a degree filtration and possibly any filtration chosen by the user.

    Parameters
    ----------
    M : array_like or sparse matrix
        Adjacency matrix of the graph.
    direction : str {'OUT', 'IN', 'TOTAL'}
        'OUT' : Compute the rich-club curve for the out-degree.

        'IN' : Compute the rich-club curve for the in-degree.

        'TOTAL' or None : Compute the rich-club curve for the total degree i.e., in-degree + out-degree.
    pre_calculated_filtration : pandas series
        To provide user defined filtration values other than degree.
        The index are the nodes of M and values the filtration values.
    sparse_bin_set : bool
        If False, all integer values between 0 and the maximum degree/filtration value will be used to generate the bins.
        If True, unique values of the degrees/filtration will be used.
        This is useful when the degrees/filtration values are sparse over the whole range.
    Returns
    -------
    Pandas series
        With index the center of the binned degrees/filtration and values the rich club coefficient at that degree.
    Notes
    -----
    The rich-club coefficient is a measure of the tendency of high-degree nodes (nodes with a high filtration value) to form tightly interconnected communities.
    """
    # TODO: Maybe expand the notes explaining this concept.
    if pre_calculated_filtration is not None:
        deg = pre_calculated_filtration
    else:
        if direction == "TOTAL":
            direction = None
        deg = node_degree(M, direction=direction)
    M = M.tocoo()
    shape = M.shape
    M = pd.DataFrame.from_dict({"row": M.row, "col": M.col})

    if not sparse_bin_set:
        degree_bins = np.arange(deg.max() + 2)
    else:
        degree_bins = np.unique(np.append(deg, [0, deg.max() + 1]))
    degree_bins_rv = degree_bins[-2::-1]
    nrn_degree_distribution = np.histogram(deg.values, bins=degree_bins)[0]
    nrn_cum_degrees = np.cumsum(nrn_degree_distribution[-1::-1])
    nrn_cum_pairs = nrn_cum_degrees * (nrn_cum_degrees - 1)

    deg_arr = np.zeros(shape[0], dtype=int)
    deg_arr[deg.index.values] = deg.values

    deg = None

    con_degree = np.minimum(deg_arr[M["row"].values], deg_arr[M["col"].values])
    M = None
    con_degree = np.histogram(con_degree, bins=degree_bins)[0]

    cum_degrees = np.cumsum(con_degree[-1::-1])

    return pd.Series(cum_degrees / nrn_cum_pairs, degree_bins_rv)[::-1]


###### TODO: This block needs docstrings ######


def _analytical_expected_rich_club_curve(m, direction="efferent"):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    indegree = np.array(m.sum(axis=0))[0]
    outdegree = np.array(m.sum(axis=1))[:, 0]

    if direction == "afferent":
        degrees = indegree
    elif direction == "efferent":
        degrees = outdegree
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)

    udegrees = np.arange(1, degrees.max() + 1)
    edge_counter = lambda i: (degrees >= i).sum() * ((degrees >= i).sum() - 1)
    res_mn = []
    res_sd = []
    for deg in udegrees:
        valid = np.nonzero(degrees >= deg)[0]
        i_v = indegree[valid]
        i_sum_all = indegree.sum() - i_v
        i_sum_s = i_v.sum() - i_v
        o_v = outdegree[valid]
        mean, variance = hypergeom.stats(i_sum_all, i_sum_s, o_v)
        res_mn.append(np.sum(mean) / edge_counter(deg))
        res_sd.append(
            np.sqrt(variance.sum()) / edge_counter(deg)
        )  # Sum the variances, but divide the std
    df = pd.DataFrame.from_dict({"mean": np.array(res_mn), "std": np.array(res_sd)})
    df.index = udegrees
    return df


def generate_degree_based_control(M, direction="efferent"):
    """A shuffled version of a connectivity matrix that aims to preserve degree distributions.

    If direction = "efferent", then the out-degree is exactly preserved, while the in-degree is
    approximately preserved. Otherwise it's the other way around.  Uses ``numpy.random``.

    Note that a CSR (efferent) or CSC (afferent) input is shuffled in place and returned.
    """
    if direction == "efferent":
        M = M.tocsr()
        idxx = np.arange(M.shape[1])
        p_out = np.array(M.mean(axis=0))[0]
    elif direction == "afferent":
        M = M.tocsc()
        idxx = np.arange(M.shape[0])
        p_out = np.array(M.mean(axis=1))[:, 0]
    else:
        raise ValueError()

    for col in range(M.shape[1]):
        p = p_out.copy()
        p[col] = 0.0
        p = p / p.sum()
        a = M.indptr[col]
        b = M.indptr[col + 1]
        M.indices[a:b] = np.random.choice(idxx, b - a, p=p, replace=False)
    return M


# The names of the directions in the functions about degrees and the ones about controls differ
_DEGREE_DIRECTION = {"efferent": "OUT", "afferent": "IN"}


def _randomized_control_rich_club_curve(m, direction="efferent", n=10, curve_direction="TOTAL"):
    res = []
    for _ in range(n):
        m_shuf = generate_degree_based_control(m, direction=direction)
        res.append(efficient_rich_club_curve(m_shuf, direction=curve_direction))
    res = pd.concat(res, axis=1)
    # TODO: Something is wrong here. rr is not defined. Should it be res?
    #      But changing rr to res causing
    df = pd.DataFrame.from_dict({"mean": np.nanmean(res, axis=1), "std": np.nanstd(res, axis=1)})
    df.index = res.index
    return df


def normalized_rich_club_curve(
    m, direction="efferent", normalize="std", normalize_with="shuffled", **kwargs
):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    data = efficient_rich_club_curve(m, direction=_DEGREE_DIRECTION.get(direction, direction))
    A = data.index.values
    B = data.values
    if normalize_with == "analytical":
        ctrl = _analytical_expected_rich_club_curve(m, direction=direction)
    elif normalize_with == "shuffled":
        ctrl = _randomized_control_rich_club_curve(
            m, direction=direction, curve_direction=_DEGREE_DIRECTION.get(direction, direction)
        )
    mn_r = ctrl["mean"].values
    sd_r = ctrl["std"].values

    if normalize == "mean":
        return pd.Series(B[: len(mn_r)] / mn_r, index=A[: len(mn_r)])
    elif normalize == "std":
        return pd.Series((B[: len(mn_r)] - mn_r) / sd_r, index=A[: len(mn_r)])
    else:
        raise Exception("Unknown normalization: %s" % normalize)


def rich_club_coefficient(m, **kwargs):
    Bn = normalized_rich_club_curve(m, normalize="std", **kwargs).values
    return np.nanmean(Bn)


###### TODO: End of block that needs docstrings ######


def ccc(matrix):
    """Computes the classical clustering coefficient of a directed graph

    Parameters
    ----------
    matrix : numpy array
        the adjaceny matrix of an unweighted graph

    Returns
    -------
    Series
        The index is the node index in ``matrix`` and the value is the clustering coefficient of the node.

    References
    ----------
    The formula is taken from the following paper.

    [1] G. Fagiolo, "Clustering in complex directed networks", 2006;
            [DOI: 10.1103/PhysRevE.76.026107](https://doi.org/10.1103/PhysRevE.76.026107).

    [2] Conceição, Pedro, et al. "An application of neighbourhoods in digraphs to the classification of binary dynamics.", 2022
            [DOI: 10.1162/netn_a_00228](https://doi.org/10.1162/netn_a_00228).

    """
    # We only analyze the udnerlying connectivity not the weights
    matrix = matrix.astype(bool).astype(int)
    # Numerator: the diagonal of the cube of the symmetric matrix s is the row sum of (s @ s) * s
    if sp.issparse(matrix):
        symmetric = matrix + matrix.transpose()
        numerator = (
            0.5 * np.asarray((symmetric @ symmetric).multiply(symmetric).sum(axis=1)).ravel()
        )
    else:
        symmetric = matrix + np.transpose(matrix)
        numerator = 0.5 * np.einsum("ij,ji->i", symmetric @ symmetric, symmetric)
    # Denominator
    deg = node_degree(matrix)
    denominator = deg * (deg - 1) - 2 * np.asarray(rc_submatrix(matrix).sum(axis=0)).ravel()
    return numerator / denominator


def reciprocal_connections(matrix, chief_only=False, center=0):
    """Counts the reciprocal connections in the graph with adjacency matrix ``matrix``.

    Parameters
    ----------
    matrix : 2d-array or sparse matrix
        Adjacency matrix of a directed network.
    chief_only : bool
        If ``True`` only the reciprocal connections of the node ``center`` are counted.
    center : int
        The node used when ``chief_only`` is ``True``.

    Returns
    -------
    int
        The number of reciprocal connections of the node ``center``, or of the whole graph (each counted once).
    """
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    if chief_only:
        return np.count_nonzero(np.multiply(matrix[center], np.transpose(matrix)[center]))
    return np.count_nonzero(np.multiply(matrix, np.transpose(matrix))) // 2


def largest_strongly_connected_component(adjacency_matrix):
    """Computes the largest strongly connected component of the graph with adjacency matrix adjacency_matrix,
        and returns the adjacency matrix of said component

    Parameters
    ----------
    adjacency_matrix : numpy array
        the adjaceny matrix of the DiGraph as a numpy array


    Returns
    -------
    numpy array
        The adjacency matrix of the largest strongly connected component

    """
    if sp.issparse(adjacency_matrix):
        sparse = adjacency_matrix.tocsr(copy=True)
        sparse.eliminate_zeros()  # stored zeros are no edges in a dense matrix
        sparse.sort_indices()
        current_tribe_nx = nx.from_scipy_sparse_array(sparse, create_using=nx.DiGraph)
    else:
        current_tribe_nx = np_to_nx(adjacency_matrix)
    largest_comp = max(nx.strongly_connected_components(current_tribe_nx), key=len)
    current_tribe_strong_nx = current_tribe_nx.subgraph(largest_comp)
    current_tribe_strong = nx_to_np(current_tribe_strong_nx)
    return current_tribe_strong


def nx_to_np(directed_graph):
    """Converts networkx digraph to numpy array of the adjacency matrix

    Parameters
    ----------
    directed_graph : networkx DiGraph
        a directed graph

    Returns
    -------
    numpy array
        the adjaceny matrix of the DiGraph as a numpy array
    """
    return nx.to_numpy_array(directed_graph, dtype=int)


def np_to_nx(adjacency_matrix):
    """Converts numpy array of an adjacency matrix to a networkx digraph

    Parameters
    ----------
    adjacency_matrix : numpy array
        the adjaceny matrix of the DiGraph as a numpy array


    Returns
    -------
    networkx DiGraph
            a directed graph

    """
    return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
