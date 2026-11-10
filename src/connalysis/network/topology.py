# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Network analysis functions based on topological constructions
#
# Author(s): D. Egas Santander, M. Santoro, JP. Smith, V. Sood
# Last modified: 03/2023

# TODO unweighted: convex hull, combinatorial ricci,
# TODO weighted: filtered_simplex_counts, persistence,

import contextlib
import itertools
import logging
import os
import tempfile
import time
from itertools import chain

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .local import neighbourhood

# The compiled flagser packages (pyflagsercount, flagserpy) are imported when a function first needs them.

LOG = logging.getLogger(__name__)

#######################################################
################# UNWEIGHTED NETWORKS #################
#######################################################


def rc_submatrix(adj):
    """Returns the symmetric submatrix of reciprocal connections of adj
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.

    Returns
    -------
    sparse matrix
        symmetric matrix of the same dtype as adj of reciprocal connections
    """
    adj = sp.csr_matrix(adj)
    if np.count_nonzero(adj.diagonal()) != 0:
        LOG.warning("The diagonal is non-zero and this may lead to errors!")
    mask = adj.astype("bool")
    mask = mask.multiply(mask.T)
    return adj.multiply(mask).astype(adj.dtype)


def underlying_undirected_matrix(adj):
    """Returns the symmetric matrix of undirected connections of `adj`.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry in `adj[i][j]` implies there is an edge from vertex `i` to vertex `j`.

    Returns
    -------
    sparse boolean matrix
        Corresponding to the symmetric underlying undirected graph
    """
    adj = sp.csr_matrix(adj)
    if np.count_nonzero(adj.diagonal()) != 0:
        LOG.warning("The diagonal is non-zero and this may lead to errors!")
    return (adj + adj.T).astype("bool")


def _series_by_dim(from_array, name_index=None, index=None, name=None):
    """A series of counts, like simplex counts:
    one count for a given value of simplex dimension.
    """
    if from_array is None:
        return None
    if index is None:
        index = pd.Index(range(len(from_array)), name=name_index)
    else:
        assert len(index) == len(from_array), "array and index are not the same length"
        index = pd.Index(index, name=name_index)
    return pd.Series(from_array, index=index, name=name)


def _padded_with_nan(rows):
    """``rows`` as they are if they all have the same length, otherwise as a float array padded with NaN."""
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    if len(lengths) == 0 or (lengths == lengths[0]).all():
        return rows
    out = np.full((len(rows), int(lengths.max())), np.nan)
    out[
        np.repeat(np.arange(len(rows)), lengths), np.concatenate([np.arange(n) for n in lengths])
    ] = np.fromiter(chain.from_iterable(rows), dtype=float, count=int(lengths.sum()))
    return out


def _pad_rows(rows, no_columns=None):
    """Integer array with one row per list in ``rows``, padded on the right with zeros."""
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    width = int(lengths.max(initial=0)) if no_columns is None else no_columns
    out = np.zeros((len(rows), width), dtype=np.int64)
    if lengths.sum():
        out[
            np.repeat(np.arange(len(rows)), lengths),
            np.concatenate([np.arange(n) for n in lengths]),
        ] = np.fromiter(chain.from_iterable(rows), dtype=np.int64, count=int(lengths.sum()))
    return out


def _frame_by_dim(from_array, no_columns, name, index):
    """A dataframe of counts, like node participation:
    one count for a node and simplex dimension.
    """
    if from_array is None:
        return None
    # Todo add method for when no_columns is not given
    columns = pd.Index(range(no_columns), name=index)
    return pd.DataFrame(_padded_with_nan(from_array), columns=columns).fillna(0).astype(int)


QUANTITIES = {"simplices", "node-participation", "bettis", "bidrectional-edges"}


def _flagser_counts(
    adjacency,
    max_simplices=False,
    count_node_participation=False,
    list_simplices=False,
    threads=8,
    max_dim=-1,
    edge_containment=False,
    vertices_todo="",
):
    """Call package `pyflagsercount's flagser_count` method that can be used to compute
    some analyses, getting counts of quantities such as simplices,
    or node-participation (a.k.a. `containment`)
    """
    import pyflagsercount

    adjacency = sp.csr_matrix(adjacency.astype(bool).astype(int))
    if np.count_nonzero(adjacency.diagonal()) != 0:
        LOG.warning("The diagonal is non-zero!  Non-zero entries in the diagonal will be ignored.")

    flagser_counts = pyflagsercount.flagser_count(
        adjacency,
        max_simplices=max_simplices,
        containment=count_node_participation,
        return_simplices=list_simplices,
        threads=threads,
        max_dim=max_dim,
        edge_containment=edge_containment,
        vertices_todo=vertices_todo,
    )

    counts = {
        "euler": flagser_counts.pop("euler"),
        "simplex_counts": _series_by_dim(
            flagser_counts.pop("cell_counts"), name="simplex_count", name_index="dim"
        ),
        "max_simplex_counts": _series_by_dim(
            flagser_counts.pop("max_cell_counts", None), name="max_simplex_count", name_index="dim"
        ),
        "simplices": flagser_counts.pop("simplices", None),
    }
    if counts["max_simplex_counts"] is None:
        max_dim_participation = counts["simplex_counts"].shape[0]
    else:
        max_dim_participation = counts["max_simplex_counts"].shape[0]
    counts["node_participation"] = _frame_by_dim(
        flagser_counts.pop("contain_counts", None),
        max_dim_participation,
        name="node_participation",
        index="node",
    )
    counts.update(flagser_counts)
    return counts


@contextlib.contextmanager
def _vertices_todo_file(nodes):
    """Path of an ``.npy`` file listing ``nodes`` for flagser (``''`` for none); removed on exit."""
    if nodes is None:
        yield ""
        return
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "vertices_todo.npy")
        np.save(path, nodes, allow_pickle=False)
        yield path


def node_degree(adj, node_properties=None, direction=None, weighted=False, **kwargs):
    """Compute degree of nodes in network adj
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].
    node_properties : data frame
        Data frame of neuron properties in adj. Only necessary if used in conjunction with TAP or connectome utilities.
    direction : string or tuple of strings
        Direction for which to compute the degree

        'IN' - In degree

        'OUT'- Out degree

        None or ('IN', 'OUT') - Total degree i.e. IN+OUT

    Returns
    -------
    series or data frame

    Raises
    ------
    Warning
        If adj has non-zero entries in the diagonal
    AssertionError
        If direction is invalid
    """
    assert not direction or direction in ("IN", "OUT") or tuple(direction) == ("IN", "OUT"), (
        f"Invalid `direction`: {direction}"
    )

    if sp.issparse(adj):
        matrix = adj.tocsr()
        if not matrix.has_canonical_format:
            matrix = matrix.copy()
            matrix.sum_duplicates()  # the same entry listed twice is one entry
        if not weighted:
            matrix = matrix.astype("bool")
        diagonal = matrix.diagonal()
    else:
        matrix = np.array(adj)
        if not weighted:
            matrix = matrix.astype("bool")
        diagonal = np.diag(matrix)
    if np.count_nonzero(diagonal) != 0:
        LOG.warning("The diagonal is non-zero!  This may cause errors in the analysis")
    index = pd.Series(range(matrix.shape[0]), name="node")

    def in_degree():
        return pd.Series(np.asarray(matrix.sum(axis=0)).ravel(), index)

    def out_degree():
        return pd.Series(np.asarray(matrix.sum(axis=1)).ravel(), index)

    if not direction:
        return in_degree() + out_degree()

    if tuple(direction) == ("IN", "OUT"):
        return pd.DataFrame({"IN": in_degree(), "OUT": out_degree()})

    if tuple(direction) == ("OUT", "IN"):
        return pd.DataFrame({"OUT": out_degree(), "IN": in_degree()})

    return in_degree() if direction == "IN" else out_degree()


def node_k_degree(adj, node_properties=None, direction=("IN", "OUT"), max_dim=-1, **kwargs):
    # TODO: Generalize from one population to another
    """Compute generalized degree of nodes in network adj.  The k-(in/out)-degree of a node v is the number of
    k-simplices with all its nodes mapping to/from the node v.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].  The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    direction : string
        Direction for which to compute the degree

        'IN' - In degree

        'OUT'- Out degree

        (’IN’, ’OUT’) - both
    max_dim : int
        Maximal dimension for which to compute the degree max_dim >=2 or -1 in
        which case it computes all dimensions.

    Returns
    -------
    data frame
        Table of of k-(in/out)-degrees

    Raises
    ------
    Warning
        If adj has non-zero entries in the diagonal which are ignored in the analysis
    AssertionError
        If direction is invalid
    AssertionError
        If not max_dim >1

    Notes
    -----
    Note that the k-in-degree of a node v is the number of (k+1) simplices the node v is a sink of.
    Dually, the k-out-degree of a node v is the number of (k+1) simplices the node v is a source of.
    """
    matrix = sp.csr_matrix(adj)
    assert (max_dim > 1) or (max_dim == -1), "max_dim should be >=2"
    assert direction in ("IN", "OUT") or tuple(direction) == ("IN", "OUT"), (
        f"Invalid `direction`: {direction}"
    )
    if np.count_nonzero(matrix.diagonal()) != 0:
        LOG.warning("The diagonal is non-zero!  Non-zero entries in the diagonal will be ignored.")
    import pyflagsercount

    flagser_out = pyflagsercount.flagser_count(matrix, return_simplices=True, max_dim=max_dim)
    max_dim_possible = len(flagser_out["cell_counts"]) - 1
    if max_dim == -1:
        max_dim = max_dim_possible
    elif max_dim > max_dim_possible:
        LOG.warning("The maximum dimension selected is not attained")
        max_dim = max_dim_possible
    if (max_dim <= 1) and (max_dim != -1):
        LOG.info("There are no simplices of dimension 2 or higher")
    else:
        index = pd.Series(range(matrix.shape[0]), name="node")
        generalized_degree = pd.DataFrame(index=index)
        for dim in np.arange(2, max_dim + 1):
            if "OUT" in direction:
                # getting source participation across dimensions
                x, y = np.unique(np.array(flagser_out["simplices"][dim])[:, 0], return_counts=True)
                generalized_degree[f"{dim - 1}_out_degree"] = pd.Series(y, index=x)
            if "IN" in direction:
                # getting sink participation across dimensions
                x, y = np.unique(
                    np.array(flagser_out["simplices"][dim])[:, dim], return_counts=True
                )
                generalized_degree[f"{dim - 1}_in_degree"] = pd.Series(y, index=x)
        return generalized_degree.fillna(0)


def simplex_counts(
    adj,
    node_properties=None,
    max_simplices=False,
    threads=8,
    max_dim=-1,
    simplex_type="directed",
    nodes=None,
    **kwargs,
):
    # TODO: ADD TRANSPOSE
    """Compute the number of simplex motifs in the network adj.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].  The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type: string
        Type of simplex to consider (See Notes):

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections
    nodes : 1d array or None(default)
        Restrict to list only the simplices whose source node is in nodes.  If None list all simplices.
        This only makes sense for directed simplices.

    Returns
    -------
    series
        simplex counts

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.

    Notes
    -----
    A directed simplex of dimension k in adj is a set of (k+1) nodes which are all to all connected in a feedforward manner.
    That is, they can be ordered from 0 to k such that there is an edge from i to j whenever i < j.

    An undirected simplex of dimension k in adj is a set of (k+1) nodes in adj which are all to all connected.  That is, they
    are all to all connected in the underlying undirected graph of adj.  In the literature this is also called a (k+1)-clique
    of the underlying undirected graph.

    A reciprocal simplex of dimension k in adj is a set of (k+1) nodes in adj which are all to all reciprocally connected.
    That is, they are all to all connected in the undirected graph of reciprocal connections of adj.  In the literature this is
    also called a (k+1)-clique of the undirected graph of reciprocal connections.
    """
    adj = sp.csr_matrix(adj)
    assert np.count_nonzero(adj.diagonal()) == 0, (
        "The diagonal of the matrix is non-zero and this may lead to errors!"
    )
    N, M = adj.shape
    assert N == M, "Dimension mismatch. The matrix must be square."

    # Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type == "undirected":
        adj = sp.triu(underlying_undirected_matrix(adj))  # symmtrize and keep upper triangular only
    elif simplex_type == "reciprocal":
        adj = sp.triu(rc_submatrix(adj))  # symmtrize and keep upper triangular only

    # Only the simplices that have sources stored in a temporary file will be considered.
    # This only makes sense for directed simplices
    if nodes is not None:
        assert simplex_type == "directed", (
            "Sub-selection of source nodes only makes sense for directed simplices"
        )

    # Count simplices
    with _vertices_todo_file(nodes) as vertices_todo:
        flagser_counts = _flagser_counts(
            adj,
            threads=threads,
            max_simplices=max_simplices,
            max_dim=max_dim,
            vertices_todo=vertices_todo,
        )

    if max_simplices:
        return flagser_counts["max_simplex_counts"]
    else:
        return flagser_counts["simplex_counts"]


def normalized_simplex_counts(
    adj, node_properties=None, max_simplices=False, threads=8, max_dim=-1, **kwargs
):
    """Compute the ratio of directed/undirected simplex counts normalized to be between 0 and 1.
    See simplex_counts and undirected_simplex_counts for details.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].  The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.

    Returns
    -------
    panda series
        Normalized simplex counts

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.

    Notes
    -----
    Maybe we should say why we choose this metric"""

    from scipy.special import factorial

    denominator = simplex_counts(
        adj,
        node_properties=node_properties,
        max_simplices=max_simplices,
        threads=threads,
        max_dim=max_dim,
        simplex_type="undirected",
        **kwargs,
    ).to_numpy()
    # Global maximum dimension since every directed simplex has an underlying undirected one of the same dimension
    max_dim_global = denominator.size
    # Maximum number of possible directed simplices for each undirected simplex across dimensions
    max_possible_directed = np.array([factorial(i + 1) for i in np.arange(max_dim_global)])
    denominator = np.multiply(denominator, max_possible_directed)
    numerator = simplex_counts(
        adj,
        node_properties=node_properties,
        max_simplices=max_simplices,
        threads=threads,
        max_dim=max_dim,
        simple_type="directed",
        **kwargs,
    ).to_numpy()
    numerator = np.pad(
        numerator, (0, max_dim_global - len(numerator)), "constant", constant_values=0
    )
    return _series_by_dim(
        np.divide(numerator, denominator)[1:],
        name="normalized_simplex_counts",
        index=np.arange(1, max_dim_global),
        name_index="dim",
    )


def node_participation(
    adj,
    node_properties=None,
    max_simplices=False,
    threads=8,
    max_dim=-1,
    simplex_type="directed",
    **kwargs,
):
    """Compute the number of simplex motifs in the network adj each node is part of.
    See simplex_counts for details.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False (default) counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type : string
        Type of simplex to consider:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections

    Returns
    -------
    data frame
        Indexed by the nodes in adj and with columns de dimension for which node participation is counted

    Raises
    -------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    """

    adj = sp.csr_matrix(adj).astype("bool")
    assert np.count_nonzero(adj.diagonal()) == 0, (
        "The diagonal of the matrix is non-zero and this may lead to errors!"
    )
    N, M = adj.shape
    assert N == M, "Dimension mismatch. The matrix must be square."

    # Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type == "undirected":
        adj = sp.triu(underlying_undirected_matrix(adj))  # symmtrize and keep upper triangular only
    elif simplex_type == "reciprocal":
        adj = sp.triu(rc_submatrix(adj))  # symmtrize and keep upper triangular only

    flagser_counts = _flagser_counts(
        adj,
        count_node_participation=True,
        threads=threads,
        max_simplices=max_simplices,
        max_dim=max_dim,
    )
    return flagser_counts["node_participation"]


def _edge_participation_frame(counts):
    """Table of edge participation: one row per edge (a tuple), one column per dimension."""
    if not counts:
        return pd.DataFrame.from_dict(counts, orient="index").astype(int)
    return pd.DataFrame(
        _pad_rows(list(counts.values())), index=pd.Index(list(counts), tupleize_cols=False)
    )


def edge_participation(
    adj,
    node_properties=None,
    max_simplices=False,
    threads=8,
    max_dim=-1,
    simplex_type="directed",
    return_simplex_counts=False,
    verbose=False,
    **kwargs,
):
    """Compute the number of simplex motifs in the network adj each edge is part of.
    See simplex_counts for details.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False (default) counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type : string
        Type of simplex to consider:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections

    Returns
    -------
    data frame
        Indexed by the edges in adj and with columns de dimension for which node participation is counted

    Raises
    -------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    """

    adj = sp.csr_matrix(adj).astype("bool")
    assert np.count_nonzero(adj.diagonal()) == 0, (
        "The diagonal of the matrix is non-zero and this may lead to errors!"
    )
    N, M = adj.shape
    assert N == M, "Dimension mismatch. The matrix must be square."

    # Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type == "undirected":
        adj = sp.triu(underlying_undirected_matrix(adj))  # symmtrize and keep upper triangular only
    elif simplex_type == "reciprocal":
        adj = sp.triu(rc_submatrix(adj))  # symmtrize and keep upper triangular only

    import pyflagsercount

    flagser_out = pyflagsercount.flagser_count(
        adj, edge_containment=True, threads=threads, max_simplices=max_simplices, max_dim=max_dim
    )
    LOG.info("Done running flagser")
    e_contain = _edge_participation_frame(flagser_out["edge_contain_counts"])
    if return_simplex_counts:
        return e_contain, flagser_out["cell_counts"]
    else:
        return e_contain


def list_simplices_by_dimension(
    adj,
    node_properties=None,
    max_simplices=False,
    max_dim=-1,
    nodes=None,
    verbose=False,
    simplex_type="directed",
    **kwargs,
):
    """List all simplex motifs in the network adj.
    Parameters
    ----------
    adj : 2d (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties :  data frame
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type : string
        Type of simplex to consider:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections
    nodes : 1d array or None(default)
        Restrict to list only the simplices whose source node is in nodes.  If None list all simplices

    Returns
    -------
    series
        Simplex lists indexed per dimension.  The dimension k entry is a (no. of k-simplices, k+1)-array
        is given, where each row denotes a simplex.

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    AssertionError
        If nodes is not a subarray of np.arange(N)

    See Also
    --------
    simplex_counts : A function that counts the simplices instead of listing them and has descriptions of the
    simplex types.
    """
    LOG.info("COMPUTE list of %ssimplices by dimension", "max-" if max_simplices else "")

    import pyflagsercount

    adj = sp.csr_matrix(adj)
    assert np.count_nonzero(adj.diagonal()) == 0, (
        "The diagonal of the matrix is non-zero and this may lead to errors!"
    )
    N, M = adj.shape
    assert N == M, "Dimension mismatch. The matrix must be square."
    if nodes is not None:
        assert np.isin(nodes, np.arange(N)).all(), (
            "nodes must be a subarray of the nodes of the matrix"
        )

    # Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type == "undirected":
        adj = sp.triu(underlying_undirected_matrix(adj))  # symmtrize and keep upper triangular only
    elif simplex_type == "reciprocal":
        adj = sp.triu(rc_submatrix(adj))  # symmtrize and keep upper triangular only

    n_threads = kwargs.get("threads", kwargs.get("n_threads", 1))

    # Only the simplices that have sources stored in a temporary file will be considered
    with _vertices_todo_file(nodes) as vertices_todo:
        original = pyflagsercount.flagser_count(
            adj,
            max_simplices=max_simplices,
            threads=n_threads,
            max_dim=max_dim,
            vertices_todo=vertices_todo,
            return_simplices=True,
        )["simplices"]

    # Format output
    max_dim = len(original)
    dims = pd.Index(np.arange(max_dim), name="dim")
    simplices = pd.Series(original, name="simplices", index=dims).apply(np.array)
    # When counting all simplices flagser doesn't list dim 0 and 1 because they correspond to vertices and edges
    if not max_simplices:
        if nodes is None:
            nodes = np.arange(0, N)
        coom = adj.tocoo()
        simplices[0] = np.reshape(nodes, (nodes.size, 1))
        mask = np.isin(coom.row, nodes)
        simplices[1] = np.stack([coom.row[mask], coom.col[mask]]).T
    return simplices


def in_degree_from_pop(adj, source_pop, max_simplices=False, threads=8, max_dim=-1, **kwargs):
    # TODO: DO THE OUTDEGREE VERSION
    # TODO: Get participation directly from flagsercount via vertices to do?
    """Compute generalized in-degree of nodes source_pop onto the rest of the nodes in adj.
    Parameters
    ----------
    adj: 2d (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    source_pop: list of indices of the source population, must be a subset of ``np.arange(0, adj.shape[0])``
    max_simplices : bool
        If False counts all simplices.
        If True counts only maximal simplices.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.
        Particularly useful for large or dense graphs.

    Returns
    -------
    Data frame
        Table of k-in-degrees from source_pop indexed by the target population.

    Raises
    ------
    AssertionError
        If adj restricted to source_pop has non-zero entries in the diagonal which can produce errors.
    """
    adj = adj.tocsr()
    source_pop = np.sort(source_pop)
    target_pop = np.setdiff1d(np.arange(adj.shape[0]), source_pop)
    adj_source = adj[np.ix_(source_pop, source_pop)]
    adj_cross = adj[np.ix_(source_pop, target_pop)]
    degs = cross_col_k_in_degree(
        adj_cross,
        adj_source,
        max_simplices=max_simplices,
        threads=threads,
        max_dim=max_dim,
        **kwargs,
    )
    degs.index = target_pop
    return degs


def cross_col_k_in_degree(
    adj_cross, adj_source, max_simplices=False, threads=8, max_dim=-1, **kwargs
):
    # TODO: DO THE OUTDEGREE VERSION
    # TODO: Get participation directly from flagsercount via vertices to do?
    """Compute generalized in-degree of nodes in adj_target from nodes in adj_source.
    The k-in-degree of a node v is the number of k-simplices in adj_source with all its nodes mapping to v
    through edges in adj_cross.
    Parameters
    ----------
    adj_cross : (n,m) array or sparse matrix
        Matrix of connections from the nodes in adj_n to the target population.
        n is the number of nodes in adj_source and m is the number of nodes in adj_target.
        A non-zero entry adj_cross[i,j] implies there is an edge from i-th node of adj_source
        to the j-th node of adj_target.
    adj_source : (n, n)-array or sparse matrix
        Adjacency matrix of the source network where n is the number of nodes in the source network.
        A non-zero entry adj_source[i,j] implies there is an edge from node i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    max_simplices : bool
        If False counts all simplices.
        If True counts only maximal simplices.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.
        Particularly useful for large or dense graphs.

    Returns
    -------
    Data frame
        Table of cross-k-in-degrees indexed by the m nodes in the target population.

    Raises
    ------
    AssertionError
        If adj_source has non-zero entries in the diagonal which can produce errors.
    """
    adj_source = sp.csr_matrix(adj_source).astype("bool")
    adj_cross = sp.csr_matrix(adj_cross).astype("bool")
    assert np.count_nonzero(adj_source.diagonal()) == 0, (
        "The diagonal of the source matrix is non-zero and this may lead to errors!"
    )
    assert adj_source.shape[0] == adj_source.shape[1], (
        "Dimension mismatch. The source matrix must be square."
    )
    assert adj_source.shape[0] == adj_cross.shape[0], (
        "Dimension mismatch. The source matrix and cross matrix must have the same number of rows."
    )

    n_source = adj_source.shape[0]  # Size of the source population
    n_target = adj_cross.shape[1]  # Size of the target population
    # Building a square matrix [[adj_source, adj_cross], [0,0]]
    adj = sp.bmat(
        [
            [adj_source, adj_cross],
            [
                sp.csr_matrix((n_target, n_source), dtype="bool"),
                sp.csr_matrix((n_target, n_target), dtype="bool"),
            ],
        ]
    )
    # Transposing to restrict computation to ``source nodes'' in adj_target in flagsercount
    adj = adj.T
    nodes = np.arange(n_source, n_source + n_target)  # nodes on target population
    slist = list_simplices_by_dimension(
        adj,
        max_simplices=max_simplices,
        max_dim=max_dim,
        nodes=nodes,
        simplex_type="directed",
        verbose=False,
        threads=threads,
        **kwargs,
    )

    # Count participation as a source in transposed matrix i.e. participation as sink in the original
    dims = slist.index[1:]
    degrees = np.zeros((n_target, len(dims)), dtype=np.int64)
    for column, dim in enumerate(dims):
        index, deg = np.unique(slist[dim][:, 0], return_counts=True)
        degrees[index - n_source, column] = deg
    if n_target == 0:
        degrees = degrees.astype(object)
    return pd.DataFrame(degrees, index=nodes, columns=dims)


def betti_counts(
    adj,
    node_properties=None,
    min_dim=0,
    max_dim=None,
    simplex_type="directed",
    approximation=None,
    **kwargs,
):
    """Count betti counts of flag complex of adj.  Type of flag complex is given by simplex_type.

    Parameters
    ----------
    adj : 2d (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.  Matrix will be cast to 0,1 entries so weights
        will be ignored.
    node_properties :  data frame
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    min_dim : int
        Minimal dimension from which betti counts are computed.
        The default min_dim = 0 (counting number of connected components).
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = None (or []) counts betti numbers up to the maximal dimension of the complex.
    simplex_type : string
        Type of flag complex to consider, given by the type of simplices it is built on.
        Possible types are:

        ’directed’ - directed simplices (directed flag complex)

        ’undirected’ - simplices in the underlying undirected graph (clique complex of the underlying undirected graph)

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections (clique complex of the
        undirected graph of reciprocal connections.)
    approximation : list of integers  or None
        Approximation parameter for the computation of the betti numbers.  Useful for large networks.
        If None all betti numbers are computed exactly.
        Otherwise, min_dim must be 0 and approximation but be a list of positive integers or -1.
        The list approximation is either extended by -1 entries on the right or sliced from [0:max_dim+1] to obtain
        a list of length max_dim.  Each entry of the list denotes the approximation value for the betti computation
        of that dimension if -1 approximation in that dimension is set to None.

        If the approximation value at a given dimension is `a` flagser skips cells creating columns in the reduction
        matrix with more than `a` entries.  This is useful for hard problems.  For large, sparse networks a good value
        if often `100,00`.  If set to `1` that dimension will be virtually ignored.  See [1]_

    Returns
    -------
    series
        Betti counts indexed per dimension from min_dim to max_dim.

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    AssertionError
        If approximation != None and min_dim != 0.

    See Also
    --------
    :func:`connalysis.network.topology.simplex_counts` :
    A function that counts the simplices forming the complex from which bettis are count.
    Simplex types are described there in detail.

    References
    ----------
    For details about the approximation algorithm see

    ..[1] D. Luetgehetmann, "Documentation of the C++ flagser library";
           [GitHub: luetge/flagser](https://github.com/luetge/flagser/blob/master/docs/documentation_flagser.pdf).

    """
    LOG.info(
        "Compute betti counts for %s-type adjacency matrix and %s-type node properties",
        type(adj),
        type(node_properties),
    )

    from flagserpy import flagser_unweighted

    # Checking matrix
    adj = sp.csr_matrix(adj).astype(bool).astype("int")
    assert np.count_nonzero(adj.diagonal()) == 0, (
        "The diagonal of the matrix is non-zero and this may lead to errors!"
    )
    N, M = adj.shape
    assert N == M, "Dimension mismatch. The matrix must be square."
    assert approximation is None or min_dim == 0, (
        "For approximation != None, min_dim must be set to 0.  \n"
        "Lower dimensions can be ignored by setting approximation to 1 on those dimensions"
    )

    # Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type == "undirected":
        adj = sp.triu(underlying_undirected_matrix(adj))  # symmtrize and keep upper triangular only
    elif simplex_type == "reciprocal":
        adj = sp.triu(rc_submatrix(adj))  # symmtrize and keep upper triangular only
    # Computing bettis
    if max_dim is None or (isinstance(max_dim, list) and len(max_dim) == 0):
        max_dim = np.inf

    if approximation is None:
        LOG.info("Run without approximation")
        bettis = flagser_unweighted(
            adj,
            min_dimension=min_dim,
            max_dimension=max_dim,
            directed=True,
            coeff=2,
            approximation=None,
        )["betti"]
    else:
        assert all(
            [isinstance(item, int) for item in approximation]
        )  # assert it's a list of integers
        approximation = np.array(approximation)
        bettis = []

        # Make approximation vector to be of size max_dim
        if max_dim != np.inf:
            if approximation.size - 1 < max_dim:  # Vector too short so pad with -1's
                approximation = np.pad(
                    approximation,
                    (0, max_dim - (approximation.size - 1)),
                    "constant",
                    constant_values=-1,
                )
            if approximation.size - 1 > max_dim:  # Vector too long, select relevant slice
                approximation = approximation[0 : max_dim + 1]
            # Sanity check
            LOG.info("Correct dimensions for approximation: %s", approximation.size == max_dim + 1)

        # Split approximation into sub-vectors of same value to speed up computation
        diff = approximation[1:] - approximation[:-1]
        slice_indx = np.array(np.where(diff != 0)[0]) + 1

        # Compute betti counts
        for dims_range in np.split(np.arange(approximation.size), slice_indx):
            n = dims_range[0]  # min dim for computation
            N = dims_range[-1]  # max dim for computation
            a = approximation[n]
            if a == -1:
                a = None
            LOG.info("Run betti for dim range %s-%s with approximation %s", n, N, a)
            bettis = (
                bettis
                + flagser_unweighted(
                    adj, min_dimension=n, max_dimension=N, directed=True, coeff=2, approximation=a
                )["betti"]
            )

        if max_dim == np.inf:
            n = approximation.size  # min dim for computation
            N = np.inf  # max dim for computation
            a = None
            LOG.info("Run betti for dim range %s-%s with approximation %s", n, N, a)
            bettis = (
                bettis
                + flagser_unweighted(
                    adj, min_dimension=n, max_dimension=N, directed=True, coeff=2, approximation=a
                )["betti"]
            )

    return pd.Series(
        bettis,
        name="betti_count",
        index=pd.Index(np.arange(min_dim, len(bettis) + min_dim), name="dim"),
    )


def _binary2simplex(address, test=None, verbosity=1000000):
    """...Not used --- keeping it here as it is of interest to understanmd
    how simplices are represented on the disc by Flagser.
    #INPUT: Address of binary file storing simplices
    #OUTPUT: A list if lists L where L[i] contains the vertex ids of the i'th simplex,
    #          note the simplices appear in no particular order

    """
    import resource  # Unix only

    LOG.info("Load binary simplex info from %s", address)
    simplex_info = pd.Series(np.fromfile(address, dtype=np.uint64))
    LOG.info("Done loading binary simplex info.")

    if test:
        simplex_info = simplex_info.iloc[0:test]

    mask64 = np.uint(1) << np.uint(63)
    mask21 = np.uint64(1 << 21) - np.uint64(1)
    mask42 = (np.uint64(1 << 42) - np.uint64(1)) ^ mask21
    mask63 = ((np.uint64(1 << 63) - np.uint64(1)) ^ mask42) ^ mask21
    end = np.uint64(2**21 - 1)

    def decode_vertices(integer):
        decode_vertices.ncalls += 1
        if decode_vertices.ncalls % verbosity == 0:
            mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            LOG.info(
                "\t progress %s / %s memory %s", decode_vertices.ncalls, len(simplex_info), mem_used
            )
        integer = np.uint64(integer)
        start = not ((integer & mask64) >> np.uint64(63))
        v0 = integer & mask21
        v1 = (integer & mask42) >> np.uint64(21)
        v2 = (integer & mask63) >> np.uint64(42)
        vertices = [v for v in [v0, v1, v2] if v != end]
        return pd.Series([start, vertices], index=["start", "vertices"])

    #    vertices = [start, v0, v1, v2]
    #    return pd.Series(vertices, index=["start", 0, 1, 2])
    decode_vertices.ncalls = 0

    LOG.info("Decode the simplices into simplex vertices")
    vertices = simplex_info.apply(decode_vertices)
    LOG.info("Done decoding to simplex vertices")

    vertices = vertices.assign(sid=np.cumsum(vertices.start)).reset_index(drop=True)

    simplices = vertices.set_index("sid").vertices.groupby("sid").apply(np.hstack)

    if not test:
        return simplices

    return (vertices.vertices, simplices)


def _generate_abstract_edges_in_simplices(dim, position="all"):
    """Generate indices of edges in a simplex with nodes 0, 1, ... dim

    Parameters
    ----------
    dim : int
        Dimension of the simplex
    position: str
        Position of the edges to extract

        'all': all edges of the simplex

        'spine': edges along the spine of the simplex

    Returns
    -------
    list
        list of pairs of nodes indexing the edges selected
    """
    if position == "all":
        edges_abstract = np.array(list(itertools.combinations(range(dim + 1), 2)))
    elif position == "spine":
        edges_abstract = np.array([[i, i + 1] for i in range(dim)])
    return edges_abstract


# def extract_submatrix_of_simplices(simplex_list, N, position="all"):
def extract_submatrix_from_simplex_list(simplex_list, N, position="spine"):
    """Generate binary submatrix of NxN matrix of edges in simplex list.

    Parameters
    ----------
    simplex list: 2d-array
        Array of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex
        indexed by the order of the nodes in an NxN matrix.
    N: int
        Number of nodes in original graph defining the NxN matrix.
    position: str
        Position of the edges to extract

        'all': all edges of the simplex (it is more efficient to do this with ``extract_submatrix_from_edge_par``)

        'spine': edges along the spine of the simplex
        (only makes sense for directed simplices)

    Returns
    -------
    coo bool matrix
        Matrix with of shape (N,N) with entries `True` corresponding to edges in simplices.
    """
    if position == "all":
        LOG.warning(
            "For edges in any position it is more efficient to use extract_submatrices_from_edge_participation"
        )
    if simplex_list.shape[0] == 0:
        return sp.csr_matrix((N, N), dtype=bool)  # no simplices in this dimension
    dim = simplex_list.shape[1] - 1
    # abstract list of edges to extract from each simplex
    edges_abstract = _generate_abstract_edges_in_simplices(dim, position=position)
    # Encode every edge (u, v) as the integer u * N + v: unique edges, in lexicographic order, from a 1d sort
    simplex_list = np.asarray(simplex_list, dtype=np.int64)
    codes = np.unique(
        np.concatenate([simplex_list[:, i] * N + simplex_list[:, j] for i, j in edges_abstract])
    )
    rows, cols = np.divmod(codes, N)
    return sp.coo_matrix((np.ones(codes.size), (rows, cols)), shape=(N, N)).astype(bool)


def extract_submatrices_from_edge_participation(edge_par, N, dims=None, thresh=1):
    """Generate binary submatrix of an NxN matrix of edges with edge participation greater than thresh.

    Parameters
    ----------
    edge_par: DataFrame
        DataFrame with edge participation values formatted as the output of ``edge_participation`` i.e.,
        indexed by the edges in adj and with columns de dimension for which edge participation is counted
    dims: list of ints
        dimensions of the simplices to consider, if ``None`` all positive dimensions are considered
    N: int
        Number of nodes in original graph defining the NxN matrix.
        The indices of edge_par must be smaller than N
    thresh: int
        Threshold value at which to consider an edge.  If thresh=1 all edges that
        elong to at least one simplex are considered.

    Returns
    -------
    dict
        keys: dimensions
        values: sparse binary matrices in coo format of shape (N,N) with entries `True` corresponding to edges
        that belong to at least thresh simplices of the dimension in its corresponding key.
    """

    if dims is None:
        dims = edge_par.columns
    dims = np.array(dims)
    assert np.isin(dims, edge_par.columns).all(), (
        "The array dims shoulds be a subset of the columns of edge participation"
    )
    mats = {}
    edges = pd.MultiIndex.from_tuples(edge_par.index)
    rows = edges.get_level_values(0).to_numpy()
    cols = edges.get_level_values(1).to_numpy()
    for dim in dims:
        keep = (edge_par[dim] >= thresh).to_numpy()
        mats[f"dimension_{dim}"] = sp.coo_matrix(
            (np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(N, N)
        ).astype(bool)
    return mats


def get_k_skeleta_graph(
    adj=None,
    dimensions=None,
    max_simplices=False,
    N=None,
    simplex_list=None,
    edge_par=None,
    simplex_type="directed",
    position="all",
    threads=8,
    **kwargs,
):
    """Return the edges of the (maximal) k-skeleton of the flag complex of adj for all k<= max_dim in the position determined
    by position.
    If simplex list are provided, it will compute the edges directly from these and not use adj,
    in which case N (the number of rows and columns of adj) is required.
    If simplex lists are not provided they will be calculated with for the flag complex whose type is determined by
    simplex_type as for simplex_counts.

    Parameters
    ----------
    adj : (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    dimensions : list of ints
        Dimensions `k` for which the `k`-skeleta is computed, if None all dimensions are computed.
    simplex_type : string
        Type of simplex to consider if computed from adj:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections
    simplex list: series
        Series 2d-arrays indexed by dimension.
        Each array is of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex.
        If provided adj will be ignored but N will be required.
    N: int
        Number of nodes in original graph.
    position: str
        Position of the edges to extract

        'all': all edges of the simplex

        'spine': edges along the spine of the simplex
        (only makes sense if simplices are directed)
    threads: int
        Number of threads into which the computation should be parallelized

    Returns
    -------
    dict
        Dictionary with keys dimensions and values boolean (N,N) matrices with entries `True`
        corresponding to edges in (maximal) simplices of that dimension.

    Raises
    ------
    AssertionError
        If neither adj nor simplex_list are provided
    AssertionError
        If N <= than an entry in the simplex list
    AssertionError
        If a dimension is required that is not an index in the simplex list

    Notes
    ------
    In order to list k-simplices and thus the k-skeleton, flagsercount needs to list all lower
    dimensional simplices anyhow.

    """

    # Check if enough inputs are provided
    if (position == "spine") and (simplex_list is not None):
        not_precomputed = False
    elif (position == "all") and (edge_par is not None):
        not_precomputed = False
    else:
        not_precomputed = True
    assert not (adj is None and not_precomputed), (
        "Either adj or simplex_list/edge_participation need to be provided"
    )

    # Determine dimensions
    if dimensions is None:
        max_dim = -1
    else:
        max_dim = np.max(np.array(dimensions))
    # Compute simplex list or edge particiption if not precomputed
    if not_precomputed:
        N = adj.shape[0]
        if position == "spine":  # Compute simplex since they are not provided
            simplex_list = list_simplices_by_dimension(
                adj,
                node_properties=None,
                max_simplices=max_simplices,
                max_dim=max_dim,
                simplex_type="directed",
                nodes=None,
                verbose=False,
                **kwargs,
            )
        elif (
            position == "all"
        ):  # More efficient than going from simplex lists if the position is not important
            edge_par, simplex_counts = edge_participation(
                adj,
                node_properties=None,
                max_simplices=max_simplices,
                threads=threads,
                max_dim=max_dim,
                simplex_type=simplex_type,
                return_simplex_counts=True,
            )
    else:
        if position == "spine":
            assert N > np.nanmax(simplex_list.explode().explode()), (
                "N must be larger than all the entries in the simplex list"
            )
    # Extract 'k'-skeleton
    if position == "spine":
        dims = simplex_list.index[
            simplex_list.index != 0
        ]  # Doesn't make sense to look at the 0-skeleton
    elif position == "all":
        dims = pd.Index(edge_par.drop(0, axis=1, errors="ignore").columns, name="dim")
    if dimensions is not None:
        dims = dims[np.isin(dims, dimensions)]
    skeleton_mats = {f"dimension_{dim}": None for dim in dims}
    if position == "spine":
        for dim in dims:
            if simplex_list[dim].size > 0:
                assert N > np.max(simplex_list[dim]), (
                    "N must be larger than all the entries in the simplex list"
                )
                skeleton_mats[f"dimension_{dim}"] = extract_submatrix_from_simplex_list(
                    simplex_list[dim], N, position=position
                )
    elif position == "all":
        skeleton_mats = extract_submatrices_from_edge_participation(
            edge_par, N, dims=dims, thresh=1
        )
    return skeleton_mats


def count_rc_edges_skeleta(
    adj=None,
    max_dim=-1,
    max_simplices=False,
    N=None,
    simplex_list=None,
    edge_par=None,
    simplex_type="directed",
    position="all",
    return_mats=False,
    threads=8,
    **kwargs,
):
    """Count the edges and reciprocal edges in the k-``skeleta`` of the directed flag complex of adj for all
    k<= max_dim. If simplex list are provided, it will compute the skeleta directly from these and not use adj.

    Parameters
    ----------
    adj : (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex list: series
        Series 2d-arrays indexed by dimension.
        Each array is of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex.
        If provided adj will be ignored but N will be required.
    edge_par: DataFrame
        DataFrame with edge participation values formatted as the output of ``edge_participation`` i.e.,
        indexed by the edges in adj and with columns de dimension for which edge participation is counted
    N: int
        Number of nodes in original graph.
    simple_type: str
        See :func:`connalysis.network.topology.simplex_counts`
    position: str
        Position of the edges to extract

        'all': all edges of the simplex

        'spine': edges along the spine of the simplex
        (only makes sense if simplices are directed)
    return_mats : bool
        If True return the matrices of the underlying graphs of the k-skeleta as in
        get_k_skeleta_graph
    threads: int
        Number of threads into which to parallelize the computation

    Returns
    -------
    data frame, (dict)
        data frame with index dimensions and columns number of (rc) edges in the corresponding skeleta
        if return_mats==True, also return the graphs of the k-skeleta as in get_k_skeleta_graph.

    Raises
    ------
    AssertionError
        If neither adj nor precomputed simplex_list or edge_par values are provided
    AssertionError
        If N <= than an entry in the simplex list

    See Also
    --------
    :func:`connalysis.network.topology.edge_participation` :
    A function that counts the number of times an edge is part of a simplex.

    :func:`connalysis.network.topology.simplex_counts` :
    A function that counts the simplices forming the complex from which bettis are count,
    where simplex types are described in detail.
    """

    # Check if enough inputs are provided
    if (position == "spine") and (simplex_list is not None):
        not_precomputed = False
    elif (position == "all") and (edge_par is not None):
        not_precomputed = False
    else:
        not_precomputed = True

    assert not (adj is None and not_precomputed), (
        "Either adj or simplex_list/edge_participation need to be provided"
    )

    if not_precomputed:
        N = adj.shape[0]
        if position == "spine":  # Compute simplex since they are not provided
            simplex_list = list_simplices_by_dimension(
                adj,
                node_properties=None,
                max_simplices=max_simplices,
                max_dim=max_dim,
                simplex_type="directed",
                nodes=None,
                verbose=False,
                **kwargs,
            )
        elif (
            position == "all"
        ):  # More efficient than going from simplex lists if the position is not important
            edge_par, simplex_counts = edge_participation(
                adj,
                node_properties=None,
                max_simplices=max_simplices,
                threads=threads,
                max_dim=max_dim,
                simplex_type=simplex_type,
                return_simplex_counts=True,
            )

    else:
        if position == "spine":
            assert N > np.nanmax(simplex_list.explode().explode()), (
                "N must be larger than all the entries in the simplex list"
            )

    # Extract 'k'-skeleton and count (rc-)edges
    if position == "spine":
        dims = simplex_list.index[
            simplex_list.index != 0
        ]  # Doesn't make sense to look at the 0-skeleton
    elif position == "all":
        dims = pd.Index(edge_par.drop(0, axis=1, errors="ignore").columns, name="dim")
    edge_counts = pd.DataFrame(
        index=dims, columns=["number_of_edges", "number_of_rc_edges", "rc/edges_percent"]
    )
    if return_mats:
        skeleton_mats = {f"dimension_{dim}": None for dim in dims}

    if position == "spine":
        for dim in dims:
            if simplex_list[dim].size > 0:
                assert N > np.max(simplex_list[dim]), (
                    "N must be larger than all the entries in the simplex list"
                )
                mat = extract_submatrix_from_simplex_list(simplex_list[dim], N, position=position)
                edges = mat.sum()
                rc_edges = rc_submatrix(mat).sum()
                edge_counts.loc[dim, "number_of_edges"] = edges
                edge_counts.loc[dim, "number_of_rc_edges"] = rc_edges
                edge_counts.loc[dim, "rc/edges_percent"] = (rc_edges * 100) / edges
            else:
                edge_counts.loc[dim, "number_of_edges"] = 0
            if return_mats:
                skeleton_mats[f"dimension_{dim}"] = mat
    elif position == "all":
        skeleton_mats = extract_submatrices_from_edge_participation(
            edge_par, N, dims=dims, thresh=1
        )
        for dim in dims:
            edges = skeleton_mats[f"dimension_{dim}"].sum()
            edge_counts.loc[dim, "number_of_edges"] = edges
            if edges > 0:
                rc_edges = rc_submatrix(skeleton_mats[f"dimension_{dim}"]).sum()
                edge_counts.loc[dim, "number_of_rc_edges"] = rc_edges
                edge_counts.loc[dim, "rc/edges_percent"] = (rc_edges * 100) / edges
            else:
                edge_counts.loc[dim, "number_of_rc_edges"] = 0
                edge_counts.loc[dim, "rc/edges_percent"] = 0

    if return_mats:
        return edge_counts, skeleton_mats
    else:
        return edge_counts


def bedge_counts(
    adjacency, simplices=None, max_simplices=False, max_dim=-1, simplex_type="directed", **kwargs
):
    r"""Counts the total number of edges in each position on the subgraphs defined by the nodes
    of the simplices listed in simplices.  If a simplex list is not passed, simplices are computed on
    the adjacency matrix.

    Parameters
    ----------
    adjacency : (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    simplices : series
        Series  of 2d-arrays indexed by dimension.
        Each array is of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type: str
        See :func:`connalysis.network.topology.simplex_counts`

    Returns
    -------
    series
        pandas series with index dimensions values (dim+1, dim+1) arrays.  The (i,j) entry counts the number of edges
        from node i to node j on all the subgraphs of adjacency on the nodes of the simplices listed.  See notes.

    Notes
    -------
    Every directed $k$-simplex $[v_o, v_1, \ldots, v_k]$ defines as subgraph of the adjacency matrix, with edges
    $v_i \to v_j$ whenever $i\leq j$, but also possibly with ''reverse'' edges.  One can represent this structure
    with a non-symmetric $(k+1, k+1)$-matrix with `1`'s for every edge in the subgraph.  The output of this function
    gives for each dimension the sum of all these matrices over all the simplices provided in `simplices` or over
    all the simplices in the adjacency matrix if none is provided.  The lower triangular part of these matrices is
    therefore a metric of recurrence within simplices, or "higher dimensional recurrence".
    In particular, in dimension 1 it is the number of reciprocal edges in the network.
    """

    adj = adjacency

    if simplices is None:
        LOG.info("COMPUTE `bedge_counts(...)`: No argued simplices.")
        return bedge_counts(
            adj,
            list_simplices_by_dimension(
                adj,
                max_simplices=max_simplices,
                max_dim=max_dim,
                simplex_type=simplex_type,
                **kwargs,
            ),
        )
    LOG.info("COMPUTE `bedge_counts(...): for simplices: %s ", simplices.shape)

    # Weights are truncated to integers, as when the matrix was converted to a dense integer array
    counts = sp.csr_matrix(adjacency).astype(int)

    def count_bedges(d_simplices):
        if d_simplices is None or d_simplices.shape[1] == 1:
            return np.nan
        if len(d_simplices) == 0:
            return pd.DataFrame(d_simplices, columns=range(d_simplices.shape[1])).sum()
        d_simplices = np.asarray(d_simplices, dtype=np.int64)
        # Entry (i, j) adds up the edges from the j-th to the i-th vertex of every simplex
        size = d_simplices.shape[1]
        total = np.empty((size, size), dtype=np.int64)
        for i in range(size):
            for j in range(size):
                total[i, j] = counts[d_simplices[:, j], d_simplices[:, i]].sum()
        return total

    results = np.empty(len(simplices), dtype=object)
    for position, d_simplices in enumerate(simplices):
        results[position] = count_bedges(d_simplices)
    return pd.Series(results, index=simplices.index, name=simplices.name).infer_objects()


# TRIAD ANALYSIS
label_edges = np.arange(3 * 3).reshape(
    3, 3
)  # Indexing edges on a 3x3 matrix going from left to right and up to down


def _get_triad_id(edges):
    """Given a the edges on a graph on nodes {0,1,2}
    return a list of edges indexed from 0 to 8 as in label_edges.

    Parameters
    ----------
    edges : tuple of pairs
        Each pair is of the form $(i,j)$ where $i, j \\in \\{0,1,2}$.

    Returns
    -------
    list of integers between 0 and 8 indexing the edges as in label_edges
    """
    row, col = tuple(zip(*edges))
    return tuple(np.sort(label_edges[row, col]))


# We list all connected triads by hand as sort them as in Gal et al., 2017
connected_triads = {
    # On two edges
    0: ((0, 2), (1, 0)),
    1: ((0, 2), (1, 2)),
    2: ((0, 1), (0, 2)),
    # On three edges
    3: ((0, 1), (1, 2), (0, 2)),
    4: ((0, 1), (0, 2), (1, 0)),
    5: ((0, 1), (1, 0), (2, 0)),
    6: ((0, 1), (1, 2), (2, 0)),
    # On four edges
    7: ((0, 1), (0, 2), (1, 0), (2, 0)),
    8: ((0, 1), (0, 2), (1, 0), (2, 1)),
    9: ((0, 1), (0, 2), (1, 0), (1, 2)),
    10: ((0, 1), (0, 2), (1, 2), (2, 1)),
    # On 5 edge
    11: ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0)),
    # On 6 edges
    12: ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
}

# Exhaustive dictionary of triads (digraphs on 3 nodes) represented by their list of edges as indexed in
# label_edges.  The 3-cycle needs to be entered twice because all edges have the same in-out degree but there
# are two graphs in the isomorphism class.
triad_dict = {_get_triad_id(connected_triads[i]): i for i in range(13)}
triad_dict[(2, 3, 7)] = 6
# Size of isomorphism class of each triad type i.e., number of permutation of the vertices giving the same graph
triad_combinations = np.array(
    [
        6,
        3,
        3,  # 2 edges
        6,
        6,
        6,
        2,  # 3 edges
        3,
        6,
        3,
        3,  # 4 edges
        6,  # 5 edges
        1,
    ]
)  # 3-clique


def _triad_table():
    """Motif id of each 3x3 adjacency pattern (its nine entries read as bits); -1 where it is no connected triad."""
    table = np.full(512, -1, dtype=np.int64)
    for edges, motif in triad_dict.items():
        table[sum(1 << int(edge) for edge in edges)] = motif
    return table


_TRIAD_TABLE = _triad_table()


def _classify_triads(adj, triples):
    """Motif ids of the triads on ``triples``, an (n, 3) array of node ids, ascending in every row.

    The nodes of each triad are sorted by (out-degree, in-degree) within the triad before the pattern of its
    3x3 sub-matrix is looked up, as in Gal et al., 2017.
    """
    n = len(triples)
    sub = np.empty((n, 3, 3), dtype=adj.dtype)
    for i in range(3):
        for j in range(3):
            sub[:, i, j] = np.asarray(adj[triples[:, i], triples[:, j]]).ravel()
    order = np.argsort(-sub.sum(axis=1) - 10 * sub.sum(axis=2), axis=1, kind="stable")
    ordered = sub[np.arange(n)[:, None, None], order[:, :, None], order[:, None, :]]
    motifs = _TRIAD_TABLE[(ordered != 0).reshape(n, 9) @ (1 << np.arange(9))]
    if (motifs < 0).any():
        raise KeyError(tuple(triples[np.argmax(motifs < 0)]))
    return motifs


def _connected_triples(undirected, max_wedges=1 << 22):
    """Yield (n, 3) arrays with the sorted nodes of every connected triple of a symmetric boolean matrix, once each.

    A triple is connected when one of its nodes (the centre) is adjacent to the other two.  Every such pair of
    neighbours is generated at its centre; a triangle has three centres and is kept at its smallest node only.
    """
    indptr, indices = undirected.indptr.astype(np.int64), undirected.indices
    degree = np.diff(indptr)
    last_wedge = np.cumsum(degree * (degree - 1) // 2)
    start = 0
    while start < len(degree):
        before = last_wedge[start - 1] if start else 0
        stop = max(int(np.searchsorted(last_wedge, before + max_wedges, side="right")), start + 1)
        entries = np.arange(indptr[start], indptr[stop])
        centre = np.repeat(np.arange(start, stop), degree[start:stop])
        later = indptr[centre + 1] - 1 - entries  # neighbours after this one in the same row
        total = int(later.sum())
        if total:
            first = np.repeat(entries, later)
            second = first + 1 + np.arange(total) - np.repeat(np.cumsum(later) - later, later)
            x, y, z = (
                indices[first].astype(np.int64),
                indices[second].astype(np.int64),
                np.repeat(centre, later),
            )
            keep = ~np.asarray(undirected[x, y]).ravel().astype(bool) | (z < x)
            yield np.sort(np.stack([x[keep], y[keep], z[keep]], axis=1), axis=1)
        start = stop


def _triples_in_original_order(undirected, max_walks=1 << 22):
    """All connected triples as an (n, 3) array, in the order ``list(set)`` gave them in the original implementation.

    That version added the sorted triple of every (x < y, common neighbour z), by increasing x, y and z, to a
    Python set and sampled from ``list(set)``.  A set's order depends on the order of insertion, so it is
    reproduced here to draw the same sample for the same random seed.
    """
    indptr, indices = undirected.indptr.astype(np.int64), undirected.indices
    degree = np.diff(indptr)
    walk_end = np.cumsum(undirected.dot(degree.astype(np.int64)))  # steps x -> z -> y out of each x
    n_nodes = len(degree)
    triples = set()
    start = 0
    while start < n_nodes:
        before = walk_end[start - 1] if start else 0
        stop = max(int(np.searchsorted(walk_end, before + max_walks, side="right")), start + 1)
        entries = np.arange(indptr[start], indptr[stop])
        x = np.repeat(np.arange(start, stop), degree[start:stop])
        z = indices[entries].astype(np.int64)
        fan = degree[z]
        total = int(fan.sum())
        if total:
            second = (
                np.repeat(indptr[z], fan) + np.arange(total) - np.repeat(np.cumsum(fan) - fan, fan)
            )
            x, y, z = np.repeat(x, fan), indices[second].astype(np.int64), np.repeat(z, fan)
            keep = y > x
            x, y, z = x[keep], y[keep], z[keep]
            order = np.lexsort((z, y, x))
            canonical = np.sort(np.stack([x[order], y[order], z[order]], axis=1), axis=1)
            codes = (canonical[:, 0] * n_nodes + canonical[:, 1]) * n_nodes + canonical[:, 2]
            _, first_seen = np.unique(codes, return_index=True)
            canonical = canonical[np.sort(first_seen)]
            triples.update(
                zip(canonical[:, 0].tolist(), canonical[:, 1].tolist(), canonical[:, 2].tolist())
            )
        start = stop
    return np.array(list(triples), dtype=np.int64).reshape(-1, 3)


def count_triads_fully_connected(adj, max_num_sampled=5000000, return_normalized=False):
    """Counts the numbers of each triadic motif in the matrix adj.

    Parameters
    ----------
    adj : 2d-array
        Adjacency matrix of a directed network.
    max_num_sampled : int
        The maximal number of connected triads classified. If the number of
        connected triads is higher than that, only the specified number is sampled at random and
        classified. The final counts are extrapolated as (actual_num_triads/ max_num_sampled) * counts.
    return_normalized : bool
        If True return the triad counts divided by the size of each isomorphism class.  That is, the total counts
        divided by the following array:

        $[6, 3, 3, 6, 6, 6, 2, 3, 6, 3, 3, 6, 1].$

    Returns
    -------
    1d-array
        The counts of the various triadic motifs in adj as ordered in Figure 5 [1]_.

    Notes
    ------
    Only connectected motifs are counted, i.e. motifs with less than 2 connections or only a single bidirectional
    connection are not counted. The connected motifs are ordered as in Figure 5 [1]_.

    References
    -------

    ..[1] Gal, Eyal, et al.
    ["Rich cell-type-specific network topology in neocortical microcircuitry."](https://www.nature.com/articles/nn.4576)
    Nature neuroscience 20.7 (2017): 1004-1013.

    """

    adj = sp.csr_matrix(adj)
    t0 = time.time()
    undirected = underlying_undirected_matrix(adj)
    undirected.eliminate_zeros()
    undirected.sort_indices()
    kept, n_triads = [], 0
    for triples in _connected_triples(undirected):
        n_triads += len(triples)
        if kept is not None:
            kept.append(triples)
            if n_triads > max_num_sampled:
                kept = None  # too many to classify them all: only keep counting
    LOG.info("Time spent finding triads: %s", time.time() - t0)
    LOG.info("Found %s connected triads", n_triads)

    t0 = time.time()
    counts = np.zeros(np.max(list(triad_dict.values())) + 1)
    # Always drawn, so that the global random state moves on as it always did
    sample_idx = np.random.choice(n_triads, np.minimum(max_num_sampled, n_triads), replace=False)
    if n_triads <= max_num_sampled:
        triples = np.concatenate(kept) if kept else np.empty((0, 3), dtype=np.int64)
    else:
        triples = _triples_in_original_order(undirected)[sample_idx]
    for block in range(0, len(triples), 1 << 20):
        counts += np.bincount(
            _classify_triads(adj, triples[block : block + (1 << 20)]), minlength=len(counts)
        )
    LOG.info("Time spent classifying triads: %s", time.time() - t0)
    if return_normalized:
        return (((n_triads / len(sample_idx)) * counts).astype(int)) / triad_combinations
    else:
        return ((n_triads / len(sample_idx)) * counts).astype(int)


def _convex_hull(adj, node_properties):  # --> topology
    """Return the convex hull of the sub gids in the 3D space using x,y,z position for gids"""
    pass


def get_all_simplices_from_max(max_simplices):
    """Takes the list of maximal simplices are returns the list of all simplices.

    Parameters
    ----------
    max_simplices : list
        A list of lists of tuples. Where max_simplices[k] is a list of the 0 dimensional maximal simplices,
        where each simplex is a tuple of the vertices of the simplex

    Returns
    -------
    list
        A list of lists of tuples. Of the same format as the inputted list but now contains all simplices.
    """
    simplices = list(max_simplices)
    for k in range(len(max_simplices) - 1, 0, -1):
        try:
            known = set(simplices[k - 1])
        except TypeError:  # simplices that cannot be hashed, e.g. lists: search the list itself
            known = simplices[k - 1]
        for simplex in simplices[k]:
            for s in range(k, -1, -1):
                face = tuple(simplex[:s] + simplex[s + 1 :])
                if face not in known:
                    simplices[k - 1].append(face)
                    if isinstance(known, set):
                        known.add(face)

    return simplices


def simplicial_rich_club_curve(M, maximal=False, sparse_bin_set=False):
    """Computes the simplicial rich club curve of a network.
       Where the i'th entry is the density of the subnetwork induced by the vertices that are contained in
       more than i (maximal) simplices.

    Parameters
    ----------
    adj : 2d-array
        Adjacency matrix of a directed network.
    max_simplices : bool
        If true then vertex participation is the number of maximal simplices each vertex is contained in.
    sparse_bin_set : bool
        If true then consecutive entries with same rich club coefficient are grouped into bins together,

    Returns
    -------
    pandas.Series
        Where the i'th entry is the rich club coefficient of the network induced by all vertices which are
        contained in more that i (maximal) simplices

    """
    import pyflagsercount

    from .classic import efficient_rich_club_curve

    M = sp.csr_matrix(M)
    contain_counts = pyflagsercount.flagser_count(M, max_simplices=maximal, containment=True)[
        "contain_counts"
    ]
    vertex_par = pd.DataFrame(_pad_rows(contain_counts))
    curves = [
        efficient_rich_club_curve(
            M, pre_calculated_filtration=vertex_par[i], sparse_bin_set=sparse_bin_set
        )
        for i in range(vertex_par.shape[1])
    ]
    return pd.DataFrame(curves).transpose().dropna(how="all")


#######################################################
################# METHODS FROM TRIDY ##################
#######################################################


def euler_characteristic(matrix):
    """Computes the Euler characteristic of the flag complex of the graph with adjacency matrix matrix

    Parameters
    ----------
    matrix : 2d-array
        Adjacency matrix of a directed network.

    Returns
    -------
    integer
        The Euler characteristic of the flag complex of matrix

    """
    return _flagser_counts(matrix)["euler"]


def tcc(matrix, center=0):
    """Computes the transitive clustering coefficient of the graph induced by
    the neighbourhood of center ``center`` in the matrix

    Parameters
    ----------

    matrix : 2d-array
        Adjacency matrix of a directed network.
    center : int
        The index of the vertex to be considered, default=0

    Returns
    -------
    float
        The transitive cluster coefficient of the neighbourhood of center

    """
    from .classic import reciprocal_connections

    outdeg = np.count_nonzero(matrix[center])
    indeg = np.count_nonzero(np.transpose(matrix)[center])
    repdeg = reciprocal_connections(matrix, chief_only=True, center=center)
    totdeg = indeg + outdeg

    # If matrix is not already neighbourhood of center, then compute neighbourhood.
    if totdeg - repdeg + 1 != len(matrix):
        matrix = neighbourhood(center, matrix)
        center = 0

    chief_containment = node_participation(matrix, max_dim=2).iloc[0]
    numerator = 0 if len(chief_containment) < 3 else chief_containment[2]
    denominator = totdeg * (totdeg - 1) - (indeg * outdeg + repdeg)

    if denominator == 0:
        return 0
    return numerator / denominator


def dc(matrix, center=0, coeff_index=2, nhbd=True):
    """Computes the density coefficient of the graph induced by
    the neighbourhood of center ``center`` in the matrix

    Parameters
    ----------

    matrix : 2d-array
        Adjacency matrix of a directed network.
    center : int
        The index of the vertex whose neighbourhood is to be considered, default=0
    coeff_index : int
         The dimension to be computed, default=2
    nhbd : bool
         If true then is assumed that matrix is the neighbourhood of center.
         If false the neighbourhood of center is computed and used, default= True.

    Returns
    -------
    float
        The density coefficient of the neighbourhood of center

    """
    assert coeff_index >= 2, "Assertion error: Density coefficient must be at least 2"
    if not nhbd:
        matrix = neighbourhood(center, matrix)
        center = 0

    flagser_output = node_participation(matrix, max_dim=coeff_index).iloc[center]
    if len(flagser_output) <= coeff_index:
        density_coeff = 0
    elif flagser_output[coeff_index] == 0:
        density_coeff = 0
    else:
        numerator = coeff_index * flagser_output[coeff_index]
        denominator = (
            (coeff_index + 1) * (len(matrix) - coeff_index) * flagser_output[coeff_index - 1]
        )
        if denominator == 0:
            density_coeff = 0
        else:
            density_coeff = numerator / denominator
    return density_coeff


# I have kept this for now for backwards compatability, in case it is used anywhere in existing code
def normalised_simplex_count(matrix, dim=2):
    """Computes the normalised simplex count of the flag complex of the graph with adjacency matrix matrix

    Parameters
    ----------
    matrix : 2d-array
        Adjacency matrix of a directed network.
    dim : int
         The dimension to be computed, default=2

    Returns
    -------
    float
        The normalised simplex count of dimension dim of the flag complex of matrix

    """
    return normalized_simplex_counts(matrix)[dim]


def nbc(matrix):
    """Computes the normalised Betti coefficient of the graph with adjacency matrix matrix

    Parameters
    ----------
    matrix : 2d-array
        Adjacency matrix of a directed network.

    Returns
    -------
    float
        The normalised Betti coefficient of the graph with adjacency matrix matrix

    """
    from flagserpy import flagser_unweighted

    flagser_output = flagser_unweighted(matrix, directed=True)
    cells = flagser_output["cell_count"]
    bettis = flagser_output["betti"]
    while (cells[-1] == 0) and (len(cells) > 1):
        cells = cells[:-1]
    while (bettis[-1] == 0) and (len(bettis) > 1):
        bettis = bettis[:-1]
    normalized_betti_list = [
        (i + 1) * bettis[i] / cells[i] for i in range(min(len(bettis), len(cells)))
    ]
    return sum(normalized_betti_list)
