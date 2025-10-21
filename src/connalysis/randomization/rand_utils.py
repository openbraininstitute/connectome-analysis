# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import scipy.sparse as sp
import pandas as pd

from typing import List, Union
from connalysis.network.topology import rc_submatrix


def subset_matrix(sparse_matrix: sp.coo_matrix, selection: np.ndarray) -> sp.coo_matrix:
    """
    Returns a subset of the matrix given a selection of edges.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param selection: Array with indices corresponding to edges to select from coo.
    :type: np.ndarray

    :return subset_matrix: Matrix containing edges specified by selection.
    :rtype: sp.coo_matrix
    """
    return sp.coo_matrix((np.ones(len(selection)), (sparse_matrix.row[selection], sparse_matrix.col[selection])),
                         shape=(sparse_matrix.shape))


def subsampled_matrix(sparse_matrix: sp.coo_matrix, n: int, generator: np.random.Generator) -> sp.coo_matrix:
    """
    Returns a random subsample of the matrix of a given size.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param n: Size of the subsample.
    :type: int
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return subsample_matrix: Matrix containing n edges.
    :rtype: sp.coo_matrix
    """
    selection = generator.choice(len(sparse_matrix.col), replace=False, size=n)
    return subset_matrix(sparse_matrix, selection)


def non_overlapping_subsampled_matrices(sparse_matrix: sp.coo_matrix, ns: List[int],
                                        generator: np.random.Generator) -> List[sp.coo_matrix]:
    """Function to retrieve an arbitrary amount of non-overlapping subsampled matrices of given edge count.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param ns: Sizes of the subsamples.
    :type: List[int]
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return subsample_matrices: Matrices containing ns edges.
    :rtype: List[sp.coo_matrix]
    """
    selection = generator.choice(len(sparse_matrix.col), replace=False, size=sum(ns))  # Samples some edges
    ls = np.append(0, np.cumsum(ns))
    return [subset_matrix(sparse_matrix, selection[ls[i]:ls[i + 1]]) for i in range(len(ls) - 1)]


def half_matrix(sparse_matrix: sp.coo_matrix, generator: np.random.Generator) -> List[sp.coo_matrix]:
    """
    Split the matrix edges exactly in half. Returns one matrix per half.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return subsample_matrices: List of two matrices, one per half.
    :rtype: List[sp.coo_matrix]
    """
    permutation = generator.permutation(len(sparse_matrix.col))
    return subset_matrix(sparse_matrix, permutation[:int(len(permutation) / 2)]), subset_matrix(sparse_matrix,
                                                                                                permutation[int(len(
                                                                                                    permutation) / 2):])


def adjust_bidirectional_connections(sparse_matrix: sp.csc_matrix, bedges_to_add: int,
                                     generator: np.random.Generator) -> sp.csc_matrix:
    """
    Turn a fixed amount of directional connections into bidirectional connections.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param bedges_to_add: Number of directional edges to transform into bedges.
    :type: int
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return adjusted_matrix: Matrix with adjusted bedges
    :rtype: sp.csc_matrix
    """
    bedge = rc_submatrix(sparse_matrix)
    dedge = sparse_matrix - bedge
    del bedge
    dedge_coo = dedge.tocoo(copy=False)  # Easier to subsample
    dedges_to_bedge, dedges_to_remove = non_overlapping_subsampled_matrices(dedge_coo,
                                                                            [bedges_to_add, bedges_to_add],
                                                                            generator)  # Coo matrices are auto casted before sum
    return sparse_matrix + dedges_to_bedge.T - dedges_to_remove


def add_bidirectional_connections(sparse_matrix: sp.csc_matrix, bedges_to_add: int,
                                  generator: np.random.Generator) -> sp.csc_matrix:
    """
    Turn a fixed amount of directional connections into bidirectional connections.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param bedges_to_add: Number bedges to add.
    :type: int
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return adjusted_matrix: Matrix with adjusted bedges.
    :rtype: sp.csc_matrix
    """
    bedge = rc_submatrix(sparse_matrix)
    dedge = sparse_matrix - bedge
    del bedge
    dedge = dedge.tocoo(copy=False)  # Easier to subsample
    dedge = subsampled_matrix(dedge, bedges_to_add, generator)
    return sparse_matrix + dedge.T


def _evaluate_probs_less_random(p_mat, adjust=None):
    """
    Evaluates a single spread step using the `less random` method.

    Parameters
    ----------
    p_mat : sparse.matrix
        Represents the current state of the spread. Each row corresponds to
        a node to spread from. Columns that we can spread to have positive entries, columns that we already
        spread to or that are excluded on account of being candidates from earlier steps have negative values.
        All others are zero.
    adjust : numpy.array
        Optional array of weight vectors. Scales the values of p_mat of the corresponding rows. In the manuscript
        (see above) this is 1 / r.
    """
    p_mat = p_mat.tocsr()
    n_pick = np.array(p_mat.sum(axis=1))[:, 0]
    if adjust is not None:
        n_pick = n_pick * adjust
    n_pick = np.round(n_pick).astype(int)

    indptr_out = [0]
    picked = []
    for a, b, c in zip(p_mat.indptr[:-1], p_mat.indptr[1:], n_pick):
        p = p_mat.data[a:b] / p_mat.data[a:b].sum()
        c = np.minimum(c, (p > 0).sum())
        indptr_out.append(c)
        if c > 0:
            picked.append(np.random.choice(p_mat.indices[a:b], c, p=p, replace=False))
    picked = np.hstack(picked)
    indptr_out = np.cumsum(indptr_out)
    m_out = sp.csr_matrix((np.ones(indptr_out[-1], dtype=bool), 
                            np.hstack(picked),
                            indptr_out),
                            shape=p_mat.shape).tocoo()
    return m_out


def _evaluate_probs(p_mat, adjust=None, less_random=False):
    """
    Evaluates a single spread step when building a stochastic spread graph (https://doi.org/10.1101/2025.08.21.671478)

    Parameters
    ----------
    p_mat : sparse.matrix
        Represents the current state of the spread. Each row corresponds to
        a node to spread from. Columns that we can spread to have positive entries, columns that we already
        spread to or that are excluded on account of being candidates from earlier steps have negative values.
        All others are zero.
    adjust : numpy.array
        Optional array of weight vectors. Scales the values of p_mat of the corresponding rows. In the manuscript
        (see above) this is 1 / r.
    less_random : bool
        If set to true, then the less stochastic version of the spread is used. That is, for each source node 
        (row in p_mat) the process spreads to exactly the expected number of nodes instead of a random number.
    """
    if less_random:
        return _evaluate_probs_less_random(p_mat, adjust=adjust)
    p_mat = p_mat.tocoo()
    thresh = p_mat.data
    if adjust is not None:
        thresh = thresh * adjust[p_mat.row]
    _v = np.random.rand(p_mat.nnz) < thresh
    return sp.coo_matrix((np.ones(_v.sum()), (p_mat.row[_v], p_mat.col[_v])), shape=p_mat.shape)

def _connection_df_to_csc_matrix(w, mirror=True, shape=None):
    """
    Transforms a representation of a graph as a DataFrame of edges to a scipy.sparse.csc_matrix.
    Used for the generation of random geometric graphs.

    Parameters
    ----------
    w : pandas.DataFrame
        Each row corresponds to a connection, columns "neuron" and "i" specify the source and target
        vertex.
    mirror : bool
        If set to True, the output matrix is made symmetrical with the following strategy: If a connection
        from vertex i to j exists, a connection is also placed from j to i if it does not already exist.
    shape : tuple
        The shape of the output matrix. If not provided, it will be guessed based on the indices in `w`.
    """
    if shape is None:
        raise ValueError("Must provide shape")
    indices = w.index.to_frame().reset_index(drop=True)
    idy = indices["neuron"].values
    idx = indices["i"].values
    w = w.values
    if mirror:
        _idx = np.hstack([idx, idy])
        _idy = np.hstack([idy, idx])
        _w = np.hstack([w, w])
        M = sp.coo_matrix((_w, (_idy, _idx)),
                              shape=shape).tocsc()
    else:
        M = sp.coo_matrix((w, (idy, idx)),
                              shape=shape).tocsc()
    return M

def _direction_or_distance_dependent_w(pts_src, pts_tgt, idx,
                                       directionality_fac=0, directionality_axis=None,
                                       distance_func=None):
    """
    Calculates weights for potential edges based on distance or direction between vertices.

    Parameters
    ----------
    pts_src : numpy.array
        Shape: (n x 3). Locations of vertices of potential source nodes.
    pts_tgt : numpy.array
        Shape: (m x 3). Locations of vertices of potential target nodes. `pts_src` and `pts_tgt` can be
        the same.
    idx : numpy.array
        Shape: (n, ). Specifies the locations of potential edges. The array has one entry per vertex in `pts_src`.
        Each entry is a list of indices into `pts_tgt` that specifies the potential targets of that source vertex.
        This corresponds to the output obtained from querying a scipy.spatial.KDTree.
    directionality_axis : numpy.array
        Shape: (3, ) This introduces a directionality bias, i.e., the calculated weight is larger if the direction 
        of a potential edge aligns with a specified vector and less likely if the direction is in the opposite
        direction of the vector. This is calculated as the dot product of the direction vector of the potential
        connection with  `directionality_axis`. 
    directionality_fac : float
        Must be between -1 and 1. This specifies how much the dot product calculated using `directionality_axis`
        (see above) is weighed to calculate a bias. Formally:
        w = (delta o directionality_axis) * directionality_fac + 1,
        where o denotes the vector dot product, delta is the vector from the source to the target vertex of a
        potential connection. 
        Directionality bias cannot be combined with distance bias (see below).
    distance_func : function
        A function that is to be evaluated on pairwise distances of candidate pairs. The function takes a
        distance as input and returns a relative weight. 
        Cannot be combined with a directionality bias (see above).
    """
    if distance_func is None:
        distance_func = lambda _x: 1.0
    A = pd.DataFrame(pts_src[:, :3], columns=pd.Index(["x", "y", "z"], name="coord"),
                                    index=pd.Index(range(len(pts_src)), name="neuron"))
    B = pd.concat([pd.DataFrame(pts_tgt[_idx, :3], index=pd.Index(_idx, name="i"),
                    columns=pd.Index(["x", "y", "z"], name="coord"))
                for _idx in idx],
                axis=0, keys=range(len(idx)), names=["neuron"])
    ab_diff = A - B
    
    pairw_D = pd.Series(np.linalg.norm(ab_diff, axis=1), index=ab_diff.index)

    if directionality_axis is not None:
        directionality_axis = np.array(directionality_axis).reshape((-1, 1))
        align = pd.Series(np.dot(ab_diff, directionality_axis)[:, 0],
                              index=ab_diff.index) / pairw_D
        return (align * directionality_fac + 1) * pairw_D.apply(distance_func)

    return pairw_D.apply(distance_func)

def _evaluate_per_node_weights(w_out, w_in, idx):
    """
    Calculates weights for potential edges, based on weights specified per vertex for outgoing 
    and/or incoming edges.

    Parameters
    ----------
    w_out : numpy.array
        Shape: (n, ). One entry per potential source vertex. Edges originating from a vertex will be
        weighted with the weight in the correponding entry of `w_out`.
    w_in : numpy.array
        Shape: (m, ). One entry per potential target vertex. Edges connecting to a vertex will be
        weighted with the weight in the correponding entry of `w_in`. This is multiplied with the weight
        calculated from `w_out`.
    idx : numpy.array
        Shape: (n, ). Specifies the locations of potential edges. The array has one entry per source vertex,
        i.e., the same length as `w_out`.
        Each entry is a list of indices into `w_in` that specifies the potential targets of that source vertex.
        This corresponds to the output obtained from querying a scipy.spatial.KDTree.
    """
    w_out = np.array(w_out)
    w_in = np.array(w_in)

    W = [pd.Series(_w_o * w_in[_idx], name="weight",
                       index=pd.Index(_idx, name="i"))
         for _w_o, _idx in zip(w_out, idx)]
    W = pd.concat(W, axis=0, keys=range(len(idx)), names=["neuron"])
    return W
