"""The functions of ``topology`` taken over from the Tridy code (``dc``, ``tcc``, ``simplicial_rich_club_curve``) and the
unused binary simplex decoder ``_binary2simplex``.

``tcc`` and ``simplicial_rich_club_curve`` cannot run in the original (an import of a name that does not exist, a keyword
the called function does not have), so their cases record the exception and are tagged ``fixed``.
"""

import os
import tempfile

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import tutorial_connectome

from cases.topology_helpers import complete, cycle, from_edges, rich, structured, transitive
from connalysis.network import topology as T


def _dense():
    """14 nodes, counts (14, 55, 67, 36, 8); used as a ready-made 'neighbourhood' matrix."""
    return rich(14, 0.12, seed=10, sizes=(5, 4), rc_sizes=(3,)).toarray()


# ----------------------------------------------------------------------------- dc


@case("topology.dc.default_matrix_is_the_neighbourhood")
def _():
    return T.dc(_dense())


@case("topology.dc.coeff_index_values")
def _():
    dense = _dense()
    return {
        (ci, nhbd): T.dc(dense, coeff_index=ci, nhbd=nhbd)
        for ci in (2, 3, 4, 5, 6)
        for nhbd in (True, False)
    }


@case("topology.dc.neighbourhood_of_other_centers")
def _():
    dense = _dense()
    return {c: T.dc(dense, center=c, nhbd=False) for c in (0, 1, 3, 7, 12)}


@case("topology.dc.centers_of_a_given_neighbourhood_matrix")
def _():
    # nhbd=True: the matrix is taken as the neighbourhood and the participation of row ``center`` is used
    dense = _dense()
    return {c: T.dc(dense, center=c) for c in (0, 2, 5, 9, 13)}


@case("topology.dc.coeff_index_with_neighbourhood_of_other_centers")
def _():
    dense = _dense()
    return {
        (c, ci): T.dc(dense, center=c, coeff_index=ci, nhbd=False)
        for c in (1, 7)
        for ci in (2, 3, 4)
    }


@case("topology.dc.dtypes")
def _():
    dense = _dense()
    return {
        np.dtype(d).name: T.dc(dense.astype(d), center=1, nhbd=False)
        for d in (bool, np.int8, np.int64, np.float64)
    }


@case("topology.dc.no_simplex_of_that_dimension_gives_zero")
def _():
    path = from_edges(4, [(0, 1), (1, 2), (2, 3)]).toarray()
    return {
        "path": T.dc(path),
        "path_other_center": T.dc(path, center=2),
        "transitive_triangle": T.dc(transitive(3).toarray()),
        "transitive_triangle_dim3": T.dc(transitive(3).toarray(), coeff_index=3),
        "single_node": T.dc(np.zeros((1, 1), dtype=bool)),
        "two_nodes": T.dc(from_edges(2, [(0, 1)]).toarray()),
    }


@case("topology.dc.complete_and_cycle_graphs")
def _():
    return {
        "complete_5": T.dc(complete(5).toarray()),
        "complete_5_dim3": T.dc(complete(5).toarray(), coeff_index=3),
        "cycle_5": T.dc(cycle(5).toarray()),
        "structured_hub": T.dc(structured().toarray(), center=18, nhbd=False),
    }


@case("topology.dc.coeff_index_one_rejected")
def _():
    return T.dc(_dense(), coeff_index=1)


@case("topology.dc.coeff_index_zero_rejected")
def _():
    return T.dc(_dense(), coeff_index=0)


@case("topology.dc.center_outside_the_matrix")
def _():
    return T.dc(_dense(), center=40)


@case("topology.dc.sparse_matrix_with_neighbourhood_flag")
def _():
    # len() of a sparse matrix is ambiguous
    return T.dc(sp.csr_matrix(_dense()))


@case("topology.dc.sparse_matrix_with_neighbourhood_computed")
def _():
    # neighbourhood() does not understand sparse matrices: it ends with a 2 x 2 empty matrix and a coefficient 0
    return T.dc(sp.csr_matrix(_dense()), nhbd=False)


@case("topology.dc.real_neighbourhoods")
def _():
    adj, _ = tutorial_connectome(300)
    dense = adj.toarray()
    return {c: T.dc(dense, center=c, nhbd=False) for c in (5, 40, 120, 251)}


# ----------------------------------------------------------------------------- tcc


@case("topology.tcc.imports_a_name_that_does_not_exist", tags=("fixed",))
def _():
    # ``from .classic import reciprocal_connections`` fails: classic has no such function
    return T.tcc(_dense())


@case("topology.tcc.center_argument", tags=("fixed",))
def _():
    return T.tcc(_dense(), center=3)


# ----------------------------------------------------------------------------- simplicial_rich_club_curve


@case("topology.simplicial_rich_club_curve.wrong_keyword", tags=("fixed",))
def _():
    # efficient_rich_club_curve has no ``pre_calculated_richness`` argument (it is ``pre_calculated_filtration``)
    return T.simplicial_rich_club_curve(rich(14, 0.12, seed=10, sizes=(5, 4), rc_sizes=(3,)))


@case("topology.simplicial_rich_club_curve.maximal_simplices", tags=("fixed",))
def _():
    return T.simplicial_rich_club_curve(
        rich(14, 0.12, seed=10, sizes=(5, 4), rc_sizes=(3,)), maximal=True
    )


@case("topology.simplicial_rich_club_curve.sparse_bin_set", tags=("fixed",))
def _():
    return T.simplicial_rich_club_curve(
        rich(14, 0.12, seed=10, sizes=(5, 4), rc_sizes=(3,)), sparse_bin_set=True
    )


# ----------------------------------------------------------------------------- _binary2simplex

END = (1 << 21) - 1  # marks an unused vertex slot in a word


def _word(vertices, continuation=False):
    vertices = list(vertices) + [END] * (3 - len(vertices))
    word = vertices[0] | (vertices[1] << 21) | (vertices[2] << 42)
    return word | (1 << 63) if continuation else word


def _words(simplices):
    """Flagser's layout: up to three 21-bit vertices per uint64, bit 63 set on the words that continue a simplex."""
    words = []
    for simplex in simplices:
        for start in range(0, len(simplex), 3):
            words.append(_word(simplex[start : start + 3], continuation=start > 0))
    return np.array(words, dtype=np.uint64)


SIMPLICES = [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9], [10, 11, 12, 13, 14, 15, 16], [2**20, 5, 7]]


def _write(directory, words, name="simplices.bin"):
    path = os.path.join(directory, name)
    words.tofile(path)
    return path


@case("topology._binary2simplex.decode_all")
def _():
    with tempfile.TemporaryDirectory() as directory:
        return T._binary2simplex(_write(directory, _words(SIMPLICES)))


@case("topology._binary2simplex.test_argument_limits_the_words_and_returns_both_views")
def _():
    with tempfile.TemporaryDirectory() as directory:
        path = _write(directory, _words(SIMPLICES))
        return {n: T._binary2simplex(path, test=n) for n in (1, 4, 100)}


@case("topology._binary2simplex.test_zero_means_everything")
def _():
    with tempfile.TemporaryDirectory() as directory:
        path = _write(directory, _words(SIMPLICES))
        return [T._binary2simplex(path, test=0), T._binary2simplex(path, test=None)]


@case("topology._binary2simplex.verbosity_does_not_change_the_result")
def _():
    with tempfile.TemporaryDirectory() as directory:
        path = _write(directory, _words(SIMPLICES))
        return T._binary2simplex(path, verbosity=2)


@case("topology._binary2simplex.file_starting_with_a_continuation_word")
def _():
    with tempfile.TemporaryDirectory() as directory:
        words = np.array([_word([1, 2, 3], continuation=True), _word([4])], dtype=np.uint64)
        return T._binary2simplex(_write(directory, words))


@case("topology._binary2simplex.many_small_simplices")
def _():
    with tempfile.TemporaryDirectory() as directory:
        simplices = [[i, i + 1, i + 2] for i in range(0, 60, 3)] + [
            [100 + i, 200 + i] for i in range(10)
        ]
        return T._binary2simplex(_write(directory, _words(simplices)))


@case("topology._binary2simplex.empty_file")
def _():
    with tempfile.TemporaryDirectory() as directory:
        return T._binary2simplex(_write(directory, np.array([], dtype=np.uint64)))


@case("topology._binary2simplex.missing_file")
def _():
    return T._binary2simplex("/nonexistent-directory/missing.bin")
