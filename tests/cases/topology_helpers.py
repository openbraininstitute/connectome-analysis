"""Graph builders shared by the ``topology_*`` case files (no cases are registered here)."""

import functools
import itertools
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, rng


def contained_case(case_id, **kwargs):
    """``support.cases.case`` that runs the function with ``tempfile.tempdir`` pointing at a throw-away directory.

    ``simplex_counts(nodes=...)`` (and ``list_simplices_by_dimension``, ``in_degree_from_pop``, ... which build on it) create
    an empty ``NamedTemporaryFile(delete=False)`` next to the ``.npy`` file they remove, and never delete it; this keeps
    those files out of the real temp directory.
    """

    def register(fn):
        @functools.wraps(fn)
        def run():
            with tempfile.TemporaryDirectory() as directory:
                previous = tempfile.tempdir
                tempfile.tempdir = directory
                try:
                    return fn()
                finally:
                    tempfile.tempdir = previous

        return case(case_id, **kwargs)(run)

    return register


def rich(n, p, seed=0, sizes=(6, 5, 4), rc_sizes=(4, 3), reciprocal=0.25, dtype=bool, fmt="csr"):
    """``digraph(n, p, reciprocal=...)`` plus feedforward cliques of ``sizes`` and fully reciprocal cliques of ``rc_sizes``.

    Unlike ``planted`` it also contains reciprocal pairs and higher-dimensional reciprocal simplices, so the
    directed, undirected and reciprocal flag complexes all differ.
    """
    r = rng(seed + 1000)
    dense = digraph(n, p, seed, reciprocal=reciprocal).toarray()
    for size in sizes:
        nodes = r.permutation(n)[:size]
        for i in range(size):
            for j in range(i + 1, size):
                dense[nodes[i], nodes[j]] = True
    for size in rc_sizes:
        nodes = r.permutation(n)[:size]
        dense[np.ix_(nodes, nodes)] = True
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


def from_edges(n, edges, dtype=bool, fmt="csr"):
    """Digraph on ``n`` nodes with the given ``(source, target)`` edges."""
    if not edges:
        return sp.csr_matrix((n, n), dtype=dtype).asformat(fmt)
    rows, cols = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges), dtype=dtype), (rows, cols)), shape=(n, n)).asformat(
        fmt
    )


def complete(n, dtype=bool):
    """Complete digraph (every ordered pair, no loops)."""
    return sp.csr_matrix((np.ones((n, n)) - np.eye(n)).astype(dtype))


def transitive(n, dtype=bool):
    """Transitive tournament: ``i -> j`` for every ``i < j``."""
    return sp.csr_matrix(np.triu(np.ones((n, n)), 1).astype(dtype))


def cycle(n, dtype=bool):
    """Directed cycle ``0 -> 1 -> ... -> n-1 -> 0``."""
    return from_edges(n, [(i, (i + 1) % n) for i in range(n)], dtype=dtype)


def octahedron(dtype=bool):
    """Cross-polytope on 6 nodes (all pairs except the opposite ones ``0-1, 2-3, 4-5``), oriented ``i -> j`` for ``i < j``.

    Its clique complex is a 2-sphere: Betti numbers (1, 0, 1).
    """
    opposite = {(0, 1), (2, 3), (4, 5)}
    return from_edges(
        6,
        [(i, j) for i in range(6) for j in range(i + 1, 6) if (i, j) not in opposite],
        dtype=dtype,
    )


def structured(dtype=bool):
    """32 nodes in eight components: a transitive 5-clique (plus a pendant node), a directed 4-cycle, a reciprocal
    triangle, an octahedron, a hub with a triangle, an isolated node, a one-way pair and a reciprocal pair."""
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 0-4: transitive 5-clique
    edges += [(31, 0), (4, 31)]  # 31: pendant node attached to the clique
    edges += [(5, 6), (6, 7), (7, 8), (8, 5)]  # 5-8: directed 4-cycle
    edges += [
        (a, b) for a in (9, 10, 11) for b in (9, 10, 11) if a != b
    ]  # 9-11: reciprocal triangle
    edges += [
        (12 + i, 12 + j)
        for i in range(6)
        for j in range(i + 1, 6)
        if (i, j) not in {(0, 1), (2, 3), (4, 5)}
    ]
    edges += [(18, k) for k in range(19, 24)] + [
        (24, 18),
        (25, 18),
        (19, 20),
    ]  # 18-25: hub (+ one triangle)
    edges += [
        (27, 28),
        (29, 30),
        (30, 29),
    ]  # 27-28 one-way pair, 29-30 reciprocal pair; 26 stays isolated
    return from_edges(32, edges, dtype=dtype)


def projective_plane():
    """Barycentric subdivision of the 6-vertex triangulation of the real projective plane, edges oriented from the lower to
    the higher dimensional face: a flag complex on 31 nodes (90 edges, 60 triangles).

    It has 2-torsion: Betti numbers (1, 1, 1) over the field with two elements, but (1, 0, 0) over the field with three.
    """
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 1),
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 1),
        (4, 5, 2),
        (5, 1, 3),
    ]
    edges = sorted({tuple(sorted(e)) for t in triangles for e in itertools.combinations(t, 2)})
    faces = [(v,) for v in range(6)] + edges + [tuple(sorted(t)) for t in triangles]
    position = {face: i for i, face in enumerate(faces)}
    pairs = [
        (position[a], position[b])
        for a in faces
        for b in faces
        if len(a) < len(b) and set(a) < set(b)
    ]
    return from_edges(len(faces), pairs)


def explicit_zeros(adj, n_zeros):
    """Float copy of ``adj`` whose first ``n_zeros`` stored entries are explicit zeros (stored, but not edges)."""
    out = adj.astype(float).tocsr()
    out.data[:n_zeros] = 0.0
    return out


def with_loops(adj, nodes):
    """Copy of ``adj`` with self-loops on ``nodes``."""
    out = adj.tolil()
    for v in nodes:
        out[v, v] = True
    return out.tocsr()


def triad_graph(seed=21):
    """Random digraph with reciprocal pairs, a reciprocal triangle and 3-cycles of both orientations, so that all
    13 connected triad classes (and both labellings of the 3-cycle) occur."""
    dense = digraph(40, 0.07, seed, reciprocal=0.3).toarray()
    for a, b, c in [
        (30, 31, 32),
        (33, 35, 34),
    ]:  # a -> b -> c -> a and a -> c -> b -> a (as labelled by position)
        dense[a, b] = dense[b, c] = dense[c, a] = True
    for a, b in [(36, 37), (37, 38), (38, 36)]:
        dense[a, b] = True
        dense[b, a] = True
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense)


# ----------------------------------------------------------------------------- precomputed inputs without the library
# ``get_k_skeleta_graph``, ``count_rc_edges_skeleta`` and ``bedge_counts`` take simplex lists / edge participation that are
# normally produced by ``list_simplices_by_dimension`` / ``edge_participation``.  Brute-force versions are used here so that
# these cases do not depend on the functions that normally produce them.


def directed_simplices(adj):
    """All simplices of the directed flag complex of ``adj`` as tuples, one list per dimension (brute force)."""
    csr = sp.csr_matrix(adj)
    successors = [
        set(csr.indices[csr.indptr[v] : csr.indptr[v + 1]].tolist()) for v in range(csr.shape[0])
    ]
    by_dim = [[(v,) for v in range(csr.shape[0])]]
    while by_dim[-1]:
        grown = []
        for simplex in by_dim[-1]:
            for w in sorted(set.intersection(*[successors[v] for v in simplex])):
                grown.append(simplex + (w,))
        by_dim.append(grown)
    return by_dim[:-1]


def _object_series(arrays, name, index_name):
    values = np.empty(len(arrays), dtype=object)
    for i, array in enumerate(arrays):
        values[i] = array
    return pd.Series(values, index=pd.Index(range(len(arrays)), name=index_name), name=name)


def simplex_series(adj, max_simplices=False):
    """Series of ``(number of simplices, dim + 1)`` int64 arrays indexed by dimension, like ``list_simplices_by_dimension``
    (empty dimensions are ``array([])``)."""
    by_dim = directed_simplices(adj)
    if max_simplices:
        covered = [set() for _ in by_dim]
        for d in range(1, len(by_dim)):
            for simplex in by_dim[d]:
                for k in range(len(simplex)):
                    covered[d - 1].add(simplex[:k] + simplex[k + 1 :])
        by_dim = [
            [s for s in simplices if s not in covered[d]] for d, simplices in enumerate(by_dim)
        ]
        by_dim[
            0
        ] = []  # flagser does not list maximal 0-simplices (isolated nodes), although it counts them
    arrays = [
        np.array(simplices, dtype=np.int64) if simplices else np.array([]) for simplices in by_dim
    ]
    return _object_series(arrays, "simplices", "dim")


def edge_participation_frame(adj, max_simplices=False):
    """DataFrame indexed by the edges ``(i, j)`` with one column per dimension holding the number of (maximal) simplices
    containing the edge, built like the output of ``edge_participation``."""
    by_dim = directed_simplices(adj)
    series = simplex_series(adj, max_simplices=max_simplices)
    counts = {}
    for d in range(1, len(by_dim)):
        array = series[d]
        for simplex in array if array.size else []:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    row = counts.setdefault((int(simplex[i]), int(simplex[j])), [0] * len(by_dim))
                    row[d] += 1
    return (
        pd.DataFrame.from_dict(dict(sorted(counts.items())), orient="index").fillna(0).astype(int)
    )
