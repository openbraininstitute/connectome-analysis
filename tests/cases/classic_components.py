"""Closeness, connected components, core numbers, the largest strongly connected component and the
networkx conversions in ``connalysis.network.classic``."""

import networkx as nx
import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, planted, weighted

from cases.classic_inputs import (
    block_scc_graph,
    complete,
    from_edges,
    multi_component,
    with_explicit_zeros,
)
from connalysis.network import classic as C

# ----------------------------------------------------------------------------- closeness_connected_components


@case("classic.closeness_connected_components.undirected_sum")
def _():
    return C.closeness_connected_components(multi_component())


@case("classic.closeness_connected_components.undirected_list")
def _():
    return C.closeness_connected_components(multi_component(), directed=False, return_sum=False)


@case("classic.closeness_connected_components.directed_sum")
def _():
    return C.closeness_connected_components(multi_component(), directed=True)


@case("classic.closeness_connected_components.directed_list")
def _():
    return C.closeness_connected_components(multi_component(), directed=True, return_sum=False)


@case("classic.closeness_connected_components.random_digraph")
def _():
    adj = planted(40, 0.05, seed=21, sizes=(5, 4))
    return [
        C.closeness_connected_components(adj, directed=d, return_sum=s)
        for d in (False, True)
        for s in (True, False)
    ]


@case("classic.closeness_connected_components.weights_are_ignored")
def _():
    adj = weighted(24, 0.12, seed=22, integers=False)
    return [C.closeness_connected_components(adj, directed=d) for d in (False, True)]


@case("classic.closeness_connected_components.input_formats")
def _():
    base = multi_component()
    inputs = [
        base.tocsc(),
        base.tocoo(),
        base.tolil(),
        base.toarray(),
        base.toarray().astype(int),
        base.astype(float),
    ]
    return [C.closeness_connected_components(m, directed=d) for m in inputs for d in (False, True)]


@case("classic.closeness_connected_components.no_edges")
def _():
    empty = sp.csr_matrix((6, 6), dtype=bool)
    return [
        C.closeness_connected_components(empty, directed=d, return_sum=s)
        for d in (False, True)
        for s in (True, False)
    ]


@case("classic.closeness_connected_components.single_node")
def _():
    return C.closeness_connected_components(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.closeness_connected_components.two_nodes_one_edge")
def _():
    adj = from_edges(2, [(0, 1)])
    return [
        C.closeness_connected_components(adj, directed=d, return_sum=s)
        for d in (False, True)
        for s in (True, False)
    ]


@case("classic.closeness_connected_components.complete_graph")
def _():
    return [C.closeness_connected_components(complete(6), directed=d) for d in (False, True)]


@case("classic.closeness_connected_components.self_loop")
def _():
    adj = from_edges(3, [(0, 0), (1, 2), (2, 1)])
    return [
        C.closeness_connected_components(adj, directed=d, return_sum=s)
        for d in (False, True)
        for s in (True, False)
    ]


@case("classic.closeness_connected_components.directed_list_with_several_strong_components")
def _():
    # the arrays come in the order of scipy's strong-component labels
    return C.closeness_connected_components(block_scc_graph(), directed=True, return_sum=False)


@case("classic.closeness_connected_components.directed_list_with_several_strong_components_sorted")
def _():
    # same call, arrays ordered by their first non-zero node, so that scipy's label numbering does not matter
    arrays = C.closeness_connected_components(block_scc_graph(), directed=True, return_sum=False)
    return sorted(arrays, key=lambda a: int(np.flatnonzero(a)[0]))


@case("classic.closeness_connected_components.directed_sum_with_several_strong_components")
def _():
    return C.closeness_connected_components(block_scc_graph(), directed=True)


@case("classic.closeness_connected_components.undirected_with_several_blocks")
def _():
    return [
        C.closeness_connected_components(block_scc_graph(), directed=False),
        C.closeness_connected_components(
            block_scc_graph(seed=1, sizes=(6, 6, 3)), directed=False, return_sum=False
        ),
    ]


@case("classic.closeness_connected_components.sparse_array_input")
def _():
    adj = sp.csr_array(multi_component())
    return [C.closeness_connected_components(adj, directed=d) for d in (False, True)]


@case("classic.closeness_connected_components.result_types")
def _():
    # the snapshot does not tell np.matrix from ndarray, so the classes are returned as text
    adj = multi_component()
    summed = C.closeness_connected_components(adj)
    listed = C.closeness_connected_components(adj, return_sum=False)
    none = C.closeness_connected_components(sp.csr_matrix((3, 3), dtype=bool))
    return [
        type(summed).__name__,
        type(listed).__name__,
        [type(a).__name__ for a in listed],
        type(none).__name__,
    ]


@case("classic.closeness_connected_components.non_square")
def _():
    return C.closeness_connected_components(sp.csr_matrix(np.ones((3, 4), dtype=bool)))


@case("classic.closeness_connected_components.explicit_zero_bridge", tags=("fixed",))
def _():
    # scipy counts the stored zero as an edge (one component); sknetwork ignores it and finds the component disconnected
    adj = sp.csr_matrix(
        (np.array([1.0, 1.0, 0.0, 1.0, 1.0]), np.array([1, 0, 2, 3, 2]), np.array([0, 1, 3, 4, 5])),
        shape=(4, 4),
    )
    return C.closeness_connected_components(adj, directed=False)


# ----------------------------------------------------------------------------- connected_components


def _both(adj, **kwargs):
    return [
        C.connected_components(adj, return_labels=False, **kwargs),
        C.connected_components(adj, return_labels=True, **kwargs),
    ]


@case("classic.connected_components.default")
def _():
    return C.connected_components(multi_component())


@case("classic.connected_components.directed_weak")
def _():
    return _both(multi_component(), directed=True, connection="weak")


@case("classic.connected_components.directed_strong")
def _():
    return _both(multi_component(), directed=True, connection="strong")


@case("classic.connected_components.undirected_weak")
def _():
    return _both(multi_component(), directed=False, connection="weak")


@case("classic.connected_components.undirected_strong")
def _():
    return _both(multi_component(), directed=False, connection="strong")


@case("classic.connected_components.random_digraph")
def _():
    adj = digraph(50, 0.04, seed=3)
    return [_both(adj, directed=d, connection=c) for d in (True, False) for c in ("weak", "strong")]


@case("classic.connected_components.giant_strong_component")
def _():
    adj = digraph(60, 0.035, seed=1)
    return _both(adj, connection="strong")


@case("classic.connected_components.weights_are_ignored")
def _():
    adj = weighted(40, 0.04, seed=4, integers=False)
    return [
        _both(adj, directed=True, connection="strong"),
        _both(adj, directed=True, connection="weak"),
    ]


@case("classic.connected_components.input_formats")
def _():
    base = multi_component()
    inputs = [
        base.tocsc(),
        base.tocoo(),
        base.tolil(),
        base.toarray(),
        base.toarray().astype(float),
        base.astype(int),
    ]
    return [_both(m, connection="strong") for m in inputs]


@case("classic.connected_components.strong_components_of_chained_blocks")
def _():
    return _both(block_scc_graph(), connection="strong")


def _first_appearance(labels):
    seen = {}
    return np.array([seen.setdefault(label, len(seen)) for label in labels.tolist()])


@case("classic.connected_components.strong_partition_up_to_label_numbering")
def _():
    # sizes sorted and labels renumbered by first appearance: the partition without scipy's numbering of it
    out = []
    for adj in (
        digraph(50, 0.04, seed=3),
        digraph(60, 0.035, seed=1),
        weighted(40, 0.04, seed=4, integers=False),
        block_scc_graph(),
        multi_component(),
    ):
        sizes, labels = C.connected_components(adj, connection="strong", return_labels=True)
        out.append((np.sort(sizes), _first_appearance(labels), sizes.dtype.name, labels.dtype.name))
    return out


@case("classic.connected_components.sparse_array_input")
def _():
    adj = sp.csr_array(multi_component())
    return [_both(adj, connection=c) for c in ("weak", "strong")]


@case("classic.connected_components.explicit_zero_is_an_edge")
def _():
    # 1 -> 2 is a stored zero: scipy still treats it as the link between {0, 1} and {2, 3}
    adj = sp.csr_matrix(
        (np.array([1.0, 1.0, 0.0, 1.0, 1.0]), np.array([1, 0, 2, 3, 2]), np.array([0, 1, 3, 4, 5])),
        shape=(4, 4),
    )
    return [_both(adj, connection="weak"), _both(adj, connection="strong")]


@case("classic.connected_components.no_edges")
def _():
    return _both(sp.csr_matrix((5, 5), dtype=bool))


@case("classic.connected_components.single_node")
def _():
    return _both(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.connected_components.empty_graph")
def _():
    return _both(sp.csr_matrix((0, 0), dtype=bool))


@case("classic.connected_components.complete_graph")
def _():
    return _both(complete(6), connection="strong")


@case("classic.connected_components.two_nodes_one_edge")
def _():
    adj = from_edges(2, [(0, 1)])
    return [_both(adj, connection=c) for c in ("weak", "strong")]


@case("classic.connected_components.result_types")
def _():
    sizes, labels = C.connected_components(multi_component(), return_labels=True)
    only_sizes = C.connected_components(multi_component())
    return [type(only_sizes).__name__, type(sizes).__name__, type(labels).__name__]


@case("classic.connected_components.bad_connection")
def _():
    return C.connected_components(multi_component(), connection="medium")


@case("classic.connected_components.non_square")
def _():
    return C.connected_components(sp.csr_matrix(np.ones((3, 4), dtype=bool)))


# ----------------------------------------------------------------------------- core_number


@case("classic.core_number.multi_component")
def _():
    return C.core_number(multi_component())


@case("classic.core_number.random_digraph")
def _():
    return C.core_number(planted(40, 0.08, seed=31, sizes=(6, 5, 4)))


@case("classic.core_number.reciprocal_pairs_count_once")
def _():
    # a reciprocal 4-clique: 3-core of the underlying simple graph although every node has total degree 6
    edges = [(a, b) for a in range(4) for b in range(4) if a != b] + [(3, 4), (4, 5), (5, 4)]
    return C.core_number(from_edges(6, edges))


@case("classic.core_number.input_formats")
def _():
    base = planted(30, 0.06, seed=32, sizes=(5, 4))
    inputs = [
        base.tocsc(),
        base.tocoo(),
        base.tolil(),
        base.toarray(),
        weighted(30, 0.1, seed=32),
        base.astype(int),
    ]
    return [C.core_number(m) for m in inputs]


@case("classic.core_number.no_edges")
def _():
    return C.core_number(sp.csr_matrix((4, 4), dtype=bool))


@case("classic.core_number.single_node")
def _():
    return C.core_number(sp.csr_matrix((1, 1), dtype=bool))


@case("classic.core_number.empty_graph")
def _():
    return C.core_number(sp.csr_matrix((0, 0), dtype=bool))


@case("classic.core_number.two_nodes_one_edge")
def _():
    return C.core_number(from_edges(2, [(0, 1)]))


@case("classic.core_number.complete_graph")
def _():
    return C.core_number(complete(6))


@case("classic.core_number.sparse_array_input")
def _():
    return C.core_number(sp.csr_array(planted(30, 0.06, seed=32, sizes=(5, 4))))


@case("classic.core_number.explicit_zero_is_an_edge")
def _():
    adj = sp.csr_matrix(
        (np.array([1.0, 0.0, 1.0]), np.array([1, 2, 0]), np.array([0, 2, 3, 3])), shape=(3, 3)
    )
    return C.core_number(adj)


@case("classic.core_number.explicit_zeros_in_random_graph")
def _():
    return C.core_number(with_explicit_zeros(planted(30, 0.08, seed=33, sizes=(5, 4)), every=4))


@case("classic.core_number.self_loop")
def _():
    # networkx refuses graphs with self-loops
    return C.core_number(from_edges(4, [(0, 1), (1, 0), (2, 2), (2, 3)]))


@case("classic.core_number.non_square")
def _():
    return C.core_number(sp.csr_matrix(np.ones((3, 4), dtype=bool)))


# ----------------------------------------------------------------------------- largest_strongly_connected_component


def _cycle(nodes):
    return [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]


def _clique(nodes):
    return [(a, b) for a in nodes for b in nodes if a != b]


@case("classic.largest_strongly_connected_component.sorted_nodes")
def _():
    adj = from_edges(8, [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6), (6, 7)], dtype=int)
    return C.largest_strongly_connected_component(adj.toarray())


@case("classic.largest_strongly_connected_component.unsorted_nodes")
def _():
    # the component is {0, 2, 5, 7}; the rows/columns of the result must follow the original node order
    edges = [(5, 2), (2, 7), (7, 0), (0, 5), (2, 0), (1, 3), (3, 4), (4, 1), (1, 5), (8, 6)]
    return C.largest_strongly_connected_component(from_edges(9, edges, dtype=int).toarray())


@case("classic.largest_strongly_connected_component.sparse_formats")
def _():
    edges = [(5, 2), (2, 7), (7, 0), (0, 5), (2, 0), (1, 3), (3, 4), (4, 1), (1, 5), (8, 6)]
    base = from_edges(9, edges, dtype=int)
    return [
        C.largest_strongly_connected_component(m)
        for m in (base, base.tocsc(), base.tocoo(), base.tolil())
    ]


@case("classic.largest_strongly_connected_component.chained_blocks")
def _():
    # blocks of 7, 5, 8 and 4 nodes in shuffled positions: the 8-node block is returned, in node order
    return C.largest_strongly_connected_component(block_scc_graph())


@case("classic.largest_strongly_connected_component.chained_blocks_with_a_tie")
def _():
    return C.largest_strongly_connected_component(block_scc_graph(sizes=(6, 6, 3, 6), seed=2))


@case("classic.largest_strongly_connected_component.sparse_array_input")
def _():
    return C.largest_strongly_connected_component(sp.csr_array(block_scc_graph()))


@case("classic.largest_strongly_connected_component.dtypes")
def _():
    edges = [(5, 2), (2, 7), (7, 0), (0, 5), (2, 0), (1, 3), (3, 4), (4, 1), (1, 5), (8, 6)]
    base = from_edges(9, edges, dtype=int).toarray()
    weights = base * np.arange(1, 10)[None, :]  # distinct integer weights per target column
    return [
        C.largest_strongly_connected_component(base.astype(bool)),
        C.largest_strongly_connected_component(sp.csr_matrix(base.astype(bool))),
        C.largest_strongly_connected_component(weights),
        C.largest_strongly_connected_component(base.astype(float) * 0.5),
    ]


@case("classic.largest_strongly_connected_component.float_weights_are_truncated")
def _():
    fm = np.array([[0, 1.7, 0.2], [2.9, 0, 0.0], [0.5, 0, 0]])
    return C.largest_strongly_connected_component(fm)


@case("classic.largest_strongly_connected_component.less_than_half_of_the_nodes")
def _():
    adj = digraph(40, 0.03, seed=1)  # largest strongly connected component: 16 of 40 nodes
    return C.largest_strongly_connected_component(adj)


@case("classic.largest_strongly_connected_component.giant_component_sparse_input")
def _():
    adj = digraph(60, 0.035, seed=1)  # 45 of 60 nodes
    return C.largest_strongly_connected_component(adj)


@case("classic.largest_strongly_connected_component.tie_cycle_then_clique")
def _():
    # two components of 3 nodes with different edge counts; nothing links them
    a, b = [0, 1, 2], [3, 4, 5]
    return C.largest_strongly_connected_component(
        from_edges(6, _cycle(a) + _clique(b), dtype=int).toarray()
    )


@case("classic.largest_strongly_connected_component.tie_clique_then_cycle")
def _():
    a, b = [0, 1, 2], [3, 4, 5]
    return C.largest_strongly_connected_component(
        from_edges(6, _clique(a) + _cycle(b), dtype=int).toarray()
    )


@case("classic.largest_strongly_connected_component.tie_linked_forward")
def _():
    a, b = [0, 1, 2], [3, 4, 5]
    edges = _cycle(a) + _clique(b) + [(2, 3)]
    return C.largest_strongly_connected_component(from_edges(6, edges, dtype=int).toarray())


@case("classic.largest_strongly_connected_component.tie_linked_backward")
def _():
    a, b = [0, 1, 2], [3, 4, 5]
    edges = _cycle(a) + _clique(b) + [(3, 2)]
    return C.largest_strongly_connected_component(from_edges(6, edges, dtype=int).toarray())


@case("classic.largest_strongly_connected_component.tie_interleaved_nodes")
def _():
    a, b = [0, 2, 4], [1, 3, 5]
    edges = _clique(a) + _cycle(b) + [(5, 0)]
    return C.largest_strongly_connected_component(from_edges(6, edges, dtype=int).toarray())


@case("classic.largest_strongly_connected_component.tie_interleaved_nodes_linked_the_other_way")
def _():
    a, b = [0, 2, 4], [1, 3, 5]
    edges = _clique(a) + _cycle(b) + [(0, 1)]
    return C.largest_strongly_connected_component(from_edges(6, edges, dtype=int).toarray())


@case("classic.largest_strongly_connected_component.tie_between_singletons_in_a_chain")
def _():
    # no cycles: the self-loop weights show which single node is returned
    dense = np.zeros((5, 5), dtype=int)
    dense[0, 1] = dense[0, 2] = dense[1, 3] = dense[2, 4] = 1
    np.fill_diagonal(dense, np.arange(1, 6))
    return C.largest_strongly_connected_component(dense)


@case("classic.largest_strongly_connected_component.tie_between_singletons_random_dag")
def _():
    dense = digraph(40, 0.03, seed=3).toarray().astype(int)  # no directed cycle at all
    np.fill_diagonal(dense, np.arange(1, 41))
    return C.largest_strongly_connected_component(dense)


@case("classic.largest_strongly_connected_component.no_edges")
def _():
    return C.largest_strongly_connected_component(np.zeros((4, 4), dtype=int))


@case("classic.largest_strongly_connected_component.single_node")
def _():
    return C.largest_strongly_connected_component(sp.csr_matrix((1, 1), dtype=int))


@case("classic.largest_strongly_connected_component.two_nodes_reciprocal")
def _():
    return C.largest_strongly_connected_component(np.array([[0, 2], [3, 0]]))


@case("classic.largest_strongly_connected_component.complete_graph")
def _():
    return C.largest_strongly_connected_component(complete(5, dtype=int))


@case("classic.largest_strongly_connected_component.self_loops_are_kept")
def _():
    return C.largest_strongly_connected_component(
        from_edges(4, [(0, 0), (0, 1), (1, 0), (2, 2)], dtype=int)
    )


@case("classic.largest_strongly_connected_component.result_types")
def _():
    edges = [(5, 2), (2, 7), (7, 0), (0, 5), (2, 0), (1, 3), (3, 4), (4, 1), (1, 5), (8, 6)]
    base = from_edges(9, edges, dtype=int)
    results = [C.largest_strongly_connected_component(m) for m in (base, base.toarray())]
    return [[type(r).__name__, r.dtype.name, r.ndim] for r in results]


@case("classic.largest_strongly_connected_component.empty_graph")
def _():
    return C.largest_strongly_connected_component(np.zeros((0, 0), dtype=int))


@case("classic.largest_strongly_connected_component.non_square")
def _():
    return C.largest_strongly_connected_component(np.ones((3, 4), dtype=int))


# ----------------------------------------------------------------------------- np_to_nx / nx_to_np


def _with_weights(graph):
    return graph, list(graph.edges(data="weight"))


@case("classic.np_to_nx.int_matrix")
def _():
    dense = from_edges(7, [(0, 1), (1, 2), (2, 0), (3, 4), (5, 5)], dtype=int).toarray()
    return _with_weights(C.np_to_nx(dense))


@case("classic.np_to_nx.weighted_int_matrix")
def _():
    return _with_weights(C.np_to_nx(weighted(8, 0.25, seed=41).toarray()))


@case("classic.np_to_nx.float_matrix")
def _():
    return _with_weights(C.np_to_nx(weighted(8, 0.25, seed=42, integers=False).toarray()))


@case("classic.np_to_nx.bool_matrix")
def _():
    return _with_weights(C.np_to_nx(digraph(8, 0.25, seed=43).toarray()))


@case("classic.np_to_nx.sparse_input")
def _():
    return _with_weights(C.np_to_nx(weighted(8, 0.25, seed=44)))


@case("classic.np_to_nx.isolated_and_single_nodes")
def _():
    return [
        _with_weights(C.np_to_nx(np.zeros((3, 3), dtype=int))),
        _with_weights(C.np_to_nx(np.zeros((1, 1), dtype=int))),
    ]


@case("classic.np_to_nx.empty_matrix")
def _():
    return _with_weights(C.np_to_nx(np.zeros((0, 0), dtype=int)))


@case("classic.np_to_nx.non_square")
def _():
    return C.np_to_nx(np.ones((2, 3), dtype=int))


@case("classic.nx_to_np.round_trip")
def _():
    dense = weighted(9, 0.3, seed=45).toarray().astype(int)
    return C.nx_to_np(C.np_to_nx(dense))


@case("classic.nx_to_np.weights_are_cast_to_int")
def _():
    graph = nx.DiGraph()
    graph.add_nodes_from([5, 2, 9])
    graph.add_edge(5, 9, weight=3.7)
    graph.add_edge(9, 2)
    graph.add_edge(2, 5, weight=0.4)
    return C.nx_to_np(graph)


@case("classic.nx_to_np.node_order_is_insertion_order")
def _():
    graph = nx.DiGraph()
    for source, target in [(7, 3), (3, 9), (9, 7), (7, 1)]:
        graph.add_edge(source, target)
    return C.nx_to_np(graph)


@case("classic.nx_to_np.subgraph")
def _():
    graph = C.np_to_nx(
        from_edges(
            9, [(5, 2), (2, 7), (7, 0), (0, 5), (2, 0), (1, 3), (3, 4), (4, 1)], dtype=int
        ).toarray()
    )
    return C.nx_to_np(graph.subgraph({5, 2, 7, 0}))


@case("classic.nx_to_np.undirected_graph")
def _():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (3, 3)])
    return C.nx_to_np(graph)


@case("classic.nx_to_np.multigraph")
def _():
    return C.nx_to_np(nx.MultiDiGraph([(0, 1), (0, 1), (1, 0), (2, 2)]))


@case("classic.nx_to_np.string_nodes")
def _():
    return C.nx_to_np(nx.DiGraph([("b", "a"), ("a", "c")]))


@case("classic.nx_to_np.result_types")
def _():
    graph = C.np_to_nx(from_edges(5, [(0, 1), (1, 2), (2, 0)], dtype=int).toarray())
    result = C.nx_to_np(graph)
    return type(result).__name__, result.dtype.name, result.ndim


@case("classic.nx_to_np.empty_graph")
def _():
    return C.nx_to_np(nx.DiGraph())
