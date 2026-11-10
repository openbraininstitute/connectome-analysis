"""``property_at_neighborhoods`` and ``properties_at_neighborhoods``: a network property on every neighbourhood.

The dict keys come from the (sorted) index of ``neighborhood_indices``, so the order of the keys is pinned too.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case
from support.data import digraph, planted, tutorial_connectome, weighted

from connalysis.network import local as L
from connalysis.network import topology as T


def _mixed(fmt="csr", dtype=bool):
    """16 nodes with reciprocal pairs, two isolated nodes (5, 11), a pure source (12) and a pure sink (13)."""
    dense = digraph(16, 0.14, seed=3, reciprocal=0.3).toarray()
    dense[[5, 11], :] = False
    dense[:, [5, 11]] = False
    dense[:, 12] = False
    dense[12, [1, 2, 7]] = True
    dense[13, :] = False
    dense[[0, 4, 9], 13] = True
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt)


def _total(m):
    return m.sum()


def _describe(m):
    """What a function sees: the container type, shape and dtype of the neighbourhood and its number of edges."""
    return (type(m).__name__, tuple(m.shape), str(m.dtype), int((m != 0).sum()))


COMBOS = {
    "both": dict(pre=True, post=True),
    "pre": dict(pre=True, post=False),
    "post": dict(pre=False, post=True),
}
CENTERS = np.array([13, 5, 2, 12, 9, 0])  # unsorted, with an isolated node, a source and a sink


# =============================================================================================== property_at_neighborhoods


@case("local.property_at_neighborhoods.node_degree")
def _():
    return L.property_at_neighborhoods(_mixed(), T.node_degree)


@case("local.property_at_neighborhoods.sum_of_the_matrix")
def _():
    return L.property_at_neighborhoods(_mixed(), _total)


@case("local.property_at_neighborhoods.lambda_function")
def _():
    return L.property_at_neighborhoods(_mixed(), lambda m: m.sum())


@case("local.property_at_neighborhoods.kwargs_are_forwarded")
def _():
    adj = weighted(16, 0.3, seed=41)
    return {
        "direction": L.property_at_neighborhoods(adj, T.node_degree, direction="IN"),
        "weighted": L.property_at_neighborhoods(adj, T.node_degree, direction="OUT", weighted=True),
        "both": L.property_at_neighborhoods(
            adj, T.node_degree, direction=("IN", "OUT"), weighted=True, pre=True, post=False
        ),
    }


@case("local.property_at_neighborhoods.keyword_arguments")
def _():
    return L.property_at_neighborhoods(
        adj=_mixed(),
        func=T.node_degree,
        pre=True,
        post=False,
        include_center=True,
        all_nodes=False,
        centers=np.array([2, 7]),
        direction="IN",
    )


@case("local.property_at_neighborhoods.simplex_counts")
def _():
    return L.property_at_neighborhoods(planted(30, 0.08, seed=42, sizes=(5, 4)), T.simplex_counts)


@case("local.property_at_neighborhoods.pre_post_include_center")
def _():
    adj = _mixed()
    return {
        f"{name}_center{int(center)}": L.property_at_neighborhoods(
            adj, _total, include_center=center, **kw
        )
        for name, kw in COMBOS.items()
        for center in (True, False)
    }


@case("local.property_at_neighborhoods.centers_unsorted")
def _():
    # all_nodes=False: only the centers, and the dict follows the sorted index of the neighbourhoods
    adj = _mixed()
    return {
        name: L.property_at_neighborhoods(adj, _describe, all_nodes=False, centers=CENTERS, **kw)
        for name, kw in COMBOS.items()
    }


@case("local.property_at_neighborhoods.centers_ignored_when_all_nodes")
def _():
    return L.property_at_neighborhoods(_mixed(), _total, all_nodes=True, centers=np.array([3, 1]))


@case("local.property_at_neighborhoods.int32_centers")
def _():
    # the keys of the dict are the (int32) centers
    return L.property_at_neighborhoods(
        _mixed(), _total, all_nodes=False, centers=np.array([9, 2, 12], dtype=np.int32)
    )


@case("local.property_at_neighborhoods.self_loops")
def _():
    # a center with a loop appears twice among the nodes of its neighbourhood, so its loop is counted four times
    dense = _mixed().toarray()
    dense[2, 2] = dense[9, 9] = True
    return L.property_at_neighborhoods(
        sp.csr_matrix(dense), _total, all_nodes=False, centers=np.array([2, 9, 5])
    )


def _clobber(m):
    """A function that overwrites its argument: the neighbourhood handed over must be a copy."""
    if sp.issparse(m):
        m.data[:] = True
    else:
        m[...] = True
    return m.sum()


@case("local.property_at_neighborhoods.function_overwrites_its_argument")
def _():
    sparse = _mixed()
    dense = sparse.toarray()
    centers = np.array([2, 0, 9])
    result = {
        "sparse": L.property_at_neighborhoods(sparse, _clobber, all_nodes=False, centers=centers),
        "dense": L.property_at_neighborhoods(dense, _clobber, all_nodes=False, centers=centers),
    }
    return result, sparse, dense


@case("local.property_at_neighborhoods.no_centers")
def _():
    return L.property_at_neighborhoods(
        _mixed(), _total, all_nodes=False, centers=np.array([], dtype=int)
    )


@case("local.property_at_neighborhoods.function_input_type")
def _():
    # sparse input of any format reaches the function as csr_matrix, dense input stays dense
    adj = _mixed()
    out = {
        fmt: L.property_at_neighborhoods(
            adj.asformat(fmt), _describe, all_nodes=False, centers=np.array([2, 5])
        )
        for fmt in ("csr", "csc", "coo", "lil", "dok")
    }
    out["ndarray"] = L.property_at_neighborhoods(
        adj.toarray(), _describe, all_nodes=False, centers=np.array([2, 5])
    )
    out["matrix"] = L.property_at_neighborhoods(
        np.matrix(adj.toarray()), _describe, all_nodes=False, centers=np.array([2, 5])
    )
    out["csr_array"] = L.property_at_neighborhoods(
        sp.csr_array(adj), _describe, all_nodes=False, centers=np.array([2, 5])
    )
    return out


@case("local.property_at_neighborhoods.dense_input")
def _():
    return L.property_at_neighborhoods(_mixed().toarray(), _total)


@case("local.property_at_neighborhoods.no_center_in_isolated_neighbourhood")
def _():
    # without the center an isolated node leaves an empty 0 x 0 matrix for the function
    adj = _mixed()
    return L.property_at_neighborhoods(adj, _describe, include_center=False)


@case("local.property_at_neighborhoods.weighted")
def _():
    return L.property_at_neighborhoods(weighted(16, 0.3, seed=43, integers=False), _total)


@case("local.property_at_neighborhoods.bool_vs_int_matrix_sum")
def _():
    # m.sum() of a bool matrix counts True entries, of an int8 matrix it widens: dtypes are part of the result
    adj = _mixed()
    return {
        "bool": L.property_at_neighborhoods(adj, _total, all_nodes=False, centers=np.array([2, 0])),
        "int8": L.property_at_neighborhoods(
            adj.astype(np.int8), _total, all_nodes=False, centers=np.array([2, 0])
        ),
        "float32": L.property_at_neighborhoods(
            adj.astype(np.float32), _total, all_nodes=False, centers=np.array([2, 0])
        ),
    }


@case("local.property_at_neighborhoods.edgeless_graph_all_nodes")
def _():
    return L.property_at_neighborhoods(sp.csr_matrix((4, 4), dtype=bool), _describe)


@case("local.property_at_neighborhoods.tutorial_simplex_counts")
def _():
    adj, _nodes = tutorial_connectome(300)
    return L.property_at_neighborhoods(adj, T.simplex_counts)


@case("local.property_at_neighborhoods.tutorial_centers")
def _():
    adj, _nodes = tutorial_connectome(600)
    centers = np.random.RandomState(44).permutation(600)[:30]
    return L.property_at_neighborhoods(
        adj, T.node_degree, pre=True, post=False, all_nodes=False, centers=centers, direction="OUT"
    )


@case("local.property_at_neighborhoods.centers_required")
def _():
    return L.property_at_neighborhoods(_mixed(), _total, all_nodes=False)


@case("local.property_at_neighborhoods.pre_and_post_false")
def _():
    return L.property_at_neighborhoods(_mixed(), _total, pre=False, post=False)


@case("local.property_at_neighborhoods.edgeless_graph_with_centers")
def _():
    # defect inherited from neighborhood_indices
    return L.property_at_neighborhoods(
        sp.csr_matrix((4, 4), dtype=bool), _total, all_nodes=False, centers=np.array([1])
    )


@case("local.property_at_neighborhoods.unknown_keyword")
def _():
    return L.property_at_neighborhoods(_mixed(), _total, no_such_argument=1)


# =============================================================================================== properties_at_neighborhoods


def _config():
    return {
        "in_degree": {"function": T.node_degree, "kwargs": {"direction": "IN"}},
        "edges": {"function": _total, "kwargs": {}},
        "shape": {"function": _describe, "kwargs": {}},
    }


@case("local.properties_at_neighborhoods.three_functions")
def _():
    return L.properties_at_neighborhoods(_mixed(), _config())


@case("local.properties_at_neighborhoods.keyword_arguments")
def _():
    return L.properties_at_neighborhoods(
        adj=_mixed(),
        func_config=_config(),
        pre=False,
        post=True,
        include_center=False,
        all_nodes=False,
        centers=np.array([2, 7]),
    )


@case("local.properties_at_neighborhoods.centers_unsorted")
def _():
    return L.properties_at_neighborhoods(_mixed(), _config(), all_nodes=False, centers=CENTERS)


@case("local.properties_at_neighborhoods.pre_post_include_center")
def _():
    adj = _mixed()
    cfg = _config()
    return {
        f"{name}_center{int(center)}": L.properties_at_neighborhoods(
            adj, cfg, include_center=center, **kw
        )
        for name, kw in COMBOS.items()
        for center in (True, False)
    }


@case("local.properties_at_neighborhoods.function_kwargs")
def _():
    adj = weighted(16, 0.3, seed=45)
    cfg = {
        "weighted_out": {
            "function": T.node_degree,
            "kwargs": {"direction": "OUT", "weighted": True},
        },
        "unweighted_both": {"function": T.node_degree, "kwargs": {"direction": ("IN", "OUT")}},
    }
    return L.properties_at_neighborhoods(adj, cfg, all_nodes=False, centers=np.array([6, 2, 10]))


@case("local.properties_at_neighborhoods.dense_input")
def _():
    return L.properties_at_neighborhoods(
        _mixed().toarray(), _config(), all_nodes=False, centers=np.array([2, 5, 13])
    )


@case("local.properties_at_neighborhoods.simplex_counts")
def _():
    cfg = {
        "counts": {"function": T.simplex_counts, "kwargs": {}},
        "counts_max": {"function": T.simplex_counts, "kwargs": {"max_simplices": True}},
    }
    return L.properties_at_neighborhoods(
        planted(30, 0.08, seed=46, sizes=(5, 4)), cfg, all_nodes=False, centers=np.arange(0, 30, 4)
    )


@case("local.properties_at_neighborhoods.empty_config")
def _():
    return L.properties_at_neighborhoods(_mixed(), {})


@case("local.properties_at_neighborhoods.tutorial")
def _():
    adj, _nodes = tutorial_connectome(300)
    cfg = {
        "simplices": {"function": T.simplex_counts, "kwargs": {}},
        "out_degree": {"function": T.node_degree, "kwargs": {"direction": "OUT"}},
    }
    centers = np.random.RandomState(47).permutation(300)[:40]
    return L.properties_at_neighborhoods(adj, cfg, all_nodes=False, centers=centers)


@case("local.properties_at_neighborhoods.missing_kwargs_entry")
def _():
    return L.properties_at_neighborhoods(_mixed(), {"edges": {"function": _total}})


@case("local.properties_at_neighborhoods.missing_function_entry")
def _():
    return L.properties_at_neighborhoods(_mixed(), {"edges": {"kwargs": {}}})


@case("local.properties_at_neighborhoods.pre_and_post_false")
def _():
    return L.properties_at_neighborhoods(_mixed(), _config(), pre=False, post=False)
