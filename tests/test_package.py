"""What importing the package does, and what it exports."""

import subprocess
import sys

import pytest


def run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout


def test_importing_the_package_has_no_side_effects():
    out = run_python(
        "import logging, sys, connalysis\n"
        "print(sorted(m for m in ('matplotlib', 'pyflagsercount', 'flagserpy', 'bigrandomgraphs') if m in sys.modules))\n"
        "print(logging.getLogger().handlers, logging.getLogger().level)\n"
    ).splitlines()
    assert out[0] == "[]", "the compiled and plotting packages are imported when they are needed"
    assert out[1] == f"[] {30}", "importing a library must not configure logging"


def test_public_names_are_where_they_always_were():
    import connalysis
    from connalysis.modelling import (  # noqa: F401
        conn_prob_2nd_order_model,
        conn_prob_2nd_order_pathway_model,
        conn_prob_3rd_order_model,
        conn_prob_3rd_order_pathway_model,
        conn_prob_model,
        conn_prob_pathway_model,
        per_group_in_out_bias_model,
        run_batch_model_building,
        run_model_building,
        run_pathway_model_building,
    )
    from connalysis.network import (  # noqa: F401
        _flagser_counts,
        classic,
        edge_participation,
        efficient_rich_club_curve,
        local,
        simplex_counts,
        stats,
        topology,
    )  # noqa: F401
    from connalysis.randomization import (  # noqa: F401
        ER_model,
        ER_shuffle,
        adjust_bidirectional_connections,
        configuration_model,
        random_geometric_model,
        run_DD2,
        run_DD3,
        run_ER,
        run_SBM,
        stochastic_spread_model,
    )

    assert isinstance(connalysis.__version__, str) and connalysis.__version__
    assert {"modelling", "network", "randomization"} <= set(vars(connalysis))
    namespace = {}
    exec("from connalysis import *", namespace)  # used to fail on a name that did not exist
    assert {"modelling", "network", "randomization"} <= set(namespace)


def test_the_third_party_packages_are_imported_when_used():
    import numpy as np
    import scipy.sparse as sp

    out = run_python(
        "import sys, numpy as np, scipy.sparse as sp\n"
        "from connalysis.network import topology\n"
        "from connalysis.randomization import run_ER\n"
        "assert 'pyflagsercount' not in sys.modules and 'bigrandomgraphs' not in sys.modules\n"
        "a = sp.csr_matrix(np.ones((4, 4), bool) & ~np.eye(4, dtype=bool))\n"
        "print(topology.simplex_counts(a).tolist(), run_ER(5, 1.0, threads=1).nnz)\n"
        "assert 'pyflagsercount' in sys.modules and 'bigrandomgraphs' in sys.modules\n"
    )
    assert out.strip() == "[4, 12, 24, 24] 20"  # ordered tuples of a complete digraph
    assert np and sp


def test_plotting_asks_for_matplotlib_when_it_is_missing(monkeypatch):
    from connalysis.modelling import modelling

    monkeypatch.setitem(
        sys.modules, "matplotlib.pyplot", None
    )  # makes `import matplotlib.pyplot` fail
    with pytest.raises(ImportError, match=r"connectome-analysis\[plot\]"):
        modelling._pyplot()
