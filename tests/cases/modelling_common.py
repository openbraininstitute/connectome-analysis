"""Shared inputs and helpers of the ``modelling_*`` case files (this module registers no cases)."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import pickle

import numpy as np
import scipy.sparse as sp
from support.data import node_table, rng, tutorial_connectome

# ----------------------------------------------------------------------------- inputs


@functools.lru_cache(maxsize=None)
def _tutorial(n):
    return tutorial_connectome(n)


def real(n: int = 500):
    """First ``n`` neurons of the tutorial connectome: bool csr adjacency and its node table (fresh copies)."""
    adj, nodes = _tutorial(n)
    return adj.copy(), nodes.copy()


@functools.lru_cache(maxsize=None)
def _spread(step):
    adj, nodes = tutorial_connectome()
    return adj[::step, ::step].tocsr(), nodes.iloc[::step].reset_index(drop=True)


def real_spread(step: int = 10):
    """Every ``step``-th neuron of the whole tutorial connectome: 672 neurons from all layers over about 1.3 mm."""
    adj, nodes = _spread(step)
    return adj.copy(), nodes.copy()


def real_spread_pathway(step: int = 10):
    """The spread-out neurons split into a first half (sources) and a second half (targets)."""
    adj, nodes = real_spread(step)
    half = adj.shape[0] // 2
    return adj[:half, half:].tocsr(), nodes.iloc[:half].copy(), nodes.iloc[half:].copy()


def real_pathway(n_src: int = 250, n_tgt: int = 250):
    """Two disjoint populations of the tutorial connectome: sources are its first neurons, targets the following ones."""
    adj, nodes = real(n_src + n_tgt)
    return adj[:n_src, n_src:].tocsr(), nodes.iloc[:n_src].copy(), nodes.iloc[n_src:].copy()


def pairwise_distances(xyz_src, xyz_tgt=None):
    """Euclidean distance matrix by plain numpy broadcasting (independent of the library)."""
    xyz_tgt = xyz_src if xyz_tgt is None else xyz_tgt
    return np.sqrt(((xyz_src[:, None, :] - xyz_tgt[None, :, :]) ** 2).sum(-1))


def planted_dd(
    n: int,
    seed: int = 0,
    a: float = 0.5,
    b: float = 0.008,
    bipolar=None,
    dtype=bool,
    fmt: str = "csr",
):
    """Node table and digraph whose connection probability is ``a * exp(-b * d)`` (no self-loops).

    ``bipolar=(a2, b2)`` uses that second law for pairs whose source lies deeper than the target, and the average
    of both laws for pairs at equal depth (``support.data.node_table`` depths are rounded, so there are ties).
    """
    nodes = node_table(n, seed=seed)
    xyz = nodes[["x", "y", "z"]].to_numpy()
    p = a * np.exp(-b * pairwise_distances(xyz))
    if bipolar is not None:
        a2, b2 = bipolar
        depth = nodes["depth"].to_numpy()
        sign = np.sign(depth[:, None] - depth[None, :])
        p2 = a2 * np.exp(-b2 * pairwise_distances(xyz))
        p = np.where(sign > 0, p2, np.where(sign < 0, p, 0.5 * (p + p2)))
    dense = rng(seed + 5000).random_sample((n, n)) < p
    np.fill_diagonal(dense, False)
    return sp.csr_matrix(dense.astype(dtype)).asformat(fmt), nodes


def planted_pathway(
    n_src: int, n_tgt: int, seed: int = 0, a: float = 0.5, b: float = 0.008, bipolar=None
):
    """Source and target node tables with a (non-square) bool csr adjacency drawn from the same kind of law."""
    src, tgt = node_table(n_src, seed=seed), node_table(n_tgt, seed=seed + 1)
    d = pairwise_distances(src[["x", "y", "z"]].to_numpy(), tgt[["x", "y", "z"]].to_numpy())
    p = a * np.exp(-b * d)
    if bipolar is not None:
        a2, b2 = bipolar
        sign = np.sign(src["depth"].to_numpy()[:, None] - tgt["depth"].to_numpy()[None, :])
        p = np.where(
            sign > 0, a2 * np.exp(-b2 * d), np.where(sign < 0, p, 0.5 * (p + a2 * np.exp(-b2 * d)))
        )
    adj = sp.csr_matrix(rng(seed + 6000).random_sample((n_src, n_tgt)) < p)
    return adj, src, tgt


# ----------------------------------------------------------------------------- observing side effects


@contextlib.contextmanager
def logged():
    """Collect ``(level name, message)`` of the log records at WARNING or above emitted inside the block.

    The root logger's level and handlers depend on the runner (``logging.basicConfig`` at import of the module only
    configures it outside pytest), so only records that pass the default WARNING threshold are collected.
    """
    records: list = []

    class _Collect(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append((record.levelname, record.getMessage()))

    handler = _Collect(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)


@contextlib.contextmanager
def without_path(tmp: str):
    """Re-raise an exception with the temporary directory written as ``<tmp>`` in its message (same class).

    The golden file stores exception messages; a random temporary path in them would change it on every regeneration.
    """
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - the class is kept
        text = str(exc)
        if tmp not in text:
            raise
        raise type(exc)(text.replace(tmp, "<tmp>")) from None


def tree(root: str) -> list:
    """Sorted relative paths (with ``/``) of all files below ``root``."""
    return sorted(
        os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
        for d, _, files in os.walk(root)
        for f in files
    )


def pickles(root: str) -> dict:
    """Unpickled content of every ``*.pickle`` file below ``root``, by sorted relative path."""
    out = {}
    for name in tree(root):
        if name.endswith(".pickle"):
            with open(os.path.join(root, name), "rb") as fh:
                out[name] = pickle.load(fh)
    return out


def figures() -> dict:
    """Text and data of the open matplotlib figures (not pixels), then close them all."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    out = {}
    for num in plt.get_fignums():
        fig = plt.figure(num)
        suptitle = fig.get_suptitle() if hasattr(fig, "get_suptitle") else fig._suptitle.get_text()
        axes = []
        for ax in fig.axes:
            legend = ax.get_legend()
            axes.append(
                {
                    "title": ax.get_title(),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                    "yscale": ax.get_yscale(),
                    "legend": [t.get_text() for t in legend.get_texts()]
                    if legend is not None
                    else [],
                    "lines": [
                        [
                            ""
                            if ln.get_label().startswith("_")
                            else ln.get_label(),  # matplotlib names unlabelled artists
                            np.asarray(ln.get_xdata(), dtype=float),
                            np.asarray(ln.get_ydata(), dtype=float),
                            mcolors.to_hex(ln.get_color()),
                        ]
                        for ln in ax.get_lines()
                    ],
                    "bar_heights": np.array(
                        [p.get_height() for p in ax.patches if type(p).__name__ == "Rectangle"]
                    ),
                    "circles": [p.get_radius() for p in ax.patches if type(p).__name__ == "Circle"],
                    "texts": [t.get_text() for t in ax.texts],
                    "image": [
                        np.ma.filled(np.ma.asarray(im.get_array()), np.nan)[::50, ::50]
                        for im in ax.images
                    ],
                }
            )
        out[num] = {
            "size": list(fig.get_size_inches()),
            "dpi": float(fig.dpi),
            "suptitle": suptitle,
            "axes": axes,
        }
    plt.close("all")
    return out


def use_agg():
    """Make plotting headless and start from a clean slate (regenerate_golden.py does not set MPLBACKEND)."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    plt.close("all")
