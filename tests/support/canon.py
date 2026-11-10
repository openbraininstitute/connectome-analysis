"""Snapshots of function results that can be stored as JSON and compared across library versions.

``snapshot(run)`` calls ``run()`` and turns whatever it returns (or raises) into plain JSON-able data.
``assert_same(actual, expected)`` compares two snapshots; floats use a tolerance, everything else
(shapes, dtypes, labels, integers, sparse structure, exception types) must match exactly.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

_STR_DTYPES = {"str", "string", "string[python]", "string[pyarrow]", "large_string[pyarrow]"}


def dtype_name(dtype) -> str:
    """Name of a numpy/pandas dtype; every string-like dtype maps to ``str``."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if isinstance(dtype, np.dtype):
        return "str" if dtype.kind in "US" else dtype.name
    name = str(dtype)
    return "str" if name in _STR_DTYPES else name


def _scalar_dtype(x) -> str:
    if isinstance(x, np.generic):
        return x.dtype.name
    return "bool" if isinstance(x, bool) else "int64" if isinstance(x, int) else "float64"


def _array(a: np.ndarray) -> dict:
    a = np.asarray(a)
    if a.dtype == object:
        values = a.ravel().tolist()
        if values and all(
            isinstance(v, str) for v in values
        ):  # pandas 2 keeps strings as object, pandas 3 as str
            return {"t": "ndarray", "dtype": "str", "shape": list(a.shape), "data": values}
        return {
            "t": "ndarray",
            "dtype": "object",
            "shape": list(a.shape),
            "data": [snap(v) for v in values],
        }
    if a.dtype.kind == "c":
        raise NotImplementedError("complex arrays are not supported")
    return {
        "t": "ndarray",
        "dtype": dtype_name(a.dtype),
        "shape": list(a.shape),
        "data": a.ravel().tolist(),
    }


def _sparse(m) -> dict:
    coo = m.tocoo(copy=True)
    coo.sum_duplicates()
    coo.eliminate_zeros()
    order = np.lexsort((coo.col, coo.row))
    return {
        "t": "sparse",
        "cls": type(m).__name__,
        "shape": list(m.shape),
        "dtype": dtype_name(m.dtype),
        "row": coo.row[order].astype(np.int64).tolist(),
        "col": coo.col[order].astype(np.int64).tolist(),
        "data": coo.data[order].tolist(),
    }


def _index(idx: pd.Index) -> dict:
    if isinstance(idx, pd.MultiIndex):
        return {
            "t": "index",
            "multi": True,
            "names": list(idx.names),
            "dtypes": [dtype_name(idx.get_level_values(i).dtype) for i in range(idx.nlevels)],
            "data": [snap(tuple(v)) for v in idx.tolist()],
        }
    values = idx.tolist()
    dtype = dtype_name(idx.dtype)
    if dtype == "object" and values and all(isinstance(v, str) for v in values):
        dtype = "str"
    return {
        "t": "index",
        "multi": False,
        "name": idx.name,
        "dtype": dtype,
        "data": [snap(v) for v in values],
    }


def snap(x: Any) -> Any:
    """Turn ``x`` into JSON-able data (see module docstring)."""
    if x is None:
        return {"t": "none"}
    if isinstance(x, (bool, np.bool_)):
        return {"t": "scalar", "dtype": "bool", "v": bool(x)}
    if isinstance(x, (int, np.integer)):
        return {"t": "scalar", "dtype": _scalar_dtype(x), "v": int(x)}
    if isinstance(x, (float, np.floating)):
        return {"t": "scalar", "dtype": _scalar_dtype(x), "v": float(x)}
    if isinstance(x, str):
        return {"t": "str", "v": str(x)}
    if isinstance(x, np.matrix):
        return {**_array(np.asarray(x)), "t": "matrix"}
    if isinstance(x, np.ndarray):
        return _array(x)
    if sp.issparse(x):
        return _sparse(x)
    if isinstance(x, pd.DataFrame):
        columns = [_array(x.iloc[:, i].to_numpy()) for i in range(x.shape[1])]
        return {
            "t": "frame",
            "index": _index(x.index),
            "columns": _index(x.columns),
            "dtypes": [c["dtype"] for c in columns],
            "data": columns,
        }
    if isinstance(x, pd.Series):
        values = _array(x.to_numpy())
        return {
            "t": "series",
            "name": snap(x.name),
            "index": _index(x.index),
            "dtype": values["dtype"],
            "data": values,
        }
    if isinstance(x, pd.Index):
        return _index(x)
    if isinstance(x, dict):
        return {"t": "dict", "data": [[snap(k), snap(v)] for k, v in x.items()]}
    if isinstance(x, (list, tuple)):
        return {"t": type(x).__name__, "data": [snap(v) for v in x]}
    if isinstance(x, (set, frozenset)):
        return {"t": "set", "data": sorted((snap(v) for v in x), key=repr)}
    if isinstance(x, nx.Graph):
        return {
            "t": "nxgraph",
            "directed": x.is_directed(),
            "nodes": [snap(n) for n in x.nodes],
            "edges": [[snap(u), snap(v)] for u, v in x.edges],
        }
    raise TypeError(f"cannot snapshot {type(x).__name__}")


def snapshot(run: Callable[[], Any]) -> dict:
    """Run ``run`` (with the global numpy seed fixed) and snapshot its result or its exception."""
    np.random.seed(12345)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - failures are part of the behaviour being recorded
        return {"t": "error", "cls": type(exc).__name__, "msg": str(exc)}
    return {"t": "ok", "value": snap(result)}


# ----------------------------------------------------------------------------- comparison


def _is_float(dtype: str) -> bool:
    return dtype.startswith("float")


def _close(a: list, e: list, rtol: float, atol: float) -> bool:
    a_arr, e_arr = np.asarray(a, dtype=float), np.asarray(e, dtype=float)
    if a_arr.shape != e_arr.shape:
        return False
    return bool(np.allclose(a_arr, e_arr, rtol=rtol, atol=atol, equal_nan=True))


def _first_difference(a: list, e: list) -> str:
    for i, (x, y) in enumerate(zip(a, e)):
        if x != y and not (x != x and y != y):
            return f"first difference at flat index {i}: {x!r} != {y!r}"
    return f"lengths {len(a)} != {len(e)}"


def _diff(
    a: Any,
    e: Any,
    path: str,
    rtol: float,
    atol: float,
    out: list[str],
    structure_only: bool = False,
) -> None:
    if len(out) >= 8:
        return
    if type(a) is not type(e) and not (isinstance(a, (int, float)) and isinstance(e, (int, float))):
        out.append(f"{path}: type {type(a).__name__} != {type(e).__name__}")
        return
    if isinstance(e, dict):
        if set(a) != set(e):
            out.append(f"{path}: keys {sorted(a)} != {sorted(e)}")
            return
        kind = e.get("t")
        for key in e:
            if key == "data" and kind in ("ndarray", "matrix", "sparse"):
                continue
            _diff(a[key], e[key], f"{path}.{key}", rtol, atol, out, structure_only)
        if kind in ("ndarray", "matrix", "sparse") and not out and not structure_only:
            dtype, a_data, e_data = e["dtype"], a["data"], e["data"]
            if kind != "sparse" and dtype == "object":
                _diff(a_data, e_data, f"{path}.data", rtol, atol, out)
            elif _is_float(dtype):
                if not _close(a_data, e_data, rtol, atol):
                    diff = np.abs(np.asarray(a_data, float) - np.asarray(e_data, float))
                    out.append(
                        f"{path}: float data differs (max abs diff {np.nanmax(diff) if diff.size else 0:.3g})"
                    )
            elif a_data != e_data:
                out.append(f"{path}: {_first_difference(a_data, e_data)}")
    elif isinstance(e, list):
        if len(a) != len(e):
            out.append(f"{path}: length {len(a)} != {len(e)}")
            return
        for i, (x, y) in enumerate(zip(a, e)):
            _diff(x, y, f"{path}[{i}]", rtol, atol, out, structure_only)
    elif isinstance(e, float) or isinstance(a, float):
        if not _close([a], [e], rtol, atol):
            out.append(f"{path}: {a!r} != {e!r}")
    elif a != e:
        out.append(f"{path}: {a!r} != {e!r}")


def assert_same(
    actual: dict,
    expected: dict,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    check_message: bool = False,
    structure_only: bool = False,
) -> None:
    """Fail with a short description of the first differences between two snapshots.

    ``structure_only`` leaves the values of arrays out: types, shapes, dtypes and indexes still have to match.
    """
    if expected["t"] == "error" and actual["t"] == "error":
        problems = (
            []
            if actual["cls"] == expected["cls"]
            else [f"raised {actual['cls']}, expected {expected['cls']}"]
        )
        if check_message and actual["msg"] != expected["msg"]:
            problems.append(f"message {actual['msg']!r} != {expected['msg']!r}")
    elif expected["t"] != actual["t"]:
        got = actual.get("cls", "a result") if actual["t"] == "error" else "a result"
        want = expected.get("cls", "a result") if expected["t"] == "error" else "a result"
        detail = f" ({actual['msg']})" if actual["t"] == "error" else ""
        problems = [f"got {got}{detail}, expected {want}"]
    else:
        problems = []
        _diff(actual["value"], expected["value"], "$", rtol, atol, problems, structure_only)
    assert not problems, "result differs from the original implementation:\n  " + "\n  ".join(
        problems
    )
