"""Private formatting helpers of ``topology`` (``_series_by_dim``, ``_frame_by_dim``) and the empty ``_convex_hull`` stub.

They are only called from inside the module, but they fix the index and column names of the counting functions' results.
"""

import numpy as np
import scipy.sparse as sp
from support.cases import case

from connalysis.network import topology as T

# ----------------------------------------------------------------------------- _series_by_dim


@case("topology._series_by_dim.none_gives_none")
def _():
    return T._series_by_dim(None, name_index="dim", name="count")


@case("topology._series_by_dim.default_index_and_names")
def _():
    return {
        "plain": T._series_by_dim([5, 6, 7]),
        "named": T._series_by_dim([5, 6, 7], name_index="dim", name="simplex_count"),
        "float_array": T._series_by_dim(np.array([0.5, 0.25]), name="fraction"),
        "numpy_ints": T._series_by_dim(np.array([4, 0, 2], dtype=np.int32), name_index="dim"),
    }


@case("topology._series_by_dim.explicit_index")
def _():
    return {
        "list_index": T._series_by_dim([5, 6, 7], index=[1, 2, 3], name_index="dim", name="x"),
        "array_index": T._series_by_dim(np.array([1, 2]), index=np.arange(1, 3)),
        "string_index": T._series_by_dim([1, 2], index=["a", "b"], name_index="label"),
    }


@case("topology._series_by_dim.empty")
def _():
    return [T._series_by_dim([]), T._series_by_dim([], name_index="dim", name="n")]


@case("topology._series_by_dim.index_of_another_length")
def _():
    return T._series_by_dim([5, 6, 7], index=[1, 2])


@case("topology._series_by_dim.scalar")
def _():
    return T._series_by_dim(5)


# ----------------------------------------------------------------------------- _frame_by_dim


@case("topology._frame_by_dim.none_gives_none")
def _():
    return T._frame_by_dim(None, 3, "node_participation", "node")


@case("topology._frame_by_dim.ragged_rows_are_padded_with_zeros")
def _():
    return T._frame_by_dim([[1, 2], [1], [1, 2, 3]], 3, "node_participation", "node")


@case("topology._frame_by_dim.column_index_gets_the_name_argument_not_the_row_index")
def _():
    # the fourth argument names the *columns*; the rows stay an unnamed range
    return {
        "node": T._frame_by_dim([[1, 2], [3]], 2, "x", "node"),
        "dim": T._frame_by_dim([[1, 2], [3]], 2, "x", "dim"),
        "none": T._frame_by_dim([[1, 2], [3]], 2, "x", None),
    }


@case("topology._frame_by_dim.values_are_cast_to_int")
def _():
    return {
        "floats": T._frame_by_dim([[1.5, 2.9], [3.2]], 2, "x", "dim"),
        "array_with_nan": T._frame_by_dim(np.array([[1.0, 2.0], [3.0, np.nan]]), 2, "x", "node"),
    }


@case("topology._frame_by_dim.empty")
def _():
    return [T._frame_by_dim([], 0, "x", "node"), T._frame_by_dim([], 3, "x", "node")]


@case("topology._frame_by_dim.more_columns_than_data")
def _():
    return T._frame_by_dim([[1, 2], [1]], 4, "x", "node")


@case("topology._frame_by_dim.fewer_columns_than_data")
def _():
    return T._frame_by_dim([[1, 2, 3], [1]], 2, "x", "node")


# ----------------------------------------------------------------------------- _convex_hull


@case("topology._convex_hull.is_an_empty_stub")
def _():
    return T._convex_hull(sp.csr_matrix((3, 3), dtype=bool), None)
