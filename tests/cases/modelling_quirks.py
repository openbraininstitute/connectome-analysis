"""Module-level behaviour of ``connalysis.modelling.modelling`` that is not a function result: what importing it does to
the logging configuration, the public names of the subpackage and the plotting constants."""

import os
import re
import subprocess
import sys
import tempfile

from support.cases import case

import connalysis.modelling as modelling_package
from connalysis.modelling import modelling as M


@case("modelling.import.logging_is_configured_for_the_whole_process")
def _():
    # Importing the module calls logging.basicConfig(level=INFO, handlers=[StreamHandler(sys.stdout)]) on the root logger.
    # That only takes effect when the root logger has no handler yet (not under pytest), so look at a fresh interpreter.
    code = (
        "import logging, sys\n"
        "import connalysis.modelling.modelling\n"
        "root = logging.getLogger()\n"
        "print('CONFIG', [root.level, [[type(h).__name__, h.level, h.formatter._fmt, h.stream is sys.stdout]"
        " for h in root.handlers]])\n"
        "logging.getLogger('some.other.library').info('hello from elsewhere')\n"
        "logging.getLogger('some.other.library').debug('not shown')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), MPLBACKEND="Agg")
    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp,
        )
    stamp = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}\]")
    lines = [stamp.sub("[<timestamp>]", line) for line in done.stdout.splitlines()]
    return {"returncode": done.returncode, "stdout_lines": lines}


@case("modelling.import.public_names_of_the_subpackage")
def _():
    names = sorted(modelling_package.__all__)
    return (
        names,
        [callable(getattr(modelling_package, n)) for n in names],
        [getattr(modelling_package, n) is getattr(M, n) for n in names],
    )


@case("modelling.import.helpers_used_by_the_tutorial_exist")
def _():
    return [
        callable(getattr(M, n))
        for n in (
            "_get_model_function",
            "_compute_dist_matrix",
            "_compute_dist_matrix_symmetric",
            "_compute_bip_matrix",
            "_extract_dependent_p_conn",
            "_generate_seeds",
            "_merge_data",
            "_save_data",
            "_get_data_part_name",
        )
    ]


@case("modelling.import.plot_style_constants")
def _():
    return {
        "data": M.DATA_COLOR,
        "model": M.MODEL_COLOR,
        "model2": M.MODEL_COLOR2,
        "colormap": M.PROB_CMAP.name,
    }
