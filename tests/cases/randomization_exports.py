"""The public surface of ``connalysis.randomization``, ``.randomization`` and ``.rand_utils``.

``connalysis.randomization.__init__`` does ``from .randomization import *`` (no ``__all__`` in that module), so the package
also exposes whatever that module imported (``KDTree``, ``np``, ``LOG`` ...) and the four helpers it takes from
``rand_utils``; its own ``__all__`` lists only eight functions.  The "callables" cases pin all of it, the "defined in
connalysis" cases only the functions the library itself defines, which is the part users rely on.
"""

import importlib
import inspect

from support.cases import case


def _module(name):
    return importlib.import_module(name)


def _public_callables(module):
    return sorted(n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n)))


def _defined_in_connalysis(module):
    return sorted(
        n
        for n in dir(module)
        if not n.startswith("_")
        and inspect.isfunction(getattr(module, n))
        and getattr(getattr(module, n), "__module__", "").startswith("connalysis")
    )


def _signature(fn):
    """Parameter names, kinds and defaults (no annotations): what a caller can rely on."""
    return [
        [
            p.name,
            p.kind.name,
            "<required>" if p.default is inspect.Parameter.empty else repr(p.default),
        ]
        for p in inspect.signature(fn).parameters.values()
    ]


@case("randomization.exports.package_callables")
def _():
    return _public_callables(_module("connalysis.randomization"))


@case("randomization.exports.package_functions_defined_in_connalysis")
def _():
    return _defined_in_connalysis(_module("connalysis.randomization"))


@case("randomization.exports.package_all")
def _():
    return list(_module("connalysis.randomization").__all__)


@case("randomization.exports.package_star_import")
def _():
    namespace = {}
    exec("from connalysis.randomization import *", namespace)
    return sorted(n for n in namespace if n != "__builtins__")


@case("randomization.exports.package_submodules_are_attributes")
def _():
    pkg = _module("connalysis.randomization")
    return {
        name: inspect.ismodule(getattr(pkg, name, None)) for name in ("randomization", "rand_utils")
    }


@case("randomization.exports.randomization_module_callables")
def _():
    return _public_callables(_module("connalysis.randomization.randomization"))


@case("randomization.exports.randomization_module_functions_defined_in_connalysis")
def _():
    return _defined_in_connalysis(_module("connalysis.randomization.randomization"))


@case("randomization.exports.randomization_module_has_no_all")
def _():
    return hasattr(_module("connalysis.randomization.randomization"), "__all__")


@case("randomization.exports.rand_utils_callables")
def _():
    return _public_callables(_module("connalysis.randomization.rand_utils"))


@case("randomization.exports.rand_utils_functions_defined_in_connalysis")
def _():
    return _defined_in_connalysis(_module("connalysis.randomization.rand_utils"))


@case("randomization.exports.rand_utils_has_no_all")
def _():
    return hasattr(_module("connalysis.randomization.rand_utils"), "__all__")


@case("randomization.exports.private_helpers_that_the_cases_and_docs_use")
def _():
    mod = _module("connalysis.randomization.randomization")
    return {
        name: callable(getattr(mod, name, None)) for name in ("_dict_to_coo", "_seed_random_state")
    }


@case("randomization.exports.top_level_package_all")
def _():
    return list(_module("connalysis").__all__)


@case("randomization.exports.top_level_star_import_fails_on_rand_utils")
def _():
    # connalysis.__all__ lists "rand_utils", which is not an attribute of the top-level package
    namespace = {}
    exec("from connalysis import *", namespace)
    return sorted(n for n in namespace if n != "__builtins__")


@case("randomization.exports.signatures_of_the_randomization_module")
def _():
    mod = _module("connalysis.randomization.randomization")
    names = _defined_in_connalysis(mod) + ["_dict_to_coo"]
    return {n: _signature(getattr(mod, n)) for n in sorted(names)}


@case("randomization.exports.signatures_of_rand_utils")
def _():
    mod = _module("connalysis.randomization.rand_utils")
    return {n: _signature(getattr(mod, n)) for n in _defined_in_connalysis(mod)}
