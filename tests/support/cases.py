"""Registry of behaviour cases.

A case is a zero-argument function that builds its input deterministically and calls the public API.
The same case runs against the original implementation (to write the golden file) and against the
current one (to check it).  Case ids are ``<group>.<function>.<variant>``; the group names the golden file.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

REGISTRY: dict[str, "Case"] = {}


@dataclass(frozen=True)
class Case:
    id: str
    run: Callable[[], Any]
    rtol: float = 1e-10
    atol: float = 1e-12
    check_message: bool = False
    tags: frozenset = field(default_factory=frozenset)

    @property
    def group(self) -> str:
        return self.id.split(".", 1)[0]


def case(
    case_id: str, *, rtol: float = 1e-10, atol: float = 1e-12, check_message: bool = False, tags=()
):
    """Register the decorated zero-argument function as a behaviour case."""

    def register(fn: Callable[[], Any]) -> Callable[[], Any]:
        assert case_id not in REGISTRY, f"duplicate case id {case_id}"
        REGISTRY[case_id] = Case(case_id, fn, rtol, atol, check_message, frozenset(tags))
        return fn

    return register


def load_all(prefix: str | None = None) -> list[Case]:
    """Import the modules of ``tests/cases`` (only those starting with ``prefix``, if given) and return their cases.

    The result is sorted by id and, with a prefix, limited to the cases whose group starts with it.
    """
    import cases as cases_pkg  # tests/cases, on sys.path through pytest's ``pythonpath``

    for module in pkgutil.iter_modules(cases_pkg.__path__):
        if not prefix or module.name.startswith(prefix):
            importlib.import_module(f"cases.{module.name}")
    return [
        REGISTRY[k] for k in sorted(REGISTRY) if not prefix or REGISTRY[k].group.startswith(prefix)
    ]
