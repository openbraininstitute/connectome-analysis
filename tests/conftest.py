import os

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports matplotlib

import pytest
from support import cases, golden


@pytest.fixture(scope="session")
def golden_store():
    """Golden snapshots by group, loaded on first use."""
    loaded: dict[str, dict] = {}

    def get(case: cases.Case) -> dict:
        if case.group not in loaded:
            loaded[case.group] = golden.load(case.group)["snapshots"]
        assert case.id in loaded[case.group], (
            f"no golden snapshot for {case.id}; generate it from the original package "
            "(see tests/golden/README.md)"
        )
        return loaded[case.group][case.id]

    return get
