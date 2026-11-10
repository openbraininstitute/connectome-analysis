"""Every registered case must give the same result as the original implementation (golden files)."""

import os

import numpy as np
import pytest
from support import canon, cases, golden
from support.intentional import CHANGES, NUMPY_1_DTYPES

CASES = cases.load_all(
    os.environ.get("CASES_PREFIX")
)  # CASES_PREFIX=topology runs one group while developing


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_same_as_original(case, golden_store):
    if case.id in CHANGES:
        pytest.skip(f"changed on purpose: {CHANGES[case.id]}")
    if case.id in NUMPY_1_DTYPES and np.lib.NumpyVersion(np.__version__) < "2.0.0":
        pytest.skip("the dtype of float32 results is that of numpy 2 (NEP 50 promotion)")
    expected = golden_store(case)
    actual = canon.snapshot(case.run)
    canon.assert_same(
        actual,
        expected,
        rtol=case.rtol,
        atol=case.atol,
        check_message=case.check_message,
        structure_only="machine_dependent" in case.tags,
    )


@pytest.mark.skipif(bool(os.environ.get("CASES_PREFIX")), reason="needs all cases")
def test_no_stale_golden_snapshots():
    ids = {c.id for c in CASES}
    for group in {c.group for c in CASES} | {
        p.name.split(".")[0] for p in golden.GOLDEN_DIR.glob("*.json.gz")
    }:
        stale = set(golden.load(group)["snapshots"]) - ids
        assert not stale, f"golden file {group} has snapshots without a case: {sorted(stale)[:5]}"


def test_intentional_changes_name_existing_cases():
    unknown = set(CHANGES) - {c.id for c in cases.load_all()}
    assert not unknown, (
        f"tests/support/intentional.py lists cases that do not exist: {sorted(unknown)}"
    )
