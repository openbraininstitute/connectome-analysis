# Golden files

`<group>.json.gz` holds, for every case of `tests/cases` in that group, what **connectome-analysis 1.2.0**
(commit `b93b82d`, the last release with the Poetry setup) returned: arrays, sparse matrices, pandas objects,
dictionaries and scalars, or the type of the exception it raised.  `tests/test_golden.py` runs the same cases against
the code under test and compares, with a relative tolerance of `1e-10` for floats and exactly for everything else
(shapes, dtypes, labels, indices, sparse structure).

They were generated on the stack the 1.2.0 release was locked to (Python 3.12, numpy 2.2.6, scipy 1.15.3,
pandas 2.3.3, statsmodels 0.14.6 for the one function that used it), with `numpy.product` restored, since two
functions of the original need it and numpy 2 removed it.  The current code reproduces them on that stack and on
scipy 1.18, numpy 2.5 and pandas 3.

The Betti counts that use flagser's approximation (cases tagged `machine_dependent`) are the exception: flagser skips
cells with a heuristic whose numbers differ between machines (CI's differ from the ones recorded here), although the
original and the rewrite give flagser identical matrices. Those cases pin the shape, index and dtypes of the result only;
`tests/test_equivalence.py` checks that the dimensions without approximation are exact.

## Regenerating

You should not need to: the files record the behaviour of a release, not of the current code.  If a case is added or
changed, regenerate its group from the original package:

```sh
git worktree add /tmp/connalysis-1.2.0 b93b82d
PYTHONPATH=/tmp/connalysis-1.2.0/src python tests/regenerate_golden.py --group <group>
```

The script refuses to run against the package in this repository, runs every case twice and rejects the ones whose
results differ, so a case cannot be added without being deterministic.

## Changes on purpose

A case whose result is deliberately not the original's is listed in `tests/support/intentional.py` with the reason
(a defect that made the original fail, an input that is accepted now, an accident of the package exports) and the new
behaviour is tested in `tests/test_changes.py`.
