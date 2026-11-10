"""Write the golden files from the ORIGINAL package.

    PYTHONPATH=/path/to/original/src python tests/regenerate_golden.py [--only SUBSTRING] [--group GROUP]

The golden files record what connectome-analysis 1.2.0 (commit b93b82d) returned for every case in
``tests/cases``.  Run this only against that original package: regenerating from the current code would
make the comparison meaningless, so the script refuses unless ``connalysis`` is imported from outside
this repository's ``src``.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from support import canon, cases, golden  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", help="only cases whose id contains this text (others keep their snapshot)"
    )
    parser.add_argument("--group", help="only this group (golden file)")
    parser.add_argument(
        "--force", action="store_true", help="allow importing connalysis from this repository"
    )
    args = parser.parse_args()

    import numpy
    import pandas
    import scipy

    import connalysis

    if not hasattr(numpy, "product"):
        # numpy 2 removed np.product, which node_stats_participation / edge_stats_participation use.  It was an
        # alias of np.prod, so restoring it records what those functions returned before numpy 2.
        numpy.product = numpy.prod

    source = Path(connalysis.__file__).resolve()
    if (HERE.parent / "src") in source.parents and not args.force:
        print(
            f"connalysis is imported from {source}, which is this repository.\n"
            "Point PYTHONPATH at the original package's src directory (see --help).",
            file=sys.stderr,
        )
        return 2

    selected = [
        c
        for c in cases.load_all(args.group)
        if (not args.only or args.only in c.id) and (not args.group or c.group == args.group)
    ]
    meta = {
        "original": str(source.parent.parent),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
    }

    by_group: dict[str, dict] = {}
    unstable = []
    for case in selected:
        first, second = canon.snapshot(case.run), canon.snapshot(case.run)
        try:
            canon.assert_same(second, first, rtol=0.0, atol=0.0)
        except AssertionError:
            unstable.append(case.id)
        by_group.setdefault(case.group, {})[case.id] = first
    if unstable:
        print(
            "These cases are not deterministic (two runs differ); fix their seeding:\n  "
            + "\n  ".join(unstable),
            file=sys.stderr,
        )
        return 1

    for group, snapshots in by_group.items():
        existing = golden.load(group)
        merged = {**existing["snapshots"], **snapshots} if args.only else snapshots
        path = golden.save(group, merged, meta if not args.only else {**existing["_meta"], **meta})
        errors = sum(1 for s in snapshots.values() if s["t"] == "error")
        print(f"{path.name}: {len(snapshots)} snapshots written ({errors} record an exception)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
