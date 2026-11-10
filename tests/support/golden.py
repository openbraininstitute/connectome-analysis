"""Reading and writing the golden files in ``tests/golden``."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"


def path_for(group: str) -> Path:
    return GOLDEN_DIR / f"{group}.json.gz"


def load(group: str) -> dict:
    path = path_for(group)
    if not path.exists():
        return {"_meta": {}, "snapshots": {}}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def save(group: str, snapshots: dict, meta: dict) -> Path:
    path = path_for(group)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": meta, "snapshots": dict(sorted(snapshots.items()))}
    with gzip.GzipFile(
        path, "wb", compresslevel=9, mtime=0
    ) as fh:  # mtime=0: byte-identical rewrites
        fh.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return path
