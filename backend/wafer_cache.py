"""Content fingerprints shared by Wafer's compiler and native launcher caches."""

import functools
import hashlib
import json
from pathlib import Path


def cache_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@functools.lru_cache(maxsize=256)
def _content_digest(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return (
        str(path),
        _content_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    )
