"""Identity and integrity checks shared by the isolated build and wheel packager."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != 2 or data.get("layout") != "wafer-only-frontend-tools":
        raise RuntimeError(f"Not an isolated Wafer build manifest: {path}")
    if data["frontend"]["source"] == data["tools"]["source"]:
        raise RuntimeError("Frontend and tools must use separate Triton source trees")
    if data["frontend"]["triton_commit"] != data["tools"]["triton_commit"]:
        raise RuntimeError("Frontend and tools have different Triton pins")
    for name in ("libtriton.so", "FileCheck", "wafer-opt", "libvr.a"):
        entry = data["artifacts"][name]
        if not Path(entry["path"]).is_file() or sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Missing or changed build artifact: {name}")
    source = Path(data["frontend"]["source"])
    for relative, expected in data["frontend"]["files"].items():
        if sha256(source / relative) != expected:
            raise RuntimeError(f"Frontend source no longer matches the binary: {relative}")
    return data
