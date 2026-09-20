#!/usr/bin/env python3
"""Freeze source, installed Python packages and the active Wafer link inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from triton.backends.dicp_triton import wafer, wafer_runtime
    import triton

    repo = Path(__file__).resolve().parents[2]
    linker, libraries = wafer._runtime_link_inputs()
    sdk = Path(os.environ["KUIPER_ROOT"])
    module = Path("/sys/bus/pci/devices/0000:3b:00.0/driver/module").resolve()
    version = module / "version"
    files = [
        args.wheel,
        linker,
        linker.parent / "riscv64-unknown-elf-ld",
        *libraries,
        wafer._find_wafer_opt(),
        wafer._find_llvm_tool("clang++"),
        wafer._find_llvm_tool("llvm-objcopy"),
        wafer._find_llvm_tool("mlir-translate"),
        Path(triton.__file__).parent / "_C/libtriton.so",
        Path(wafer.__file__),
        Path(wafer_runtime.__file__),
        sdk / "lib/libhpgr.so",
    ]
    files += sorted((sdk / "include").rglob("*.h"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "source_commit": run("git", "-C", str(repo), "rev-parse", "HEAD"),
        "source_status": run("git", "-C", str(repo), "status", "--short"),
        "submodules": run("git", "-C", str(repo), "submodule", "status"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "driver_module": str(module),
        "driver_version": version.read_text().strip() if version.exists() else None,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "modes": {
            name: os.getenv(name)
            for name in (
                "DICP_BACKEND",
                "USE_SIM_MODE",
                "WAFER_ENABLE_RUNTIME",
                "WAFER_DEVICE_LOG_ABI",
            )
        },
        "python_packages": sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
        ),
        "artifacts": [fingerprint(path) for path in files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.output}: {len(files)} artifacts")


if __name__ == "__main__":
    main()
