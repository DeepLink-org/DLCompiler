#!/usr/bin/env python3
"""Package the verified Wafer-only build without invoking DLCompiler setup.py."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/wafer"))
from wafer_artifacts import read_manifest
from package_wafer import prepare_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path,
                        default=Path(os.getenv("WAFER_BUILD_DIR", ROOT / "third_party/wafer/build_manual")))
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    manifest_path = build / "wafer-build.json"
    manifest = read_manifest(manifest_path)
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    version = "3.5.0+wafer.git" + revision[:8]
    output = args.wheel_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # All staging is private to this invocation. Never remove ROOT/build or
    # install over the caller's existing Triton environment.
    with tempfile.TemporaryDirectory(prefix="wafer-package-", dir=build) as temporary:
        staging = Path(temporary)
        identity = prepare_package(ROOT, staging, manifest_path, manifest, version, revision)
        subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)],
                       cwd=staging, check=True)
    (output / "wafer-package.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
