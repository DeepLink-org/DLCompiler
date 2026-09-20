"""Assemble a Wafer-only wheel from pinned Python sources and audited binaries."""

import hashlib
import json
from pathlib import Path
import shutil
import sysconfig


def prepare_package(repo, staging, manifest_path, manifest, version, revision):
    expected_abi = manifest.get("python_soabi")
    if expected_abi != sysconfig.get_config_var("SOABI"):
        raise RuntimeError(f"Frontend Python ABI {expected_abi!r} differs from packager ABI")
    source = Path(manifest["frontend"]["source"])
    package = staging / "triton"
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.a", "*.o")
    shutil.copytree(source / "python/triton", package, ignore=ignored)
    # Keep the Python DICP dispatch entry point, but no Ascend language package
    # or original DICP C++ plugin. Vendor imports remain behind target branches.
    backend = package / "backends/dicp_triton"
    shutil.copytree(repo / "backend", backend, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.so", "*.a", "*.o", "dicp_opt", "bin"))
    language = package / "language/extra"
    for name in ("wafer", "txda"):
        # txda only re-exports the Wafer language API for existing callers.
        shutil.copytree(repo / "third_party/wafer/language" / name, language / name, ignore=ignored)
    shutil.copytree(repo / "third_party/wafer/experimental/tle", package / "experimental/tle", ignore=ignored)
    (package / "_C").mkdir(exist_ok=True)
    (package / "_C/__init__.py").touch()
    destinations = {
        "libtriton.so": package / "_C/libtriton.so",
        "FileCheck": package / "_C/FileCheck",
        "wafer-opt": backend / "bin/wafer-opt",
        "libvr.a": backend / "lib/libvr.a",
    }
    for name, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest["artifacts"][name]["path"], destination)
    init = package / "__init__.py"
    init.write_text(init.read_text().replace("__version__ = '3.5.0'", f"__version__ = {version!r}"))
    identity = {
        "schema": 1, "version": version, "dlcompiler_commit": revision,
        "package_backends": ["wafer"], "python_entry_point": "dicp_triton",
        "build": manifest,
        "python_sources": {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*.py"))
        },
    }
    (backend / "wafer-package.json").write_text(json.dumps(identity, indent=2) + "\n")
    shutil.copy2(manifest_path, backend / "bin/wafer-build.json")
    shutil.copy2(source / "LICENSE", staging / "LICENSE.triton")
    for name in ("LICENSE", "LICENSE.txt"):
        if (repo / name).exists():
            shutil.copy2(repo / name, staging / "LICENSE.dlcompiler")
            break
    (staging / "setup.py").write_text('''from pathlib import Path
from setuptools import Distribution, find_namespace_packages, setup

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

root = Path("triton")
setup(
    name="triton", version=VERSION,
    description="DLCompiler Wafer-only Triton compiler",
    packages=find_namespace_packages(include=["triton", "triton.*"]),
    package_data={"triton": [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]},
    include_package_data=False,
    distclass=BinaryDistribution,
    python_requires=">=3.10",
    install_requires=["setuptools>=40.8.0", "pybind11>=2.13.1"],
    entry_points={"triton.backends": ["dicp_triton = triton.backends.dicp_triton"]},
    license_files=["LICENSE.*"],
)
'''.replace('version=VERSION', 'version=' + repr(version)))
    return identity
