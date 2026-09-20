#!/usr/bin/env python3
"""Check an installed Wafer wheel and mode/cache separation without device work.

Run outside the repository root after sourcing init_wafer_env.sh. SDK and LLVM
are still required; the CRT archive must come from the installed wheel.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.pop("WAFER_RUNTIME_LIB_DIR", None)
    os.environ["USE_SIM_MODE"] = "0"

    import triton
    from triton.backends import _find_concrete_subclasses
    from triton.backends.compiler import BaseBackend, GPUTarget
    from triton.backends.dicp_triton import wafer, wafer_runtime
    from triton.compiler import ASTSource
    from verify_wafer_runtime import wafer_vector
    from audit_wafer_elf import audit_kernel

    repo = Path(__file__).resolve().parents[2]
    package = Path(triton.__file__).resolve().parent
    assert repo not in package.parents, f"Source checkout shadows installed wheel: {package}"
    assert wafer.TXDABackend is wafer.WaferBackend
    assert wafer.TXDAOptions is wafer.WaferOptions
    assert wafer_runtime.TXDALauncher is wafer_runtime.WaferLauncher
    assert wafer_runtime.TXDAUtils is wafer_runtime.WaferUtils
    assert _find_concrete_subclasses(wafer, BaseBackend) is wafer.WaferBackend
    canonical = importlib.import_module("triton.language.extra.wafer.libdevice")
    legacy = importlib.import_module("triton.language.extra.txda.libdevice")
    exports = [name for name in dir(canonical) if not name.startswith("_")]
    assert exports
    for name in exports:
        assert getattr(legacy, name) is getattr(canonical, name), name

    _, libraries = wafer._runtime_link_inputs()
    archive = libraries[3].resolve()
    assert archive == Path(wafer.__file__).resolve().parent / "lib/libvr.a", archive
    source = ASTSource(
        fn=wafer_vector,
        signature=dict(lhs="*fp32", rhs="*fp32", output="*fp32", alpha="fp32",
                       size="i32", BLOCK="constexpr"),
        constexprs=dict(BLOCK=256),
    )
    compiled = []
    for runtime in (False, True, False, True):
        os.environ["WAFER_ENABLE_RUNTIME"] = str(int(runtime))
        kernel = triton.compile(source, target=GPUTarget("wafer", "wafer", 32))
        extension = "so" if runtime else "o"
        assert list(kernel.asm) == ["source", "ttir", "coreir", "wafer_ir", "llir", extension]
        if runtime:
            audit_kernel(kernel.metadata.kernel_path, kernel.metadata.device_log_abi)
        compiled.append({"runtime": runtime, "hash": kernel.hash, "extension": extension})
    assert compiled[0] == compiled[2]
    assert compiled[1] == compiled[3]
    assert compiled[0]["hash"] != compiled[1]["hash"]
    report = {
        "installed_package": str(package),
        "version": importlib.metadata.version("triton"),
        "packaged_crt": str(archive),
        "legacy_language_exports_checked": len(exports),
        "mode_sequence": compiled,
        "device_work_submitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
