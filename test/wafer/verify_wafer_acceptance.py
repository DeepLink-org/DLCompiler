#!/usr/bin/env python3

import argparse
import os
import tempfile
from pathlib import Path


def parse_args():
    workspace_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description="Compile a Triton GEMM through Wafer to LLVM dialect IR."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=workspace_root / "gemm_ll_0.mlir",
        help="LLVM dialect IR used for structural comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory outside the Git worktree (default: temporary directory).",
    )
    return parser.parse_args()


def require_markers(text, markers, label):
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"{label} is missing structural markers: {missing}")


def main():
    args = parse_args()
    output_dir = args.output_dir or Path(
        tempfile.mkdtemp(prefix="wafer-gemm-acceptance-", dir="/tmp")
    )
    output_dir = output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise RuntimeError("Acceptance artifacts must be stored outside the Git worktree")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.reference.is_file():
        raise FileNotFoundError(f"Reference IR not found: {args.reference}")

    os.environ["DICP_BACKEND"] = "wafer"
    os.environ["USE_SIM_MODE"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_DUMP_PATH"] = str(output_dir)

    import triton
    import triton.language as tl
    from triton._C import libtriton
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if "dicp_triton" not in backends:
        raise RuntimeError(f"dicp_triton backend was not discovered: {list(backends)}")

    target = GPUTarget("wafer", "wafer", 32)
    backend = backends["dicp_triton"].compiler(target)
    backend.load_dialects(libtriton.ir.context())

    import triton.language.extra.wafer  # noqa: F401

    @triton.jit
    def wafer_gemm(
        a_base,
        b_base,
        out,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.arange(0, BLOCK_M)
        columns = tl.arange(0, BLOCK_N)
        reduction = tl.arange(0, BLOCK_K)
        a_offsets = rows[:, None] * K + reduction[None, :]
        b_offsets = reduction[:, None] * N + columns[None, :]
        out_offsets = rows[:, None] * N + columns[None, :]
        a_ptrs = a_base + a_offsets
        b_ptrs = b_base + b_offsets
        out_ptrs = out + out_offsets
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b)
        tl.store(out_ptrs, accumulator)

    source = ASTSource(
        fn=wafer_gemm,
        signature={
            "a_base": "*bf16",
            "b_base": "*bf16",
            "out": "*bf16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
    )
    kernel = triton.compile(source, target=target)
    required_stages = {"ttir", "coreir", "wafer_ir", "llir"}
    missing_stages = sorted(required_stages.difference(kernel.asm))
    if missing_stages:
        raise RuntimeError(f"Wafer compilation is missing stages: {missing_stages}")

    llvm_mlir_path = output_dir / "llvm.mlir"
    if not llvm_mlir_path.is_file():
        raise RuntimeError(f"Wafer did not dump LLVM dialect IR: {llvm_mlir_path}")

    generated = llvm_mlir_path.read_text(encoding="utf-8")
    reference = args.reference.read_text(encoding="utf-8")
    common_markers = (
        "llvm.func",
        "@__Gemm",
        'section = "ExportedDYNSYMTab"',
        "triton_tsm.spm_use",
    )
    require_markers(generated, common_markers, "generated LLVM dialect IR")
    require_markers(reference, common_markers, "reference LLVM dialect IR")
    require_markers(generated, ("@wafer_gemm",), "generated LLVM dialect IR")

    print("Wafer GEMM compiler acceptance passed:")
    print(f"  triton: {triton.__file__}")
    print(f"  libtriton: {libtriton.__file__}")
    print(f"  backend: {target}")
    print(f"  stages: {list(kernel.asm)}")
    print(f"  llvm dialect IR: {llvm_mlir_path}")
    print(f"  reference: {args.reference.resolve()}")
    print(f"  structural markers: {', '.join(common_markers)}")


if __name__ == "__main__":
    main()