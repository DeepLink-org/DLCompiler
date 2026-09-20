#!/usr/bin/env python3

import argparse
import functools
import os
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parents[1]
    parser = argparse.ArgumentParser(
        description="Compile representative Triton kernels with baseline and new Wafer compilers."
    )
    parser.add_argument(
        "--baseline-opt",
        type=Path,
        default=workspace_root / "DLCompiler/third_party/wafer/build_manual/install/bin/wafer-opt",
    )
    parser.add_argument(
        "--new-opt",
        type=Path,
        default=repo_root / "third_party/wafer/build_manual/third_party/wafer/bin/wafer-opt",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


@functools.lru_cache(maxsize=None)
def uses_wafer_pass_names(wafer_opt):
    help_text = subprocess.check_output([str(wafer_opt), "--help"], text=True)
    return "--mk-to-wafer" in help_text


def run_stage(wafer_opt, source, output, arguments):
    if not uses_wafer_pass_names(wafer_opt):
        raise RuntimeError(f"Compiler does not support Wafer pass names: {wafer_opt}")
    subprocess.run(
        [str(wafer_opt), str(source), *arguments, "-o", str(output)],
        check=True,
    )


def lower_case(wafer_opt, ttir_path, output_dir):
    coreir_path = output_dir / "coreir.mlir"
    wafer_ir_path = output_dir / "wafer_ir.mlir"
    llvm_path = output_dir / "llvm.mlir"
    run_stage(
        wafer_opt,
        ttir_path,
        coreir_path,
        [
            "--triton-to-core-dialects",
            "--tle-to-mk",
            "--dsa-memory-to-core",
            "--linalg-tiling",
            "--core-dialects-to-mk",
            "--linalg-fusion",
            "--legalize-tensor-form-loops",
            "--one-shot-bufferize",
            "--convert-bufferization-to-memref",
            "--cse",
            "--canonicalize",
        ],
    )
    run_stage(
        wafer_opt,
        coreir_path,
        wafer_ir_path,
        ["--spmd-allocate-shared-memory", "--expand-strided-metadata", "--lower-affine", "--mk-to-wafer", "--cse"],
    )
    run_stage(
        wafer_opt,
        wafer_ir_path,
        llvm_path,
        [
            "--wafer-memref-to-llvm",
            "--addr-to-llvm",
            "--convert-scf-to-cf",
            "--convert-math-to-llvm",
            "--convert-math-to-libm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--kernel-arg-buffer",
            "--wafer-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "--canonicalize",
            "--export-kernel-symbols",
        ],
    )
    if "llvm.func" not in llvm_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"LLVM dialect output has no llvm.func: {llvm_path}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (args.output_dir or Path(tempfile.mkdtemp(prefix="wafer-examples-", dir="/tmp"))).resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise RuntimeError("Regression artifacts must be stored outside the Git worktree")
    for compiler in (args.baseline_opt, args.new_opt):
        if not compiler.is_file():
            raise FileNotFoundError(compiler)

    os.environ["DICP_BACKEND"] = "wafer"
    os.environ["USE_SIM_MODE"] = "1"

    import triton
    import triton.language as tl
    from triton._C import libtriton
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    @triton.jit
    def addptr(in_ptr, out_ptr):
        for offset in range(0, 10, 2):
            first = in_ptr + 1 + offset
            second = first + 1
            tl.store(out_ptr + 1 + offset, tl.load(first))
            tl.store(out_ptr + 2 + offset, tl.load(second))

    @triton.jit
    def block_copy(in_ptr, out_ptr):
        source = tl.make_block_ptr(in_ptr + 8, (2, 2), (2, 1), (0, 0), (2, 2), (1, 0))
        destination = tl.make_block_ptr(out_ptr, (2, 2), (2, 1), (0, 0), (2, 2), (1, 0))
        tl.store(destination, tl.load(source, boundary_check=(0,)), boundary_check=(0,))

    @triton.jit
    def reduce_2d(in_ptr, out_ptr, stride, elements, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        block = tl.make_block_ptr(in_ptr, (elements * tl.num_programs(0),), (1,), (stride * row,), (BLOCK_SIZE,), (0,))
        tl.store(out_ptr + row, tl.sum(tl.load(block, boundary_check=(0,)), axis=0))

    @triton.jit
    def scan_1d(out_ptr, in_ptr, elements, M: tl.constexpr, N: tl.constexpr):
        offsets = tl.arange(0, M)
        values = tl.load(in_ptr + offsets, mask=offsets < elements, other=0.0)
        result = tl.cumsum(values).reshape((1, M)).broadcast_to((N, M))
        rows = tl.arange(0, N)
        columns = tl.arange(0, M)
        tl.store(out_ptr + M * rows[:, None] + columns[None, :], result, mask=columns[None, :] < elements)

    cases = {
        "addptr": ASTSource(addptr, {"in_ptr": "*fp32", "out_ptr": "*fp32"}, {}),
        "blockptr": ASTSource(block_copy, {"in_ptr": "*fp32", "out_ptr": "*fp32"}, {}),
        "reduce": ASTSource(
            reduce_2d,
            {"in_ptr": "*fp32", "out_ptr": "*fp32", "stride": "i32", "elements": "i32", "BLOCK_SIZE": "constexpr"},
            {"BLOCK_SIZE": 32},
        ),
        "scan": ASTSource(
            scan_1d,
            {"out_ptr": "*fp32", "in_ptr": "*fp32", "elements": "i32", "M": "constexpr", "N": "constexpr"},
            {"M": 32, "N": 2},
        ),
    }

    target = GPUTarget("wafer", "wafer", 32)
    backend = backends["dicp_triton"].compiler(target)
    options = backend.parse_options({})
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, source in cases.items():
        case_dir = output_dir / name
        case_dir.mkdir()
        module = source.make_ir(
            target,
            options,
            backend.get_codegen_implementation(options),
            backend.get_module_map(),
            context,
        )
        module = backend.make_ttir(module, {}, options)
        ttir_path = case_dir / "input.ttir.mlir"
        ttir_path.write_text(str(module), encoding="utf-8")

        for label, compiler in (("baseline", args.baseline_opt), ("new", args.new_opt)):
            result_dir = case_dir / label
            result_dir.mkdir()
            lower_case(compiler, ttir_path, result_dir)
            print(f"PASS {name}: {label}")

    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
