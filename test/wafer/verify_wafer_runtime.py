#!/usr/bin/env python3
"""Numerical Wafer acceptance through the installed CompiledKernel launcher.

Run outside the source root after sourcing the workspace environment. Requires
USE_SIM_MODE=0, WAFER_ENABLE_RUNTIME=1, WAFER_RUNTIME_LIB_DIR, and a matching
WAFER_DEVICE_LOG_ABI (rcs on Kuiper 1.4). --torch uses actual TXDA tensors.
"""

import argparse
import ctypes
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import sys
import time

import numpy as np
import triton
from triton import knobs
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/wafer"))
from audit_wafer_elf import audit_kernel


@triton.jit
def wafer_vector(lhs, rhs, output, alpha, size, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < size
    value = tl.load(lhs + offset, mask=mask) + tl.load(rhs + offset, mask=mask) + alpha
    tl.store(output + offset, value, mask=mask)


@triton.jit
def wafer_reduction(values, output, size, BLOCK: tl.constexpr):
    offset = tl.arange(0, BLOCK)
    values = tl.load(values + offset, mask=offset < size, other=0.0)
    tl.store(output, tl.sum(values, 0))


@triton.jit
def wafer_matmul(
    lhs, rhs, output, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    inner = tl.arange(0, BK)
    a = tl.load(lhs + rows[:, None] * K + inner[None, :])
    b = tl.load(rhs + inner[:, None] * N + cols[None, :])
    result = tl.dot(a, b)
    tl.store(output + rows[:, None] * N + cols[None, :], result)


def compile_kernel(fn, signature, constants):
    source = ASTSource(fn=fn, signature=signature, constexprs=constants)
    started = time.monotonic()
    kernel = triton.compile(source, target=GPUTarget("wafer", "wafer", 32))
    assert list(kernel.asm) == ["source", "ttir", "coreir", "wafer_ir", "llir", "so"]
    audit_kernel(kernel.metadata.kernel_path, kernel.metadata.device_log_abi)
    print(
        f"compiled {kernel.name} in {time.monotonic() - started:.3f}s: {kernel.metadata.kernel_path}",
        flush=True,
    )
    return kernel


class RawRuntime:
    def __init__(self):
        root = Path(os.environ["KUIPER_ROOT"])
        self.library = ctypes.CDLL(str(root / "lib/libhpgr.so"))
        for name, argtypes in {
            "txSetDevice": [ctypes.c_uint32],
            "txMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64],
            "txFree": [ctypes.c_void_p],
            "txMemcpy": [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int,
            ],
        }.items():
            fn = getattr(self.library, name)
            fn.argtypes, fn.restype = argtypes, ctypes.c_int
        self.check(self.library.txSetDevice(0), "txSetDevice")

    @staticmethod
    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation} failed with status 0x{status:x}")

    @contextmanager
    def buffer(self, host):
        pointer = ctypes.c_void_p()
        self.check(
            self.library.txMalloc(ctypes.byref(pointer), host.nbytes), "txMalloc"
        )
        buffer = RawBuffer(self, pointer, host)
        failed = False
        try:
            self.check(
                self.library.txMemcpy(
                    pointer, ctypes.c_void_p(host.ctypes.data), host.nbytes, 1
                ),
                "H2D",
            )
            yield buffer
        except BaseException:
            failed = True
            raise
        finally:
            status = self.library.txFree(pointer)
            if status and failed:
                print(f"cleanup txFree also failed: 0x{status:x}", flush=True)
            else:
                self.check(status, "txFree")


class RawBuffer:
    def __init__(self, runtime, pointer, host):
        self.runtime, self.pointer, self.host = runtime, pointer, host

    def data_ptr(self):
        return self.pointer.value

    def reset(self):
        self.runtime.check(
            self.runtime.library.txMemcpy(
                self.pointer,
                ctypes.c_void_p(self.host.ctypes.data),
                self.host.nbytes,
                1,
            ),
            "reset output H2D",
        )

    def cpu(self):
        result = np.empty_like(self.host)
        self.runtime.check(
            self.runtime.library.txMemcpy(
                ctypes.c_void_p(result.ctypes.data), self.pointer, result.nbytes, 2
            ),
            "D2H",
        )
        return result


def invoke(kernel, args, grid, iterations, reset, verify):
    calls = []
    saved = (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    knobs.runtime.launch_enter_hook = lambda metadata: calls.append(
        ("enter", metadata.get()["name"])
    )
    knobs.runtime.launch_exit_hook = lambda metadata: calls.append(
        ("exit", metadata.get()["name"])
    )
    try:
        reset()
        kernel[grid](*args)
        verify()
        launcher, module = kernel._run, kernel.module
        memory_before = free_device_memory()
        elapsed = 0.0
        for _ in range(iterations - 1):
            reset()
            started = time.monotonic()
            kernel[grid](*args)
            elapsed += time.monotonic() - started
            verify()
        print(
            f"repeat launch mean={elapsed / (iterations - 1) * 1000:.3f}ms; device free memory change={free_device_memory() - memory_before} bytes",
            flush=True,
        )
        assert (
            kernel._run is launcher and kernel.module is module and module is not None
        )
        assert calls == [
            (event, kernel.name)
            for _ in range(iterations)
            for event in ("enter", "exit")
        ]
    finally:
        knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook = saved


def free_device_memory():
    library = ctypes.CDLL(str(Path(os.environ["KUIPER_ROOT"]) / "lib/libhpgr.so"))
    query = library.txMemGetInfo
    query.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    free, total = ctypes.c_uint64(), ctypes.c_uint64()
    RawRuntime.check(query(ctypes.byref(free), ctypes.byref(total)), "txMemGetInfo")
    return free.value


def run_case(case, iterations, torch_mode, compile_only=False):
    if case in ("vector", "grid"):
        size = 256 if case == "vector" else 700
        hosts = [
            np.arange(size, dtype=np.float32) * 0.5,
            np.arange(size, dtype=np.float32)[::-1].copy() * 0.25,
            np.zeros(size, dtype=np.float32),
        ]
        expected = hosts[0] + hosts[1] + np.float32(1.25)
        kernel = compile_kernel(
            wafer_vector,
            dict(
                lhs="*fp32",
                rhs="*fp32",
                output="*fp32",
                alpha="fp32",
                size="i32",
                BLOCK="constexpr",
            ),
            dict(BLOCK=256),
        )
        scalars, grid = [1.25, size], (triton.cdiv(size, 256), 1, 1)
    elif case == "reduction":
        hosts = [np.arange(1, 257, dtype=np.float32), np.zeros(1, dtype=np.float32)]
        expected = np.array([32896.0], dtype=np.float32)
        kernel = compile_kernel(
            wafer_reduction,
            dict(values="*fp32", output="*fp32", size="i32", BLOCK="constexpr"),
            dict(BLOCK=256),
        )
        scalars, grid = [256], (1, 1, 1)
    else:
        hosts = [
            np.full((128, 64), 0x3F80, dtype=np.uint16),
            np.full((64, 256), 0x3F80, dtype=np.uint16),
            np.zeros((128, 256), dtype=np.uint16),
        ]
        expected = np.full((128, 256), 0x4280, dtype=np.uint16)
        kernel = compile_kernel(
            wafer_matmul,
            dict(
                lhs="*bf16",
                rhs="*bf16",
                output="*bf16",
                M="i32",
                N="i32",
                K="i32",
                BM="constexpr",
                BN="constexpr",
                BK="constexpr",
            ),
            dict(BM=128, BN=256, BK=64),
        )
        scalars, grid = [128, 256, 64], (1, 1, 1)

    if compile_only:
        print(f"PASS compile/link/ELF audit: {case}; no device launch", flush=True)
        return

    with ExitStack() as stack:
        if torch_mode:
            import torch
            import torch_txda  # noqa: F401
            from triton.backends.dicp_triton.wafer_runtime import get_runtime

            assert get_runtime() is torch.txda
            buffers = [
                (
                    torch.from_numpy(host).view(torch.bfloat16).to("txda")
                    if host.dtype == np.uint16
                    else torch.from_numpy(host).to("txda")
                )
                for host in hosts
            ]
            stream = torch.txda.Stream()
            stack.enter_context(torch.txda.stream(stream))
            from triton.runtime import driver

            assert driver.active.get_current_stream(0) == stream.txda_stream
            args = buffers + scalars
            reset_host = torch.from_numpy(hosts[-1])
            if hosts[-1].dtype == np.uint16:
                reset_host = reset_host.view(torch.bfloat16)

            def reset():
                buffers[-1].copy_(reset_host)

        else:
            runtime = RawRuntime()
            buffers = [stack.enter_context(runtime.buffer(host)) for host in hosts]
            # Exercise both integer pointers and data_ptr() objects in one launch.
            args = [buffers[0].data_ptr(), *buffers[1:], *scalars]
            reset = buffers[-1].reset

        def verify():
            if torch_mode:
                stream.synchronize()
                output = buffers[-1].cpu()
                result = (
                    output.view(torch.uint16).numpy()
                    if hosts[-1].dtype == np.uint16
                    else output.numpy()
                )
            else:
                result = buffers[-1].cpu()
            np.testing.assert_array_equal(result, expected)

        invoke(kernel, args, grid, iterations, reset, verify)
        print(
            f"PASS {case}: {expected.size} elements, grid={grid}, iterations={iterations}, "
            f"{'TXDA tensor/non-default stream' if torch_mode else 'raw pointers/data_ptr'}, hooks and every iteration verified",
            flush=True,
        )


def run_jit(iterations, compile_only=False):
    import torch
    from triton.runtime import driver

    if compile_only:
        active = driver.active
        saved = active.get_current_device, active.get_current_stream
        active.get_current_device = lambda: 0
        active.get_current_stream = lambda device: None
        try:
            for size in (700, 1):
                # warmup specializes tensor dtypes and arguments without loading
                # a device binary. CPU tensors are sufficient for this check.
                values = torch.empty(size, dtype=torch.float32)
                kernel = wafer_vector.warmup(
                    values,
                    values,
                    values,
                    1.25,
                    size,
                    BLOCK=256,
                    grid=(triton.cdiv(size, 256),),
                )
                audit_kernel(
                    kernel.metadata.kernel_path, kernel.metadata.device_log_abi
                )
                assert ((4,) in kernel.src.constants) == (size == 1)
                print(
                    f"PASS JIT compile/link/ELF audit: size={size}, constexpr/specialization; no device launch",
                    flush=True,
                )
        finally:
            active.get_current_device, active.get_current_stream = saved
        return

    import torch_txda  # noqa: F401

    assert driver.active.get_active_torch_device() == torch.device("txda", 0)
    stream = torch.txda.Stream()
    for size in (700, 1):
        host = torch.arange(size, dtype=torch.float32)
        lhs, rhs = host.to("txda"), (host * 0.5).to("txda")
        output = torch.empty_like(lhs)
        prepared = wafer_vector.warmup(
            lhs, rhs, output, 1.25, size, BLOCK=256, grid=(triton.cdiv(size, 256),)
        )
        audit_kernel(prepared.metadata.kernel_path, prepared.metadata.device_log_abi)
        expected = host * 1.5 + 1.25
        reset_host = torch.zeros_like(host)
        with torch.txda.stream(stream):
            output.copy_(reset_host)
            kernel = wafer_vector[(triton.cdiv(size, 256),)](
                lhs, rhs, output, 1.25, size, BLOCK=256
            )
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            launcher = kernel._run
            memory_before = free_device_memory()
            elapsed = 0.0
            for _ in range(iterations - 1):
                output.copy_(reset_host)
                started = time.monotonic()
                cached = wafer_vector[(triton.cdiv(size, 256),)](
                    lhs, rhs, output, 1.25, size, BLOCK=256
                )
                elapsed += time.monotonic() - started
                assert cached is kernel and cached._run is launcher
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            stream.synchronize()
            print(
                f"repeat JIT launch mean={elapsed / (iterations - 1) * 1000:.3f}ms; device free memory change={free_device_memory() - memory_before} bytes",
                flush=True,
            )
        assert kernel.metadata.device_log_abi == os.environ["WAFER_DEVICE_LOG_ABI"]
        print(
            f"PASS JIT: size={size}, constexpr=256, iterations={iterations}, specialization/cache reuse and every iteration verified",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("vector", "grid", "reduction", "gemm", "jit", "all"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile, link and audit without allocating or launching on the device",
    )
    args = parser.parse_args()
    if args.case == "jit" and not args.torch:
        parser.error("--case jit requires --torch")
    if args.iterations < 2:
        parser.error("--iterations must be at least 2 to verify initialization reuse")
    if os.getenv("USE_SIM_MODE") != "0" or os.getenv("WAFER_ENABLE_RUNTIME") != "1":
        parser.error(
            "Set USE_SIM_MODE=0 and WAFER_ENABLE_RUNTIME=1 before starting Python"
        )
    started = time.monotonic()
    cases = (
        ("vector", "reduction", "gemm", "grid") if args.case == "all" else (args.case,)
    )
    if args.case == "all" and args.torch:
        cases += ("jit",)
    for case in cases:
        if case == "jit":
            run_jit(args.iterations, args.compile_only)
        else:
            run_case(case, args.iterations, args.torch, args.compile_only)
    print(
        f"{'Compile-only checks' if args.compile_only else 'Hardware acceptance'} passed in {time.monotonic() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
