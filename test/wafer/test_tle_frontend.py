"""Compile the TLE frontend against the installed Wafer/Triton IR bindings."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


@triton.jit
def local_kernel(out):
    buf = tle.dsa.alloc((16,), tl.float32)
    i = tl.arange(0, 16)
    ptr = tle.dsa.local_ptr(buf, [i])
    tl.store(ptr, i.to(tl.float32))
    tl.store(out + i, tl.load(ptr))


@triton.jit
def buffer_helper(buf, out):
    i = tl.arange(0, 16)
    tl.store(out + i, tl.load(tle.dsa.local_ptr(buf, [i])))


@triton.jit
def helper_kernel(out):
    buf = tle.dsa.alloc((16,), tl.float32)
    buffer_helper(buf, out)


@triton.jit
def remote_kernel(out):
    buf = tle.dsa.alloc((16,), tl.float32)
    remote = tle.remote(buf, tl.program_id(0))
    ptr = tle.dsa.local_ptr(remote, [tl.arange(0, 16)])
    tl.store(ptr, tl.full((16,), 1, tl.float32))


@triton.jit
def copy_kernel(out):
    src = tle.dsa.alloc((16,), tl.float32)
    dst = tle.dsa.alloc((16,), tl.float32)
    tle.dsa.copy(src, dst, (16,))
    tle.dsa.copy(dst, out, (16,))


@triton.jit
def barrier_kernel(out):
    tle.distributed_barrier()


@triton.jit
def invalid_alloc_kernel(out):
    tle.dsa.alloc((-1,), tl.float32)


@triton.jit
def pipeline_kernel(out):
    for i in tle.dsa.pipeline(0, 16, num_stages=2):
        tl.store(out + i, i.to(tl.float32))


def make_ttir(fn):
    target = GPUTarget("wafer", "wafer", 32)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    src = ASTSource(fn, signature={"out": "*fp32"})
    return str(src.make_ir(target, options, backend.get_codegen_implementation(options),
                           backend.get_module_map(), context))


@pytest.mark.parametrize("kernel,operations", [
    (local_kernel, ["dsa.alloc", "dsa.local_pointers", "tt.load", "tt.store"]),
    (helper_kernel, ["dsa.alloc", "tt.call", "dsa.local_pointers"]),
    (remote_kernel, ["dsa.remote_pointers", "tt.get_program_id"]),
    (copy_kernel, ["dsa.copy"]),
    (pipeline_kernel, ["scf.for", "tt.num_stages = 2"]),
])
def test_tle_frontend(kernel, operations):
    module = make_ttir(kernel)
    for operation in operations:
        assert operation in module


@pytest.mark.parametrize("kernel,message", [
    (barrier_kernel, "cross-tile CRT implementation"),
    (invalid_alloc_kernel, "must be a positive integer"),
])
def test_tle_rejects_unsupported_semantics(kernel, message):
    with pytest.raises(triton.compiler.errors.CompilationError, match=message):
        make_ttir(kernel)
