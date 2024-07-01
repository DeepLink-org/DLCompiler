import os

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import Autotuner

from dlblas.op_struct import OpImpl


def compile_and_bench(op: OpImpl, args):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
    '''
    kernel_file = op.file_path
    with open(kernel_file, 'r') as file:
        src_code = file.read()

    # dynamically written to a python file and compiled as a python module
    #
    # XXX
    # the mod is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time?
    #
    # mod = PyCodeCache.load(src_code, extra=str(counter))
    mod = PyCodeCache.load(src_code)
    PyCodeCache.clear()  # we want a fresh copy every time

    call_name = op.call.__name__
    bench_fn_name = op.bench_fn.__name__
    if isinstance(op.kernel, Autotuner):
        kernel_name = op.kernel.fn.__name__
    else:
        kernel_name = op.kernel.__name__

    # swap the impl
    op.call = getattr(mod, call_name)
    op.bench_fn = getattr(mod, bench_fn_name)
    op.kernel = getattr(mod, kernel_name)

    # actual bench
    perf = op.bench(*args)
    return perf
