import os

from torch._inductor.codecache import PyCodeCache

from dlblas.op_struct import OpImpl


def compile_and_bench(op: OpImpl, args):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
    '''
    # path-to-deeplink/python/dlBLAS/dlblas
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    kernel_file_name = op.call.__globals__['__name__']
    kernel_file = os.path.join(this_file_dir, 'kernels',
                               kernel_file_name + '.py')

    with open(kernel_file, 'r') as file:
        src_code = file.read()

    # dynamically written to a python file and compiled as a python module
    #
    # XXX
    # the file is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time?
    #
    # mod = PyCodeCache.load(src_code, extra=str(counter))
    mod = PyCodeCache.load(src_code)
    PyCodeCache.clear()  # we want a fresh copy every time

    call_name = op.call.__name__
    bench_fn_name = op.bench_fn.__name__
    kernel_name = op.kernel.__name__

    # swap the impl
    op.call = getattr(mod, call_name)
    op.bench_fn = getattr(mod, bench_fn_name)
    op.kernel = getattr(mod, kernel_name)

    # actual bench
    perf = op.bench(*args)
    return perf
