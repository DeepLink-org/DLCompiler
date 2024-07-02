import os
import re
import tempfile
from typing import Union

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import Autotuner

from dlblas.op_struct import OpImpl
from dlblas.cache import TritonCompiledKernel


def compile_and_bench(op: OpImpl, args):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
    '''
    compile_op(op)
    perf = op.bench(*args)
    return perf


def compile_op(op: Union[OpImpl, TritonCompiledKernel]):
    kernel_file = op.file_path
    with open(kernel_file, 'r') as file:
        src_code = file.readlines()

    # a hack to avoid re-register the func; XXX this seems to be significant overhead
    buffer = []
    for line in src_code:
        # Replace the line with 'pass'
        if 'register_dlblas_op' in line and 'import' not in line:
            tmp = re.sub(r'register_dlblas_op.*?(?=\n|$)',
                         'pass\n',
                         line,
                         flags=re.MULTILINE)
            buffer.append(tmp)
        else:
            buffer.append(line)  # Keep the line as is

    with tempfile.NamedTemporaryFile(mode='w+', delete=True,
                                     suffix='.tmp') as temp_file:
        temp_file.writelines(buffer)
        temp_file.seek(0)  # Move the file pointer to the beginning to read
        replace = temp_file.read()

    # dynamically write to a python file and compiled as a python module
    #
    # XXX
    # the mod is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time?
    #
    # mod = PyCodeCache.load(src_code, extra=str(counter))
    #
    mod = PyCodeCache.load(replace)
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


def preload(op: TritonCompiledKernel, args):
    # XXX there may be better way to do it
    assert isinstance(op, TritonCompiledKernel)

    # jit and fill the cache
    op(*args)

    #
    if isinstance(op.kernel, Autotuner):
        cache = op.kernel.fn.cache
    else:
        cache = op.kernel.cache

    # FIXME assume only one cache for now
    for dev_id, vals in cache.items():
        for workload_keys, compiled_kernel in vals.items():

            # write the kernel data
            compiled_kernel.kernel = op.binary
            compiled_kernel.module = None  # this force it to reload
