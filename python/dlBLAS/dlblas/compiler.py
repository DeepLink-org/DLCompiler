import os
import re
import tempfile

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import Autotuner

from dlblas.op_struct import OpImpl


def compile_and_bench(op: OpImpl, args):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
    '''
    compile_op(op)
    perf = op.bench(*args)
    return perf


def compile_op(op: OpImpl):
    kernel_file = op.file_path
    with open(kernel_file, 'r') as file:
        src_code = file.read()

    # a hack to avoid re-register the func
    ## replace = re.sub(r'register_dlblas_op.*?(?=\n|$)', 'pass', src_code, flags=re.MULTILINE)

    ## regex seems to unable to express conditional replace
    processed_content = []
    for line in src_code:
        # Check if the line contains 'register_dlblas_op' and does not contain 'import'
        if 'register_dlblas_op' in line and 'import' not in line:
            tmp = re.sub(r'register_dlblas_op.*?(?=\n|$)',
                         'pass\n',
                         line,
                         flags=re.MULTILINE)
            processed_content.append(
                tmp)  # Replace the line with 'pass', keeping the indentation
        else:
            processed_content.append(line)  # Keep the line as is

    with tempfile.NamedTemporaryFile(mode='w+', delete=True,
                                     suffix='.tmp') as temp_file:
        temp_file.writelines(processed_content)
        temp_file.seek(0)  # Move the file pointer to the beginning to read
        replace = temp_file.read()

    # dynamically write to a python file and compiled as a python module
    #
    # XXX
    # the mod is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time?
    #
    # mod = PyCodeCache.load(src_code, extra=str(counter))
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
