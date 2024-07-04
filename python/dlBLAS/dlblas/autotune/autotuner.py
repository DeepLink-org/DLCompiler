import os

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import OutOfResources

from dlblas.op_struct import OpImpl
from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.autotune.policy import RandomPolicy
from dlblas.autotune.dynamic_compiler import Parser


def tunning(op: OpImpl, args):
    parser = parse_op(op)
    policy = RandomPolicy('random')

    # TODO a simple loop for now
    for _ in range(10):
        config = policy.generate(op.spaces)
        src = parser.build(config)
        op.src = src
        compile_op(op)
        try:
            perf = op.bench(*args)
            test_ok = True
        except OutOfResources:
            test_ok = False

        if test_ok:
            break

    return perf


def parse_op(op: OpImpl):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
    '''
    kernel_file = op.file_path
    with open(kernel_file, 'r') as file:
        # src_code = file.readlines()  # list[str]
        src_code = file.read()  # str

    parser = Parser().process(src_code, op)
    return parser


def compile_op(op: OpImpl):
    #
    # dynamically write to a python file and compiled as a python module
    # the mod is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time
    #
    # mod = PyCodeCache.load(src_code, extra=str(counter))
    #
    assert (op.src is not None and isinstance(op.src, str))

    # XXX may be try catch
    mod = PyCodeCache.load(op.src)
    PyCodeCache.clear()  # we want a fresh copy every time

    call_name = op.call.__name__
    bench_fn_name = op.bench_fn.__name__
    kernel_name = op.kernel.__name__

    # swap the impl
    op.call = getattr(mod, call_name)
    op.bench_fn = getattr(mod, bench_fn_name)
    op.kernel = getattr(mod, kernel_name)
