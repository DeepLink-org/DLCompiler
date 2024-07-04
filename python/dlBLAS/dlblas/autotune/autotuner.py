import os

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import OutOfResources

from dlblas.op_struct import OpImpl
from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.autotune.policy import get_policy, Policy
from dlblas.autotune.dynamic_compiler import Parser
from dlblas.autotune.configs import AutotuneConfig


def tunning(op: OpImpl, args: tuple, configs: AutotuneConfig):
    parser: Parser = parse_op(op)
    policy: Policy = get_policy(configs)

    # tunning loop
    for iteration in range(configs.iteration):

        # policy generate suggestions
        kernel_configs = policy.generate(op.spaces)

        # compile
        src = parser.build(kernel_configs)
        op.src = src
        compile_op(op)

        # feedback signal
        bench_ok = True
        try:
            perf = op.bench(*args)
        except OutOfResources:
            bench_ok = False

        # early stop creiteria
        if bench_ok:
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
