import os

import numpy as np

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import OutOfResources

from dlblas.op_struct import OpImpl
from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.autotune.policy import get_policy, Policy
from dlblas.autotune.dynamic_compiler import Parser
from dlblas.autotune.configs import AutotuneConfig


def tunning(op: OpImpl, args: tuple, configs: AutotuneConfig):
    parser: Parser = parse_op(op)
    policy: Policy = get_policy(op.spaces, configs)

    # tunning loop
    perfs = [float('inf') for _ in range(configs.total_iteration)]
    srcs = [None for _ in range(configs.total_iteration)]
    iteration = 0
    while iteration < configs.total_iteration:

        # policy generate suggestions
        kernel_configs = policy.generate()
        if kernel_configs is None:
            # exhausted
            break

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
        except Exception as e:
            # this actually triggers JIT compile,
            # if there's problem with the compilation
            # it will raise an exception
            # TODO not sure if we simply ignore here?
            # if all fails, then just return a inf perf?
            raise e

        # update tunner
        if bench_ok:
            policy.feedback(perf)

            perfs[iteration] = perf
            srcs[iteration] = src
        else:
            policy.feedback(None)

        iteration += 1

    # get best
    best_kernel_configs_idx = int(np.argmin(perfs))
    if srcs[best_kernel_configs_idx] is None:
        raise RuntimeError(f'''
            unable to tune for op {op.name},
            consider to config more tunning iteration,
            or widen the search space
        ''')
    op.src = srcs[best_kernel_configs_idx]
    best_perf = perfs[best_kernel_configs_idx]
    return best_perf


def parse_op(op: OpImpl):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
        also note that registration is ok means that the python file can be parsed successfully
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
