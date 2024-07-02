import os
import re
import tempfile
from enum import Enum, auto
from typing import Union
from dataclasses import dataclass, field, astuple

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import Autotuner, Config

from dlblas.op_struct import OpImpl
from dlblas.cache import TritonCompiledKernel
from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.autotune.policy import RandomPolicy
'''
The compiler dynamically parse and execute the kernel file as string,
    this is because kernels defined under the kerenl/ folder have been `executed` 
    it would have been easier to define kerenl as templates,
    dynamic parsing and execution `templatfy` the kernels
'''


class State(Enum):
    DEFAULT = auto()
    CALL = auto()
    CALL_KERNEL = auto()


@dataclass
class Parser:
    _buffer: list = field(default_factory=list)
    kernel_name: str = None
    call_name: str = None
    tunable_params: set = field(default_factory=set)

    def get_tunable_params(self, op: OpImpl):
        space = op.spaces
        if isinstance(space, ChoiceSpace):
            first = space[0]
            assert isinstance(first, Config)
            tunable_params = list(
                first.kwargs.keys()) + ['num_warps', 'num_stages', 'num_ctas']
        elif isinstance(space, DictSpace):
            tunable_params = list(space.params.keys())
        else:
            raise TypeError(
                f"space must be ChoiceSpace or DictSpace, but got {type(space)}"
            )

        self.tunable_params = set(tunable_params)

    def process(self, src_code: str, op: OpImpl):
        self.get_tunable_params(op)
        self.kernel_name = op.kernel.__name__
        self.call_name = op.call.__name__

        register_pat = r'register_dlblas_op\(.*?(?=\n|$)'

        def rewrite_register(match: re.Match):
            return 'pass\n'

        # replace all `register_dlblas_op` call
        text = re.sub(
            register_pat,
            rewrite_register,
            src_code,
            flags=re.MULTILINE,
        )

        # find invoke kernel idx; {kernel_name}[{grid_name}]
        start_idx = []
        invoke_kernel_pattern = fr'{self.kernel_name}\[[a-zA-Z0-9_]+\]'
        matches: list[re.Match] = re.finditer(
            invoke_kernel_pattern,
            text,
            flags=re.DOTALL,
        )
        for match in matches:
            start_idx.append(match.start())

        # find (start, end) pair
        start_end_idx = []
        for start in start_idx:
            end = start
            while True:
                if text[end] == '(':
                    break
                end += 1

            # find the last closing )
            open_count = 1
            end += 1
            while True:
                if text[end] == '(':
                    open_count += 1
                elif text[end] == ')':
                    open_count -= 1
                    if open_count == 0:
                        break
                end += 1
            end += 1
            start_end_idx.append((start, end))

        for (start, end) in start_end_idx:
            line = text[start:end]
            line = line.replace('\n', '').replace('#', '').replace(' ', '')
            arg_start_index = line.find('(')
            args_line = line[arg_start_index + 1:-1]
            for arg in args_line.split(','):
                # TODO
                # print(arg)
                pass

        return self

    @property
    def buffer(self):
        return self._buffer

    def build(self) -> str:
        src = ''.join(self.buffer)
        return src

    def replace_tunable_args(self, replacement):
        # TODO
        # assert self.kernel_name is not None, f'parser has no kernel name'

        # for i in range(self.call_def_idx + 1, self.call_end_idx):
        #     line = self.buffer[i]
        #     if self.kernel_name in line:
        #         self.mode = State.KERNEL
        pass


def tunning(op: OpImpl, args):
    parser = parse_op(op)
    policy = RandomPolicy('random')
    decision = policy.generate(op.spaces)
    parser.replace_tunable_args(decision)
    src = parser.build()
    compile_op(op, src)
    perf = op.bench(*args)
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


def compile_op(op: Union[OpImpl, TritonCompiledKernel], src):
    #
    # dynamically write to a python file and compiled as a python module
    # the mod is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time
    #
    # mod = PyCodeCache.load(src_code, extra=str(counter))
    #
    mod = PyCodeCache.load(src)
    PyCodeCache.clear()  # we want a fresh copy every time

    call_name = op.call.__name__
    bench_fn_name = op.bench_fn.__name__
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
