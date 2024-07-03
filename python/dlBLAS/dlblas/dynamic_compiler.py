import os
import re
import tempfile
from enum import Enum, auto
from typing import Union
from dataclasses import dataclass, field, astuple
from copy import deepcopy

from torch._inductor.codecache import PyCodeCache

from triton.runtime.autotuner import Autotuner, Config, OutOfResources

from dlblas.op_struct import OpImpl
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
    kernel_name: str = None
    call_name: str = None
    tunable_params: set = field(default_factory=set)
    src_code: str = None
    start_end_idx: list[tuple[int]] = field(default_factory=list)
    kernel_args_names: list[str] = field(default_factory=list)
    kernel_constexprs_idx: list[int] = field(default_factory=list)

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
        self.kernel_args_names = deepcopy(op.kernel.arg_names)
        self.kernel_constexprs_idx = deepcopy(op.kernel.constexprs)

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

        # find (start, end) pair in the kernel text
        start_end_idx = []
        for start in start_idx:
            # goes to the first '('
            end = start
            while True:
                if text[end] == '(':
                    break
                end += 1

            # find the last closing ')'
            # there must be one, otherwise the file will report error at import time
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

        self.src_code = text
        self.start_end_idx = start_end_idx
        return self

    def build(self, replacement: dict) -> str:
        ''' build src code text with replacement value for tunable params
        '''
        new_src = ''
        last_end = 0
        for i, (start, end) in enumerate(self.start_end_idx):
            new_args = []
            #
            # for each kernel invocation
            # dynamically fill in the tunable args
            #
            line = self.src_code[start:end]
            line = line.replace('\n', '').replace('#', '').replace(' ', '')
            # find the first '('
            arg_start_index = line.find('(')
            # strip '(' and ')'
            args_line = line[arg_start_index + 1:-1]
            # the last arg could have a trailing comma...
            args_line = args_line.rstrip(',')
            # convert to list
            args_line_list = args_line.split(',')
            for arg_idx, arg in enumerate(args_line_list):
                if '=' not in arg:
                    # positional
                    if arg_idx in self.kernel_constexprs_idx:
                        # tl.constexpr pass as positional args
                        pass
                    else:
                        new_args.append(arg)

                else:
                    # kwawgs
                    kw = arg.split('=')[0]
                    if kw not in self.tunable_params:
                        # not tunable
                        new_args.append(arg)

            # now fill in the replacement
            for k, v in replacement.items():
                new_args.append(f'{k}={v}')
            new_line = line[:arg_start_index] + '(' + ','.join(new_args) + ')'

            # build new src code
            new_src += self.src_code[last_end:start]
            new_src += new_line
            last_end = end

        # the remaining part
        new_src += self.src_code[last_end:]
        return new_src


def tunning(op: OpImpl, args):
    parser = parse_op(op)
    policy = RandomPolicy('random')

    # TODO a simple loop for now
    for _ in range(10):
        decision = policy.generate(op.spaces)
        src = parser.build(decision)
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
    mod = PyCodeCache.load(op.src)
    PyCodeCache.clear()  # we want a fresh copy every time

    call_name = op.call.__name__
    bench_fn_name = op.bench_fn.__name__
    kernel_name = op.kernel.__name__

    # swap the impl
    op.call = getattr(mod, call_name)
    op.bench_fn = getattr(mod, bench_fn_name)
    op.kernel = getattr(mod, kernel_name)
