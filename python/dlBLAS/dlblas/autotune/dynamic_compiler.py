import os
import re
import tempfile
from enum import Enum, auto
from typing import Union
from dataclasses import dataclass, field, astuple
from copy import deepcopy

from triton.runtime.autotuner import Config

from dlblas.op_struct import OpImpl
from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.autotune.passes import rewrite_dlblas_registration_pass, analyse_kernel_call_pass
'''
The compiler dynamically parse and execute the kernel file as string,
    this is because kernels defined under the kerenl/ folder have been `executed` 
    it would have been easier to define kerenl as templates,
    dynamic parsing and execution `templatfy` the kernels
'''


@dataclass
class Parser:
    kernel_name: str = None
    call_name: str = None
    tunable_params: set = field(default_factory=set)
    src_code: str = None
    kernel_call_start_end_idx: [[int]] = field(default_factory=list)
    kernel_args_names: [str] = field(default_factory=list)
    kernel_constexprs_idx: [int] = field(default_factory=list)

    def get_tunable_params(self, op: OpImpl):
        space = op.spaces
        if space is None:
            tunable_params = []
        elif isinstance(space, ChoiceSpace):
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

        # run passes
        text = rewrite_dlblas_registration_pass(src_code)
        kernel_call_start_end_idx = analyse_kernel_call_pass(
            text,
            self.kernel_name,
        )

        # populate self
        self.src_code = text
        self.kernel_call_start_end_idx = kernel_call_start_end_idx
        return self

    def build(self, replacement: dict) -> str:
        ''' build src code text with replacement value for tunable params
        '''
        new_src = ''
        last_end = 0
        for i, (start, end) in enumerate(self.kernel_call_start_end_idx):
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
