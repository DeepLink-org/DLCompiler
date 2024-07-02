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


class State(Enum):
    DEFAULT = auto()
    KERNEL = auto()
    CALL = auto()


@dataclass
class Parser:
    _buffer: list = field(default_factory=list)
    kernel_def_idx: int = -1
    kernel_end_idx: int = -1
    call_def_idx: int = -1
    call_end_idx: int = -1
    mode: State = State.DEFAULT
    tunable_params: field = field(default_factory=dict)

    def pre_process(self, op: OpImpl):
        space = op.spaces
        if isinstance(space, ChoiceSpace):
            first = space[0]
            assert isinstance(first, Config)
            self.tunable_params = list(
                first.kwargs.keys()) + ['num_warps', 'num_stages', 'num_ctas']
        elif isinstance(space, DictSpace):
            self.tunable_params = list(space.params.keys())
        else:
            raise TypeError(
                f"space must be ChoiceSpace or DictSpace, but got {type(space)}"
            )

        self.tunable_params = set(self.tunable_params)

    def parse(self, src_code: str, op: OpImpl):
        kernel_name = op.kernel.__name__
        call_name = op.call.__name__
        self.pre_process(op)

        # XXX significant overhead?
        for i, line in enumerate(src_code):
            if 'register_dlblas_op' in line and 'import' not in line:
                # avoid re-register the func;
                replace = re.sub(
                    r'register_dlblas_op.*?(?=\n|$)',
                    'pass\n',
                    line,
                    flags=re.MULTILINE,
                )
                self._buffer.append(replace)
                continue

            if "def" in line and kernel_name in line:
                self.kernel_def_idx = i
                self.mode = State.KERNEL

            if "def" in line and call_name in line:
                self.call_def_idx = i
                self.mode = State.CALL

            # keep line
            self._buffer.append(line)

        return self

    @property
    def buffer(self):
        return self._buffer

    def get_tunable_args(self):
        for line in self.buffer[self.kernel_def_idx + 1:self.kernel_end_idx]:
            pass

    def replace_tunable_args(self):
        return


def compile_and_bench(op: OpImpl, args):
    compile_op(op)
    perf = op.bench(*args)
    return perf


def compile_op(op: Union[OpImpl, TritonCompiledKernel]):
    '''at kernel-register time, the same triton kernel is used
        here we want to intercept the source code and compile it
        so that each args can correspond to only one instance of triton kernel

        we use inductor's PyCodeCache for now
    '''
    kernel_file = op.file_path
    with open(kernel_file, 'r') as file:
        src_code = file.readlines()

    parser = Parser().parse(src_code, op)
    with tempfile.NamedTemporaryFile(mode='w+', delete=True,
                                     suffix='.tmp') as temp_file:
        temp_file.writelines(parser.buffer)
        temp_file.seek(0)  # Move the file pointer to the beginning to read
        replace = temp_file.read()

    # dynamically write to a python file and compiled as a python module
    # the mod is cached in PyCodeCache, but we want a fresh copy each time, so we clear each time
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
