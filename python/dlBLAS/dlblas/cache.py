from dataclasses import dataclass
from typing import Optional
import pickle

import torch

from triton.runtime.autotuner import Autotuner
from triton.runtime.driver import driver

from dlblas.op_struct import OpImpl, OpParams

# XXX not sure why dataclass is not working
# @dataclass(frozen=True)
# class TritonCompiledKernel:
#     file_path: str
#     call: callable
#     bench_fn: callable
#     kernel: callable
#
#     # triton-related
#     # may add more in the future, but essentially a kernel is just bytes
#     kernel: bytes


class TritonCompiledKernel:

    def __init__(self, file_path, call, bench_fn, kernel, binary):
        self.file_path = file_path
        self.call = call
        self.bench_fn = bench_fn
        self.kernel = kernel
        self.binary = binary


def convert_dtype(t: torch.Tensor):
    if t.dtype == torch.float32:
        return 'f32'
    elif t.dtype == torch.float16:
        return 'f16'
    elif t.dtype == torch.bfloat16:
        return 'bf16'
    elif t.dtype == torch.int32:
        return 'i32'
    elif t.dtype == torch.int8:
        return 'i8'
    else:
        raise LookupError(f"unsupported dtype {t.dtype}")


def convert_shapes(t: torch.Tensor):
    ans = ''
    for s in t.shape:
        ans += str(s) + 'x'
    return ans[:-1]


def convert_device(t: torch.Tensor):
    if t.device.type == 'cuda':
        return 'cuda'
    elif t.device.type == 'cpu':
        return 'cpu'
    else:
        raise LookupError(f"unsupported device {t.device}")


class Cache:

    def __init__(self):
        self._cache = {}

    def gen_key(self, op_name, args):
        key = op_name
        for i, arg in enumerate(args):
            key += '-' + str(i) + ':'
            if isinstance(arg, torch.Tensor):
                key += convert_dtype(arg) + '_' + convert_shapes(arg)
                device = convert_device(arg)
            else:
                key += str(arg)  # let it fail if not implemented

        # XXX assume all tensor in the same device
        key += '-' + device
        return key

    def put(self, op: OpImpl, op_name, args):
        key = self.gen_key(op_name, args)
        #
        # so Triton has already a cache mechanism during compilation
        # https://github.com/triton-lang/triton/blob/8e96b71b1b47a5d09f1cfb1826a16178f58dbef0/python/triton/compiler/compiler.py#L258
        #
        # XXX
        # do we maintain our own cache system? or we just trigger triton, which perform cache look-up for us?
        # we maintain our own cache
        #

        # this is triton's driver interface, and may subject to change
        binary_ext = driver.binary_ext
        if isinstance(op.kernel, Autotuner):
            jit_fn = op.kernel.fn
        else:
            jit_fn = op.kernel

        # FIXME assume only one cache for now
        for dev_id, vals in jit_fn.cache.items():
            for workload_keys, compiled_kernel in vals.items():
                binary: bytes = compiled_kernel.kernel

        local_scope = {}

        # we just need the function name to compile dynamically
        call = f"""
def {op.call.__name__}():
    pass
"""
        bench_fn = f"""
def {op.bench_fn.__name__}():
    pass
"""
        kernel = f"""
def {jit_fn.__name__}():
    pass
"""
        exec(call, globals(), local_scope)
        exec(bench_fn, globals(), local_scope)
        exec(kernel, globals(), local_scope)
        call = local_scope[op.call.__name__]
        bench_fn = local_scope[op.bench_fn.__name__]
        kernel = local_scope[jit_fn.__name__]

        self._cache[key] = TritonCompiledKernel(
            op.file_path,
            call,
            bench_fn,
            kernel,
            binary,
        )

    def get(self, op_name, args) -> Optional[TritonCompiledKernel]:
        key = self.gen_key(op_name, args)
        if key in self._cache:
            return self._cache[key]

    def to_file(self, fname):
        with open(f'{fname}.pkl', 'wb') as handle:
            # pickle.dump(self._cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._cache, handle)

    def from_file(self, fname):
        with open(f'{fname}.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self._cache = data
