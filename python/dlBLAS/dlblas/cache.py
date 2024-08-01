from dataclasses import dataclass
from typing import Optional, Any, Union
import pickle

import torch

from dlblas.op_struct import OpImpl, OpParams
from dlblas.autotune.space import ChoiceSpace, DictSpace


@dataclass(frozen=True)
class OpImplCache:
    params: OpParams
    file_path: str
    src: str
    spaces: Union[ChoiceSpace, DictSpace]
    call_name: str
    bench_fn_name: str
    kernel_name: str


def convert_dtype(t: torch.Tensor):
    if t.dtype == torch.float32:
        return 'f32'
    elif t.dtype == torch.float16:
        return 'f16'
    elif t.dtype == torch.bfloat16:
        return 'bf16'
    elif t.dtype == torch.int64:
        return 'i64'
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
        self._cache[key] = op
        # new_op = OpImplCache(
        #     op.params,
        #     op.file_path,
        #     op.src,
        #     op.spaces,
        #     op.call.__name__,
        #     op.bench_fn.__name__,
        #     op.kernel.__name__,
        # )
        # self._cache[key] = new_op
       

    def get(self, op_name, args) -> Optional[OpImpl]:
        key = self.gen_key(op_name, args)
        if key in self._cache:
            op_cache = self._cache[key]
            if isinstance(op_cache, OpImpl):
                return op_cache
            assert isinstance(op_cache, OpImplCache)
            local_scope = {}

            # we just need the function name to compile dynamically
            call = f"""
def {op_cache.call_name}():
    pass
"""
            bench_fn = f"""
def {op_cache.bench_fn_name}():
    pass
"""
            kernel = f"""
def {op_cache.kernel_name}():
    pass
"""
            exec(call, globals(), local_scope)
            exec(bench_fn, globals(), local_scope)
            exec(kernel, globals(), local_scope)
            call = local_scope[op_cache.call_name]
            bench_fn = local_scope[op_cache.bench_fn_name]
            kernel = local_scope[op_cache.kernel_name]
            op = OpImpl(
                op_cache.params,
                op_cache.file_path,
                op_cache.src,
                op_cache.spaces,
                call,
                bench_fn,
                kernel,
            )
            return op

    def to_file(self, fname):
        with open(f'{fname}.pkl', 'wb') as handle:
            # pickle.dump(self._cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._cache, handle)

    def from_file(self, fname):
        with open(f'{fname}.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self._cache = data
