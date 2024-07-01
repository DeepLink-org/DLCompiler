import inspect
import pickle

import torch

from dlblas.op_struct import OpImpl


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
        #
        # for now just key -> OpImpl
        #
        # if key not in self._cache:
        #     self._cache[key] = op

    def to_file(self, fname):
        with open(f'{fname}.pkl', 'wb') as handle:
            # pickle.dump(self._cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._cache, handle)

    def from_file(self, fname):
        with open(f'{fname}.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self._cache = data
