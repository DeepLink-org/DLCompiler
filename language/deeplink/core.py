import numpy as np
from triton.language import semantic as tl_semantic
from triton.language.core import (
    _tensor_member_fn,
    builtin,
    constexpr,
    tensor,
    range,
)
import builtins
from . import semantic as dl_semantic

def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


def _extract_slice(sl: slice, shape: constexpr):
    def constexpr_or_none_to_value(v, default: int):
        if v is None:
            return default
        assert isinstance(v, (constexpr, int)), f"slice only can be constexpr or int, got: {v}"
        return _constexpr_to_value(v)

    start = constexpr_or_none_to_value(sl.start, 0)
    stop = constexpr_or_none_to_value(
        sl.stop, _constexpr_to_value(shape))
    step = constexpr_or_none_to_value(sl.step, 1)
    size = (stop - start + step - 1) // step
    assert start >= 0 and stop >= 0 and step >= 0 and size >= 0, f"slice should be greater than 0"
    return start, size, step


@_tensor_member_fn
@builtin
def __getitem__(self, slices, _builder=None):
    if isinstance(slices, (slice, constexpr, tensor)) or slices is None:
        slices = [slices]
    ret = self
    offsets = []
    sizes = []
    strides = []
    dst_shape = []
    need_extract_slice = False
    for dim, sl in enumerate(slices):
        if sl is None or isinstance(sl, constexpr) and sl.value is None:
            ret = tl_semantic.expand_dims(ret, dim, _builder)
            offsets.append(_builder.get_int32(0))
            dst_shape.append(constexpr(1))
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
        elif isinstance(sl, slice) and sl.start is None and sl.stop is None and sl.step is None:
            pass
        elif sl is None or isinstance(sl, (constexpr, int)) and sl.value is not None:
            offsets.append(_builder.get_int32(_constexpr_to_value(sl)))
            need_extract_slice = True
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
        elif isinstance(sl, tensor):
            offsets.append(sl.handle)
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
            need_extract_slice = True
        elif isinstance(sl, (slice, builtins.slice)):
            start, size, step = _extract_slice(sl, ret.shape[dim])
            offsets.append(start)
            strides.append(constexpr(step))
            sizes.append(constexpr(size))
            dst_shape.append(constexpr(size))
            need_extract_slice = True
        else:
            raise ValueError(f"unsupported tensor index: {sl}")

    if need_extract_slice:
        new_offsets = [
            tl_semantic.to_tensor(o, _builder) if not isinstance(o, tensor) else o
            for o in offsets
        ]
        ret = dl_semantic.extract_slice(self, new_offsets, sizes, strides, _builder)
    return ret


@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to receive tensor.
    :type ful: Tensor
    :param sub: The tensor to be inserted.
    :type sub: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    new_offsets = [
        tl_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    out = dl_semantic.insert_slice(ful, sub, new_offsets, sizes, strides, _builder)
    return out


@builtin
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to split.
    :type ful: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert len(ful.shape) > 0
    new_offsets = [
        tl_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    sub = dl_semantic.extract_slice(ful, new_offsets, sizes, strides, _builder)
    return sub


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """
    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, bind_sub_block: bool = False):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block