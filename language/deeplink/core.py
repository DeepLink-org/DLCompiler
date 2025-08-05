import numpy as np
from triton.language import semantic as tl_semantic
from triton.language.core import (
    _tensor_member_fn,
    _shape_check_impl,
    _constexpr_to_value,
    _unwrap_if_constexpr,
    builtin,
    constexpr,
    tensor,
    range,
)
import builtins
from . import semantic as dl_semantic


class layout:
    ASCEND = ['ND', 'NZ']

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in layout.ASCEND, name
    
    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'


ND = layout('ND')
NZ = layout('NZ')

class scope:
    GPU = ['fragment']
    ASCEND = ['UB', 'L1', 'L0A', 'L0B', 'L0C']

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in scope.ASCEND + scope.GPU, name

    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'


fragment = scope('fragment')
UB = scope('UB')
L1 = scope('L1')
L0A = scope('L0A')
L0B = scope('L0B')
L0C = scope('L0C')


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


@builtin
def compile_hint(ptr, hint_name, hint_val=None, _builder=None):
    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    dl_semantic.compile_hint(ptr, hint_name, hint_val, _builder)


@builtin
def alloc(shape, value, dtype, layout=None, scope=None, _builder=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param value: A scalar value to fill the array with
    :type value: scalar
    :param dtype: Data type of the new array, e.g., :code:`tl.float16`
    :type dtype: tl.dtype
    """
    shape = _shape_check_impl(shape)
    value = _constexpr_to_value(value)
    dtype = _constexpr_to_value(dtype)
    layout = _constexpr_to_value(layout)
    scope = _constexpr_to_value(scope)
    return dl_semantic.alloc(shape, value, dtype, layout, scope, _builder)


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

class inline_lambda:
    """
    Inline a lambda function into the current block.
    This class is used to inline a lambda function into the current block.
    """
    def __init__(self, node, closure_values, closure_names, arg_names):
        self.node = node
        self.closure_values = closure_values
        self.closure_names = closure_names
        self.arg_names = arg_names

    def __call__(self, *args, generator=None):
        if generator is None:
            raise RuntimeError("Generator must be provided for Lambda inlining")

        # save old state
        old_lscope = generator.lscope.copy()
        old_local_defs = generator.local_defs.copy()
        old_insert_block = generator.builder.get_insertion_block()
        try:
            # create closure parameters map
            closure_map = {}
            for name, value in zip(self.closure_names, self.closure_values):
                closure_map[name] = value

            # create parameter map
            param_map = {}
            for name, value in zip(self.arg_names, args):
                param_map[name] = value

            # merge closure and parameter maps
            generator.lscope = {**old_lscope, **closure_map, **param_map}
            generator.local_defs = {**old_local_defs, **closure_map, **param_map}

            # visit the lambda body
            return generator.visit(self.node.body)
        finally:
            # restore old state
            generator.lscope = old_lscope
            generator.local_defs = old_local_defs
            if old_insert_block:
                generator.builder.set_insertion_point_to_end(old_insert_block)
