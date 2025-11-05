from typing import List
from triton.language import core as tl
from triton.language import semantic as tl_semantic
from triton._C.libtriton import ir

def insert_slice(ful: tl.tensor, sub: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = builder.create_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def extract_slice(ful: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = builder.create_extract_slice(ful.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def compile_hint(ptr: tl.tensor, hint_name: str, hint_val, builder: ir.builder):
    if not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation(ptr.handle, hint_name, hint_val)

def alloc(shape: List[int], value, dtype: tl.dtype, layout, scope, builder: ir.builder) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        value = tl_semantic.cast(value, dtype, builder)
    else:
        # scalar
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = builder.get_null_value(dtype.to_ir(builder))
        else:
            get_value_fn = getattr(builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        value = tl.tensor(value, dtype)
    if len(shape) == 0:
        return value
    ret_ty = tl.block_type(value.dtype, shape)
    x = tl.tensor(builder.create_splat(value.handle, shape), ret_ty)
    if layout is not None:
        builder.create_annotation(x.handle, "layout", builder.get_string_attr(str(layout)))
    if scope is not None:
        builder.create_annotation(x.handle, "scope", builder.get_string_attr(str(scope)))
    return x


def custom_sync_op(builder: ir.builder, op_name: str, **kwargs):
    if op_name == "sync_block_all":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["mode"], kwargs["event_id"])

    elif op_name == "sync_block_set":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])
    
    elif op_name == "sync_block_wait":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])
    
    raise ValueError(f"Unsupported custom op: {op_name}")