from typing import List
from contextlib import contextmanager
from triton.language import core as tl
from triton.language import semantic as tl_semantic
from triton._C.libtriton import ir, dicp_triton

_dry_run = False


@contextmanager
def dry_run_context():
    global _dry_run
    _dry_run = True
    try:
        yield
    finally:
        _dry_run = False


_SENDER_RECEIVER_MAP = {
    "cube": ("vector", dicp_triton.ir.PIPE.PIPE_FIX, dicp_triton.ir.PIPE.PIPE_MTE2),
    "vector": ("cube", dicp_triton.ir.PIPE.PIPE_MTE3, dicp_triton.ir.PIPE.PIPE_MTE2),
}


def create_address_space(
    address_space: dicp_triton.ir.AddressSpace,
    builder,
) -> ir.attribute:
    return builder.get_target_attribute(address_space)


def insert_slice(
    ful: tl.tensor,
    sub: tl.tensor,
    offsets: List[tl.tensor],
    sizes: List[int],
    strides: List[int],
    _semantic: tl_semantic.TritonSemantic,
) -> tl.tensor:
    assert len(ful.shape) == len(offsets)
    assert len(ful.shape) == len(sizes)
    assert len(ful.shape) == len(strides)
    assert all([s >= 1 for s in sizes])
    assert all([s >= 0 for s in strides])
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = _semantic.builder.create_insert_slice(
        ful.handle, sub.handle, new_offsets, sizes, strides
    )
    return tl.tensor(out, ret_type)


def extract_slice(
    ful: tl.tensor,
    offsets: List[tl.tensor],
    sizes: List[int],
    strides: List[int],
    _semantic: tl_semantic.TritonSemantic,
) -> tl.tensor:
    assert len(ful.shape) == len(offsets)
    assert len(ful.shape) == len(sizes)
    assert len(ful.shape) == len(strides)
    assert all([s >= 1 for s in sizes])
    assert all([s >= 0 for s in strides])
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = _semantic.builder.create_extract_slice(
        ful.handle, new_offsets, sizes, strides
    )
    return tl.tensor(out, ret_type)


def compile_hint(ptr: tl.tensor, hint_name: str, hint_val, builder: ir.builder):
    # TODO: simt mode does not support hint annotations
    # if builder.is_simt_mode():
    #     return
    # Check isinstance(hint_val, bool) first to handle False explicitly
    if isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, tl.constexpr):
        hint_val = builder.get_string_attr(hint_val.value)
    elif isinstance(hint_val, (list, tl.tuple)):
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation_mark(ptr.handle, hint_name, hint_val)


def alloc(
    shape: List[int], value, dtype: tl.dtype, layout, scope, builder: ir.builder
) -> tl.tensor:
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
        builder.create_annotation_mark(
            x.handle, "layout", builder.get_string_attr(str(layout))
        )
    if scope is not None:
        builder.create_annotation_mark(
            x.handle, "scope", builder.get_string_attr(str(scope))
        )
    return x


def custom_sync_op(builder: ir.builder, op_name: str, **kwargs):
    if _dry_run:
        return None
    if op_name == "sync_block_all":
        return builder.sync_block_all(kwargs["mode"], kwargs["event_id"])

    elif op_name == "sync_block_set":
        sender = kwargs["sender"]
        receiver, sender_pipe, receiver_pipe = _SENDER_RECEIVER_MAP[sender]
        event_id = kwargs["event_id"]
        id_value = builder.get_int64(event_id)
        return builder.sync_block_set(
            sender, receiver, id_value, sender_pipe, receiver_pipe
        )

    elif op_name == "sync_block_wait":
        sender = kwargs["sender"]
        receiver, sender_pipe, receiver_pipe = _SENDER_RECEIVER_MAP[sender]
        event_id = kwargs["event_id"]
        id_value = builder.get_int64(event_id)
        return builder.sync_block_wait(
            sender, receiver, id_value, sender_pipe, receiver_pipe
        )

    raise ValueError(f"Unsupported custom op: {op_name}")
