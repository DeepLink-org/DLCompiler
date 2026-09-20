# flagtree tle
import builtins
import triton.language.core as tl
from triton._C.libtriton.wafer import tle as _dsa_ir
from typing import Optional, Sequence
from enum import Enum
from . import types as tle

from triton.language.core import (
    constexpr,
    tensor,
    range,
)

# Address space 3 matches the shared-memory space used in TritonGPU lowering.
SHARED_MEMORY_ADDRESS_SPACE = 3


# Triton 3.5 recognizes loop iterators by class identity, so the otherwise empty
# FlagTree range subclass cannot compile here. The alias preserves num_stages
# and loop_unroll_factor and emits the same tt.num_stages loop annotation.
# Actual software pipelining is selected by the Wafer enable_pipeline option.
pipeline = range


@tl.builtin
def memory_space(input, space, _semantic=None):
    """
    Annotate a tensor with a target memory-space tag.

    The attribute ``tt.memory_space`` is propagated through the IR and can be
    consumed by downstream DSA passes (e.g. ``--dsa-memory-to-core``) to make
    allocation / placement decisions.

    Args:
        input: Tensor to annotate.
        space: Memory-space name string, e.g. ``"spm"`` or ``"shared_memory"``.
    """
    space = tl._unwrap_if_constexpr(space)
    if _semantic is not None and hasattr(input, 'handle') and hasattr(input.handle, 'set_attr'):
        input.handle.set_attr("tt.memory_space", _semantic.builder.get_string_attr(str(space)))
    return input


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout: Optional[object] = None,
    scope: tle.scope = None,
    _semantic=None,
) -> tle.buffered_tensor:
    """
    Allocate local memory buffer

    Args:
        shape: Buffer shape
        dtype: Data type
        layout: Memory layout encoding (optional)
        scope: Storage type (default to shared memory)
        _semantic: Semantic analyzer (internal use)

    Returns:
        Allocated buffer tensor

    Raises:
        ValueError: When parameters are invalid
        RuntimeError: When allocation fails
    """
    from .semantic import DSASemantic

    if _semantic is None:
        raise ValueError("alloc must be used inside @triton.jit")
    if layout is not None:
        raise ValueError("alloc(): layout parameter is not yet support for DSA backend")

    # --- Validate inputs via semantic layer ---
    unwrapped_shape = DSASemantic.validate_alloc_shape(shape)
    elem_dtype = DSASemantic.validate_alloc_dtype(dtype)
    resolved_scope = DSASemantic.validate_alloc_scope(scope)

    elem_ir_ty = elem_dtype.to_ir(_semantic.builder)

    if not hasattr(_dsa_ir, "create_dsa_alloc"):
        raise RuntimeError("builder missing create_dsa_alloc for DSA alloc")

    alloc_value = _dsa_ir.create_dsa_alloc(_semantic.builder, list(unwrapped_shape), elem_ir_ty)
    buf_ty = tle.buffered_tensor_type(unwrapped_shape, elem_dtype, resolved_scope)
    buf_ty._ir_type = alloc_value.get_type()
    return tle.buffered_tensor(alloc_value, buf_ty)


class CopyDirection(Enum):
    """Copy direction enum for data transfer operations"""
    GM_TO_LOCAL = "GMTOLOCAL"  # Global memory to local memory
    LOCAL_TO_GM = "LOCALTOGM"  # Local memory to global memory


@tl.builtin
def copy(src, dst, shape, offsets: Sequence[constexpr | tensor] = None,
         _semantic=None) -> None:
    """Copy an entire contiguous buffer, or an equally shaped tensor of pointers.

    DSA CopyOp only accepts memrefs. GM transfers are expressed as explicit
    loads/stores through a local pointer view so their IR types remain valid.
    """
    from .semantic import DSASemantic

    # Copy the complete local buffer; sub-buffer offsets are not implemented.
    # A caller can offset the GM pointer before passing it to this function.
    if offsets is not None:
        raise NotImplementedError("DSA copy offsets are not supported; pass an adjusted pointer")
    shape = DSASemantic.validate_alloc_shape(tl._unwrap_if_constexpr(shape))
    src_is_buf = isinstance(src, tle.buffered_tensor)
    dst_is_buf = isinstance(dst, tle.buffered_tensor)
    if not (src_is_buf or dst_is_buf):
        raise ValueError("copy requires at least one buffered_tensor")
    for value in (src, dst):
        if isinstance(value, tle.buffered_tensor):
            if value.type.shape != shape:
                raise ValueError("copy shape must match the complete DSA buffer")
            # Remote buffer handles need an explicit remote pointer view.
            if hasattr(value.type, "_tle_remote_shard_id"):
                raise NotImplementedError("Use local_ptr(remote(buffer, tile)) for NoC transfers")

    # Local -> local: both handles are memrefs, as required by dsa.copy.
    # The binding takes the Triton 3.5 semantic object's builder explicitly.
    if src_is_buf and dst_is_buf:
        DSASemantic.validate_copy_dtype_compat(src.dtype, dst.dtype)
        _dsa_ir.create_dsa_copy(_semantic.builder, src.handle, dst.handle)
        return

    # GM <-> local: exactly one operand is a buffer. A Triton pointer cannot
    # be passed to dsa.copy, so use pointer views and emit load/store IR below.
    buffer = src if src_is_buf else dst
    gm = dst if src_is_buf else src
    if not isinstance(gm, tl.tensor) or not gm.dtype.is_ptr():
        raise ValueError("The global operand of DSA copy must be a pointer tensor")
    DSASemantic.validate_copy_dtype_compat(buffer.dtype, gm.dtype.element_ty)

    # Broadcast each coordinate axis to the full buffer shape. local_ptr only
    # builds an address view of the buffer; it does not copy any data itself.
    indices = _make_full_indices(buffer, _semantic)
    ptr = local_ptr(buffer, indices, _semantic=_semantic)
    if not gm.shape:
        # A scalar GM base pointer denotes contiguous row-major storage.
        # Expand it to one pointer per element: shape (M, N) gives i * N + j.
        # Pointer addition uses element offsets, so no byte-size factor is needed.
        linear = tl.full(shape, 0, tl.int32, _semantic=_semantic)
        stride = 1
        for axis in builtins.range(len(shape) - 1, -1, -1):
            offset = indices[axis].__mul__(stride, _semantic=_semantic)
            linear = linear.__add__(offset, _semantic=_semantic)
            stride *= shape[axis]
        gm = gm.__add__(linear, _semantic=_semantic)
    elif tuple(gm.shape) != shape:
        raise ValueError("Global pointer tensor shape must match the DSA buffer")

    # An already-shaped pointer tensor keeps its caller-supplied addressing.
    # These transfers are unmasked: every supplied GM address must be valid.
    # _semantic makes these tl calls generate IR during JIT compilation.
    if src_is_buf:
        # Local -> GM: read the local pointer view and write global pointers.
        tl.store(gm, tl.load(ptr, _semantic=_semantic), _semantic=_semantic)
    else:
        # GM -> local: read global pointers and write the local pointer view.
        tl.store(ptr, tl.load(gm, _semantic=_semantic), _semantic=_semantic)


def _expand_index_to_shape(index: tl.tensor, shape: Sequence[int], axis: int, _semantic) -> tl.tensor:
    idx = index
    for _ in builtins.range(axis):
        idx = tl.expand_dims(idx, 0, _semantic=_semantic)
    for _ in builtins.range(len(shape) - axis - 1):
        idx = tl.expand_dims(idx, len(idx.shape), _semantic=_semantic)
    return tl.broadcast_to(idx, *shape, _semantic=_semantic)


def _make_full_indices(buffer: tle.buffered_tensor, _semantic) -> tuple[tl.tensor, ...]:
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    indices = []
    for axis, dim in enumerate(shape):
        idx = tl.arange(0, dim, _semantic=_semantic)
        idx = _expand_index_to_shape(idx, shape, axis, _semantic)
        indices.append(idx)
    return tuple(indices)


@tl.builtin
def local_ptr(
    buffer: tle.buffered_tensor,
    indices: Optional[Sequence] = None,
    _semantic=None,
    _generator=None,
) -> tl.tensor:
    """
    Materialize shared-memory pointers that cover the given buffered tensor.

    Args:
        buffer: Local memory buffer tensor returned by ``tle.alloc``.
        indices: Tuple of integer index tensors. The tuple length must equal
            the rank of ``buffer`` and every tensor must have the same shape.
            The output pointer tensor will have that same shape.

    Returns:
        Tensor of pointers suitable for ``tl.load``/``tl.store``.
    """
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"Buffer parameter must be tle.buffered_tensor, but got {type(buffer)}")

    if _semantic is None:
        raise ValueError("local_ptr must be used inside @triton.jit")

    # Preferred metadata source: buffered_tensor.type (survives JIT value
    # reconstruction). Keep value attrs as backward-compatibility fallback.
    remote_shard_id = getattr(buffer.type, "_tle_remote_shard_id", None)
    remote_scope = getattr(buffer.type, "_tle_remote_scope", None)
    if remote_shard_id is None:
        remote_shard_id = getattr(buffer, "_tle_remote_shard_id", None)
        remote_scope = getattr(buffer, "_tle_remote_scope", None)
    remote_buffer_marker = remote_shard_id is not None

    indices = tl._unwrap_if_constexpr(indices)
    if indices is None:
        raise ValueError("local_ptr indices must be provided as a tuple of tensors")
    if isinstance(indices, tl.tuple):
        indices_tuple = tuple(indices.values)
    elif isinstance(indices, (tuple, list)):
        indices_tuple = tuple(indices)
    else:
        raise ValueError("local_ptr indices must be a tuple or list of tensors")

    buffer_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    if len(indices_tuple) != len(buffer_shape):
        raise ValueError(f"local_ptr indices must provide {len(buffer_shape)} tensors, got {len(indices_tuple)}")

    idx_tensors: list[tensor] = []
    view_shape: Optional[tuple[int, ...]] = None
    scalar_index_flags: list[bool] = []
    for idx in indices_tuple:
        idx_tensor = idx if isinstance(idx, tensor) else _semantic.to_tensor(idx)
        if not idx_tensor.dtype.is_int():
            raise ValueError("local_ptr indices must use integer dtypes")
        is_scalar_index = not idx_tensor.type.is_block()
        scalar_index_flags.append(is_scalar_index)
        if is_scalar_index:
            idx_tensors.append(idx_tensor)
            continue
        if view_shape is None:
            view_shape = tuple(idx_tensor.shape)
        elif tuple(idx_tensor.shape) != view_shape:
            raise ValueError("local_ptr indices must have identical shapes")
        idx_tensors.append(idx_tensor)

    if not idx_tensors:
        raise ValueError("local_ptr indices cannot be empty")
    all_scalar_indices = all(scalar_index_flags)
    any_scalar_indices = any(scalar_index_flags)
    if any_scalar_indices and not all_scalar_indices:
        raise ValueError("local_ptr indices must be either all scalar or all tensors with identical shapes")
    if not all_scalar_indices and view_shape is None:
        view_shape = tuple()

    ptr_dtype = tl.pointer_type(buffer.type.element_ty)
    insert_block = _semantic.builder.get_insertion_block()
    if insert_block is None:
        raise RuntimeError("TLE local_ptr called without an insertion block")
    if all_scalar_indices:
        result_ty = ptr_dtype
        result_ir = ptr_dtype.to_ir(_semantic.builder)
    else:
        result_ty = tl.block_type(ptr_dtype, list(view_shape))
        result_ir = result_ty.to_ir(_semantic.builder)
    handles = [idx.handle for idx in idx_tensors]
    if not hasattr(_dsa_ir, "create_dsa_local_pointers"):
        raise RuntimeError("builder missing create_dsa_local_pointers for DSA local_ptr")
    local_ptr_op = _dsa_ir.create_dsa_local_pointers(_semantic.builder, result_ir, buffer.handle, *handles)

    result_tensor = tl.tensor(local_ptr_op.get_result(0), result_ty)

    if remote_buffer_marker:
        if remote_scope is not None:
            raise NotImplementedError("Wafer DSA remote pointers currently require physical tile IDs without a scope")
        if all_scalar_indices:
            raise ValueError("local_ptr does not yet support scalar indices on remote buffers")
        if not hasattr(_dsa_ir, "create_dsa_remote_pointers"):
            raise RuntimeError("builder missing create_dsa_remote_pointers for remote buffers")
        shard_val = (remote_shard_id.handle if isinstance(remote_shard_id, tl.tensor) else _semantic.to_tensor(remote_shard_id).handle)
        remote_op = _dsa_ir.create_dsa_remote_pointers(_semantic.builder, result_ir, result_tensor.handle, shard_val)
        result_tensor = tl.tensor(remote_op.get_result(0), result_ty)

    return result_tensor


@tl.builtin
def to_tensor(buffer: tle.buffered_tensor, writable: bool = True, _semantic=None) -> tl.tensor:
    """
    Convert a DSA ``buffered_tensor`` (on-chip buffer) into a ``tl.tensor`` view.

    This is a zero-copy view: the returned tensor aliases the on-chip buffer, so it
    can participate in standard Triton tensor expressions without any data
    movement. Lowered by ``--tle-to-mk`` into ``bufferization.to_tensor``.

    Args:
        buffer: A ``buffered_tensor`` previously allocated with ``tle.language.dsa.alloc``.
        writable: Mark the resulting tensor as writable (default ``True``).

    Returns:
        ``tl.tensor`` aliasing the on-chip buffer contents.
    """
    builder = _semantic.builder
    if builder is None:
        raise ValueError("to_tensor must be used inside @triton.jit")
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"to_tensor requires a buffered_tensor, got {type(buffer).__name__}")
    if hasattr(buffer.type, "_tle_remote_shard_id"):
        raise NotImplementedError("to_tensor requires a local buffer; use local_ptr for NoC")
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    result_ty = tl.block_type(buffer.type.element_ty, list(shape))
    result_ir = result_ty.to_ir(builder)
    handle = _dsa_ir.create_dsa_to_tensor(builder, result_ir, buffer.handle, bool(writable))
    return tl.tensor(handle, result_ty)


@tl.builtin
def to_buffer(src: tl.tensor, space=None, _semantic=None) -> tle.buffered_tensor:
    """
    Copy a ``tl.tensor`` into a newly-allocated DSA buffer and return it.

    This is the reverse bridge of :func:`to_tensor`: it materialises the tensor
    value into a fresh on-chip buffer. Lowered by ``--tle-to-mk`` into
    ``bufferization.to_buffer`` + ``memref.copy``.

    Args:
        src: A ``tl.tensor`` value to store.
        space: Storage scope for the new buffer. Defaults to the internal
            scratchpad scope; may be a per-backend selector such as
            ``tle.dsa.wafer.SPM``.
            may be an address-space selector such as
            ``tle.language.dsa.wafer.SPM``.

    Returns:
        A new ``buffered_tensor`` containing a copy of ``src``.
    """
    builder = _semantic.builder
    if builder is None:
        raise ValueError("to_buffer must be used inside @triton.jit")
    if not isinstance(src, tl.tensor):
        raise ValueError(f"to_buffer src must be a tl.tensor, got {type(src).__name__}")
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in src.shape)
    if not shape:
        raise ValueError("to_buffer src must be a non-scalar tensor")
    buf = alloc(shape, src.dtype, scope=space, _semantic=_semantic)
    _dsa_ir.create_dsa_to_buffer(builder, src.handle, buf.handle)
    return buf


def _check_binary_operands(opname, input, other, result):
    """Validate three-operand elementwise arithmetic operands.

    All three must be ``buffered_tensor`` with identical shape and element
    dtype (tle.md: no implicit broadcast in this API layer).
    """
    for name, val in (("input", input), ("other", other), ("result", result)):
        if not isinstance(val, tle.buffered_tensor):
            raise ValueError(f"{opname} {name} must be a buffered_tensor, "
                             f"got {type(val).__name__}")
    input_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in input.type.shape)
    other_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in other.type.shape)
    result_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in result.type.shape)
    if input_shape != other_shape or input_shape != result_shape:
        raise ValueError(f"{opname} shape mismatch: input={input_shape}, "
                         f"other={other_shape}, result={result_shape}")
    input_dtype = input.type.element_ty
    other_dtype = other.type.element_ty
    result_dtype = result.type.element_ty
    if input_dtype != other_dtype or input_dtype != result_dtype:
        raise ValueError(f"{opname} dtype mismatch: input={input_dtype}, "
                         f"other={other_dtype}, result={result_dtype}")
    if not input_dtype.is_floating():
        raise NotImplementedError(f"{opname}: DSA buffer arithmetic currently requires floating point")
    if any(hasattr(value.type, "_tle_remote_shard_id") for value in (input, other, result)):
        raise NotImplementedError(f"{opname}: use local_ptr for NoC transfers before local arithmetic")


@tl.builtin
def add(input, other, result, _semantic=None):
    """``result = input + other`` elementwise on on-chip buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("add", input, other, result)
    _dsa_ir.create_dsa_add(builder, input.handle, other.handle, result.handle)


@tl.builtin
def sub(input, other, result, _semantic=None):
    """``result = input - other`` elementwise on on-chip buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("sub", input, other, result)
    _dsa_ir.create_dsa_sub(builder, input.handle, other.handle, result.handle)


@tl.builtin
def mul(input, other, result, _semantic=None):
    """``result = input * other`` elementwise on on-chip buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("mul", input, other, result)
    _dsa_ir.create_dsa_mul(builder, input.handle, other.handle, result.handle)


@tl.builtin
def max(input, other, result, _semantic=None):
    """``result = max(input, other)`` elementwise on on-chip buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("max", input, other, result)
    _dsa_ir.create_dsa_maximum(builder, input.handle, other.handle, result.handle)


@tl.builtin
def min(input, other, result, _semantic=None):
    """``result = min(input, other)`` elementwise on on-chip buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("min", input, other, result)
    _dsa_ir.create_dsa_minimum(builder, input.handle, other.handle, result.handle)


@tl.builtin
def div(input, other, result, _semantic=None):
    """``result = input / other`` elementwise on on-chip buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("div", input, other, result)
    _dsa_ir.create_dsa_div(builder, input.handle, other.handle, result.handle)


# ---------------------------------------------------------------------------
# dsa.extract_slice / dsa.insert_slice
#
# Element-level strided slicing (spec 3.3.2.4). The tile grid-coordinate
# convenience forms are provided by the generic tle-lite tier
# (``tle.extract_tile`` / ``tle.insert_tile``) and lower through the shared
# tle dialect with backend-specific conversion patterns.
# Offsets may mix static (int/constexpr) and dynamic (scalar tl.tensor) dims,
# encoded with ShapedType::kDynamic as the sentinel in `static_offsets`.
# ---------------------------------------------------------------------------

_DYNAMIC = -(1 << 63)  # int64 min == ShapedType::kDynamic sentinel


def _try_unwrap_int(val):
    """Return ``val`` as a Python int if int/constexpr-like, else ``None``."""
    if isinstance(val, int):
        return val
    v = tl._unwrap_if_constexpr(val)
    return v if isinstance(v, int) else None


def _static_dims(shape, fn):
    """Unwrap a shape into compile-time ints, erroring on dynamic dims."""
    shape = tl._unwrap_if_constexpr(shape)
    dims = [tl._unwrap_if_constexpr(d) for d in shape]
    if any(not isinstance(d, int) for d in dims):
        raise ValueError(f"{fn}: shape must be compile-time constants, got {shape}")
    return dims


def _split_offsets(offsets, fn):
    """Split per-dim offsets (int/constexpr or scalar tl.tensor) into
    ``(static_offsets, dyn_handles)``; dynamic dims use ``_DYNAMIC`` sentinel."""
    offsets = tl._unwrap_if_constexpr(offsets)
    static = []
    dyn = []
    for v in offsets:
        if isinstance(v, tl.tensor):
            if len(v.shape) != 0:
                raise ValueError(f"{fn}: dynamic offsets must be scalar tl.tensor")
            static.append(_DYNAMIC)
            dyn.append(v.handle)
        else:
            iv = _try_unwrap_int(v)
            if iv is None:
                raise ValueError(f"{fn}: offsets must be int/constexpr or scalar tl.tensor")
            static.append(iv)
    return static, dyn


def _check_static_slice_bounds(fn, src_shape, static_offsets, sizes, strides):
    """Validate the static (non-sentinel) offsets against src/sizes/strides."""
    for i, (src, off, size, stride) in enumerate(zip(src_shape, static_offsets, sizes, strides)):
        if size <= 0:
            raise ValueError(f"{fn}: size[{i}]={size} must be positive")
        if stride <= 0:
            raise ValueError(f"{fn}: stride[{i}]={stride} must be positive")
        if off == _DYNAMIC:
            continue
        if off < 0:
            raise ValueError(f"{fn}: offset[{i}]={off} must be non-negative")
        end = off + (size - 1) * stride + 1
        if end > src:
            raise ValueError(f"{fn}: slice [{off}:{off}+{size}*{stride}] exceeds source dim "
                             f"{i} ({src})")


@tl._tensor_member_fn
@tl.builtin
def extract_slice(x: tl.tensor, offsets, sizes, strides, _semantic=None) -> tl.tensor:
    """Extract a strided slice from ``x``.

    Args:
        x:       Source ``tl.tensor``.
        offsets: Per-dim offsets; each element is an int/constexpr (static) or
            a scalar ``tl.tensor`` (dynamic).
        sizes:   Per-dim slice sizes (compile-time constants); this is the
            result shape (``tensor.extract_slice`` semantics).
        strides: Per-dim strides (compile-time constants).

    Returns a tensor whose shape is ``sizes``.
    """
    if not isinstance(x, tl.tensor):
        raise ValueError(f"extract_slice: source must be tl.tensor, got {type(x)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("extract_slice must be used inside @triton.jit")

    src_shape = _static_dims(x.type.shape, "extract_slice")
    sizes = _static_dims(sizes, "extract_slice")
    strides = _static_dims(strides, "extract_slice")
    offsets = tl._unwrap_if_constexpr(offsets)
    if len(offsets) != len(src_shape) or len(sizes) != len(src_shape) \
            or len(strides) != len(src_shape):
        raise ValueError("extract_slice: offsets/sizes/strides rank must match source rank")

    static_offsets, dyn_offsets = _split_offsets(offsets, "extract_slice")
    _check_static_slice_bounds("extract_slice", src_shape, static_offsets, sizes, strides)

    # tensor.extract_slice semantics: sizes is the result shape.
    result_ty = tl.block_type(x.type.element_ty, sizes)
    result_ir = result_ty.to_ir(builder)

    handle = _dsa_ir.create_dsa_extract_slice(builder, result_ir, x.handle, static_offsets, dyn_offsets, sizes, strides)
    return tl.tensor(handle, result_ty)


@tl._tensor_member_fn
@tl.builtin
def insert_slice(x: tl.tensor, tile: tl.tensor, offsets, sizes=None, strides=None, _semantic=None) -> tl.tensor:
    """Insert ``tile`` into ``x`` at ``offsets``.

    ``sizes`` defaults to ``tile.shape``; ``strides`` defaults to all ones.
    Returns a new tensor with the same shape/type as ``x``.
    """
    if not isinstance(x, tl.tensor):
        raise ValueError(f"insert_slice: source must be tl.tensor, got {type(x)}")
    if not isinstance(tile, tl.tensor):
        raise ValueError(f"insert_slice: tile must be tl.tensor, got {type(tile)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("insert_slice must be used inside @triton.jit")

    src_shape = _static_dims(x.type.shape, "insert_slice")
    tile_shape = _static_dims(tile.type.shape, "insert_slice")
    offsets = tl._unwrap_if_constexpr(offsets)
    if len(offsets) != len(src_shape):
        raise ValueError("insert_slice: offsets rank must match source rank")
    if sizes is None:
        sizes = tile_shape
    else:
        sizes = _static_dims(sizes, "insert_slice")
    if strides is None:
        strides = [1] * len(src_shape)
    else:
        strides = _static_dims(strides, "insert_slice")
    if len(sizes) != len(src_shape) or len(strides) != len(src_shape):
        raise ValueError("insert_slice: sizes/strides rank must match source rank")
    if tuple(sizes) != tuple(tile_shape):
        raise ValueError(f"insert_slice: sizes {sizes} must match tile shape {tile_shape}")
    if x.type.element_ty != tile.type.element_ty:
        raise ValueError(f"insert_slice: element type mismatch source={x.type.element_ty}, "
                         f"tile={tile.type.element_ty}")

    static_offsets, dyn_offsets = _split_offsets(offsets, "insert_slice")
    _check_static_slice_bounds("insert_slice", src_shape, static_offsets, sizes, strides)

    handle = _dsa_ir.create_dsa_insert_slice(builder, x.type.to_ir(builder), x.handle, tile.handle, static_offsets, dyn_offsets,
                                             sizes, strides)
    return tl.tensor(handle, x.type)
