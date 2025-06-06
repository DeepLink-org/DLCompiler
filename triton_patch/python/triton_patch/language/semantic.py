from typing import List, Optional, Union, Tuple
import numbers
import triton.language as tl
from triton._C.libtriton import ir
from triton.language.semantic import wrap_tensor, _str_to_rounding_mode, not_equal, _str_to_dot_input_precision, \
    binary_op_type_checking_impl, integer_promote_impl, broadcast_impl_shape, _str_to_sem, _str_to_scope


def arange(start: int, end: int, builder: ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")
    is_start_int64 = bool(start >> 32)
    is_end_int64 = bool(end >> 32)
    if is_start_int64 or is_end_int64:
        raise ValueError("arange must fit in int32")
    if end <= start:
        raise ValueError("arange's end argument must be greater than the start argument")
    range = end - start
    # if (range & (range - 1)) != 0:
    #     raise ValueError("arange's range must be a power of 2")
    shape = [range]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.create_make_range(start, end), ret_ty)

def cast(input: tl.tensor, dst_ty: tl.dtype, builder: ir.builder,
         fp_downcast_rounding: Optional[str] = None) -> tl.tensor:
    src_ty = input.type
    if isinstance(dst_ty, tl.constexpr):
        dst_ty = dst_ty.value
    if isinstance(fp_downcast_rounding, tl.constexpr):
        fp_downcast_rounding = fp_downcast_rounding.value
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input

    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar

    # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
    # not be set
    fp_downcast_rounding = _str_to_rounding_mode(fp_downcast_rounding)
    use_custom_rounding = False
    if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
    ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
        if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
        elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
    else:
        if fp_downcast_rounding is not None:
            raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                             "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))

    if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
        raise ValueError("[fp8, fp64] is unsupported on Ascend for now."
                         "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))
    if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
        assert builder.codegen_fns.get(
            "convert_custom_types") is not None, "target doesn't provide conversion for this type."
        return builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _builder=builder)
    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    # and non-default rounding modes for downcasting
    if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
       use_custom_rounding:
        return tl.tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder), fp_downcast_rounding), dst_ty)

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)

    # Standard floating types' casting: truncation
    #   fp64 => fp32, fp16, bf16
    #   fp32 => fp16, bf16
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tl.tensor(builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        else:
            return tl.tensor(builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend), dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tl.tensor(builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_ui(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to integer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tl.tensor(builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)), dst_ty)
        if bitwidth == 1:
            return not_equal(cast(input, tl.int64, builder), tl.tensor(builder.get_int64(0), tl.int64), builder)

    # Casting integer types to pointer types
    if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to pointer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'

def dot(lhs: tl.tensor, rhs: tl.tensor, acc: tl.tensor, input_precision: Optional[str], max_num_imprecise_acc: int,
        out_dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()

    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        # All combinations of supported fp8 x fp8 are permitted
        pass
    else:
        assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32), f"Unsupported lhs dtype {lhs.dtype}"
        assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32), f"Unsupported rhs dtype {rhs.dtype}"
        assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"

    if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
        lhs = cast(lhs, tl.float16, builder)
        rhs = cast(rhs, tl.float16, builder)

    if input_precision is None:
        input_precision = builder.options.default_dot_input_precision

    input_precision = _str_to_dot_input_precision(input_precision, builder)

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    assert lhs.shape[-1].value == rhs.shape[
        -2].value, f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({lhs.shape[-1].value}) must be equal to first index of second shape ({rhs.shape[-2].value})"
    assert builder.codegen_fns.get("min_dot_size") is not None, "target doesn't provide lower shape bounds for dot."
    min_dot_size = builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
    assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
        and rhs.shape[-1].value >= min_dot_size[1], \
            f"Input shapes should have M >= {min_dot_size[0]}, N >= {min_dot_size[1]} and K >= {min_dot_size[2]}"
    if lhs.type.scalar.is_int():
        assert lhs.type.scalar == tl.int8, "only int8 supported!"
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError(
            "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`")
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    else:
        _0 = builder.get_fp16(0) if out_dtype.is_fp16() else builder.get_fp32(0)
        ret_scalar_ty = out_dtype

    M = lhs.type.shape[-2]
    N = rhs.type.shape[-1]
    K = lhs.type.shape[-1]
    B = lhs.type.shape[0] if lhs_rank == 3 else None
    ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
    if acc is None:
        acc_handle = builder.create_splat(_0, [B, M, N] if B else [M, N])
    else:
        acc_handle = acc.handle
        assert acc.type == ret_ty

    if (input_precision == getattr(ir.INPUT_PRECISION, "HF32")):
        if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32() or not ret_scalar_ty.is_fp32()):
            raise ValueError("input_precision = 'hf32' must be used with f32 * f32 = f32 on Ascend")

    if max_num_imprecise_acc is not None:
        tl.static_print("max_num_imprecise_acc is not supported on Ascend yet. Thus it is ignored.")
    max_num_imprecise_acc = 0
    return tl.tensor(builder.create_dot(lhs.handle, rhs.handle, acc_handle, input_precision, max_num_imprecise_acc),
                     ret_ty)

# Use Union instead of |. Becase python 3.9 does not support |.
# It will reports error: TypeError: unsupported operand type(s) for |: 'type' and 'ABCMeta'
def floordiv(input: Union[tl.tensor, numbers.Number], other: Union[tl.tensor, numbers.Number], builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_bool() or other_scalar_ty.is_bool():
        raise TypeError(f"unexpected type {input_scalar_ty}")
    if input_scalar_ty.is_int() and other_scalar_ty.is_int():
        ret_ty = integer_promote_impl(input_scalar_ty, other_scalar_ty)
        input = cast(input, ret_ty, builder)
        other = cast(other, ret_ty, builder)
        if ret_ty.is_int_signed():
            return tl.tensor(builder.create_sdiv(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_udiv(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {input_scalar_ty}")

def gather(src: tl.tensor, index: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    assert index.dtype.is_int(), "index must be an integer tensor"

    rank = len(src.type.shape)
    assert len(index.type.shape) == rank, "source and index tensors must have the same rank"

    assert -rank <= axis < rank, f"gather axis {axis} must be < source rank ({rank})"
    if axis < 0:
        axis += rank

    for d in range(rank):
        if d == axis:
            continue
        assert index.type.shape[d] == src.type.shape[d], f"index dim {axis} must match the corresponding source dim"

    gather = builder.create_gather(src.handle, index.handle, axis)
    return wrap_tensor(gather, src.type.scalar, index.type.shape)

def insert(ful: tl.tensor, sub: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = builder.create_insert(ful.handle, sub.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

    # Check `mask`, `other`, `boundary_check`, and `padding` arguments
    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                         "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                         "use block pointers (defined by `make_block_ptr`) instead")

    # For a pointer of scalar, check the type of `mask` and `other`
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
        if other and other.type.is_block():
            raise ValueError("Other argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `other` into the same shape as `ptr`
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other is not None:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    # Get `pointer_type<elt_ty>` and `elt_ty`
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast `other` into `elt_ty` type
    if other is not None:
        other = cast(other, elt_ty, builder)

    # Create loaded result type `dst_ty`
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = elt_ty

    # Build IR
    if mask is None:
        ret = tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
    else:
        ret = tl.tensor(
            builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache, eviction,
                                       is_volatile), dst_ty)
    # Do not cast back to int1 when is_bool=true. We directly use the int8 tensor given by tl.load
    return ret

def minimum(x: tl.tensor, y: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    x, y = binary_op_type_checking_impl(x, y, builder)
    dtype = x.dtype
    if dtype.is_bool():
        raise TypeError(f"Unexpected dtype {dtype}")
    if dtype.is_floating():
        if propagate_nan == tl.PropagateNan.ALL:
            return tl.tensor(builder.create_minimumf(x.handle, y.handle), x.type)
        elif propagate_nan == tl.PropagateNan.NONE:
            return tl.tensor(builder.create_minnumf(x.handle, y.handle), x.type)
        else:
            raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_minsi(x.handle, y.handle), x.type)
    elif dtype.is_int_unsigned():
        return tl.tensor(builder.create_minui(x.handle, y.handle), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}")

def maximum(x: tl.tensor, y: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    x, y = binary_op_type_checking_impl(x, y, builder)
    dtype = x.dtype
    if dtype.is_bool():
        raise TypeError(f"Unexpected dtype {dtype}")
    if dtype.is_floating():
        if propagate_nan == tl.PropagateNan.ALL:
            return tl.tensor(builder.create_maximumf(x.handle, y.handle), x.type)
        elif propagate_nan == tl.PropagateNan.NONE:
            return tl.tensor(builder.create_maxnumf(x.handle, y.handle), x.type)
        else:
            raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_maxsi(x.handle, y.handle), x.type)
    elif dtype.is_int_unsigned():
        return tl.tensor(builder.create_maxui(x.handle, y.handle), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}")

def subview(ful: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = builder.create_slice(ful.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def atom_red_typechecking_impl(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, op: str,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
    if ptr.type.is_const() or ptr.type.element_ty.is_const():
        raise ValueError("Cannot store to a constant pointer")
    element_ty = ptr.type.scalar.element_ty
    # Add `tl.int64` restriction for NPU
    if element_ty in [tl.int1, tl.int64, tl.float64]:
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, float16, float32, bfloat16.")
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val is not None:
            val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask

def atomic_max(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'max', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
        else:
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    # Design for NPU
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

def atomic_min(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'min', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
        else:
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    # Design for NPU
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)