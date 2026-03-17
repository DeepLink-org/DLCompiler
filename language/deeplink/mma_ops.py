import triton.language.core as tl
from typing import Optional, Tuple, Any
from . import types as tlx
from triton._C.libtriton import getenv


def require_nv_mma_shared_layout(x: tlx.buffered_tensor, swizzled: bool, _builder=None):
    assert isinstance(
        x.type.layout, tlx.shared_layout_encoding
    ), "input must be a shared tensor"
    rank = len(x.shape)
    layout = tlx.nv_mma_shared_layout_encoding(
        shape=x.shape,
        order=x.type.layout.order,
        elemType=x.dtype,
        numCTAsPerCGA=[1] * rank,
        numCTASplit=[1] * rank,
        numCTAOrder=[1] * rank,
        fp4Padded=False,
        swizzled=swizzled,
    )

    layout_handle = _builder.make_nv_mma_shared_encoding_attr(
        [int(x) for x in layout.shape],
        layout.order,
        layout.elemType.to_ir(_builder),
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
        layout.fp4Padded,
        layout.swizzled,
    )
    return _builder.create_require_layout(x.handle, layout_handle)


def require_dot_operand_layout(opnd: tl.tensor, opIdx, parent_layout, _builder=None):
    layout_handle = _builder.make_dot_operand_encoding_attr(
        opnd.handle, opIdx, parent_layout
    )
    return _builder.create_require_layout(opnd.handle, layout_handle)


def dot_precheck(
    _semantic,
    lhs: tl.tensor,
    rhs: tl.tensor,
    acc: tl.tensor,
    input_precision: Optional[str],
    allow_tf32,
    max_num_imprecise_acc: int,
    out_dtype: tl.dtype,
    tlx_paired_ctas: bool = False,
) -> Tuple[Any]:
    input_precision = tl._unwrap_if_constexpr(input_precision)
    allow_tf32 = tl._unwrap_if_constexpr(allow_tf32)
    assert (
        input_precision is None or tl._unwrap_if_constexpr(allow_tf32) is None
    ), "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = getenv("TRITON_F32_DEFAULT") or (
            "tf32" if (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
        )

    input_precision = tl._unwrap_if_constexpr(input_precision)
    out_dtype = tl._unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = tl._unwrap_if_constexpr(max_num_imprecise_acc)
    acc = tl._unwrap_if_constexpr(acc)

    assert lhs.type.is_block() and rhs.type.is_block()

    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        # All combinations of supported fp8 x fp8 are permitted
        pass
    else:
        assert lhs.dtype in (
            tl.int8,
            tl.uint8,
            tl.float16,
            tl.bfloat16,
            tl.float32,
            tl.float64,
        ), f"Unsupported lhs dtype {lhs.dtype}"
        assert rhs.dtype in (
            tl.int8,
            tl.uint8,
            tl.float16,
            tl.bfloat16,
            tl.float32,
            tl.float64,
        ), f"Unsupported rhs dtype {rhs.dtype}"
        assert (
            lhs.dtype == rhs.dtype
        ), f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"

    if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
        # We upcast because there's no fp8e4b15 type in MLIR
        lhs = _semantic.cast(lhs, tl.float16)
        rhs = _semantic.cast(rhs, tl.float16)

    uses_fp8e4b8 = lhs.dtype.is_fp8e4b8() or rhs.dtype.is_fp8e4b8()
    uses_fp8e5b16 = lhs.dtype.is_fp8e5b16() or rhs.dtype.is_fp8e5b16()
    if uses_fp8e4b8 or uses_fp8e5b16:
        type_name = "fp8e4b8" if uses_fp8e4b8 else "fp8e5b16"
        if type_name in _semantic.builder.options.deprecated_fp8_dot_operand_dtypes:
            arch = _semantic.builder.options.arch
            lhs = _semantic.cast(lhs, tl.float16)
            rhs = _semantic.cast(rhs, tl.float16)

    if input_precision is None:
        input_precision = _semantic.builder.options.default_dot_input_precision

    input_precision = _semantic._str_to_dot_input_precision(input_precision)

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert (
        lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3
    ), f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"

    assert tl._unwrap_if_constexpr(lhs.shape[-1]) == tl._unwrap_if_constexpr(
        rhs.shape[-2]
    ), f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({tl._unwrap_if_constexpr(lhs.shape[-1])}) must be equal to first index of second shape ({tl._unwrap_if_constexpr(rhs.shape[-2])})"

    assert (
        _semantic.builder.codegen_fns.get("min_dot_size") is not None
    ), "target doesn't provide lower shape bounds for dot."
    min_dot_size = _semantic.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
    assert (
        tl._unwrap_if_constexpr(lhs.shape[-2]) >= min_dot_size[0]
        and tl._unwrap_if_constexpr(lhs.shape[-1]) >= min_dot_size[2]
        and tl._unwrap_if_constexpr(rhs.shape[-1]) >= min_dot_size[1]
    ), f"Input shapes should have M >= {min_dot_size[0]}, N >= {min_dot_size[1]} and K >= {min_dot_size[2]}"
    if lhs.type.scalar.is_int():
        assert lhs.type.scalar == tl.int8, "only int8 supported!"
        _0 = _semantic.builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError(
            "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`"
        )
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = _semantic.builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    elif lhs.type.scalar.is_fp64():
        _0 = _semantic.builder.get_fp64(0)
        ret_scalar_ty = tl.float64
    else:
        _0 = (
            _semantic.builder.get_fp16(0)
            if out_dtype.is_fp16()
            else _semantic.builder.get_fp32(0)
        )
        ret_scalar_ty = out_dtype

    M = lhs.type.shape[-2]
    if tlx_paired_ctas:
        N = (
            2 * rhs.type.shape[-1]
        )  # rhs is actually [K, N/2] in two_ctas mode so we scale it back
    else:
        N = rhs.type.shape[-1]
    K = lhs.type.shape[-1]
    B = lhs.type.shape[0] if lhs_rank == 3 else None
    ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])

    if acc is None:
        acc_handle = _semantic.builder.create_splat(ret_ty.to_ir(_semantic.builder), _0)
    else:
        acc_handle = acc.handle
        assert acc.type.shape == ret_ty.shape and acc.type.element_ty == out_dtype

    # max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
    if max_num_imprecise_acc is None:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
            max_num_imprecise_acc = _semantic.builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    else:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and max_num_imprecise_acc > K:
            raise ValueError(
                f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})"
            )
    return (lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc, ret_ty)


# async dot signature needs to be close to tl.dot as much as possible
@tl.builtin
def async_dot(
    A: tlx.buffered_tensor | tl.tensor,
    B: tlx.buffered_tensor,
    acc: tlx.buffered_tensor | tl.tensor | None = None,
    use_acc: (
        tl.constexpr | tl.tensor
    ) = None,  # For blackwell, compute D = A @ B + D instead of D = A @ B. If None, default to True.
    pred=None,
    mBarriers: list[tlx.mbarrier] = [],
    two_ctas: bool = False,
    input_precision=None,
    out_dtype=tl.float32,
    _semantic=None,
) -> tl.tensor:
    """
    Performs a warp-group matrix multiply-accumulate operation of two blocks and return the matrix product.

    This maps directly to NVIDIA Hopper’s wgmma.mma_async instructions, enabling high-throughput matrix multiplication
    across multiple warps within a warpgroup, or Blackwell's tcgen05.mma instruction.

    The operation computes:
        D = A @ B + C

    Where:

        A: A matrix tile held in registers or shared memory

        B: A matrix tile loaded from shared memory

        C is an accumulator tile in registers

        D is the output tile in registers

    input_precision can be one of: tf32, tf32x3, ieee.
    """

    # Perform dot_precheck shared by tl.dot
    (A, B, acc_handle, input_precision, max_num_imprecise_acc, ret_ty) = dot_precheck(_semantic,
        A, B, acc, input_precision, None, None, out_dtype, two_ctas
    )

    assert A.shape[0] >= 64, "M must be at least 64"
    assert A.shape[1] >= 16, "K must be at least 16"
    assert B.shape[1] >= 32, "N must be at least 32"

    cuda_compute_capability = 90
    version = 5 if cuda_compute_capability >= 100 else 3

    # TODO. batched dot is not supported yet
    if isinstance(A, tlx.buffered_tensor) and A.type.storage == tlx.storage_kind.smem:
        A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder)
    elif isinstance(A, tl.tensor):
        assert (
            cuda_compute_capability < 100
        ), "register operand is not supported on Blackwell"
        A_handle = A.handle
    else:
        assert False, "unreach"

    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder)

    if version == 5:
        assert False, "unreach"
    else:
        mma_layout = _semantic.builder.make_nv_mma_encoding_attr(
            A_handle, acc_handle, version, 0, _semantic.builder.options.num_warps
        )
        acc = _semantic.builder.create_require_layout(acc_handle, mma_layout)
        if isinstance(A, tl.tensor):
            A_handle = require_dot_operand_layout(A, 0, mma_layout, _semantic.builder)
        output = _semantic.builder.create_warp_group_dot(
            A_handle, B_handle, acc, input_precision, max_num_imprecise_acc, True
        )
        # Release the mma layout for the output to conform to what the user expects
        output = _semantic.builder.create_release_layout(output)
        return tl.tensor(output, ret_ty)


@tl.builtin
def async_dot_wait(
    pendings: tl.constexpr,
    inp: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Wait for completion of prior asynchronous dot operations.
    Each input must be the tensors corresponding to the async dot ops that we're
    waiting on.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    return tl.tensor(
        _semantic.builder.create_warp_group_dot_wait([inp.handle], pendings)[0],
        inp.type,
    )
