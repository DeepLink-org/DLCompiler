import torch
import triton
import triton.language as tl
from triton_dist.language.extra.cuda.language_extra import tid, st, ld
from triton.language import core
from triton_dist.language import core as dist_core
from typing import Any


@triton.jit
def zero_vec_f32(vec_size: tl.constexpr, _semantic=None):
    z = tl.cast(0, tl.float32)
    if vec_size == 1:
        return z
    elif vec_size == 2:
        return z, z
    elif vec_size == 4:
        return z, z, z, z
    elif vec_size == 8:
        return z, z, z, z, z, z, z, z
    else:
        assert False, "unsupported vec_size"


@core.extern
def unpack_bf16x2_f32(
    v1,
    v2,
    v3,
    v4,
    _semantic=None,
):
    callee_name = f"__triton_maca_unpack_bf16x2_f32"
    inpTy = v1.dtype
    assert inpTy == tl.int32, "maca unpack_bf16x2_f32 only support input type int32"
    outTy = tl.float32
    return dist_core.extern_call(
        "",
        "",
        [v1, v2, v3, v4],
        {(inpTy, inpTy, inpTy, inpTy): (callee_name, (outTy, outTy, outTy, outTy, outTy, outTy, outTy, outTy))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def pack_f32_bf16x2(
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    _semantic=None,
):
    callee_name = f"__triton_maca_pack_f32_bf16x2"
    inpTy = v1.dtype
    assert inpTy == tl.float32, "maca pack_f32_bf16x2 only support input type float32"
    outTy = tl.int32
    return dist_core.extern_call(
        "",
        "",
        [v1, v2, v3, v4, v5, v6, v7, v8],
        {(inpTy, inpTy, inpTy, inpTy, inpTy, inpTy, inpTy, inpTy): (callee_name, (outTy, outTy, outTy, outTy))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def load_v4(
    ptr,
    suffix: core.constexpr,
    _semantic=None,
):
    assert ptr.dtype.is_ptr(), "load_v4 ptr should be a pointer"
    assert core._unwrap_if_constexpr(suffix) in ["b32"], "maca load_v4 only supports b32*4 for now"
    elemTy = ptr.dtype.element_ty
    assert elemTy.name in ["int32", "fp32", "bf16", "fp16"], "maca load_v4 only supports int32, fp32, fp16, bf16"
    outTy = tl.int32

    callee_name = f"__triton_maca_loadv4_b128"
    return dist_core.extern_call(
        "",
        "",
        [core.cast(ptr, dtype=core.pointer_type(tl.uint32), _semantic=_semantic)],
        {(core.pointer_type(tl.uint32), ): (callee_name, (outTy, outTy, outTy, outTy))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def store_v4(ptr, v0, v1, v2, v3, suffix: core.constexpr, _semantic=None):
    assert ptr.dtype.is_ptr(), "store_v4 ptr should be a pointer"
    assert core._unwrap_if_constexpr(suffix) in ["b32"], "maca store_v4 only supports b32*4 for now"

    elemTy = v0.dtype
    assert (elemTy in [core.dtype("int32"), core.dtype("fp32"),
                       core.dtype("uint32")] and elemTy == v1.dtype and elemTy == v2.dtype
            and elemTy == v3.dtype), "maca load_v4 only supports int32, fp32, uint32"

    callee_name = f"__triton_maca_storev4_b128"
    return dist_core.extern_call(
        "",
        "",
        [
            core.cast(ptr, dtype=core.pointer_type(tl.uint32), _semantic=_semantic),
            core.cast(v0, dtype=tl.uint32, _semantic=_semantic),
            core.cast(v1, dtype=tl.uint32, _semantic=_semantic),
            core.cast(v2, dtype=tl.uint32, _semantic=_semantic),
            core.cast(v3, dtype=tl.uint32, _semantic=_semantic),
        ],
        {(core.pointer_type(tl.uint32), tl.uint32, tl.uint32, tl.uint32, tl.uint32): (callee_name, ())},
        is_pure=False,
        _semantic=_semantic,
    )
