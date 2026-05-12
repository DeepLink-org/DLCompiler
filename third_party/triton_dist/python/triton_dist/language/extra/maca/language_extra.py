import triton
import triton.language as tl
from triton.language import core
from triton_dist.language import core as dist_core

# Scope name mapping: legacy names -> LLVM scope names
_SCOPE_MAP = {
    "cta": "block",
    "gpu": "device",
    "sys": "system",
    "device": "device",
    "system": "system",
    "block": "block",
    "warp": "warp",
    "one-as": "one-as",
    "device-one-as": "device-one-as",
    "block-one-as": "block-one-as",
    "warp-one-as": "warp-one-as",
    "singlethread-one-as": "singlethread-one-as",
}

# Valid scope names after mapping
_VALID_SCOPES = [
    "device", "system", "block", "warp", "one-as", "device-one-as", "block-one-as", "warp-one-as", "singlethread-one-as"
]


def _get_type_suffix(dtype):
    """Get type suffix for function name."""
    dtype_str = str(dtype)
    if dtype_str == "int32":
        return "i32"
    elif dtype_str == "uint32":
        return "u32"
    elif dtype_str == "int64":
        return "i64"
    elif dtype_str == "uint64":
        return "u64"
    elif dtype_str == "fp32":
        return "f32"
    elif dtype_str == "fp16":
        return "f16"
    elif dtype_str == "bf16":
        return "bf16"
    else:
        return dtype_str


def _translate_scope(scope):
    """Translate scope name to LLVM scope name."""
    _scope = core._unwrap_if_constexpr(scope)
    assert _scope in _SCOPE_MAP, f"scope should be one of {list(_SCOPE_MAP.keys())}"
    return _SCOPE_MAP[_scope]


def _translate_semantic(semantic):
    """Translate semantic name to LLVM semantic name."""
    _semantic = core._unwrap_if_constexpr(semantic)
    if _semantic == "relaxed":
        return "monotonic"
    elif _semantic in ["monotonic", "acquire", "release", "acq_rel"]:
        return _semantic
    else:
        raise ValueError(f"Unsupported semantic: {_semantic}")


@core.extern
def __syncthreads(_semantic=None):
    return tl.debug_barrier(_semantic=_semantic)


@core.extern
def __tid__(axis: core.constexpr, _semantic=None):
    return tl.inline_intrinsic_elementwise(
        intrinsic=f"llvm.mxc.thread.id.{axis.value}",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def tid(axis: core.constexpr, _semantic=None):
    if axis == 0:
        return __tid__(core.constexpr("x"), _semantic=_semantic)
    elif axis == 1:
        return __tid__(core.constexpr("y"), _semantic=_semantic)
    elif axis == 2:
        return __tid__(core.constexpr("z"), _semantic=_semantic)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


@core.extern
def laneid(_semantic=None):
    return core.tensor(_semantic.builder.create_laneid(), core.int32)


@core.extern
def ld(
    ptr,
    scope="gpu",
    semantic="relaxed",
    _semantic=None,
):
    """
    semantic should be one of ["relaxed", "acquire"]
    scope should be one of ["device", "system", "block", "warp", "one-as",
                            "device-one-as", "block-one-as", "warp-one-as", "singlethread-one-as"]
    """
    assert ptr.dtype.is_ptr(), "ld(ptr, scope) should be a pointer"
    assert core._unwrap_if_constexpr(semantic) in [
        "relaxed",
        "acquire",
    ], "load only supports 'relaxed' and 'acquire' semantics"

    semantic = _translate_semantic(semantic)
    _scope = _translate_scope(scope)
    assert core._unwrap_if_constexpr(_scope) in _VALID_SCOPES, f"scope should be one of {_VALID_SCOPES}"

    base_name = f"__triton_maca_load_{core._unwrap_if_constexpr(semantic)}_{_scope}"
    return dist_core.extern_elementwise(
        "",
        "",
        [ptr],
        {(core.pointer_type(dtype), ): (f"{base_name}_{_get_type_suffix(dtype)}", dtype)
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
             core.dtype("fp32"),
             core.dtype("fp16"),
             core.dtype("bf16"),
         ]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ld_b32(ptr, _semantic=None):
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int32(),
        "ld_b32(ptr) argument 0 `ptr` should be a pointer of int type",
        _semantic=_semantic,
    )
    return ld(ptr, scope="gpu", semantic="relaxed", _semantic=_semantic)


@core.extern
def ld_acquire(ptr, scope: core.constexpr = "gpu", _semantic=None):
    return ld(ptr, scope, "acquire", _semantic=_semantic)


@core.extern
def st(
    ptr,
    val,
    scope="gpu",
    semantic="relaxed",
    _semantic=None,
):
    assert core._unwrap_if_constexpr(semantic) in [
        "relaxed",
        "release",
    ], "store only supports 'monotonic' and 'release' semantics"

    semantic = _translate_semantic(semantic)
    scope = _translate_scope(scope)
    assert core._unwrap_if_constexpr(scope) in _VALID_SCOPES, f"scope should be one of {_VALID_SCOPES}"

    base_name = f"__triton_maca_store_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}"
    return dist_core.extern_elementwise(
        "",
        "",
        [ptr, core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic)],
        {(core.pointer_type(dtype), dtype): (f"{base_name}_{_get_type_suffix(dtype)}", dtype)
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
             core.dtype("fp32"),
             core.dtype("fp16"),
             core.dtype("bf16"),
         ]},
        is_pure=False,
        _semantic=_semantic,
        check_args=False,
    )


@core.extern
def atomic_add(
    ptr,
    value,
    scope="device",
    semantic="relaxed",
    _semantic=None,
):
    """
    semantic should be one of ["monotonic", "release", "acquire", "acq_rel"]
    scope should be one of ["device", "system", "block", "warp", "one-as",
                            "device-one-as", "block-one-as", "warp-one-as", "singlethread-one-as"]
    """
    semantic = _translate_semantic(semantic)
    scope = _translate_scope(scope)
    assert core._unwrap_if_constexpr(semantic) in [
        "monotonic",
        "release",
        "acquire",
        "acq_rel",
    ], "semantic should be one of ['monotonic', 'release', 'acquire', 'acq_rel']"
    assert core._unwrap_if_constexpr(scope) in _VALID_SCOPES, f"scope should be one of {_VALID_SCOPES}"

    base_name = f"__triton_maca_atomic_add_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}"
    return dist_core.extern_elementwise(
        "",
        "",
        [
            ptr,
            core.cast(value, dtype=ptr.dtype.element_ty, _semantic=_semantic),
        ],
        {(core.pointer_type(dtype), dtype): (f"{base_name}_{_get_type_suffix(dtype)}", dtype)
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
         ]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def __shfl_sync_with_mode_i32(
    value,
    offset,
    mode: core.constexpr = "up",
    width: int = 64,
    _semantic=None,
):
    shfl_mode = mode.value

    callee_name = f"__triton_maca_shfl_{shfl_mode}"
    return dist_core.extern_elementwise(
        "",
        "",
        [value, core.cast(offset, dtype=core.dtype("int32"), _semantic=_semantic)],
        {(value.dtype, core.dtype("int32")): (callee_name, value.dtype)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def __shfl_sync_i32(value, laneid):
    return __shfl_sync_with_mode_i32(value, laneid, "idx", 64)


@triton.jit
def atomic_add_per_warp(barrier_ptr, value, scope: core.constexpr, semantic: core.constexpr):
    _laneid = laneid()
    x = tl.cast(0, barrier_ptr.dtype.element_ty)
    if _laneid == 0:
        x = atomic_add(barrier_ptr, value, scope, semantic)
    return __shfl_sync_i32(x, 0)


__all__ = [
    "__syncthreads",
    "tid",
    "laneid",
    "ld",
    "ld_b32",
    "ld_acquire",
    "st",
    "atomic_add",
    "__shfl_sync_i32",
    "atomic_add_per_warp",
]
