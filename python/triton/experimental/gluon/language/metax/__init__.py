from triton._C.libtriton import ir

from .. import _core as ttgl
from .._core import _unwrap_if_constexpr, builtin
from .._layouts import MACAMmaLayout
from .._semantic import _check

__all__ = [
    "MACAMmaLayout",
    "async_copy_global_to_shared",
    "bsm_perm",
    "extract_tensor",
    "insert_tensor",
    "gvm_arrive",
    "barrier",
    "barrier_shared",
    "sched_bound",
    "iglp",
]


@builtin
def async_copy_global_to_shared(smem, pointer, mask=None, other=None, cache_modifier="", eviction_policy="",
                                volatile=False, intrinsic=True, _semantic=None):
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    volatile = _unwrap_if_constexpr(volatile)
    intrinsic = _unwrap_if_constexpr(intrinsic)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)

    _check(pointer.type.is_block(), lambda: "expected pointer to be a tensor")
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape={smem.shape}, pointer.shape={pointer.shape}"
    )
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, pointer.dtype.element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)

    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, other_handle,
                                                        cache_modifier, eviction_policy, volatile, intrinsic)


@builtin
def bsm_perm(value, dtype, _semantic=None):
    dtype = _unwrap_if_constexpr(dtype)
    _check(isinstance(value, ttgl.tensor), lambda: "value must be a tensor")
    _check(isinstance(value.type, ttgl.distributed_type), lambda: "value must have a distributed_type")
    ret_ty = ttgl.distributed_type(dtype, value.shape, value.type.layout)
    handle = _semantic.builder.create_bsm_perm(ret_ty.to_ir(_semantic.builder), value.handle)
    return ttgl.tensor(handle, ret_ty)


@builtin
def extract_tensor(source, shape, cta_idx, elem_idx, _semantic=None):
    shape = [_unwrap_if_constexpr(dim) for dim in shape]
    cta_idx = [_unwrap_if_constexpr(idx) for idx in cta_idx]
    elem_idx = [_unwrap_if_constexpr(idx) for idx in elem_idx]
    _check(isinstance(source, ttgl.tensor), lambda: "source must be a tensor")
    _check(isinstance(source.type, ttgl.distributed_type), lambda: "source must have a distributed_type")
    ret_ty = ttgl.distributed_type(source.dtype, shape, source.type.layout)
    handle = _semantic.builder.create_extract_tensor(ret_ty.to_ir(_semantic.builder), source.handle, cta_idx, elem_idx)
    return ttgl.tensor(handle, ret_ty)


@builtin
def insert_tensor(inserted, insert, cta_idx, elem_idx, _semantic=None):
    cta_idx = [_unwrap_if_constexpr(idx) for idx in cta_idx]
    elem_idx = [_unwrap_if_constexpr(idx) for idx in elem_idx]
    _check(isinstance(inserted, ttgl.tensor), lambda: "inserted must be a tensor")
    _check(isinstance(insert, ttgl.tensor), lambda: "insert must be a tensor")
    _check(isinstance(inserted.type, ttgl.distributed_type), lambda: "inserted must have a distributed_type")
    handle = _semantic.builder.create_insert_tensor(inserted.type.to_ir(_semantic.builder), inserted.handle,
                                                    insert.handle, cta_idx, elem_idx)
    return ttgl.tensor(handle, inserted.type)


@builtin
def gvm_arrive(num, _semantic=None):
    num = _unwrap_if_constexpr(num)
    _semantic.builder.create_gvm_arrive(num)


@builtin
def barrier(_semantic=None):
    _semantic.builder.create_maca_barrier()


@builtin
def barrier_shared(_semantic=None):
    _semantic.builder.create_maca_barrier_shared()


@builtin
def sched_bound(_semantic=None):
    _semantic.builder.create_maca_sched_bound()


@builtin
def iglp(config_0=0, config_1=-1, config_2=-1, config_3=-1, config_4=-1, config_5=-1, config_6=-1, config_7=-1,
         _semantic=None):
    config_0 = _unwrap_if_constexpr(config_0)
    config_1 = _unwrap_if_constexpr(config_1)
    config_2 = _unwrap_if_constexpr(config_2)
    config_3 = _unwrap_if_constexpr(config_3)
    config_4 = _unwrap_if_constexpr(config_4)
    config_5 = _unwrap_if_constexpr(config_5)
    config_6 = _unwrap_if_constexpr(config_6)
    config_7 = _unwrap_if_constexpr(config_7)
    _semantic.builder.create_maca_iglp(config_0, config_1, config_2, config_3, config_4, config_5, config_6, config_7)
