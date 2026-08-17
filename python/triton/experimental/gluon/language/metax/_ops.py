from triton._C.libtriton import ir

from .._core import _unwrap_if_constexpr, builtin
from .. import _core as ttgl
from .._semantic import _check

__all__ = [
    "async_copy_global_to_shared",
    "bsm_perm",
    "barrier",
    "barrier_shared",
    "gvm_arrive",
    "sched_bound",
    "iglp",
]


@builtin
def async_copy_global_to_shared(
    smem,
    pointer,
    mask=None,
    other=None,
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    _semantic=None,
):
    """Asynchronously copy a global tensor into MetaX shared memory."""
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    volatile = _unwrap_if_constexpr(volatile)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)

    _check(pointer.type.is_block(), lambda: "expected pointer to be a tensor")
    _check(
        smem.shape == pointer.shape,
        lambda: (
            "expected smem shape to match pointer shape but got "
            f"smem.shape={smem.shape}, pointer.shape={pointer.shape}"
        ),
    )
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, pointer.dtype.element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)

    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(
        smem.handle,
        pointer.handle,
        mask_handle,
        other_handle,
        cache_modifier,
        eviction_policy,
        volatile,
    )


@builtin
def bsm_perm(value, _semantic=None):
    """Declare the logical C500 BSM permutation.

    The C500 i32 carrier is deliberately introduced only by the late TTGIR
    legalization pass, after the dot operand layout is concrete.
    """
    _check(isinstance(value, ttgl.tensor), lambda: "value must be a tensor")
    _check(
        isinstance(value.type, ttgl.distributed_type),
        lambda: "value must have a distributed_type",
    )
    handle = _semantic.builder.create_bsm_perm(value.handle)
    return ttgl.tensor(handle, value.type)


@builtin
def barrier(_semantic=None):
    _semantic.builder.create_maca_barrier()


@builtin
def barrier_shared(_semantic=None):
    _semantic.builder.create_maca_barrier_shared()


@builtin
def gvm_arrive(num, _semantic=None):
    num = _unwrap_if_constexpr(num)
    _semantic.builder.create_gvm_arrive(num)


@builtin
def sched_bound(_semantic=None):
    _semantic.builder.create_maca_sched_bound()


@builtin
def iglp(
    config_0=0,
    config_1=-1,
    config_2=-1,
    config_3=-1,
    config_4=-1,
    config_5=-1,
    config_6=-1,
    config_7=-1,
    _semantic=None,
):
    configs = [
        _unwrap_if_constexpr(config)
        for config in (
            config_0,
            config_1,
            config_2,
            config_3,
            config_4,
            config_5,
            config_6,
            config_7,
        )
    ]
    _semantic.builder.create_maca_iglp(*configs)
