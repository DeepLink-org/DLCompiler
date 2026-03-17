import triton.language.core as tl

import re
import triton.runtime.driver as driver


def is_hip():
    target = driver.active.get_current_target()
    return target.backend == "hip"


@tl.builtin
def thread_id(axis, _semantic=None):
    """
    Returns the id of the current thread instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = tl._unwrap_if_constexpr(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"thread_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(_semantic.builder.create_thread_id(axis), tl.int32)


@tl.builtin
def async_task_replica_id(_semantic=None):
    from triton.language.extra.deeplink.compiler.code_generator import region_replica_id_stack

    assert len(region_replica_id_stack) > 0, (
        "async_task_replica_id must be called inside an async region where the stack must be non-empty")
    return tl.constexpr(region_replica_id_stack[-1])


@tl.builtin
def dtype_of(v, _semantic=None) -> tl.dtype:
    """
    Returns the element type of a given tensor or tensor descriptor.
    """
    if isinstance(v, tl.tensor):
        dtype = v.type.element_ty
        if dtype.is_ptr():
            dtype = dtype.element_ty
        return dtype
    elif isinstance(v, tl.tensor_descriptor_base):
        return v.dtype
    else:
        raise ValueError(f"dtype_of only works on tensors and tensor descriptors, but got {v}")


@tl.builtin
def size_of(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    """
    Returns the size of a given dtype.
    """
    assert isinstance(dtype, tl.dtype), f"size_of expects a dtype, but got {type(dtype)}"
    return tl.constexpr(dtype.primitive_bitwidth // 8)

