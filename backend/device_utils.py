# Copyright (c) 2025, DeepLink.
import functools
from typing import Optional

import torch
import triton

WARPS_PER_SM = {
    (8, 0): 64,
    (8, 6): 48,
    (8, 7): 48,
    (8, 9): 48,
    (9, 0): 64,
    (10, 0): 64,
    (10, 1): 48,
    (12, 0): 48,
}


@functools.lru_cache
def get_device_props(device=None):
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    warps_per_sm = WARPS_PER_SM.get((props.major, props.minor), 32)
    out = dict(
        multi_processor_count=props.multi_processor_count,
        warps_per_sm=warps_per_sm,
    )
    return out


@functools.lru_cache
def get_number_cores():
    if is_npu():
        import triton.runtime.driver as driver

        device = torch.npu.current_device()
        return driver.active.utils.get_device_properties(device)["num_aicore"]
    elif is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    else:
        raise RuntimeError("Please implement this function.")


def is_mlu_592():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "mlu" and target.arch == 592


def is_muxi():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "maca"


@functools.lru_cache
def is_cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@functools.lru_cache
def is_npu():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


@functools.lru_cache
def is_tesla():
    try:
        return "Tesla" in torch.cuda.get_device_name(0)
    except Exception:
        return False


@functools.lru_cache
def is_nvidia_hopper():
    try:
        return is_cuda() and (
            "NVIDIA H" in torch.cuda.get_device_name(0)
            or torch.cuda.get_device_capability()[0] >= 9
        )
    except Exception:
        return False


def set_allocator(device_: str):
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device=device_, dtype=torch.int8)

    triton.set_allocator(alloc_fn)


@functools.lru_cache
def is_tma_supported():
    try:
        is_tma_supported = (
            is_cuda()
            and torch.cuda.get_device_capability(0)[0] >= 9
            and (
                hasattr(triton.language, "_experimental_make_tensor_descriptor")
                or hasattr(triton.language, "make_tensor_descriptor")
            )
        )
        if is_tma_supported:
            set_allocator("cuda")
        return is_tma_supported
    except Exception:
        return False


@functools.lru_cache
def infer_device():
    """
    Get current device name based on available devices
    """
    if is_npu():
        return "npu"
    elif is_mlu_592():
        return "mlu"
    elif is_muxi():
        return "cuda"
    elif is_nvidia_hopper():
        return "cuda"
    elif is_cuda():
        return "cuda"
    else:
        return "cpu"


NUM_CORES = get_number_cores()
DEVICE = infer_device()
