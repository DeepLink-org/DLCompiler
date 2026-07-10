import argparse
import os
import sys
import time

import torch

import triton
import triton.language as tl

try:
    import backend.ascend_autotune_hooks  # noqa: F401
except ImportError:
    pass

try:
    import triton.backends.dicp_triton.ascend_autotune_hooks  # noqa: F401
except ImportError:
    pass

try:
    import triton.language.extra.deeplink.cann.extension as cann_ext
except ImportError:
    try:
        import triton.language.extra.cann.extension as cann_ext
    except ImportError:
        cann_ext = None

if cann_ext is not None:
    extract_slice = cann_ext.extract_slice
    insert_slice = cann_ext.insert_slice
    get_element = cann_ext.get_element
else:
    extract_slice = getattr(tl, "extract_slice", None)
    insert_slice = getattr(tl, "insert_slice", None)
    get_element = getattr(tl, "get_element", None)


class _TLDeviceFallback:
    fast_dividef = staticmethod(lambda x, y: x / y)
    fast_expf = staticmethod(tl.exp)
    fast_logf = staticmethod(tl.log)
    fast_log2f = staticmethod(tl.log2)


tldevice = _TLDeviceFallback()


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    if "pytest" in sys.modules:
        return parser.parse_args([])
    return parser.parse_args()


def case_values(cases):
    cases = tuple(cases)
    limit = int(os.getenv("DLC_AUTOTUNE_CASE_LIMIT", "0"))
    if limit <= 0:
        return cases
    return cases[:limit]


def dtype_from_name(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def init(seed=0):
    torch.manual_seed(seed)


def get_vectorcore_num(device=None):
    if device is None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.npu.current_device()
        elif torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = 0
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    return props.get("num_vectorcore", 1)


def sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup=10, repeat=50):
    out = fn()
    sync()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=repeat)
    return ms, out


def assert_close(name, actual, expected, rtol=None, atol=None):
    if rtol is None:
        rtol = 1e-2 if actual.dtype in (torch.float16, torch.bfloat16) else 1e-4
    if atol is None:
        atol = 1e-2 if actual.dtype in (torch.float16, torch.bfloat16) else 1e-4
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu()
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def report(name, torch_ms, triton_ms, models):
    speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
    print(f"op: {name}")
    print(f"torch_ms: {torch_ms:.6f}")
    print(f"triton_ms: {triton_ms:.6f}")
    print(f"speedup: {speedup:.4f}x")
    print(f"models: {models}")
