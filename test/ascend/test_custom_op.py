"""
Triton add kernel using dl.custom() with bitcode auto-resolution.

Verifies the custom op `add` produces correct results on Ascend NPU.
"""

import os

# Ensure bishengir tools are in PATH before triton imports read BISHENG_INSTALL_PATH.
_BISHENG_INSTALL = (
    "/mnt/data01/zmz/workspace/04ttshared/fordlc/ascendnpu-ir-0514/build/install/bin/"
)
if os.path.isdir(_BISHENG_INSTALL):
    os.environ.setdefault("BISHENG_INSTALL_PATH", _BISHENG_INSTALL)
    if _BISHENG_INSTALL not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _BISHENG_INSTALL + os.pathsep + os.environ.get("PATH", "")

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.deeplink.cann.extension as dl


# ======================================================================
# DSL custom op registration — bitcode auto-resolved by name
# ======================================================================


@dl.register_custom_op
class add:
    core = dl.CORE.VECTOR
    pipe = dl.PIPE.PIPE_V
    mode = dl.MODE.SIMD

    def __init__(self, a, b, out=None):
        assert out is not None, "dl.custom() requires out= parameter"
        self.symbol = "custom_add_" + str(a.dtype)
        self.bitcode = "add"  # auto-resolved to add.aiv.bc


# ======================================================================
# Triton kernel
# ======================================================================


@triton.jit
def custom_add_kernel(output_ptr, a_ptr, b_ptr, L: tl.constexpr):
    idx = tl.arange(0, L)
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    buf = tl.full([L], 0, a.dtype)
    res = dl.custom("add", a, b, out=buf)
    tl.store(output_ptr + idx, res)


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.parametrize("L", [32, 128, 1024])
def test_custom_add_int32(L):
    a = torch.randint(0, 1000, (L,), dtype=torch.int32).npu()
    b = torch.randint(0, 1000, (L,), dtype=torch.int32).npu()
    out = torch.empty(L, dtype=torch.int32).npu()

    custom_add_kernel[1, 1, 1](out, a, b, L=L)

    ref = a.cpu() + b.cpu()
    assert torch.equal(out.cpu(), ref), f"L={L}: out={out.cpu()}, ref={ref}"


if __name__ == "__main__":
    for L in [32, 128, 1024]:
        test_custom_add_int32(L)
        print(f"[PASS] L={L}")
    print("Done.")
