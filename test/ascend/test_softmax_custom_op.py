"""
Triton softmax kernel using dl.custom() with bitcode auto-resolution.

Computes: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

The three vector ops (vsub, vexp, vdiv) are dl.custom() calls that link
against softmax_ops.aiv.bc.  The bitcode path is auto-resolved from the
name "softmax_ops" — no absolute path needed.
"""
import os

# Ensure bishengir tools are in PATH before triton imports read BISHENG_INSTALL_PATH.
_BISHENG_INSTALL = "/mnt/data01/zmz/workspace/04ttshared/fordlc/ascendnpu-ir-0514/build/install/bin/"
if os.path.isdir(_BISHENG_INSTALL):
    os.environ.setdefault("BISHENG_INSTALL_PATH", _BISHENG_INSTALL)
    if _BISHENG_INSTALL not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _BISHENG_INSTALL + os.pathsep + os.environ.get("PATH", "")

import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl

# ======================================================================
# DSL custom op registration — bitcode auto-resolved by name
# ======================================================================


@dl.register_custom_op
class vsub_fp32:
    core = dl.CORE.VECTOR
    pipe = dl.PIPE.PIPE_V
    mode = dl.MODE.SIMD

    def __init__(self, a, b, out=None):
        assert out is not None, "dl.custom() requires out= parameter"
        self.symbol = "custom_vsub_fp32"
        self.bitcode = "softmax_ops"   # auto-resolved to softmax_ops.aiv.bc


@dl.register_custom_op
class vexp_fp32:
    core = dl.CORE.VECTOR
    pipe = dl.PIPE.PIPE_V
    mode = dl.MODE.SIMD

    def __init__(self, a, out=None):
        assert out is not None, "dl.custom() requires out= parameter"
        self.symbol = "custom_vexp_fp32"
        self.bitcode = "softmax_ops"


@dl.register_custom_op
class vdiv_fp32:
    core = dl.CORE.VECTOR
    pipe = dl.PIPE.PIPE_V
    mode = dl.MODE.SIMD

    def __init__(self, a, b, out=None):
        assert out is not None, "dl.custom() requires out= parameter"
        self.symbol = "custom_vdiv_fp32"
        self.bitcode = "softmax_ops"


# ======================================================================
# Triton kernel
# ======================================================================

CHUNK_SIZE = 1024          # max vector length for fp32 DSL ops
MIN_CHUNK_SIZE = 8         # min vector length (SIMD width >= 2)


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    N,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Row-wise fused softmax using dl.custom() on Ascend NPU.

    One program = one row.  Processing in chunks of CHUNK_SIZE.
    Reductions (tl.max, tl.sum) use native Triton.
    """
    pid = tl.program_id(axis=0)
    row_start = pid * row_stride

    # --- Pass 1: row-wise maximum ---
    row_max = float("-inf")
    for start in tl.static_range(0, BLOCK_SIZE, CHUNK_SIZE):
        offsets = row_start + start + tl.arange(0, CHUNK_SIZE)
        mask = (start + tl.arange(0, CHUNK_SIZE)) < N
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        row_max = tl.maximum(row_max, tl.max(x_chunk))

    # --- Pass 2: denominator sum(exp(x - max)) ---
    denom = 0.0
    for start in tl.static_range(0, BLOCK_SIZE, CHUNK_SIZE):
        offsets = row_start + start + tl.arange(0, CHUNK_SIZE)
        mask = (start + tl.arange(0, CHUNK_SIZE)) < N
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x_chunk.to(tl.float32)

        zero = tl.full([CHUNK_SIZE], 0, tl.float32)
        row_max_vec = zero + row_max

        # vsub: shifted = x - max
        buf_sub = tl.full([CHUNK_SIZE], 0, tl.float32)
        shifted = dl.custom("vsub_fp32", x_f32, row_max_vec, out=buf_sub)

        # vexp: exp_vals = exp(shifted)
        buf_exp = tl.full([CHUNK_SIZE], 0, tl.float32)
        exp_vals = dl.custom("vexp_fp32", shifted, out=buf_exp)

        denom += tl.sum(tl.where(mask, exp_vals, 0.0))

    # --- Pass 3: normalize and store ---
    for start in tl.static_range(0, BLOCK_SIZE, CHUNK_SIZE):
        offsets = row_start + start + tl.arange(0, CHUNK_SIZE)
        mask = (start + tl.arange(0, CHUNK_SIZE)) < N
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x_chunk.to(tl.float32)

        zero = tl.full([CHUNK_SIZE], 0, tl.float32)
        row_max_vec = zero + row_max
        denom_vec = zero + denom

        # vsub: shifted = x - max
        buf_sub = tl.full([CHUNK_SIZE], 0, tl.float32)
        shifted = dl.custom("vsub_fp32", x_f32, row_max_vec, out=buf_sub)

        # vexp: exp_vals = exp(shifted)
        buf_exp = tl.full([CHUNK_SIZE], 0, tl.float32)
        exp_vals = dl.custom("vexp_fp32", shifted, out=buf_exp)

        # vdiv: result = exp_vals / denom
        buf_div = tl.full([CHUNK_SIZE], 0, tl.float32)
        result = dl.custom("vdiv_fp32", exp_vals, denom_vec, out=buf_div)

        tl.store(output_ptr + offsets, result.to(x_chunk.dtype), mask=mask)


# ======================================================================
# Public wrapper
# ======================================================================

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax using dl.custom() Triton kernel.

    Args:
        x: Input tensor (fp16, fp32, or bf16).
        dim: Dimension for softmax (default: -1).

    Returns:
        Output tensor with same shape/dtype as x.
    """
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim

    # Handle non-last-dim by transposition
    permuted = False
    inv_perm = list(range(ndim))
    if dim != ndim - 1:
        perm = list(range(ndim))
        perm[-1], perm[dim] = perm[dim], perm[-1]
        x = x.permute(perm).contiguous()
        dim = ndim - 1
        permuted = True
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
    else:
        x = x.contiguous()

    N = x.shape[-1]
    M = x.numel() // N

    BLOCK_SIZE = triton.next_power_of_2(N)
    CS = max(MIN_CHUNK_SIZE, min(CHUNK_SIZE, BLOCK_SIZE))

    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (M,)
    softmax_kernel[grid](
        x, output, N, N,
        BLOCK_SIZE=BLOCK_SIZE,
        CHUNK_SIZE=CS,
    )

    if permuted:
        output = output.permute(inv_perm)

    return output


# ======================================================================
# Test
# ======================================================================

if __name__ == "__main__":
    print("=== Softmax Triton Kernel Test (dl.custom) ===")

    for shape, dim in [((4, 128), -1), ((4, 1024), -1), ((2, 4096), -1), ((16, 256), 0)]:
        x = torch.randn(shape, dtype=torch.float32, device="npu")
        y = softmax(x, dim=dim)
        ref = torch.nn.functional.softmax(x.float(), dim=dim)

        max_diff = (y - ref).abs().max().item()
        passed = torch.allclose(y, ref, atol=1e-5)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] shape={shape}, dim={dim}, max_diff={max_diff:.2e}")

    print("Done.")
