# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch, torch_npu
import triton
import triton.language as tl
import triton.language.math as tl_math
import pytest
import test_common
import random


def test_ldst_indirect_00():

    @triton.jit
    def triton_ldst_indirect_00_kernel(
        out_ptr0, in_ptr0, in_ptr1, OFFSET0: tl.constexpr, XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        offset1 = tl.load(in_ptr0 + OFFSET0)
        idx_in1 = offset1 + pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr1 + idx_in1)
        tmp1 = tl.exp(tmp0)
        idx_out0 = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + idx_out0, tmp1)

    def triton_ldst_indirect_00_func(x0, x1, s, xs):
        n = x1.numel()
        ns = n - s
        assert ns == xs, "test only single core"
        y0 = torch.empty((ns,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_00_kernel[ns // xs, 1, 1](y0, x0, x1, OFFSET0=s, XS=xs)
        return y0

    def torch_ldst_indirect_00_func(x0, x1, s):
        offset = x0[s]
        return torch.exp(x1[offset:])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 0
    N0, N1 = 16, 16
    blocksize = 16
    assert N0 > offset, "offset must be < N0"
    N1 = N1 + offset
    x0 = torch.arange(0, N0, dtype=torch.int32, device=DEV)
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_00_func(x0, x1, offset)
    triton_cal = triton_ldst_indirect_00_func(x0, x1, offset, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_01():

    @triton.jit
    def triton_ldst_indirect_01_kernel(
        out_ptr0, in_ptr0, in_ptr1, OFFSET0: tl.constexpr, XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        offset1 = tl.load(in_ptr0 + OFFSET0)
        idx_in1 = offset1 + pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr1 + idx_in1)
        tmp1 = tl_math.exp(tmp0)
        idx_out0 = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + idx_out0, tmp1)

    def triton_ldst_indirect_01_func(x0, x1, s, xs):
        n = x1.numel()
        ns = n - s
        assert ns == xs, "test only single core"
        y0 = torch.empty((ns,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_01_kernel[ns // xs, 1, 1](y0, x0, x1, OFFSET0=s, XS=xs)
        return y0

    def torch_ldst_indirect_01_func(x0, x1, s):
        offset = x0[s]
        return torch.exp(x1[offset:])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 0
    N0, N1 = 16, 16
    blocksize = 16
    assert N0 > offset, "offset must be < N0"
    N1 = N1 + offset
    x0 = torch.arange(0, N0, device=DEV)  # int64
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_01_func(x0, x1, offset)
    triton_cal = triton_ldst_indirect_01_func(x0, x1, offset, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_02():

    @triton.jit
    def triton_ldst_indirect_02_kernel(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
        pid = tl.program_id(0)
        for i in tl.range(0, XS):
            tmp0 = tl.load(in_ptr0 + i)
            tmp1 = tl.load(in_ptr1 + tmp0)
            tmp2 = tl_math.exp(tmp1)
            tl.store(out_ptr0 + i, tmp2)

    def triton_ldst_indirect_02_func(x0, x1, xs):
        n0 = x0.numel()
        assert n0 == xs, "test only single core"
        y0 = torch.empty((n0,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_02_kernel[n0 // xs, 1, 1](y0, x0, x1, XS=xs)
        return y0

    def torch_ldst_indirect_02_func(x0, x1):
        return torch.exp(x1[x0])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 16
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == blocksize, "N0 must be == blocksize"
    x0 = offset + torch.arange(0, N0, device=DEV)  # int64
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_02_func(x0, x1)
    triton_cal = triton_ldst_indirect_02_func(x0, x1, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_03():

    @triton.jit
    def triton_ldst_indirect_03_kernel(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp1 = tl.load(in_ptr1 + tmp0)
        tmp2 = tl_math.exp(tmp1)
        out0_idx = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_03_func(x0, x1, xs):
        n0 = x0.numel()
        assert n0 == xs, "test only single core"
        y0 = torch.empty((n0,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_03_kernel[n0 // xs, 1, 1](y0, x0, x1, XS=xs)
        return y0

    def torch_ldst_indirect_03_func(x0, x1):
        return torch.exp(x1[x0])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 16
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == blocksize, "N0 must be == blocksize"
    x0 = offset + torch.arange(0, N0, device=DEV)  # int64
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_03_func(x0, x1)
    triton_cal = triton_ldst_indirect_03_func(x0, x1, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_04():

    @triton.jit
    def triton_ldst_indirect_04_kernel(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp0min = tl.min(tmp0, axis=0)
        tmp0max = tl.max(tmp0, axis=0)
        tmp0 = tmp0 * 2.0
        tmp0 = tl.clamp(tmp0, tmp0min, tmp0max)
        tmp0 = tmp0.to(tl.int32)
        tmp1 = tl.load(in_ptr1 + tmp0)
        tmp2 = tl_math.exp(tmp1)
        out0_idx = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_04_func(x0, x1, xs):
        n0 = x0.numel()
        assert n0 == xs, "test only single core"
        y0 = torch.empty((n0,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_04_kernel[n0 // xs, 1, 1](y0, x0, x1, XS=xs)
        return y0

    def torch_ldst_indirect_04_func(x0, x1):
        x0min = torch.min(x0)
        x0max = torch.max(x0)
        idx = torch.clamp(x0 * 2, x0min, x0max)
        return torch.exp(x1[idx.to(torch.int32)])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 16
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == blocksize, "N0 must be == blocksize"
    x0 = offset + torch.arange(0, N0, dtype=torch.float32, device=DEV)
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_04_func(x0, x1)
    triton_cal = triton_ldst_indirect_04_func(x0, x1, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_05():

    @triton.jit
    def triton_ldst_indirect_05_kernel(
        out_ptr0, in_ptr1, in_ptr2, stride_in_r, XS: tl.constexpr, RS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = tl.arange(0, XS)
        tmp1 = tl.load(in_ptr1 + in_idx1)
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        tmp2 = tl_math.exp(tmp2)
        out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_05_func(xc, x2, xs, rs):
        nr = x2.size()[0]
        nc = xc.numel()
        stride_in_r = x2.stride()[0]
        assert nr == xs, "test only single core"
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_05_kernel[nr // xs, 1, 1](
            y0, xc, x2, stride_in_r, XS=xs, RS=rs
        )
        return y0

    def torch_ldst_indirect_05_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return torch.exp(extracted)

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 8
    lowdimsize = N0
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_05_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_05_func(xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_06():

    @triton.jit
    def triton_ldst_indirect_06_kernel(
        out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr2,
        stride_in_r,
        XS: tl.constexpr,
        RS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp1 = tl.load(in_ptr1 + in_idx1)
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        tmp2 = tl_math.exp(tmp2)
        out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_06_func(xr, xc, x2, xs, rs):
        nr = x2.size()[0]
        nc = xc.numel()
        stride_in_r = x2.stride()[0]
        assert nr == xs, "test only single core"
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_06_kernel[nr // xs, 1, 1](
            y0, xr, xc, x2, stride_in_r, XS=xs, RS=rs
        )
        return y0

    def torch_ldst_indirect_06_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return torch.exp(extracted)

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 4
    lowdimsize = N0
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_06_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_06_func(xr, xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_07():

    @triton.jit
    def triton_ldst_indirect_07_kernel(
        out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr2,
        stride_in_r,
        XS: tl.constexpr,
        RS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp1 = tl.load(in_ptr1 + in_idx1)
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_07_func(xr, xc, x2, xs, rs):
        nr = x2.size()[0]
        nc = xc.numel()
        stride_in_r = x2.stride()[0]
        assert nr == xs, "test only single core"
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_07_kernel[nr // xs, 1, 1](
            y0, xr, xc, x2, stride_in_r, XS=xs, RS=rs
        )
        return y0

    def torch_ldst_indirect_07_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return extracted

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 4
    lowdimsize = N0
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_07_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_07_func(xr, xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_08():

    @triton.jit
    def triton_ldst_indirect_08_kernel(
        out_ptr0,
        in_ptr_xc,
        in_ptr_x2,
        stride_in_r,
        OUT_COLS: tl.constexpr,
        XS: tl.constexpr,
        RS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_idx_full = pid * XS + tl.arange(0, XS)
        col_pos = tl.arange(0, RS)
        xc_vals = tl.load(in_ptr_xc + col_pos)
        row_arange = tl.arange(0, XS)
        gather_flat = row_arange[:, None] * stride_in_r + xc_vals[None, :]
        vals = tl.load(in_ptr_x2 + gather_flat)
        vals = tl_math.exp(vals)
        out_flat = row_idx_full[:, None] * OUT_COLS + xc_vals[None, :]
        tl.store(out_ptr0 + out_flat, vals)

    def triton_ldst_indirect_08_func(xc, x2, xs, rs):
        nr = x2.size(0)
        out_cols = x2.size(1)
        stride_in_r = x2.stride(0)
        assert nr == xs, "test only single core"
        y0 = torch.zeros((nr, out_cols), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_08_kernel[nr // xs, 1, 1](
            y0, xc, x2, stride_in_r, OUT_COLS=out_cols, XS=xs, RS=rs
        )
        return y0

    def torch_ldst_indirect_08_func(xr, xc, x2):
        out = torch.zeros((xr.numel(), x2.size(1)), dtype=x2.dtype, device=x2.device)
        gathered = torch.exp(x2[xr[:, None], xc[None, :]])
        out.scatter_(1, xc.expand(xr.numel(), -1), gathered)
        return out

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 8
    lowdimsize = N0
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_08_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_08_func(xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_09():

    @triton.jit
    def triton_ldst_indirect_09_kernel(
        out_ptr0,
        in_ptr1,
        in_ptr2,
        stride_in_r,
        offset: tl.constexpr,
        XS: tl.constexpr,
        RS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        in_idx0 = tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = pid * XS + tl.load(in_ptr1 + in_idx0)
        tmp1 = tl.arange(0, RS) + offset
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        tmp2 = tl_math.exp(tmp2)
        out0_idx = pid * XS * RS + in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_09_func(xr, x2, offset, xs, rs):
        nr = xr.numel()
        nc = rs
        stride_in_r = x2.stride()[0]
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_09_kernel[nr // xs, 1, 1](
            y0, xr, x2, stride_in_r, offset=offset, XS=xs, RS=rs
        )
        return y0

    def torch_ldst_indirect_09_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return torch.exp(extracted)

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 8
    lowdimsize = N0
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_09_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_09_func(xr, x2, offset, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


@triton.jit
def unstructured_mask_2d_kernel(
    in_ptr, out_ptr, mask_m_ptr, mask_n_ptr, m, n, M: tl.constexpr, N: tl.constexpr
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    mask_m = tl.load(mask_m_ptr + offs_m, mask=offs_m < m, other=0) != 0
    mask_n = tl.load(mask_n_ptr + offs_n, mask=offs_n < n, other=0) != 0

    in_ptrs = in_ptr + offs_m[:, None] * N + offs_n[None, :]
    # dim 0 with unstructured mask.
    v = tl.load(in_ptrs, mask=mask_m[:, None] and offs_n[None, :] < n, other=-2)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    # dim 1 with unstructured mask.
    tl.store(out_ptrs, v, mask=offs_m[:, None] < m and mask_n[None, :])


# helper to get torch dtype from string
def torch_dtype(dtype_str):
    return eval(f"torch.{dtype_str}")


@pytest.mark.parametrize(
    "param_list",
    [
        ["float32", (8, 16)],
    ],
)
def test_unstructured_mask_2d(param_list):
    dtype_str, shape = param_list
    dtype = torch_dtype(dtype_str)
    M, N = shape

    # make deterministic
    random.seed(0)
    torch.manual_seed(0)

    # input: use distinct values per element for easy checking
    # use arange and cast to dtype
    total = M * N
    if dtype.is_floating_point:
        in_tensor = (
            torch.arange(total, dtype=torch.float32).reshape(M, N).to(dtype).npu()
        )
    else:
        in_tensor = torch.arange(total, dtype=torch.int64).reshape(M, N).to(dtype).npu()

    # masks: random 0/1 tensors (1D)
    mask_m = torch.randint(0, 2, (M,), dtype=torch.int32).npu()  # rows
    mask_n = torch.randint(0, 2, (N,), dtype=torch.int32).npu()  # cols

    # out: initialize with a sentinel so we can tell which positions are untouched
    if dtype.is_floating_point:
        sentinel = torch.tensor(-999.0, dtype=torch.float32).to(dtype)
    else:
        sentinel = torch.tensor(-999, dtype=torch.int64).to(dtype)

    out_init = torch.full((M, N), sentinel.item(), dtype=dtype).npu()
    out = out_init.clone()

    # call kernel: single program covering full matrix; M,N passed as constexpr
    # signature: (in_ptr, out_ptr, mask_m_ptr, mask_n_ptr, m, n, M:tl.constexpr, N:tl.constexpr)
    # set m=M, n=N to simplify masks (see analysis)
    unstructured_mask_2d_kernel[1, 1](in_tensor, out, mask_m, mask_n, M, N, M=M, N=N)

    # construct reference output according to kernel logic described in analysis:
    # when mask_n[j] == 0 -> out should remain the initial sentinel (kernel does not store)
    # when mask_n[j] == 1:
    #    if mask_m[i] == 1 -> out[i,j] == in[i,j]
    #    else -> out[i,j] == -2
    expected = out_init.clone()
    for i in range(M):
        row_mask = bool(mask_m[i].item())
        for j in range(N):
            col_mask = bool(mask_n[j].item())
            if not col_mask:
                # kernel does not store here; keep initial sentinel
                expected[i, j] = out_init[i, j]
            else:
                if row_mask:
                    expected[i, j] = in_tensor[i, j]
                else:
                    # -2 with the same dtype
                    if dtype.is_floating_point:
                        expected[i, j] = torch.tensor(-2.0, dtype=torch.float32).to(
                            dtype
                        )
                    else:
                        expected[i, j] = torch.tensor(-2, dtype=expected.dtype)

    # validate using project's common validator
    test_common.validate_cmp(dtype_str, out, expected)


if __name__ == "__main__":
    test_ldst_indirect_08()
    print("success: test_ldst_indirect_05")
