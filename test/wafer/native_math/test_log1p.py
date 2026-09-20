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

"""Ascend log1p algorithm and original parameter matrix, using native Wafer tensors."""
import pytest
import torch
import torch_txda  # noqa: F401 -- registers the TXDA device
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def triton_log1p(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tmp0 + libdevice.log1p(tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize("param_list", [["float32", (2, 4096, 8), 2, 32768, 1024]])
def test_log1p(param_list):
    sigtype, shape, ncore, xblock, xblock_sub = param_list
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=getattr(torch, sigtype))
    b = torch.randn_like(a)
    out = (torch.zeros_like(a)).to("txda")
    triton_log1p[(ncore,)]((a).to("txda"), (b).to("txda"), out, a.numel(), xblock, xblock_sub)
    torch.testing.assert_close(out.cpu(), a + torch.log1p(b), rtol=1e-4, atol=1e-4, equal_nan=True)


@triton.jit
def log1p_kernel(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, libdevice.log1p(tl.load(X + i)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log1p_special_values(dtype):
    host = torch.tensor([-2, -1, -1 + torch.finfo(dtype).eps, -1e-7, -1e-8, -0.0, 0.0, 1e-8,
                         1e-7, 1e-4, 1, 2, 10, float("inf"), -float("inf"), float("nan")], dtype=dtype)
    out = (torch.zeros_like(host)).to("txda")
    log1p_kernel[(1,)]((host).to("txda"), out, host.numel())
    # No absolute tolerance: log(1+x) incorrectly rounds tiny x to zero.
    expected, actual = torch.log1p(host), out.cpu()
    tolerance = 1e-6 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.signbit(actual[5:7]), torch.signbit(expected[5:7]))
