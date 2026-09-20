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

"""Ascend relu algorithm and original parameter matrix, using native Wafer tensors."""
import pytest
import torch
import torch_txda  # noqa: F401 -- registers the TXDA device
import triton
import triton.language as tl


@triton.jit
def wafer_relu(x):
    # Ordered comparison keeps NaN and -0, matching torch.relu. The current
    # Wafer maximum instruction discards NaN, so use its compare/select path.
    return tl.where(x < 0, 0, x)

@triton.jit
def triton_relu(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tmp0 + wafer_relu(tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize("param_list", [
    ["float32", (2, 4096, 8), 2, 32768, 512],
    ["float16", (2, 4096, 8), 2, 32768, 512],
])
def test_relu(param_list):
    sigtype, shape, ncore, xblock, xblock_sub = param_list
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=getattr(torch, sigtype))
    b = torch.randn_like(a)
    out = (torch.zeros_like(a)).to("txda")
    triton_relu[(ncore,)]((a).to("txda"), (b).to("txda"), out, a.numel(), xblock, xblock_sub)
    tolerance = 1e-3 if sigtype == "float16" else 1e-4
    torch.testing.assert_close(out.cpu(), a + torch.relu(b), rtol=tolerance, atol=tolerance, equal_nan=True)


@triton.jit
def relu_kernel(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(X + i)
    tl.store(Y + i, wafer_relu(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu_special_values(dtype):
    host = torch.tensor([-float("inf"), -1, -0.0, 0.0, 1e-5, 1, float("inf"), float("nan")], dtype=dtype)
    out = (torch.zeros_like(host)).to("txda")
    relu_kernel[(1,)]((host).to("txda"), out, host.numel())
    expected, actual = torch.relu(host), out.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.signbit(actual[:-1]), torch.signbit(expected[:-1]))
