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

"""Ascend atan/scalar coverage and FlagTree isnan coverage on native Wafer APIs.

Sources: test/ascend/passed_tests/test_{atan,isnan,scalar_calc}.py at b1991f1;
FlagTree third_party/tsingmicro/examples/test_libdevice.py at 22f4ff0.
The original license notice is retained above. Shapes, dtypes and comparison
tolerances are preserved; only device transport, references and OP entry change.
"""
import pytest
import torch
import torch_txda  # noqa: F401 -- registers the TXDA device
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def unary_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, 0)
    result = getattr(libdevice, OP)(x)
    tl.store(Y + offsets, result, offsets < N)


@triton.jit
def scalar_tanh_kernel(X, Y):
    tl.store(Y, libdevice.tanh(tl.load(X)))


@pytest.mark.parametrize("dtype,sigtype", [(torch.float32, "float32"), (torch.float16, "float16")])
@pytest.mark.parametrize("N,NUMEL", [(3, 32), (-32, 32), (37, 64), (-256, 256), (781, 1024)])
def test_elementwsie_common(dtype, sigtype, N, NUMEL):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N
    torch.manual_seed(0)
    host = torch.randn((N,), dtype=dtype)
    x, out = (host).to("txda"), (torch.zeros_like(host)).to("txda")
    unary_kernel[(1,)](x, out, N, NUMEL, "atan", debug=True)
    tolerance = 1e-3 if sigtype == "float16" else 1e-4
    torch.testing.assert_close(out.cpu(), torch.atan(host), rtol=tolerance, atol=tolerance, equal_nan=True)


@pytest.mark.parametrize("sigtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("N", [256])
def test_isnan(sigtype, N):
    # Extend FlagTree's FP32/4/128 cases with Ascend's three dtype/256 cases.
    torch.manual_seed(0)
    host = torch.randn((N,), dtype=getattr(torch, sigtype))
    host[1] = float("nan")
    host[N // 4] = float("inf")
    host[N // 2] = -float("inf")
    out = (torch.zeros(N, dtype=torch.bool)).to("txda")
    unary_kernel[(1,)]((host).to("txda"), out, N, N, "isnan")
    torch.testing.assert_close(out.cpu(), torch.isnan(host), rtol=0, atol=0)
    assert out.cpu()[1].item() is True


@pytest.mark.parametrize("param_list", [["float32", 16]])
def test_scalar_tanh_calc(param_list):
    sigtype, N = param_list
    torch.manual_seed(0)
    host = torch.randn(N, dtype=getattr(torch, sigtype))
    out = (torch.zeros(1, dtype=host.dtype)).to("txda")
    scalar_tanh_kernel[(1,)]((host).to("txda"), out)
    torch.testing.assert_close(out.cpu()[0], torch.tanh(host[0]), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", ["atan", "tanh", "isnan"])
def test_unary_special_values(dtype, op):
    host = torch.tensor([-float("inf"), -10, -1, -0.0, 0.0, 1e-5, 1, 10,
                         float("inf"), float("nan")], dtype=dtype)
    expected = getattr(torch, op)(host)
    out = (torch.zeros_like(expected)).to("txda")
    unary_kernel[(1,)]((host).to("txda"), out, host.numel(), 16, op)
    actual = out.cpu()
    tolerance = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-4}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
    if op != "isnan":
        torch.testing.assert_close(torch.signbit(actual[3:5]), torch.signbit(expected[3:5]))
