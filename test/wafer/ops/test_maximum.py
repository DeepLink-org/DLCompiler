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

import triton
import triton.language as tl
import torch
import torch_txda  # noqa: F401
import pytest
import test_common


def torch_maximum(x0, x1):
    res = torch.maximum(x0, x1)
    return res


@triton.jit
def triton_maximum(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tl.maximum(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize(
    "param_list",
    [
        ["float32", (2, 4096, 8), 2, 32768, 1024],
        ["float16", (2, 4096, 8), 2, 32768, 1024],
        ["int8", (2, 4096, 8), 2, 32768, 1024],
    ],
)
def test_maximum(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).cpu()
    x1 = test_common.generate_tensor(shape, dtype).cpu()
    # torch结果
    torch_res = torch_maximum(x0, x1)
    # triton结果
    triton_res = test_common.generate_tensor(shape, dtype).cpu()
    x0_txda = x0.to("txda")
    x1_txda = x1.to("txda")
    triton_res_txda = triton_res.to("txda")
    triton_maximum[ncore, 1, 1](x0_txda, x1_txda, triton_res_txda, x0_txda.numel(), xblock, xblock_sub)
    with torch.no_grad():
        triton_res.copy_(triton_res_txda.cpu())
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
