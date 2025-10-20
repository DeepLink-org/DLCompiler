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

import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


def torch_not(x0):
    res = torch.bitwise_not(x0)
    return res


@triton.jit
def triton_not(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp2 = not(tmp0)
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['int16', (2, 4096, 8), 2, 32768, 1024],
                             ['int32', (2, 4096, 8), 2, 32768, 1024],
                             ['int64', (2, 4096, 8), 2, 32768, 1024],
                             ['bool', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_not(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch_not(x0)
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_not[ncore, 1, 1](x0, triton_res, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
