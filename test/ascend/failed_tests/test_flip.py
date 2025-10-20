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
import torch_npu
import pytest
import test_common
# from triton.runtime.libentry import libentry


@pytest.mark.parametrize(
    "para_type,data_type,shape",
    [
        ["float32", torch.float32, (4, 16, 64)],
        ["float16", torch.float16, (4, 16, 64)],
        ["int8", torch.int8, (4, 16, 64)],
    ],
)
def test_flip(para_type, data_type, shape):

    def torch_func(x):
        return torch.flip(x, dims=(2,))

    # @libentry()
    @triton.jit
    def triton_kernel(
        output_ptr0, in_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr
    ):
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = (
            xidx[:, None, None] * YB * ZB
            + yidx[None, :, None] * ZB
            + zidx[None, None, :]
        )
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.flip(tmp0, 2)
        tl.store(output_ptr0 + idx, tmp1)

    def triton_func(x):
        XB, YB, ZB = shape
        y = torch.empty_like(x)
        triton_kernel[1, 1, 1](y, x, XB, YB, ZB)
        return y

    x = torch.randint(low=-128, high=128, size=shape, dtype=data_type).npu()
    torch_ref = torch_func(x)
    triton_cal = triton_func(x)
    test_common.validate_cmp(para_type, torch_ref, triton_cal)
