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

import torch
import torch_txda  # noqa: F401
import numpy as np
import triton
import triton.language as tl
from numpy.random import RandomState


# inp the two 32 bit signed integers.
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)


# accuracy reference
def umulhi32(a, b):
    a_64 = a.astype(np.int64)
    b_64 = b.astype(np.int64)
    product_64 = a_64 * b_64
    # get the high part
    result_high_32 = product_64 >> 32
    return result_high_32.astype(np.int32)


def test_umulhi():
    N = 128
    x = torch.randint(low=0, high=2000, size=(N,), dtype=torch.int32)
    y = torch.randint(low=0, high=2000, size=(N,), dtype=torch.int32)
    xx = x.cpu()
    yy = y.cpu()
    z_tri = torch.zeros(size=(N,), dtype=torch.int32).cpu()
    xx_txda = xx.to("txda")
    yy_txda = yy.to("txda")
    z_tri_txda = z_tri.to("txda")
    umulhi_kernel[(1,)](xx_txda, yy_txda, z_tri_txda, N=N)
    with torch.no_grad():
        z_tri.copy_(z_tri_txda.cpu())

    xxx = x.numpy()
    yyy = y.numpy()
    z_ref = umulhi32(xxx, yyy)
    z_ref1 = torch.from_numpy(z_ref).cpu()
    assert torch.equal(z_tri, z_ref1)
