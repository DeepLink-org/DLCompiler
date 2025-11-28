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


@triton.jit
def fn_npu_(
    output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr
):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)

    idx = xidx[:, None] * YB + yidx[None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.join(X, Y)

    oidx = (
        xidx[:, None, None] * YB * 2
        + yidx[None, :, None] * 2
        + tl.arange(0, 2)[None, None, :]
    )

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize(
    "para_type,data_type,XB,YB,ZB",
    [
        ["float32", torch.float32, 4, 64, 4],
        ["float32", torch.float32, 8, 8, 4],
        ["float16", torch.float16, 4, 64, 4],
        ["float16", torch.float16, 8, 8, 4],
        ["int8", torch.int8, 4, 128, 4],
        ["int8", torch.int8, 8, 8, 4],
    ],
)
def test_join(para_type, data_type, XB, YB, ZB):
    x = torch.full((XB, YB), 100, dtype=data_type).npu()
    y = torch.full((XB, YB), 30, dtype=data_type).npu()

    ans = torch.stack((x, y), dim=-1)
    print(ans)

    output = torch.randint(1, (XB, YB, 2), dtype=data_type).npu()
    fn_npu_[1, 1, 1](output, x, y, XB, YB, ZB, debug=True)

    print(output)
    test_common.validate_cmp(para_type, ans, output)


@triton.jit
def fn_npu_concat_axis_(
    output_ptr,
    x_ptr,
    y_ptr,
    XB: tl.constexpr,
    YB: tl.constexpr,
    ZB: tl.constexpr,
    axis: tl.constexpr,  # 0 / 1 / 2
):
    """
    新增 kernel：将两个 shape=(XB, YB, ZB) 的张量沿 axis 拼接到 output。
    只新增，不修改原有 fn_npu_。
    说明：这个 kernel 假定输入张量是连续的、按行主序扁平化。
    """

    # 构造三维索引 (i,j,k) -> linear index i*(YB*ZB) + j*ZB + k
    xidx = tl.arange(0, XB)  # (XB,)
    yidx = tl.arange(0, YB)  # (YB,)
    zidx = tl.arange(0, ZB)  # (ZB,)

    # idx shape: (XB, YB, ZB)
    idx = (
        xidx[:, None, None] * (YB * ZB) + yidx[None, :, None] * ZB + zidx[None, None, :]
    )

    # 从输入加载完整块
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    # 根据 axis 计算输出偏移并存储
    if axis == 0:
        # out shape: (XB*2, YB, ZB)
        oidx_x = idx  # X 放在前半段
        oidx_y = (
            (xidx + XB)[:, None, None] * (YB * ZB)
            + yidx[None, :, None] * ZB
            + zidx[None, None, :]
        )
        tl.store(output_ptr + oidx_x, X)
        tl.store(output_ptr + oidx_y, Y)

    elif axis == 1:
        # out shape: (XB, YB*2, ZB)
        # 线性化为 i*(YB*2*ZB) + j*(ZB) + k
        base_x = xidx[:, None, None] * (YB * 2 * ZB)
        oidx_x = base_x + yidx[None, :, None] * ZB + zidx[None, None, :]
        oidx_y = base_x + (yidx + YB)[None, :, None] * ZB + zidx[None, None, :]
        tl.store(output_ptr + oidx_x, X)
        tl.store(output_ptr + oidx_y, Y)

    elif axis == 2:
        # out shape: (XB, YB, ZB*2)
        # 线性化为 i*(YB*(ZB*2)) + j*(ZB*2) + k
        base_x = xidx[:, None, None] * (YB * (ZB * 2)) + yidx[None, :, None] * (ZB * 2)
        oidx_x = base_x + zidx[None, None, :]
        oidx_y = base_x + (zidx + ZB)[None, None, :]
        tl.store(output_ptr + oidx_x, X)
        tl.store(output_ptr + oidx_y, Y)

    else:
        # 不支持其它 axis，早退出
        return


@triton.jit
def fn_npu_concat_axis_tiled_(
    output_ptr,
    x_ptr,
    y_ptr,
    XB: tl.constexpr,
    YB: tl.constexpr,
    ZB: tl.constexpr,
    axis: tl.constexpr,  # 0/1/2
    TI: tl.constexpr,  # tile size for i (XB dim)
    TJ: tl.constexpr,  # tile size for j (YB dim)
    TK: tl.constexpr,  # tile size for k (ZB dim)
):
    """
    分块 kernel：每个 program 处理一个 tile，支持沿 axis 拼接。
    - 输入 x,y 形状为 (XB, YB, ZB)
    - 输出为拼接后的形状 (根据 axis)
    - TI/TJ/TK 为每个方向的 tile 大小（constexpr）
    """

    # block id per dimension
    bid_i = tl.program_id(0)
    bid_j = tl.program_id(1)
    bid_k = tl.program_id(2)

    # tile origin indices
    i0 = bid_i * TI
    j0 = bid_j * TJ
    k0 = bid_k * TK

    # ranges within tile (实际长度考虑边界)
    i_range = tl.arange(0, TI)
    j_range = tl.arange(0, TJ)
    k_range = tl.arange(0, TK)

    # compute actual masks for boundaries
    ii = i0 + i_range  # shape (TI,)
    jj = j0 + j_range  # (TJ,)
    kk = k0 + k_range  # (TK,)

    # masks whether indices are in bounds
    mask_i = ii < XB
    mask_j = jj < YB
    mask_k = kk < ZB

    # create 3D grid of indices using broadcasting
    # shapes: ii[:,None,None], jj[None,:,None], kk[None,None,:]
    idx = (
        ii[:, None, None] * (YB * ZB) + jj[None, :, None] * ZB + kk[None, None, :]
    )  # shape (TI, TJ, TK) but some entries out-of-bounds

    # combined mask for load/store (True where within X/Y/Z bounds)
    mask = mask_i[:, None, None] & mask_j[None, :, None] & mask_k[None, None, :]

    # load X and Y with mask (out-of-bounds read returns undefined, so mask=False avoids)
    X = tl.load(x_ptr + idx, mask=mask, other=0)
    Y = tl.load(y_ptr + idx, mask=mask, other=0)

    # Now compute output base depending on axis
    if axis == 0:
        # out shape: (XB*2, YB, ZB)
        # linear index for output: i*(YB*ZB) + j*ZB + k, but for Y we shift i by XB
        oidx_x = idx  # write X to i
        oidx_y = (
            (ii[:, None, None] + XB) * (YB * ZB)
            + jj[None, :, None] * ZB
            + kk[None, None, :]
        )
        # store with mask (only store valid positions)
        tl.store(output_ptr + oidx_x, X, mask=mask)
        tl.store(output_ptr + oidx_y, Y, mask=mask)

    elif axis == 1:
        # out shape: (XB, YB*2, ZB)
        # linear index: i*(YB*2*ZB) + j*(ZB) + k
        base = ii[:, None, None] * (YB * 2 * ZB)
        oidx_x = base + jj[None, :, None] * ZB + kk[None, None, :]
        oidx_y = base + (jj[None, :, None] + YB) * ZB + kk[None, None, :]
        tl.store(output_ptr + oidx_x, X, mask=mask)
        tl.store(output_ptr + oidx_y, Y, mask=mask)

    elif axis == 2:
        # out shape: (XB, YB, ZB*2)
        # linear index: i*(YB*(ZB*2)) + j*(ZB*2) + k
        base = ii[:, None, None] * (YB * (ZB * 2)) + jj[None, :, None] * (ZB * 2)
        oidx_x = base + kk[None, None, :]
        oidx_y = base + (kk[None, None, :] + ZB)
        tl.store(output_ptr + oidx_x, X, mask=mask)
        tl.store(output_ptr + oidx_y, Y, mask=mask)

    else:
        return


def _ceil_div(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize(
    "para_type,data_type,XB,YB,ZB,axis",
    [
        ["float32", torch.float32, 4, 4, 4, 0],
        ["float32", torch.float32, 512, 256, 512, 0],
        ["float32", torch.float32, 512, 256, 512, 1],
        ["float32", torch.float32, 512, 256, 512, 2],
        ["float16", torch.float16, 2, 8, 4, 1],
        ["int8", torch.int8, 4, 8, 2, 2],
    ],
)
def test_join_axis_added_tiled(para_type, data_type, XB, YB, ZB, axis):
    """
    增强测试：对小输入使用 grid=(1,1,1)（调用简单 kernel），对大输入使用 tiling kernel 并计算 grid。
    - 该测试仅新增，不改动原来的 fn_npu_ 或其他测试。
    """

    x = torch.full((XB, YB, ZB), 100, dtype=data_type).npu()
    y = torch.full((XB, YB, ZB), 30, dtype=data_type).npu()
    ans = torch.cat((x, y), dim=axis)

    out_shape = list(x.shape)
    out_shape[axis] = out_shape[axis] * 2
    output = torch.randint(1, tuple(out_shape), dtype=data_type).npu()

    LARGE_THRESHOLD = 16

    if max(XB, YB, ZB) <= LARGE_THRESHOLD:
        fn_npu_concat_axis_[1, 1, 1](output, x, y, XB, YB, ZB, axis, debug=True)
    else:
        TI, TJ, TK = 16, 16, 16

        # 计算每维需要多少 tile
        gx = _ceil_div(XB, TI)
        gy = _ceil_div(YB, TJ)
        gz = _ceil_div(ZB, TK)

        # launch tiled kernel with grid (gx, gy, gz)
        fn_npu_concat_axis_tiled_[gx, gy, gz](
            output, x, y, XB, YB, ZB, axis, TI, TJ, TK, debug=True
        )

    test_common.validate_cmp(para_type, ans, output)
