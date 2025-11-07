import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common


def torch_pointwise_even_blocks(x0, x1, xblock):
    """
    参考实现：只有第偶数个 block 的元素会相加（block id 从 0 开始计数），其他 block 输出保持为 0（或 y_cal 初始值）。
    xblock: 一个 block 的大小（线性元素数），与 Triton kernel 中的 XBLOCK 对应
    """
    # 展平为一维线性内存，保持 dtype & device
    x0_flat = x0.reshape(-1)
    x1_flat = x1.reshape(-1)
    n = x0_flat.numel()
    idx = torch.arange(n, device=x0.device)
    block_id = (idx // xblock) % 2
    mask = block_id == 0
    res_flat = torch.zeros_like(x0_flat)
    # 只有偶数 block 做加法
    res_flat[mask] = x0_flat[mask] + x1_flat[mask]
    return res_flat.reshape(x0.shape)


@triton.jit
def triton_add(
    in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr
):
    """
    Triton kernel：只有 program_id(0) 为偶数的 block 才会执行 load/add/store；奇数 block 跳过（不修改输出）
    """
    bid = tl.program_id(0)  # block id
    # 如果此 block 是奇数，直接返回（不做任何 store）
    if bid % 2 != 0:
        return

    offset = bid * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = x0_prime
        tmp0 = tl.load(in_ptr0 + x0, None)
        tmp1 = tl.load(in_ptr1 + x0, None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + x0, tmp2, None)


@pytest.mark.parametrize(
    "param_list",
    [
        ["float32", (2, 4096, 8), 2, 32768, 1024],
        ["float16", (2, 4096, 8), 2, 32768, 1024],
        ["int8", (2, 4096, 8), 2, 32768, 1024],
    ],
)
def test_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # 参考结果：只在偶数 block 做加法
    y_ref = torch_pointwise_even_blocks(x0, x1, xblock)
    y_cal = torch.zeros(shape, dtype=eval("torch." + dtype)).npu()
    triton_add[ncore, 1, 1](x0, x1, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize(
    "param_list",
    [
        ["float32", (2, 4096, 8), 2, 32768, 1024],
        ["float32", (128, 4096, 1280), 1310720, 512, 64],
        ["float16", (128, 4096, 1280), 1310720, 512, 64],
        ["int8", (128, 4096, 1280), 1310720, 512, 64],
    ],
)
def test_all_blocks_parallel(param_list, monkeypatch):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_pointwise_even_blocks(x0, x1, xblock)
    y_cal = torch.zeros(shape, dtype=eval("torch." + dtype)).npu()
    triton_add[ncore, 1, 1](x0, x1, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)
    monkeypatch.delenv("TRITON_ALL_BLOCKS_PARALLEL")
