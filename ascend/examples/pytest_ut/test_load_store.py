import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


@triton.jit
def triton_load_store(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (8, 8, 4), 2, 128, 64],
                             ['float16', (8, 8, 4), 2, 128, 64],
                             ['int8', (8, 8, 4), 2, 128, 64],
                         ]
                         )
def test_load_store(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_ref = x0
    # triton结果
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_load_store[ncore, 1, 1](x0, y_cal, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, y_cal, y_ref)
