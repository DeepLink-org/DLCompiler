import torch
import triton
import triton.language as tl

import triton.backends.dicp_triton.driver as dicp
triton.runtime.driver.set_active(dicp.DICPDriver('ascend'))

@triton.jit
def kernel1(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    get_block_start = pid * BLOCK_SIZE
    block_start = get_block_start

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N 

    input_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.store(output_ptr + offsets, input_data, mask=mask)


def test_kernel():
    # 定义参数
    BLOCK_SIZE = 32
    N = 1024
    # 初始化输入数据
    input_tensor = torch.arange(N, dtype=torch.float32, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    # 计算网格大小
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # 启动内核
    kernel1[grid](input_tensor, output_tensor, N=N, BLOCK_SIZE=BLOCK_SIZE)

    # 验证结果
    assert torch.allclose(input_tensor, output_tensor), "Output does not match input"
    print("Test passed!")

if __name__ == "__main__":
    test_kernel()
