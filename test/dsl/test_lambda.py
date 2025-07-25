import torch
import triton
import triton.language as tl

import triton.backends.dicp_triton.driver as dicp
triton.runtime.driver.set_active(dicp.DICPDriver('ascend'))

@triton.jit
def lambda_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # 原始方法
    # block_start = pid * BLOCK_SIZE
    # 使用 lambda 函数计算块的起始位置
    get_block_start = lambda pid: pid * BLOCK_SIZE
    block_start = get_block_start(pid)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # 创建掩码，防止越界访问

    input_data = tl.load(input_ptr + offsets, mask=mask, other=0)  # 使用掩码加载数据
    tl.store(output_ptr + offsets, input_data, mask=mask)  # 使用掩码存储数据


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
    lambda_kernel[grid](input_tensor, output_tensor, N=N, BLOCK_SIZE=BLOCK_SIZE)

    # 验证结果
    assert torch.allclose(input_tensor, output_tensor), "Output does not match input"
    print("Test passed!")

if __name__ == "__main__":
    test_kernel()
