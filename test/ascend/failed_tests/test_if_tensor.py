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
import triton
import triton.language as tl


@triton.jit
def if_tensor_kernel(
        kv_start_idx,  # tensor
        output_ptr,
):
    pid = tl.program_id(0)
    if kv_start_idx is not None:
        value = tl.load(kv_start_idx + pid)
        tl.store(output_ptr + pid, value)


# 测试函数
def test_kernel():
    n = 8
    device = 'npu'

    kv_start_idx = torch.arange(n, dtype=torch.float32, device=device)
    output1 = torch.zeros(n, dtype=torch.float32, device=device)
    if_tensor_kernel[(n,)](
        kv_start_idx, output1,
    )

    expected = torch.arange(n, dtype=torch.float32, device=device)
    assert torch.allclose(output1, expected), f"Output {output1} != Expected {expected}"
    print(f"RESULT: output1 = {output1}")
    print("✅ Test passed!")


if __name__ == "__main__":
    test_kernel()
