import os

import torch
import torch_npu
import triton
import triton.language as tl

try:
    from triton.runtime.libentry import libentry
except ImportError:
    libentry = None
from backend.testing import do_bench_npu
import backend.ascend_autotune_hooks  # noqa: F401 — install proxy before @triton.autotune


if libentry is None:

    def test_add(size: int):
        print("SKIPPED: libentry not available in stock triton 3.5")

    if __name__ == "__main__":
        test_add(98432)

else:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1 * 1024, 'multibuffer': True}),
            triton.Config({'BLOCK_SIZE': 12 * 1024, 'multibuffer': True}),
            triton.Config({'BLOCK_SIZE': 12 * 1024, 'multibuffer': False}),
            triton.Config({'BLOCK_SIZE': 8 * 1024, 'multibuffer': True}),
        ], key=["n_elements"])
    @libentry()
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add_torch(x, y):
        return x + y

    def add_autotune(x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()
        add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )](x, y, output, n_elements)
        return output

    def test_add(size: int):
        x = torch.rand(size, device="npu")
        y = torch.rand(size, device="npu")
        output_torch = add_torch(x, y)
        output_triton = add_autotune(x, y)
        assert torch.allclose(output_triton, output_torch)
        print(f"Vector Add {size} with libentry PASSED!")

    if __name__ == "__main__":
        test_add(98432)
