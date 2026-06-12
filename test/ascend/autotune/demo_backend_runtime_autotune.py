import torch
import torch_npu
import triton
import triton.language as tl

from backend.ascend_autotune_runtime import autotune as ascend_autotune


@ascend_autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
    hints={"compile_options": "vector"},
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements)
    return out


def main():
    n_elements = 98432
    x = torch.rand(n_elements, device="npu", dtype=torch.float32)
    y = torch.rand(n_elements, device="npu", dtype=torch.float32)

    out = add(x, y)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + y, rtol=1e-3, atol=1e-3)
    print("backend.ascend_autotune_runtime autotune demo PASSED")


if __name__ == "__main__":
    main()
