import torch

import triton
import triton.language as tl
import triton.compiler as tc

# from triton.backends.triton.driver import DICPDriver
# from triton.common.backend import register_backend
import triton.backends.dicp_triton.driver as dicp
import triton.compiler as tc
from pathlib import Path
# import backend.driver as bd


triton.runtime.driver.set_active(dicp.DICPDriver('npu'))
# triton.runtime.driver.set_active(cu.CudaDriver())
# register_backend("dicp", cl.DICPBackend("mlu"))

@triton.jit
def softmax_native_kernel(output_ptr, input_ptr, input_row_stride, 
                          output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def get_configs_io_bound():
    configs = []
    for row_block in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        configs.append(triton.Config(kwargs = {'ROW_BLOCK': row_block}, num_stages = 0, num_warps =1))
    
    return configs

# @triton.autotune(configs=get_configs_io_bound(), key=['n_cols'])
@triton.jit
def softmax_optimize_kernel(output_ptr, input_ptr, input_row_stride, 
                           output_row_stride, n_rows, n_cols, 
                           BLOCK_SIZE: tl.constexpr, 
                           OUTER_ROW_BLOCK: tl.constexpr,
                           ROW_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ub = tl.minimum(n_rows, (pid + 1) * OUTER_ROW_BLOCK)
    outer_raw_idx = pid * OUTER_ROW_BLOCK
    for inner_row in range(outer_raw_idx, ub, ROW_BLOCK):
        row_offset = tl.arange(0, ROW_BLOCK)
        row_start_ptr = input_ptr + (inner_row + row_offset) * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        input_ptrs = row_start_ptr[:, None] + col_offsets
        
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis = 1)[:, None]
        numerator = tl.exp(row_minus_max)
        dominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / dominator

        output_row_start_ptr = output_ptr + (inner_row + row_offset) * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets

        tl.store(output_ptrs, softmax_output, mask = col_offsets < n_cols)

# @triton.autotune(
#     configs=[
#         triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_stages=1, num_warps=8),
#         triton.Config(kwargs={'BLOCK_SIZE': 10240}, num_stages=1, num_warps=4),
#         triton.Config(kwargs={'BLOCK_SIZE': 18432}, num_stages=1, num_warps=4),
#         triton.Config(kwargs={'BLOCK_SIZE': 32768}, num_stages=1, num_warps=4),
#         triton.Config(kwargs={'BLOCK_SIZE': 43520}, num_stages=1, num_warps=4),
#     ],
#     key=['n_elements'],
# )
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
        x_ptr: *Pointer* to first input vector.
        y_ptr: *Pointer* to second input vector.
        output_ptr: *Pointer* to output vector.
        n_elements: Size of the vector.
        BLOCK_SIZE: Number of elements each program should process. `constexpr` so it can be used as a shape value.
    """
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    # assert x.is_mlu and y.is_mlu and output.is_mlu
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to MLU launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable MLU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[(grid)](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.mlu.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

# torch.manual_seed(0)
# size = 98432
# x = torch.rand(size, device='cuda')
# y = torch.rand(size, device='cuda')
# # output_torch = x + y
# output_triton = add(x, y)
# src = tc.ASTSource(
#     fn=add_kernel,
#     constants={"BLOCK_SIZE": 1024},
#     signature="*fp32, *fp32, *fp32, i32",
# )
# ret = triton.compile(src)
# src_path = "add_kernel.mlir"
# str = ret.asm["ttlinalgdir"]
# Path(src_path).write_bytes(ret.asm["ttlinalgdir"])


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid = lambda META:(48, 1, 1)
    OUTER_ROW_BLOCK = triton.cdiv(n_rows, 48)
    softmax_optimize_kernel[(48, 1, 1)](y, x, x.stride(0), y.stride(0), 
                                    n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, 
                                    OUTER_ROW_BLOCK=OUTER_ROW_BLOCK)
    return y

src = tc.ASTSource(
    fn=softmax_optimize_kernel,
    constants={"BLOCK_SIZE": 64, "OUTER_ROW_BLOCK": 8, "ROW_BLOCK": 1},
    signature="*fp32, *fp32, i32, i32, i32, i32",
)
ret = triton.compile(src)
src_path = "softmax_optimize_kernel.mlir"
Path(src_path).write_bytes(ret.asm["ttir"].encode())

# ==============================

# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
assert torch.allclose(output_torch, output_triton)
print("success")
