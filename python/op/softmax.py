import torch

import triton
import triton.language as tl

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

@triton.autotune(configs=get_configs_io_bound(), key=['n_cols'])
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

torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
