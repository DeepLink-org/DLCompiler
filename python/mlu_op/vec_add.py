import os
import argparse

import torch
import numpy as np

import triton
import triton.language as tl


def parse_args():
    parser = argparse.ArgumentParser(description="???")

    parser.add_argument("--stages", type=int, default=0)
    parser.add_argument('-b', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--dump', default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def main():
    args = parse_args()

    @triton.autotune(
        configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_stages=0, num_warps=8),
            triton.Config(kwargs={'BLOCK_SIZE': 10240}, num_stages=0, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 18432}, num_stages=0, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 32768}, num_stages=0, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 43520}, num_stages=0, num_warps=4),

            # this is a I/O bound kernel, so pipeline doesn't help
            # triton.Config(kwargs={'BLOCK_SIZE': 43520}, num_stages=1, num_warps=4),
            # triton.Config(kwargs={'BLOCK_SIZE': 43520}, num_stages=2, num_warps=4),
            # triton.Config(kwargs={'BLOCK_SIZE': 43520}, num_stages=3, num_warps=4),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)




    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        assert x.is_mlu and y.is_mlu and output.is_mlu
        n_elements = output.numel()

        # The SPMD launch grid denotes the number of kernel instances that run in parallel.
        # It is analogous to MLU launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
        # In this case, we use a 1D grid where the size is the number of blocks:

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable MLU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.

        kernel = add_kernel[grid](x, y, output, n_elements)

        # We return a handle to z but, since `torch.mlu.synchronize()` hasn't been called, the kernel is still
        # running asynchronously at this point.

        if args.dump:
            assert not args.b, f'cannot bench and dump ttir'
            print()
            print(list(kernel.metadata.keys()))
            print(list(kernel.asm.keys()))

            dir_path = os.path.dirname(os.path.realpath(__file__))
            f_name = os.path.basename(__file__)
            f_name = f'{f_name[:-3]}_ttir.mlir'
            full_path = os.path.join(dir_path, f_name)

            with open(f"{full_path}", "w") as f:
                f.write(kernel.asm['ttir'])

        return output

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='mlu')
    y = torch.rand(size, device='mlu')
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')




    @triton.jit
    def add_large_kernel(
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
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        # add to tl.int64 for large tensor
        block_start = block_start.to(tl.int64)
        step_size = tl.num_programs(axis=0) * BLOCK_SIZE
        while 0 <= block_start and block_start < n_elements:
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
            block_start += step_size


    def add_large(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, N):
        grid = lambda meta: (min(triton.cdiv(N, meta['BLOCK_SIZE']), 65535), )
        add_large_kernel[grid](x, y, z, N, BLOCK_SIZE=8192)
        return z

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],  # Argument names to use as an x-axis for the plot.
            x_vals=[2**i for i in range(12, 28, 1)
                    ],  # Different possible values for `x_name`.
            x_log=True,  # x axis is logarithmic.
            line_arg=
            'provider',  # Argument name whose value corresponds to a different line in the plot.
            line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
            line_names=['Triton', 'Torch'],  # Label name for the lines.
            styles=[('blue', '-'), ('green', '-')],  # Line styles.
            ylabel='GB/s',  # Label name for the y-axis.
            plot_name=
            'vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        ))
    def benchmark(size, provider):
        x = torch.rand(size, device='mlu', dtype=torch.float32)
        y = torch.rand(size, device='mlu', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y,
                                                         quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y),
                                                         quantiles=quantiles)
        gbps = lambda ms: 12 * size / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    if args.b:
        benchmark.run(print_data=True, show_plots=True)

if __name__ == '__main__':
    main()