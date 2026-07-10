import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.autotune(
    configs=[],
    key={"x": "n_elements"},
    hints={
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SIZE_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    },
)
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE
    loops1 = (BLOCK_SIZE + BLOCK_SIZE_SUB - 1) // BLOCK_SIZE_SUB
    for loop in range(0, loops1):
        x0 = offset + loop * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        mask = x0 < n_elements
        x = tl.load(x_ptr + x0, mask)
        y = tl.load(y_ptr + x0, mask)
        output = x + y
        tl.store(output_ptr + x0, output)


def add_torch(x, y):
    return x + y


def add_autotune(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)](
        x, y, output, n_elements
    )
    return output


@pytest.mark.autotune
@pytest.mark.parametrize(
    "size",
    [
        2048,
    ],
)
def test_add(size: int):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output_torch = add_torch(x, y)
    output_triton = add_autotune(x, y)
    assert torch.allclose(output_triton, output_torch)


@pytest.mark.autotune
def test_add_no_reduction_axes():
    with pytest.raises(ValueError, match="reduction_axes must be a list"):

        @triton.autotune(
            configs=[],
            key={"x": "n_elements"},
            hints={
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
            },
        )
        @triton.jit
        def add_kernel_exception():
            pass


@pytest.mark.autotune
def test_add_no_keyname():
    with pytest.raises(ValueError, match="All keys in 'key' must be valid axis names"):

        @triton.autotune(
            configs=[],
            key={"x0": "n_elements"},
            hints={
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            },
        )
        @triton.jit
        def add_kernel_exception():
            pass
