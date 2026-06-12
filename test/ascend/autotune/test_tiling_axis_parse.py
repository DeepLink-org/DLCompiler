import pytest
import triton
import triton.language as tl
from test_common import check_axes_parse_res, mock_autotuner


def test_tiling_axis_parse_base_case1(mock_autotuner):
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_tiling_axis_parse_base_case1(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SUB: tl.constexpr,
    ):
        offset = tl.program_id(axis=0) * BLOCK_SIZE
        base = tl.arange(0, BLOCK_SUB)
        loops = (BLOCK_SIZE + BLOCK_SUB - 1) // BLOCK_SUB
        for loop in range(loops):
            offsets = offset + (loop * BLOCK_SUB) + base
            mask = offsets < min(BLOCK_SIZE + offset, n_elements)

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y

            tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    act_res = triton_tiling_axis_parse_base_case1[grid]()

    check_axes_parse_res(act_res, ref_res)


@pytest.mark.skip
def test_tiling_axis_parse_base_case2(mock_autotuner):
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_tiling_axis_parse_base_case2(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SUB: tl.constexpr,
    ):
        offset = tl.program_id(axis=0) * BLOCK_SIZE
        base = tl.arange(0, BLOCK_SUB)
        for offset_sub in range(0, BLOCK_SIZE, BLOCK_SUB):
            offsets = offset + offset_sub + base[:]
            mask = offsets < min(BLOCK_SIZE + offset, n_elements)

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y

            tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    act_res = triton_tiling_axis_parse_base_case2[grid]()

    check_axes_parse_res(act_res, ref_res)


@pytest.mark.skip
def test_tiling_axis_parse_base_case3(mock_autotuner):
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_tiling_axis_parse_base_case3(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SUB: tl.constexpr,
    ):
        offset = tl.program_id(axis=0) * BLOCK_SIZE
        base = tl.arange(0, BLOCK_SUB)[:]
        for offset_sub in range(0, BLOCK_SIZE, BLOCK_SUB):
            offsets = offset + offset_sub + base
            mask = offsets < min(BLOCK_SIZE + offset, n_elements)

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y

            tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    act_res = triton_tiling_axis_parse_base_case3[grid]()

    check_axes_parse_res(act_res, ref_res)
