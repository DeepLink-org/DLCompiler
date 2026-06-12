import os

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import triton.backends.dicp_triton.ascend_autotune_hooks  # noqa: F401 - install proxy before decorators

os.environ.setdefault("TRITON_AUTOTUNE_PARALLEL_COMPILE", "0")


@triton.autotune(
    configs=[
        triton.Config({"XS": 128, "multibuffer": True}),
        triton.Config({"XS": 1024, "multibuffer": True}),
        triton.Config({"XS": 1024, "multibuffer": False}),
    ],
    key=["numel"],
)
@triton.jit
def _explicit_config_exp_add_kernel(out_ptr, x_ptr, y_ptr, numel, XS: tl.constexpr):
    offsets = tl.program_id(0) * XS + tl.arange(0, XS)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = tl.full((XS,), 0.0, tl.float32)
    for i in range(8):
        out = tl.exp(x) + y + i
    tl.store(out_ptr + offsets, out, mask=mask)


def _explicit_config_exp_add(x, y):
    out = torch.empty_like(x)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["XS"]), 1, 1)
    _explicit_config_exp_add_kernel[grid](out, x, y, numel)
    return out


@triton.autotune(configs=[], key=["n_elements"])
@triton.jit
def _auto_tiling_add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _auto_tiling_add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    _auto_tiling_add_kernel[grid](x, y, out, n_elements)
    return out


@triton.autotune(
    configs=[],
    key=["n_elements"],
    hints={"compile_options": "vector"},
)
@triton.jit
def _auto_tiling_compile_options_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y * 3.0, mask=mask)


def _auto_tiling_compile_options_add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    _auto_tiling_compile_options_add_kernel[grid](x, y, out, n_elements)
    return out


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
def _hinted_tiling_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    sub_offsets = tl.arange(0, BLOCK_SIZE_SUB)
    loops = (BLOCK_SIZE + BLOCK_SIZE_SUB - 1) // BLOCK_SIZE_SUB
    for loop in range(loops):
        offsets = block_start + loop * BLOCK_SIZE_SUB + sub_offsets
        mask = offsets < min(block_start + BLOCK_SIZE, n_elements)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def _hinted_tiling_add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    _hinted_tiling_add_kernel[grid](x, y, out, n_elements)
    return out


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128, "multibuffer": False})],
    key=["n_elements"],
    hints={"auto_gen_config": True},
)
@triton.jit
def _auto_and_user_config_add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x - y, mask=mask)


def _auto_and_user_config_add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    _auto_and_user_config_add_kernel[grid](x, y, out, n_elements)
    return out


@triton.max_autotune(
    configs=[triton.Config({"BLOCK_SIZE": 256})],
    key=["n_elements"],
    kernel_type="vector",
    num_stages=[1, 2],
    enable_ubuf_saving=[True, False],
)
@triton.jit
def _max_autotune_vector_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0 + y, mask=mask)


def _max_autotune_vector(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    _max_autotune_vector_kernel[grid](x, y, out, n_elements)
    return out


@triton.max_autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128})],
    key=["n_elements"],
    kernel_type="vector",
    enable_ubuf_saving=[True, False],
)
@triton.jit
def _max_autotune_default_stage_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + 1.0, mask=mask)


def _max_autotune_default_stage(x):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    _max_autotune_default_stage_kernel[grid](x, out, n_elements)
    return out


@pytest.mark.autotune
def test_community_autotune_explicit_configs_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")
    y = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _explicit_config_exp_add(x, y)
    expected = torch.exp(x) + y + 7

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.autotune
def test_advanced_autotune_empty_configs_auto_tiling_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")
    y = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _auto_tiling_add(x, y)
    expected = x + y

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.autotune
def test_advanced_autotune_auto_tiling_compile_options_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")
    y = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _auto_tiling_compile_options_add(x, y)
    expected = x + y * 3.0

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.autotune
def test_advanced_autotune_hints_dict_key_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")
    y = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _hinted_tiling_add(x, y)
    expected = x + y

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.autotune
def test_advanced_autotune_user_configs_merge_auto_configs_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")
    y = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _auto_and_user_config_add(x, y)
    expected = x - y

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.autotune
def test_max_autotune_vector_expanded_configs_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")
    y = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _max_autotune_vector(x, y)
    expected = x * 2.0 + y

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.autotune
def test_max_autotune_uses_ascend_default_num_stages_e2e():
    x = torch.randn(4096, dtype=torch.float32, device="npu")

    actual = _max_autotune_default_stage(x)
    expected = x + 1.0

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
