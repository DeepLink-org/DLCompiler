import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel_device_assert_scalar(COND, BLOCK: tl.constexpr):
    tl.device_assert(COND, "test scalar")


@triton.jit
def kernel_device_assert_tensor(COND, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    cond_value = tl.load(COND + offsets, mask=mask, other=1)
    tl.device_assert(cond_value, "test tensor")


@pytest.mark.parametrize('cond', [True, False])
def test_assert_scalar(cond, request):
    if not cond and request.config.getoption("--wafer-hardware"):
        pytest.skip("False device assertions terminate through firmware __assert_func; compile coverage only on this shared card")
    kernel_device_assert_scalar[(1, )](cond, BLOCK=16, debug=True)


@pytest.mark.parametrize('cond_list', [
    [True, True, True],
    [True, False, True],
    [False, False, False],
    [True],
    [False],
])
def test_assert_tensor(cond_list, request):
    if not all(cond_list) and request.config.getoption("--wafer-hardware"):
        pytest.skip("False device assertions terminate through firmware __assert_func; compile coverage only on this shared card")
    cond_tensor = torch.tensor(cond_list, dtype=torch.bool, device="cpu")
    n_elements = cond_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK']), )
    cond_tensor_txda = cond_tensor.to("txda")
    kernel_device_assert_tensor[grid](cond_tensor_txda, n_elements, BLOCK=16, debug=True)


def run_all_tests():
    test_assert_scalar()
    test_assert_tensor()
    print("Manually check that the printout is correct!")


if __name__ == "__main__":
    # Run the test with pytest
    run_all_tests()
