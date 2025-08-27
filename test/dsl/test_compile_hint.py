import torch
import triton
import triton.language as tl
import pytest
import triton.language.extra.deeplink as dl

# eg: pytest -v test_compile_hint.py::test_compile_hint
#############################


@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        dl.compile_hint(tmp0, "hint_a")
        dl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        dl.compile_hint(tmp2, "hint_b", 42)
        dl.compile_hint(tmp2, "hint_c", True)
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                         ]
                         )
def test_compile_hint(param_list):
    dtype_str, shape, ncore, xblock, xblock_sub = param_list
    dtype = getattr(torch, dtype_str)
    x0 = torch.rand(shape, dtype=dtype).npu()
    y_ref = x0
    y_cal = torch.rand(shape, dtype=dtype).npu()
    triton_compile_hint[(ncore, )](x0, y_cal, x0.numel(), xblock, xblock_sub)
    assert torch.allclose(y_cal, y_ref)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(y_cal - y_ref))}')
    assert y_cal.dtype == y_ref.dtype
    print(f"dtype is same.")
