import pytest
import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl

from triton.language.extra import libdevice


@pytest.mark.parametrize("dtype_str", ["float32"])
@pytest.mark.parametrize("size", [128, 4])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [("tanh", "tanh"), ("pow", "pow"), ("fmod", "fmod"), ("isnan", "isnan"),
     ("isinf", "isinf"), ("finitef", "isfinite"), ("ceil", "ceil"), ("floor", "floor"), ("rint", "round"),
     ("trunc", "trunc"),  # Add the new function test
     ],
)
def test_special(dtype_str, size, libdevice_fn, torch_special_fn, device):
    SIZE = size
    dtype = getattr(torch, dtype_str)

    if torch_special_fn in ["pow", "fmod"]:
        x = torch.randn((SIZE, ), dtype=dtype, device="cpu")
        y = torch.randn((SIZE, ), dtype=dtype, device="cpu")
        y_exp = torch.empty((SIZE, ), dtype=dtype, device="cpu")
        if torch_special_fn == "pow":
            y_ref = torch.pow(x, y)
        elif torch_special_fn == "fmod":
            y_ref = torch.fmod(x, y)
    elif torch_special_fn in ["isnan", "isinf", "isfinite"]:  # Add isfinite to this condition
        x = torch.randn((SIZE, ), dtype=dtype, device="cpu")
        # Set some element as nan&-nan
        x[SIZE // 4] = float('inf')
        x[SIZE // 2] = float('-inf')
        x[3 * SIZE // 4] = float('nan')

        # Use bool as return value
        y_exp = torch.empty((SIZE, ), dtype=torch.bool, device="cpu")
        if torch_special_fn == "isnan":
            y_ref = torch.isnan(x)
        elif torch_special_fn == "isinf":
            y_ref = torch.isinf(x)
        else:  # isfinite
            y_ref = torch.isfinite(x)
    elif torch_special_fn in ["ceil", "floor", "trunc", "round"]:
        # For ceil, floor, and trunc, we can use the same input
        # as they are unary operations.
        x = torch.randn((SIZE, ), dtype=dtype, device="cpu")
        y_exp = torch.empty((SIZE, ), dtype=dtype, device="cpu")
        if torch_special_fn == "ceil":
            y_ref = torch.ceil(x)
        elif torch_special_fn == "floor":
            y_ref = torch.floor(x)
        elif torch_special_fn == "trunc":
            y_ref = torch.trunc(x)
        elif torch_special_fn == "round":
            y_ref = torch.round(x)
    else:
        x = torch.randn((SIZE, ), dtype=dtype, device="cpu")
        y_exp = torch.empty((SIZE, ), dtype=dtype, device="cpu")
        if torch_special_fn == "tanh":
            y_ref = torch.tanh(x)
        else:
            y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel_pow(x_ptr, y_ptr, out_ptr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        res = libdevice.pow(x, y)
        tl.store(out_ptr + off, res)

    @triton.jit
    def kernel_fmod(x_ptr, y_ptr, out_ptr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        res = libdevice.fmod(x, y)
        tl.store(out_ptr + off, res)

    @triton.jit
    def kernel_rint(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get rounded result
        res = libdevice.rint(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_unary(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_isnan(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get bool result
        res = libdevice.isnan(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_isinf(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get bool result
        res = libdevice.isinf(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_finitef(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get bool result
        res = libdevice.finitef(x)
        tl.store(out_p + off, res)

    if torch_special_fn == "pow":
        x_txda = x.to("txda")
        y_txda = y.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_pow[(1, )](x_txda, y_txda, y_exp_txda, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())
    elif torch_special_fn == "round":
        x_txda = x.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_rint[(1, )](x_txda, y_exp_txda, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())
    elif torch_special_fn == "fmod":
        x_txda = x.to("txda")
        y_txda = y.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_fmod[(1, )](x_txda, y_txda, y_exp_txda, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())
    elif torch_special_fn == "isnan":
        x_txda = x.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_isnan[(1, )](x_txda, y_exp_txda, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())
    elif torch_special_fn == "isinf":
        x_txda = x.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_isinf[(1, )](x_txda, y_exp_txda, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())
    elif torch_special_fn == "isfinite":
        x_txda = x.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_finitef[(1, )](x_txda, y_exp_txda, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())
    else:
        x_txda = x.to("txda")
        y_exp_txda = y_exp.to("txda")
        kernel_unary[(1, )](x_txda, y_exp_txda, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)
        with torch.no_grad():
            y_exp.copy_(y_exp_txda.cpu())

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)


def test_libdevice_rename(device):

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device="cpu")
    out = torch.empty_like(inp)

    inp_txda = inp.to("txda")
    out_txda = out.to("txda")
    triton_copy[(1, )](inp_txda, out_txda, BLOCK_SIZE)
    with torch.no_grad():
        out.copy_(out_txda.cpu())
    torch.testing.assert_close(out, inp)


def test_libdevice_erf(device):
    """Exercise the extern wrapper, independently of tl.erf's frontend entry."""
    @triton.jit
    def erf_kernel(inp, out, BLOCK: tl.constexpr):
        indices = tl.arange(0, BLOCK)
        values = tl.load(inp + indices)
        tl.store(out + indices, libdevice.erf(values))

    values = torch.tensor([-3.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 3.0],
                          dtype=torch.float32, device="cpu")
    output = torch.empty_like(values)
    values_txda = values.to("txda")
    output_txda = output.to("txda")
    erf_kernel[(1,)](values_txda, output_txda, BLOCK=8)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    torch.testing.assert_close(output, torch.erf(values), atol=1e-3, rtol=1e-3)
