import triton
import torch
import textwrap
import inspect
import triton.backends.dicp_triton.driver as dicp
import triton.language as tl
import triton.compiler as tc
from pathlib import Path


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel

def _get_unary_kernel(expr):
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    return kernel

def _get_binary_kernel(expr):
    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    replacements = {'GENERATE_TEST_HERE': expr}
    kernel = patch_kernel(kernel, replacements)
    return kernel

def test_unary_kernel(expr):
    kernel = _get_unary_kernel(f'tl.{expr}(x)')
    src = tc.ASTSource(
        fn=kernel,
        constants={"SIZE": 128},
        signature="*fp32, *fp32",
    )
    ret = triton.compile(src)
    src_path = f"{expr}.mlir"
    Path(src_path).write_bytes(ret.asm["ttlinalgdir"])

def test_binary_kernel(name, expr, use_int=False):
    if use_int:
        signature="*i32, *i32, *i32"
    else:
        signature="*fp32, *fp32, *fp32"
    kernel = _get_binary_kernel(expr)
    src = tc.ASTSource(
        fn=kernel,
        constants={"SIZE": 128},
        signature=signature,
    )
    ret = triton.compile(src)
    src_path = f"{name}.mlir"
    Path(src_path).write_bytes(ret.asm["ttlinalgdir"])


def test_custom_kernel(name, kernel, constants, signature):
    src = tc.ASTSource(
        fn=kernel,
        constants=constants,
        signature=signature,
    )
    ret = triton.compile(src)
    src_path = f"{name}.mlir"
    Path(src_path).write_bytes(ret.asm["ttlinalgdir"])

@triton.jit
def clamp_max_min_kernel(x_ptr, min_ptr, max_ptr, out_ptr, ref_ptr, N, BLOCK_SIZE: tl.constexpr):
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask)
    min = tl.load(min_ptr + off, mask=mask)
    max = tl.load(max_ptr + off, mask=mask)
    out = out_ptr + off
    ref = ref_ptr + off
    tl.store(out, tl.clamp(x, min, max), mask=mask)
    ref_val = tl.minimum(tl.maximum(x, min), max)
    tl.store(ref, ref_val, mask=mask)

@triton.jit
def fma_kernel(Z, X, Y, W, SIZE: tl.constexpr):
    off = tl.arange(0, SIZE)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    w = tl.load(W + off)
    z = tl.math.fma(x, y, w)
    tl.store(Z + off, z)

@triton.jit
def umulhi_kernel(X, Y, Z, SIZE: tl.constexpr):
    offs = tl.arange(0, SIZE)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, SIZE), z)

@triton.jit
def broadcast_kernel(x_ptr, y_ptr, y_broadcasted_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + N * offset1[:, None] + offset2[None, :])
    y = tl.load(y_ptr + offset2)
    _, y_broadcasted = tl.broadcast(x, y)
    tl.store(y_broadcasted_ptr + N * offset1[:, None] + offset2[None, :], y_broadcasted)

@triton.jit
def shape_kernel(X, Y, Z):
    x = tl.arange(0, 32).expand_dims(-1).broadcast_to(32, 32).view(1024)
    y = tl.arange(0, 32).reshape(4, 8).permute(1, 0).view(32)
    z = tl.arange(0, 64).view(2,32).trans(1, 0).view(64)
    tl.store(X + tl.arange(0, 1024), x)
    tl.store(Y + tl.arange(0, 32), y)
    tl.store(Z + tl.arange(0, 64), z)

@triton.jit
def interleave_kernel(Z, N: tl.constexpr):
    z = tl.interleave(tl.arange(0, N), tl.arange(N, 2 * N))
    tl.store(Z + tl.arange(0, 2 * N), z)

@triton.jit
def join_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.join(x, y)
    tl.store(Z + tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :], z)

@triton.jit
def split_kernel(X, Z1, Z2, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    x1 = tl.reshape(x, (N // 2, 2))
    z1, z2 = tl.split(x1)
    tl.store(Z1 + tl.arange(0, N // 2), z1)
    tl.store(Z2 + tl.arange(0, N // 2), z2)


if __name__ == "__main__":
    unary_exprs = ['exp', 'log', 'cos', 'sin', 'exp2', 'log2', 'sqrt', 'rsqrt', 'sqrt_rn', 'sigmoid', 'softmax', 'floor', 'ceil', 'abs', 'erf',]
    int_binary_exprs = {"idiv":"x//y", "and":"x&y", "or":"x|y", "xor":"x^y", "lshift":"x<<y", "rshift":"x>>y", "umulhi":"tl.umulhi(x, y)"}
    float_binary_exprs = {
        "add":"x+y", "sub":"x-y", "mul":"x*y", "div":"x/y", "mod":"x%y", 
        "eq":"x==y", "neq":"x!=y", "gt":"x>y", "lt":"x<y", "nlt":"x>=y", "ngt":"x<=y", 
        "fdiv":"tl.fdiv(x, y)", "div_rn":"tl.div_rn(x, y)"
    }
    for expr in unary_exprs:
        test_unary_kernel(expr)
    for name, expr in int_binary_exprs.items():
        test_binary_kernel(name, expr, use_int=True)
    for name, expr in float_binary_exprs.items():
        test_binary_kernel(name, expr)
    test_custom_kernel("clamp_max_min",clamp_max_min_kernel,{"BLOCK_SIZE":64}, "*fp32, *fp32, *fp32, *fp32, *fp32, i32")
    test_custom_kernel("fma",fma_kernel,{"SIZE":64}, "*fp32, *fp32, *fp32, *fp32")
    test_custom_kernel("broadcast",broadcast_kernel,{"M":64, "N":128}, "*fp32, *fp32, *fp32")
    test_custom_kernel("shape",shape_kernel,None,"*fp32, *fp32, *fp32")
    test_custom_kernel("interleave",interleave_kernel,{"N":64}, "*i32")
    test_custom_kernel("join",join_kernel,{"N":64}, "*fp32, *fp32, *fp32")
    test_custom_kernel("split",split_kernel,{"N":64}, "*fp32, *fp32, *fp32")