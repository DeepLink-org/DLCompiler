"""Compile device printing without loading a kernel on the card."""
import re

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def print_vector(X, HEX: tl.constexpr):
    value = tl.load(X + tl.arange(0, 8))
    tl.device_print("abi", value, hex=HEX)


@triton.jit
def print_scalar(X, HEX: tl.constexpr):
    tl.device_print("abi", tl.load(X), hex=HEX)


@pytest.mark.parametrize("kernel", [print_scalar, print_vector])
@pytest.mark.parametrize("dtype,hex_mode,expected", [
    ("fp16", False, r"fpext half .* to double"),
    ("bf16", False, r"fpext bfloat .* to double"),
    ("i16", False, r"sext i16 .* to i32"),
    ("u16", False, r"zext i16 .* to i32"),
    ("fp16", True, r"zext i16 .* to i32"),
])
def test_printf_uses_c_vararg_promotions(kernel, dtype, hex_mode, expected):
    source = ASTSource(kernel, signature={"X": "*" + dtype, "HEX": "constexpr"},
                       constexprs={"HEX": hex_mode})
    result = triton.compile(source, target=GPUTarget("wafer", "wafer", 32))
    assert re.search(expected, result.asm["llir"]), result.asm["llir"]

