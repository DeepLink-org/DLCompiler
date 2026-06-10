# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from triton.language.core import _unwrap_if_constexpr

__all__ = [
    "layout",
    "ND",
    "NZ",
    "fragment",
    "UB",
    "L1",
    "L0A",
    "L0B",
    "L0C",
]


class layout:
    ASCEND = ["ND", "NZ"]

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in layout.ASCEND, name

    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        return self.name

    def __repr__(self):
        return f"triton.language.{self.codegen_name()}"


ND = layout("ND")
NZ = layout("NZ")


class _memory_scope:
    GPU = ["fragment"]
    ASCEND = ["UB", "L1", "L0A", "L0B", "L0C"]

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in _memory_scope.ASCEND + _memory_scope.GPU, name

    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        return self.name

    def __repr__(self):
        return f"triton.language.{self.codegen_name()}"


fragment = _memory_scope("fragment")
UB = _memory_scope("UB")
L1 = _memory_scope("L1")
L0A = _memory_scope("L0A")
L0B = _memory_scope("L0B")
L0C = _memory_scope("L0C")
