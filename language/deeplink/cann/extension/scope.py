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


class scope:
    def __init__(self, core_mode: str, _builder=None, _semantic=None, **kwargs):
        self.core_mode = (
            _unwrap_if_constexpr(core_mode) if _builder is None else core_mode
        )
        self._builder = _builder
        self._semantic = _semantic
        self.disable_auto_sync = kwargs.get("disable_auto_sync", False)
        if self.core_mode not in ("cube", "vector"):
            raise ValueError(
                f'core_mode must be "cube" or "vector", got {self.core_mode}'
            )

    def __enter__(self):
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
