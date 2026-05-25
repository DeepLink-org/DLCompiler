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

from triton._C.libtriton import ir, dicp_triton

# MLIR affine bindings (same objects as triton._C.libtriton.ascend.ir).
affine_expr = dicp_triton.ir.affine_expr
affine_constant_expr = dicp_triton.ir.affine_constant_expr
affine_dim_expr = dicp_triton.ir.affine_dim_expr
affine_symbol_expr = dicp_triton.ir.affine_symbol_expr
affine_binary_op_expr = dicp_triton.ir.affine_binary_op_expr
affine_map = dicp_triton.ir.affine_map

AffineExpr = affine_expr
AffineConstantExpr = affine_constant_expr
AffineDimExpr = affine_dim_expr
AffineSymbolExpr = affine_symbol_expr
AffineBinaryOpExpr = affine_binary_op_expr
AffineMap = affine_map

from .core import (
    ascend_address_space,
    builtin,
    CORE,
    copy_from_ub_to_l1,
    copy,
    debug_barrier,
    fixpipe,
    FixpipeDMAMode,
    FixpipeDualDstMode,
    FixpipePreQuantMode,
    FixpipePreReluMode,
    int64,
    is_builtin,
    MODE,
    PIPE,
    IteratorType,
    sub_vec_id,
    sub_vec_num,
    sync_block_all,
    sync_block_set,
    sync_block_wait,
    SYNC_IN_VF,
)

from .scope import scope

from .custom_op import (
    custom,
    custom_semantic,
    register_custom_op,
)

from .math_ops import (
    atan2,
    isfinited,
    finitef,
)

from .aux_ops import (
    parallel,
    compile_hint,
    multibuffer,
)

from .vec_ops import (
    get_element,
    sort,
    flip,
)

from .mem_ops import (
    index_put,
    gather_out_to_ub,
    scatter_ub_to_out,
    index_select_simd,
)

# Re-export from deeplink.core for backward compatibility.
from ...core import (
    insert_slice,
    extract_slice,
    alloc,
    ND,
    NZ,
    fragment,
    UB,
    L1,
    L0A,
    L0B,
    L0C,
    SyncFlag,
    set_cross_flag,
    wait_cross_flag,
    inline_lambda,
)

# gather is a standard triton op; re-export for backward compat.
from triton.language.core import gather

__all__ = [
    # core
    "builtin",
    "copy_from_ub_to_l1",
    "copy",
    "CORE",
    "debug_barrier",
    "fixpipe",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "int64",
    "is_builtin",
    "MODE",
    "PIPE",
    "IteratorType",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_all",
    "sync_block_set",
    "sync_block_wait",
    "SYNC_IN_VF",
    # address space
    "ascend_address_space",
    # MLIR affine
    "affine_expr",
    "affine_constant_expr",
    "affine_dim_expr",
    "affine_symbol_expr",
    "affine_binary_op_expr",
    "affine_map",
    "AffineExpr",
    "AffineConstantExpr",
    "AffineDimExpr",
    "AffineSymbolExpr",
    "AffineBinaryOpExpr",
    "AffineMap",
    # scope
    "scope",
    # custom op
    "custom",
    "custom_semantic",
    "register_custom_op",
    # math ops
    "atan2",
    "isfinited",
    "finitef",
    # aux ops
    "parallel",
    "compile_hint",
    "multibuffer",
    # vec ops
    "get_element",
    "sort",
    "flip",
    # mem ops
    "index_put",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_select_simd",
    # backward compat from deeplink.core
    "insert_slice",
    "extract_slice",
    "alloc",
    "ND",
    "NZ",
    "fragment",
    "UB",
    "L1",
    "L0A",
    "L0B",
    "L0C",
    "SyncFlag",
    "set_cross_flag",
    "wait_cross_flag",
    "inline_lambda",
    # standard
    "gather",
]
