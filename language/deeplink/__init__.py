from triton.language import math

from . import libdevice
from . import extension
from .async_task import async_task
from .core import (
    insert_slice,
    extract_slice,
    sync_block_all,
    set_cross_flag,
    wait_cross_flag,
    parallel,
    inline_lambda,
    alloc,
    compile_hint,
    ND,
    NZ,
    fragment,
    UB,
    L1,
    L0A,
    L0B,
    L0C,
    SyncFlag,
)
from .custom_op import (
    custom,
    custom_semantic,
    register_custom_op,
    CORE,
    PIPE,
    MODE,
)

# ---------------------------------------------------------------------------
# Glue layer: delegate standard math functions to triton.language.math
# (aligned with triton-ascend cann/__init__.py)
# ---------------------------------------------------------------------------

libdevice.umulhi = math.umulhi
libdevice.exp = math.exp
libdevice.exp2 = math.exp2
libdevice.log = math.log
libdevice.log2 = math.log2
libdevice.cos = math.cos
libdevice.sin = math.sin
libdevice.sqrt = math.sqrt
libdevice.sqrt_rn = math.sqrt_rn
libdevice.rsqrt = math.rsqrt
libdevice.div_rn = math.div_rn
libdevice.erf = math.erf
libdevice.floor = math.floor
libdevice.ceil = math.ceil
libdevice.fdiv = math.fdiv
libdevice.fma = math.fma
libdevice.abs = math.abs

# Reverse override: libdevice's tanh supports bf16 via cast, replace math.tanh
math.tanh = libdevice.tanh

__all__ = [
    "libdevice",
    "extension",
    "insert_slice",
    "extract_slice",
    "sync_block_all",
    "set_cross_flag",
    "wait_cross_flag",
    "parallel",
    "inline_lambda",
    "alloc",
    "compile_hint",
    "ND",
    "NZ",
    "fragment",
    "UB",
    "L1",
    "L0A",
    "L0B",
    "L0C",
    "SyncFlag",
    "async_task",
    "custom",
    "custom_semantic",
    "register_custom_op",
    "CORE",
    "PIPE",
    "MODE",
]


def ensure_driver_initialized():
    from triton.runtime.driver import driver

    if driver._active is None:
        from triton.backends.dicp_triton.utils import init_dicp_driver

        init_dicp_driver()
