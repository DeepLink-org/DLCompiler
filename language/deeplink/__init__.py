from triton.backends.dicp_triton.utils import init_dicp_driver
from . import libdevice
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

__all__ = [
    "libdevice",
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
]

init_dicp_driver()
