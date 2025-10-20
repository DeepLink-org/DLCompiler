from triton.backends.dicp_triton.utils import init_dicp_driver
from . import libdevice
from .core import (
    insert_slice, 
    extract_slice, 
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
)

__all__ = [
    "libdevice",
    "insert_slice", 
    "extract_slice", 
    "parallel", 
    "inline_lambda"
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
]

init_dicp_driver()
