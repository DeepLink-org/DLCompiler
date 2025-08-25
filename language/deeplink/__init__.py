from triton.backends.dicp_triton.utils import init_dicp_driver
from .core import (
    insert_slice, 
    extract_slice, 
    parallel, 
    inline_lambda,
    alloc,
    compile_hint,
    multibuffer,
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
    "insert_slice", 
    "extract_slice", 
    "parallel", 
    "inline_lambda"
    "alloc",
    "compile_hint",
    "multibuffer",
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
