from triton.backends.dicp_triton.utils import init_dicp_driver
from .core import (insert_slice, extract_slice, parallel, InlineLambda)

__all__ = [
    "insert_slice", "extract_slice", "parallel", "InlineLambda"
]

init_dicp_driver()
