from triton.backends.dicp_triton.utils import init_dicp_driver
from .core import (insert_slice, extract_slice, parallel, inline_lambda)

__all__ = [
    "insert_slice", "extract_slice", "parallel", "inline_lambda"
]

init_dicp_driver()
