"""
Dispatch table for Ascend-specific 'with' statement context managers.
"""

from .extension import scope
from .code_generator import handle_scope_with, mangle_ty

__all__ = ["DEEPLINK_WITH_DISPATCH"]

DEEPLINK_WITH_DISPATCH = {
    scope: handle_scope_with,
    "mangle_ty": mangle_ty,
}
