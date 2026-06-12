# Backward compatibility: deeplink.extension is now a thin wrapper around cann.extension.
# The canonical Ascend extension APIs have moved to deeplink.cann.extension to align
# with triton-ascend's directory structure.
from .cann.extension import *
from .cann.extension import __all__ as _cann_all

__all__ = _cann_all
