from dlblas.frontend import *
from dlblas.op_registry import op_registry
from dlblas.op_struct import OpParams, OpImpl
from dlblas.symbolic_var import SymVar, Tensor

register_dlblas_op = op_registry.register

# this import all kernels dynamically
import dlblas.kernels

__version__ = "0.0.1"
