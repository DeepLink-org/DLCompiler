from dlblas.op_registry import op_registry
from dlblas.op_struct import OpParams, OpImpl
from dlblas.symbolic_var import SymVar, Tensor
from dlblas.autotune.space import (
    RangeSapce,
    DiscreteSpace,
    PowerOfTwoSpace,
    ChoiceSpace,
    FixedSpace,
    DictSpace,
)

register_dlblas_op = op_registry.register

# this import all kernels dynamically
import dlblas.kernels


def get_list_op_names() -> list[str]:
    return op_registry.get_list_op_names()


def get_args_from_op_name(name: str):
    return op_registry.get_args_from_op_name(name)


def get_op(name: str, args):
    '''based on name and args,
    return OpImpl
    '''
    return op_registry.get_op(name, args)


__version__ = "0.0.1"
