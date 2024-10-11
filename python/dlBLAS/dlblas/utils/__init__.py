from dlblas.op_registry import op_registry
from dlblas.op_struct import OpParams, OpImpl, Tensor
from dlblas.symbolic_var import SymVar
from dlblas.autotune.space import (
    RangeSapce,
    DiscreteSpace,
    PowerOfTwoSpace,
    ChoiceSpace,
    FixedSpace,
    DictSpace,
)
from dlblas.autotune.configs import AutotuneConfig

register_dlblas_op = op_registry.register

def get_list_op_names() -> [str]:
    return op_registry.get_list_op_names()


def get_args_from_op_name(name: str):
    return op_registry.get_args_from_op_name(name)


def get_op(name: str, args):
    '''based on name and args,
    return OpImpl
    '''
    return op_registry.get_op(name, args)
