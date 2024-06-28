from dataclasses import dataclass, field, astuple
from typing import Any

from dlblas.symbolic_var import Tensor


@dataclass(frozen=True)
class OpParams:
    n_args: int
    args_types: list[str]
    args: tuple


@dataclass(frozen=True)
class OpImpl:
    params: OpParams
    kernel: callable
    bench_fn: callable

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.kernel(*args, **kwargs)


def parse_args(args: tuple):
    types = []
    for i, arg in enumerate(args):
        if isinstance(arg, Tensor):
            types.append('tensor')
        elif isinstance(arg, str):
            types.append('str')
        elif isinstance(arg, int):
            types.append('int')
        else:
            raise TypeError(f"arg {i} has unsupported type {type(arg)}")

    # TODO generate shape constraint at here?
    params = OpParams(
        n_args=len(args),
        args_types=types,
        args=args,
    )
    return params


def match(user_args, op_params: OpParams):
    if not isinstance(user_args, tuple):
        raise TypeError(
            f"user_args must be a tuple, but got {type(user_args)}")
    if not isinstance(op_params, OpParams):
        raise TypeError(
            f"op_params must be an OpParams, but got {type(op_params)}")

    if len(user_args) != op_params.n_args:
        return False

    for i, arg in enumerate(user_args):
        # type check
        if op_params.args_types[i] == 'tensor' and not isinstance(arg, Tensor):
            return False
        if op_params.args_types[i] == 'str' and not isinstance(arg, str):
            return False
        if op_params.args_types[i] == 'int' and not isinstance(arg, int):
            return False

        # py_val check
        if isinstance(arg, str):
            if arg != op_params.args[i]:
                return False
        if isinstance(arg, int):
            if arg != op_params.args[i]:
                return False

        # tensor check
        if isinstance(arg, Tensor):
            if arg.device != op_params.args[i].device:
                return False
            if arg.dtype != op_params.args[i].dtype:
                return False

            # shape check
            sym_shape = op_params.args[i].shape
            concrete_shape = arg.shape
            # TODO check shape

    return True
