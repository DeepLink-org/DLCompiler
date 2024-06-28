from dataclasses import dataclass, field, astuple
from typing import Any

import torch

from dlblas.symbolic_var import Tensor
from dlblas.utils.logger import get_logger

logger = get_logger(__name__)


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

    def bench(self, *args: Any, **kwargs: Any) -> Any:
        return self.bench_fn(*args, **kwargs)


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

    # TODO generate shape constraint at register time?
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

    concrete_shapes = []
    sym_shapes = []
    for i, arg in enumerate(user_args):
        # type check
        if op_params.args_types[i] == 'tensor' and not isinstance(
                arg, torch.Tensor):
            # user would want to pass with torch.Tensor
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
        if isinstance(arg, torch.Tensor):
            if arg.dtype != op_params.args[i].dtype:
                return False

            # TODO consider add a logging for this
            dev_str = str(arg.device)
            op_dev_str = op_params.args[i].device
            if not dev_str.startswith(op_dev_str):
                return False

            # appending shape for check
            sym_shape = op_params.args[i].shape
            concrete_shape = arg.shape

            sym_shapes.append(sym_shape)
            concrete_shapes.append(concrete_shape)

    # shape check
    if violate_symbolic_constraints(concrete_shapes, sym_shapes):
        return False

    return True


def violate_symbolic_constraints(concrete_shapes, sym_shapes):
    # NOTE this could be expensive,
    # we should probably improve if noticable slowdown

    # for now, we just check when symbolic variables are the same
    # whether the corresponding concrete values are the same
    sym2loc = {}
    for i in range(len(sym_shapes)):
        for j in range(len(sym_shapes[i])):
            symbol = sym_shapes[i][j]
            if symbol in sym2loc:
                sym2loc[symbol].append((i, j))
            else:
                sym2loc[symbol] = [(i, j)]

    for sym, locs in sym2loc.items():
        first_loc = locs[0]
        first_val = concrete_shapes[first_loc[0]][first_loc[1]]

        for loc in locs:
            if concrete_shapes[loc[0]][loc[1]] != first_val:
                return True

    return False
