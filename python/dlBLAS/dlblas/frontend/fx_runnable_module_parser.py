import sys
import textwrap

import ast
import importlib
import inspect
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

import torch


# Define a function to evaluate expressions
def eval_expr(expr_node, tensor_dict):
    if isinstance(expr_node, ast.Name):
        # Variable reference
        return tensor_dict[expr_node.id]
    elif isinstance(expr_node, ast.Call):

        # Function call
        # func_name = expr_node.func.attr
        func = expr_node.func
        assert isinstance(func, ast.Name)
        func_name = func.id

        args = [eval_expr(arg) for arg in expr_node.args]
        kwargs = {kw.arg: eval_expr(kw.value) for kw in expr_node.keywords}

        # TODO; convert this to appropriate op, and
        aten_op = getattr(torch.ops.aten, func_name)
        output = aten_op(*args, **kwargs)
        return output

    elif isinstance(expr_node, ast.Constant):
        # Constant value
        return expr_node.value
    else:
        raise NotImplementedError(
            f"Expression type {type(expr_node)} not implemented")


# Define a function to execute assignments
def exec_assign(node, tensor_dict):
    value = eval_expr(node.value)
    for target in node.targets:
        if isinstance(target, ast.Name):
            tensor_dict[target.id] = value
        else:
            raise NotImplementedError(
                f"Assignment target {type(target)} not implemented")


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        type=str,
        default=
        '/heguoliang/triton_deeplink/custom_bench/llama_fx/llama_fx_graph/prefill/module.py',
    )
    parser.add_argument(
        "-m",
        type=str,
        default='cddb4645ik6eir7vfcqcgsfyc3fm2imkftwug55kk2pmqzxkowk6',
    )
    args = parser.parse_args()

    # dynamically exec the module.py file
    arg_path = Path(args.p)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nn_mod = getattr(mod, args.m)
    forward_method = getattr(nn_mod, 'forward')

    # Inspect the forward method to get the parameter names
    forward_input_args = inspect.signature(forward_method).parameters

    # Create a mapping from variable names to tensors
    # TODO how to get args shape? then write into tensor_dict
    dummy_tensors = {name: torch.randn(1) for name in forward_input_args}
    tensor_dict = defaultdict(lambda: torch.randn(1))

    # execute function line-by-line
    ## convert to fake tensors
    ## iterate over torch ops, execute it on fake tensor
    ## store output to a dict
    forward_src = inspect.getsource(forward_method)
    forward_src = textwrap.dedent(forward_src)
    tree: ast.Module = ast.parse(forward_src)

    # Walk through the AST and execute assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # assume forward has no control flow; only sequential torch ops
            exec_assign(node)
