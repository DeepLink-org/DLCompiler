import sys
import textwrap

import ast
import importlib
import inspect
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

import torch

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
    # parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    # parser.add_argument("--num-stages", "-ns", type=int, default=3,
    #                     help="Number of stages (meta-parameter of the kernel)")
    args = parser.parse_args()

    arg_path = Path(args.p)

    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nn_mod = getattr(mod, args.m)
    forward_method = getattr(nn_mod, 'forward')

    # Inspect the forward method to get the parameter names
    forward_input_args = inspect.signature(forward_method).parameters

    # TODO how to get args shape? then
    dummy_tensors = {name: torch.randn(1) for name in forward_input_args}

    # execute src

    ## convert to fake tensors
    ## iterate over torch ops, execute it on fake tensor
    ## store output to a dict
    forward_src = inspect.getsource(forward_method)
    forward_src = textwrap.dedent(forward_src)
    tree: ast.Module = ast.parse(forward_src)

    # Create a mapping from variable names to tensors
    tensor_dict = defaultdict(lambda: torch.randn(1))

    # Define a function to evaluate expressions
    def eval_expr(expr_node):
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
    def exec_assign(node):
        value = eval_expr(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                tensor_dict[target.id] = value
            else:
                raise NotImplementedError(
                    f"Assignment target {type(target)} not implemented")

    # Walk through the AST and execute assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            exec_assign(node)
