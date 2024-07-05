import sys
import textwrap

import ast
import importlib
import inspect
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser


def dynamic_import_and_parse(path_to_src, nn_module_name):
    # dynamically exec the module.py file
    arg_path = Path(path_to_src)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nn_mod = getattr(mod, nn_module_name)
    forward_method = getattr(nn_mod, 'forward')

    forward_src = inspect.getsource(forward_method)
    forward_src = textwrap.dedent(forward_src)
    return forward_src


def extract_shape_dtype(annotation):
    # Extracting the dtype and shape from the annotation string
    dtype, shape = annotation.split('[')
    shape = shape.rstrip(']')  # Remove the closing bracket
    return dtype, shape


def parse_fx_readable(source_code):
    tensor_info = {}
    op_info = defaultdict(list)

    # Process the function body.
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Handle function arguments.
            for arg_node in node.args.args:
                if arg_node.annotation:
                    tensor_name = arg_node.arg
                    dtype, shape = extract_shape_dtype(
                        ast.get_source_segment(source_code,
                                               arg_node.annotation))
                    if dtype and shape:
                        tensor_info[tensor_name] = {
                            'dtype': dtype,
                            'shape': shape
                        }

        elif isinstance(node, ast.AnnAssign):
            # Handle a stmt
            ## fisrt deal with the output tensor
            target_tensor_name = node.target.id
            if isinstance(node.annotation, ast.Constant):
                dtype, shape = extract_shape_dtype(node.annotation.s)
                if dtype and shape:
                    tensor_info[target_tensor_name] = {
                        'dtype': dtype,
                        'shape': shape
                    }
            else:
                raise ValueError(f"Unsupported annotation: {node.annotation}")

            ## then deal with the function and args
            if isinstance(node.value, ast.Call):
                call = node.value

                # get the full function name, e.g. `torch.ops.xxx`
                op_names = []
                attr = call.func
                while isinstance(attr, ast.Attribute):
                    op_names.append(attr.attr)
                    attr = attr.value

                assert isinstance(attr, ast.Name)
                op_names.append(attr.id)
                op_names = '.'.join(op_names[::-1])

                # get the args
                args = call.args
                kwargs = call.keywords

                for arg in args:
                    assert isinstance(arg, ast.Name)
                args_names = [arg.id for arg in args]
                assert len(
                    kwargs) == 0, f"Unexpected keyword arguments: {kwargs}"

                # store
                op_info[op_names].append(
                    (target_tensor_name, args_names, kwargs))
            else:
                raise ValueError(f"Unsupported value: {node.value}")

    return tensor_info, op_info


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        type=str,
        default=
        '/heguoliang/triton_deeplink/custom_bench/fx_graphs/fx_graph_readable.py',
    )
    parser.add_argument(
        "-m",
        type=str,
        default='g',
    )
    args = parser.parse_args()

    forward_src = dynamic_import_and_parse(args.p, args.m)
    tensor_info, op_info = parse_fx_readable(forward_src)

    print("Tensor Information:")
    print(tensor_info)

    print("\nOperation Inputs:")
    print(op_info)
