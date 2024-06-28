import importlib.util
import os


def import_all_modules_from_folder(folder_path):
    """
    dynamically import all Python modules under this folder

    Note that all kernels implementation under this folder must be registered a call interface
    """
    # Ensure the path is absolute
    folder_path = os.path.abspath(folder_path)

    # Check if the provided path is a directory and exists
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(
            f"The provided path '{folder_path}' is not a valid directory.")

    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter out only Python files (.py extension) and ignore __init__.py if present
    python_files = [
        file for file in files
        if file.endswith('.py') and file != '__init__.py'
    ]

    # Import each Python file as a module
    for file in python_files:
        # Construct the module name by removing the .py extension
        module_name = file[:-3]

        # Use importlib.util.spec_from_file_location to create a spec for the module
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(folder_path, file))

        # Create the module from the spec
        module = importlib.util.module_from_spec(spec)

        # Execute the module
        spec.loader.exec_module(module)

        # print(f"Successfully imported kernel module: {module_name}")


# Usage example
folder_path = dir_path = os.path.dirname(os.path.realpath(__file__))

import_all_modules_from_folder(folder_path)
