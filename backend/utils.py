import contextlib
import io
import functools
from pathlib import Path
import os
import shutil
import subprocess
import sys
from triton.runtime.cache import get_dump_manager


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def command_exists(cmd):
    try:
        subprocess.run(
            ["which", cmd], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        pass
    if shutil.which(cmd):
        return True
    try:
        subprocess.run([cmd, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


backend = None


def get_current_backend():
    global backend
    if backend is not None:
        return backend
    elif command_exists("npu-smi"):
        backend = "ascend"
    elif command_exists("cnmon"):
        backend = "mlu"
    elif command_exists("mx-smi"):
        backend = "maca"
    elif command_exists("nvidia-smi"):
        backend = "nvidia"
    else:
        backend = None
    return backend


def init_dicp_driver():
    backend = get_current_backend()
    if backend is not None:
        from triton.backends.dicp_triton.driver import DICPDriver
        from triton.runtime.driver import driver

        driver.set_active(DICPDriver(backend))
    else:
        raise RuntimeError("No supported backend found.")


def _dump_stage_ir(ir_str, key, filename, cmd_list=None):
    dump_manager = get_dump_manager(key)
    print("Dumping intermediate results to " + dump_manager.cache_dir + "/" + filename)
    dump_manager.put(ir_str, filename, binary=False)
    if cmd_list:
        cmd_list[1] = dump_manager.cache_dir + "/" + filename
        print(f"DEBUG dump ir command: {cmd_list}")
