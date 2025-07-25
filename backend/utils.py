import contextlib
import io
import shutil
import subprocess
import sys

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
        subprocess.run(["which", cmd], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        pass
    if shutil.which(cmd):
        return True
    try:
        subprocess.run([cmd, "--help"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


backend = None
def get_current_backend():
    global backend
    if backend is not None:
        return backend
    elif command_exists("npu-smi"):
        backend = 'ascend'
    else:
        backend = None
    return backend


def init_dicp_driver():
    backend = get_current_backend()
    if backend is not None:
        from triton.backends.dicp_triton.driver import DICPDriver
        from triton.runtime.driver import driver
        driver.set_active(DICPDriver(backend))
