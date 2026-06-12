import contextlib
import io
import functools
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from triton.runtime.cache import get_cache_manager, get_dump_manager

import pybind11


# ---------------------------------------------------------------------------
# Backend function dispatch (torch_npu only, aligned with triton-ascend)
# ---------------------------------------------------------------------------


_BACKEND_POLICY = None


def get_backend_func(name, *args, **kwargs):
    global _BACKEND_POLICY
    if _BACKEND_POLICY is None:
        import torch
        import torch_npu

        _BACKEND_POLICY = "torch_npu"
    return _TORCH_NPU_BACKEND_FUNCS[name](*args, **kwargs)


def _version_hash():
    import torch
    import torch_npu

    return [torch.version.git_version, torch_npu.version.git_version]


def _cxx_abi():
    import torch

    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


def _header_file(enable_taskqueue):
    taskqueue_include = (
        "#include <torch_npu/csrc/framework/OpCommand.h>" if enable_taskqueue else ""
    )
    return f"""#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
{taskqueue_include}"""


def _allocate_memory(size, stream):
    return f"workspace_addr_ptr = const_cast<void *>(at::empty({size}, at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte)).storage().data());"


def _allocate_sync_block_lock(size, stream):
    return f"syncBlockLock_ptr = const_cast<void *>(at_npu::native::allocate_workspace({size}, {stream}).storage().data());"


def _pre_launch(first_call):
    return ""


def _async_launch(func):
    return f"""at_npu::native::OpCommand cmd;
    cmd.Name(name.c_str()).SetCustomHandler({func}).Run();"""


def _get_cc_cmd(build_pch):
    import torch
    import torch_npu

    torch_path = os.path.dirname(os.path.realpath(torch.__file__))
    torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
    cc_cmd = [
        f"-I{os.path.join(torch_path, 'include')}",
        f"-I{os.path.join(torch_npu_path, 'include')}",
        f"-D_GLIBCXX_USE_CXX11_ABI={_cxx_abi()}",
    ]
    if not build_pch:
        cc_cmd += [
            f"-L{os.path.join(torch_npu_path, 'lib')}",
            "-ltorch_npu",
        ]
    return cc_cmd


def _get_tensor_params_shape(*args):
    import torch

    tensor_params = [arg for arg in args if isinstance(arg, torch.Tensor)]
    tensor_params_shape = []
    for t in tensor_params:
        tensor_params_shape.append([s for s in t.shape])
    return tensor_params_shape


_TORCH_NPU_BACKEND_FUNCS = {
    "version_hash": _version_hash,
    "cxx_abi": _cxx_abi,
    "header_file": _header_file,
    "allocate_memory": _allocate_memory,
    "allocate_sync_block_lock": _allocate_sync_block_lock,
    "pre_launch": _pre_launch,
    "async_launch": _async_launch,
    "get_cc_cmd": _get_cc_cmd,
    "get_tensor_params_shape": _get_tensor_params_shape,
}


# ---------------------------------------------------------------------------
# Quiet context manager
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------

TRITON_PROFILER_REGISTERED = False

replace_dicp_ir = os.environ.get("DLC_REPLACE_DICP_IR_FILE", None)
if os.environ.get("TRITON_DEBUG", "0") == "1" or replace_dicp_ir is not None:
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    dump_dir = "./tmp"
    os.environ["TRITON_DUMP_DIR"] = os.environ.get("TRITON_DUMP_DIR", dump_dir)
    if os.path.exists(dump_dir):
        print(f"Directory **{dump_dir}** exists. Deleting the entire directory...")
        shutil.rmtree(dump_dir)


def _dump_stage_ir(ir_str, key, filename, cmd_list=None):
    dump_manager = get_dump_manager(key)
    print("Dumping intermediate results to " + dump_manager.cache_dir + "/" + filename)
    dump_manager.put(ir_str, filename, binary=False)
    if cmd_list:
        cmd_list[1] = dump_manager.cache_dir + "/" + filename
        print(f"DEBUG dump ir command: {cmd_list}")


# ---------------------------------------------------------------------------
# BishengIR path initialisation (local _C/bishengir override)
# ---------------------------------------------------------------------------

local_bishengir_path = os.path.join(os.path.dirname(__file__), "../../_C/bishengir")
bisheng_install_path = os.environ.get("BISHENG_INSTALL_PATH", None)
if (
    bisheng_install_path is None
    and os.path.exists(local_bishengir_path)
    and os.path.isdir(local_bishengir_path)
    and os.path.exists(os.path.join(local_bishengir_path, "bishengir-compile"))
    and os.path.exists(os.path.join(local_bishengir_path, "bishengir-hivm-compile"))
    and os.path.exists(os.path.join(local_bishengir_path, "bishengir-opt"))
    and os.path.exists(os.path.join(local_bishengir_path, "hivmc"))
):
    os.environ["BISHENG_INSTALL_PATH"] = local_bishengir_path
    os.environ["PATH"] = local_bishengir_path + os.pathsep + os.environ["PATH"]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _get_npucompiler_path():
    ascend_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    npu_compiler_path = os.path.join(ascend_dir, "../../_C/bishengir/bishengir-compile")
    if os.path.exists(npu_compiler_path) and os.access(npu_compiler_path, os.X_OK):
        npuir_env_path = os.path.dirname(npu_compiler_path)
        env["PATH"] = npuir_env_path + os.pathsep + env["PATH"]
    else:
        npu_compiler_path = shutil.which("bishengir-compile")
        if npu_compiler_path is None:
            npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", None)
            if npu_compiler_root is None:
                raise EnvironmentError(
                    "Couldn't find executable bishengir-compile or TRITON_NPU_COMPILER_PATH."
                )
            npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    return os.path.abspath(npu_compiler_path), env


def _get_bishengir_opt_path():
    ascend_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    bishengir_opt_path = os.path.join(ascend_dir, "../../_C/bishengir/bishengir-opt")
    if os.path.exists(bishengir_opt_path) and os.access(bishengir_opt_path, os.X_OK):
        npuir_env_path = os.path.dirname(bishengir_opt_path)
        env["PATH"] = npuir_env_path + os.pathsep + env["PATH"]
    else:
        bishengir_opt_path = shutil.which("bishengir-opt")
        if bishengir_opt_path is None:
            bishengir_opt_root = os.getenv("TRITON_NPU_COMPILER_PATH", None)
            if bishengir_opt_root is None:
                raise EnvironmentError(
                    "Couldn't find executable bishengir-opt or TRITON_NPU_COMPILER_PATH"
                )
            bishengir_opt_path = os.path.join(bishengir_opt_root, "bishengir-opt")
    return os.path.abspath(bishengir_opt_path), env


def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", None)
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bisheng or TRITON_NPU_COMPILER_PATH"
            )
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path


@functools.lru_cache(None)
def _get_ascend_path() -> Path:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError(
            "ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first"
        )
    return Path(path)


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


def _is_ascend_sanitizer_enabled() -> bool:
    return os.getenv("TRITON_ENABLE_SANITIZER", "false").lower() in ("true", "1")


def _is_auto_map_parallel_blocks_enabled() -> bool:
    return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")


def get_ascend_arch_from_env():
    return os.getenv("TRITON_ASCEND_ARCH", "")


def is_ffts_supported(arch: str) -> bool:
    if is_compile_on_910_95:
        return False
    if arch in ("Ascend910A", "Ascend310B4"):
        return False
    return True


def force_disable_ffts() -> bool:
    if is_compile_on_910_95:
        return True
    return os.getenv("TRITON_DISABLE_FFTS", "false").lower() in ("true", "1")


# ---------------------------------------------------------------------------
# C++ compilation helpers
# ---------------------------------------------------------------------------


def _check_cxx11_abi():
    return get_backend_func("cxx_abi")


def _get_cxx():
    cxx = os.environ.get("CC")
    if cxx is not None:
        return cxx
    clangxx = shutil.which("clang++")
    gxx = shutil.which("g++")
    cxx = clangxx if clangxx is not None else gxx
    if cxx is None:
        raise RuntimeError("Failed to find C++ compiler")
    return cxx


def _get_cxx_precompiled(header_path):
    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        if clangxx is not None:
            return [clangxx, "-include", header_path]
        elif gxx is not None:
            return [gxx]
        else:
            raise RuntimeError("Failed to find C++ compiler")
    return [cxx]


def _precompile_npu_hash(header_src):
    cxx = _get_cxx()
    py_version = sys.version
    asc_path = str(_get_ascend_path())
    version_txt = [header_src, cxx, py_version, asc_path]
    version_txt += get_backend_func("version_hash")
    return hashlib.sha256("_".join(version_txt).encode("utf-8")).hexdigest()


def _precompile_npu_ext(header_path, gch_path):
    cc_cmd = [_get_cxx(), "-x", "c++-header", header_path]
    cc_cmd += ["-w"]

    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]

    asc_path = _get_ascend_path()
    rt_path = os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h")
    if not os.path.exists(rt_path):
        cc_cmd += [
            f"-I{os.path.join(asc_path, 'pkg_inc')}",
            f"-I{os.path.join(asc_path, 'pkg_inc/profiling')}",
        ]

    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
    ]

    cc_cmd += get_backend_func("get_cc_cmd", build_pch=True)

    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", gch_path]

    result = subprocess.run(cc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to precompile {gch_path}, error: {result.stderr}, cmd={cc_cmd}"
        )
    return header_path


def _precompile_npu_ext_with_lock(header_src, enable_precompile):
    import fcntl

    precompile_hash = _precompile_npu_hash(header_src)
    cache = get_cache_manager(precompile_hash)
    gch_path = cache.get_file("precompiled.h.gch")
    header_path = cache.get_file("precompiled.h")

    if enable_precompile:
        if header_path is not None and gch_path is not None:
            return header_path
    else:
        if header_path is not None:
            return header_path

    cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip()
    lock_path = os.path.join(cache_dir, f"{precompile_hash}.lock")
    with open(lock_path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            header_path = cache.get_file("precompiled.h")
            if enable_precompile:
                gch_path = cache.get_file("precompiled.h.gch")
                if header_path is not None and gch_path is not None:
                    return header_path
            else:
                if header_path is not None:
                    return header_path
            header_path = cache.put(header_src, "precompiled.h", binary=False)
            if not enable_precompile:
                return header_path
            src_dir = os.path.dirname(header_path)
            gch_path = os.path.join(src_dir, "precompiled.h.gch")
            _precompile_npu_ext(header_path, gch_path)
            return header_path
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _build_npu_ext(
    obj_name: str, header_path, src_path, *, kernel_launcher="torch", precompile=False
) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    src_dir = os.path.dirname(src_path)
    so_path = os.path.join(src_dir, f"{obj_name}{suffix}")

    if precompile:
        cc_cmd = _get_cxx_precompiled(header_path)
        cc_cmd += [src_path]
    else:
        cc_cmd = [_get_cxx(), src_path]

    cc_cmd += ["-w"]

    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]

    asc_path = _get_ascend_path()
    if header_path is not None:
        cc_cmd += [f"-I{os.path.dirname(header_path)}"]

    rt_path = os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h")
    if not os.path.exists(rt_path):
        cc_cmd += [
            f"-I{os.path.join(asc_path, 'pkg_inc')}",
            f"-I{os.path.join(asc_path, 'pkg_inc/profiling')}",
        ]

    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]

    if kernel_launcher:
        cc_cmd += get_backend_func("get_cc_cmd", build_pch=False)

    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-Winvalid-pch", "-o", so_path]

    result = subprocess.run(cc_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return so_path
    if "precompiled.h.gch" in result.stderr:
        return _build_npu_ext(
            obj_name,
            header_path,
            src_path,
            kernel_launcher=kernel_launcher,
            precompile=False,
        )
    raise RuntimeError(
        f"Failed to compile {src_path}, error: {result.stderr}, cmd={cc_cmd}"
    )


# ---------------------------------------------------------------------------
# Type mapping for msprof tensor reporting
# ---------------------------------------------------------------------------


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        "i1": 12,
        "i4": 29,
        "i8": 2,
        "i16": 6,
        "i32": 3,
        "i64": 9,
        "u1": 30,
        "u8": 4,
        "u16": 7,
        "u32": 8,
        "u64": 10,
        "fp16": 1,
        "bf16": 27,
        "fp32": 0,
        "fp64": 11,
        "fp8e5": 35,
        "fp8e4nv": 36,
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")
    return MAP_SIGTYPE_TO_INT[sigty]


# ---------------------------------------------------------------------------
# BishengIR API detection
# ---------------------------------------------------------------------------


def _check_bishengir_api_change() -> bool:
    bishengir_path, _ = _get_npucompiler_path()
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and "limit-auto-multi-buffer-buffer" in result.stdout:
            return True
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# Additional feature flags
# ---------------------------------------------------------------------------


def _is_debug_line_info_disabled() -> bool:
    return os.getenv("TRITON_DISABLE_LINE_INFO", "true").lower() in ("true", "1")


def _enable_print_ub_bits() -> bool:
    return os.getenv("ENABLE_PRINT_UB_BITS", "false").lower() in ("true", "1")


def _enable_dump_memory_info() -> bool:
    return os.getenv("TRITON_MEMORY_DISPLAY", "false").lower() in ("true", "1")


def _enable_msdebug() -> bool:
    return os.getenv("LLVM_EXTRACT_DI_LOCAL_VARIABLES", "false").lower() in (
        "true",
        "1",
    )


def _enable_unpublished_feature() -> bool:
    return os.getenv("ENABLE_UNPUBLISHED_FEATURE", "false").lower() in ("true", "1")


# ---------------------------------------------------------------------------
# 910_95 device detection
# ---------------------------------------------------------------------------


def _get_ascend_devices():
    import glob as _glob

    devices = []
    pci_path = "/sys/bus/pci/devices/*"
    for dev in _glob.glob(pci_path):
        try:
            vendor_path = os.path.join(dev, "vendor")
            device_path = os.path.join(dev, "device")
            if os.path.exists(vendor_path):
                with open(vendor_path, "r") as f:
                    vendor = f.read().strip()
                if vendor == "0x19e5" and os.path.exists(device_path):
                    with open(device_path, "r") as f:
                        device = f.read().strip()
                        devices.append(device)
        except (IOError, OSError):
            continue
    return devices


def _check_npu_smi_device():
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=100,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            return (
                "ascend910_95" in output
                or "ascend950" in output
                or "910_958b" in output
            )
        return False
    except Exception:
        return False


_ascend_devices = _get_ascend_devices()
_pci_condition = any("0xd806" in dev for dev in _ascend_devices)
_npu_smi_condition = _check_npu_smi_device()
is_compile_on_910_95 = _pci_condition or _npu_smi_condition


# ---------------------------------------------------------------------------
# CANN version & libdevice helpers
# ---------------------------------------------------------------------------


def get_machine_arch():
    ARCHITECTURE_ALIASES = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "i386": "x86_64",
        "i686": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "armv7l": "aarch64",
        "armv8l": "aarch64",
        "arm": "aarch64",
    }
    system_arch = platform.machine()
    return ARCHITECTURE_ALIASES.get(system_arch, system_arch)


def get_cann_version():
    ascend_path = _get_ascend_path()
    arch = get_machine_arch()
    cann_version_file_path = os.path.join(
        ascend_path, arch + "-linux", "ascend_toolkit_install.info"
    )
    if not os.path.exists(cann_version_file_path):
        cann_version_file_path = os.path.join(
            ascend_path, arch + "-linux", "ascend_all_cann_install.info"
        )
    version = ""
    innerversion = ""
    with open(cann_version_file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("version="):
                version = line.split("=")[1]
            elif line.startswith("innerversion="):
                innerversion = line.split("=")[1]
    if version and innerversion:
        return "CANN-" + version + "-" + innerversion
    if version:
        return "CANN-" + version
    raise ValueError("get_cann_version is empty!")


def triton_enable_libdevice_simt():
    enable_libdevice_simt = os.getenv("TRITON_ENABLE_LIBDEVICE_SIMT", False)
    return enable_libdevice_simt


def _check_bishengir_able_save_ir() -> bool:
    bishengir_path, _ = _get_npucompiler_path()
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and "save-linked-ir" in result.stdout:
            return True
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
