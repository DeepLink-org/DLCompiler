from pathlib import Path
import tempfile
import os
import subprocess
import sysconfig
from typing import Optional
import functools
import hashlib
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.dicp_triton.utils import get_current_backend

import importlib
import shutil

import setuptools
import sys
import contextlib
import io


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def build_for_backend(name, src, srcdir):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so = os.path.join(srcdir, "{name}{suffix}".format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError(
                "Failed to find C compiler. Please specify via CC environment variable."
            )
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    ret = subprocess.check_call(
        [cc, src, f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-o", so]
    )
    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = []
    include_dirs = [srcdir]
    libraries = []
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language="c",
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ["-O3"],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ["build_ext"]
    args.append("--build-temp=" + srcdir)
    args.append("--build-lib=" + srcdir)
    args.append("-q")
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so


class DICUtils:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DICUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "extension_backend.c")).read_text()
        key = hashlib.sha256(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "ext_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_for_backend("ext_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("ext_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class DICPDriver(DriverBase):
    def __init__(self, target=None):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__()
        if target == "mlu":
            from triton.backends.dicp_triton.mlu import BangLauncher, BangUtils

            self.target = "mlu"
            self.utils = BangUtils()
            self.launcher_cls = BangLauncher
            import torch
            import torch_mlu

            self.get_current_device = torch.mlu.current_device
            self.set_current_device = torch.mlu.set_device
            self.get_current_stream = lambda idx: torch.mlu.current_stream(
                idx
            ).mlu_stream
            self.is_linear_pointer = lambda ptr, device: self.utils.is_linear_pointer(
                ptr, device
            )
        elif target == "maca":
            from triton.backends.dicp_triton.maca import MacaLauncher, MacaUtils

            self.target = "maca"
            self.utils = MacaUtils()
            self.launcher_cls = MacaLauncher
        elif target == "ascend":
            from triton.backends.dicp_triton.npu import NPULauncher, NPUUtils

            self.target = "ascend"
            self.utils = NPUUtils()
            self.launcher_cls = NPULauncher
        elif target == "nvidia":
            from triton.backends.nvidia.driver import CudaLauncher, CudaUtils

            self.target = "nvidia"
            self.utils = CudaUtils()
            self.launcher_cls = CudaLauncher
        else:
            self.target = "dicp"

    def __new__(cls, target=None):
        if not hasattr(cls, "instance"):
            cls.instance = super(DICPDriver, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    @staticmethod
    def is_active():
        return True

    @classmethod
    def is_active(self):
        try:
            current_backend = get_current_backend()
            if current_backend == "ascend":

                def test_npucompiler():
                    from triton.backends.dicp_triton.npu import _get_bisheng_path

                    npucompiler = _get_bisheng_path()
                    targets = (
                        subprocess.check_output([npucompiler, "-print-targets"])
                        .decode()
                        .strip()
                        .split()
                    )
                    return "hiipu64" in targets

                try:
                    return test_npucompiler()
                except Exception as e_npucompiler:
                    import warnings

                    red = "\x1b[31;20m"
                    reset = "\x1b[0m"
                    warnings.warn(red + str(e_npucompiler) + reset)
                    return False
            elif self.target == "muxi":
                import torch

                return True
        except Exception as e:
            try:
                import torch

                return True
            except Exception as e:
                raise RuntimeError(f"dicp triton exception:{e}")
        return True

    def launch_as_union_task(self, device, grid):
        if self.target == "mlu":
            import math

            cluster_num = self.utils.get_device_properties(device).get("cluster_num")
            core_num_per_cluster = self.utils.get_device_properties(device).get(
                "core_num_per_cluster"
            )
            total_cores = cluster_num * core_num_per_cluster
            return (
                grid[0] % core_num_per_cluster == 0 and math.prod(grid) <= total_cores
            )

    def get_device_capability(self):
        if self.target == "mlu":
            return ("mlu", 0)
        elif self.target == "maca":
            return ("maca", 0)
        elif self.target == "ascend":
            return ("ascend", 0)
        elif self.target == "nvidia":
            capability = torch.cuda.get_device_capability(self.get_current_device())
            return ("cuda", capability)
        return ("dicp", 0)

    def get_current_stream(self, device):
        import torch

        if self.target == "mlu":
            import torch_mlu

            if device is None:
                device = self.get_current_device()
            return torch.mlu.current_stream(device).mlu_stream
        elif self.target == "maca":
            if device is None:
                device = self.get_current_device()
            return torch.cuda.current_stream(device).cuda_stream
        elif self.target == "ascend":
            if device is None:
                device = self.get_current_device()
            return torch.npu.current_stream(device).npu_stream
        elif self.target == "nvidia":
            if device is None:
                device = self.get_current_device()
            return torch.cuda.current_stream(device).cuda_stream
        return None

    def get_current_device(self):
        import torch

        # dicp doesn't have a device to return. Return something.
        if self.target == "mlu":
            import torch_mlu

            return torch.mlu.current_device()
        elif self.target == "maca":
            return torch.cuda.current_device()
        elif self.target == "ascend":
            import torch_npu

            return torch.npu.current_device()
        elif self.target == "nvidia":
            return torch.cuda.current_device()
        return "dicp"

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def set_current_device(self, device):
        # dicp doesn't have a device to set
        if self.target == "mlu":
            return torch.mlu.set_device(device)
        elif self.target == "maca":
            return torch.cuda.set_device(device)
        elif self.target == "ascend":
            import torch_npu

            return torch.npu.set_device(device)
        elif self.target == "nvidia":
            return torch.cuda.set_device(device)
        return

    def get_current_target(self):
        if self.target == "mlu":
            device = self.get_current_device()
            capability = self.utils.get_device_properties(device).get("isa_version")
            # As compile func in compiler.py just support GPUTarget, and this type
            # can also represent MLU information, we will temporarily use GPUTarget here.
            return GPUTarget("mlu", capability, 0)
        elif self.target == "maca":
            return GPUTarget("maca", "x86", 32)
        elif self.target == "ascend":
            backend = "ascend"
            arch = self.utils.get_arch()
            warp_size = 0
            return GPUTarget(backend, arch, warp_size)
        elif self.target == "nvidia":
            device = self.get_current_device()
            capability = torch.cuda.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]
            warp_size = 32
            return GPUTarget("cuda", capability, warp_size)
        return GPUTarget("dicp", "x86", 32)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def get_device_interface(self):
        if self.target == "ascend":
            import torch

            return torch.npu
        elif self.target == "mlu":
            import torch

            return torch.mlu
        else:
            assert False, f"Not implemented for {self.target}"

    def get_empty_cache_for_benchmark(self):
        if self.target == "ascend":
            import torch

            cache_size = 192 * 1024 * 1024
            return torch.empty(cache_size // 4, dtype=torch.int, device="npu")
        elif self.target == "mlu":
            import torch

            # We maintain a buffer of 256 MB that we clear
            # before each kernel call to make sure that the L2 cache
            # doesn't contain any input data before the run
            cache_size = 256 * 1024 * 1024
            return torch.empty(int(cache_size // 4), dtype=torch.int, device="mlu")
        else:
            assert False, f"Not implemented for {self.target}"

    def get_active_torch_device(self):
        # todo: fix it.
        import torch

        return torch.device("cpu")

    def map_python_to_cpp_type(self, ty: str) -> str:
        if ty[0] == "*":
            return "void*"
        if ty == "constexpr":
            return "PyObject*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    @classmethod
    def clear_cache(self, cache):
        cache.zero_()
