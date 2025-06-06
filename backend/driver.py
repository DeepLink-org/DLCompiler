import functools
import hashlib
import importlib
import os
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

import setuptools
import torch

from triton.runtime.cache import get_cache_manager
from triton.runtime.build import quiet
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
# from backend.mlu.mlu_driver import MLUUtils, MLUDriver

def build_for_backend(name, src, srcdir):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    ret = subprocess.check_call([cc, src, f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-o", so])
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
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-O3'],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
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
        if not hasattr(cls, 'instance'):
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
        if(self.__initialized): return
        self.__initialized = True
        super().__init__()
        if target == "mlu":
            from triton.backends.dicp_triton.mlu import MluLauncher, MLUUtils
            self.target = "mlu"
            self.utils = MLUUtils()
            self.launcher_cls = MluLauncher
        elif target == "maca":
            from triton.backends.dicp_triton.maca import MacaLauncher, MacaUtils
            self.target = "maca"
            self.utils = MacaUtils()
            self.launcher_cls = MacaLauncher
        elif target == "npu":
            from triton.backends.dicp_triton.npu import NpuLauncher, NpuUtils
            self.target = "npu"
            self.utils = NpuUtils()
            self.launcher_cls = NpuLauncher
        else:
            self.target = "dicp"
           
    
    def __new__(cls, target=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DICPDriver, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    @staticmethod
    def is_active(self):
        if self.target == "npu":
            def test_npucompiler():
                from triton.backends.ascend.utils import _get_bisheng_path
                npucompiler = _get_bisheng_path()
                targets = subprocess.check_output([npucompiler, "-print-targets"]).decode().strip().split()
                return "hiipu64" in targets
            try:
                return test_npucompiler()
            except Exception as e_npucompiler:
                import warnings
                red = "\x1b[31;20m"
                reset = "\x1b[0m"
                warnings.warn(red + str(e_npucompiler) + reset)
                return False
        return True

    def get_device_capability(self):
        if self.target == "mlu":
            return ("mlu", 0)
        elif self.target == "maca":
            return ("maca", 0)
        elif self.target == "npu":
            return ("npu", 0)
        return ("dicp", 0)

    def get_current_stream(self, device):
        if self.target == "mlu":
            if device is None:
                device = self.get_current_device()
            return torch.mlu.current_stream(device).mlu_stream
        elif self.target == "maca":
            if device is None:
                device = self.get_current_device()
            return torch.cuda.current_stream(device).cuda_stream
        elif self.target == "npu":
            if device is None:
                device = self.get_current_device()
            return torch.npu.current_stream(device).npu_stream
        return None

    def get_current_device(self):
        # dicp doesn't have a device to return. Return something.
        if self.target == "mlu":
            return torch.mlu.current_device()
        elif self.target == "maca":
            return torch.cuda.current_device()
        elif self.target == "npu":
            import torch_npu
            return torch.npu.current_device()
        return "dicp"

    def set_current_device(self, device):
        # dicp doesn't have a device to set
        if self.target == "mlu":
            return torch.mlu.set_device(device)
        elif self.target == "maca":
            return torch.cuda.set_device(device)
        elif self.target == "npu":
            import torch_npu
            return torch.npu.set_device(device)
        #assert device == "dicp"
        return

    def get_current_target(self):
        if self.target == "mlu":
            return GPUTarget("mlu", "x86", 32)
        elif self.target == "maca":
            return GPUTarget("maca", "x86", 32)
        elif self.target == "npu":
            backend = "npu"
            arch = self.utils.get_arch()
            warp_size = 0
            return GPUTarget(backend, arch, warp_size)
        return GPUTarget("dicp", "x86", 32)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
    
    def get_device_interface(self):
        if self.target == "npu":
            import torch
            return torch.npu
        
    def get_empty_cache_for_benchmark(self):
        if self.target == "npu":
            import torch
            cache_size = 192 * 1024 * 1024
            return torch.empty(cache_size // 4, dtype=torch.int, device='npu')
        
