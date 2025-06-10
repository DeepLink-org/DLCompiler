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
from triton.backends.dicp_triton.utils import _build_npu_ext, _check_cxx11_abi, convert_sigtype_to_int

from triton.backends.compiler import BaseBackend, GPUTarget, AttrsDescriptor, register_descriptor
from triton._C.libtriton import ir, passes
from triton.runtime import driver
from triton.runtime.cache import get_dump_manager
from dataclasses import dataclass
import functools
from typing import Any, Union, Tuple, Dict
from types import ModuleType
from pathlib import Path
import tempfile
import os
import subprocess
import hashlib
import ctypes
from typing import Optional

from triton.backends.dicp_triton.utils import downgrade_llir, _get_llvm_path, _get_mlir_path, _get_triton_adapter_opt_path, \
    _get_kernel_target, _get_npucompiler_path, _is_ascend_sanitizer_enabled

# from triton.backends.compiler import DICPOptions


# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    # return lambda lhsType, rhsType: (16, 16, 16)
    return lambda lhsType, rhsType: (1, 1, 1)

def make_ttir(mod, metadata, opt):
    if 'hash' not in metadata:
        metadata['hash'] = hashlib.md5(f"{mod}-{metadata}".encode()).hexdigest()
    # the same optimize pass for triton-ir as all other backends
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)
    if opt.debug:
        dump_manager = get_dump_manager(metadata['hash'])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary = False)

    return mod

def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
    # use triton_adapter to lower Triton-MLIR to linalg
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        cmd_list = [triton_adapter_opt_path, src_path,
            f'--triton-to-linalg=global-kernel=false named-ops={named_ops}',
            "-o", dst_path]
        if _is_ascend_sanitizer_enabled():
            cmd_list += ["--mlir-print-debuginfo"] # pass debug info

        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(dst_path).read_text(),
                             "kernel.ttadapter.mlir", binary = False)

        return Path(dst_path).read_text()


def linalg_to_llir(linalg: str, metadata, opt):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        llmlir_path = os.path.join(tmpdir, "kernel.llir.mlir")
        llir_path = os.path.join(tmpdir, "kernel.ll")
        Path(ttadapter_path).write_text(linalg)
        mlir_opt_path = _get_mlir_path("bin", "mlir-opt")
        # TritonAdapter-MLIR to LLVM-MLIR
        subprocess.check_call([mlir_opt_path, ttadapter_path,
            "--convert-linalg-to-affine-loops",
            "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine",
            "--convert-linalg-to-loops",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-complex-to-llvm",
            "--convert-vector-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            # Lowering memrefs creates more affine.apply ops.
            # Lowering these affine ops again creates further arith ops,
            # so we have to run these two passes again here.
            "--lower-affine",
            "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts",
            "-o",
            llmlir_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(llmlir_path).read_text(), "kernel.llir.mlir", binary = False)

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_mlir_path("bin", "mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path,
            "--mlir-to-llvmir",
            "-o",
            llir_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(llir_path).read_text(), "kernel.ll", binary = False)

        return Path(llir_path).read_text()

def llir_to_cpuasm(llir: str, metadata, opt):
    # add metadata at final stage
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, CPU backend requires no limit on shared memory size
    metadata['shared'] = 1
    # We can get a function name (C naming) from
    # LLVM-IR by getting the first "define void @".
    fn_name = llir.split("define void @")[1].split("(")[0].strip()
    metadata['name'] = fn_name + " cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        linked_path = os.path.join(tmpdir, "kernel_linked.ll")
        dst_path = os.path.join(tmpdir, "kernel.s")

        llir = downgrade_llir(llir)
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(llir, "kernel_downgrade.ll", binary = False)

        Path(src_path).write_text(llir)

        linker_path = _get_llvm_path("bin", "llvm-link")
        libclc_path = _get_llvm_path("lib", "clc", "libspirv-aarch64--.bc")
        subprocess.check_call([linker_path, src_path, libclc_path, "--only-needed", "-S", "-o", linked_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(linked_path).read_text(), "kernel_linked.ll", binary = False)

        llc_path = _get_llvm_path("bin", "llc")
        subprocess.check_call([llc_path, linked_path, "-o", dst_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(dst_path).read_text(), "kernel.s", binary = False)

        # Actually it's text-format assembly.  Use read_text().
        return Path(dst_path).read_text()


def linalg_to_bin_enable_npu_compile(linalg: str, metadata, opt):
    import re
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata['shared'] = 1
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata['mix_mode'] = re.search(r'mix_mode\s*=\s*"([^"]+)"', linalg).group(1)
    metadata['kernel_name'] = re.search(r'func\.func\s+@(\w+)', linalg).group(1)
    # Use while space to split kernel_name and mix_mode.
    # Check the function load_binary in npu_driver.py.
    metadata['name'] = metadata['kernel_name'] + " " + metadata['mix_mode']
    # remove the mix_mode attribute
    linalg = re.sub(r', mix_mode\s*=\s*"[^"]*"', '', linalg)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel_reloc.o")
        callback_path = os.path.join(tmpdir, "libkernel.so")
        multibuffer = metadata['multibuffer']
        _compile_option_list = [
            f"--enable-auto-multi-buffer={multibuffer}",
        ]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        npu_compiler_path = _get_npucompiler_path()
        if (npu_compiler_path.endswith("bishengir-compile")):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                "--enable-hivm-compile=true",
                "--enable-triton-kernel-compile=true",
            ]
        cmd_list = [npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file]
        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            callback_func = getattr(lib, metadata['kernel_name'] +
                                         "_infer_workspace_shape_function")
            callback_func.restype = ctypes.c_int64
            callback_func.argtypes = []
            metadata['workspace_size'] = callback_func()

        return Path(bin_path).read_bytes()

@dataclass(frozen=True)
class NPUOptions():
    debug: bool = False
    # sanitize_overflow: bool = False
    sanitize_overflow: bool = True
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = 2
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    enable_npu_compile: bool = True
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    multibuffer: bool = True

    stream: int = None

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = -1

    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()

@register_descriptor
class AscendAttrsDescriptor(AttrsDescriptor):

    def _add_backend_properties(self, params=None, values=None):
        if params is None or values is None:
            return

        for i in range(len(values)):
            if (params[i].is_constexpr):
                continue
            val = values[i]

            if hasattr(val, 'shape'):
                self.arg_properties[f"tt.shape_{i}"] = list(val.shape)
                self.property_values[f"tt.shape_{i}"] = 0
            else:
                # Scalar
                pass

    def get_shapes(self):
        shapes = {}
        for name, val in self.arg_properties.items():
            if name.startswith("tt.shape"):
                idx = int(name.split('_')[-1])
                shapes[idx] = val
        return shapes


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu' or target.backend == 'npu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if (target.backend == "cpu"):
            self.binary_ext = "cpuasm"
        elif (target.backend == "npu"):
            self.binary_ext = "npubin"

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == 'npu':
            args = {k: opts[k] for k in NPUOptions.__dataclass_fields__.keys() if k in opts}
            options = NPUOptions(**args)
        else:
            args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
            options = CPUOptions(**args)
        return options

    def pack_metadata(self, metadata):
        from triton.backends.dicp_triton.utils import TRITON_PROFILER_REGISTERED
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Ascend backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig, mix_mode = metadata.name.split()
        if (len(kernel_name_orig) > KERNEL_NAME_MAX_LEN):
            kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
            # import warnings
            # # red = "\x1b[31;20m"
            # # reset = "\x1b[0m"
            # warnings.warn(kernel_name_orig + " is truncated to " + kernel_name)
            # warnings.warn("because '" + kernel_name_orig + "' exceeds torchnpu profiler's length limit < 50")
        else:
            kernel_name = kernel_name_orig
        return {
            "kernel_name": kernel_name,
            "hash": metadata.hash,
            "debug": metadata.debug,
            "profiler_registered": TRITON_PROFILER_REGISTERED,
        }

    def get_codegen_implementation(self):
        # Note: a dict of functions is required to generate vendor-specific code piecies
        #       e.g. convert custom types like fp8e4b15
        codegen_fns = {
            "min_dot_size": min_dot_size(self.target)
        }
        return codegen_fns

    def load_dialects(self, ctx):
        pass

    def get_attrs_descriptor(self, params, args):
        return AscendAttrsDescriptor(params, args)

    def add_stages(self, stages, options):
        if self.target.backend == 'npu':
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            if options.enable_npu_compile:
                stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
                stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile(src, metadata, options)
            else:
                pass
        else:
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options)
            stages["llir"] = lambda src, metadata: linalg_to_llir(src, metadata, options)
            stages["cpuasm"] = lambda src, metadata: llir_to_cpuasm(src, metadata, options)


    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = self.target
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}




class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "npu_utils.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "npu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "npu_utils.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_npu_ext("npu_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device):
        fnname, mix_mode = name.split()
        return self.npu_utils_mod.load_kernel_binary(fnname, kernel, shared, device, mix_mode)

    @functools.lru_cache()
    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid triton-compiler complain
        # fetch available memory at runtime
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"max_shared_mem" : 1, "num_aicore" : num_aic, "num_vectorcore" : num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_aicore_num()

class NPULauncher(object):
    def __init__(self, src, metadata):
        debug_mode = metadata.debug
        workspace_size = int(metadata.workspace_size) \
                              if hasattr(metadata, 'workspace_size') else -1
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        shapes = src.attrs.get_shapes()
        wrapper_src = generate_npu_wrapper_src(constants, signature, shapes, \
                                               workspace_size)
        so_launcher_path = make_npu_launcher_stub(wrapper_src, debug_mode)
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")

    def __call__(self, *args, **kwargs):
        profiler_registered = self.launch(*args, **kwargs)
        import triton
        triton.backends.dicp_triton.utils.TRITON_PROFILER_REGISTERED = True if profiler_registered == 1 else False

class NPUDriver(DriverBase):
    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        super().__init__()

    @classmethod
    def is_active(cls):
        def test_npucompiler():
            from triton.backends.dicp_triton.utils import _get_bisheng_path
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

    def get_current_target(self):
        backend = "npu"
        arch = self.utils.get_arch()
        warp_size = 0
        return GPUTarget(backend, arch, warp_size)

    def get_current_device(self):
        """
        Get current device
        """
        import torch
        import torch_npu
        return torch.npu.current_device()

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        import torch
        import torch_npu
        return torch.npu.set_device(device)

    def get_current_stream(self, device: Optional[int] = None) -> int:
        """
        Get stream for current device
        """
        # According to torch_npu, the content of a torch.npu.Stream is essentilly an rtStream_t
        # TODO: use CANN API instead of torchnpu
        import torch
        import torch_npu
        if device is None:
            device = self.get_current_device()
        return torch.npu.current_stream(device).npu_stream

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        import torch
        return torch.npu

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 192 * 1024 * 1024
        return torch.empty(cache_size // 4, dtype=torch.int, device='npu')

def make_npu_launcher_stub(src, debug=False):
    """
    Generate the launcher stub to launch the kernel
    """
    # try to get cached file
    so_cache_key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    so_cache_manager = get_cache_manager(so_cache_key)
    # append the cxx11_abi value to the launcher name to avoid
    # linking to a launcher with wrong cxx11_abi.
    use_cxx11_abi = _check_cxx11_abi()
    name = f"launcher_cxx11abi{use_cxx11_abi}"
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_name = f"{name}{suffix}"

    if debug:
        dump_manager = get_dump_manager(so_cache_key)
        print(f"Dumping {name}.cxx to {dump_manager.cache_dir}")
        dump_manager.put(src, f"{name}.cxx", binary = False)

    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is not None:
        return cache_path

    with tempfile.TemporaryDirectory() as tmpdir:
        if debug:
            so_cache_manager.put(src, f"{name}.cxx", binary=False)
        src_path = os.path.join(tmpdir, f"{name}.cxx")
        with open(src_path, "w") as f:
            f.write(src)
        enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
        if (enable_taskqueue):
            kernel_launcher_type = "torch"
        else:
            kernel_launcher_type = None
        so = _build_npu_ext(name, src_path, tmpdir, kernel_launcher=kernel_launcher_type)
        if debug:
            with open(so, "rb") as f:
                return dump_manager.put(f.read(), so_name, binary=True)
        with open(so, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)


# the template is from triton-adapter HEAD. Wrapping the generated kernel binary into a python module
def generate_npu_wrapper_src(constants, signature, shapes, workspace_size):
    import os
    def _ty_to_cpp(ty):
        if ty[0] == '*':
            return "void*"
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

    def _extracted_ty(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gridX, gridY, gridZ;
        rtStream_t stream;
        const void *functon;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])

    grid_info = {'X': 'i32', 'Y': 'i32', 'Z': 'i32'}

    enable_device_print = os.getenv("TRITON_DEVICE_PRINT", 'false').lower() in ('true', '1')
    enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')

    LINE_CHANGE_CHAR = chr(10) # it is \n
    # tensorData <=5, cause MSPROF_GE_TENSOR_DATA_LEN = 5
    max_tensors_num = min(len(shapes), 5)
    # Take only the first 8 tensors of signature.items(). MSPROF_GE_TENSOR_DATA_SHAPE_LEN = 8
    limited_tensors = [
                          (i, ty) for i, ty in signature.items()
                          if i not in constants and i in shapes and ty.startswith("*")
                      ][:8]

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  if (!(*profilerRegistered)) {
     MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
     *profilerRegistered = 1;
  }
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    // FIXME: to avoid bug in msprof, currently we disable these checks
    // if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    // FIXME: to avoid bug in msprof, currently we disable these checks
    // if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    // FIXME: to avoid bug in msprof, currently we disable these checks
    // if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AI_CORE;
      nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));
    }}
    {{
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      profTensorData->tensorNum = {max_tensors_num};
      for (int i = 0; i < {max_tensors_num}; i++) {{
        profTensorData->tensorData[i].tensorType = MSPROF_GE_TENSOR_TYPE_INPUT; // FIXME
        profTensorData->tensorData[i].format = 2; // GeDataFormat: ND = 2
      }}
      // What if the index is non-contiguous?
      // Scalars don't have a '*' and are returned without cropping
      {LINE_CHANGE_CHAR.join(
        f'profTensorData->tensorData[{i}].dataType = {convert_sigtype_to_int(ty[1:])};'
        for i, ty in signature.items()
        if i not in constants and i in shapes and ty.startswith("*")
      )}
      // The scalar is not in the shapes array, and the shape assignment operation is not performed
      {LINE_CHANGE_CHAR.join(
        f'profTensorData->tensorData[{i}].shape[{j}] = {shapes[i][j]};'
        for i, ty in limited_tensors
        for j in range(min(len(shapes[i]), 8))
      )}

      // Set to 0 except for the true dimension of the tensor. The total is MSPROF_GE_TENSOR_DATA_SHAPE_LEN = 8
      {LINE_CHANGE_CHAR.join(
        f'profTensorData->tensorData[{i}].shape[{j}] = 0;'
        for i, ty in limited_tensors
        for j in range(min(len(shapes[i]), 8), 8)
      )}
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string>
#include <sys/syscall.h>
{'#include <pybind11/pybind11.h>' if enable_device_print else ''}
{'#include <pybind11/iostream.h>' if enable_device_print else ''}

#define PY_SSIZE_T_CLEAN
#include <Python.h>
{'#include <torch_npu/csrc/framework/OpCommand.h>' if enable_taskqueue else ''}
#include "experiment/runtime/runtime/rt.h"
{'#include "device_print.h"' if enable_device_print else ''}

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, int *profilerRegistered, {arg_decls}) {{
  {'pybind11::scoped_ostream_redirect output;' if enable_device_print else ''}
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  {cpp_msprof_callback}
  {'auto launch_call = [=]()' if enable_taskqueue else ''} {{
    uint32_t blockNum = gridX * gridY * gridZ;
    {'TTAscDebug::DebugTunnelData *DTData = TTAscDebug::Open(blockNum);' if enable_device_print else ''}
    rtError_t ret;
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    // stub argument for workspace
    void *workspace_addr = NULL;
    {f'''
    uint16_t ModuleId = 0;
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    ret = rtMalloc(reinterpret_cast<void **>(&workspace_addr),
                   totalWorkSpaceSize, RT_MEMORY_HBM, ModuleId);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if workspace_size > 0 else ''}
    struct __attribute__((packed)) {{
      void* ffts_addr __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants)}
      {' '.join(f'{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
      {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
    }} args = {{
      static_cast<void*>(ffts_addr),
      static_cast<void*>(workspace_addr),
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants)},
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
      {', static_cast<void*>(DTData)' if enable_device_print else ''}
    }};
    {cpp_msprof_call_before_launch}
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
    {'TTAscDebug::Close(DTData, stream);' if enable_device_print else ''}
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else ''}
   }};
   {'at_npu::native::OpCommand cmd; cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();' if enable_taskqueue else ''}
  return;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  PyObject *profilerRegisteredObj = PyDict_GetItemString(packedMetadata, "profiler_registered");
  int profilerRegistered = PyObject_IsTrue(profilerRegisteredObj);
  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0]=="*" else "" for i, ty in signature.items()])};
  _launch(kernelName, function, stream, gridX, gridY, gridZ, &profilerRegistered, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items())});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}

  return Py_BuildValue("I", profilerRegistered);
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""