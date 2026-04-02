# CPU Backend for verification
# Merged from triton_shared backend/compiler.py and backend/driver.py

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import triton
from pathlib import Path
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
import sysconfig
import importlib.util
import sys
import platform
import triton.backends.dicp_triton.utils as dicp_utils

dump_ir = os.environ.get("DLC_DUMP_IR", "0") == "1"


def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _get_sanitizer_type():
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")
    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")
    return sanitizer_type


def _ttir_to_ttsharedir(mod, metadata):
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess_args = [
            triton_shared_opt_path,
            src_path,
            "--triton-to-linalg-experimental",
            "--mlir-print-debuginfo",
            "-o",
            dst_path,
        ]
        if _get_sanitizer_type() != "":
            print("Building with sanitizer support...")
            subprocess_args.insert(2, "--add-llvm-debug-info")
        subprocess.check_call(subprocess_args)
        result = Path(dst_path).read_text()
        if dump_ir:
            dicp_utils._dump_stage_ir(
                result, metadata["hash"], "kernel.ttsharedir.mlir"
            )
        return result


def _optimize_ttsharedir(ttsharedir: str):
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        subprocess.check_call(
            [
                mlir_opt_path,
                ttshared_path,
                "--convert-elementwise-to-linalg",
                "--convert-linalg-to-affine-loops",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--expand-strided-metadata",
                "--convert-scf-to-cf",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--convert-cf-to-llvm",
                "--lower-affine",
                "--convert-arith-to-llvm",
                "--reconcile-unrealized-casts",
                "--mlir-print-debuginfo",
                "-o",
                llmlir_path,
            ]
        )
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        result = Path(llir_path).read_text()
        if dump_ir:
            dicp_utils._dump_stage_ir(result, metadata["hash"], "kernel.llir.mlir")
        return result


def _optimize_llir(llir: str):
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)
        sanitizer_type = _get_sanitizer_type()
        if sanitizer_type != "":
            instrumented_src_path = os.path.join(tmpdir, "kernel-instrumented.ll")
            opt_path = _get_llvm_bin_path("opt")
            top_level_triton_path = os.path.dirname(triton.__file__)
            sanitizer_attributes_pass_path = str(
                next(
                    Path(top_level_triton_path).rglob("libSanitizerAttributes.so"), None
                )
            )
            if not sanitizer_attributes_pass_path:
                raise Exception(f"libSanitizerAttributes.so does not exist.")
            subprocess.check_call(
                [
                    opt_path,
                    "-load-pass-plugin",
                    sanitizer_attributes_pass_path,
                    "-passes=sanitizer-attributes",
                    f"-sanitizer-type={sanitizer_type}",
                    "-S",
                    src_path,
                    "-o",
                    instrumented_src_path,
                ]
            )
            clang_path = _get_llvm_bin_path("clang++")
            subprocess_args = [clang_path, "-c", instrumented_src_path, "-o", dst_path]
            if sanitizer_type == "asan":
                subprocess_args.extend(
                    ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
                )
            elif sanitizer_type == "tsan":
                subprocess_args.extend(["-g", "-fsanitize=thread"])
            subprocess.check_call(subprocess_args)
        else:
            llc_path = _get_llvm_bin_path("llc")
            subprocess.check_call(
                [
                    llc_path,
                    src_path,
                    "-filetype=obj",
                    "-relocation-model=pic",
                    "-o",
                    dst_path,
                ]
            )
        return Path(dst_path).read_bytes()


# -------------------- Compiler --------------------


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None

    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = 2

    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = True

    extern_libs: dict = None

    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False

    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False

    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")

    sanitize_overflow: bool = True
    enable_npu_compile: bool = True

    kernel_name: str = "triton_"

    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False

    max_num_imprecise_acc_default: bool = None
    multibuffer: bool = True

    inject_barrier_all: bool = False
    disable_auto_inject_block_sync: bool = False
    unit_flag: bool = False

    disable_auto_cv_work_space_manage: bool = False
    enable_auto_bind_sub_block: bool = True

    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None

    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    set_workspace_multibuffer: int = None

    stream: int = None

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = "obj"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        )
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        if dump_ir:
            dicp_utils._dump_stage_ir(str(mod), metadata["hash"], "kernel.ttir.mlir")
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(
            _ttir_to_ttsharedir(src, metadata)
        )
        stages["llir"] = lambda src, metadata: _optimize_llir(
            _ttsharedir_to_llir(src, metadata)
        )
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}


# -------------------- Driver --------------------


def _ty_to_cpp(ty):
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
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _extracted_type(ty):
    if ty[0] == "*":
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join(
        [_format_of(_extracted_type(ty)) for ty in signature.values()]
    )
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    kernel_arg_decls = ", ".join(
        _ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_arg_decls += ", " if kernel_arg_decls else ""

    kernel_parameters = ", ".join(
        f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_parameters += ", " if kernel_parameters else ""

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    // apply parallelization to the triton grid when using ThreadSanitizer (TSan) 
    // to help detect potential data races across program instances during kernel execution
    {"#pragma omp parallel for collapse(3)" if _get_sanitizer_type() == "tsan" else ""}
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          // Use some random type "char" here.
          {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};' for i, ty in signature.items() if i not in constants and ty[0] == "*")}
          {kernel_name}({kernel_parameters}
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, "include")
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib = "{name}{major}{minor}.lib".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    else:
        py_include_dir = os.path.join(
            sys.base_prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        py_lib_dir = os.path.join(sys.base_prefix, "lib")
        py_lib = "{name}{major}.{minor}".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        kernel_obj = cu_function
        kernel_name = kernel_metadata[6]
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)
        key = hashlib.sha256(src.encode("utf-8") + kernel_obj).hexdigest()
        cache = get_cache_manager(key)
        name = "__triton_shared_ref_cpu_kernel_launcher"
        if platform.system() == "Windows":
            filename = f"{name}.pyd"
        else:
            filename = f"{name}.so"
        cache_path = cache.get_file(filename)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                sanitizer_type = _get_sanitizer_type()
                if platform.system() == "Windows":
                    if sanitizer_type != "":
                        raise Exception(
                            "Sanitizers are not supported on Windows with triton-shared."
                        )
                    obj_path = os.path.join(tmpdir, "kernel.obj")
                    launcher_src_path = os.path.join(tmpdir, "main.cxx")
                    so_path = os.path.join(tmpdir, "kernel.pyd")
                    Path(obj_path).write_bytes(kernel_obj)
                    Path(launcher_src_path).write_text(src)
                    subprocess.check_call(
                        [
                            "cl",
                            "/LD",
                            "/std:c++17",
                            launcher_src_path,
                            obj_path,
                            f"-I{py_include_dir}",
                            f"-I{include_dir}",
                            "/link",
                            f"/LIBPATH:{py_lib_dir}",
                            "/link",
                            f"{py_lib}",
                            f"/OUT:{so_path}",
                        ]
                    )
                else:
                    obj_path = os.path.join(tmpdir, "kernel.o")
                    launcher_src_path = os.path.join(tmpdir, "main.cxx")
                    so_path = os.path.join(tmpdir, "kernel.so")
                    Path(obj_path).write_bytes(kernel_obj)
                    Path(launcher_src_path).write_text(src)
                    if sanitizer_type != "":
                        clang_path = _get_llvm_bin_path("clang++")
                        subprocess_args = [
                            clang_path,
                            "-std=c++17",
                            launcher_src_path,
                            obj_path,
                            f"-I{py_include_dir}",
                            f"-I{include_dir}",
                            f"-L{py_lib_dir}",
                            "-shared",
                            f"-l{py_lib}",
                            "-fPIC",
                            "-o",
                            so_path,
                        ]
                        if sanitizer_type == "asan":
                            subprocess_args.extend(
                                ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
                            )
                        elif sanitizer_type == "tsan":
                            subprocess_args.extend(["-g", "-fsanitize=thread"])
                        subprocess.check_call(subprocess_args)
                    else:
                        subprocess.check_call(
                            [
                                "g++",
                                "-std=c++17",
                                launcher_src_path,
                                obj_path,
                                f"-I{py_include_dir}",
                                f"-I{include_dir}",
                                f"-L{py_lib_dir}",
                                "-shared",
                                f"-l{py_lib}",
                                "-fPIC",
                                "-o",
                                so_path,
                            ]
                        )
                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(
            gridX,
            gridY,
            gridZ,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            *args,
        )

    return launch


class CPULauncher(object):
    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        self.launch = compile_module(launcher_src, kernel_placeholder_name)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
            None,
            kernel_obj,
            None,
            None,
        )


class CPUDriver(DriverBase):
    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "obj"

    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        return "cpu"

    def set_current_device(self, device):
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
