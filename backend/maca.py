import hashlib
import os
import tempfile
import shutil
import subprocess
import sysconfig
import contextlib
import sys
import io
import functools
import importlib
import setuptools
from pathlib import Path
from triton.runtime.cache import get_cache_manager
from triton.runtime import JITFunction
from .utils import quiet

from triton.backends.dicp_triton.libtriton.triton import (
    add_external_libs, compile_ptx_to_cubin,
    get_shared_memory_size, ir,
    translate_llvmir_to_hsaco, translate_llvmir_to_ptx,
    translate_triton_gpu_to_llvmir, translate_llvmir_to_mcfatbin)

if os.environ.get('MACA_PATH') is not None:
    USE_MACA = True
else:
    USE_MACA = False
assert USE_MACA, "Please set MACA_PATH!"

def get_architecture_descriptor(capability=None):
    try:
        import torch
    except ImportError:
        raise ImportError("Triton requires PyTorch to be installed")
    if capability is None:
        if torch.version.hip is None:
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]
        else:
             raise ImportError("HIP not supported")
    return capability

def parse_mlir_module(mod):
    context = ir.context()
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        Path(src_path).write_text(ttir_code)
        src_path = "/home/pujiang/tangding/Triton/python/op/add_kernel.ttir.mx"
        module = ir.parse_mlir_module(src_path, context)
        # module takes ownership of the context
        module.context = context
        return module

def ttir_to_ttgir(mod, num_warps):
    mod = parse_mlir_module(mod)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.run(mod)
    return mod

def optimize_ttgir(mod, num_stages, arch):
    """
    MACA supported pass combination:
    1. tritongpu_pipeline_maca4_pass
    2. tritongpu_pipeline_maca4_pass + tritongpu_prefetch_maca2_pass
    1. tritongpu_pipeline_maca5_pass - tritongpu_optimize_dot_operands_pass
    2. tritongpu_pipeline_maca5_pass + tritongpu_prefetch_maca2_pass - tritongpu_optimize_dot_operands_pass
    """
    use_opt_maca_mma = (USE_MACA and os.getenv("TRITON_ENABLE_MACA_OPT_MMA") and num_stages == 2)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    if isinstance(arch, int):
        pm.add_tritongpu_accelerate_matmul_pass(arch, num_stages)
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    if use_opt_maca_mma:
        pm.add_tritongpu_pipeline_maca5_pass(num_stages)
        pm.add_tritongpu_prefetch_maca2_pass()
    else:
        pm.add_tritongpu_pipeline_pass(num_stages)
        pm.add_tritongpu_prefetch_pass()
    # TODO(fix): add_tritongpu_pipeline_maca5_pass induces endless loop
    if not use_opt_maca_mma:
        pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    if not use_opt_maca_mma:
        pm.add_tritongpu_reorder_instructions_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod

def ttgir_to_llir(mod, arch):
    return translate_triton_gpu_to_llvmir(mod, arch, False)

def llir_to_mcfatbin(mod, mxcc_arch: str, maca_path: str):
    '''
    Translate TritonGPU module to mcfatbin code.
    :param mod: a TritonGPU dialect module
    :return:
        - Path to mcfatbin object
    '''
    return translate_llvmir_to_mcfatbin(mod, mxcc_arch, maca_path)


# ----- source code generation --------


def ty_to_cpp(ty):
    if ty[0] == '*':
        if USE_MACA:
            return "mcDeviceptr_t"
        else:
            return "CUdeviceptr"
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


def generate_launcher(constants, signature):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
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

    def format_of(ty):
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

    format = "iiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    if USE_MACA:
        src = f"""
    #include <mcr/mc_runtime.h>
    #include <Python.h>

    static inline void gpuAssert(mcError_t code, const char *file, int line)
    {{
    if (code != mcSuccess)
    {{
        const char* prefix = "Triton Error [MACA]: ";
        const char* str = mcGetErrorString(code);
        char err[1024] = {{0}};
        strcat(err, prefix);
        strcat(err, str);
        PyErr_SetString(PyExc_RuntimeError, err);
    }}
    }}

    #define MACA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

    void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, mcStream_t stream, mcFunction_t function, {arg_decls}) {{
    void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
    if(gridX*gridY*gridZ > 0){{
        MACA_CHECK(mcModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
    }}
    }}

    typedef struct _DevicePtrInfo {{
        mcDeviceptr_t dev_ptr;
        bool valid;
    }} DevicePtrInfo;

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
    DevicePtrInfo ptr_info;
    ptr_info.dev_ptr = (mcDeviceptr_t)0;
    ptr_info.valid = true;
    if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (mcDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
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
        ptr_info.dev_ptr = (mcDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
        if(!ptr_info.dev_ptr)
        return ptr_info;
        uint64_t dev_ptr;
        int status = mcPointerGetAttribute(&dev_ptr, mcPointerAttributeDevicePointer, ptr_info.dev_ptr);
        if (status == mcErrorInvalidValue) {{
            PyErr_Format(PyExc_ValueError,
                        "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
            ptr_info.valid = false;
        }}
        ptr_info.dev_ptr = (mcDeviceptr_t)dev_ptr;
        Py_DECREF(ret);
        return ptr_info;
    }}
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    return ptr_info;
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{
    int gridX, gridY, gridZ;
    uint64_t _stream;
    uint64_t _function;
    int num_warps;
    int shared_memory;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *compiled_kernel = NULL;
    PyObject *hook_ret = NULL;
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
    if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
        return NULL;
    }}

    if (launch_enter_hook != Py_None) {{
        PyObject *new_args = PyTuple_Pack(1, compiled_kernel);
        hook_ret = PyObject_CallObject(launch_enter_hook, new_args);
        Py_DECREF(new_args);
    }}

    // raise exception asap
    {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
    _launch(gridX, gridY, gridZ, num_warps, shared_memory, (mcStream_t)_stream, (mcFunction_t)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

    if (launch_exit_hook != Py_None) {{
        PyObject *new_args = NULL;
        if (hook_ret) {{
            new_args = PyTuple_Pack(2, compiled_kernel, hook_ret);
        }} else {{
            new_args = PyTuple_Pack(1, compiled_kernel);
        }}
        hook_ret = PyObject_CallObject(launch_exit_hook, new_args);
        Py_DECREF(new_args);
    }}

    if (hook_ret) {{
        Py_DECREF(hook_ret);
    }}
    if(PyErr_Occurred()) {{
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
    elif is_hip():
        src = f"""
    #define __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <Python.h>
    #include <stdio.h>

    static inline void gpuAssert(hipError_t code, const char *file, int line)
    {{
      if (code != HIP_SUCCESS)
      {{
         const char* prefix = "Triton Error [HIP]: ";
         const char* str = hipGetErrorString(code);
         char err[1024] = {{0}};
         snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
         PyErr_SetString(PyExc_RuntimeError, err);
      }}
    }}

    #define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

    static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, hipStream_t stream, hipFunction_t function, {arg_decls}) {{
      void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
      if (gridX*gridY*gridZ > 0) {{
          HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
      }}
    }}

    typedef struct _DevicePtrInfo {{
      hipDeviceptr_t dev_ptr;
      bool valid;
    }} DevicePtrInfo;

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
      DevicePtrInfo ptr_info;
      ptr_info.dev_ptr = 0;
      ptr_info.valid = true;

      if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
        return ptr_info;
      }}

      if (obj == Py_None) {{
        // valid nullptr
        return ptr_info;
      }}

      PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");

      if (ptr) {{
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);

        if (!PyLong_Check(ret)) {{
          PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
          ptr_info.valid = false;
          return ptr_info;
        }}

        ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);

        if (!ptr_info.dev_ptr)
          return ptr_info;

        uint64_t dev_ptr;
        hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
        if (status == hipErrorInvalidValue) {{
            PyErr_Format(PyExc_ValueError,
                         "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
            ptr_info.valid = false;
        }}

        ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
        return ptr_info;
      }}

      PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
      return ptr_info;
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      uint64_t _stream;
      uint64_t _function;
      int num_warps;
      int shared_memory;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *compiled_kernel = NULL;

      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if (!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
        return NULL;
      }}

      if (launch_enter_hook != Py_None) {{
        PyObject_CallObject(launch_enter_hook, args);
      }}

      // raise exception asap
      {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
      _launch(gridX, gridY, gridZ, num_warps, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items())});
      if (launch_exit_hook != Py_None) {{
        PyObject_CallObject(launch_exit_hook, args);
      }}
      if (PyErr_Occurred()) {{
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
    else:
        src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, CUstream stream, CUfunction function, {arg_decls}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  if(gridX*gridY*gridZ > 0){{
    CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
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
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject_CallObject(launch_enter_hook, args);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (launch_exit_hook != Py_None) {{
    PyObject_CallObject(launch_exit_hook, args);
  }}

  if(PyErr_Occurred()) {{
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
    return src

@functools.lru_cache()
def maca_home_dirs():
    return os.getenv("MACA_PATH")

@functools.lru_cache()
def libmaca_dir():
    return os.path.join(maca_home_dirs(), "lib")

def _build(name, src, srcdir):
    maca_lib_dir = libmaca_dir()
    maca_include_dir = os.path.join(maca_home_dirs(), "include")
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

    ret = subprocess.check_call(["/usr/bin/g++", src, f"-I{maca_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", f"-D__MACA__", "-shared", "-fPIC", f"-L{maca_lib_dir}", "-lmcruntime", "-o", so])

    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    include_dirs = [srcdir]
    libraries = ['cuda']
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


class MacaUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MacaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "maca.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "maca_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("maca_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("maca_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class MacaLauncher(object):

    def __init__(self, src, metadata):
        if isinstance(src.fn, JITFunction):
            name, _ = src.fn.__name__, "ast"
        else:
            name, _ = os.path.basename(src.fn).split(".")
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        so_cache_key = metadata['hash']
        self.so_path = self.make_launcher_stub(name, so_cache_key, signature, constants)
        spec = importlib.util.spec_from_file_location("__triton_launcher", self.so_path)
        mod = importlib.util.module_from_spec(spec)
        self.launch = mod.launch

    def __call__(self, grid_0, grid_1, grid_2, stream, kernel_function, kernel_packed_metadata, launch_metadata, launch_enter_hook,launch_exit_hook, *args, **kwargs):
        self.launch(grid_0, grid_1, grid_2, 4, 0, stream, kernel_function, launch_enter_hook, launch_exit_hook, self, *args)

    def make_launcher_stub(self, name, so_cache_key, signature, constants):
        # name of files that are cached
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = generate_launcher(constants, signature)
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)

                so = _build(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path