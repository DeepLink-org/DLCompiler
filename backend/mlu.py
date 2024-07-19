
import os
from typing import Any
from collections import namedtuple
import triton.backends.dicp_triton.libtriton.triton as _triton
from triton.backends.dicp_triton.libtriton.triton import ir
import torch_mlu
from pathlib import Path
import tempfile
from triton.runtime.cache import get_cache_manager, make_so_cache_key
import sysconfig
import shutil
import subprocess
import setuptools

def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CNaddr"
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

def generate_launcher(constants, signature, ids):
    start_desc = len(signature)
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

    #folded_without_constexprs = [c for c in ids['ids_of_folded_args'] if c not in ids['ids_of_const_exprs']]
    #params = [i for i in signature.keys() if i >= start_desc or (i not in constants and i not in folded_without_constexprs)]
    print("signature: ", signature)
    params = [i for i in signature.keys() if i not in constants]
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format
    # Generate glue code.
    src = f"""

#include \"cn_api.h\"
#include \"cnrt.h\"
#include \"cnrtc.h\"

#include <stdbool.h>
#include <stdio.h>
#include <Python.h>

static inline void cnAssert(CNresult code, const char *file, int line) {{
  if (code != CN_SUCCESS) {{
    const char *prefix = "Triton Error [MLU]: ";
    const char *str;
    cnGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    PyErr_SetString(PyExc_RuntimeError, err);
  }}
}}

#define CN_CHECK(ans) {{ cnAssert((ans), __FILE__, __LINE__); }}

static void _launch(unsigned int dimx, unsigned int dimy, unsigned int dimz, KernelClass func_type, CNqueue stream, CNkernel function, {arg_decls}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};

  if(dimx*dimy*dimz > 0) {{
    CN_CHECK(cnInvokeKernel(function, dimx, dimy, dimz, func_type, 0, stream, params, NULL));
  }}
}}

typedef struct _DevicePtrInfo {{
    uint64_t dev_ptr;
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
    cnrtPointerAttributes_t attributes;
    cnrtRet_t status = cnrtPointerGetAttributes(&attributes, (void*)ptr_info.dev_ptr);
    if (status != CNRT_RET_SUCCESS) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    attributes.devicePointer = (void*)dev_ptr;
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
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  // TODO: how to decide func_type?
  uint64_t func_type = 1;

  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

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
  _launch(gridX, gridY, gridZ, (KernelClass)func_type, (CNqueue)_stream, (CNkernel)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;

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

def default_neuware_dir():
    default_dir = "/usr/local/neuware/"
    return os.getenv("NEUWARE_HOME", default=default_dir)

def _build_mlu_ext(name, src, srcdir):
    cn_path = default_neuware_dir()
    cn_lib_dirs = [os.path.join(cn_path, "lib64")]
    cn_include_dir = os.path.join(cn_path, "include")

    # The main code is copied from triton/python/triton/common/build.py.

    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))

    # Try to avoid setuptools if possible
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

    cc_cmd = [cc, src, "-O3", f"-I{cn_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-o", so]
    cc_cmd += [f"-L{dir}" for dir in cn_lib_dirs]
    cc_cmd += ["-lcndrv", "-lcnrt"]
    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = cuda_lib_dirs
    include_dirs = [srcdir, cu_include_dir]
    libraries = ['mlu']
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

from triton.runtime import JITFunction
import hashlib
import importlib

class MLUUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MLUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "mlu.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "mlu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_mlu_ext("mlu_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        print(cache_path)
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.is_linear_pointer = mod.is_linear_pointer


class MluLauncher(object):

    def __init__(self, src, metadata):
        if isinstance(src.fn, JITFunction):
            name, _ = src.fn.__name__, "ast"
        else:
            name, _ = os.path.basename(src.fn).split(".")
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        so_cache_key = src.hash()
        self.so_path = self.make_launcher_stub(name, so_cache_key, signature, constants, ids)
        print("sopath: ", self.so_path)
        spec = importlib.util.spec_from_file_location("__triton_launcher", self.so_path)
        mod = importlib.util.module_from_spec(spec)
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)

    def make_launcher_stub(self, name, so_cache_key, signature, constants, ids):
        # name of files that are cached
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = generate_launcher(constants, signature, ids)
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)

                so = _build_mlu_ext(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path

def set_num_warps(mod: Any, num_warps: int):
    '''
    Set num warps on triton module.
    :param: mod: tt ir module.
    :num_warps: num warps set by user, it will attach attributes
                triton.xpe = num_warps > 1 ? 4 : 1 and
                triton.xtask = num_warps / 4.
    '''
    if num_warps not in [1, 4, 8, 16, 32]:
        raise ValueError("num_warps only support 1/4/8/16/32")

    builder = ir.builder(mod.context)
    if num_warps > 1:
        mod.set_attr("triton.xpe", builder.get_int32_attr(4))
        mod.set_attr("triton.xtask", builder.get_int32_attr(num_warps // 4))
    else:
        mod.set_attr("triton.xpe", builder.get_int32_attr(1))

def set_num_stages(mod: Any, num_stages: int):
    builder = ir.builder(mod.context)
    mod.set_attr("triton.num_stages", builder.get_int32_attr(num_stages))

def set_linear_attr(mod: Any, is_linear: bool):
    if is_linear is not None:
        builder = ir.builder(mod.context)
        mod.set_attr("triton.is_linear", builder.get_bool_attr(is_linear))

def set_debug_attr(mod: Any):
    debug = os.environ.get("TRITON_DEBUG", "0")
    if debug not in ["0", "1"]:
        raise ValueError("TRITON_DEBUG must be '0' or '1', but got {}", debug)
    if debug == "1":
        builder = ir.builder(mod.context)
        mod.set_attr("mlu.debug", builder.get_bool_attr(True))

def set_kernel_sym(mod: Any, kernel_name: str):
    if kernel_name is not None:
        builder = ir.builder(mod.context)
        mod.set_attr("triton.kernel_name", builder.get_str_attr(kernel_name))

def set_silence(mod: Any):
    builder = ir.builder(mod.context)
    mod.set_attr("mlu.silence", builder.get_bool_attr(True))

def ttir_to_cnfatbin(mod: Any, metadata: dict, arch_desc: dict, tuning_mode: bool, is_linear: bool):
    '''
    Compile triton module to cnfatbin.
    :param mod: tt ir module.
    :param arch_desc: contains num_warps and arch.
    :return: str
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "temp_ttir.mlir")
        ttir_code = str(mod)
        kernel_name = metadata.get('name', None)
        Path(src_path).write_text(ttir_code)
        context = ir.context()
        context.load_triton()
        mod = ir.parse_mlir_module(src_path, context)

        context2 = ir.context()
        context2.load_triton()
        builder = ir.builder(context2)
        builder.target = arch_desc
        mod2 = builder.create_module()
        if kernel_name is None:
            mod2.push_back(mod.get_single_function())
        else:
            mod2.push_back(mod.get_function(kernel_name))
        mod2.context = context2

        set_num_warps(mod2, arch_desc["num_warps"])
        set_num_stages(mod2, arch_desc["num_stages"])
        set_linear_attr(mod2, is_linear)
        set_debug_attr(mod2)
        set_kernel_sym(mod2, kernel_name)
        if tuning_mode:
            set_silence(mod2)
        return _triton.compile_ttir_to_cnfatbin(mod2, arch_desc["isa_version"])


def get_architecture_descriptor(driver, **kwargs):
    # Currently only support Block task.
    device = driver.get_current_device()
    num_warps = kwargs.get("num_warps", 1)
    num_stages = kwargs.get("num_stages", 1)
    isa_version = driver.utils.get_device_properties(device).get('isa_version')
    isa_version_passed = kwargs.get("isa_version", None)
    isa_version = isa_version if isa_version_passed is None else isa_version_passed
    capability = {
        "num_warps": num_warps,
        "isa_version": isa_version,
        "num_stages" : num_stages,
    }
    return capability