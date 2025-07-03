import functools
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, mlu
# from triton.backends.mlu.driver import default_neuware_dir
from triton.runtime.errors import OutOfResources

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional, Dict
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import warnings
import subprocess
from pathlib import Path

dirname = os.path.dirname(os.path.realpath(__file__))
libraries = ['cndrv', 'cnrt']


@functools.lru_cache()
def default_neuware_dir():
    default_dir = "/usr/local/neuware/"
    return os.getenv("NEUWARE_HOME", default=default_dir)


@functools.lru_cache()
def library_dirs():
    return [os.path.join(default_neuware_dir(), "lib64")]


@functools.lru_cache()
def include_dirs():
    return [os.path.join(default_neuware_dir(), "include")]


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs(), include_dirs(),
                        libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class BangUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(BangUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            Path(os.path.join(dirname, "driver.c")).read_text(), "bang_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.is_linear_pointer = mod.is_linear_pointer


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty, ptr_ty="CNaddr"):
    if ty[0] == '*':
        return ptr_ty
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
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "PyObject*": "O",
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
        }[ty_to_cpp(ty)]

    def serialize_idx(signature):
        flat_list = []
        mapping = {}
        idx = 0

        def traverse(sub_lst, path=()):
            nonlocal idx
            for i, item in enumerate(sub_lst):
                new_path = path + (i, )
                if isinstance(item, tuple):
                    traverse(item, new_path)
                else:
                    flat_list.append(item)
                    mapping[new_path] = idx
                    idx += 1

        traverse(signature)
        return mapping

    idx_map = serialize_idx(signature.values())
    constants = {
        idx_map[idx]: value
        for idx, value in constants.items() if idx in idx_map
    }
    remove_idx = {
        idx_map[(i, )]
        for i, ty in enumerate(signature.values()) if ty == "constexpr"
    }

    args_format = ''.join(
        [format_of(ty) for ty in signature.values() if ty != "constexpr"])
    format = "iiiKKOOOO" + args_format
    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    non_const_args = {i: ty for i, ty in signature.items() if i not in remove_idx}
    args_list = ', ' + ', '.join(
        f"&_arg{i}"
        for i, ty in non_const_args.items()) if len(non_const_args) > 0 else ''

    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items()
                          if ty != "constexpr" and i not in constants)
    # generate glue code
    params = [
        i for i, ty in signature.items() if ty != "constexpr" and i not in constants
    ]

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr" and i not in constants:
            internal_args_list.append(f"_arg{i}")

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
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
  }}
}}

#define CN_CHECK(ans) {{ cnAssert((ans), __FILE__, __LINE__); }}

static void _launch(unsigned int dimx, unsigned int dimy, unsigned int dimz, KernelClass func_type, CNqueue stream, CNkernel function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
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
    if (status != cnrtSuccess) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    attributes.devicePointer = (void*)dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
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
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int num_warps;
  if (!PyArg_ParseTuple(kernel_metadata, \"i\", &num_warps)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  gridX *= num_warps;

  int ordinal = -1;
  cnrtGetDevice(&ordinal);
  cnrtDeviceProp_t prop;

  cnrtGetDeviceProperties(&prop, ordinal);
  int cluster_cnt = prop.clusterCount;
  int core_num_per_cluster = prop.McorePerCluster;
  int total_cores = cluster_cnt * core_num_per_cluster;

  uint64_t func_type = ((num_warps == 1) && (gridX % core_num_per_cluster == 0) && (gridX * gridY * gridZ <= total_cores)) ? core_num_per_cluster : num_warps;

  if (launch_enter_hook != Py_None) {{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, (KernelClass)func_type, (CNqueue)_stream, (CNkernel)_function{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  Py_END_ALLOW_THREADS;

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


class BangLauncher(object):

    def __init__(self, src, metadata):
        ids = {
            "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants | src.attrs.get_constants(), signature, ids)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


def path_to_binary(binary: str):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(default_neuware_dir(), "bin", binary)
    ]

    for p in paths:
        bin = p.split(" ")[0]
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"],
                                             stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*" + binary + "( version:)? (\d+\.\d+\.\d+).*",
                                    result.decode("utf-8"),
                                    flags=re.MULTILINE)
                if version is not None:
                    return p, version.group(2)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_cnas_version():
    _, version = path_to_binary("cnas")
    return version


MIN_REQUIRED_CNTOOLKIT_VERSION = "4.1.0"


def check_cntoolkit_version():
    from packaging import version
    version_file = os.path.join(default_neuware_dir(), "version.txt")
    if not os.path.exists(version_file):
        return warnings.warn(
            f"{version_file} is not found, please install cntoolkit-cloud")

    with open(version_file, 'r') as f:
        line = f.readline().strip()
    match = re.search(r"Version\s+([\d.]+)", line)
    if match:
        cntoolkit_version = match.group(1)
    else:
        raise RuntimeError(f"Cannot find cntoolkit version")

    if version.parse(cntoolkit_version) < version.parse(MIN_REQUIRED_CNTOOLKIT_VERSION):
        raise RuntimeError(
            f"cntoolkit version {cntoolkit_version} is lower than required {MIN_REQUIRED_CNTOOLKIT_VERSION}"
        )


check_cntoolkit_version()


def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


def _extract_memory_info(log: str) -> dict:
    pattern = r'(NRAM|WRAM|SRAM)\s+(\d+)\s+([-]?\d+)\s+(\d+)\s+'
    return re.findall(pattern, log)


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@functools.lru_cache()
def default_libdevice_dir():
    return str(Path(__file__).parent / "lib")


@dataclass(frozen=True)
class MLUOptions:
    num_warps: int = 1
    num_stages: int = 0
    cnas_version: int = None
    enable_soft_i64: bool = False
    is_linear: bool = False
    kernel_name: str = None

    # These options only used in the GPU, here, we are just setting default values.
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None

    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dtypes: Tuple[str] = ()
    allow_half_div: bool = True
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    precision_mode: str = "precision"
    backend_name: str = "mlu"
    sanitize_overflow: bool = False
    opt_level: str = "O3"
    restrict_ptr: bool = None
    restrict_ptr_hint: bool = False
    can_promote_relay: bool = False
    # Default bottleneck set to I/O, default behavior for software pipeline.
    bottleneck: str = None
    pipeline_strategies: Tuple[str] = ()
    onchip_mem_analysis: str = False
    # Eanble internal mlu instruction bound check, it will slow down the running
    # speed, only used for debug.
    enable_mlu_bound_check: bool = False

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"
        assert self.bottleneck in [None, "io", "mv", "simd"]
        if self.num_warps not in [1, 4, 8, 16, 32]:
            warnings.warn("num_warps should in 1/4/8/16/32 for mlu backend")
        assert self.opt_level in ["O0", "O1", "O2", "O3", "Om", "Os"]

        # Only block and u1 task are supported.
        if self.num_warps > 4:
            warnings.warn(
                f"num_warps is currently set to {self.num_warps}; values greater "
                f"than 4 are not supported, falling back to 4", UserWarning)
            object.__setattr__(self, 'num_warps', 4)

        # Fallback to 1 if num_warps set to 2.
        if self.num_warps == 2:
            warnings.warn(
                "num_warps equals to 2 is not supported currently, "
                "fallback to 1 if encountered.", UserWarning)
            object.__setattr__(self, 'num_warps', 1)

        if self.debug is None:
            object.__setattr__(self, 'debug', False)

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple(
            (k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

@staticmethod
def set_num_warps(mod: Any, num_warps: int, builder):
    '''
    Set num warps on triton module.
    :param: mod: tt ir module.
    :num_warps: num warps set by user, it will attach attributes
                triton.xpe = num_warps > 1 ? 4 : 1 and
                triton.xtask = num_warps / 4.
    '''
    if num_warps >= 4:
        mod.set_attr("triton.xpe", builder.get_int32_attr(4))
        mod.set_attr("triton.xtask", builder.get_int32_attr(num_warps // 4))
    else:
        mod.set_attr("triton.xpe", builder.get_int32_attr(1))

@staticmethod
def ttir_get_kernel_info(ttir: str) -> dict:
    '''
    Get kernel info from ttir.
    '''
    info = dict(kernel_name='', contain_readperf=False)
    for line in ttir.split('\n'):
        line = line.strip()
        if line.startswith('tt.func public'):
            info['kernel_name'] = line.split('@')[1].split("(")[0]
        if line.startswith('mlu.readperf'):
            info['contain_readperf'] = True
    return info

@staticmethod
def stringify_arch(capability):
    return f'mtp_{capability}'

@staticmethod
def onchip_mem_analysis(mod, opt):
    return mlu.analysis_onchip_mem_usage(mod, opt)

@staticmethod
def get_estimate_onchip_memory_usage_fn(code: str, func_name: str):
    namespace = {}
    exec(code, namespace, namespace)
    if func_name in namespace:
        return namespace[func_name]
    else:
        raise ValueError(f"not found function '{func_name}'")

@staticmethod
def make_ttir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    mlu.passes.add_arith_canonicalizer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)

    kernel_info = ttir_get_kernel_info(str(mod))
    if not kernel_info['kernel_name']:
        raise RuntimeError("can not get kernel name from ttir")

    if kernel_info['contain_readperf'] and (opt.num_warps > 1
                                            or opt.num_stages > 1):
        raise RuntimeError(
            "the readperf op is not allowed when num_warps or num_stages is greater than 1"
        )

    # Set the required fields for metadata.
    metadata["name"] = kernel_info['kernel_name']
    metadata["shared"] = 0

    builder = ir.builder(mod.context)
    set_num_warps(mod, opt.num_warps, builder)

    mod.set_attr("tt.num_stages", builder.get_int32_attr(opt.num_stages))
    if opt.bottleneck:
        mod.set_attr("tt.bottleneck_stream", builder.get_str_attr(opt.bottleneck))
    if opt.pipeline_strategies is not None:
        mod.set_attr("tt.pipeline_strategies",
                        builder.get_str_array_attr(opt.pipeline_strategies))
    if capability < 600:
        mod.set_attr("triton.enable_soft_i64",
                        builder.get_bool_attr(opt.enable_soft_i64))
    else:
        if opt.enable_soft_i64:
            warnings.warn("Ignore enable_soft_i64 for capability {capability}")

    mod.set_attr("triton.is_linear", builder.get_bool_attr(opt.is_linear))
    if opt.kernel_name is not None:
        mod.set_attr("triton.kernel_name", builder.get_str_attr(opt.kernel_name))
    if opt.restrict_ptr is not None:
        mod.set_attr("genesis.restrict_ptr",
                        builder.get_bool_attr(opt.restrict_ptr))
    elif opt.restrict_ptr_hint:
        mod.set_attr("genesis.restrict_ptr_hint", builder.get_bool_attr(True))
    mod.set_attr(
        "genesis.promote_relay",
        builder.get_bool_attr(opt.can_promote_relay
                                and os.getenv("TRITON_MLU_PROMOTE_RELAY") == "1"))

    assert opt.precision_mode in ["fast", "precision"]

    mod.set_attr("genesis.assert",
                    builder.get_bool_attr(opt.enable_mlu_bound_check))
    mod.set_attr("genesis.precision_mode", builder.get_str_attr(opt.precision_mode))
    arch = stringify_arch(capability)
    mod.set_attr("genesis.arch", builder.get_str_attr(arch))

    return mod

@staticmethod
def make_linalg(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_canonicalizer(pm)
    mlu.passes.add_hot_cold_splitting(pm)
    mlu.passes.add_wrap_func_body_with_single_block(pm)
    mlu.passes.add_inliner(pm)
    mlu.passes.add_conservate_pointer_mode_set(pm)
    passes.common.add_canonicalizer(pm)
    mlu.passes.add_canonicalize_triton(pm)
    mlu.passes.add_pointer_strength_reduction(pm)
    mlu.passes.add_pointer_contiguity_enhancement(pm)
    mlu.passes.add_pointer_constancy_degeneration(pm)
    mlu.passes.add_refine_elementwise_symbol_attr(pm)
    mlu.passes.add_canonicalize_triton(pm)
    mlu.passes.add_optimize_triangle_mask(pm)
    mlu.passes.add_triton_to_arith(pm)
    passes.common.add_canonicalizer(pm)
    mlu.passes.add_arith_canonicalizer(pm)
    mlu.passes.add_tensor_canonicalizer(pm)
    mlu.passes.add_arithext_to_linalg(pm)
    mlu.passes.add_triton_to_linalg(pm)
    passes.common.add_cse(pm)
    mlu.passes.add_extract_like_move_backward(pm)
    passes.common.add_canonicalizer(pm)
    mlu.passes.add_convert_scalar_i64_to_tensor(pm)
    passes.common.add_canonicalizer(pm)
    mlu.passes.add_arith_to_linalg(pm)
    mlu.passes.add_math_to_linalg(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    mlu.passes.add_wrap_func_body_with_single_block(pm)
    mlu.passes.add_convert_triton_to_scf(pm)
    mlu.passes.add_generate_triton_executalbe(pm)
    mlu.passes.add_set_attr_to_forop(pm)
    pm.run(mod)
    return mod

@staticmethod
def make_optimized_linalg(mod, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    mlu.passes.add_auto_tile_pipeline(pm, opt.opt_level)
    pm.run(mod)
    return mod

@staticmethod
def make_mluir(mod, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    mlu.passes.add_post_process_pipeline(pm, opt.opt_level)
    pm.run(mod)
    return mod

@staticmethod
def make_optimize_mluir(mod, opt, capability):
    mlu.optimize_mluir(mod, opt, capability, default_libdevice_dir())
    return mod

@staticmethod
def make_mlisa(mod):
    pm = ir.pass_manager(mod.context, "builtin.module", ir.NESTING.IMPLICIT)
    pm.enable_debug()
    mlu.passes.serialize_to_mlisa(pm)
    pm.run(mod)

    return mlu.get_mlisa_from_module(mod).decode('utf-8')

@staticmethod
def make_cnbin(src, opt, capability):
    cnas, _ = path_to_binary("cnas")
    cnlink, _ = path_to_binary("cnlink")

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.mlisa') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='a+', suffix='.log') as flog:
        fsrc.write(src)
        fsrc.flush()
        fbin = fsrc.name + '.cnbin'
        ffatbin = fsrc.name + '.cnfatbin'

        opt_level = opt.opt_level
        if opt_level == "Om":
            opt_level = "O3"
        debug = []
        # We only enable -g debugging when opt_level == O0.
        if opt_level == "O0" and opt.debug:
            debug = ['-g']
        arch = stringify_arch(capability)
        line_info = [] if os.environ.get('TRITON_DISABLE_LINE_INFO') else [
            '-lineinfo'
        ]
        cnas_cmd = [
            cnas, *line_info, *debug, f'-{opt_level}', '--verbose', '-a', arch,
            '-i', fsrc.name, '-o', fbin
        ]
        # FIXME: remove cnlink when cnModuleLoadData support cnbin.
        cnlink_cmd = [cnlink, '--fatbin', '-i', fbin, '-o', ffatbin]
        try:
            subprocess.run(cnas_cmd,
                            check=True,
                            close_fds=False,
                            stdout=flog,
                            stderr=flog)
            subprocess.run(cnlink_cmd,
                            check=True,
                            close_fds=False,
                            stdout=flog,
                            stderr=flog)
            if os.path.exists(fsrc.name):
                os.remove(fsrc.name)
            if os.path.exists(fbin):
                os.remove(fbin)
            if os.path.exists(flog.name):
                os.remove(flog.name)
        except subprocess.CalledProcessError as e:
            with open(flog.name) as log_file:
                log = log_file.read()
            if os.path.exists(flog.name):
                os.remove(flog.name)

            meminfo = _extract_memory_info(log)
            for info in meminfo:
                memory_type, used, avail, total = info
                if int(avail) < 0:
                    raise OutOfResources(int(used), int(total), memory_type)

            raise RuntimeError(
                f'`cnas+cnlink` failed with error code {e.returncode}: \n{log}\n'
                f'Repro cnas command: {" ".join(cnas_cmd)}\n'
                f'Repro cnlink command: {" ".join(cnlink_cmd)}\n')

        with open(ffatbin, 'rb') as f:
            cnfatbin = f.read()
        if os.path.exists(ffatbin):
            os.remove(ffatbin)

    return cnfatbin
