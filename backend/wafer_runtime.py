import ctypes
import importlib.util
import os
import shutil
import subprocess
import sysconfig
import tempfile
import weakref
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from triton.runtime.cache import get_cache_manager

from .wafer_cache import cache_digest, file_fingerprint


def _sdk_path(name):
    root = os.getenv("KUIPER_ROOT")
    if not root:
        raise RuntimeError("KUIPER_ROOT is not set; source init_wafer_env.sh first.")
    return os.path.join(root, name)


class _KuiperRuntime:
    def __init__(self):
        self.library = ctypes.CDLL(os.path.join(_sdk_path("lib"), "libhpgr.so"))
        self.library.txGetDevice.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.library.txGetDevice.restype = ctypes.c_int
        self.library.txSetDevice.argtypes = [ctypes.c_uint32]
        self.library.txSetDevice.restype = ctypes.c_int

    def current_device(self):
        device = ctypes.c_uint32()
        status = self.library.txGetDevice(ctypes.byref(device))
        if status != 0:
            raise RuntimeError(f"txGetDevice failed with status 0x{status:x}")
        return device.value

    def set_device(self, device):
        status = self.library.txSetDevice(device)
        if status != 0:
            raise RuntimeError(f"txSetDevice failed with status 0x{status:x}")

    def current_stream(self, device=None):
        # Without torch_txda there is no framework stream context.
        return None


def get_runtime():
    try:
        import torch
        import torch_txda  # noqa: F401

        if hasattr(torch, "txda"):
            return torch.txda
    except (ImportError, AttributeError):
        pass
    return _KuiperRuntime()


def _launcher_compiler():
    compiler = os.getenv("CXX") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Failed to find a C++ compiler; set CXX.")
    return shutil.which(compiler) or compiler


def _build_launcher(name, source, directory):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = os.path.join(directory, f"{name}{suffix}")
    compiler = _launcher_compiler()
    include_dirs = [_sdk_path("include"), sysconfig.get_path("include")]
    library_dirs = [_sdk_path("lib")]
    libraries = ["hpgr"]
    command = [
        compiler,
        source,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-Wno-psabi",
        "-o",
        output,
    ]
    command += [f"-I{path}" for path in include_dirs]
    command += [f"-L{path}" for path in library_dirs]
    command += [f"-l{library}" for library in libraries]
    subprocess.check_call(command)
    return output


def _launcher_cache_key(source):
    headers = sorted(Path(_sdk_path("include")).rglob("*.h"))
    return cache_digest(
        {
            "source": source,
            "compiler": file_fingerprint(_launcher_compiler()),
            "python": [
                sysconfig.get_config_var("SOABI"),
                sysconfig.get_config_var("EXT_SUFFIX"),
                file_fingerprint(Path(sysconfig.get_path("include")) / "Python.h"),
                file_fingerprint(sysconfig.get_config_h_filename()),
            ],
            "sdk_headers": [file_fingerprint(path) for path in headers],
            "runtime": file_fingerprint(_sdk_path("lib/libhpgr.so")),
        }
    )


def compile_launcher(source):
    name = "__triton_launcher"
    cache = get_cache_manager(_launcher_cache_key(source))
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, f"{name}.cpp")
            Path(source_path).write_text(source, encoding="utf-8")
            shared_object = _build_launcher(name, source_path, directory)
            cache_path = cache.put(
                Path(shared_object).read_bytes(), f"{name}.so", binary=True
            )
    spec = importlib.util.spec_from_file_location(name, cache_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpp_type(type_name):
    if type_name.startswith("*"):
        return "PyObject*"
    if type_name in ("fp16", "bf16"):
        raise NotImplementedError(
            "Wafer fp16/bf16 scalar packing is not implemented; pass an fp32 scalar "
            "and cast inside the kernel. fp16/bf16 tensor pointers are supported."
        )
    return {
        "i1": "int32_t",
        # Parse signed narrow values as C int, then copy their low bytes into
        # the 64-bit slot. KernelArgBufferPass loads the declared scalar width.
        "i8": "int32_t",
        "i16": "int32_t",
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
    }[type_name]


def _parse_format(type_name):
    if type_name.startswith("*"):
        return "O"
    return {
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "L",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
        "float": "f",
        "double": "d",
    }[_cpp_type(type_name)]


def make_launcher(signature, launch_mode="simt"):
    if launch_mode not in ("simt", "cluster"):
        raise ValueError(f"Unknown Wafer launch mode: {launch_mode}")
    cluster_check = ""
    launch_function = "txLaunchKernelGGL"
    cluster_argument = ""
    if launch_mode == "cluster":
        launch_function = "txLaunchClusterKernelGGL"
        cluster_argument = "dim3({1, 1, 1}), "
        cluster_check = '''
    if (grid_y != 1 || grid_z != 1 || grid_x > 16) {
        PyErr_SetString(PyExc_ValueError, "Wafer cluster grid must be (1..16, 1, 1)"); return NULL;
    }
'''
    declarations = " ".join(
        f"{_cpp_type(type_name)} arg{index};" for index, type_name in signature.items()
    )
    parse_format = "iiiOKOOOO" + "".join(
        _parse_format(type_name) for type_name in signature.values()
    )
    parse_args = "".join(f", &arg{index}" for index in signature)
    pointer_setup = "\n".join(
        f"void *ptr{index} = get_pointer(arg{index}); if (PyErr_Occurred()) return NULL;"
        for index, type_name in signature.items()
        if type_name.startswith("*")
    )
    kernel_args = "\n".join(
        (
            f"runtime_args.push_back(1); runtime_args.push_back((uint64_t)ptr{index});"
            if type_name.startswith("*")
            else f"uint64_t scalar{index} = 0; memcpy(&scalar{index}, &arg{index}, sizeof(arg{index})); runtime_args.push_back(scalar{index});"
        )
        for index, type_name in signature.items()
    )
    return f"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "tx_runtime.h"

static void *get_pointer(PyObject *object) {{
    if (object == Py_None) return nullptr;
    if (PyLong_Check(object)) return PyLong_AsVoidPtr(object);
    PyObject *value = PyObject_CallMethod(object, "data_ptr", nullptr);
    if (!value) return nullptr;
    void *pointer = PyLong_AsVoidPtr(value);
    Py_DECREF(value);
    return pointer;
}}

static PyObject *launch(PyObject *, PyObject *args) {{
    int grid_x, grid_y, grid_z;
    unsigned long long function;
    PyObject *stream_object, *kernel_metadata, *launch_metadata, *enter_hook, *exit_hook;
    {declarations}
    if (!PyArg_ParseTuple(args, "{parse_format}", &grid_x, &grid_y, &grid_z, &stream_object, &function,
            &kernel_metadata, &launch_metadata, &enter_hook, &exit_hook{parse_args})) return NULL;
    if (grid_x < 0 || grid_y < 0 || grid_z < 0) {{
        PyErr_SetString(PyExc_ValueError, "Wafer grid dimensions must be nonnegative"); return NULL;
    }}
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) Py_RETURN_NONE;
    {cluster_check}
    txStream_t stream = stream_object == Py_None ? nullptr : (txStream_t)PyLong_AsVoidPtr(stream_object);
    if (PyErr_Occurred()) return NULL;
    if (enter_hook != Py_None) {{
        PyObject *result = PyObject_CallFunctionObjArgs(enter_hook, launch_metadata, NULL);
        if (!result) return NULL;
        Py_DECREF(result);
    }}
    {pointer_setup}
    std::vector<uint64_t> runtime_args;
    {kernel_args}
    runtime_args.insert(runtime_args.end(), {{(uint64_t)grid_x, (uint64_t)grid_y, (uint64_t)grid_z, 0, 0, 0}});
    PyObject *path_object = PyObject_GetAttrString(kernel_metadata, "kernel_path");
    if (!path_object) return NULL;
    PyObject *name_object = PyObject_GetAttrString(kernel_metadata, "name");
    if (!name_object) {{ Py_DECREF(path_object); return NULL; }}
    const char *kernel_path = PyUnicode_AsUTF8(path_object);
    if (!kernel_path) {{ Py_DECREF(path_object); Py_DECREF(name_object); return NULL; }}
    const char *kernel_name = PyUnicode_AsUTF8(name_object);
    if (!kernel_name) {{ Py_DECREF(path_object); Py_DECREF(name_object); return NULL; }}
    void *binary = nullptr;
    size_t size = 0;
    if (!function) {{
    FILE *file = fopen(kernel_path, "rb");
    if (!file) {{ PyErr_SetFromErrnoWithFilename(PyExc_OSError, kernel_path); Py_DECREF(path_object); Py_DECREF(name_object); return NULL; }}
    long length = -1;
    if (fseek(file, 0, SEEK_END) == 0) length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {{
        PyErr_Format(PyExc_OSError, "Invalid or empty Wafer kernel: %s", kernel_path);
        fclose(file); Py_DECREF(path_object); Py_DECREF(name_object); return NULL;
    }}
    size = (size_t)length;
    binary = malloc(size);
    if (!binary || fread(binary, 1, size, file) != size) {{ fclose(file); free(binary); Py_DECREF(path_object); Py_DECREF(name_object); PyErr_SetString(PyExc_RuntimeError, "Failed to read Wafer kernel"); return NULL; }}
    fclose(file);
    }}
    txError_t status;
    // The argument tuple keeps tensor objects alive while the GIL is released.
    // Both paths remain synchronous, including stream errors and module lifetime.
    Py_BEGIN_ALLOW_THREADS
    if (function) {{
        status = {"txLaunchClusterKernel" if launch_mode == "cluster" else "txLaunchKernel"}(
            (txFunction_t)function, {cluster_argument}
            dim3({{(uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z}}), dim3({{1, 1, 1}}),
            runtime_args.data(), runtime_args.size() * sizeof(uint64_t), 0, stream);
    }} else {{
        status = {launch_function}(kernel_name, (uint64_t)binary, size, {cluster_argument}
            dim3({{(uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z}}), dim3({{1, 1, 1}}),
            runtime_args.data(), runtime_args.size() * sizeof(uint64_t), 0, stream);
    }}
    if (status == TX_SUCCESS) status = txStreamSynchronize(stream);
    Py_END_ALLOW_THREADS
    free(binary);
    if (status != TX_SUCCESS) {{
        PyErr_Format(PyExc_RuntimeError, "Wafer kernel %s (%s) failed with Kuiper status 0x%x, stream=%p",
                     kernel_name, kernel_path, (unsigned int)status, (void*)stream);
    }}
    Py_DECREF(path_object);
    Py_DECREF(name_object);
    if (status != TX_SUCCESS) return NULL;
    if (exit_hook != Py_None) {{
        PyObject *result = PyObject_CallFunctionObjArgs(exit_hook, launch_metadata, NULL);
        if (!result) return NULL;
        Py_DECREF(result);
    }}
    Py_RETURN_NONE;
}}
static PyMethodDef methods[] = {{{{"launch", launch, METH_VARARGS, "Launch a Wafer kernel"}}, {{NULL, NULL, 0, NULL}}}};
static struct PyModuleDef module = {{PyModuleDef_HEAD_INIT, "__triton_launcher", NULL, -1, methods}};
PyMODINIT_FUNC PyInit___triton_launcher(void) {{ return PyModule_Create(&module); }}
"""


def __getattr__(name):
    aliases = {"TXDAUtils": WaferUtils, "TXDALauncher": WaferLauncher}
    if name in aliases:
        return aliases[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class WaferUtils:
    def load_binary(self, name, kernel, shared_mem, device):
        api = os.getenv("WAFER_LAUNCH_API", "ggl")
        if api == "module":
            owner = _LoadedModule(name, kernel, device)
            return owner, owner.function, 0, 0, 1024
        if api != "ggl":
            raise ValueError("WAFER_LAUNCH_API must be 'ggl' or 'module'")
        # Kuiper loads the ELF during launch. Retain the binary as an opaque,
        # non-null lifetime token so CompiledKernel initializes only once.
        return kernel, 0, 0, 0, 1024

    def get_device_properties(self, device=None):
        return {"max_shared_mem": 3 * 1024 * 1024 - 2 * 0x10000}


class _LoadedModule:
    """Own one SDK module for CompiledKernel's lifetime, including ELF storage.

    Device launches synchronize before returning, so normal destruction cannot
    unload a module with an outstanding launch. The SDK ABI stays unchanged.
    """
    def __init__(self, name, binary, device):
        runtime = _KuiperRuntime()
        library = runtime.library
        library.txModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32]
        library.txModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        library.txModuleUnload.argtypes = [ctypes.c_void_p]
        for operation in (library.txModuleLoad, library.txModuleGetFunction, library.txModuleUnload):
            operation.restype = ctypes.c_int
        if not binary or len(binary) > 0xFFFFFFFF:
            raise ValueError("Wafer module ELF size must fit a nonzero uint32")
        self.binary = ctypes.create_string_buffer(binary)
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        previous = runtime.current_device()
        runtime.set_device(device)
        try:
            status = library.txModuleLoad(ctypes.byref(module), self.binary, len(binary))
            if status:
                raise RuntimeError(f"txModuleLoad({name}) failed with status 0x{status:x}")
            self._release = weakref.finalize(self, self._unload, runtime, device, module)
            status = library.txModuleGetFunction(ctypes.byref(function), module, name.encode())
            if status or not function.value:
                self._release()
                raise RuntimeError(f"txModuleGetFunction({name}) failed with status 0x{status:x}")
            self.function = function.value
        finally:
            runtime.set_device(previous)

    @staticmethod
    def _unload(runtime, device, module):
        previous = runtime.current_device()
        runtime.set_device(device)
        try:
            status = runtime.library.txModuleUnload(module)
            if status:
                raise RuntimeError(f"txModuleUnload failed with status 0x{status:x}")
        finally:
            runtime.set_device(previous)

    def close(self):
        self._release()


class SimulatorUtils:
    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so", delete=False) as file:
            file.write(kernel)
            path = file.name
        import ctypes

        module = ctypes.CDLL(path)
        os.unlink(path)
        function = ctypes.cast(getattr(module, name), ctypes.c_void_p).value
        return module, function, 0, 0, 1024

    def get_device_properties(self, device=None):
        return {"max_shared_mem": 3 * 1024 * 1024 - 2 * 0x10000}


class WaferLauncher:
    def __init__(self, src, metadata):
        argument_names = getattr(getattr(src, "fn", None), "arg_names", ())

        def argument_index(key):
            key = key[0] if isinstance(key, tuple) else key
            return argument_names.index(key) if isinstance(key, str) else int(key)

        signature = dict(
            sorted(
                (argument_index(index), type_name)
                for index, type_name in src.signature.items()
            )
        )
        constants = {argument_index(index) for index in getattr(src, "constants", {})}
        self.source_argument_count = len(argument_names) or (
            max(signature, default=-1) + 1
        )
        signature = {
            index: type_name
            for index, type_name in signature.items()
            if index not in constants
        }
        self.runtime_argument_indices = tuple(signature)
        self.metadata = metadata
        self.launch = compile_launcher(
            make_launcher(signature, getattr(metadata, "launch_mode", "simt"))
        ).launch

    def __call__(self, *args, **kwargs):
        arguments = list(args)
        count = len(arguments) - 9
        if count == self.source_argument_count:
            # JITFunction passes all bound arguments, including constexpr and
            # specialized values. CompiledKernel also accepts runtime-only args.
            arguments = arguments[:9] + [
                arguments[9 + index] for index in self.runtime_argument_indices
            ]
        elif count != len(self.runtime_argument_indices):
            raise TypeError(
                f"Wafer launcher expected {len(self.runtime_argument_indices)} runtime arguments "
                f"or {self.source_argument_count} source arguments, got {count}"
            )
        arguments[5] = self.metadata
        return self.launch(*arguments, **kwargs)


@lru_cache(maxsize=8)
def _noc_initializer(toolchain_key):
    from . import wafer

    declarations = {
        "module_init": "void @module_init(ptr)",
        "module_cleanup": "void @module_cleanup(ptr)",
        "__NoCRingInit": "void @__NoCRingInit()",
    }
    entry = "__wafer_noc_init"
    source = ""
    for name in (entry, *declarations):
        source += (f'@name_{name} = weak constant [{len(name) + 1} x i8] '
                   f'c"{name}\\00", section ".rodata.name", align 1\n')
        source += (f'@export_{name} = constant {{ptr, ptr}} '
                   f'{{ptr @{name}, ptr @name_{name}}}, section "ExportedDYNSYMTab", align 8\n')
    source += "\n".join("declare " + declaration for declaration in declarations.values())
    source += (f"\ndefine void @{entry}(ptr %args) {{\n"
               "  call void @__NoCRingInit()\n  ret void\n}\n")
    metadata = {"name": entry, "launch_mode": "cluster"}
    wafer.object_to_binary(wafer.llir_to_object(source, metadata, simulator=False),
                           metadata, simulator=False)
    src = SimpleNamespace(signature={})
    return WaferLauncher(src, SimpleNamespace(**metadata))


def initialize_noc(stream=None):
    """Clear the ring's two sync words on all 16 tiles and wait for completion.

    Call before a NoC collective, after previous device work has completed.
    The separate cluster kernel prevents late tile initialization from erasing
    a peer's request. Firmware recovery alone does not clear these SPM words.
    """
    from triton.backends.compiler import GPUTarget
    from .wafer import WaferBackend, runtime_binary_enabled, simulator_enabled

    if simulator_enabled() or not runtime_binary_enabled():
        raise RuntimeError("NoC initialization requires the Wafer hardware runtime")
    key = WaferBackend(GPUTarget("wafer", "wafer", 32)).hash()
    launcher = _noc_initializer(key)
    launcher(16, 1, 1, stream, 0, None, None, None, None)
