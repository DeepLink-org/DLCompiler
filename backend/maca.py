from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, metax

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


@functools.lru_cache()
def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return bin, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_ptxas_version():
    version = subprocess.check_output([_path_to_binary("ptxas")[0], "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
    

def maca_get_kernel_name(src: str) -> str:
    '''
    Get kernel name from llvm ir.
    This Kernel name is required when launching the kernel.
    '''
    assert src
    import re
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith('define metaxgpu_kernel void @'):
            return re.match(r"define metaxgpu_kernel void @(.+?)\(", line).groups()[0]

def parse_option(string):
    return [item for item in string.split(';') if item]


@dataclass(frozen=True)
class MACAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'maca'
    # MACA: new args
    pipeline: str = "basic"
    scenario: str = ""
    extra_options: str = ""
    pipeline_load_num: int = -1

    def __post_init__(self):
        default_libdir = os.getenv("MACA_PATH") + '/lib'
        ext_default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            # ext_maca_mathlib.bc
            env_ext_libdevice_path = os.getenv("TRITON_EXT_LIBDEVICE_PATH", None)
            ext_libdevice_path = env_ext_libdevice_path if env_ext_libdevice_path is not None else str(ext_default_libdir) + '/ext_maca_mathlib.bc'
            assert os.path.exists(ext_libdevice_path), "ext_maca_mathlib.bc do not exit, please check!"
            extern_libs['ext_libdevice'] = ext_libdevice_path
            # maca_kernellib.bc
            env_kernel_libdevice_path = os.getenv("TRITON_KERNEL_LIBDEVICE_PATH", None)
            kernel_libdevice_path = env_kernel_libdevice_path if env_kernel_libdevice_path is not None else default_libdir + '/maca_kernellib.bc'
            extern_libs['kernel_libdevice'] = kernel_libdevice_path
            # maca_mathlib.bc
            env_libdevice_path = os.getenv("TRITON_LIBDEVICE_PATH", None)
            libdevice_path = env_libdevice_path if env_libdevice_path is not None else default_libdir + '/maca_mathlib.bc'
            extern_libs['libdevice'] = libdevice_path
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and self.num_warps <= 16 and (self.num_warps & (self.num_warps - 1)) == 0, \
                "num_warps must be a power of 2 or greater than 0 and less than or equal to 16"

    def hash(self):
        hash_dict = dict(self.__dict__)
        # hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MACABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'maca'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        print(f"zmz debug self.capability: {self.capability}")
        assert isinstance(self.capability, int)
        self.binary_ext = "mcfatbin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in MACAOptions.__dataclass_fields__.keys() if k in opts}
        # USE_MACA: support allow_fp8e4nv(i.e. float8_e4m3fn)
        args["allow_fp8e4nv"] = True
        # args["allow_fp8e4nv"] = False
        args["allow_fp8e4b15"] = False
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return MACAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        import triton.language.extra.cuda as cuda
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if self.capability >= 80 else cuda.convert_custom_float8_sm70
        }
        return codegen_fns

    def load_dialects(self, ctx):
        metax.load_dialects(ctx)


@staticmethod
def make_ttir(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)
    return mod

@staticmethod
def make_ttgir(mod, metadata, opt, capability):
    assert opt.pipeline_load_num >= -1, "invalid pipeline_load_num value!"
    scenarios = parse_option(opt.scenario)
    disable_prefetch = "unprefetch" in scenarios 
    fullstage = "fullstage" in scenarios
    store_coalesce = "storeCoalesce" in scenarios
    mla = "mla" in scenarios
    single_shm = "singleshm" in scenarios
    use_opt_maca_mma = True
    use_opt_maca_mma = (opt.pipeline != "" and not os.getenv("TRITON_DISABLE_MACA_OPT_MMA"))
    # TTIR -> TTGIR
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 64, opt.num_ctas)
    # optimize TTGIR
    passes.ttgpuir.add_coalesce(pm)
    if capability // 10 >= 8:
        passes.ttgpuir.add_f32_dot_tc(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    
    if opt.pipeline == "cpasync" :
        disable_prefetch = True
    metax.passes.ttgpuir.add_accelerate_matmul(pm, opt.num_stages, disable_prefetch, store_coalesce, "c500")
    passes.ttgpuir.add_remove_layout_conversions(pm)
    if store_coalesce:
        metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_from_repn_to_elemn_pass(pm)
        metax.passes.ttgpuir.add_tritonmetaxgpu_optimize_cstore_pass(pm, opt.num_stages)
        passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.common.add_cse(pm)
    if capability // 10 >= 8:
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        if use_opt_maca_mma:
            if opt.pipeline == "basic":
                if mla and single_shm:
                    # only mla=True and single_shm=True
                    metax.passes.ttgpuir.add_pipeline_maca_4(pm, opt.num_stages, opt.pipeline_load_num, fullstage, True)
                else:
                    metax.passes.ttgpuir.add_pipeline_maca_4(pm, opt.num_stages, opt.pipeline_load_num, fullstage, False)
            elif opt.pipeline == "cpasync" and not mla:
                metax.passes.ttgpuir.add_pipeline_async_tn(pm, opt.num_stages)
                metax.passes.ttgpuir.add_pipeline_async_tt(pm, opt.num_stages)
                metax.passes.ttgpuir.add_pipeline_async_base(pm, opt.num_stages, fullstage)
            elif mla and opt.num_stages == 2 and opt.pipeline == "cpasync":
                metax.passes.ttgpuir.add_pipeline_async_multidot_mla_mixed(pm, opt.num_stages, fullstage, opt.pipeline_load_num, single_shm, True)
            elif mla and opt.num_stages == 2 and opt.pipeline == "mixed":
                metax.passes.ttgpuir.add_pipeline_async_multidot_mla_mixed(pm, opt.num_stages, fullstage, opt.pipeline_load_num, single_shm, False)
            else:
                print("no avalilable pipeline for maca")
        else:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages)
    if use_opt_maca_mma and opt.pipeline == "basic" and "unprefetch" not in scenarios:
        metax.passes.ttgpuir.add_prefetch_maca_2(pm)
    elif not use_opt_maca_mma:
        passes.ttgpuir.add_prefetch(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    if os.getenv("TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP"):
        metax.passes.ttgpuir.add_tritonmetaxgpu_move_dot_operands_out_loop_pass(pm)
    if os.getenv("TRITON_ENABLE_MACA_MERGE_EQUAL_SHARED_LAYOUT"):
        metax.passes.ttgpuir.add_tritonmetaxgpu_merge_equal_shared_layout_pass(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(mod)
    return mod

@staticmethod
def make_mlir(src, metadata, options, capability):
    # warp-specialization mutates num_warps
    num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
    if num_warp_groups is not None:
        metadata["num_warps"] *= num_warp_groups
    mod = src

    # TritonGPU -> LLVM-IR (MLIR)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.convert.add_scf_to_cf(pm)
    passes.convert.add_index_to_llvmir(pm)
    passes.ttgpuir.add_allocate_shared_memory(pm)
    metax.passes.ttgpuir.add_to_llvmir(pm, capability)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
        passes.llvmir.add_di_scope(pm)
    pm.run(mod)

    # Get some metadata
    metadata["shared"] = src.get_int_attr("triton_gpu.shared")
    ret = str(mod)
    return ret

@staticmethod
def make_llir(src, metadata, options, capability):
    mlir_opt_path = os.path.dirname(__file__) + "/bin/mlir-opt"
    opted_mlir = metax.mlir_opt(src, mlir_opt_path)
    mlir_translate_path = os.path.dirname(__file__) + "/bin/mlir-translate"
    maca_path = os.environ.get('MACA_PATH')
    assert maca_path, "Not found MACA_PATH"
    llir = metax.translate_mlir_to_llir(opted_mlir, maca_path)
    if options.extern_libs:
        paths = [path for (name, path) in options.extern_libs]
        llir = metax.link_extern_libs(llir, paths, maca_path)
    metadata["name"] = maca_get_kernel_name(llir)
    return llir


@staticmethod
def make_mcfatbin(src, metadata, opt, capability):
    scenarios = parse_option(opt.scenario)
    opt_mxcc = os.environ.get("TRITON_COMPILER_OPT_PATH")
    mxcc_arch = os.environ.get('MACA_PATH') + "/mxgpu_llvm/bin/mxcc"
    if opt_mxcc:
        mxcc_arch = opt_mxcc + "/mxgpu_llvm/bin/mxcc"
    if mxcc_arch is None:
        raise RuntimeError('mxcc_arch is None (not specified)')
    compile_options = ""
    if (opt.pipeline == "basic" or opt.pipeline == "basic-prefetch") and "mla" not in scenarios:
        compile_options = " -mllvm -metaxgpu-sched-regpressure=false -mllvm -metaxgpu-PostRA-Scheduler=false -mllvm -metaxgpu-mma-sched=true "
        if "fullstage" in scenarios: 
            compile_options += " -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-igroup "
        else:
            compile_options += " -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-sched-mma-maxnum=3 "
        if "roll" not in scenarios: 
            compile_options += " -mllvm -metaxgpu-mma-unroll-count=" + str(opt.num_stages) + " "
    elif opt.pipeline == "cpasync" and "mla" not in scenarios:
        compile_options = " -mllvm -metaxgpu-sched-regpressure=true "
        compile_options += " -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-igroup -mllvm -metaxgpu-aggressive-4g-addr-opt=true \
                            -mllvm -metaxgpu-shl-add-combine=false -mllvm -misched-postra=true -mllvm -enable-post-misched=true "
        
        if os.getenv("TRITON_ENABLE_MACA_COMPILER_INT8_OPT"):
            compile_options += " -mllvm -metaxgpu-slp-vectorize-i8=true"

        if "unroll" in scenarios:
            compile_options += " -mllvm -metaxgpu-mma-unroll-count=" + str(opt.num_stages) + " "
    if "flashattn-fwd" in scenarios:
        compile_options = " -mllvm -metaxgpu-mma-sched=true -mllvm -metaxgpu-sched-select=metaxgpu-minreg -mllvm -map-use-pk-fma=1 "
    elif "flashattn-bwd" in scenarios:
        compile_options = " -mllvm -metaxgpu-sched-regpressure=true "
        compile_options += " -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true "
    if "mla" in scenarios:
        # maybe will change the compile options in mla later
        if opt.num_stages == 2:
            if opt.pipeline == "cpasync":
                compile_options = " -mllvm -metaxgpu-sched-regpressure=true "
                compile_options += " -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-igroup -mllvm -metaxgpu-aggressive-4g-addr-opt=true \
                                    -mllvm -metaxgpu-shl-add-combine=false -mllvm -misched-postra=true -mllvm -enable-post-misched=true "
                if "unroll" in scenarios:
                    compile_options += " -mllvm -metaxgpu-mma-unroll-count=" + str(opt.num_stages) + " "
            elif opt.pipeline == "basic" or opt.pipeline == "mixed":
                compile_options = " -mllvm -metaxgpu-mma-sched=true -mllvm -map-use-pk-fma=1 -mllvm -metaxgpu-split-regalloc=true -mllvm -metaxgpu-aggressive-fold=true \
                                    -mllvm -metaxgpu-disable-licm=true "
            else:
                assert False, "Please set pipeline for mla!"
        else:
            compile_options = " -mllvm -metaxgpu-mma-sched=true -mllvm -map-use-pk-fma=1 -mllvm -metaxgpu-split-regalloc=true -mllvm -metaxgpu-aggressive-fold=true "
    if opt.extra_options != "":
        compile_options = opt.extra_options
    return metax.translate_llvmir_to_mcfatbin(src, mxcc_arch, os.environ.get('MACA_PATH'), compile_options)
    

    # def add_stages(self, stages, options):
    #     stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
    #     stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
    #     stages["mlir"] = lambda src, metadata: self.make_mlir(src, metadata, options, self.capability)
    #     stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
    #     stages["mcfatbin"] = lambda src, metadata: self.make_mcfatbin(src, metadata, options, self.capability)

    # @functools.lru_cache()
    # def hash(self):
    #     mxcc_arch = os.environ.get('MACA_PATH') + "/mxgpu_llvm/bin/mxcc"
    #     if mxcc_arch is None:
    #         raise RuntimeError('mxcc_arch is None (not specified)')
    #     version = subprocess.check_output([mxcc_arch, "--version"]).decode("utf-8").split('\n', 1)[0]
    #     return f'{version}-{self.capability}'


##################################################################################################



import functools
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
# libraries = ['cuda']
libraries = []

@functools.lru_cache()
def maca_home_dirs():
    return os.getenv("MACA_PATH")

@functools.lru_cache()
def libmaca_dirs():
    maca_path = maca_home_dirs()
    return ["{}/lib/".format(maca_path)]

maca_lib_dir = libmaca_dirs()
maca_include_dir = [os.path.join(maca_home_dirs(), "include")]


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libmaca_dirs()]


def compile_module_from_src(src, name):
    # print(f"zmz debug compile_module_from_src: {src}")
    print(f"zmz debug name, {name}")
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            # TODO(MACA): fix it
            so = _build(name, src_path, tmpdir, library_dirs(), maca_include_dir, libraries)
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


class MacaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MacaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "maca.c")).read_text(), "maca_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        # self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        # self.fill_1d_tma_descriptor = mod.fill_1d_tma_descriptor
        # self.fill_2d_tma_descriptor = mod.fill_2d_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "mcDeviceptr_t"
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
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
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
        }[ty]

    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    # generate glue code
    params = [i for i in signature.keys() if i not in constants]
    src = f"""
#include <mcr/mc_runtime.h>
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(mcError_t code, const char *file, int line)
{{
   if (code != mcSuccess)
   {{
      const char* prefix = "Triton Error [MACA]: ";
      const char* str = mcGetErrorString(code);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define MACA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, mcStream_t stream, mcFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
    assert(num_ctas == 1);
    MACA_CHECK(mcModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    mcDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
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
    Py_DECREF(ret);  // Thanks ChatGPT!
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
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (mcStream_t)_stream, (mcFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});
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


class MacaLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class MacaDriver(GPUDriver):

    def __init__(self):
        self.utils = MacaUtils()  # TODO: make static
        self.launcher_cls = MacaLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 64
        return GPUTarget("maca", capability, warp_size)

    @staticmethod
    def is_active():
        import torch
        return torch.cuda.is_available() and (torch.version.hip is None)
