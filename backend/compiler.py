from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path
from triton.backends.dicp_triton.driver import DICPDriver
from typing import Any, Tuple

def _get_dicp_triton_opt_path() -> str:
    path = os.getenv("DICP_TRITON_OPT_PATH", "")
    if path == "":
        raise Exception("DICP_TRITON_OPT_PATH is not set.")
    return path

def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

def _get_triton_linalg_opt_path() -> str:
    # path = os.getenv("TRITON_LINALG_OPT_PATH", "")
    path = "triton-shared-opt"
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path 

def _ttir_to_linalgdir(mod):
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "triton_linalg.mlir")
        Path(src_path).write_text(ttir_code)
        triton_linalg_opt_path = _get_triton_linalg_opt_path()
        subprocess.check_call([triton_linalg_opt_path, src_path, "--triton-to-linalg", "-o", dst_path])
        return Path(dst_path).read_text()

def _optimize_ttlinalgdir(ttlinalgdir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttlinalgdir

# get kernel name to recompile kernel
def _linalgir_get_kernel_name(ttir: str) -> str:
    '''
    Get kernel name from ttir.
    This Kernel name is required when launching the kernel.
    '''
    for line in ttir.split('\n'):
        line = line.strip()
        if line.startswith('func.func'):
            return line.split('@')[1].split("(")[0]
    raise RuntimeError("can not get kernel name from ttir")

def _ttir_get_kernel_name(ttir: str):
    '''
    Get kernel name from ttir.
    This Kernel name is required when launching the kernel.
    '''
    for line in ttir.split('\n'):
        line = line.strip()
        if line.startswith('tt.func'):
            return line.split('@')[1].split("(")[0]
    return None

# call llvm compiler to generate bin file
def _linalg_to_fatbin(ttlinalgdir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "temp_linalg.mlir")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(ttlinalgdir)
        # llc_path = _get_llvm_bin_path("llc")
        # subprocess.check_call([llc_path, src_path, "-o", dst_path])
        # Actually it's text-format assembly.  Use read_text().
        return ttlinalgdir


@dataclass(frozen=True)
class DICPOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")      
    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

class DICPBackend(BaseBackend):
    binary_ext = "ttlinalgdir"
    def __init__(self, target:str) -> None:
        # if target is "mlu":
        #     device_type = "370"
        #     MLUBackend().__init__(device_type)
        super().__init__(target)
        self.driver = DICPDriver(target)
        if self.driver.target == 'dicp':
            self.binary_ext = "ttlinalgdir"
        elif self.driver.target == 'mlu':
            self.binary_ext = "cnbin"
        elif self.driver.target == 'maca':
            self.binary_ext = "mcfatbin"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ['dicp', 'mlu', 'maca']

    @staticmethod
    def make_ttir(mod, metadata, opt):
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
        metadata["name"] = _ttir_get_kernel_name(str(mod))
        metadata["shared"] = 0
        return mod

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        if self.driver.target == 'dicp':
            stages["ttlinalgdir"] = lambda src, metadata: _optimize_ttlinalgdir(_ttir_to_linalgdir(src))
            stages["fatbin"] = lambda src, metadata: _linalg_to_fatbin(src, metadata)
        elif self.driver.target == 'mlu':
            from triton.backends.dicp_triton.mlu import ttir_to_cnfatbin, get_architecture_descriptor
            stages["cnbin"] = lambda src, metadata: ttir_to_cnfatbin(src, metadata, get_architecture_descriptor(self.driver, options), False, True)
        elif self.driver.target == 'maca':
            from triton.backends.dicp_triton.maca import ttir_to_ttgir, optimize_ttgir, ttgir_to_llir, llir_to_mcfatbin, get_architecture_descriptor
            arch = get_architecture_descriptor()
            extern_libs = dict()
            stages["ttgir"] = lambda src, metadata: optimize_ttgir(ttir_to_ttgir(src, 4), options.num_stages, arch)
            stages["llir"] = lambda src, metadata: ttgir_to_llir(src, arch)
            mxcc_arch = os.environ.get('MACA_PATH') + "/mxgpu_llvm/bin/mxcc"
            if mxcc_arch is None:
                raise RuntimeError('mxcc_arch is None (not specified)')
            stages["mcfatbin"] = lambda src, metadata: llir_to_mcfatbin(src, mxcc_arch, os.environ.get('MACA_PATH'))
        else:
            raise RuntimeError("backend not supported")

    def load_dialects(self, ctx):
        return
    
    @functools.lru_cache()
    def hash(self):
        return self.target
    
    def get_driver(self):
        return self.driver
    
    # parse  add_kernel[(16,)](x, y, output, n_elements, BLOCK_SIZE=1024)
    def parse_options(self, options: dict) -> Any:
        args = {'arch': self.target}
        args.update({k: options[k] for k in DICPOptions.__dataclass_fields__.keys() if k in options})
        return DICPOptions(**args)
    
    def get_codegen_implementation(self):
        codegen_fns = dict()
        return codegen_fns
    
    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )
