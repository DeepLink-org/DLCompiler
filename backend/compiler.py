from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path
from backend.driver import DICPDriver

def _get_dicp_triton_opt_path() -> str:
    path = os.getenv("DICP_TRITON_OPT_PATH", "")
    if path == "":
        raise Exception("DICP_TRITON_OPT_PATH is not set.")
    return path

def _get_triton_linalg_opt_path() -> str:
    # path = os.getenv("TRITON_LINALG_OPT_PATH", "")
    path = "/home/sheng.yuan/workspace/deeplink_triton/Triton/third_party/triton_linalg/triton/build/third_party/dicp_triton/third_party/triton_linalg/bin/triton-linalg-opt"
    if path == "":
        raise Exception("TRITON_LINALG_OPT_PATH is not set.")
    return path 

def _ttir_to_linalgdir(mod):
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "triton_linalg.mlir")
        Path(src_path).write_text(ttir_code)
        triton_linalg_opt_path = _get_triton_linalg_opt_path()
        import pdb
        pdb.set_trace()
        subprocess.check_call([triton_linalg_opt_path, src_path, "--triton-to-linalg", "-o", dst_path])
        print (Path(dst_path).read_text())
        return Path(dst_path).read_text()

def _optimize_ttlinalgdir(ttlinalgdir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttlinalgdir

def _linalg_to_fatbin(ttlinalgdir: str):
    return ttlinalgdir

def _optimize_fatbin(ttlinalgdir: str):
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
    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

class DICPBackend(BaseBackend):
    def __init__(self, target:str) -> None:
        # if target is "mlu":
        #     device_type = "370"
        #     MLUBackend().__init__(device_type)
        super().__init__(target)
        self.driver = DICPDriver()


    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'dicp'

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
        return mod
    
    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttlinalgdir"] = lambda src, metadata: _optimize_ttlinalgdir(_ttir_to_linalgdir(src))
        stages["fatbin"] = lambda src, metadata: _optimize_fatbin(_linalg_to_fatbin(src))

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
