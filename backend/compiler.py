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
from typing import Any, Tuple, Dict
from types import ModuleType


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
        subprocess.check_call(
            [triton_linalg_opt_path, src_path, "--triton-to-linalg", "-o", dst_path]
        )
        return Path(dst_path).read_text()


def _optimize_ttlinalgdir(ttlinalgdir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttlinalgdir


# get kernel name to recompile kernel
def _linalgir_get_kernel_name(ttir: str) -> str:
    """
    Get kernel name from ttir.
    This Kernel name is required when launching the kernel.
    """
    for line in ttir.split("\n"):
        line = line.strip()
        if line.startswith("func.func"):
            return line.split("@")[1].split("(")[0]
    raise RuntimeError("can not get kernel name from ttir")


def _ttir_get_kernel_name(ttir: str):
    """
    Get kernel name from ttir.
    This Kernel name is required when launching the kernel.
    """
    for line in ttir.split("\n"):
        line = line.strip()
        if line.startswith("tt.func"):
            return line.split("@")[1].split("(")[0]
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
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class DICPBackend(BaseBackend):
    binary_ext = "ttlinalgdir"

    def __init__(self, target: str) -> None:
        super().__init__(target)
        self.driver = DICPDriver(target)
        if self.driver.target == "dicp":
            self.binary_ext = "ttlinalgdir"
        elif self.driver.target == "mlu":
            self.capability = target.arch
            assert isinstance(self.capability, int)
            self.binary_ext = "cnbin"
        elif self.driver.target == "maca":
            self.capability = 80
            self.binary_ext = "mcfatbin"
        elif self.driver.target == "ascend":
            self.binary_ext = "npubin"
            self.capability = target.arch
        else:
            raise RuntimeError(f"Target '{self.target_type}' is not supported.")

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ["dicp", "mlu", "maca", "ascend"]

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

    def get_attrs_descriptor(self, params, args):
        if self.driver.target == "ascend":
            from triton.backends.dicp_triton.npu import AscendAttrsDescriptor

            return AscendAttrsDescriptor(params, args)
        else:
            raise RuntimeError(
                f"backend {self.driver.target} not supported for get_attrs_descriptor."
            )

    def add_stages(self, stages, options, language=None):
        if self.driver.target not in ["ascend", "mlu"]:
            stages["ttir"] = lambda src, metadata: self.make_ttir(
                src, metadata, options
            )
        if self.driver.target == "dicp":
            stages["ttlinalgdir"] = lambda src, metadata: _optimize_ttlinalgdir(
                _ttir_to_linalgdir(src)
            )
            stages["fatbin"] = lambda src, metadata: _linalg_to_fatbin(src, metadata)
        elif self.driver.target == "mlu":
            from triton.backends.dicp_triton.mlu import (
                onchip_mem_analysis,
                make_ttir,
                make_linalg,
                make_optimized_linalg,
                make_mluir,
                make_optimize_mluir,
                make_mlisa,
                make_cnbin,
            )

            stages["ttir"] = lambda src, metadata: make_ttir(
                src, metadata, options, self.capability
            )
            if options.onchip_mem_analysis:
                stages["onchip_mem_analysis"] = (
                    lambda src, metadata: onchip_mem_analysis(src, options)
                )
                return
            stages["linalg"] = lambda src, metadata: make_linalg(src, metadata, options)
            stages["linalgopt"] = lambda src, metadata: make_optimized_linalg(
                src, options
            )
            stages["mluir"] = lambda src, metadata: make_mluir(src, options)
            stages["mluiropt"] = lambda src, metadata: make_optimize_mluir(
                src, options, self.capability
            )
            stages["mlisa"] = lambda src, metadata: make_mlisa(src)
            stages["cnbin"] = lambda src, metadata: make_cnbin(
                src, options, self.capability
            )

            # from triton.backends.dicp_triton.mlu import ttir_to_cnfatbin, get_architecture_descriptor
            # stages["cnbin"] = lambda src, metadata: ttir_to_cnfatbin(src, metadata, get_architecture_descriptor(self.driver, options), False, True)
        elif self.driver.target == "maca":
            from triton.backends.dicp_triton.maca import (
                make_ttir,
                make_ttgir,
                make_mlir,
                make_llir,
                make_mcfatbin,
            )

            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: make_ttgir(
                src, metadata, options, self.capability
            )
            stages["mlir"] = lambda src, metadata: make_mlir(
                src, metadata, options, self.capability
            )
            stages["llir"] = lambda src, metadata: make_llir(
                src, metadata, options, self.capability
            )
            stages["mcfatbin"] = lambda src, metadata: make_mcfatbin(
                src, metadata, options, self.capability
            )
        elif self.driver.target == "ascend":
            from triton.backends.dicp_triton.npu import (
                make_ttir,
                ttir_to_linalg,
                ttir_to_ttsharedir_ascend,
                ttsharedir_to_linkedir,
                linalg_to_bin_enable_npu_compile,
            )

            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            lower_by_ttshared = os.getenv("LOWER_BY_TTSHARED", "1")
            if lower_by_ttshared == "0":
                if options.enable_npu_compile:
                    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
                        src, metadata, options, named_ops=True
                    )
                    stages["npubin"] = (
                        lambda src, metadata: linalg_to_bin_enable_npu_compile(
                            src, metadata, options, self.capability
                        )
                    )
            else:
                if options.enable_npu_compile:
                    stages["ttshared"] = (
                        lambda src, metadata: ttir_to_ttsharedir_ascend(
                            src, metadata, options, named_ops=True
                        )
                    )
                    stages["linkedir"] = lambda src, metadata: ttsharedir_to_linkedir(
                        src, metadata, options, named_ops=True
                    )
                    stages["npubin"] = (
                        lambda src, metadata: linalg_to_bin_enable_npu_compile(
                            src, metadata, options, self.capability
                        )
                    )
        else:
            raise RuntimeError("backend not supported")

    def load_dialects(self, ctx):
        # TODO  Warning If additional backends are integrated into the common IR with customized passes, their respective Dialect interfaces must be registered here. A decoupled registration mechanism for each backend is preferred to maintain modularity.
        if self.driver.target == "mlu":
            from triton._C.libtriton import mlu
            mlu.load_dialects(ctx)
        else:
            from triton._C.libtriton import dicp_triton
            dicp_triton.load_dialects(ctx)
        return

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_driver(self):
        return self.driver

    # parse  add_kernel[(16,)](x, y, output, n_elements, BLOCK_SIZE=1024)
    def parse_options(self, options: dict) -> Any:
        if self.target.backend == "ascend":
            from triton.backends.dicp_triton.npu import NPUOptions

            args = {
                k: options[k]
                for k in NPUOptions.__dataclass_fields__.keys()
                if k in options
            }
            options = NPUOptions(**args)
            return options
        elif self.target.backend == "mlu":
            from triton.backends.dicp_triton.mlu import MLUOptions

            args = {
                k: options[k]
                for k in MLUOptions.__dataclass_fields__.keys()
                if k in options
            }
            # When arch is less than mtp_5xx, tf32 is not supported, use fp32 for calculation.
            if "allowed_dot_input_precisions" not in args:
                if self.capability < 500:
                    args["allowed_dot_input_precisions"] = "ieee"

            if "supported_fp8_dtypes" not in args:
                supported_fp8_dtypes = set(MLUOptions.supported_fp8_dtypes)
                if self.capability >= 600:
                    supported_fp8_dtypes = supported_fp8_dtypes.union(
                        ("fp8e5", "fp8e4nv")
                    )
                args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

            args["max_num_imprecise_acc_default"] = 0

            if "enable_fp_fusion" not in args:
                args["enable_fp_fusion"] = (
                    os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"
                )

            if "enable_mlu_bound_check" not in args:
                args["enable_mlu_bound_check"] = (
                    os.getenv("TRITON_ENABLE_MLU_BOUND_CHECK", "0") == "1"
                )
            return MLUOptions(**args)
        elif self.target.backend == "maca":
            from triton.backends.dicp_triton.maca import MACAOptions

            # args = {k: options[k] for k in MACAOptions.__dataclass_fields__.keys() if k in options}
            # return MACAOptions(**args)
            args = {
                k: options[k]
                for k in MACAOptions.__dataclass_fields__.keys()
                if k in options
            }
            # USE_MACA: support allow_fp8e4nv(i.e. float8_e4m3fn)
            args["allow_fp8e4nv"] = True
            # args["allow_fp8e4nv"] = False
            args["allow_fp8e4b15"] = False
            args["max_num_imprecise_acc_default"] = (
                2**30 if self.capability == 90 else 0
            )
            return MACAOptions(**args)
        else:
            args = {"arch": self.target}
            args.update(
                {
                    k: options[k]
                    for k in DICPOptions.__dataclass_fields__.keys()
                    if k in options
                }
            )
            return DICPOptions(**args)

    def get_codegen_implementation(self, options=None):
        codegen_fns = dict()
        if self.target.backend == "ascend":
            from triton.backends.dicp_triton.npu import min_dot_size

            codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        elif self.target.backend == "mlu":
            from triton.backends.dicp_triton.mlu import min_dot_size

            codegen_fns = {
                "convert_custom_types": lambda arg, dst_ty: arg,
                "min_dot_size": min_dot_size(self.target),
            }
        elif self.target.backend == "maca":
            import triton.language.extra.cuda as cuda

            codegen_fns = {
                "convert_custom_types": (
                    cuda.convert_custom_float8_sm80
                    if self.capability >= 80
                    else cuda.convert_custom_float8_sm70
                )
            }
        return codegen_fns

    def pack_metadata(self, metadata):
        if self.target.backend == "ascend":
            from triton.backends.dicp_triton.npu import TRITON_PROFILER_REGISTERED

            # collect necessary metadata to launch kernels
            # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
            # Get this name as the kernel_name to CANN runtime.
            # kernel_name is unique to Ascend backend and should not be public.
            # CANN runtime limits the length of kernel name <= 50.
            # Considering '\n' is appended, thus the real kernel name <= 49.
            KERNEL_NAME_MAX_LEN = 49
            kernel_name_orig, mix_mode = metadata.name.split()
            if len(kernel_name_orig) > KERNEL_NAME_MAX_LEN:
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
        elif self.target.backend == "mlu":
            return (metadata.num_warps,)
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    @functools.lru_cache()
    def hash(self):
        if self.target.backend == "mlu":
            from triton.backends.dicp_triton.mlu import get_cnas_version

            version = get_cnas_version()
            return f"{version}-{self.capability}"
        # TODO fetch compiler version
        version_key = self.target
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        if self.target.backend == "mlu":
            from triton.language.extra.mlu import libdevice

            return {"triton.language.extra.libdevice": libdevice}
        return {}
