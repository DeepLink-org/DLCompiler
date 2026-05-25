import functools
import os
import tempfile
from pathlib import Path
from typing import Any

from triton._C.libtriton import ir, dicp_triton, passes
from ..compiler import DICPOptions
from ..driver import DICPDriver
from ..utils import get_current_backend

replace_commonir_ir = os.environ.get("DLC_REPLACE_COMMON_IR_FILE", None)
replace_commonir_linked_ir = os.environ.get("DLC_REPLACE_COMMONIR_LINKED_IR_FILE", None)


def commonir_to_linkedir(commonir, metadata, opt, named_ops=True):
    if replace_commonir_ir is not None:
        print(f"[DEBUG] Replace common ir with {replace_commonir_ir}")
        commonir = Path(replace_commonir_ir).read_text()

    compile_on_910_95 = metadata["compile_on_910_95"]
    enable_nd2nz_on_vector = metadata["enable_nd2nz_on_vector"]
    enable_select_analysis = metadata["enable_select_analysis"]

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.commonir.mlir")
        Path(src_path).write_text(commonir)
        context = ir.context()
        commonir_backend.load_dialects(context)
        mod = ir.parse_mlir_module(src_path, context)
        pm = ir.pass_manager(context)
        dicp_triton.passes.commonir.add_vectorize_parallel_loop(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        dicp_triton.passes.commonir.add_annotate_kernel_attrs(pm)
        dicp_triton.passes.ttir.add_triton_to_linalg(
            pm,
            True,
            named_ops,
            enable_nd2nz_on_vector,
            enable_select_analysis,
            compile_on_910_95,
        )
        dicp_triton.passes.ttir.add_ascend_npu_ir_legalize(pm, False)
        pm.run(mod)
        content = str(mod)
        print(content)

    if replace_commonir_linked_ir is not None:
        print(f"[DEBUG] Replace Linkedir with {replace_commonir_linked_ir}")
        return Path(replace_commonir_linked_ir).read_text()
    return content


class CommonIRBackend:
    binary_ext = "ttlinalgdir"

    def __init__(self) -> None:
        target = get_current_backend()
        self.driver = DICPDriver(target)
        self.target = target
        if self.driver.target == "mlu":
            self.capability = target.arch
            assert isinstance(self.capability, int)
            self.binary_ext = "cnbin"
        elif self.driver.target == "maca":
            self.capability = 80
            self.binary_ext = "mcfatbin"
        elif self.driver.target == "ascend":
            self.binary_ext = "npubin"
        else:
            raise RuntimeError(f"Target '{self.driver.target}' is not supported.")

    def add_stages(self, stages, options, language=None):
        if self.driver.target == "ascend":
            from triton.backends.dicp_triton.npu import (
                linalg_to_bin_enable_npu_compile_910_95,
                linalg_to_bin_enable_npu_compile_A2_A3,
            )

            stages["linkedir"] = lambda src, metadata: commonir_to_linkedir(
                src, metadata, options, named_ops=True
            )

            if options.compile_on_910_95:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(
                        src, metadata, options
                    )
                )
            else:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(
                        src, metadata, options
                    )
                )
        else:
            raise RuntimeError("backend not supported")

    def load_dialects(self, ctx):
        if self.driver.target == "mlu":
            from triton._C.libtriton import mlu

            mlu.load_dialects(ctx)
        elif self.driver.target == "ascend":
            from triton._C.libtriton import dicp_triton

            dicp_triton.load_dialects(ctx)
        from triton._C.libtriton import dicp_triton

        dicp_triton.ir.load_dialects(ctx)
        return

    def get_driver(self):
        return self.driver

    # parse  add_kernel[(16,)](x, y, output, n_elements, BLOCK_SIZE=1024)
    def parse_options(self, options: dict) -> Any:
        if self.driver.target == "ascend":
            from triton.backends.dicp_triton.npu import NPUOptions

            args = {
                k: options[k]
                for k in NPUOptions.__dataclass_fields__.keys()
                if k in options
            }
            options = NPUOptions(**args)
            return options
        elif self.driver.target == "mlu":
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
        elif self.driver.target == "maca":
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
        if self.driver.target == "ascend":
            from triton.backends.dicp_triton.npu import min_dot_size

            codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        elif self.driver.target == "mlu":
            from triton.backends.dicp_triton.mlu import min_dot_size

            codegen_fns = {
                "convert_custom_types": lambda arg, dst_ty: arg,
                "min_dot_size": min_dot_size(self.target),
            }
        elif self.driver.target == "maca":
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
        if self.driver.target == "ascend":
            KERNEL_NAME_MAX_LEN = 49
            kernel_name_orig = metadata.kernel_name
            if len(kernel_name_orig) > KERNEL_NAME_MAX_LEN:
                kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
            else:
                kernel_name = kernel_name_orig
            return {
                "kernel_name": kernel_name,
                "hash": metadata.hash,
                "debug": metadata.debug,
                "tensor_kinds": metadata.tensor_kinds,
            }
        elif self.driver.target == "mlu":
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
        if self.driver.target == "mlu":
            from triton.backends.dicp_triton.mlu import get_cnas_version

            version = get_cnas_version()
            return f"{version}-{self.capability}"
        version_key = self.driver.target
        return str(version_key)


commonir_backend = CommonIRBackend()
