import os
import re
from typing import Callable, List
from triton.backends.dicp_triton.commonir.compiler import (
    CommonIRCompiler,
    CommonIRSource,
    CompiledKernel,
)


class AdapterWrapper:
    def __init__(self) -> None:
        from tilelang import tvm as tvm
        from tvm import tir
        from tilelang.engine.param import KernelParam
        from tilelang.jit.adapter import BaseKernelAdapter

        class Artifact:
            def __init__(self) -> None:
                self.kernel_source: str = None
                self.params: List[KernelParam] = None

            def set_kernel_source(self, kernel_source) -> None:
                self.kernel_source = str(kernel_source)
                self.params = self._extrac_params(kernel_source)

            def _extrac_params(self, func: tir.PrimFunc) -> List[KernelParam]:
                tensor_types = []
                for var in func.params:
                    if var in func.buffer_map:
                        tensor_types.append(
                            KernelParam.from_buffer(func.buffer_map[var])
                        )
                    else:
                        tensor_types.append(KernelParam.from_var(var))
                return tensor_types

        class Adapter(BaseKernelAdapter):
            def __init__(self) -> None:
                self.mod = None
                self.func = None
                self.libpath = None
                self.kernel_source = None

            def set_info(self, mod, kernel_source, func: CompiledKernel) -> None:
                self.mod = mod
                self.func = func
                self.libpath = func._run.so_launcher_path
                self.kernel_source = str(kernel_source)

            def _convert_torch_func(self) -> Callable:
                return self.func

            def get_kernel_source(self) -> str:
                return self.kernel_source

        self.adapter = Adapter()
        self.artifact = Artifact()

    @classmethod
    def compile_and_create_adapter(cls, tilelang_module):
        adapter_wrapper = AdapterWrapper()
        adapter_wrapper.artifact.set_kernel_source(tilelang_module)
        mlir_content = cls._tilelang_to_commonir(tilelang_module)
        grid = cls._parse_grid(tilelang_module)
        signature = cls._parse_signature(mlir_content)

        commonir_compiler = CommonIRCompiler()
        func = commonir_compiler.compile(CommonIRSource(mlir_content, grid, signature))
        adapter_wrapper.adapter.set_info(mlir_content, tilelang_module, func)

        return adapter_wrapper

    @classmethod
    def from_database(
        cls,
        params,
        result_idx,
        target,
        func_or_mod,
        host_kernel_source,
        kernel_lib_path,
        pass_configs,
    ):
        return cls.compile_and_create_adapter(func_or_mod)

    @classmethod
    def _tilelang_to_commonir(cls, tilelang_module):
        from tilelang.engine.lower import extrac_params
        from tilelang.engine.param import CompiledArtifact
        from tilelang.engine.phase import (
            PreLowerSemanticCheck,
            LowerAndLegalize,
        )
        from tilelang import tvm as tvm
        from tvm.target import Target
        from tvm import tir, IRModule
        from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll

        def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
            mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
            mod = tir.transform.LowerOpaqueBlock()(mod)
            mod = tir.transform.RemoveNoOp()(mod)
            return mod

        def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
            return tir.transform.BindTarget(target)(mod)

        def canon_target_host(target: str | Target, target_host: str | Target | None):
            if not target_host:
                target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"

            return target_host

        def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
            return tvm.ffi.get_global_func("target.build.tilelang_commonir")(
                device_mod, target
            )

        def lower(
            func_or_mod: tir.PrimFunc | tvm.IRModule,
            target: str | Target = "auto",
            target_host: str | Target | None = None,
            runtime_only=False,
        ) -> CompiledArtifact:
            mod = func_or_mod
            params = None
            if isinstance(func_or_mod, tir.PrimFunc):
                func = func_or_mod
                params = extrac_params(func) if not runtime_only else None
                mod = tvm.IRModule({func.attrs["global_symbol"]: func})
            target = "commonir"
            target_host = canon_target_host(target, target_host)
            target_host = tvm.target.Target.canon_target(target_host)
            target = tvm.target.Target(target, target_host)
            # Before lowering, do semantic check
            PreLowerSemanticCheck(mod)
            # Phase 1: Lower and legalize the IR
            mod = LowerAndLegalize(mod, target)
            # Phase 2: Optimize the IR for the target
            mod = OptimizeForTarget(mod, target)
            codegen_mod = device_codegen(mod, target)
            # print(codegen_mod.inspect_source())
            return CompiledArtifact(
                None, codegen_mod, params, codegen_mod.inspect_source(), None
            )

        debug_enabled = os.environ.get("TILELANG_PRINT_COMMONIR", "0") in (
            "1",
            "true",
            "on",
        )

        instruments = [PrintAfterAll(), PrintBeforeAll()] if debug_enabled else []
        with tvm.transform.PassContext(instruments=instruments):
            lower_result = lower(tilelang_module)
            mlir_content = lower_result.kernel_source
        return mlir_content

    @classmethod
    def _parse_grid(cls, tilelang_module):
        patterns = {
            "x": r'T\.launch_thread\("blockIdx\.x",\s*(\d+)\)',
            "y": r'T\.launch_thread\("blockIdx\.y",\s*(\d+)\)',
            "z": r'T\.launch_thread\("blockIdx\.z",\s*(\d+)\)',
        }
        block_indices = {"x": None, "y": None, "z": None}
        for dim, pattern in patterns.items():
            match = re.search(pattern, str(str(tilelang_module)))
            if match:
                block_indices[dim] = int(match.group(1))
        return [
            block_indices["x"] if block_indices["x"] is not None else 1,
            block_indices["y"] if block_indices["y"] is not None else 1,
            block_indices["z"] if block_indices["z"] is not None else 1,
        ]

    @classmethod
    def _read_mlir_file(cls, file_path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: File '{file_path}' does not exist")
            return None
        except Exception as e:
            print(f"Error occurred while reading the file: {e}")
            return None

    @classmethod
    def _parse_signature(cls, mlir_content) -> dict:
        target_types = {
            "i1",
            "i8",
            "i16",
            "i32",
            "i64",
            "u32",
            "u64",
            "fp16",
            "bf16",
            "fp32",
            "f32",
            "fp64",
            "f16",
        }

        pattern = r"func\.func\s*@[^(]*\(([^)]*)\)"
        match = re.search(pattern, mlir_content)

        if not match:
            return {}

        params_str = match.group(1)

        params = []
        current_param = ""
        brace_count = 0
        angle_count = 0

        for char in params_str:
            if char == "," and brace_count == 0 and angle_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "<":
                    angle_count += 1
                elif char == ">":
                    angle_count -= 1

        if current_param:
            params.append(current_param.strip())

        result = {}
        index = 0

        for param in params:
            if re.match(r"%args\d+", param.strip()):
                continue

            found_type = None
            for t_type in target_types:
                x_pattern = r"\bx" + t_type + r"\b"
                if re.search(x_pattern, param):
                    found_type = "*" + t_type
                    break
                elif re.search(r"\b" + t_type + r"\b", param):
                    found_type = t_type
                    break

            if found_type:
                if found_type == "f16":
                    found_type = "fp16"
                elif found_type == "*f16":
                    found_type = "*fp16"
                elif found_type == "f32":
                    found_type = "fp32"
                elif found_type == "*f32":
                    found_type = "*fp32"

                result[index] = found_type
                index += 1

        return result
