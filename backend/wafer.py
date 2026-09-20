import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget

from .wafer_cache import cache_digest, file_fingerprint


@dataclass(frozen=True)
class WaferOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    precision_mode: int = 0
    enable_pipeline: bool = False
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs: tuple = None
    cluster_dims: tuple = (1, 1, 1)
    launch_mode: str = "simt"
    shared: bool = False
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str, ...] = ("ieee",)
    sanitize_overflow: bool = True
    max_num_imprecise_acc_default: int = 0
    supported_fp8_dtypes: Tuple[str, ...] = ("fp8e5", "fp8e4b15", "fp8e4nv")
    deprecated_fp8_dtypes: Tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.precision_mode) is not int or self.precision_mode not in (0, 1, 2):
            raise ValueError("Wafer precision_mode must be 0, 1 or 2")
        if self.launch_mode not in ("simt", "cluster"):
            raise ValueError("Wafer launch_mode must be 'simt' or 'cluster'")
        if self.launch_mode == "cluster" and tuple(self.cluster_dims) != (1, 1, 1):
            raise ValueError("Wafer cluster launch uses one cluster: cluster_dims=(1, 1, 1)")

    def hash(self):
        key = "_".join(f"{name}-{value}" for name, value in self.__dict__.items())
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _run_tool(arguments):
    subprocess.check_call(
        arguments,
        stdout=None if os.getenv("MLIR_ENABLE_DUMP") == "1" else subprocess.DEVNULL,
    )


def _dump_file(path):
    dump_dir = os.getenv("TRITON_DUMP_PATH")
    if dump_dir:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(path, Path(dump_dir) / Path(path).name)


def _find_wafer_opt():
    override = os.getenv("WAFER_OPT_PATH")
    if override:
        path = Path(override)
        if path.is_file():
            return str(path)
        raise RuntimeError(f"WAFER_OPT_PATH does not name a file: {path}")

    backend_dir = Path(__file__).resolve().parent
    candidates = (
        backend_dir / "bin" / "wafer-opt",
        backend_dir.parent
        / "third_party"
        / "wafer"
        / "build_manual"
        / "third_party"
        / "wafer"
        / "bin"
        / "wafer-opt",
        backend_dir.parent
        / "third_party"
        / "wafer"
        / "build_manual"
        / "install"
        / "triton"
        / "backends"
        / "wafer"
        / "bin"
        / "wafer-opt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    path = shutil.which("wafer-opt")
    if path:
        return path
    raise RuntimeError(
        "wafer-opt not found; run scripts/wafer/compile_wafer.sh or set WAFER_OPT_PATH"
    )


def _find_llvm_tool(name):
    llvm_bin = os.getenv("LLVM_BINARY_DIR")
    if llvm_bin:
        candidate = Path(llvm_bin) / name
        if candidate.is_file():
            return str(candidate)
    path = shutil.which(name)
    if path:
        return path
    raise RuntimeError(f"{name} not found; set LLVM_BINARY_DIR")


def _run_wafer_stage(source, arguments, source_name, output_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / source_name
        output_path = Path(tmpdir) / output_name
        source_path.write_text(str(source), encoding="utf-8")
        command = [
            _find_wafer_opt(),
            str(source_path),
            *arguments,
            "-o",
            str(output_path),
        ]
        _run_tool(command)
        _dump_file(source_path)
        _dump_file(output_path)
        return output_path.read_text(encoding="utf-8")


def _precision_mode_from_env():
    # Keep the old switch as mode 2; explicit FlagTree-style mode takes priority.
    value = os.getenv("PRECISION_MODE")
    if value is None:
        return 2 if os.getenv("PRECISION_PRIORITY", "0").lower() in ("1", "true", "yes") else 0
    if value not in ("0", "1", "2"):
        raise ValueError("PRECISION_MODE must be 0, 1 or 2")
    return int(value)


def ttir_to_coreir(module, precision_mode=None, enable_pipeline=False, num_stages=1):
    if precision_mode is None:
        precision_mode = _precision_mode_from_env()
    core_to_mk = f"--core-dialects-to-mk=precision-mode={precision_mode}"
    return _run_wafer_stage(
        module,
        [
            "--triton-to-core-dialects",
            "--tle-to-mk",
            "--dsa-memory-to-core",
            "--linalg-tiling",
            core_to_mk,
            "--linalg-fusion",
            "--legalize-tensor-form-loops",
            "--one-shot-bufferize",
            "--convert-bufferization-to-memref",
            "--materialize-strided-linalg-inputs",
            *([f"--mk-pipeline=num-stages={num_stages} max-stages=2",
               "--mk-loop-bound-canonicalize"] if enable_pipeline else []),
            "--cse",
            "--canonicalize",
        ],
        "ttir.mlir",
        "coreir.mlir",
    )


def coreir_to_wafer_ir(module, enable_pipeline=False):
    return _run_wafer_stage(
        module,
        [
            "--spmd-allocate-shared-memory",
            "--expand-strided-metadata",
            "--lower-affine",
            "--mk-to-wafer",
            *(["--wafer-insert-barrier"] if enable_pipeline else []),
            "--cse",
        ],
        "coreir.mlir",
        "wafer_ir.mlir",
    )


def wafer_ir_to_llir(module, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "wafer_ir.mlir"
        llvm_mlir_path = Path(tmpdir) / "llvm.mlir"
        llvm_ir_path = Path(tmpdir) / "kernel.ll"
        source_path.write_text(str(module), encoding="utf-8")
        wafer_arguments = [
            _find_wafer_opt(),
            str(source_path),
            "--wafer-memref-to-llvm",
            "--addr-to-llvm",
            "--convert-scf-to-cf",
            # Keep log1p for libm: log(1+x) loses tiny inputs and signed zero.
            "--convert-math-to-llvm=approximate-log1p=false",
            "--convert-math-to-libm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--kernel-arg-buffer",
            "--wafer-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "--canonicalize",
            "--export-kernel-symbols",
            "-o",
            str(llvm_mlir_path),
        ]
        _run_tool(wafer_arguments)
        _run_tool(
            [
                _find_llvm_tool("mlir-translate"),
                str(llvm_mlir_path),
                "--mlir-to-llvmir",
                "-o",
                str(llvm_ir_path),
            ]
        )
        llvm_ir = llvm_ir_path.read_text(encoding="utf-8")
        names = re.findall(r"define\s+(?:\w+\s+)*@([\w.$]+)\(", llvm_ir)
        if names:
            metadata["name"] = names[0]
        metadata.setdefault("shared", 0)
        _dump_file(llvm_mlir_path)
        _dump_file(llvm_ir_path)
        return llvm_ir


def llir_to_object(llvm_ir, metadata, simulator=None):
    if simulator is None:
        simulator = simulator_enabled()
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "kernel.ll"
        object_path = Path(tmpdir) / "kernel.o"
        source_path.write_text(llvm_ir, encoding="utf-8")
        compiler = _find_llvm_tool("clang++")
        arguments = [
            compiler,
            str(source_path),
            "-O2",
            "-c",
            "-fPIC",
            "-o",
            str(object_path),
        ]
        if not simulator:
            arguments.extend(
                ["--target=riscv64-unknown-elf", "-march=rv64imfdc", "-mabi=lp64d"]
            )
        _run_tool(arguments)
        _dump_file(object_path)
        return object_path.read_bytes()


def _find_linker_library(linker, name):
    output = subprocess.check_output(
        [str(linker), "-march=rv64imfdc", "-mabi=lp64d", f"-print-file-name={name}"],
        text=True,
    ).strip()
    path = Path(output)
    if output == name or not path.is_file():
        raise RuntimeError(f"{linker} could not locate {name}: {output}")
    return path


LINK_FLAGS = (
    "-shared",
    "-march=rv64imfdc",
    "-mabi=lp64d",
    "-O2",
    "-nostartfiles",
    # All libraries are supplied explicitly and included in the cache key.
    # Do not let GCC append the original libc after our firmware-adapted copy.
    "-nodefaultlibs",
    "-Wl,--allow-shlib-undefined",
    "-Wl,--no-dynamic-linker",
    "-Wl,--gc-sections",
    "-Wl,--unique=.rodata.name",
)

# Kuiper 1.4 firmware renamed the device logging API. Rename references in
# private archive/object copies; preserve the vendor implementation and varargs
# ABI instead of supplying empty logging stubs or modifying the installed SDK.
RCS_LOG_SYMBOLS = {
    "tx8_kernel_printf": "rcs_kernel_printf",
    "tx8_kernel_vprintf": "rcs_kernel_vprintf",
    "tx8_kernel_vsnprintf": "rcs_kernel_vsnprintf",
    "tsm_ep_log": "rcs_ep_log",
    "_tsm_ep_log": "_rcs_ep_log",
}

# Keep the CRT assertion bound to the firmware's newlib service. Pulling the
# toolchain's static newlib implementation also pulls unsupported POSIX syscalls.
# Rename only its private archive copy; internal libc references remain paired
# with that implementation, while the CRT's __assert_func stays a firmware import.
FIRMWARE_LIBC_SYMBOLS = {"__assert_func": "__wafer_newlib_assert_func"}


def device_log_abi():
    abi = os.getenv("WAFER_DEVICE_LOG_ABI", "wafer")
    if abi not in ("wafer", "rcs"):
        raise ValueError(
            f"Unsupported WAFER_DEVICE_LOG_ABI={abi!r}; expected wafer or rcs"
        )
    return abi


def _runtime_link_inputs():
    deps_root = os.getenv("WAFER_DEPS_ROOT")
    if not deps_root:
        raise RuntimeError("WAFER_DEPS_ROOT is not set; source init_wafer_env.sh first.")
    wafer_deps_root = Path(deps_root)
    toolchain_root = Path(
        os.getenv(
            "XUANTIE_NAME", wafer_deps_root / "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
        )
    )
    linker = toolchain_root / "bin" / "riscv64-unknown-elf-gcc"
    wafer_lib_dir = Path(
        os.getenv("WAFER_RUNTIME_LIB_DIR", Path(__file__).resolve().parent / "lib")
    )
    if not wafer_lib_dir.is_dir():
        wafer_lib_dir = (
            Path(__file__).resolve().parent.parent / "third_party" / "wafer" / "lib"
        )

    required = [linker, wafer_lib_dir, wafer_deps_root / "lib"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Wafer runtime link dependencies are missing: " + ", ".join(missing)
        )
    libraries = [
        wafer_deps_root / "lib" / name
        for name in ("libcommon_util.a", "libinstr_tx81.a", "liblibc_stub.a")
    ]
    libraries.append(wafer_lib_dir / "libvr.a")
    # Xuantie GCC normally supplies libgloss along with libc. Keep that
    # existing dependency explicit when using -nodefaultlibs, including it
    # in the cache identity instead of relying on GCC's hidden defaults.
    libraries.extend(
        _find_linker_library(linker, name)
        for name in ("libm.a", "libc.a", "libgcc.a", "libgloss.a")
    )
    for library in libraries:
        if not library.is_file():
            raise RuntimeError(f"Wafer runtime link library is missing: {library}")
    return linker, libraries


def _link_fingerprint(linker, libraries, log_abi=None):
    log_abi = device_log_abi() if log_abi is None else log_abi
    result = {
        "linker": file_fingerprint(linker),
        "ld": file_fingerprint(linker.parent / "riscv64-unknown-elf-ld"),
        "flags": LINK_FLAGS,
        "archive_groups": [4, len(libraries) - 4],
        "libraries": [file_fingerprint(path) for path in libraries],
        "device_log_abi": log_abi,
        "firmware_libc_symbols": FIRMWARE_LIBC_SYMBOLS,
        "objcopy": file_fingerprint(_find_llvm_tool("llvm-objcopy")),
    }
    if log_abi == "rcs":
        result["log_symbols"] = RCS_LOG_SYMBOLS
    return result


def _adapt_logging_file(source, destination):
    _run_tool(
        [
            _find_llvm_tool("llvm-objcopy"),
            *(f"--redefine-sym={old}={new}" for old, new in RCS_LOG_SYMBOLS.items()),
            str(source),
            str(destination),
        ]
    )


def _adapt_logging_libraries(libraries, link_fingerprint):
    from triton.runtime.cache import get_cache_manager

    cache = get_cache_manager(cache_digest({"logging_archives": link_fingerprint}))
    adapted = []
    # Only Wafer and Wafer archives contain the renamed device APIs.
    for index, library in enumerate(libraries[:4]):
        name = f"{index}-{library.name}"
        path = cache.get_file(name)
        if path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / name
                _adapt_logging_file(library, output)
                path = cache.put(output.read_bytes(), name, binary=True)
        adapted.append(Path(path))
    return adapted + libraries[4:]


def _adapt_firmware_libc(libraries):
    from triton.runtime.cache import get_cache_manager

    adapted = list(libraries)
    for index, library in enumerate(libraries):
        if library.name != "libc.a":
            continue
        objcopy = _find_llvm_tool("llvm-objcopy")
        cache = get_cache_manager(cache_digest({
            "firmware_libc": file_fingerprint(library),
            "symbols": FIRMWARE_LIBC_SYMBOLS,
            "objcopy": file_fingerprint(objcopy),
        }))
        path = cache.get_file("libc.a")
        if path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / "libc.a"
                _run_tool([
                    objcopy,
                    *(f"--redefine-sym={old}={new}" for old, new in FIRMWARE_LIBC_SYMBOLS.items()),
                    str(library), str(output),
                ])
                path = cache.put(output.read_bytes(), "libc.a", binary=True)
        adapted[index] = Path(path)
    return adapted


def object_to_binary(obj, metadata, simulator=None, log_abi=None):
    if simulator is None:
        simulator = simulator_enabled()
    if simulator:
        raise RuntimeError(
            "Wafer simulator linking requires libvr, libtriton_cmodel, "
            "libtx8be_op_cmodel, and libneuralcore_qemu; they are not part of the current SDK."
        )
    linker, libraries = _runtime_link_inputs()
    log_abi = device_log_abi() if log_abi is None else log_abi
    link_fingerprint = _link_fingerprint(linker, libraries, log_abi)

    key = cache_digest(
        {"object": hashlib.sha256(obj).hexdigest(), "link": link_fingerprint}
    )
    from triton.runtime.cache import get_cache_manager

    cache = get_cache_manager(key)
    cache_path = cache.get_file("kernel.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_path = Path(tmpdir) / "kernel.o"
            binary_path = Path(tmpdir) / "kernel.so"
            object_path.write_bytes(obj)
            if log_abi == "rcs":
                libraries = _adapt_logging_libraries(libraries, link_fingerprint)
                adapted_object = Path(tmpdir) / "kernel-rcs.o"
                _adapt_logging_file(object_path, adapted_object)
                object_path = adapted_object
            libraries = _adapt_firmware_libc(libraries)
            command = [
                str(linker),
                *LINK_FLAGS,
                str(object_path),
                "-Wl,--start-group",
                *(str(path) for path in libraries[:4]),
                "-Wl,--end-group",
                "-Wl,--start-group",
                *(str(path) for path in libraries[4:]),
                "-Wl,--end-group",
                "-o",
                str(binary_path),
            ]
            cache.put(obj, "kernel.o", binary=True)
            cache.put(
                json.dumps({"command": command, "inputs": link_fingerprint}, indent=2),
                "link.json",
                binary=False,
            )
            _run_tool(command)
            _dump_file(binary_path)
            cache_path = cache.put(binary_path.read_bytes(), "kernel.so", binary=True)

    metadata["kernel_path"] = cache_path
    metadata["so_key"] = Path(cache_path).parent.name
    metadata["device_log_abi"] = log_abi
    return Path(cache_path).read_bytes()


def runtime_binary_enabled():
    return os.getenv("WAFER_ENABLE_RUNTIME", "0").lower() in ("1", "true", "yes")


def simulator_enabled():
    return os.getenv("USE_SIM_MODE", "0").lower() in ("1", "true", "yes")


def __getattr__(name):
    # Keep explicit legacy imports without advertising duplicate backend classes.
    aliases = {"TXDAOptions": WaferOptions, "TXDABackend": WaferBackend}
    if name in aliases:
        return aliases[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class WaferBackend(BaseBackend):
    def __init__(self, target):
        super().__init__(target)
        self.simulator = simulator_enabled()
        self.runtime = runtime_binary_enabled()
        self.device_log_abi = device_log_abi()
        self.precision_mode = _precision_mode_from_env()
        self.enable_pipeline = os.getenv("TRITON_PIPELINE", "0").lower() in ("1", "true", "yes")
        self.binary_ext = "so" if self.runtime else "o"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ("wafer", "txda")

    def parse_options(self, options: dict) -> Any:
        arguments = {
            name: options[name]
            for name in WaferOptions.__dataclass_fields__
            if name in options
        }
        arguments.setdefault("arch", self.target.arch)
        arguments.setdefault("precision_mode", self.precision_mode)
        arguments.setdefault("enable_pipeline", self.enable_pipeline)
        return WaferOptions(**arguments)

    def hash(self):
        inputs = {
            "target": [self.target.backend, self.target.arch, self.target.warp_size],
            "simulator": self.simulator,
            "runtime": self.runtime,
            "precision_mode": self.precision_mode,
            "enable_pipeline": self.enable_pipeline,
            "tools": [
                file_fingerprint(_find_wafer_opt()),
                file_fingerprint(_find_llvm_tool("mlir-translate")),
                file_fingerprint(_find_llvm_tool("clang++")),
            ],
            "source": file_fingerprint(__file__),
        }
        if self.runtime and not self.simulator:
            inputs["link"] = _link_fingerprint(
                *_runtime_link_inputs(), self.device_log_abi
            )
        return cache_digest(inputs)

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def load_dialects(self, context):
        from triton._C.libtriton import wafer

        wafer.load_dialects(context)
        wafer.tle.load_dialects(context)

    @staticmethod
    def make_ttir(module, metadata, options):
        pass_manager = ir.pass_manager(module.context)
        pass_manager.enable_debug()
        passes.common.add_inliner(pass_manager)
        passes.ttir.add_combine(pass_manager)
        passes.common.add_canonicalizer(pass_manager)
        passes.ttir.add_reorder_broadcast(pass_manager)
        passes.common.add_cse(pass_manager)
        passes.common.add_licm(pass_manager)
        passes.common.add_symbol_dce(pass_manager)
        pass_manager.run(module)
        metadata.setdefault("shared", 0)
        return module

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda source, metadata: self.make_ttir(
            source, metadata, options
        )
        stages["coreir"] = lambda source, metadata: ttir_to_coreir(
            source, options.precision_mode, options.enable_pipeline, options.num_stages)
        stages["wafer_ir"] = lambda source, metadata: coreir_to_wafer_ir(source, options.enable_pipeline)
        stages["llir"] = lambda source, metadata: wafer_ir_to_llir(source, metadata)
        if self.runtime:
            stages["so"] = lambda source, metadata: object_to_binary(
                llir_to_object(source, metadata, self.simulator),
                metadata,
                self.simulator,
                self.device_log_abi,
            )
        else:
            stages["o"] = lambda source, metadata: llir_to_object(
                source, metadata, self.simulator
            )

    def get_module_map(self) -> Dict[str, ModuleType]:
        try:
            from triton.language.extra.wafer import libdevice

            return {"triton.language.extra.libdevice": libdevice}
        except ImportError:
            return {}
