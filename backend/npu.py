from pathlib import Path
import tempfile
import os
import subprocess
import sysconfig
import functools
import hashlib
import logging
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir, passes, dicp_triton
import triton.backends.dicp_triton.utils as dicp_utils
from dataclasses import dataclass
from typing import Any, Union, Tuple, Dict
import ctypes
import re
import sys

from .utils import (
    TRITON_PROFILER_REGISTERED,
    replace_dicp_ir,
    _get_npucompiler_path,
    _get_bisheng_path,
    _get_ascend_path,
    _get_bishengir_opt_path,
    _is_ascend_sanitizer_enabled,
    _is_auto_map_parallel_blocks_enabled,
    _check_bishengir_api_change,
    _check_bishengir_able_save_ir,
    _is_debug_line_info_disabled,
    _enable_print_ub_bits,
    _enable_dump_memory_info,
    _enable_msdebug,
    _enable_unpublished_feature,
    _check_cxx11_abi,
    _build_npu_ext,
    convert_sigtype_to_int,
    force_disable_ffts,
    triton_enable_libdevice_simt,
    get_cann_version,
    is_compile_on_910_95,
)
from .npu_driver import NPUUtils


def _get_dicp_opt_path() -> str:
    import triton

    triton_dir = os.path.dirname(os.path.realpath(triton.__file__))
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    cmake_dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    return os.path.join(
        triton_dir,
        os.pardir,
        "build",
        cmake_dir_name,
        "third_party",
        "dicp_triton",
        "tools",
        "dicp_triton_opt",
        "dicp_opt",
    )


def _get_mlir_path(path: str, *paths) -> str:
    root_path = os.getenv("MLIR_ROOT", "")
    if root_path == "":
        raise EnvironmentError("MLIR_ROOT is not set.")
    return os.path.join(root_path, path, *paths)


def _get_llvm_path(path: str, *paths) -> str:
    root_path = os.getenv("LLVM_ROOT", "")
    if root_path == "":
        raise EnvironmentError("LLVM_ROOT is not set.")
    return os.path.join(root_path, path, *paths)


def downgrade_llir(llir):
    llir = _downgrade_mem_attrs(llir)
    llir = _downgrade_stacksaverestore_intrinsics(llir)
    return llir


def _downgrade_mem_attrs(llir: str):
    memory_pattern = r"memory\([^()]*\)"

    def replace_mem_attr(m):
        attrs = m[0][7:-1].split(",")
        if len(attrs) == 0:
            return "readnone"
        loc_map = {"argmem": 1, "inaccessiblemem": 2, "other": 4}
        loc_attr = 0
        rw_map = {"readwrite": 3, "write": 2, "read": 1, "none": 0}
        rw_attr = 0
        for attr_pair in attrs:
            pair = attr_pair.split(":")
            assert len(pair) <= 2
            if len(pair) == 1:
                rw = rw_map[pair[0].strip()]
                loc = loc_map["other"]
            else:
                rw = rw_map[pair[1].strip()]
                loc_str = pair[0].strip()
                if loc_str == "argmem" or loc_str == "inaccessiblemem":
                    loc = loc_map[loc_str]
                else:
                    loc = loc_map["other"]
            if rw > 0:
                loc_attr = loc_attr | loc
                rw_attr = rw_attr | rw
        rev_rw_map = {0: "readnone", 1: "readonly", 2: "writeonly"}
        if rw_attr in rev_rw_map:
            rw_attr_str = rev_rw_map[rw_attr]
        else:
            rw_attr_str = ""
        rev_loc_map = {
            1: "argmemonly",
            2: "inaccessiblememonly",
            3: "inaccessiblemem_or_argmemonly",
        }
        if loc_attr in rev_loc_map:
            loc_attr_str = rev_loc_map[loc_attr]
        else:
            loc_attr_str = ""
        return rw_attr_str + " " + loc_attr_str

    return re.sub(memory_pattern, replace_mem_attr, llir)


def _downgrade_stacksaverestore_intrinsics(llir: str):
    llir = re.sub(r"llvm\.stacksave\.\w+", "llvm.stacksave", llir)
    llir = re.sub(r"llvm\.stackrestore\.\w+", "llvm.stackrestore", llir)
    return llir


@functools.lru_cache(None)
def _get_bishengir_llvm_version() -> int:
    try:
        npu_compiler_path, _ = _get_npucompiler_path()
        result = subprocess.run(
            [npu_compiler_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout + result.stderr).lower()
        m = re.search(r"llvm\s+(\d+)", output)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 19


# def _downgrade_mlir_for_legacy_llvm(content: str) -> str:
#     content = content.replace("*xf", "?xf")
#     content = content.replace("*xi", "?xi")
#     content = content.replace("*xbf", "?xbf")
#     # 匹配形如 "memref<...> to tensor<...>" 的模式
#     pattern = r"(memref\<.*?\>)\s+to\s+(tensor\<.*?\>)"
#     # 使用正则替换，保留memref和tensor类型，中间插入注释
#     content = re.sub(pattern, r"\1 // to \2", content)
#     if len(re.findall("hivm\.hir\.custom", content)) > 0:
#         content = re.sub(r'"#hivm\.pipe<([A-Za-z0-9_]*)>"', r"#hivm.pipe<\1>", content)
#         content = re.sub(
#             r'"#hivm\.tcore_type<([A-Za-z0-9_]*)>"', r"#hivm.tcore_type<\1>", content
#         )
#         content = re.sub(
#             r'"#hivm\.vf_mode<([A-Za-z0-9_]*)>"', r"#hivm.vf_mode<\1>", content
#         )
#     return content


def _downgrade_mlir_for_legacy_llvm(content: str) -> str:
    _TO_BUFFER_RE = re.compile(
        r"\bbufferization\.to_buffer\b\s+(?P<value>%[^\s:]+).*?:\s*"
        r"(?P<tensor>tensor<.+?>)\s+to\s+(?P<memref>memref<.+?>)",
        re.DOTALL,
    )
    _TO_TENSOR_RE = re.compile(
        r"\bbufferization\.to_tensor\b\s+(?P<value>%[^\s:]+)"
        r"(?P<attrs>(?:\s+(?:restrict|writable))*)\s*:\s*"
        r"(?P<memref>memref<.+?>)\s+to\s+tensor<.+?>",
        re.DOTALL,
    )
    _HIVM_ATTR_RE = re.compile(r'"#hivm\.(pipe|tcore_type|vf_mode)<([A-Za-z0-9_]*)>"')
    content = content.replace("*x", "?x")
    content = _TO_BUFFER_RE.sub(
        lambda m: f'bufferization.to_memref {m.group("value")} : {m.group("memref")}',
        content,
    )
    content = _TO_TENSOR_RE.sub(
        lambda m: f'bufferization.to_tensor {m.group("value")}{m.group("attrs")} : {m.group("memref")}',
        content,
    )
    return _HIVM_ATTR_RE.sub(r"#hivm.\1<\2>", content)


def _check_bishengir_is_regbased() -> bool:
    bishengir_path, _ = _get_npucompiler_path()
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and "reg-based" in result.stdout:
            return True
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# Compiler pipeline functions
# ---------------------------------------------------------------------------


def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    if opt.arch:
        target_attr_str = f'#hacc.target<"{opt.arch}">'
        try:
            builder = dicp_triton.ir.dicp_npu_ir_builder(mod.context, opt.arch)
            mod.set_attr("hacc.target", builder.parse_attr(target_attr_str))
        except Exception as e:
            logging.warning(f"[DICP] Failed to set hacc.target: {e}")
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
    if opt.debug:
        dicp_utils._dump_stage_ir(str(mod), metadata["hash"], "kernel.ttir.mlir")
    return mod


def ttir_to_linalg_dicp(mod, metadata, opt, *, named_ops=False):
    """Lower TTIR to linalg using triton-dicp pipeline."""
    pm = ir.pass_manager(mod.context)

    enable_mask_fallback = metadata["enable_mask_fallback_conversion"]
    optimize_dynamic_offset = metadata["optimize_dynamic_offset"]
    compile_on_910_95 = metadata["compile_on_910_95"]
    force_simt_template = metadata["force_simt_template"]
    enable_sync_block_lock = metadata["enable_sync_block_lock"]
    enable_nd2nz_on_vector = metadata["enable_nd2nz_on_vector"]
    enable_select_analysis = metadata["enable_select_analysis"]
    auto_blockify_size = metadata["auto_blockify_size"]
    if not _is_auto_map_parallel_blocks_enabled():
        auto_blockify_size = 1

    dicp_triton.passes.ttir.add_ascend_legalize(pm)
    dicp_triton.passes.ttir.add_auto_blockify(pm, auto_blockify_size)

    if metadata["add_auto_scheduling"]:
        dicp_triton.passes.ttir.add_dag_sync(pm)
        dicp_triton.passes.ttir.add_dag_scope(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        dicp_triton.passes.ttir.add_dag_ssbuffer(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

    dicp_triton.passes.ttir.add_triton_to_structure(
        pm, enable_mask_fallback, optimize_dynamic_offset
    )
    dicp_triton.passes.ttir.add_discrete_mask_access_conversion(
        pm, compile_on_910_95, force_simt_template, enable_sync_block_lock
    )
    dicp_triton.passes.ttir.add_triton_to_annotation(pm)
    dicp_triton.passes.ttir.add_triton_to_unstructure(
        pm, compile_on_910_95, force_simt_template
    )
    dicp_triton.passes.ttir.add_triton_to_hivm(pm)
    dicp_triton.passes.ttir.add_triton_to_hfusion(pm)
    dicp_triton.passes.ttir.add_triton_to_llvm(pm)
    dicp_triton.passes.ttir.add_bubble_up_operation(pm)
    dicp_triton.passes.ttir.add_triton_to_structure(
        pm, enable_mask_fallback, optimize_dynamic_offset
    )
    dicp_triton.passes.ttir.add_triton_to_linalg(
        pm,
        False,
        named_ops,
        enable_nd2nz_on_vector,
        enable_select_analysis,
        compile_on_910_95,
    )
    dicp_triton.passes.ttir.add_ascend_npu_ir_legalize(pm, False)

    if metadata["enable_dynamic_cv_pipeline"]:
        dicp_triton.passes.ttir.add_dynamic_cv_pipeline(pm, compile_on_910_95)

    pm.run(mod)

    content = str(mod)

    if opt.debug:
        pipeline_str = pm.get_pipeline_str()
        dicp_opt_path = _get_dicp_opt_path()
        cmd_list = [
            dicp_opt_path,
            "",
            f"--pass-pipeline={pipeline_str}",
            "--mlir-print-debuginfo",
            "-o",
            "/dev/null",
        ]
        dicp_utils._dump_stage_ir(
            content, metadata["hash"], "kernel.dicp.mlir", cmd_list
        )

    return content


def linalg_to_llir(linalg: str, metadata, opt):
    with tempfile.TemporaryDirectory() as tmpdir:
        dicp_path = os.path.join(tmpdir, "kernel.dicp.mlir")
        llmlir_path = os.path.join(tmpdir, "kernel.llir.mlir")
        llir_path = os.path.join(tmpdir, "kernel.ll")
        Path(dicp_path).write_text(linalg)
        mlir_opt_path = _get_mlir_path("bin", "mlir-opt")
        subprocess.check_call(
            [
                mlir_opt_path,
                dicp_path,
                "--convert-linalg-to-affine-loops",
                "--eliminate-empty-tensors",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--expand-strided-metadata",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--lower-affine",
                "--convert-arith-to-llvm",
                "--reconcile-unrealized-casts",
                "-o",
                llmlir_path,
            ]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(
                Path(llmlir_path).read_text(), "kernel.llir.mlir", binary=False
            )

        mlir_translate_path = _get_mlir_path("bin", "mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(Path(llir_path).read_text(), "kernel.ll", binary=False)

        return Path(llir_path).read_text()


def llir_to_cpuasm(llir: str, metadata, opt):
    metadata["shared"] = 1
    fn_name = llir.split("define void @")[1].split("(")[0].strip()
    metadata["name"] = fn_name + " cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        linked_path = os.path.join(tmpdir, "kernel_linked.ll")
        dst_path = os.path.join(tmpdir, "kernel.s")

        llir = downgrade_llir(llir)
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(llir, "kernel_downgrade.ll", binary=False)

        Path(src_path).write_text(llir)

        linker_path = _get_llvm_path("bin", "llvm-link")
        libclc_path = _get_llvm_path("lib", "clc", "libspirv-aarch64--.bc")
        subprocess.check_call(
            [
                linker_path,
                src_path,
                libclc_path,
                "--only-needed",
                "-S",
                "-o",
                linked_path,
            ]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(
                Path(linked_path).read_text(), "kernel_linked.ll", binary=False
            )

        llc_path = _get_llvm_path("bin", "llc")
        subprocess.check_call([llc_path, linked_path, "-o", dst_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(Path(dst_path).read_text(), "kernel.s", binary=False)

        return Path(dst_path).read_text()


def __get_metadata_attr_by_callback(lib, postfix: str, metadata, meta_key: str):
    func_symbol = metadata["kernel_name"] + postfix
    if hasattr(lib, func_symbol):
        callback_func = getattr(lib, func_symbol)
        callback_func.restype = ctypes.c_int64
        callback_func.argtypes = []
        metadata[meta_key] = callback_func()


def _parse_linalg_metadata(linalg: str, metadata: dict):
    DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX = (
        r"hivm.disable_auto_tile_and_bind_subblock"
    )
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'
    PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"
    TENSOR_KIND_REGEX = (
        r"%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}"
    )

    # Example: bitcode = "a.bc"
    BITCODES_REGEX = r'bitcode\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+))'

    # Example removal:   ', mix_mode = "aiv"' → ''
    REMOVE_MIX_MODE_REGEX = r', mix_mode\s*=\s*"[^"]*"'
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    metadata["auto_tile_and_bind_subblock"] = not re.search(
        DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg
    )
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    metadata["name"] = metadata["kernel_name"]
    metadata["tensor_kinds"] = [
        int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)
    ]
    metadata["required_ub_bits"] = 0

    bitcodes = re.findall(BITCODES_REGEX, linalg)
    metadata["bitcodes"] = [val for group in bitcodes for val in group if val]
    return linalg, metadata


def _parse_ttir_metadata(ttir: str, metadata: dict):
    KERNEL_NAME_REGEX = r"tt\.func\spublic\s+@(\w+)"
    TENSOR_KIND_REGEX = (
        r"%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}"
    )

    metadata["shared"] = 1
    metadata["mix_mode"] = "aiv"
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, ttir).group(1)
    metadata["name"] = metadata["kernel_name"]
    metadata["tensor_kinds"] = [
        int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)
    ]
    return metadata


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata["target"].arch
    bishengir_target_opt = f"--target={bishengir_target}"
    return [bishengir_target_opt]


def get_auto_bind_sub_block_option(metadata):
    enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
    return (
        metadata["auto_tile_and_bind_subblock"]
        if enable_auto_bind_sub_block is None
        else enable_auto_bind_sub_block
    )


def _save_npuir_debug_output(
    stdout_bytes: bytes, stderr_bytes: bytes, tmpdir: str, metadata_hash: str
):
    stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""
    combined = stdout + stderr
    if not combined.strip():
        combined = "No output captured."
    output_path = os.path.join(tmpdir, "kernel.npuir.mlir")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    dump_manager = get_dump_manager(metadata_hash)
    dump_manager.put(
        Path(output_path).read_text(encoding="utf-8"), "kernel.npuir.mlir", binary=False
    )


def get_libdevice():
    current = os.path.dirname(__file__)
    return os.path.join(current, "lib/libdevice.10.bc")


# ---------------------------------------------------------------------------
# Shared NPU compilation orchestration
# ---------------------------------------------------------------------------


def _compile_linalg_to_npu_bin(linalg, metadata, opt, *,
                                build_options_fn,
                                bishengir_hivm_opt=None,
                                extra_cmd_args=None,
                                debug_stage_name="kernel.npuir_input.mlir"):
    """Shared orchestration for linalg → npubin compilation.

    Parameters
    ----------
    build_options_fn : callable
        ``(metadata, opt) -> list[str]`` that builds the initial
        ``_compile_option_list`` for this platform.
    bishengir_hivm_opt : str or None
        If set, injected between ``--enable-hfusion-compile`` and
        ``--enable-triton-kernel-compile`` in the bishengir-compile block
        (A2/A3 only).
    extra_cmd_args : callable or None
        Optional ``(metadata, opt) -> list[str]`` for appending extra args
        after ``-o bin_file`` (910_95: vf_merge_level + hfusion multi-consumer).
    debug_stage_name : str
        File name used when dumping the input IR in debug mode.
    """
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    if replace_dicp_ir is not None:
        print(f"[DEBUG] Replace dicp ir with {replace_dicp_ir}")
        linalg = Path(replace_dicp_ir).read_text()
    if _get_bishengir_llvm_version() < 22:
        linalg = _downgrade_mlir_for_legacy_llvm(linalg)
    if opt.debug:
        dicp_utils._dump_stage_ir(linalg, metadata["hash"], debug_stage_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        dicp_path = os.path.join(tmpdir, "kernel.dicp.mlir")
        Path(dicp_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")

        # --- platform-specific option list ---
        _compile_option_list = build_options_fn(metadata, opt)

        # --- bishengir-compile wrapper ---
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += ["--enable-hfusion-compile=true"]
            if bishengir_hivm_opt:
                _compile_option_list.append(bishengir_hivm_opt)
            _compile_option_list += ["--enable-triton-kernel-compile=true"]

        # --- debug ---
        if opt.debug:
            _compile_option_list += [
                "--bishengir-print-ir-after=hivm-graph-sync-solver"
            ]

        cmd_list = (
            [npu_compiler_path, dicp_path] + _compile_option_list + ["-o", bin_file]
        )
        if extra_cmd_args:
            cmd_list.extend(extra_cmd_args(metadata, opt))

        if opt.debug:
            print(f"[DEBUG] cmd_list: {' '.join(cmd_list)}")

        # --- execute ---
        try:
            ret = subprocess.run(
                cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        stdout_str = ret.stdout.decode("utf-8") if ret.stdout else ""
        match = re.search(r"UB\s+size\s*=\s*(\d+)\s*bits", stdout_str)
        if match:
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode("utf-8") if ret.stderr else ""
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(
                ret.returncode, cmd_list, ret.stdout, ret.stderr
            )

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(
                lib, "_infer_task_type_function", metadata, "bs_task_type"
            )
            __get_metadata_attr_by_callback(
                lib, "_infer_workspace_shape_function", metadata, "workspace_size"
            )
            __get_metadata_attr_by_callback(
                lib, "_infer_sync_block_lock_num_function", metadata, "lock_num"
            )
            __get_metadata_attr_by_callback(
                lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val"
            )

        return Path(bin_path).read_bytes()


# ---------------------------------------------------------------------------
# 910_95 compilation path
# ---------------------------------------------------------------------------


def linalg_to_bin_enable_npu_compile_910_95(linalg: str, metadata, opt):
    def _build_options(m, o):
        opts = get_common_bishengir_compile_options(m)

        multibuffer = m.get("multibuffer")
        num_stages = m.get("num_stages")
        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            opts.append(f"--enable-auto-multi-buffer={multi_buffer_value}")

        if m.get("disable_tightly_coupled_buffer_reuse"):
            opts.append("--disable-tightly-coupled-buffer-reuse")

        opts.append(
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(m)}"
        )

        if force_disable_ffts():
            opts.append("--disable-ffts")
        if _is_ascend_sanitizer_enabled():
            opts.append("--enable-sanitizer=true")
        if not _is_debug_line_info_disabled():
            opts.append("--enable-debug-info=true")
        if _enable_print_ub_bits():
            opts.append("--enable-print-memory-allocated-size")

        enable_hivm_auto_cv_balance = m["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            opts.append(
                f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"
            )

        sync_solver = m["sync_solver"]
        if sync_solver is not None:
            opts.append(f"--enable-hivm-graph-sync-solver={sync_solver}")

        unit_flag = m["unit_flag"]
        if unit_flag is not None:
            opts.append(f"--enable-hivm-unit-flag-sync={unit_flag}")

        inject_barrier_all = m["inject_barrier_all"]
        if inject_barrier_all is not None:
            opts.append(
                f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"
            )

        inject_block_all = m["inject_block_all"]
        if inject_block_all is not None:
            opts.append(
                f"--enable-hivm-inject-block-all-sync={inject_block_all}"
            )

        limit_auto_multi_buffer_only_for_local_buffer = m[
            "limit_auto_multi_buffer_only_for_local_buffer"
        ]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            opts.append(
                f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"
            )

        set_workspace_multibuffer = m["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            opts.append(
                f"--set-workspace-multibuffer={set_workspace_multibuffer}"
            )

        auto_multi_buffer = m["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            opts.append(
                f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"
            )

        enable_mixed_cv = m["enable_mixed_cv"]
        if enable_mixed_cv is not None:
            opts.append(f"--enable-mixed-cv={enable_mixed_cv}")

        enable_cce_vf_auto_sync = m["enable_cce_vf_auto_sync"]
        if enable_cce_vf_auto_sync is not None:
            opts.append(
                f"--append-bisheng-options=-mllvm --cce-vf-auto-sync={enable_cce_vf_auto_sync}"
            )

        enable_cce_vf_remove_membar = m["enable_cce_vf_remove_membar"]
        if enable_cce_vf_remove_membar is not None:
            opts.append(
                f"--append-bisheng-options=-mllvm --cce-vf-remove-membar={enable_cce_vf_remove_membar}"
            )

        env_vf = os.getenv("TRITON_ENABLE_VF_FUSION")
        enable_vf_fusion = (
            env_vf.lower() in ("true", "1", "yes")
            if env_vf is not None
            else m.get("enable_vf_fusion", False)
        )
        if enable_vf_fusion:
            opts.append("--enable-vf-fusion")

        enable_drop_unit_dims = m["enable_drop_unit_dims"]
        if enable_drop_unit_dims is not None:
            opts.append(f"--enable-drop-unit-dims={enable_drop_unit_dims}")

        enable_flatten = m["enable_flatten"]
        if enable_flatten is not None:
            opts.append(f"--enable-flatten={enable_flatten}")

        enable_auto_vectorize_v2 = m["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            opts.append(
                f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"
            )

        auto_vectorize_v2_max_fused_ops_num = m[
            "auto_vectorize_v2_max_fused_ops_num"
        ]
        if auto_vectorize_v2_max_fused_ops_num is not None:
            opts.append(
                f"--hfusion-max-fused-ops-in-auto-vectorize-v2={auto_vectorize_v2_max_fused_ops_num}"
            )

        prevec_max_fused_ops_num = m["prevec_max_fused_ops_num"]
        if prevec_max_fused_ops_num is not None:
            opts.append(
                f"--hfusion-max-fused-elementwise-ops={prevec_max_fused_ops_num}"
            )

        disable_auto_inject_block_sync = m["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            opts.append(
                f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"
            )

        bitcodes = m["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                opts.append(f"--link-aicore-bitcode={bitcode}")

        enable_auto_blockify = m["enable_auto_blockify"]
        if _is_auto_map_parallel_blocks_enabled():
            if enable_auto_blockify is None or enable_auto_blockify:
                opts.append("--enable-auto-blockify-loop")
        elif enable_auto_blockify:
            opts.append("--enable-auto-blockify-loop")

        bisheng_options = m["bisheng_options"]
        if bisheng_options is not None:
            opts.append(f"--append-bisheng-options={bisheng_options}")

        if o.mix_mode in ("aic",):
            opts.append("--disable-hfusion-vectorize=true")

        return opts

    def _extra_cmd_args(m, o):
        args = []
        vf_merge_level = m.get("vf_merge_level")
        if vf_merge_level is not None and vf_merge_level != 1:
            args.append(f"--enable-vf-merge-level={vf_merge_level}")
        hfusion = m.get("hfusion_enable_multiple_consumer_fusion")
        if hfusion:
            args.append(
                f"--hfusion-enable-multiple-consumer-fusion={hfusion}"
            )
        return args

    return _compile_linalg_to_npu_bin(
        linalg, metadata, opt,
        build_options_fn=_build_options,
        extra_cmd_args=_extra_cmd_args,
        debug_stage_name="kernel.dicp.mlir",
    )


# ---------------------------------------------------------------------------
# A2/A3 compilation path
# ---------------------------------------------------------------------------


def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    if _check_bishengir_is_regbased():
        bishengir_hivm_opt = "--reg-based=true"
    else:
        bishengir_hivm_opt = "--enable-hivm-compile=true"

    def _build_options(m, o):
        opts = [f"--target={NPUUtils().get_arch()}"]

        multibuffer = m.get("multibuffer")
        num_stages = m.get("num_stages")
        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            opts.append(
                f"--enable-auto-multi-buffer={multi_buffer_value}"
            )

        enable_ubuf_saving = m["enable_ubuf_saving"]
        if enable_ubuf_saving is not None:
            opts.append(f"--enable-ubuf-saving={enable_ubuf_saving}")

        enable_preload = m["enable_preload"]
        if enable_preload is not None:
            opts.append(f"--enable-preload={enable_preload}")

        opts.append(
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(m)}"
        )

        if _is_ascend_sanitizer_enabled():
            opts.append("--enable-sanitizer=true")
        if not _is_debug_line_info_disabled():
            opts.append("--enable-debug-info=true")
        if _enable_print_ub_bits():
            opts.append("--enable-print-memory-allocated-size")
        if _enable_dump_memory_info():
            opts.append("--enable-memory-display=true")
        if _enable_msdebug():
            opts.append("--enable-ms-debug=true")

        enable_hivm_auto_cv_balance = m["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            opts.append(
                f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"
            )

        sync_solver = m["sync_solver"]
        if sync_solver is not None:
            opts.append(
                f"--enable-hivm-graph-sync-solver={sync_solver}"
            )
            opts.append(
                f"--enable-hivm-cross-core-gss={sync_solver}"
            )

        unit_flag = m["unit_flag"]
        if unit_flag is not None:
            opts.append(f"--enable-hivm-unit-flag-sync={unit_flag}")

        enable_drop_unit_dims = m["enable_drop_unit_dims"]
        if enable_drop_unit_dims is not None:
            opts.append(f"--enable-drop-unit-dims={enable_drop_unit_dims}")

        enable_flatten = m["enable_flatten"]
        if enable_flatten is not None:
            opts.append(f"--enable-flatten={enable_flatten}")

        enable_auto_vectorize_v2 = m["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            opts.append(
                f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"
            )

        inject_barrier_all = m["inject_barrier_all"]
        if inject_barrier_all is not None:
            opts.append(
                f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"
            )

        inject_block_all = m["inject_block_all"]
        if inject_block_all is not None:
            opts.append(
                f"--enable-hivm-inject-block-all-sync={inject_block_all}"
            )

        limit_auto_multi_buffer_only_for_local_buffer = m[
            "limit_auto_multi_buffer_only_for_local_buffer"
        ]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            opts.append(
                f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"
            )

        set_workspace_multibuffer = m["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            opts.append(
                f"--set-workspace-multibuffer={set_workspace_multibuffer}"
            )

        tile_mix_vector_loop = m["tile_mix_vector_loop"]
        if tile_mix_vector_loop is not None:
            opts.append(f"--tile-mix-vector-loop={tile_mix_vector_loop}")

        tile_mix_cube_loop = m["tile_mix_cube_loop"]
        if tile_mix_cube_loop is not None:
            opts.append(f"--tile-mix-cube-loop={tile_mix_cube_loop}")

        auto_multi_buffer = m["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            opts.append(
                f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"
            )

        disable_auto_inject_block_sync = m["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            opts.append(
                f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"
            )

        bitcodes = m["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                opts.append(f"--link-aicore-bitcode={bitcode}")

        if m.get("disable_auto_cv_work_space_manage") is True:
            opts.append("--disable-auto-cv-work-space-manage=True")

        opts.append(f"--link-aicore-bitcode={get_libdevice()}")

        disable_size_align_for_cast = m["disable_size_align_for_cast"]
        if disable_size_align_for_cast is not None:
            opts.append(
                f"--disable-size-align-for-cast={disable_size_align_for_cast}"
            )

        if _is_auto_map_parallel_blocks_enabled():
            opts.append("--enable-auto-blockify-loop")

        return opts

    return _compile_linalg_to_npu_bin(
        linalg, metadata, opt,
        build_options_fn=_build_options,
        bishengir_hivm_opt=bishengir_hivm_opt,
    )


# ---------------------------------------------------------------------------
# SIMT-only TTIR -> npubin path
# ---------------------------------------------------------------------------


def ttir_to_npubin(mod, metadata, opt):
    ttir_code = str(mod)
    metadata = _parse_ttir_metadata(ttir_code, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        Path(src_path).write_text(ttir_code)
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        if opt.force_simt_only:
            _compile_option_list += ["--enable-hivm-compile=false"]
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]
            if opt.enable_bishengir_simt_optimization != 000:
                _compile_option_list += [
                    f"--enable-bishengir-simt-optimization={opt.enable_bishengir_simt_optimization}"
                ]
            if opt.simt_stack_limit:
                _compile_option_list += [f"--simt-stack-limit={opt.simt_stack_limit}"]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [
                    f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"
                ]
            if opt.enable_simt_reorder_instruction:
                _compile_option_list += ["--enable-simt-reorder-instruction=true"]
            if opt.disable_fma:
                _compile_option_list += [f"--disable-fma"]
            enable_libdevice_simt = triton_enable_libdevice_simt()
            if enable_libdevice_simt:
                bisheng_options = metadata["bisheng_options"]
                if bisheng_options is not None:
                    _compile_option_list += [
                        f"--append-bisheng-options={bisheng_options}"
                    ]

        npu_compiler_path, env = _get_npucompiler_path()
        cmd_list = (
            [npu_compiler_path, src_path] + _compile_option_list + ["-o", bin_file]
        )
        ret = subprocess.run(cmd_list, env=env, capture_output=True, check=True)
        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode("utf-8")
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(
                ret.returncode, cmd_list, ret.stdout, ret.stderr
            )
        return Path(bin_path).read_bytes()


# ---------------------------------------------------------------------------
# Options dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    llvm_version: int = 22
    kernel_name: str = "triton_"
    arch: str = ""

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 32
    num_ctas: int = 1
    num_stages: int = 1 if is_compile_on_910_95 else 2
    warp_size: int = 32
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    auto_blockify_size: int = 1
    enable_auto_blockify: bool = None
    compile_on_910_95: bool = is_compile_on_910_95
    optimize_dynamic_offset: bool = False
    enable_mask_fallback_conversion: bool = False
    enable_warp_specialization: bool = False
    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    auto_tile_and_bind_subblock: bool = True
    supported_fp8_dtypes: Tuple[str] = (
        "fp8e5",
        "fp8e4b15",
        "fp8e4nv",
        "fp8e4b8",
        "fp8e5b16",
    )
    deprecated_fp8_dtypes: Tuple[str] = ()
    vf_merge_level: int = 1
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    bisheng_options: str = "-cce-link-aicore-ll-module " + get_libdevice()

    multibuffer: bool = not is_compile_on_910_95
    enable_ubuf_saving: bool = None
    enable_preload: bool = None
    enable_auto_bind_sub_block: bool = None
    disable_tightly_coupled_buffer_reuse: bool = False
    enable_select_analysis: bool = True
    enable_hivm_auto_cv_balance: bool = None
    sync_solver: bool = None
    unit_flag: bool = None
    enable_cce_vf_auto_sync: bool = None
    enable_cce_vf_remove_membar: bool = None
    enable_drop_unit_dims: bool = None
    enable_flatten: bool = None
    enable_auto_vectorize_v2: bool = None
    auto_vectorize_v2_max_fused_ops_num: int = None
    prevec_max_fused_ops_num: int = None
    inject_barrier_all: bool = None
    inject_block_all: bool = None
    disable_size_align_for_cast: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None
    disable_auto_inject_block_sync: bool = None
    disable_auto_cv_work_space_manage: bool = False
    enable_mixed_cv: bool = None
    enable_vf_fusion: bool = False
    add_auto_scheduling: bool = False
    enable_dynamic_cv_pipeline: bool = False
    hfusion_enable_multiple_consumer_fusion: bool = False

    stream: int = None
    parallel_mode: str = "simd"
    force_simt_only: bool = False
    force_simt_template: bool = False
    enable_sync_block_lock: bool = False
    shared_mem_dynamic_size: int = None
    enable_bishengir_simt_optimization: int = 000
    compile_mode: str = "simd"
    mix_mode: str = ""
    simt_stack_limit: int = None
    enable_simt_reorder_instruction: bool = False
    disable_fma: bool = False

    def __post_init__(self):
        if self.compile_mode == "simd":
            object.__setattr__(self, "parallel_mode", "simd")
        elif self.compile_mode == "unstructured_in_simt":
            object.__setattr__(self, "force_simt_template", True)
        elif self.compile_mode == "simt_only":
            object.__setattr__(self, "force_simt_only", True)
            object.__setattr__(self, "parallel_mode", "simt")

        if self.force_simt_only:
            if self.shared_mem_dynamic_size is None:
                object.__setattr__(self, "shared_mem_dynamic_size", 122880)
        else:
            object.__setattr__(self, "shared_mem_dynamic_size", 221184)

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        key = "_".join([key, get_cann_version()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    llvm_version: int = 22
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = -1

    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()
