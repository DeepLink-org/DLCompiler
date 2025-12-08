from pathlib import Path
import tempfile
import os
import subprocess
import sysconfig
import functools
import hashlib
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir, passes, dicp_triton
from triton.runtime.cache import get_dump_manager
import triton.backends.dicp_triton.utils as dicp_utils
from dataclasses import dataclass
from typing import Any, Union, Tuple, Dict
import ctypes
import re
import pybind11
import shutil


###################### utils.py start ######################

TRITON_PROFILER_REGISTERED = False
dump_ir = os.environ.get("DLC_DUMP_IR", "0") == "1"
replace_ttshared_ir = os.environ.get("DLC_REPLACE_TTSHARED_IR_FILE", None)
replace_linked_ir = os.environ.get("DLC_REPLACE_LINKED_IR_FILE", None)
if dump_ir or (replace_ttshared_ir is not None) or (replace_linked_ir is not None):
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    dump_dir = "./tmp"
    os.environ["TRITON_DUMP_DIR"] = os.environ.get("TRITON_DUMP_DIR", dump_dir)
    if os.path.exists(dump_dir):
        print(f"Directory **{dump_dir}** exists. Deleting the entire directory...")
        shutil.rmtree(dump_dir)

local_bishengir_path = os.path.join(os.path.dirname(__file__), "../../_C/bishengir")
bisheng_install_path = os.environ.get("BISHENG_INSTALL_PATH", None)
if (
    bisheng_install_path is None
    and os.path.exists(local_bishengir_path)
    and os.path.isdir(local_bishengir_path)
    and os.path.exists(os.path.join(local_bishengir_path, "bishengir-compile"))
    and os.path.exists(os.path.join(local_bishengir_path, "bishengir-hivm-compile"))
    and os.path.exists(os.path.join(local_bishengir_path, "bishengir-opt"))
    and os.path.exists(os.path.join(local_bishengir_path, "hivmc"))
):
    os.environ["BISHENG_INSTALL_PATH"] = local_bishengir_path
    os.environ["PATH"] = local_bishengir_path + os.pathsep + os.environ["PATH"]


def downgrade_llir(llir):
    llir = _downgrade_mem_attrs(llir)
    llir = _downgrade_stacksaverestore_intrinsics(llir)
    return llir


def _replace_mod_ir_with_file(mod, filepath: str, stage_name: str):
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Replacement MLIR file not found: {filepath}")
    print(f"[DEBUG] replacing '{stage_name}' IR with file '{filepath}'")
    try:
        new_mod = ir.parse_mlir_module(str(p), mod.context)
        new_mod.context = mod.context
        return new_mod
    except Exception as e:
        raise RuntimeError(f"Failed to parse replacement MLIR file '{filepath}': {e}")


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
                loc = loc_map["other"]  # all location
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


def _get_triton_adapter_opt_path() -> str:
    path = os.path.dirname(__file__)
    path = os.path.join(path, "triton-adapter-opt")
    return path


def _get_dicp_opt_path() -> str:
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, "../../_C", "dicp_opt")
    return path


def _get_triton_shared_opt_path() -> str:
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, "../../_C", "triton-shared-opt-v3_2")
    path34 = os.path.join(base_path, "../../_C", "triton-shared-opt-v3_4")
    if os.path.exists(path34):
        path = path34
    path = os.getenv("TRITON_SHARED_OPT_PATH", path)  # allow user override
    if not os.path.exists(path):
        raise EnvironmentError(
            f"Couldn't find triton-shared-opt at {path}, set TRITON_SHARED_OPT_PATH to override"
        )
    return path


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


def _get_npucompiler_path() -> str:
    npu_compiler_path = shutil.which("bishengir-compile")
    if npu_compiler_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bishengir-compile or TRITON_NPU_COMPILER_PATH."
            )
        npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    npu_compiler_path = os.path.abspath(npu_compiler_path)
    return npu_compiler_path


def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bisheng or TRITON_NPU_COMPILER_PATH"
            )
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path


@functools.lru_cache(None)
def _get_ascend_path() -> str:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError(
            "ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first"
        )
    return Path(path)


def _is_ascend_sanitizer_enabled() -> bool:
    return os.getenv("TRITON_ENABLE_SANITIZER", "false").lower() in ("true", "1")


def _is_auto_map_parallel_blocks_enabled() -> bool:
    return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")


def _build_npu_ext(obj_name: str, src_path, src_dir, *, kernel_launcher=None) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = os.path.join(src_dir, f"{obj_name}{suffix}")

    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = clangxx if clangxx is not None else gxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    cc_cmd = [cxx, src_path]
    # disable all warnings
    cc_cmd += [f"-w"]
    # find the python library
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    # device_print.h
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
    # find the ascend library
    asc_path = _get_ascend_path()
    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]

    if kernel_launcher == "torch":
        import torch
        import torch_npu

        torch_path = os.path.dirname(os.path.realpath(torch.__file__))
        torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
        use_cxx11_abi = _check_cxx11_abi()
        cc_cmd += [
            f"-I{os.path.join(torch_path, 'include')}",
            f"-I{os.path.join(torch_npu_path, 'include')}",
            f"-L{os.path.join(torch_npu_path, 'lib')}",
            "-ltorch_npu",
            f"-D_GLIBCXX_USE_CXX11_ABI={use_cxx11_abi}",
        ]

    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", so_path]

    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    else:
        raise RuntimeError("Failed to compile " + src_path)


def _get_kernel_target(metadata: dict):
    if "target" not in metadata:
        raise Exception("No target provided!")
    sub_target = metadata["target"].arch
    assert isinstance(sub_target, str)
    if sub_target.startswith("Ascend910B"):
        mix_mode = metadata["mix_mode"]
        if mix_mode.lower().strip("_").startswith("aiv"):
            return "ascend_910b_vec", "c220-vec", "aiv"
        elif mix_mode.lower().strip("_").startswith("aic"):
            return "ascend_910b_cube", "c220-cube", "aic"
        else:
            return "ascend_910b", "c220", "mix"
    elif sub_target.startswith("Ascend910"):
        return "ascend_910", "c100", "mix"
    else:
        raise NotImplementedError(f"NPU subtarget {sub_target} not supported yet")


def _check_cxx11_abi():
    import torch

    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12,  # BOOL
        # Integer types
        "i8": 2,  # INT8
        "i16": 6,  # INT16
        "i32": 3,  # INT32
        "i64": 9,  # INT64
        # Unsigned integer types
        "u32": 8,  # UINT32
        "u64": 10,  # UINT64
        # Floating point types
        "fp16": 1,  # FLOAT16
        "bf16": 27,  # DT_BF16
        "fp32": 0,  # FLOAT
        "fp64": 11,  # DOUBLE
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")

    return MAP_SIGTYPE_TO_INT[sigty]


def _check_bishengir_is_regbased() -> bool:
    bishengir_path = _get_npucompiler_path()
    try:
        result = subprocess.run(
            f"{bishengir_path} --help | grep 'reg-based'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            # bishengir-compile is regbased version
            return True
        else:
            # bishengir-compile is membased version
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


###################### utils.py end ######################


# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    # return lambda lhsType, rhsType: (16, 16, 16)
    return lambda lhsType, rhsType: (1, 1, 1)


def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.md5(f"{mod}-{metadata}".encode()).hexdigest()
    mod.set_attr("dicp.backend", ir.builder(mod.context).get_string_attr("ascend"))
    # the same optimize pass for triton-ir as all other backends
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
    if opt.debug or dump_ir:
        dicp_utils._dump_stage_ir(str(mod), metadata["hash"], "kernel.ttir.mlir")
    return mod


def ttir_to_linalg(mod, metadata, opt, *, named_ops=True):
    # use triton_adapter to lower Triton-MLIR to linalg
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        cmd_list = [
            triton_adapter_opt_path,
            src_path,
            f"--triton-to-linalg=global-kernel=false named-ops={named_ops}",
            "-o",
            dst_path,
        ]
        if _is_ascend_sanitizer_enabled():
            cmd_list += ["--mlir-print-debuginfo"]  # pass debug info

        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(
                Path(dst_path).read_text(), "kernel.ttadapter.mlir", binary=False
            )

        return Path(dst_path).read_text()


def ttir_to_ttsharedir_ascend(mod, metadata, opt, *, named_ops=False):
    pm = ir.pass_manager(mod.context)
    dicp_triton.passes.triton_shared_ascend.add_canonicalize_cmpi(pm)
    dicp_triton.passes.triton_shared_ascend.add_canonicalize_triton_ir_ascend(pm)
    dicp_triton.passes.triton_shared_ascend.add_triton_to_linalg_npu(pm)
    pm.run(mod)
    if opt.debug or dump_ir:
        cmd_list = [
            _get_dicp_opt_path(),
            "kernel.ttir.mlir",
            "--canonicalize-cmpi",
            "--canonicalize-triton-ir-ascend",
            "--triton-to-linalg-npu-conversion",
        ]
        dicp_utils._dump_stage_ir(
            str(mod), metadata["hash"], "kernel.ttshared.mlir", cmd_list
        )
    if replace_ttshared_ir is not None:
        return _replace_mod_ir_with_file(
            mod, replace_ttshared_ir, "ttir_to_ttsharedir_ascend"
        )
    return mod


def ttsharedir_to_linkedir(mod, metadata, opt, *, named_ops=False):
    pm = ir.pass_manager(mod.context)
    dicp_triton.passes.linked_npu.add_lower_affine(pm)
    dicp_triton.passes.linked_npu.add_normalize_slice_ops(pm)
    dicp_triton.passes.linked_npu.add_linalg_if_to_select(pm)
    dicp_triton.passes.linked_npu.add_linalg_generic_to_scf(pm)
    dicp_triton.passes.linked_npu.add_scalar_to_1d_tensor(pm)
    dicp_triton.passes.linked_npu.add_linalg_to_linked(pm, False, True)
    dicp_triton.passes.linked_npu.add_linked_to_hivm(pm)
    pm.run(mod)

    # TODO(zmz): 修改test_path 中内容，暂时在python中处理，bishengir-compile后续会支持，去掉这里逻辑。
    content = str(mod)
    # 将"*xfxxx"替换成"?xfxxx"
    content = content.replace("*xf", "?xf")
    content = content.replace("*xi", "?xi")
    content = content.replace("*xbf", "?xbf")
    # 匹配形如 "memref<...> to tensor<...>" 的模式
    pattern = r"(memref\<.*?\>)\s+to\s+(tensor\<.*?\>)"
    # 使用正则替换，保留memref和tensor类型，中间插入注释
    content = re.sub(pattern, r"\1 // to \2", content)

    if opt.debug or dump_ir:
        cmd_list = [
            _get_dicp_opt_path(),
            "kernel.ttshared.mlir",
            "--lower-affine",
            "--normalize-slice-ops",
            "--linalg-if-to-select",
            "--linalg-generic-to-scf",
            "--scalar-to-1d-tensor",
            f"--linalg-to-linked=global-kernel=false named-ops=true",
            "--linked-to-hivm",
        ]
        dicp_utils._dump_stage_ir(
            content, metadata["hash"], "kernel.linkedir.mlir", cmd_list
        )

    if replace_linked_ir is not None:
        print(f"[DEBUG] Replace Linkedir with {replace_linked_ir}")
        return Path(replace_linked_ir).read_text()
    return content


def linalg_to_llir(linalg: str, metadata, opt):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        llmlir_path = os.path.join(tmpdir, "kernel.llir.mlir")
        llir_path = os.path.join(tmpdir, "kernel.ll")
        Path(ttadapter_path).write_text(linalg)
        mlir_opt_path = _get_mlir_path("bin", "mlir-opt")
        # TritonAdapter-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                mlir_opt_path,
                ttadapter_path,
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
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
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

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_mlir_path("bin", "mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(Path(llir_path).read_text(), "kernel.ll", binary=False)

        return Path(llir_path).read_text()


def llir_to_cpuasm(llir: str, metadata, opt):
    # add metadata at final stage
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, CPU backend requires no limit on shared memory size
    metadata["shared"] = 1
    # We can get a function name (C naming) from
    # LLVM-IR by getting the first "define void @".
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

        # Actually it's text-format assembly.  Use read_text().
        return Path(dst_path).read_text()


def __get_metadata_attr_by_callback(lib, postfix: str, metadata, meta_key: str):
    func_symbol = metadata["kernel_name"] + postfix
    if hasattr(lib, func_symbol):
        callback_func = getattr(lib, func_symbol)
        callback_func.restype = ctypes.c_int64
        callback_func.argtypes = []
        metadata[meta_key] = callback_func()


def _parse_linalg_metadata(linalg: str, metadata: dict):
    """
    Parse Linalg IR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - mix_mode
      - kernel_name
      - tensor_kinds
      - shared (currently hardcoded)
      - name (combined kernel_name and mix_mode)
    Additionally, removes the mix_mode attribute from the IR.
    """
    # --- Regular expressions and examples ---
    # Example: mix_mode = "aiv" -> aiv
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'
    # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"
    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = (
        r"%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}"
    )
    # Example removal:   ', mix_mode = "aiv"' → ''
    REMOVE_MIX_MODE_REGEX = r', mix_mode\s*=\s*"[^"]*"'
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    # Use while space to split kernel_name and mix_mode.
    # Check the function load_binary in npu_driver.py.
    metadata["name"] = metadata["kernel_name"] + " " + metadata["mix_mode"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [
        int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)
    ]
    # remove the mix_mode attribute
    linalg = re.sub(REMOVE_MIX_MODE_REGEX, "", linalg)
    return linalg, metadata


def linalg_to_bin_enable_npu_compile(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        lower_by_ttshared = os.getenv("LOWER_BY_TTSHARED", "1")
        if lower_by_ttshared == "1":
            ttadapter_path = os.path.join(tmpdir, "kernel.linkedir.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_is_regbased():
            bishengir_hivm_opt = "--reg-based=true"
        else:
            bishengir_hivm_opt = "--enable-hivm-compile=true"
        bin_path = os.path.join(tmpdir, "kernel_reloc.o")
        callback_path = os.path.join(tmpdir, "libkernel.so")
        multibuffer = metadata["multibuffer"]
        _compile_option_list = []
        _compile_option_list += [
            f"--enable-auto-multi-buffer={multibuffer}",
        ]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if _is_auto_map_parallel_blocks_enabled():
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path = _get_npucompiler_path()

        # support bishengir-compile more version
        if "8.2.RC1.alpha002" in npu_compiler_path:
            bin_path = os.path.join(tmpdir, "kernel_reloc.o")
        elif "8.2.RC1.alpha003" in npu_compiler_path:
            bin_path = os.path.join(tmpdir, "kernel.o")
        else:
            bin_path = os.path.join(tmpdir, "kernel.o")

        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                bishengir_hivm_opt,
                "--enable-triton-kernel-compile=true",
            ]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += [
                f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"
            ]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += [
                f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"
            ]

        disable_auto_cv_work_space_manage = metadata[
            "disable_auto_cv_work_space_manage"
        ]
        if disable_auto_cv_work_space_manage is True:
            _compile_option_list += [
                f"--disable-auto-cv-work-space-manage={disable_auto_cv_work_space_manage}"
            ]

        enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
        if enable_auto_bind_sub_block is not None:
            _compile_option_list += [
                f"--enable-auto-bind-sub-block={enable_auto_bind_sub_block}"
            ]

        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        if dump_ir:
            print(f"DEBUG dump ir[bishengir-compile] command: {cmd_list}")
        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if not Path(bin_path).is_file():
            print(ret.stderr.decode("utf-8"))
        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
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


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = False
    # sanitize_overflow: bool = True
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = 2
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    enable_warp_specialization: bool = False
    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    enable_npu_compile: bool = True
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    multibuffer: bool = True
    inject_barrier_all: bool = False
    disable_auto_inject_block_sync: bool = False
    disable_auto_cv_work_space_manage: bool = False
    enable_auto_bind_sub_block: bool = True

    stream: int = None

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    llvm_version: int = 15
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


class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "npu_utils.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "npu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "npu_utils.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_npu_ext("npu_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device):
        fnname, mix_mode = name.split()
        return self.npu_utils_mod.load_kernel_binary(
            fnname, kernel, shared, device, mix_mode
        )

    @functools.lru_cache()
    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid triton-compiler complain
        # fetch available memory at runtime
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_aicore_num()

    @functools.lru_cache()
    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]


class NPULauncher(object):
    def __init__(self, src, metadata):
        debug_mode = metadata.debug
        workspace_size = (
            int(metadata.workspace_size) if hasattr(metadata, "workspace_size") else -1
        )
        lock_init_value = (
            int(metadata.lock_init_value) if hasattr(metadata, "lock_init_value") else 0
        )
        lock_num = int(metadata.lock_num) if hasattr(metadata, "lock_num") else -1
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        mix_mode = metadata.mix_mode
        wrapper_src = generate_npu_wrapper_src(
            constants, signature, workspace_size, mix_mode, lock_num, lock_init_value
        )
        so_launcher_path = make_npu_launcher_stub(wrapper_src, debug_mode)
        # initialize launcher
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "__triton_launcher", so_launcher_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


def make_npu_launcher_stub(src, debug=False):
    """
    Generate the launcher stub to launch the kernel
    """
    # try to get cached file
    so_cache_key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    so_cache_manager = get_cache_manager(so_cache_key)
    # append the cxx11_abi value to the launcher name to avoid
    # linking to a launcher with wrong cxx11_abi.
    use_cxx11_abi = _check_cxx11_abi()
    name = f"launcher_cxx11abi{use_cxx11_abi}"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_name = f"{name}{suffix}"

    if debug:
        dump_manager = get_dump_manager(so_cache_key)
        print(f"Dumping {name}.cxx to {dump_manager.cache_dir}")
        dump_manager.put(src, f"{name}.cxx", binary=False)

    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is not None:
        return cache_path

    with tempfile.TemporaryDirectory() as tmpdir:
        if debug:
            so_cache_manager.put(src, f"{name}.cxx", binary=False)
        src_path = os.path.join(tmpdir, f"{name}.cxx")
        with open(src_path, "w") as f:
            f.write(src)
        enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", "true").lower() in (
            "true",
            "1",
        )
        if enable_taskqueue:
            kernel_launcher_type = "torch"
        else:
            kernel_launcher_type = None
        so = _build_npu_ext(
            name, src_path, tmpdir, kernel_launcher=kernel_launcher_type
        )
        if debug:
            with open(so, "rb") as f:
                return dump_manager.put(f.read(), so_name, binary=True)
        with open(so, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)


def extract_device_print_code_from_cann():
    from triton.backends.dicp_triton.npu import _get_bisheng_path

    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(
        ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib"
    )

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), "r") as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = " ".join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = "\n".join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer("\[aicore\]", code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if (
                    src[i] == "}"
                ):  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == "{":
                    brace_nest += 1
                elif src[j] == "}":
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ""
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace("__gm__", " ")
        new_code = new_code.replace("__CCELIB_RT_ERROR_NONE", "RT_ERROR_NONE")
        new_code = new_code.replace("__CCELIB_RT_MEMORY_HBM", "RT_MEMORY_HBM")
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_HOST_TO_DEVICE", "RT_MEMCPY_HOST_TO_DEVICE"
        )
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_DEVICE_TO_HOST", "RT_MEMCPY_DEVICE_TO_HOST"
        )
        return new_code

    # the following headers should be included in this order
    headers_combined = "\n".join(
        [
            read_header("common/common_impl.h"),
            read_header("internal/debug_tunnel/payload.h"),
            read_header("internal/debug_tunnel/payload_impl.h"),
            read_header("internal/debug_tunnel/tunnel.h"),
            read_header("internal/debug_tunnel/tunnel_impl.h"),
        ]
    )
    # Prepend the needed include so generated code has std::cout / std::endl available
    return "#include <iostream>\n" + headers_combined


# the template is from triton-adapter HEAD. Wrapping the generated kernel binary into a python module
def generate_npu_wrapper_src(
    constants, signature, workspace_size, mix_mode, lock_num, lock_ini_val
):
    import os

    # TODO(zmz)，临时方案signature 的value 中，如果有*u1，换成*i1
    signature = {k: v.replace("*u1", "*i1") for k, v in signature.items()}

    def _ty_to_cpp(ty):
        if ty[0] == "*":
            return "void*"
        if ty == "constexpr":
            return "PyObject*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _extracted_ty(ty):
        if ty[0] == "*":
            return "PyObject*"
        if ty == "constexpr":
            return "PyObject*"
        return {
            "i1": "int32_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gridX, gridY, gridZ;
        rtStream_t stream;
        const void *functon;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + "".join(
        [_format_of(_extracted_ty(ty)) for ty in signature.values()]
    )

    grid_info = {"X": "i32", "Y": "i32", "Z": "i32"}

    enable_device_print = os.getenv("TRITON_DEVICE_PRINT", "false").lower() in (
        "true",
        "1",
    )
    enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", "true").lower() in (
        "true",
        "1",
    )
    enable_auto_map_parallel_blocks = _is_auto_map_parallel_blocks_enabled()
    npu_utils = NPUUtils()
    num_physical_blocks = (
        npu_utils.get_aivector_core_num()
        if mix_mode == "aiv"
        else npu_utils.get_aicore_num()
    )
    task_type = (
        "MSPROF_GE_TASK_TYPE_AIV"
        if mix_mode == "aiv"
        else "MSPROF_GE_TASK_TYPE_AI_CORE"
    )
    LINE_CHANGE_CHAR = chr(10)  # it is \n

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {LINE_CHANGE_CHAR.join(
          f'dataTypes[{i}] = {convert_sigtype_to_int(ty[1:])};'
          for i, ty in signature.items()
          if ty.startswith("*") and i < 5
        )}
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string>
#include <sys/syscall.h>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
{'#include <torch_npu/csrc/framework/OpCommand.h>' if enable_taskqueue else ''}
#include "experiment/runtime/runtime/rt.h"
{extract_device_print_code_from_cann() if enable_device_print else ''}

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds{', ' + arg_decls if len(signature) > 0 else ''}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
   {'auto launch_call = [=]()' if enable_taskqueue else ''} {{
    uint32_t blockNum = gridX * gridY * gridZ;
    {'if (blockNum > (uint32_t)' + str(num_physical_blocks) + ') { std::cout << "WARNING: Grid " << blockNum << " > physical limit ' + str(num_physical_blocks) + ', performance maybe reduced." << std::endl;if (blockNum > 65535 && !' + str(enable_auto_map_parallel_blocks).lower() + ') {std::cout << "Grid " << blockNum << " > 65535, Please set TRITON_ALL_BLOCKS_PARALLEL=1 to enable all blocks parallel execution." << std::endl; } }'}

    {'blockNum = std::min(blockNum, (uint32_t)' + str(num_physical_blocks) + ');' if enable_auto_map_parallel_blocks else ''}
    {'cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);' if enable_device_print else ''}
    rtError_t ret;
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    // stub argument for workspace
    void *syncBlockLock = NULL;
    void *workspace_addr = NULL;
    uint16_t ModuleId = 0;
    {f'''
    uint64_t syncBlockLockSize = {lock_num} * sizeof(int64_t);
    ret = rtMalloc(reinterpret_cast<void **>(&syncBlockLock),
                   syncBlockLockSize, RT_MEMORY_HBM, 0);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    std::vector<int64_t> lockInitData({lock_num}, {lock_ini_val});
    ret = rtMemcpy(syncBlockLock, syncBlockLockSize, reinterpret_cast<void *>(lockInitData.data()),
                   syncBlockLockSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if lock_num > 0 else ''}
    {f'''
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    ret = rtMalloc(reinterpret_cast<void **>(&workspace_addr),
                   totalWorkSpaceSize, RT_MEMORY_HBM, ModuleId);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if workspace_size > 0 else ''}
    struct __attribute__((packed)) {{
      void* ffts_addr __attribute__((aligned(8)));
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if ty != "constexpr")}
      {' '.join(f'{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items() if ty != "constexpr")}
      {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
    }} args = {{
      static_cast<void*>(ffts_addr),
      static_cast<void*>(syncBlockLock),
      static_cast<void*>(workspace_addr),
      {(', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if ty != "constexpr") + ',') if len(signature) > 0 else ''}
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items() if ty != "constexpr")}
      {', static_cast<void*>(DTData)' if enable_device_print else ''}
    }};
    {cpp_msprof_call_before_launch}
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
    {'void *&stream_ref = const_cast<void*&>(stream);' if enable_device_print else ''}
    {'cce::internal::DebugTunnel::Close(DTData, stream_ref);' if enable_device_print else ''}
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else ''}
   }};
   {'at_npu::native::OpCommand cmd; cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();' if enable_taskqueue else ''}
  return;
}}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {{
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {{
    return shape;
  }}

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {{
    return shape;
  }}
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {{
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {{
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {{
        shape.push_back(PyLong_AsLong(dim));
      }}
    }}
  }}
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}
  if (__MsprofFlagL1)
  {{
    {
      LINE_CHANGE_CHAR.join(
        f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
        for i, ty in signature.items() if ty[0] == "*"
      )
    }
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {{
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {{
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }}
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0]=="*" else "" for i, ty in signature.items()])};
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}
  Py_RETURN_NONE;
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
  {cpp_msprof_callback}
  return m;
}}
"""
