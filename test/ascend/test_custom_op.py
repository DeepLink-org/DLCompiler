#!/usr/bin/env python3
import subprocess
import os
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl

from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
import hashlib
from triton.backends.dicp_triton.npu import NPUUtils
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend, IRSource, filter_traceback
from triton import __version__, knobs
from triton.runtime.cache import (
    get_cache_manager,
    get_dump_manager,
    get_override_manager,
    get_cache_key,
)

from triton.backends.dicp_triton.npu import (
    make_ttir,
    ttir_to_linalg,
    ttir_to_ttsharedir_ascend,
    ttsharedir_to_linkedir,
    linalg_to_bin_enable_npu_compile,
    NPUOptions,
)


@dl.register_custom_op
class add:
    core = dl.CORE.VECTOR
    pipe = dl.PIPE.PIPE_V
    mode = dl.MODE.SIMD

    def __init__(self, a, b, out=None):
        assert out, "out is required"
        self.symbol = "custom_add_" + str(a.dtype)
        # self.bitcode defaults to the Ascend installation directory
        # Typically it would be a specific bitcode file like /path/to/kernel.aiv.bc
        self.bitcode = "/usr/local/Ascend/"


@triton.jit
def triton_custom_add(output_ptr, a_ptr, b_ptr, L: tl.constexpr):
    idx = tl.arange(0, L)

    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)

    buf = tl.full([L], 0, a.dtype)
    res = dl.custom("add", a, b, out=buf)

    tl.store(output_ptr + idx, res)


def compile(src, target=None, options=None, _env_vars=None):
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        context = ir.context()
        src = IRSource(src, context, backend)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = knobs.compilation.override
    enable_ir_dump = knobs.compilation.dump_ir
    store_only_binary = knobs.compilation.store_binary_only
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res

    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options, src.language)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource):
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    try:
        module = src.make_ir(target, options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise

    if ir_source:
        ir_filename = f"{file_name}.{src.ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    use_ir_loc = knobs.compilation.use_ir_loc
    if ir_source and use_ir_loc:
        module.create_location_snapshot(src.path)
        print(f"Creating new locations for {src.path}")

    if compilation_listener:
        timer.finished_ir_initialization()

    if "npubin" in stages.keys():
        del stages["npubin"]

    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if fn_override_manager is None:
            # Users can override kernels at scale by setting `ir_override` in autotune config
            # without TRITON_KERNEL_OVERRIDE
            if (
                ir_override := metadata.get("ir_override", None)
            ) and ir_override.endswith(f".{ext}"):
                next_module = parse(ir_override, ext, context)
        elif full_name := fn_override_manager.get_file(ir_filename):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
            if ext == "cubin":
                sass = get_sass(next_module)
                fn_dump_manager.put(sass, file_name + ".sass")
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)
    return module


if __name__ == "__main__":
    npuutiles = NPUUtils()
    src = ASTSource(
        triton_custom_add,
        {"output_ptr": "*i32", "a_ptr": "*i32", "b_ptr": "*i32"},
        {"L": 32},
    )
    target = GPUTarget(backend="ascend", arch=npuutiles.get_arch(), warp_size=0)
    options = {
        "debug": False,
        "sanitize_overflow": False,
        "llvm_version": 15,
        "kernel_name": "triton_",
        "cluster_dims": (1, 1, 1),
        "num_warps": -1,
        "num_ctas": -1,
        "num_stages": 2,
        "num_buffers_warp_spec": 0,
        "num_consumer_groups": 0,
        "reg_dec_producer": 0,
        "reg_inc_consumer": 0,
        "enable_warp_specialization": False,
        "enable_nd2nz_on_vector": False,
        "enable_persistent": False,
        "optimize_epilogue": False,
        "enable_fp_fusion": True,
        "allow_fp8e4nv": False,
        "allowed_dot_input_precisions": ("ieee", "hf32"),
        "enable_npu_compile": True,
        "max_num_imprecise_acc_default": None,
        "extern_libs": None,
        "multibuffer": True,
        "inject_barrier_all": False,
        "disable_auto_inject_block_sync": False,
        "unit_flag": False,
        "disable_auto_cv_work_space_manage": False,
        "enable_auto_bind_sub_block": True,
        "tile_mix_vector_loop": None,
        "tile_mix_cube_loop": None,
        "limit_auto_multi_buffer_only_for_local_buffer": None,
        "set_workspace_multibuffer": None,
        "stream": None,
    }
    linkedir = compile(src, target, options, {})

    print("=== MLIR (linkedir) ===")
    print(linkedir)
