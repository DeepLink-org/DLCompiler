import functools
import hashlib
import json
from pathlib import Path
from typing import Any, List
from triton._C.libtriton import get_cache_invalidating_env_vars

from triton.runtime.cache import triton_key
from .backend import commonir_backend
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import AsmDict, _raise_error
from triton.compiler.compiler import LazyDict
from triton.runtime.cache import get_cache_manager


class CommonIRSource:
    def __init__(self, src: str, grid: List[int], signature: dict):
        self.src = src
        self.grid = grid
        self.signature = signature


class CompiledKernel:
    def __init__(self, src: CommonIRSource, metadata_group, hash):
        from collections import namedtuple

        metadata_path = next(
            (Path(p) for c, p in metadata_group.items() if c.endswith(".json"))
        )
        metadata = json.loads(metadata_path.read_text())
        metadata["cluster_dims"] = tuple(metadata["cluster_dims"])
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata["target"]
        metadata["target"] = GPUTarget(
            target["backend"], target["arch"], target["warp_size"]
        )
        KernelMetadata = namedtuple("KernelMetadata", sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        self.packed_metadata = commonir_backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        self.grid = src.grid
        # stores the text of each level of IR that was generated during compilation
        asm_files = [
            Path(p) for c, p in metadata_group.items() if not c.endswith(".json")
        ]
        binary_ext = commonir_backend.binary_ext
        self.asm = AsmDict(
            {
                file.suffix[1:]: (
                    file.read_bytes()
                    if file.suffix[1:] == binary_ext
                    else file.read_text()
                )
                for file in asm_files
            }
        )
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None
        self._run = None

    def _init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            self._run = functools.partial(_raise_error, err)
            raise err

        device = commonir_backend.get_driver().get_current_device()
        # create launcher
        self._run = commonir_backend.get_driver().launcher_cls(self.src, self.metadata)
        (
            self.module,
            self.function,
            self.n_regs,
            self.n_spills,
        ) = commonir_backend.get_driver().utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device
        )

    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run

    def launch_metadata(self, grid, stream, *args):
        self._init_handles()
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        return ret

    def __call__(self, *args: Any) -> Any:
        device = commonir_backend.get_driver().get_current_device()
        stream = commonir_backend.get_driver().get_current_stream(device)
        # launch kernel

        launch_metadata = self.launch_metadata(self.grid, stream, *args)
        self.run(
            self.grid[0],
            self.grid[1],
            self.grid[2],
            stream,
            self.function,
            self.packed_metadata,
            launch_metadata,
            None,  # knobs.runtime.launch_enter_hook,
            None,  # knobs.runtime.launch_exit_hook,
            *args,
        )

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = commonir_backend.get_driver().get_current_device()
                stream = commonir_backend.get_driver().get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(
                grid[0],
                grid[1],
                grid[2],
                stream,
                self.function,
                self.packed_metadata,
                launch_metadata,
                None,  # knobs.runtime.launch_enter_hook,
                None,  # knobs.runtime.launch_exit_hook,
                *args,
            )

        return runner


class CommonIRCompiler(object):

    def compile(self, commonir_src: CommonIRSource, options=None, _env_vars=None):

        target = commonir_backend.get_driver().get_current_target()
        assert isinstance(target, GPUTarget), "target must be of GPUTarget type"

        extra_options = {}
        options = commonir_backend.parse_options(
            dict(options or dict(), **extra_options)
        )
        # create cache manager
        env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars

        src_hash = hashlib.sha256(commonir_src.src.encode("utf-8")).hexdigest()
        key = f"{triton_key()}-{src_hash}-{commonir_backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
        hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        fn_cache_manager = get_cache_manager(hash)
        store_only_binary = False
        file_name = "tilelang-commonir"
        metadata_filename = f"{file_name}.json"
        metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
        # initialize metadata
        metadata = {
            "hash": hash,
            "target": target,
            **options.__dict__,
            **env_vars,
        }
        # run compilation pipeline  and populate metadata
        stages = dict()
        commonir_backend.add_stages(stages, options)
        module = commonir_src.src
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

        for ext, compile_ir in list(stages.items()):
            next_module = compile_ir(module, metadata)
            ir_filename = f"{file_name}.{ext}"
            if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
                metadata_group[ir_filename] = fn_cache_manager.put(
                    next_module, ir_filename
                )
            module = next_module
        # write-back metadata
        metadata_group[metadata_filename] = fn_cache_manager.put(
            json.dumps(metadata, default=vars), metadata_filename, binary=False
        )
        fn_cache_manager.put_group(metadata_filename, metadata_group)
        return CompiledKernel(commonir_src, metadata_group, hash)

    @functools.lru_cache()
    def hash(self):
        return "CommonIRCompiler"
