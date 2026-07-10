"""Ascend autotune runtime: auto-tiling + compile-option search.

Previously from triton-ascend/third_party/ascend/backend/runtime.
Adapted for DLCompiler (triton.backends.dicp_triton).
"""

from .kernel_ast_analyzer import (
    AutoParser,
    AxesKeyParser,
    SplitAxesParser,
    TilingAxesParser,
    ReductionAxesParser,
    LowDimsAxesParser,
    PtrNumsParser,
)
from .tile_candidate_generator import AxisInfo, BlockInfo, KernelMeta, TileGenerator
from .schedule_profiles import (
    CompileOptionsSpec,
    CompileFailureRegionSet,
    classify_compile_failure,
    compile_profile_to_config,
    effective_compile_profile_key,
    expand_compile_option_configs,
    generate_linked_compile_neighbors,
    get_stage1_probe_configs,
    get_stage1_probe_profiles,
    make_stage2_seed_profiles,
    parse_compile_options_hint,
    validate_compile_profile,
)
from .ascend_kernel_autotuner import (
    AutoTilingTuner,
    autotune,
    max_autotune,
    get_max_configs,
    BaseAutotuner,
    CubeAutotuner,
    MixcvAutotuner,
    VectorAutotuner,
    get_autotune_cube_config,
    get_autotune_cv_config,
    get_autotune_vector_config,
)

__all__ = [
    "AutoParser",
    "AxesKeyParser",
    "SplitAxesParser",
    "TilingAxesParser",
    "ReductionAxesParser",
    "LowDimsAxesParser",
    "PtrNumsParser",
    "AxisInfo",
    "BlockInfo",
    "KernelMeta",
    "TileGenerator",
    "CompileOptionsSpec",
    "CompileFailureRegionSet",
    "classify_compile_failure",
    "compile_profile_to_config",
    "effective_compile_profile_key",
    "expand_compile_option_configs",
    "generate_linked_compile_neighbors",
    "get_stage1_probe_configs",
    "get_stage1_probe_profiles",
    "make_stage2_seed_profiles",
    "parse_compile_options_hint",
    "validate_compile_profile",
    "AutoTilingTuner",
    "autotune",
    "max_autotune",
    "get_max_configs",
    "BaseAutotuner",
    "CubeAutotuner",
    "MixcvAutotuner",
    "VectorAutotuner",
    "get_autotune_cube_config",
    "get_autotune_cv_config",
    "get_autotune_vector_config",
]
