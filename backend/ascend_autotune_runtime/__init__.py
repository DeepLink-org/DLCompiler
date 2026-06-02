"""Ascend autotune runtime: auto-tiling + compile-option search.

Previously from triton-ascend/third_party/ascend/backend/runtime.
Adapted for DLCompiler (triton.backends.dicp_triton).
"""

from .autoparser import (
    AutoParser,
    AxesKeyParser,
    SplitAxesParser,
    TilingAxesParser,
    ReductionAxesParser,
    LowDimsAxesParser,
    PtrNumsParser,
)
from .tile_generator import AxisInfo, BlockInfo, KernelMeta, TileGenerator
from .autotuner import (
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
