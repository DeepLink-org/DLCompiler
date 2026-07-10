from __future__ import annotations

import inspect
import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from triton.runtime.autotuner import Config
from triton.backends.dicp_triton.utils import is_compile_on_910_95


DEFAULT_MAX_CONFIGS = None

_VALID_VALUES = {
    "num_stages": [1, 2],
    "limit_auto_multi_buffer_of_local_buffer": ["no-limit", "no-l0c"],
    "set_workspace_multibuffer": [2, 4],
    "tile_mix_vector_loop": [1, 2, 4, 8],
    "tile_mix_cube_loop": [1, 2, 4, 8],
}

_BOOLEAN_PARAMS = {
    "enable_tuning_mode",
    "multibuffer",
    "unit_flag",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "enable_hivm_auto_cv_balance",
    "enable_ubuf_saving",
    # "enable_preload",
    "enable_auto_bind_sub_block",
}

_SUPPORTED_PARAMS = {
    "cube": {
        "enable_tuning_mode",
        "num_stages",
        "unit_flag",
        "limit_auto_multi_buffer_of_local_buffer",
    },
    "mixcv": {
        "enable_tuning_mode",
        "num_stages",
        "multibuffer",
        "unit_flag",
        "limit_auto_multi_buffer_only_for_local_buffer",
        "limit_auto_multi_buffer_of_local_buffer",
        "set_workspace_multibuffer",
        "enable_hivm_auto_cv_balance",
        "tile_mix_vector_loop",
        "tile_mix_cube_loop",
        "enable_ubuf_saving",
        # "enable_preload",
        "enable_auto_bind_sub_block",
    },
    "vector": {
        "num_stages",
        "enable_ubuf_saving",
    },
}

_ALL_PARAMS = set().union(*_SUPPORTED_PARAMS.values())

_MIXCV_910_95_UNSUPPORTED_PARAMS = {
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
}

COMPILE_MODE_KEY = "mode"
COMPILE_MODE_VECTOR = "VECTOR"
COMPILE_MODE_MB_OFF = "MB_OFF"
COMPILE_MODE_LOCAL_MB = "LOCAL_MB"
COMPILE_MODE_WORKSPACE_CV = "WORKSPACE_CV"

FAILURE_EXACT_ONLY = "EXACT_ONLY"
FAILURE_RESOURCE_UB = "RESOURCE_UB"
FAILURE_RESOURCE_L0C = "RESOURCE_L0C"
FAILURE_RESOURCE_L1 = "RESOURCE_L1"
FAILURE_RESOURCE_WORKSPACE = "RESOURCE_WORKSPACE"
FAILURE_SYNC_OR_CORRECTNESS = "SYNC_OR_CORRECTNESS"
FAILURE_COMPILER_INTERNAL = "COMPILER_INTERNAL"

_RESOURCE_FAILURES = {
    FAILURE_RESOURCE_UB,
    FAILURE_RESOURCE_L0C,
    FAILURE_RESOURCE_L1,
    FAILURE_RESOURCE_WORKSPACE,
}

_ALL_FAILURE_KINDS = {
    FAILURE_EXACT_ONLY,
    FAILURE_RESOURCE_UB,
    FAILURE_RESOURCE_L0C,
    FAILURE_RESOURCE_L1,
    FAILURE_RESOURCE_WORKSPACE,
    FAILURE_SYNC_OR_CORRECTNESS,
    FAILURE_COMPILER_INTERNAL,
}

_MULTI_BUFFER_CHILD_PARAMS = {
    "limit_auto_multi_buffer_only_for_local_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
}

_WORKSPACE_CV_PARAMS = {
    "set_workspace_multibuffer",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
}

WORKSPACE_CV_AGGRESSIVE_PROBE = {
    COMPILE_MODE_KEY: COMPILE_MODE_WORKSPACE_CV,
    "enable_tuning_mode": True,
    "num_stages": 2,
    "multibuffer": True,
    "enable_auto_bind_sub_block": True,
    "enable_hivm_auto_cv_balance": True,
    "enable_ubuf_saving": True,
    "limit_auto_multi_buffer_only_for_local_buffer": False,
    "limit_auto_multi_buffer_of_local_buffer": "no-limit",
    "set_workspace_multibuffer": 4,
    "tile_mix_cube_loop": 4,
    "tile_mix_vector_loop": 4,
    "unit_flag": False,
}

WORKSPACE_CV_MIX1_PROBE = {
    COMPILE_MODE_KEY: COMPILE_MODE_WORKSPACE_CV,
    "enable_tuning_mode": True,
    "num_stages": 2,
    "multibuffer": True,
    "enable_auto_bind_sub_block": True,
    "enable_hivm_auto_cv_balance": True,
    "enable_ubuf_saving": True,
    "limit_auto_multi_buffer_only_for_local_buffer": False,
    "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
    "set_workspace_multibuffer": 2,
    "tile_mix_cube_loop": 1,
    "tile_mix_vector_loop": 1,
    "unit_flag": False,
}

WORKSPACE_CV_LOW_RESOURCE_PROBE = {
    COMPILE_MODE_KEY: COMPILE_MODE_WORKSPACE_CV,
    "enable_tuning_mode": True,
    "num_stages": 2,
    "multibuffer": True,
    "enable_auto_bind_sub_block": True,
    "enable_hivm_auto_cv_balance": True,
    "enable_ubuf_saving": True,
    "limit_auto_multi_buffer_only_for_local_buffer": False,
    "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
    "set_workspace_multibuffer": 2,
    "tile_mix_cube_loop": 4,
    "tile_mix_vector_loop": 4,
    "unit_flag": False,
}

DEFAULT_STAGE1_PROBE_PROFILES = [
    WORKSPACE_CV_AGGRESSIVE_PROBE,
    WORKSPACE_CV_LOW_RESOURCE_PROBE,
    WORKSPACE_CV_MIX1_PROBE,
]

VECTOR_STAGE1_PROBE = {
    COMPILE_MODE_KEY: COMPILE_MODE_VECTOR,
    "num_stages": 2,
    "enable_ubuf_saving": True,
}

DEFAULT_VECTOR_STAGE1_PROBE_PROFILES = [
    VECTOR_STAGE1_PROBE,
]

CONSERVATIVE_MIXCV_STAGE1_PROBE_PROFILES = [
    WORKSPACE_CV_LOW_RESOURCE_PROBE,
    WORKSPACE_CV_MIX1_PROBE,
]

DEFAULT_COMPILE_ANNEALING_OPTIONS = {
    "seed_budget": 8,
    "max_compile_trials_per_shape": 16,
    "neighbors_per_step": 2,
    "compile_initial_temperature": 0.20,
    "compile_cooling": 0.85,
    "candidate_pool_per_shape": 3,
    "random_seed": 0,
}

DEFAULT_UB_LIMIT_BYTES = 192 * 1024
LOCAL_ONLY_PROBE_UB_THRESHOLD_BYTES = 100 * 1024
TIGHT_UB_MARGIN_BYTES = 32 * 1024

_AUTO_SEARCH_PRESETS = {
    "cube": {
        "enable_tuning_mode": [True],
        "num_stages": [1, 2],
        "unit_flag": [False, True],
        "limit_auto_multi_buffer_of_local_buffer": ["no-limit", "no-l0c"],
    },
    "mixcv": {
        "enable_tuning_mode": [True],
        "num_stages": [1, 2],
        "unit_flag": [False, True],
        "limit_auto_multi_buffer_only_for_local_buffer": [True, False],
        "limit_auto_multi_buffer_of_local_buffer": ["no-limit", "no-l0c"],
        "set_workspace_multibuffer": [2, 4],
        "enable_hivm_auto_cv_balance": [True],
        "tile_mix_vector_loop": [1, 2, 4],
        "tile_mix_cube_loop": [1, 2, 4],
        "enable_ubuf_saving": [False, True],
        # "enable_preload": [False, True],
        "enable_auto_bind_sub_block": [True],
    },
    "vector": {
        "num_stages": [1, 2],
        "enable_ubuf_saving": [True, False],
    },
}


def _is_mixcv_multi_buffer_auto_enabled(
    num_stages: int,
    combo: Dict[str, Any],
    config: Config,
    fixed_options: Dict[str, Any],
) -> bool:
    """Whether `enable_auto_multi_buffer` takes effect for this combination."""
    if num_stages == 1:
        return False
    multibuffer = _resolve_compile_option(
        "multibuffer", combo, config, fixed_options, default=None
    )
    return multibuffer is not False


def _is_mixcv_limit_to_local_only_active(
    num_stages: int,
    combo: Dict[str, Any],
    config: Config,
    fixed_options: Dict[str, Any],
) -> bool:
    return _is_mixcv_multi_buffer_auto_enabled(num_stages, combo, config, fixed_options)


def _is_mixcv_workspace_multibuffer_active(
    num_stages: int,
    combo: Dict[str, Any],
    config: Config,
    fixed_options: Dict[str, Any],
) -> bool:
    if not _is_mixcv_multi_buffer_auto_enabled(
        num_stages, combo, config, fixed_options
    ):
        return False
    limit_to_local_only = _resolve_compile_option(
        "limit_auto_multi_buffer_only_for_local_buffer",
        combo,
        config,
        fixed_options,
        default=True,
    )
    return limit_to_local_only is False


_MIXCV_OPTION_ACTIVITY_RULES = {
    "limit_auto_multi_buffer_only_for_local_buffer": _is_mixcv_limit_to_local_only_active,
    "limit_auto_multi_buffer_of_local_buffer": _is_mixcv_limit_to_local_only_active,
    "set_workspace_multibuffer": _is_mixcv_workspace_multibuffer_active,
    "tile_mix_vector_loop": _is_mixcv_workspace_multibuffer_active,
    "tile_mix_cube_loop": _is_mixcv_workspace_multibuffer_active,
}


@dataclass
class CompileOptionsSpec:
    enabled: bool = False
    kernel_type: str = "mixcv"
    params: Dict[str, List[Any]] = field(default_factory=dict)
    max_configs: Optional[int] = DEFAULT_MAX_CONFIGS


def _normalize_kernel_type(kernel_type: str) -> str:
    if kernel_type == "mix":
        return "mixcv"
    if kernel_type not in _SUPPORTED_PARAMS:
        raise ValueError(
            "compile_options kernel_type must be one of: cube, mix, mixcv, vector"
        )
    return kernel_type


def _as_value_list(name: str, value: Any) -> List[Any]:
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"compile_options parameter '{name}' must not be empty")
    return values


def validate_compile_option_values(name: str, values: List[Any]) -> None:
    if name in _BOOLEAN_PARAMS and not all(isinstance(v, bool) for v in values):
        raise ValueError(f"compile_options parameter '{name}' expects boolean values")

    if name in _VALID_VALUES and not all(v in _VALID_VALUES[name] for v in values):
        raise ValueError(
            f"compile_options parameter '{name}' expects values in {_VALID_VALUES[name]}"
        )


def parse_compile_options_hint(hint: Any) -> CompileOptionsSpec:
    if hint is None or hint is False:
        return CompileOptionsSpec(enabled=False)

    if hint is True:
        return CompileOptionsSpec(enabled=True)

    if isinstance(hint, str):
        return CompileOptionsSpec(
            enabled=True,
            kernel_type=_normalize_kernel_type(hint),
        )

    if not isinstance(hint, dict):
        raise TypeError("hints['compile_options'] must be bool, str, or dict")

    raw = dict(hint)
    kernel_type = _normalize_kernel_type(
        raw.pop("kernel_type", raw.pop("type", "mixcv"))
    )
    max_configs = raw.pop("max_configs", DEFAULT_MAX_CONFIGS)
    if max_configs is not None and (
        not isinstance(max_configs, int) or max_configs <= 0
    ):
        raise ValueError(
            "compile_options max_configs must be a positive integer or None"
        )

    nested_options = raw.pop("options", {})
    if nested_options:
        if not isinstance(nested_options, dict):
            raise TypeError("compile_options options must be a dict")
        raw.update(nested_options)

    supported = _SUPPORTED_PARAMS[kernel_type]
    params: Dict[str, List[Any]] = {}
    for name, value in raw.items():
        if name not in _ALL_PARAMS:
            raise ValueError(f"Unknown compile_options parameter: {name}")
        if name not in supported:
            raise ValueError(
                f"compile_options parameter '{name}' is not supported "
                f"for kernel_type '{kernel_type}'"
            )
        values = _as_value_list(name, value)
        validate_compile_option_values(name, values)
        params[name] = values

    return CompileOptionsSpec(
        enabled=True,
        kernel_type=kernel_type,
        params=params,
        max_configs=max_configs,
    )


def _make_config_compat(**kwargs):
    supported_config_args = inspect.signature(Config).parameters
    return Config(
        **{key: value for key, value in kwargs.items() if key in supported_config_args}
    )


def _drop_unsupported_profile_options(profile: Mapping[str, Any]) -> Dict[str, Any]:
    copied = dict(profile)
    if is_compile_on_910_95:
        for name in _MIXCV_910_95_UNSUPPORTED_PARAMS:
            copied.pop(name, None)
    return copied


def _profile_mode(profile: Mapping[str, Any]) -> str:
    if COMPILE_MODE_KEY in profile:
        return profile[COMPILE_MODE_KEY]
    num_stages = profile.get("num_stages")
    multibuffer = profile.get("multibuffer", None)
    if num_stages == 1 or multibuffer is False:
        return COMPILE_MODE_MB_OFF
    if profile.get("limit_auto_multi_buffer_only_for_local_buffer", True) is True:
        return COMPILE_MODE_LOCAL_MB
    return COMPILE_MODE_WORKSPACE_CV


def _compile_profile_error(profile: Mapping[str, Any]) -> Optional[str]:
    mode = _profile_mode(profile)
    if mode not in {
        COMPILE_MODE_VECTOR,
        COMPILE_MODE_MB_OFF,
        COMPILE_MODE_LOCAL_MB,
        COMPILE_MODE_WORKSPACE_CV,
    }:
        return f"unknown compile profile mode: {mode}"

    supported = (
        _SUPPORTED_PARAMS["vector"]
        if mode == COMPILE_MODE_VECTOR
        else _SUPPORTED_PARAMS["mixcv"]
    ) | {COMPILE_MODE_KEY}
    unknown = sorted(name for name in profile if name not in supported)
    if unknown:
        return f"unknown compile profile option(s): {unknown}"

    if "num_stages" not in profile:
        return "compile profile must include num_stages"

    for name, value in profile.items():
        if name == COMPILE_MODE_KEY:
            continue
        if name == "num_stages":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return "compile profile num_stages must be a positive integer"
            continue
        try:
            validate_compile_option_values(name, [value])
        except ValueError as exc:
            return str(exc)

    if is_compile_on_910_95:
        unsupported = sorted(_MIXCV_910_95_UNSUPPORTED_PARAMS & set(profile))
        if unsupported:
            return f"compile profile uses unsupported 910_95 option(s): {unsupported}"

    num_stages = profile["num_stages"]
    if mode == COMPILE_MODE_VECTOR:
        return None

    multibuffer = profile.get("multibuffer", None)
    auto_mb_disabled = num_stages == 1 or multibuffer is False
    child_params = _MULTI_BUFFER_CHILD_PARAMS & set(profile)

    if mode == COMPILE_MODE_MB_OFF:
        if num_stages != 1:
            return "MB_OFF profile must use num_stages=1"
        if child_params:
            return f"MB_OFF profile must not include multi-buffer child option(s): {sorted(child_params)}"
        return None

    if auto_mb_disabled:
        return (
            "num_stages=1 or multibuffer=False disables auto multi-buffer; "
            "use MB_OFF without child options"
        )

    if num_stages != 2:
        return f"{mode} profile must use num_stages=2"
    if multibuffer is not True:
        return f"{mode} profile must include multibuffer=True"

    limit_local = profile.get("limit_auto_multi_buffer_only_for_local_buffer", None)
    if mode == COMPILE_MODE_LOCAL_MB:
        if limit_local is not True:
            return "LOCAL_MB profile must set limit_auto_multi_buffer_only_for_local_buffer=True"
        workspace_params = _WORKSPACE_CV_PARAMS & set(profile)
        if workspace_params:
            return f"LOCAL_MB profile must not include workspace option(s): {sorted(workspace_params)}"
        if "limit_auto_multi_buffer_of_local_buffer" not in profile:
            return (
                "LOCAL_MB profile must include limit_auto_multi_buffer_of_local_buffer"
            )
        return None

    if limit_local is not False:
        return "WORKSPACE_CV profile must set limit_auto_multi_buffer_only_for_local_buffer=False"
    if "set_workspace_multibuffer" not in profile:
        return "WORKSPACE_CV profile must include set_workspace_multibuffer"
    if not is_compile_on_910_95:
        for name in ("tile_mix_cube_loop", "tile_mix_vector_loop"):
            if name not in profile:
                return f"WORKSPACE_CV profile must include {name}"
    return None


def validate_compile_profile(
    profile: Mapping[str, Any], *, raise_on_error: bool = False
) -> bool:
    """Validate a generated search profile without repairing inactive fields."""
    error = _compile_profile_error(profile)
    if error is None:
        return True
    if raise_on_error:
        raise ValueError(error)
    return False


def apply_fixed_compile_options_to_profile(
    profile: Mapping[str, Any],
    fixed_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Overlay runtime-fixed compile options onto a generated profile."""
    fixed_options = fixed_options or {}
    copied = dict(profile)
    if not fixed_options:
        return copied

    mode = _profile_mode(copied)
    supported = (
        _SUPPORTED_PARAMS["vector"]
        if mode == COMPILE_MODE_VECTOR
        else _SUPPORTED_PARAMS["mixcv"]
    )
    unsupported = sorted(name for name in fixed_options if name not in supported)
    if unsupported:
        raise ValueError(
            f"fixed compile option(s) {unsupported} are not valid for profile mode {mode}"
        )
    copied.update(fixed_options)
    return copied


def effective_compile_profile_key(profile: Mapping[str, Any]) -> tuple:
    validate_compile_profile(profile, raise_on_error=True)
    effective = {
        key: value for key, value in profile.items() if key != COMPILE_MODE_KEY
    }
    return (
        _profile_mode(profile),
        tuple(
            sorted((key, _hashable_value(value)) for key, value in effective.items())
        ),
    )


def compile_profile_to_config(
    profile: Mapping[str, Any],
    *,
    shape_kwargs: Optional[Mapping[str, Any]] = None,
    base_config: Optional[Config] = None,
) -> Config:
    """Convert a validated profile to Triton Config and keep mode internal."""
    validate_compile_profile(profile, raise_on_error=True)
    kwargs = dict(base_config.kwargs) if base_config is not None else {}
    if shape_kwargs:
        kwargs.update(shape_kwargs)
    for name in _ALL_PARAMS:
        kwargs.pop(name, None)
    for name, value in profile.items():
        if name in {COMPILE_MODE_KEY, "num_stages"}:
            continue
        kwargs[name] = value

    return _make_config_compat(
        kwargs=kwargs,
        num_warps=getattr(base_config, "num_warps", 4),
        num_stages=profile["num_stages"],
        num_ctas=getattr(base_config, "num_ctas", 1),
        maxnreg=getattr(base_config, "maxnreg", None),
        pre_hook=getattr(base_config, "pre_hook", None),
        ir_override=getattr(base_config, "ir_override", None),
        num_buffers_warp_spec=getattr(base_config, "num_buffers_warp_spec", None),
        num_consumer_groups=getattr(base_config, "num_consumer_groups", None),
        reg_dec_producer=getattr(base_config, "reg_dec_producer", None),
        reg_inc_consumer=getattr(base_config, "reg_inc_consumer", None),
    )


def get_stage1_probe_profiles(profile_family: str = "mixcv") -> List[Dict[str, Any]]:
    if profile_family == "vector":
        profiles = DEFAULT_VECTOR_STAGE1_PROBE_PROFILES
    elif profile_family == "conservative_mixcv":
        profiles = CONSERVATIVE_MIXCV_STAGE1_PROBE_PROFILES
    else:
        profiles = DEFAULT_STAGE1_PROBE_PROFILES
    return [_drop_unsupported_profile_options(profile) for profile in profiles]


def get_stage1_probe_configs(
    shape_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    base_config: Optional[Config] = None,
    profile_family: str = "mixcv",
) -> List[Config]:
    return [
        compile_profile_to_config(
            profile, shape_kwargs=shape_kwargs, base_config=base_config
        )
        for profile in get_stage1_probe_profiles(profile_family)
    ]


def _shape_size(shape_kwargs: Optional[Mapping[str, Any]]) -> int:
    size = 1
    if not shape_kwargs:
        return size
    for value in shape_kwargs.values():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        size *= max(1, value)
    return size


def _is_large_shape(
    shape_kwargs: Optional[Mapping[str, Any]], large_tile: Optional[bool]
) -> bool:
    if large_tile is not None:
        return large_tile
    return _shape_size(shape_kwargs) >= 128 * 128


def _workspace_cv_profile(
    *,
    set_workspace_multibuffer: int,
    tile_mix_cube_loop: int,
    tile_mix_vector_loop: int,
    local_buffer_strategy: str = "no-l0c",
    enable_ubuf_saving: bool = True,
    unit_flag: bool = False,
) -> Dict[str, Any]:
    profile = {
        COMPILE_MODE_KEY: COMPILE_MODE_WORKSPACE_CV,
        "num_stages": 2,
        "multibuffer": True,
        "enable_tuning_mode": True,
        "enable_auto_bind_sub_block": True,
        "enable_hivm_auto_cv_balance": True,
        "enable_ubuf_saving": enable_ubuf_saving,
        "limit_auto_multi_buffer_only_for_local_buffer": False,
        "limit_auto_multi_buffer_of_local_buffer": local_buffer_strategy,
        "set_workspace_multibuffer": set_workspace_multibuffer,
        "tile_mix_cube_loop": tile_mix_cube_loop,
        "tile_mix_vector_loop": tile_mix_vector_loop,
        "unit_flag": unit_flag,
    }
    return _drop_unsupported_profile_options(profile)


def _local_mb_profile(
    *,
    local_buffer_strategy: str,
    enable_ubuf_saving: bool = True,
    unit_flag: bool = False,
) -> Dict[str, Any]:
    return {
        COMPILE_MODE_KEY: COMPILE_MODE_LOCAL_MB,
        "num_stages": 2,
        "multibuffer": True,
        "enable_tuning_mode": True,
        "enable_auto_bind_sub_block": True,
        "enable_hivm_auto_cv_balance": True,
        "enable_ubuf_saving": enable_ubuf_saving,
        "limit_auto_multi_buffer_only_for_local_buffer": True,
        "limit_auto_multi_buffer_of_local_buffer": local_buffer_strategy,
        "unit_flag": unit_flag,
    }


def _mb_off_profile(
    *, enable_ubuf_saving: bool = True, unit_flag: bool = False
) -> Dict[str, Any]:
    return {
        COMPILE_MODE_KEY: COMPILE_MODE_MB_OFF,
        "num_stages": 1,
        "enable_tuning_mode": True,
        "enable_auto_bind_sub_block": True,
        "enable_hivm_auto_cv_balance": True,
        "enable_ubuf_saving": enable_ubuf_saving,
        "unit_flag": unit_flag,
    }


def _append_valid_unique_profile(
    profiles: List[Dict[str, Any]],
    profile: Mapping[str, Any],
    seen: set,
    *,
    allow_unit_flag: bool,
) -> None:
    candidate = _drop_unsupported_profile_options(profile)
    if not allow_unit_flag and candidate.get("unit_flag") is True:
        return
    if not validate_compile_profile(candidate):
        return
    key = effective_compile_profile_key(candidate)
    if key in seen:
        return
    seen.add(key)
    profiles.append(candidate)


def _workspace_profile_from_probe(
    probe: Mapping[str, Any],
    *,
    set_workspace_multibuffer: int,
    tile_mix_cube_loop: int,
    tile_mix_vector_loop: int,
    local_buffer_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    if _profile_mode(probe) == COMPILE_MODE_WORKSPACE_CV:
        return _replace_profile(
            probe,
            set_workspace_multibuffer=set_workspace_multibuffer,
            tile_mix_cube_loop=tile_mix_cube_loop,
            tile_mix_vector_loop=tile_mix_vector_loop,
            limit_auto_multi_buffer_of_local_buffer=(
                local_buffer_strategy
                if local_buffer_strategy is not None
                else probe.get("limit_auto_multi_buffer_of_local_buffer", "no-l0c")
            ),
        )
    return _workspace_cv_profile(
        set_workspace_multibuffer=set_workspace_multibuffer,
        tile_mix_cube_loop=tile_mix_cube_loop,
        tile_mix_vector_loop=tile_mix_vector_loop,
        local_buffer_strategy=local_buffer_strategy or "no-l0c",
        enable_ubuf_saving=probe.get("enable_ubuf_saving", True),
        unit_flag=probe.get("unit_flag", False),
    )


def _tile_mix_probe_order(
    base_pair: tuple[int, int], *, stage1_ub_bytes: Optional[int]
) -> List[tuple[int, int]]:
    if stage1_ub_bytes is None:
        order = [(4, 4), (4, 2), (2, 2), (2, 4), (1, 1)]
    elif stage1_ub_bytes < LOCAL_ONLY_PROBE_UB_THRESHOLD_BYTES:
        order = [base_pair, (4, 2), (2, 2), (2, 4), (1, 1), (4, 4)]
    elif stage1_ub_bytes >= DEFAULT_UB_LIMIT_BYTES - TIGHT_UB_MARGIN_BYTES:
        order = [base_pair, (4, 4), (4, 2), (2, 4), (2, 2)]
    else:
        order = [base_pair, (4, 4), (4, 2), (2, 2), (2, 4), (1, 1)]

    result: List[tuple[int, int]] = []
    for pair in order:
        if pair not in result:
            result.append(pair)
    return result


def _workspace_probe_order(
    base_workspace: int, *, stage1_ub_bytes: Optional[int]
) -> List[int]:
    if stage1_ub_bytes is None:
        order = [base_workspace, 4, 2]
    elif stage1_ub_bytes < LOCAL_ONLY_PROBE_UB_THRESHOLD_BYTES:
        order = [4, base_workspace, 2]
    elif stage1_ub_bytes >= DEFAULT_UB_LIMIT_BYTES - TIGHT_UB_MARGIN_BYTES:
        order = [base_workspace, 2, 4]
    else:
        order = [base_workspace, 4, 2]

    result: List[int] = []
    for value in order:
        if value in (2, 4) and value not in result:
            result.append(value)
    return result


def _stage2_workspace_tile_profiles_from_probe(
    stage1_profile: Mapping[str, Any],
    *,
    stage1_ub_bytes: Optional[int],
) -> List[Dict[str, Any]]:
    mode = _profile_mode(stage1_profile)
    if mode == COMPILE_MODE_WORKSPACE_CV:
        base_workspace = stage1_profile.get("set_workspace_multibuffer", 2)
        base_pair = (
            stage1_profile.get("tile_mix_cube_loop", 4),
            stage1_profile.get("tile_mix_vector_loop", 4),
        )
        base_local_strategy = stage1_profile.get(
            "limit_auto_multi_buffer_of_local_buffer", "no-l0c"
        )
    else:
        base_workspace = 2
        base_pair = (4, 4)
        base_local_strategy = stage1_profile.get(
            "limit_auto_multi_buffer_of_local_buffer", "no-l0c"
        )

    workspaces = _workspace_probe_order(base_workspace, stage1_ub_bytes=stage1_ub_bytes)
    tile_pairs = _tile_mix_probe_order(base_pair, stage1_ub_bytes=stage1_ub_bytes)

    profiles: List[Dict[str, Any]] = []
    for workspace in workspaces:
        for cube, vector in tile_pairs:
            profiles.append(
                _workspace_profile_from_probe(
                    stage1_profile,
                    set_workspace_multibuffer=workspace,
                    tile_mix_cube_loop=cube,
                    tile_mix_vector_loop=vector,
                    local_buffer_strategy=base_local_strategy,
                )
            )

    # After workspace/tile_mix ranking, try the no-limit variant of the same
    # probe-derived tile order. If the Stage 1 probe was already no-limit this
    # only contributes non-duplicates.
    for workspace in workspaces:
        for cube, vector in tile_pairs:
            profiles.append(
                _workspace_profile_from_probe(
                    stage1_profile,
                    set_workspace_multibuffer=workspace,
                    tile_mix_cube_loop=cube,
                    tile_mix_vector_loop=vector,
                    local_buffer_strategy="no-limit",
                )
            )
    return profiles


def _make_vector_stage2_seed_profiles(
    stage1_profile: Mapping[str, Any],
    *,
    seed_budget: int,
) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    seen = set()
    base = {
        COMPILE_MODE_KEY: COMPILE_MODE_VECTOR,
        "num_stages": stage1_profile.get("num_stages", 2),
        "enable_ubuf_saving": stage1_profile.get("enable_ubuf_saving", True),
    }
    candidates = [
        base,
        {**base, "num_stages": 1 if base["num_stages"] == 2 else 2},
        {**base, "enable_ubuf_saving": not bool(base["enable_ubuf_saving"])},
        {
            **base,
            "num_stages": 1 if base["num_stages"] == 2 else 2,
            "enable_ubuf_saving": not bool(base["enable_ubuf_saving"]),
        },
    ]
    for profile in candidates:
        _append_valid_unique_profile(seeds, profile, seen, allow_unit_flag=False)
        if len(seeds) >= seed_budget:
            break
    return seeds


def make_stage2_seed_profiles(
    stage1_profile: Mapping[str, Any],
    *,
    shape_kwargs: Optional[Mapping[str, Any]] = None,
    large_tile: Optional[bool] = None,
    seed_budget: int = 8,
    allow_unit_flag: bool = False,
    stage1_ub_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build Stage 2 seeds from the Stage 1 winning probe.

    The first seed is always the exact Stage 1 probe. Subsequent workspace CV
    seeds change workspace/tile_mix around that probe, ordered by Stage 1 UB
    margin. Fixed branch seeds are fallback only.
    """
    if _profile_mode(stage1_profile) == COMPILE_MODE_VECTOR:
        return _make_vector_stage2_seed_profiles(
            stage1_profile, seed_budget=seed_budget
        )

    seeds: List[Dict[str, Any]] = []
    seen = set()
    _append_valid_unique_profile(
        seeds, stage1_profile, seen, allow_unit_flag=allow_unit_flag
    )
    if allow_unit_flag:
        _append_valid_unique_profile(
            seeds,
            _replace_profile(stage1_profile, unit_flag=True),
            seen,
            allow_unit_flag=allow_unit_flag,
        )

    workspace_tile_profiles = _stage2_workspace_tile_profiles_from_probe(
        stage1_profile, stage1_ub_bytes=stage1_ub_bytes
    )
    if (
        stage1_ub_bytes is not None
        and stage1_ub_bytes < LOCAL_ONLY_PROBE_UB_THRESHOLD_BYTES
    ):
        early_workspace_profile_count = min(4, max(1, seed_budget - len(seeds) - 2))
    else:
        early_workspace_profile_count = len(workspace_tile_profiles)

    for profile in workspace_tile_profiles[:early_workspace_profile_count]:
        _append_valid_unique_profile(
            seeds, profile, seen, allow_unit_flag=allow_unit_flag
        )

    local_strategies = []
    if (
        stage1_ub_bytes is not None
        and stage1_ub_bytes < LOCAL_ONLY_PROBE_UB_THRESHOLD_BYTES
    ):
        local_strategies.append("no-l0c")
    local_strategies.append("no-limit")
    for local_strategy in local_strategies:
        _append_valid_unique_profile(
            seeds,
            _local_mb_profile(
                local_buffer_strategy=local_strategy,
                enable_ubuf_saving=stage1_profile.get("enable_ubuf_saving", True),
            ),
            seen,
            allow_unit_flag=allow_unit_flag,
        )

    for profile in workspace_tile_profiles[early_workspace_profile_count:]:
        _append_valid_unique_profile(
            seeds, profile, seen, allow_unit_flag=allow_unit_flag
        )

    _append_valid_unique_profile(
        seeds, _mb_off_profile(), seen, allow_unit_flag=allow_unit_flag
    )
    return seeds[:seed_budget]


def _replace_profile(profile: Mapping[str, Any], **updates: Any) -> Dict[str, Any]:
    copied = dict(profile)
    copied.update(updates)
    return copied


def _next_tile_mix_up(value: int) -> Optional[int]:
    order = [1, 2, 4]
    if value not in order:
        return None
    index = order.index(value)
    return order[index + 1] if index + 1 < len(order) else None


def _next_tile_mix_down(value: int) -> Optional[int]:
    order = [1, 2, 4]
    if value not in order:
        return None
    index = order.index(value)
    return order[index - 1] if index > 0 else None


def _tile_pair_balance_candidates(
    profile: Mapping[str, Any], *, large_shape: bool
) -> List[Dict[str, Any]]:
    if _profile_mode(profile) != COMPILE_MODE_WORKSPACE_CV:
        return []
    if "tile_mix_cube_loop" not in profile or "tile_mix_vector_loop" not in profile:
        return []

    order = (
        [(4, 4), (4, 2), (2, 4), (2, 2), (1, 1)]
        if large_shape
        else [(2, 2), (4, 2), (2, 4), (4, 4), (1, 1)]
    )
    current = (profile["tile_mix_cube_loop"], profile["tile_mix_vector_loop"])
    if current not in order:
        return [
            _replace_profile(
                profile,
                tile_mix_cube_loop=order[0][0],
                tile_mix_vector_loop=order[0][1],
            )
        ]
    index = order.index(current)
    candidates = []
    for next_index in (index + 1, index - 1):
        if 0 <= next_index < len(order):
            cube, vector = order[next_index]
            candidates.append(
                _replace_profile(
                    profile,
                    tile_mix_cube_loop=cube,
                    tile_mix_vector_loop=vector,
                )
            )
    return candidates


def generate_linked_compile_neighbors(
    profile: Mapping[str, Any],
    *,
    shape_kwargs: Optional[Mapping[str, Any]] = None,
    large_tile: Optional[bool] = None,
    limit: Optional[int] = None,
    allow_unit_flag: bool = False,
    prefer_resource_relax: bool = False,
) -> List[Dict[str, Any]]:
    """Generate legal action-based neighbors for compile-option annealing."""
    validate_compile_profile(profile, raise_on_error=True)
    large_shape = _is_large_shape(shape_kwargs, large_tile)
    mode = _profile_mode(profile)
    candidates: List[Dict[str, Any]] = []
    if allow_unit_flag and mode != COMPILE_MODE_VECTOR:
        candidates.append(
            _replace_profile(
                profile,
                unit_flag=not bool(profile.get("unit_flag", False)),
            )
        )

    if mode == COMPILE_MODE_VECTOR:
        candidates.append(
            _replace_profile(
                profile,
                num_stages=1 if profile.get("num_stages") == 2 else 2,
            )
        )
        candidates.append(
            _replace_profile(
                profile,
                enable_ubuf_saving=not bool(profile.get("enable_ubuf_saving", True)),
            )
        )

    elif mode == COMPILE_MODE_WORKSPACE_CV:
        resource_relax = []
        if profile.get("set_workspace_multibuffer") == 4:
            resource_relax.append(
                _replace_profile(profile, set_workspace_multibuffer=2)
            )
        for name in ("tile_mix_vector_loop", "tile_mix_cube_loop"):
            if name in profile:
                next_value = _next_tile_mix_up(profile[name])
                if next_value is not None:
                    resource_relax.append(
                        _replace_profile(profile, **{name: next_value})
                    )
        if profile.get("limit_auto_multi_buffer_of_local_buffer") == "no-limit":
            resource_relax.append(
                _replace_profile(
                    profile, limit_auto_multi_buffer_of_local_buffer="no-l0c"
                )
            )
        if profile.get("enable_ubuf_saving") is False:
            resource_relax.append(_replace_profile(profile, enable_ubuf_saving=True))
        resource_relax.append(
            _local_mb_profile(
                local_buffer_strategy=profile.get(
                    "limit_auto_multi_buffer_of_local_buffer", "no-l0c"
                ),
                enable_ubuf_saving=profile.get("enable_ubuf_saving", True),
                unit_flag=profile.get("unit_flag", False),
            )
        )

        perf_push = []
        if profile.get("set_workspace_multibuffer") == 2:
            perf_push.append(_replace_profile(profile, set_workspace_multibuffer=4))
        if profile.get("limit_auto_multi_buffer_of_local_buffer") == "no-l0c":
            perf_push.append(
                _replace_profile(
                    profile, limit_auto_multi_buffer_of_local_buffer="no-limit"
                )
            )
        if profile.get("enable_ubuf_saving") is True:
            perf_push.append(_replace_profile(profile, enable_ubuf_saving=False))
        for name in ("tile_mix_vector_loop", "tile_mix_cube_loop"):
            if name in profile:
                next_value = _next_tile_mix_down(profile[name])
                if next_value is not None:
                    perf_push.append(_replace_profile(profile, **{name: next_value}))

        balance = _tile_pair_balance_candidates(profile, large_shape=large_shape)
        candidates.extend(resource_relax if prefer_resource_relax else perf_push)
        candidates.extend(balance)
        candidates.extend(perf_push if prefer_resource_relax else resource_relax)

    elif mode == COMPILE_MODE_LOCAL_MB:
        candidates.append(
            _workspace_cv_profile(
                set_workspace_multibuffer=2,
                tile_mix_cube_loop=4 if large_shape else 2,
                tile_mix_vector_loop=4 if large_shape else 2,
                local_buffer_strategy=profile.get(
                    "limit_auto_multi_buffer_of_local_buffer", "no-l0c"
                ),
                enable_ubuf_saving=profile.get("enable_ubuf_saving", True),
                unit_flag=profile.get("unit_flag", False),
            )
        )
        candidates.append(
            _mb_off_profile(
                enable_ubuf_saving=profile.get("enable_ubuf_saving", True),
                unit_flag=profile.get("unit_flag", False),
            )
        )
        if profile.get("limit_auto_multi_buffer_of_local_buffer") == "no-l0c":
            candidates.append(
                _replace_profile(
                    profile, limit_auto_multi_buffer_of_local_buffer="no-limit"
                )
            )
        else:
            candidates.append(
                _replace_profile(
                    profile, limit_auto_multi_buffer_of_local_buffer="no-l0c"
                )
            )

    else:
        candidates.append(
            _local_mb_profile(
                local_buffer_strategy="no-l0c",
                enable_ubuf_saving=profile.get("enable_ubuf_saving", True),
                unit_flag=profile.get("unit_flag", False),
            )
        )

    selected: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        _append_valid_unique_profile(
            selected, candidate, seen, allow_unit_flag=allow_unit_flag
        )
        if limit is not None and len(selected) >= limit:
            break
    return selected


_SPACE_OVERFLOW_RE = re.compile(r"\b([a-z][a-z0-9_]*)\s+overflow\b")
_SYNC_FAILURE_TOKENS = (
    "injectsync",
    "block-sync",
    "syncblock",
    "graphsyncsolver",
    "syncsolver",
    "barrier",
    "event",
    "set flag",
    "wait flag",
    "memory conflict",
)
_INTERNAL_FAILURE_TOKENS = (
    "internal error",
    "report_fatal_error",
    "llvm_unreachable",
    "assertion failed",
    "segmentation fault",
    "core dumped",
    "unhandled case",
    "unexpected op",
    "error in cv-pipelining",
    "postprocesscubefunc failed",
    "postprocessvectorfunc failed",
)
_WORKSPACE_FAILURE_TOKENS = (
    "workspace",
    "alloc_workspace",
    "allocworkspace",
    "failed to multibuffer",
    "multibuffer",
    "unknown buffer size",
    "alloc-like op",
    "cv-pipelining",
    "unable to pipeline",
    "cannot pipeline",
    "failed to pipelinine",
)


def _failure_text(error: Any) -> str:
    if error is None:
        return ""
    if isinstance(error, (list, tuple)):
        text = " ".join(_failure_text(item) for item in error)
    elif isinstance(error, BaseException):
        text = f"{type(error).__name__} {error}"
    else:
        text = str(error)
    return re.sub(r"\s+", " ", text.lower()).strip()


def classify_compile_failure(error: Any) -> str:
    text = _failure_text(error)
    match = _SPACE_OVERFLOW_RE.search(text)
    if match:
        space = match.group(1)
        if space in {"ub", "ubuf"}:
            return FAILURE_RESOURCE_UB
        if space in {"cbuf", "l1", "l1a", "l1b"}:
            return FAILURE_RESOURCE_L1
        if space in {"l0c", "cc"}:
            return FAILURE_RESOURCE_L0C
        if space in {"workspace", "gm"}:
            return FAILURE_RESOURCE_WORKSPACE
        return FAILURE_EXACT_ONLY
    if any(token in text for token in _SYNC_FAILURE_TOKENS):
        return FAILURE_SYNC_OR_CORRECTNESS
    if any(token in text for token in _INTERNAL_FAILURE_TOKENS):
        return FAILURE_COMPILER_INTERNAL
    if any(token in text for token in _WORKSPACE_FAILURE_TOKENS):
        return FAILURE_RESOURCE_WORKSPACE
    return FAILURE_EXACT_ONLY


def is_resource_failure(failure_kind: str) -> bool:
    return failure_kind in _RESOURCE_FAILURES


def _is_workspace_safest_tile_point(profile: Mapping[str, Any]) -> bool:
    return (
        _profile_mode(profile) == COMPILE_MODE_WORKSPACE_CV
        and profile.get("tile_mix_vector_loop") == 4
        and profile.get("tile_mix_cube_loop") == 4
        and profile.get("enable_ubuf_saving", True) is True
    )


def should_prune_compile_direction(
    profile: Mapping[str, Any], failure_kind: str
) -> bool:
    """Whether a failed profile can prune higher-resource neighbors.

    Resource overflow failures are always directional. For workspace CV, a
    failure at tile_mix=(4,4) with ubuf_saving=True is also treated as a
    directional failure because it is the lowest-UB point for that workspace /
    local-buffer strategy; smaller tile_mix values should not be tested first.
    """
    if is_resource_failure(failure_kind):
        return True
    return _is_workspace_safest_tile_point(profile)


def _pressure_rank_local_buffer(value: Any) -> int:
    return {"no-l0c": 0, "no-limit": 1}.get(value, 0)


def _pressure_rank_ubuf_saving(value: Any) -> int:
    return 0 if value is True else 1


def compile_profile_resource_not_less(
    candidate: Mapping[str, Any], failed_profile: Mapping[str, Any]
) -> bool:
    if _profile_mode(candidate) != _profile_mode(failed_profile):
        return False
    if _profile_mode(candidate) != COMPILE_MODE_WORKSPACE_CV:
        return False

    for name in ("tile_mix_vector_loop", "tile_mix_cube_loop"):
        if name in candidate and name in failed_profile:
            if candidate[name] > failed_profile[name]:
                return False
    if (
        "set_workspace_multibuffer" in candidate
        and "set_workspace_multibuffer" in failed_profile
        and candidate["set_workspace_multibuffer"]
        < failed_profile["set_workspace_multibuffer"]
    ):
        return False
    if _pressure_rank_local_buffer(
        candidate.get("limit_auto_multi_buffer_of_local_buffer", "no-l0c")
    ) < _pressure_rank_local_buffer(
        failed_profile.get("limit_auto_multi_buffer_of_local_buffer", "no-l0c")
    ):
        return False
    if _pressure_rank_ubuf_saving(
        candidate.get("enable_ubuf_saving", True)
    ) < _pressure_rank_ubuf_saving(failed_profile.get("enable_ubuf_saving", True)):
        return False
    return True


@dataclass
class CompileFailureRegionSet:
    exact_keys: set = field(default_factory=set)
    resource_failures: List[Dict[str, Any]] = field(default_factory=list)
    forbidden_modes: set = field(default_factory=set)

    def add(self, profile: Mapping[str, Any], failure: Any) -> str:
        failure_kind = (
            failure
            if isinstance(failure, str) and failure in _ALL_FAILURE_KINDS
            else classify_compile_failure(failure)
        )
        self.exact_keys.add(effective_compile_profile_key(profile))
        mode = _profile_mode(profile)
        if should_prune_compile_direction(profile, failure_kind):
            if mode == COMPILE_MODE_WORKSPACE_CV:
                self.resource_failures.append(dict(profile))
            elif mode == COMPILE_MODE_LOCAL_MB:
                self.forbidden_modes.add(COMPILE_MODE_LOCAL_MB)
        elif mode == COMPILE_MODE_LOCAL_MB:
            self.forbidden_modes.add(COMPILE_MODE_LOCAL_MB)
        return failure_kind

    def is_forbidden(self, profile: Mapping[str, Any]) -> bool:
        key = effective_compile_profile_key(profile)
        if key in self.exact_keys:
            return True
        if _profile_mode(profile) in self.forbidden_modes:
            return True
        return any(
            compile_profile_resource_not_less(profile, failed)
            for failed in self.resource_failures
        )


def _value_space_for_config(
    config: Config, spec: CompileOptionsSpec, *, generated_tiling: bool
) -> Dict[str, List[Any]]:
    supported = _SUPPORTED_PARAMS[spec.kernel_type]
    preset = _AUTO_SEARCH_PRESETS[spec.kernel_type]

    value_space = {}
    for name in sorted(supported):
        if name in spec.params:
            values = spec.params[name]
        elif name not in preset:
            continue
        else:
            values = preset[name]
        validate_compile_option_values(name, values)
        value_space[name] = values
    if spec.kernel_type == "mixcv" and is_compile_on_910_95:
        for name in _MIXCV_910_95_UNSUPPORTED_PARAMS:
            value_space.pop(name, None)
    return value_space


def _is_inactive_reason(
    name: str,
    num_stages: int,
    combo: Dict[str, Any],
    config: Config,
    fixed_options: Dict[str, Any],
) -> Optional[str]:
    if not _is_param_effective(name, num_stages, combo, config, fixed_options):
        if name in {
            "limit_auto_multi_buffer_only_for_local_buffer",
            "limit_auto_multi_buffer_of_local_buffer",
            "set_workspace_multibuffer",
            "tile_mix_vector_loop",
            "tile_mix_cube_loop",
        }:
            return "depends on auto multi-buffer"
    return None


def _is_param_effective(
    name: str,
    num_stages: int,
    combo: Dict[str, Any],
    config: Config,
    fixed_options: Dict[str, Any],
) -> bool:
    if name not in _MIXCV_OPTION_ACTIVITY_RULES:
        return True
    return _MIXCV_OPTION_ACTIVITY_RULES[name](num_stages, combo, config, fixed_options)


def _resolve_compile_option(
    name: str,
    combo: Dict[str, Any],
    config: Config,
    fixed_options: Dict[str, Any],
    default: Any,
) -> Any:
    if name in combo:
        return combo[name]
    if name in fixed_options:
        return fixed_options[name]
    return default


def _effective_values(
    name: str,
    value_space: Dict[str, List[Any]],
    fixed_options: Dict[str, Any],
    default: Any,
) -> List[Any]:
    if name in fixed_options:
        return [fixed_options[name]]
    if name in value_space:
        return value_space[name]
    return [default]


def _emit_values(
    names: List[str],
    value_space: Dict[str, List[Any]],
    fixed_options: Dict[str, Any],
) -> List[tuple[str, List[Any]]]:
    return [
        (name, value_space[name])
        for name in names
        if name in value_space and name not in fixed_options
    ]


def _product_dict(items: List[tuple[str, List[Any]]]):
    if not items:
        yield {}
        return
    names = [name for name, _ in items]
    values = [values for _, values in items]
    for combo in itertools.product(*values):
        yield dict(zip(names, combo))


def _mixcv_branch_items(
    *,
    num_stages: int,
    value_space: Dict[str, List[Any]],
    fixed_options: Dict[str, Any],
) -> List[List[tuple[str, List[Any]]]]:
    independent_names = [
        "enable_tuning_mode",
        "unit_flag",
        "enable_ubuf_saving",
        "enable_hivm_auto_cv_balance",
        "enable_auto_bind_sub_block",
    ]
    independent_items = _emit_values(independent_names, value_space, fixed_options)

    multibuffer_values = _effective_values(
        "multibuffer", value_space, fixed_options, None
    )
    branches = []
    for multibuffer in multibuffer_values:
        multibuffer_items = []
        if "multibuffer" in value_space and "multibuffer" not in fixed_options:
            multibuffer_items = [("multibuffer", [multibuffer])]
        branch_base = independent_items + multibuffer_items

        if num_stages == 1 or multibuffer is False:
            branches.append(branch_base)
            continue

        limit_only_values = _effective_values(
            "limit_auto_multi_buffer_only_for_local_buffer",
            value_space,
            fixed_options,
            True,
        )
        for limit_only in limit_only_values:
            limit_only_items = []
            if (
                "limit_auto_multi_buffer_only_for_local_buffer" in value_space
                and "limit_auto_multi_buffer_only_for_local_buffer" not in fixed_options
            ):
                limit_only_items = [
                    ("limit_auto_multi_buffer_only_for_local_buffer", [limit_only])
                ]
            if limit_only is True:
                branch_names = ["limit_auto_multi_buffer_of_local_buffer"]
            else:
                branch_names = [
                    "limit_auto_multi_buffer_of_local_buffer",
                    "set_workspace_multibuffer",
                    "tile_mix_vector_loop",
                    "tile_mix_cube_loop",
                ]
            branches.append(
                branch_base
                + limit_only_items
                + _emit_values(branch_names, value_space, fixed_options)
            )

    return branches


def _make_expanded_config(
    config: Config,
    spec: CompileOptionsSpec,
    combo_value: Dict[str, Any],
    num_stages: int,
) -> Config:
    new_kwargs = dict(config.kwargs)
    for name in _ALL_PARAMS:
        new_kwargs.pop(name, None)
    for name, value in combo_value.items():
        new_kwargs[name] = value

    return _make_config_compat(
        kwargs=new_kwargs,
        num_warps=getattr(config, "num_warps", 4),
        num_stages=num_stages,
        num_ctas=getattr(config, "num_ctas", 1),
        maxnreg=getattr(config, "maxnreg", None),
        pre_hook=getattr(config, "pre_hook", None),
        ir_override=getattr(config, "ir_override", None),
        num_buffers_warp_spec=getattr(config, "num_buffers_warp_spec", None),
        num_consumer_groups=getattr(config, "num_consumer_groups", None),
        reg_dec_producer=getattr(config, "reg_dec_producer", None),
        reg_inc_consumer=getattr(config, "reg_inc_consumer", None),
    )


def _hashable_value(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_value(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_hashable_value(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _config_key(config: Config) -> tuple:
    return (
        tuple(
            sorted(
                (key, _hashable_value(value)) for key, value in config.kwargs.items()
            )
        ),
        getattr(config, "num_warps", 4),
        getattr(config, "num_stages", None),
        getattr(config, "num_ctas", 1),
        getattr(config, "maxnreg", None),
        id(getattr(config, "pre_hook", None)),
        _hashable_value(getattr(config, "ir_override", None)),
        getattr(config, "num_buffers_warp_spec", None),
        getattr(config, "num_consumer_groups", None),
        getattr(config, "reg_dec_producer", None),
        getattr(config, "reg_inc_consumer", None),
    )


def expand_compile_option_configs(
    configs: List[Config],
    spec: CompileOptionsSpec,
    *,
    generated_tiling: bool,
    fixed_options: Optional[Dict[str, Any]] = None,
) -> List[Config]:
    if not spec.enabled or not configs:
        return configs

    fixed_options = fixed_options or {}
    expanded_configs = []
    emitted_config_keys = set()
    for config in configs:
        value_space = _value_space_for_config(
            config, spec, generated_tiling=generated_tiling
        )
        if "num_stages" in fixed_options:
            num_stage_values = [fixed_options["num_stages"]]
            value_space.pop("num_stages", None)
        else:
            num_stage_values = value_space.pop("num_stages")

        for name in fixed_options:
            if name != "num_stages":
                value_space.pop(name, None)

        for num_stages in num_stage_values:
            if spec.kernel_type == "mixcv":
                branch_items = _mixcv_branch_items(
                    num_stages=num_stages,
                    value_space=value_space,
                    fixed_options=fixed_options,
                )
                combo_iter = itertools.chain.from_iterable(
                    _product_dict(items) for items in branch_items
                )
            else:
                combo_iter = _product_dict(list(value_space.items()))

            for combo_value in combo_iter:
                new_config = _make_expanded_config(
                    config, spec, combo_value, num_stages
                )
                config_key = _config_key(new_config)
                if config_key in emitted_config_keys:
                    continue
                emitted_config_keys.add(config_key)

                if (
                    spec.max_configs is not None
                    and len(expanded_configs) >= spec.max_configs
                ):
                    raise ValueError(
                        "compile_options generated more than "
                        f"{spec.max_configs} configs. Narrow the search space or raise max_configs."
                    )
                expanded_configs.append(new_config)

    return expanded_configs


def get_compile_option_param_names(spec: CompileOptionsSpec) -> set[str]:
    if not spec.enabled:
        return set()
    return set(_SUPPORTED_PARAMS[spec.kernel_type])


def format_compile_option_result(
    config: Config,
    spec: CompileOptionsSpec,
    fixed_options: Optional[Dict[str, Any]] = None,
) -> str:
    if not spec.enabled:
        return str(config)

    fixed_options = fixed_options or {}
    compile_param_names = _SUPPORTED_PARAMS[spec.kernel_type] - {"num_stages"}
    selected_meta = {
        key: value
        for key, value in sorted(config.kwargs.items())
        if key not in compile_param_names
    }
    selected_meta["num_stages"] = getattr(config, "num_stages", None)

    effective = {
        key: value
        for key, value in sorted(config.kwargs.items())
        if key in compile_param_names
    }
    effective.update(
        {
            key: value
            for key, value in sorted(fixed_options.items())
            if key in compile_param_names
        }
    )

    if spec.kernel_type == "mixcv":
        num_stages = selected_meta["num_stages"]
        multibuffer = effective.get("multibuffer", None)
        effective["enable_auto_multi_buffer"] = (
            False if multibuffer is False or num_stages == 1 else True
        )
        for name in _SUPPORTED_PARAMS[spec.kernel_type]:
            if name == "num_stages":
                continue
            reason = _is_inactive_reason(
                name, num_stages, config.kwargs, config, fixed_options
            )
            if reason is not None and name not in effective:
                effective[name] = f"<inactive: {reason}>"
            if reason is not None and name in effective:
                effective[name] = f"<inactive: {reason}>"
        if is_compile_on_910_95:
            for name in _MIXCV_910_95_UNSUPPORTED_PARAMS:
                effective[name] = "<unsupported: compile_on_910_95>"

    selected_items = [f"{key}={value}" for key, value in selected_meta.items()]
    effective_items = [f"{key}={value}" for key, value in sorted(effective.items())]
    return (
        "selected_meta: "
        + ", ".join(selected_items)
        + "; effective_compile_options: "
        + ", ".join(effective_items)
    )


def summarize_compile_option_configs(
    configs: List[Config], limit: Optional[int] = None
) -> List[str]:
    summary = []
    selected_configs = configs if limit is None else configs[:limit]
    for config in selected_configs:
        items = [f"{key}={value}" for key, value in sorted(config.kwargs.items())]
        items.append(f"num_stages={getattr(config, 'num_stages', None)}")
        summary.append(", ".join(items))
    return summary
