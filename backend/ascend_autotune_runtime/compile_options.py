from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        "enable_tuning_mode",
        "num_stages",
        "enable_ubuf_saving",
    },
}

_ALL_PARAMS = set().union(*_SUPPORTED_PARAMS.values())

_MIXCV_910_95_UNSUPPORTED_PARAMS = {
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
}

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
        "enable_tuning_mode": [True],
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
    if not _is_mixcv_multi_buffer_auto_enabled(num_stages, combo, config, fixed_options):
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
    kernel_type = _normalize_kernel_type(raw.pop("kernel_type", raw.pop("type", "mixcv")))
    max_configs = raw.pop("max_configs", DEFAULT_MAX_CONFIGS)
    if max_configs is not None and (not isinstance(max_configs, int) or max_configs <= 0):
        raise ValueError("compile_options max_configs must be a positive integer or None")

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
            print(
                f"[WARNING] compile_options parameter '{name}' is not supported "
                f"for kernel_type '{kernel_type}' and will be ignored."
            )
            continue
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
    return Config(**{
        key: value
        for key, value in kwargs.items()
        if key in supported_config_args
    })


def _value_space_for_config(config: Config, spec: CompileOptionsSpec, *, generated_tiling: bool) -> Dict[str, List[Any]]:
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


def _is_inactive_reason(name: str, num_stages: int, combo: Dict[str, Any], config: Config, fixed_options: Dict[str, Any]) -> Optional[str]:
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

    multibuffer_values = _effective_values("multibuffer", value_space, fixed_options, None)
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


def _make_expanded_config(config: Config, spec: CompileOptionsSpec, combo_value: Dict[str, Any], num_stages: int) -> Config:
    new_kwargs = dict(config.kwargs)
    for name in _SUPPORTED_PARAMS[spec.kernel_type]:
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
        tuple(sorted((key, _hashable_value(value)) for key, value in config.kwargs.items())),
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
        value_space = _value_space_for_config(config, spec, generated_tiling=generated_tiling)
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
                new_config = _make_expanded_config(config, spec, combo_value, num_stages)
                config_key = _config_key(new_config)
                if config_key in emitted_config_keys:
                    continue
                emitted_config_keys.add(config_key)

                if spec.max_configs is not None and len(expanded_configs) >= spec.max_configs:
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
    effective.update({
        key: value
        for key, value in sorted(fixed_options.items())
        if key in compile_param_names
    })

    if spec.kernel_type == "mixcv":
        num_stages = selected_meta["num_stages"]
        multibuffer = effective.get("multibuffer", None)
        effective["enable_auto_multi_buffer"] = (
            False if multibuffer is False or num_stages == 1 else True
        )
        for name in _SUPPORTED_PARAMS[spec.kernel_type]:
            if name == "num_stages":
                continue
            reason = _is_inactive_reason(name, num_stages, config.kwargs, config, fixed_options)
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
        "selected_meta: " + ", ".join(selected_items) +
        "; effective_compile_options: " + ", ".join(effective_items)
    )


def summarize_compile_option_configs(configs: List[Config], limit: Optional[int] = None) -> List[str]:
    summary = []
    selected_configs = configs if limit is None else configs[:limit]
    for config in selected_configs:
        items = [f"{key}={value}" for key, value in sorted(config.kwargs.items())]
        items.append(f"num_stages={getattr(config, 'num_stages', None)}")
        summary.append(", ".join(items))
    return summary
