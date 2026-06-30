# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import builtins
import copy
import functools
import ast
import inspect
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import itertools
from typing import Dict, List

from torch import Tensor

import triton
from triton.runtime.autotuner import Autotuner, Config
from triton.backends.dicp_triton.utils import is_compile_on_910_95

from .kernel_ast_analyzer import (
    DotCallParser,
    LowDimsAxesParser,
    ReductionAxesParser,
    SplitAxesParser,
    TilingAxesParser,
)
from .schedule_profiles import (
    apply_fixed_compile_options_to_profile,
    classify_compile_failure,
    compile_profile_to_config,
    effective_compile_profile_key,
    get_stage1_probe_profiles,
    expand_compile_option_configs,
    format_compile_option_result,
    get_compile_option_param_names,
    _hashable_value,
    parse_compile_options_hint,
    summarize_compile_option_configs,
)
from .kernel_archetype import (
    OperatorKind,
    SearchPolicy,
    analyze_operator_policy,
    make_search_policy,
)
from .tile_search_policy import (
    apply_no_dot_search_defaults,
    filter_shapes_by_ub_and_timing,
    parse_search_params_hint,
    propose_evolved_shape_configs,
    select_initial_percentile_shapes,
    select_top_shape_entries,
    stage1_children_per_round,
    stage1_initial_budget,
    stage1_initial_percentiles,
    stage1_total_budget,
    ub_timing_weighted_key,
)
from .tile_acquisition_model import resource_overflow_ratio
from .measurement_cache import SearchMeasureCache
from .tile_shape_space import (
    effective_search_value_map,
    expand_search_param_shapes,
    extract_shape,
    shape_key,
)
from .schedule_profile_search import Stage2CompileSearcher
from .measurement_strategy import select_benchmark_strategy, ub_bytes_of
from .utils import get_byte_per_numel, is_valid_axis_name, valid_axis_names


def _make_config_compat(**kwargs):
    supported_config_args = inspect.signature(Config).parameters
    return Config(
        **{key: value for key, value in kwargs.items() if key in supported_config_args}
    )


def _empty_npu_cache_after_failure():
    try:
        import torch

        torch.npu.empty_cache()
    except Exception:
        pass


def _unwrap_parse_target(fn):
    seen = set()
    cur = fn
    while cur is not None and id(cur) not in seen:
        if callable(getattr(cur, "parse", None)):
            return cur

        jit_function = getattr(cur, "jit_function", None)
        if callable(getattr(jit_function, "parse", None)):
            return jit_function

        seen.add(id(cur))
        cur = getattr(cur, "fn", None)
    return fn


class AutoTilingTuner(Autotuner):
    """
    Automatic generateing candidate tiling configs and evaluating their performance to get the best config.
    """

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        auto_profile_dir=None,
        hints=None,
    ):
        """
        :param key: a list of argument name, where the change of arguments in value will triger re-generating candidates configs and evaluating.
            The parameters in the list will be assigned axis names in sequence, with the axis name being in
            {'x','y','z','w','v','t','rx','ry','rz','rw','rv','rt}, where the prefix 'r' means a reduction axis.
            Only the axis name in this param should add perfix 'r' if it's a reduction axis.
        :type key: List[str]
        """
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
            False,
        )
        self.user_defined_do_bench = do_bench is not None
        if not hints:
            self.hints = {}
        else:
            self.hints = dict(hints)
        self.ast_fn = _unwrap_parse_target(self.fn)
        split_params = self.hints.get("split_params", None)
        tiling_params = self.hints.get("tiling_params", None)
        low_dim_axes = self.hints.get("low_dim_axes", None)
        reduction_axes = self.hints.get("reduction_axes", None)
        self._init_axis_params(
            key,
            split_params,
            tiling_params,
            low_dim_axes,
            reduction_axes,
        )

        self.auto_gen_config = not configs or self.hints.get("auto_gen_config", False)
        self.operator_features = None
        self.operator_policy = self._infer_operator_policy()
        self._infer_compile_options_hint_if_needed()
        self.compile_options = parse_compile_options_hint(
            self.hints.get("compile_options", None)
        )
        self.search_params = parse_search_params_hint(
            self.hints.get("search_params", None)
        )
        if self.operator_policy.no_dot_shape_values:
            apply_no_dot_search_defaults(self.search_params)
        self.gen_configs = []  # generated configs from TileGenerator
        self.auto_profile_dir = auto_profile_dir
        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.is_simt_mode = False
        self.simt_stack_limit = 8192
        self.user_specified_warps = None
        self.user_specified_multibuffer = None
        self.default_multibuffer = not is_compile_on_910_95
        self.print_autotuning = os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1"
        self.print_autotuning_timings = (
            os.getenv("TRITON_PRINT_AUTOTUNING_TIMINGS", None) == "1"
        )
        self.fixed_compile_options = {}
        self._search_params_runtime_enabled = False
        self._search_param_anchor_shape_keys = set()
        self._search_ub_cache = {}
        self._search_measure_cache = SearchMeasureCache(
            self.search_params.params, self._search_ub_cache
        )
        self._search_stage1_measured_summary = []
        self._search_stage1_failed_summary = []
        self._search_bench_call_count = 0
        self._search_bench_config_count = 0
        self.last_search_stats = {}
        # AsyncCompileMode is available for triton.runtime.JITFunction, but not
        # for wrappers such as LibEntry.
        self._can_async_compile = isinstance(self.fn, triton.runtime.JITFunction)
        # General autotune keeps the historical env switch. Stage 1 search_params
        # probe batches force async compile independently via _batch_bench(...,
        # force_parallel=True).
        self.compile_parallel = (
            self._can_async_compile
            and os.getenv("TRITON_AUTOTUNE_PARALLEL_COMPILE", "1") == "1"
        )

    def _infer_compile_options_hint_if_needed(self):
        if "compile_options" in self.hints:
            return
        self.hints["compile_options"] = self.operator_policy.compile_options_kernel_type

    def _use_search_params_runtime(self) -> bool:
        return (
            self.search_params.enabled
            and self.compile_options.enabled
            and self.compile_options.kernel_type
            == self.operator_policy.compile_options_kernel_type
        )

    def _use_search_params_mixcv(self) -> bool:
        return (
            self._use_search_params_runtime()
            and self.compile_options.kernel_type == "mixcv"
        )

    def _apply_fixed_compile_profile(self, profile):
        return apply_fixed_compile_options_to_profile(
            profile, getattr(self, "fixed_compile_options", {})
        )

    def _apply_fixed_compile_profiles(self, profiles):
        deduped = []
        seen = set()
        for profile in profiles:
            fixed_profile = self._apply_fixed_compile_profile(profile)
            key = effective_compile_profile_key(fixed_profile)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fixed_profile)
        return deduped

    def _infer_operator_policy(self) -> SearchPolicy:
        try:
            features, policy = analyze_operator_policy(
                self._parse_ast(), self._get_capture_scope(), {id(self.ast_fn)}
            )
            self.operator_features = features
            return policy
        except Exception as exc:  # noqa: BLE001 - autotune must fail closed.
            self.operator_features = None
            return make_search_policy(OperatorKind.UNKNOWN)

    def _parse_ast(self):
        parse = getattr(self.ast_fn, "parse", None)
        if not callable(parse):
            raise ValueError(
                "Cannot parse Ascend autotune kernel AST. "
                "Please pass explicit autotune hints for wrapped kernels."
            )
        return parse()

    def _get_capture_scope(self):
        get_capture_scope = getattr(self.ast_fn, "get_capture_scope", None)
        if callable(get_capture_scope):
            return get_capture_scope()
        return {}

    def _autoparse_has_dot(self) -> bool:
        try:
            func_ast = self._parse_ast()
            return DotCallParser(
                func_ast, self._get_capture_scope(), {id(self.ast_fn)}
            ).parse()
        except Exception as e:
            raise ValueError(
                "Cannot infer Ascend compile_options from kernel AST. "
                "Please pass hints={'compile_options': 'mixcv'} or "
                "hints={'compile_options': 'vector'} explicitly."
            ) from e

    def _expand_simt_num_warps_configs(
        self, base_configs: List[Config]
    ) -> List[Config]:
        _default_cand_num_warps = [8, 16, 32, 64]
        cand_num_warps = (
            _default_cand_num_warps
            if self.user_specified_warps is None
            else [self.user_specified_warps]
        )

        simt_configs = []
        for base_cfg in base_configs:
            for num_warps in cand_num_warps:
                new_cfg = copy.deepcopy(base_cfg)
                new_cfg.num_warps = num_warps
                simt_configs.append(new_cfg)

        if self.print_autotuning:
            print(
                f"Triton autotuning: Expanded to {len(simt_configs)} SIMT configs (with warps: {cand_num_warps})"
            )
        return simt_configs

    def _expand_simd_multibuffer_configs(
        self, base_configs: List[Config]
    ) -> List[Config]:
        if self.user_specified_multibuffer is not None:
            if self.print_autotuning:
                print(
                    "Triton autotuning: Skip SIMD multibuffer expansion because user "
                    f"specified multibuffer={self.user_specified_multibuffer}"
                )
            return base_configs

        opposite_default_multibuffer = not self.default_multibuffer
        simd_configs = []
        for base_cfg in base_configs:
            simd_configs.append(base_cfg)
            new_cfg = copy.deepcopy(base_cfg)
            new_cfg.kwargs["multibuffer"] = opposite_default_multibuffer
            simd_configs.append(new_cfg)

        if self.print_autotuning:
            print(
                "Triton autotuning: Expanded to "
                f"{len(simd_configs)} SIMD configs (toggle multibuffer={opposite_default_multibuffer})"
            )
        return simd_configs

    def _init_axis_params(
        self, key, split_params, tiling_params, low_dim_axes, reduction_axes
    ):
        if isinstance(key, list):
            if split_params or tiling_params or low_dim_axes or reduction_axes:
                raise ValueError(
                    "If any axis-related parameters (split_params, tiling_params, low_dim_axes, reduction_axes)"
                    " are provided, 'key' must be a dict, not a list."
                )
            if len(key) > len(valid_axis_names):
                raise ValueError(
                    "Number of parameters exceeds the number of available axes."
                )
            self.keys = {axis: param for axis, param in zip(valid_axis_names, key)}
        elif isinstance(key, dict):
            if not set(key.keys()).issubset(set(valid_axis_names)):
                raise ValueError(
                    "All keys in 'key' must be valid axis names. Got unexpected keys."
                )
            self.keys = key
            if any([split_params, tiling_params, low_dim_axes, reduction_axes]) is None:
                raise ValueError(
                    "If 'key' is a dict, all axis-related parameters (split_params, tiling_params, low_dim_axes,"
                    " reduction_axes) must be provided."
                )
            if not isinstance(split_params, dict):
                raise ValueError(
                    "split_params must be a dict, got: {}".format(type(split_params))
                )
            if not isinstance(tiling_params, dict):
                raise ValueError(
                    "tiling_params must be a dict, got: {}".format(type(tiling_params))
                )
            if not isinstance(low_dim_axes, list):
                raise ValueError(
                    "low_dim_axes must be a list, got: {}".format(type(low_dim_axes))
                )
            if not isinstance(reduction_axes, list):
                raise ValueError(
                    "reduction_axes must be a list, got: {}".format(
                        type(reduction_axes)
                    )
                )

            used_axes = set(split_params.keys()).union(
                tiling_params.keys(),
                low_dim_axes,
                reduction_axes,
            )
            if not used_axes.issubset(self.keys.keys()):
                raise ValueError(
                    "The following axes are used but not present in the 'key': {}".format(
                        used_axes - set(self.keys.keys())
                    )
                )

        self.split_params = split_params
        self.all_split_params = {}
        self.fixed_split_params = {}
        self.tiling_params = tiling_params
        self.low_dim_axes = low_dim_axes
        self.reduction_axes = reduction_axes
        self.fixed_grid_dims = set()
        self.fixed_grid_dim_values = {}
        self.split_axis_pid_dims = {}
        self.axis_pid_dims = {}
        self.dual_reduction = False
        self.persistent_reduction = False
        self.num_buffers = -1

    def _autoparse_axis_params(self, all_args):
        miss_params = [arg for arg in self.arg_names if arg not in all_args.keys()]
        search_param_names = set()
        search_params = getattr(self, "search_params", None)
        if search_params is not None and getattr(search_params, "enabled", False):
            search_param_names = set(search_params.params)
        # parse pointer params nums
        if self.num_buffers == -1:
            self.num_buffers = self._autoparse_ptr_nums(all_args)

        # parse autotiling axes
        # reduction axis must be parsed before other axes. it will alter the key
        if not self.reduction_axes:
            self.reduction_axes = self._autoparse_reduction_axes()
        if len(self.reduction_axes) >= 2:
            self.dual_reduction = True

        if not self.low_dim_axes:
            self.low_dim_axes = self._autoparse_low_dim_axes()

        if len(self.reduction_axes) == 1:
            reduction_axis = self.reduction_axes[0]
            reduction_param = self.keys.get(reduction_axis, None)
            reduction_numel = all_args.get(reduction_param, float("inf"))
            persistent_threshold = self._get_persistent_reduction_threshold(
                reduction_axis
            )
            if reduction_numel <= persistent_threshold:
                self.persistent_reduction = True

        if not self.split_params:
            all_split_params = self._autoparse_split_params(
                self._get_constexpr_candidates()
            )
            self.all_split_params = dict(all_split_params)
            self.fixed_split_params = {}
            self.fixed_grid_dim_values = self._get_fixed_grid_dim_values(
                all_args.get("grid", None),
                all_args,
            )
            self.fixed_grid_dims = set(self.fixed_grid_dim_values.keys())

            fixed_grid_axes = {
                axis
                for axis, pid_dim in self.axis_pid_dims.items()
                if pid_dim in self.fixed_grid_dims
            }

            # Only missing constexpr params are tunable, and fixed-grid axes
            # should not be tuned on split.
            self.split_params = {
                axis: param
                for axis, param in all_split_params.items()
                if param in miss_params and axis not in fixed_grid_axes
            }

            # Fixed split is inferred only from fixed grid dims.
            for axis, pid_dim in self.axis_pid_dims.items():
                if pid_dim not in self.fixed_grid_dims:
                    continue
                core_num = self.fixed_grid_dim_values.get(pid_dim, 0)
                axis_len_name = self.keys.get(axis, None)
                axis_len = all_args.get(axis_len_name, None)
                if not isinstance(core_num, int) or core_num <= 0:
                    continue
                if not isinstance(axis_len, int) or axis_len <= 0:
                    continue

                self.fixed_split_params[axis] = (axis_len + core_num - 1) // core_num
        elif not self.axis_pid_dims:
            # When split axes are provided by hints, parse axis->program_id mapping
            # independently for fixed-grid semantics and diagnostics.
            self._autoparse_axis_pid_dims()
        miss_params = [
            arg for arg in miss_params if arg not in self.split_params.values()
        ]
        if not self.tiling_params:
            self.tiling_params = self._autoparse_tiling_params(miss_params)
        miss_params = [
            arg for arg in miss_params if arg not in self.tiling_params.values()
        ]
        miss_params = [arg for arg in miss_params if arg not in search_param_names]
        if miss_params:
            raise ValueError(
                f"Missing required arguments: {miss_params}. "
                f"These arguments must be explicitly provided and cannot be automatically tuned. "
                f"Please ensure that these arguments are passed when calling the function."
            )

    def _gen_tile_configs(
        self, kv_dict: Dict[str, int], dtype: torch.dtype
    ) -> List[Config]:
        from .tile_candidate_generator import KernelMeta, TileGenerator

        axis_sizes = {}
        for k, v in kv_dict.items():
            if not is_valid_axis_name(k):
                continue
            if not isinstance(v, int):
                raise ValueError(
                    f"Not supported dim type: {type(v)}, `int` is the only supported type"
                )
            axis_sizes[k] = v

        kernel_meta = KernelMeta(
            axis_sizes,
            self.split_params,
            self.fixed_split_params,
            self.tiling_params,
            self.low_dim_axes,
            dtype,
            self.persistent_reduction,
            self.dual_reduction,
            self.num_buffers,
            self.is_simt_mode,
        )
        tile_gen = TileGenerator(kernel_meta=kernel_meta)
        tile_gen.descend_split_tiling()

        self.gen_configs.clear()
        self.gen_configs = tile_gen.configs

        if self.is_simt_mode:
            self.gen_configs = self._expand_simt_num_warps_configs(self.gen_configs)
        else:
            self.gen_configs = self._expand_simd_multibuffer_configs(self.gen_configs)

        if len(self.gen_configs) == 0:
            print(
                "[WARNING] The generated candidate tiling configs are empty based on provided parameters!"
            )

        if self.print_autotuning:
            print("Generated configs number: {}".format(len(self.gen_configs)))

    def generate_key_and_configs(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        self.is_simt_mode = kwargs.get("force_simt_only", False)
        if "num_warps" in kwargs and kwargs["num_warps"] is not None:
            self.user_specified_warps = kwargs["num_warps"]
        else:
            self.user_specified_warps = None
        if "multibuffer" in kwargs and kwargs["multibuffer"] is not None:
            self.user_specified_multibuffer = kwargs["multibuffer"]
        else:
            self.user_specified_multibuffer = None

        # generate key
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        key = [_args[v] for _, v in self.keys.items() if v in _args]

        # Currently, we use the dtype with maximum byte length
        dtype = None
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
                dtype = (
                    arg.dtype
                    if get_byte_per_numel(arg.dtype) >= get_byte_per_numel(dtype)
                    else dtype
                )
        if dtype is None:
            raise NotImplementedError("Not support for non-Tensor inputs")

        fixed_compile_options = {
            name: kwargs[name]
            for name in get_compile_option_param_names(self.compile_options)
            if name in kwargs and kwargs[name] is not None
        }
        if fixed_compile_options:
            key.append(
                tuple(
                    sorted(
                        (name, _hashable_value(value))
                        for name, value in fixed_compile_options.items()
                    )
                )
            )
        self.fixed_compile_options = fixed_compile_options
        self._search_params_runtime_enabled = self._use_search_params_runtime()
        key = tuple(key)
        if key not in self.cache:
            if self.auto_gen_config and not self._search_params_runtime_enabled:
                self._autoparse_axis_params(all_args)
                _kv_dict = {k: _args[v] for k, v in self.keys.items() if v in _args}
                self._gen_tile_configs(_kv_dict, dtype)
            if self._search_params_runtime_enabled:
                self.configs = self.gen_configs + self.user_configs
                if not self.configs:
                    self.configs = [
                        _make_config_compat(
                            kwargs={},
                            num_warps=4,
                            num_stages=2,
                            num_ctas=1,
                            num_buffers_warp_spec=0,
                            num_consumer_groups=0,
                            reg_dec_producer=0,
                            reg_inc_consumer=0,
                        )
                    ]
                if self.print_autotuning or self.search_params.debug:
                    print(
                        "Search params autotuning: enabled; "
                        f"operator_kind={self.operator_policy.operator_kind.value}, "
                        f"compile_options={self.compile_options.kernel_type}, "
                        f"params={self.search_params.params}, "
                        f"values={self.search_params.values}, "
                        f"fixed_runtime_params={sorted(fixed_compile_options)}, "
                        f"base_configs={len(self.configs)}, "
                        "skip old compile_options cartesian expansion",
                        flush=True,
                    )
                    notice = self._search_params_correctness_notice()
                    if notice:
                        print(notice, flush=True)
                return key
            gen_configs = expand_compile_option_configs(
                self.gen_configs,
                self.compile_options,
                generated_tiling=True,
                fixed_options=fixed_compile_options,
            )
            user_configs = expand_compile_option_configs(
                self.user_configs,
                self.compile_options,
                generated_tiling=False,
                fixed_options=fixed_compile_options,
            )
            if self.print_autotuning and self.compile_options.enabled:
                print(
                    "Triton autotuning compile_options: "
                    f"kernel_type={self.compile_options.kernel_type}, "
                    f"manual_params={sorted(self.compile_options.params.keys())}, "
                    f"fixed_runtime_params={sorted(fixed_compile_options.keys())}, "
                    f"max_configs={self.compile_options.max_configs}, "
                    f"generated_tiling_configs={len(self.gen_configs)}->{len(gen_configs)}, "
                    f"user_configs={len(self.user_configs)}->{len(user_configs)}"
                )
                for idx, sample in enumerate(
                    summarize_compile_option_configs(gen_configs + user_configs), 1
                ):
                    print(f"Triton autotuning compile_options sample[{idx}]: {sample}")
            if len(gen_configs) == 0 and len(user_configs) == 0:
                self.configs = [
                    _make_config_compat(
                        kwargs={},
                        num_warps=4,
                        num_stages=2,
                        num_ctas=1,
                        num_buffers_warp_spec=0,
                        num_consumer_groups=0,
                        reg_dec_producer=0,
                        reg_inc_consumer=0,
                    )
                ]
                self.configs = expand_compile_option_configs(
                    self.configs,
                    self.compile_options,
                    generated_tiling=True,
                    fixed_options=fixed_compile_options,
                )
                if self.print_autotuning and self.compile_options.enabled:
                    print(
                        "Triton autotuning compile_options fallback: "
                        f"configs=1->{len(self.configs)}"
                    )
                    for idx, sample in enumerate(
                        summarize_compile_option_configs(self.configs), 1
                    ):
                        print(
                            f"Triton autotuning compile_options sample[{idx}]: {sample}"
                        )
            else:
                self.configs = gen_configs + user_configs
        return key

    def run(self, *args, **kwargs):
        key = self.generate_key_and_configs(*args, **kwargs)
        if self.is_simt_mode and kwargs.get("simt_stack_limit", None) is None:
            kwargs["simt_stack_limit"] = self.simt_stack_limit
        used_cached_result = True
        if key not in self.cache:
            if self._search_params_runtime_enabled:
                used_cached_result = False
                bench_start = time.time()
                try:
                    config, timings = self._run_search_params_autotune(*args, **kwargs)
                finally:
                    self.bench_time = time.time() - bench_start
                    self.last_search_stats["bench_time"] = self.bench_time
                self.cache[key] = config
                full_nargs = {
                    **self.nargs,
                    **kwargs,
                    **self.cache[key].all_kwargs(),
                }
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
                if self.print_autotuning_timings:
                    self._print_config_timings(timings)
                config = self.cache[key]
            else:
                # prune configs
                pruned_configs = self.prune_configs(kwargs)
                if len(pruned_configs) > 1:
                    used_cached_result = False

                    def benchmark():
                        bench_start = time.time()
                        timings = self._batch_bench(
                            *args, configs=pruned_configs, **kwargs
                        )
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = builtins.min(timings, key=timings.get)
                        full_nargs = {
                            **self.nargs,
                            **kwargs,
                            **self.cache[key].all_kwargs(),
                        }
                        self.pre_hook(full_nargs, reset_only=True)
                        self.configs_timings = timings
                        if self.print_autotuning_timings:
                            self._print_config_timings(timings)

                    benchmark()
                    config = self.cache[key]
                else:
                    self.cache[key] = pruned_configs[0]
                    config = self.cache[key]
        else:
            config = self.cache[key]

        self.best_config = config
        if self._search_params_runtime_enabled and used_cached_result:
            self._record_search_stats(searched=False)
        if self.print_autotuning and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: "
                f"{format_compile_option_result(self.best_config, self.compile_options, getattr(self, 'fixed_compile_options', {}))};"
            )

        if not used_cached_result and self.auto_profile_dir is not None:
            self._profile(*args, config=self.best_config, **kwargs)
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        final_kwargs = dict(kwargs, **config.all_kwargs())
        ret = self.fn.run(
            *args,
            **final_kwargs,
        )
        self.nargs = None
        return ret

    @staticmethod
    def _timing_sort_key(cost):
        if isinstance(cost, (list, tuple)) and cost:
            return cost[0]
        return cost

    def _is_finite_timing(self, cost):
        value = self._timing_sort_key(cost)
        return isinstance(value, (int, float)) and math.isfinite(value)

    def _print_config_timings(self, timings):
        sorted_timings = sorted(
            timings.items(), key=lambda item: self._timing_sort_key(item[1])
        )
        for idx, (config, cost) in enumerate(sorted_timings, 1):
            print(
                "Triton autotuning timing"
                f"[{idx}]: cost={cost}; "
                f"{format_compile_option_result(config, self.compile_options, getattr(self, 'fixed_compile_options', {}))};",
                flush=True,
            )

    def _batch_bench(
        self, *args, configs, force_parallel=False, return_errors=False, **kwargs
    ):
        from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources

        kernels_call = {
            config: self._make_kernel_call(*args, config=config, **kwargs)
            for config in configs
        }
        run_fns = {}
        errors = {}
        exc = None
        exc_stack = ""

        use_parallel = self.compile_parallel or (
            force_parallel and self._can_async_compile
        )
        if use_parallel:
            import psutil

            cpu_count = psutil.cpu_count(logical=False) or 1
            max_workers = max(1, min(max(1, cpu_count // 2), len(kernels_call)))
            future_kernels = []
            try:
                with (
                    ThreadPoolExecutor(max_workers=max_workers) as executor,
                    triton.AsyncCompileMode(executor),
                ):
                    for config, fn in kernels_call.items():
                        future_kernels.append((config, fn(warmup=True)))

                    for config, fut in future_kernels:
                        try:
                            if hasattr(fut, "result"):
                                fut = fut.result()
                            run_fns[config] = functools.partial(
                                kernels_call[config], warmup=False
                            )
                        except (
                            CompileTimeAssertionFailure,
                            CompilationError,
                            OutOfResources,
                            Exception,
                        ) as e:
                            _empty_npu_cache_after_failure()
                            import traceback

                            exc_stack = traceback.format_exc()
                            exc = e
                            errors[config] = e
            except Exception as e:
                # ignore exception from __exit__() of AsyncCompileMode
                triton.runtime._async_compile.active_mode.set(None)
                if exc is None:
                    _empty_npu_cache_after_failure()
                    import traceback

                    exc_stack = traceback.format_exc()
                    exc = e
                for config in configs:
                    if config not in run_fns:
                        errors.setdefault(config, e)
        else:
            for config, fn in kernels_call.items():
                try:
                    fn(warmup=False)
                    run_fns[config] = functools.partial(fn, warmup=False)
                except (
                    CompileTimeAssertionFailure,
                    CompilationError,
                    OutOfResources,
                    Exception,
                ) as e:
                    _empty_npu_cache_after_failure()
                    import traceback

                    exc_stack = traceback.format_exc()
                    exc = e
                    errors[config] = e

        if len(run_fns) == 0:
            if return_errors:
                return {}, errors
            raise RuntimeError(
                f"No valid triton configs. {type(exc).__name__}: {exc} \nStack trace: {exc_stack}"
            )

        bench_fn = self.do_bench
        if (
            getattr(self, "search_params", None) is not None
            and self.search_params.bench_warmup is not None
            and self.search_params.bench_active is not None
        ):
            bench_fn = functools.partial(
                self.do_bench,
                warmup=self.search_params.bench_warmup,
                active=self.search_params.bench_active,
            )
        strategy = select_benchmark_strategy(
            bench_fn,
            self.user_defined_do_bench,
            len(run_fns),
        )
        timings = {}
        try:
            timings = strategy.bench(run_fns)
        except Exception:
            _empty_npu_cache_after_failure()
            fallback_timings, fallback_errors = self._batch_bench_fallback(
                strategy, run_fns
            )
            timings.update(fallback_timings)
            errors.update(fallback_errors)
        if return_errors:
            return timings, errors
        return timings

    def _collect_search_ub_for_configs(self, *args, configs, **kwargs):
        """Cache compile-time UB bytes for Stage 1 shape ranking.

        UB collection is intentionally separate from timing benchmark. It reads
        ``CompiledKernel.metadata.required_ub_bits``, which the backend fills
        from ``memory_info_{aic,aiv}.json`` when ``TRITON_MEMORY_DISPLAY=1`` enables
        ``--enable-memory-display=true``.
        """
        if not getattr(self, "search_params", None):
            return
        if not hasattr(self, "_search_ub_cache"):
            self._search_ub_cache = {}
        configs = [config for config in configs if config not in self._search_ub_cache]
        if not configs:
            return

        prev_memory_display = os.environ.get("TRITON_MEMORY_DISPLAY")
        os.environ["TRITON_MEMORY_DISPLAY"] = "1"
        try:
            if self._can_async_compile:
                import psutil

                cpu_count = psutil.cpu_count(logical=False) or 1
                max_workers = max(1, min(max(1, cpu_count // 2), len(configs)))
                future_kernels = []
                try:
                    with (
                        ThreadPoolExecutor(max_workers=max_workers) as executor,
                        triton.AsyncCompileMode(executor),
                    ):
                        for config in configs:
                            kernel_call = self._make_kernel_call(
                                *args, config=config, **kwargs
                            )
                            future_kernels.append((config, kernel_call(warmup=True)))

                        for config, fut in future_kernels:
                            try:
                                if hasattr(fut, "result"):
                                    fut = fut.result()
                                ub = ub_bytes_of(fut)
                                if ub is not None:
                                    self._search_ub_cache[config] = ub
                            except Exception:
                                _empty_npu_cache_after_failure()
                except Exception:
                    triton.runtime._async_compile.active_mode.set(None)
                    _empty_npu_cache_after_failure()
                return

            for config in configs:
                try:
                    kernel_call = self._make_kernel_call(*args, config=config, **kwargs)
                    compiled = kernel_call(warmup=True)
                    if hasattr(compiled, "result"):
                        compiled = compiled.result()
                except Exception:
                    _empty_npu_cache_after_failure()
                    continue
                ub = ub_bytes_of(compiled)
                if ub is not None:
                    self._search_ub_cache[config] = ub
        finally:
            if prev_memory_display is None:
                os.environ.pop("TRITON_MEMORY_DISPLAY", None)
            else:
                os.environ["TRITON_MEMORY_DISPLAY"] = prev_memory_display

    def _batch_bench_fallback(self, strategy, run_fns):
        timings = {}
        errors = {}
        for config, fn in run_fns.items():
            try:
                cost = strategy.bench({config: fn})
                if isinstance(cost, dict):
                    timings[config] = cost.get(config, float("inf"))
                elif isinstance(cost, (list, tuple)) and len(cost) == 1:
                    timings[config] = cost[0]
                elif isinstance(cost, (int, float)):
                    timings[config] = cost
                else:
                    timings[config] = float("inf")
                    errors[config] = RuntimeError(
                        "benchmark fallback returned non-scalar timing"
                    )
            except Exception as exc:
                _empty_npu_cache_after_failure()
                timings[config] = float("inf")
                errors[config] = exc
        return timings, errors

    def _bench_stage1_fast_configs(self, *args, configs, **kwargs):
        """Compile Stage-1 probe configs in parallel, then time them cheaply.

        Stage 1 only ranks shape candidates, so it intentionally avoids the
        CANN profiler based do_bench_npu path. Stage 2 uses the same fast
        timing path; Stage 3 is the only precise do_bench_npu pass.
        """
        from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources
        from triton.testing import do_bench

        self._search_bench_call_count += 1
        self._search_bench_config_count += len(configs)

        kernels_call = {
            config: self._make_kernel_call(*args, config=config, **kwargs)
            for config in configs
        }
        run_fns = {}
        errors = {}

        prev_memory_display = os.environ.get("TRITON_MEMORY_DISPLAY")
        os.environ["TRITON_MEMORY_DISPLAY"] = "1"
        try:
            if self._can_async_compile:
                import psutil

                cpu_count = psutil.cpu_count(logical=False) or 1
                max_workers = max(1, min(max(1, cpu_count // 2), len(kernels_call)))
                future_kernels = []
                try:
                    with (
                        ThreadPoolExecutor(max_workers=max_workers) as executor,
                        triton.AsyncCompileMode(executor),
                    ):
                        for config, fn in kernels_call.items():
                            future_kernels.append((config, fn(warmup=True)))

                        for config, fut in future_kernels:
                            try:
                                if hasattr(fut, "result"):
                                    fut = fut.result()
                                ub = ub_bytes_of(fut)
                                if ub is not None:
                                    self._search_ub_cache[config] = ub
                                run_fns[config] = functools.partial(
                                    kernels_call[config], warmup=False
                                )
                            except (
                                CompileTimeAssertionFailure,
                                CompilationError,
                                OutOfResources,
                                Exception,
                            ) as exc:
                                _empty_npu_cache_after_failure()
                                errors[config] = exc
                except Exception as exc:
                    triton.runtime._async_compile.active_mode.set(None)
                    _empty_npu_cache_after_failure()
                    for config in configs:
                        errors.setdefault(config, exc)
            else:
                for config, fn in kernels_call.items():
                    try:
                        compiled = fn(warmup=True)
                        if hasattr(compiled, "result"):
                            compiled = compiled.result()
                        ub = ub_bytes_of(compiled)
                        if ub is not None:
                            self._search_ub_cache[config] = ub
                        run_fns[config] = functools.partial(fn, warmup=False)
                    except (
                        CompileTimeAssertionFailure,
                        CompilationError,
                        OutOfResources,
                        Exception,
                    ) as exc:
                        _empty_npu_cache_after_failure()
                        errors[config] = exc
        finally:
            if prev_memory_display is None:
                os.environ.pop("TRITON_MEMORY_DISPLAY", None)
            else:
                os.environ["TRITON_MEMORY_DISPLAY"] = prev_memory_display

        timings = {}
        for config, run_fn in run_fns.items():
            try:
                cost = do_bench(
                    run_fn,
                    warmup=self.search_params.stage1_bench_warmup,
                    rep=self.search_params.stage1_bench_rep,
                    quantiles=(0.5, 0.2, 0.8),
                )
                if isinstance(cost, (list, tuple)):
                    cost = cost[0] if cost else float("inf")
                if self._is_finite_timing(cost):
                    timings[config] = cost
                else:
                    errors[config] = RuntimeError("stage1 fast benchmark returned inf")
            except Exception as exc:  # noqa: BLE001
                _empty_npu_cache_after_failure()
                errors[config] = exc
        return timings, errors

    def _bench_stage2_fast_configs(self, *args, configs, **kwargs):
        """Stage-2 fast timing for every compile-profile candidate."""
        from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources
        from triton.testing import do_bench

        self._search_bench_call_count += 1
        self._search_bench_config_count += len(configs)

        prev_memory_display = os.environ.pop("TRITON_MEMORY_DISPLAY", None)
        prev_memory_display_debug = os.environ.pop("TRITON_MEMORY_DISPLAY_DEBUG", None)
        try:
            kernels_call = {
                config: self._make_kernel_call(*args, config=config, **kwargs)
                for config in configs
            }
            run_fns = {}
            errors = {}

            if self._can_async_compile:
                import psutil

                cpu_count = psutil.cpu_count(logical=False) or 1
                max_workers = max(1, min(max(1, cpu_count // 2), len(kernels_call)))
                future_kernels = []
                try:
                    with (
                        ThreadPoolExecutor(max_workers=max_workers) as executor,
                        triton.AsyncCompileMode(executor),
                    ):
                        for config, fn in kernels_call.items():
                            future_kernels.append((config, fn(warmup=True)))

                        for config, fut in future_kernels:
                            try:
                                if hasattr(fut, "result"):
                                    fut = fut.result()
                                run_fns[config] = functools.partial(
                                    kernels_call[config], warmup=False
                                )
                            except (
                                CompileTimeAssertionFailure,
                                CompilationError,
                                OutOfResources,
                                Exception,
                            ) as exc:
                                _empty_npu_cache_after_failure()
                                errors[config] = exc
                except Exception as exc:
                    triton.runtime._async_compile.active_mode.set(None)
                    _empty_npu_cache_after_failure()
                    for config in configs:
                        if config not in run_fns:
                            errors.setdefault(config, exc)
            else:
                for config, fn in kernels_call.items():
                    try:
                        compiled = fn(warmup=True)
                        if hasattr(compiled, "result"):
                            compiled = compiled.result()
                        run_fns[config] = functools.partial(fn, warmup=False)
                    except (
                        CompileTimeAssertionFailure,
                        CompilationError,
                        OutOfResources,
                        Exception,
                    ) as exc:
                        _empty_npu_cache_after_failure()
                        errors[config] = exc

            timings = {}
            for config, run_fn in run_fns.items():
                try:
                    cost = do_bench(
                        run_fn,
                        warmup=self.search_params.stage1_bench_warmup,
                        rep=self.search_params.stage1_bench_rep,
                        quantiles=(0.5, 0.2, 0.8),
                    )
                    if isinstance(cost, (list, tuple)):
                        cost = cost[0] if cost else float("inf")
                    if self._is_finite_timing(cost):
                        timings[config] = cost
                    else:
                        errors[config] = RuntimeError(
                            "stage2 fast benchmark returned inf"
                        )
                except Exception as exc:  # noqa: BLE001
                    _empty_npu_cache_after_failure()
                    errors[config] = exc
            return timings, errors
        finally:
            if prev_memory_display is not None:
                os.environ["TRITON_MEMORY_DISPLAY"] = prev_memory_display
            if prev_memory_display_debug is not None:
                os.environ["TRITON_MEMORY_DISPLAY_DEBUG"] = prev_memory_display_debug

    def _search_debug(self, message: str):
        if self.search_params.debug or self.print_autotuning:
            print(f"Search params autotuning: {message}", flush=True)

    def _format_search_error_log(self, error) -> str:
        if error is None:
            return "<none>"
        text = str(error)
        if not text:
            text = repr(error)
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _record_search_stats(self, *, searched: bool, stage2_candidates: int = 0):
        self.last_search_stats = {
            "searched": searched,
            "bench_time": getattr(self, "bench_time", 0.0),
            "bench_calls": self._search_bench_call_count if searched else 0,
            "bench_configs": self._search_bench_config_count if searched else 0,
            "measurements": len(self._search_measure_cache) if searched else 0,
            "stage1_success": (
                len(self._search_stage1_measured_summary) if searched else 0
            ),
            "stage1_fail": (len(self._search_stage1_failed_summary) if searched else 0),
            "stage2_final_candidates": stage2_candidates if searched else 0,
        }

    def _bench_search_config(self, *args, config, **kwargs):
        try:
            timings, errors = self._batch_bench(
                *args, configs=[config], return_errors=True, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            return None, exc
        cost = timings.get(config, float("inf"))
        if not self._is_finite_timing(cost):
            return None, errors.get(config) or RuntimeError("benchmark returned inf")
        return cost, None

    def _bench_search_configs(self, *args, configs, force_parallel=False, **kwargs):
        self._search_bench_call_count += 1
        self._search_bench_config_count += len(configs)
        try:
            return self._batch_bench(
                *args,
                configs=configs,
                force_parallel=force_parallel,
                return_errors=True,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            return {}, {config: exc for config in configs}

    def _shape_config_pool(self, kwargs):
        base_config = self.configs[0]
        self._search_param_anchor_shape_keys = {
            shape_key(
                extract_shape(config, self.search_params.params),
                self.search_params.params,
            )
            for config in self.configs
            if all(name in config.kwargs for name in self.search_params.params)
        }
        shape_configs = expand_search_param_shapes(base_config, self.search_params)
        old_configs = self.configs
        try:
            self.configs = shape_configs
            pruned_configs = self.prune_configs(kwargs)
        finally:
            self.configs = old_configs
        self._search_debug(
            f"Stage 0 shape pool: total={len(shape_configs)}, after_prune={len(pruned_configs)}"
        )
        return pruned_configs

    def _run_stage1_probe_for_shape(self, *args, shape_config, **kwargs):
        results, _ = self._run_stage1_probes_for_shapes(
            *args, shape_configs=[shape_config], **kwargs
        )
        return results[0] if results else None

    def _run_stage1_probes_for_shapes(self, *args, shape_configs, **kwargs):
        operator_policy = getattr(self, "operator_policy", None)
        profile_family = (
            operator_policy.stage1_profile_family
            if operator_policy is not None
            else "mixcv"
        )
        profiles = self._apply_fixed_compile_profiles(
            get_stage1_probe_profiles(profile_family)
        )
        if not profiles:
            return [], list(shape_configs)

        pending = [
            {
                "shape": extract_shape(config, self.search_params.params),
                "shape_config": config,
            }
            for config in shape_configs
        ]
        if not pending:
            return [], []

        results = []
        failed_configs = []

        def _bench_profile(items, profile, label):
            configs = [
                compile_profile_to_config(
                    profile,
                    shape_kwargs=item["shape"],
                    base_config=item["shape_config"],
                )
                for item in items
            ]
            timings = {}
            errors = {}
            batch_configs = []
            for shape_item, config in zip(items, configs):
                self._search_debug(
                    f"Stage 1 {label} shape={shape_item['shape']}, profile={profile}"
                )
                cached = self._search_measure_cache.get(shape_item["shape"], profile)
                if cached is None:
                    batch_configs.append(config)
                    continue
                if cached["ok"] and cached.get("source") == "stage1_fast":
                    timings[config] = cached["time"]
                    if cached.get("ub") is not None:
                        self._search_ub_cache[config] = cached["ub"]
                    self._search_debug(
                        f"Stage 1 {label} cache hit "
                        f"shape={shape_item['shape']}, cost={cached['time']}"
                    )
                    continue
                if not cached["ok"]:
                    errors[config] = cached.get("error") or RuntimeError(
                        f"cached failure classified_as={cached.get('failure')}"
                    )
                    self._search_debug(
                        f"Stage 1 {label} cache failure "
                        f"shape={shape_item['shape']}, "
                        f"classified_as={cached.get('failure')}"
                    )
                    continue
                batch_configs.append(config)
            if batch_configs:
                batch_timings, batch_errors = self._bench_stage1_fast_configs(
                    *args, configs=batch_configs, **kwargs
                )
                timings.update(batch_timings)
                errors.update(batch_errors)
            return configs, timings, errors

        remaining = pending
        last_failed = {}
        for probe_index, profile in enumerate(profiles, 1):
            if not remaining:
                break
            probe_configs, probe_timings, probe_errors = _bench_profile(
                remaining, profile, f"probe[{probe_index}]"
            )
            next_remaining = []
            for item, config in zip(remaining, probe_configs):
                shape = item["shape"]
                cost = probe_timings.get(config, float("inf"))
                if self._is_finite_timing(cost):
                    self._search_measure_cache.put(
                        shape, profile, config, cost=cost, source="stage1_fast"
                    )
                    self._search_debug(
                        f"Stage 1 probe[{probe_index}] success shape={shape}, "
                        f"cost={cost}, "
                        f"ub={getattr(self, '_search_ub_cache', {}).get(config)}"
                    )
                    results.append(
                        {
                            "shape": shape,
                            "shape_config": item["shape_config"],
                            "profile": profile,
                            "config": config,
                            "time": cost,
                            "ub": getattr(self, "_search_ub_cache", {}).get(config),
                        }
                    )
                    continue

                error = probe_errors.get(config)
                failure = (
                    classify_compile_failure(error)
                    if error is not None
                    else "EXACT_ONLY"
                )
                self._search_measure_cache.put(
                    shape,
                    profile,
                    config,
                    error=error,
                    failure=failure,
                    source="stage1_fast",
                )
                last_failed[shape_key(shape, self.search_params.params)] = (
                    item,
                    failure,
                    error,
                )
                next_remaining.append(item)
                self._search_debug(
                    f"Stage 1 probe[{probe_index}] fail shape={shape}, "
                    f"classified_as={failure}, "
                    f"resource_ratio={resource_overflow_ratio(error)}, "
                    f"try_next={probe_index < len(profiles)}, "
                    f"error_log:\n{self._format_search_error_log(error)}"
                )
            remaining = next_remaining

        for item in remaining:
            shape = item["shape"]
            failure_info = last_failed.get(shape_key(shape, self.search_params.params))
            failure = failure_info[1] if failure_info is not None else "EXACT_ONLY"
            error = failure_info[2] if failure_info is not None else None
            failed_configs.append(item["shape_config"])
            self._search_debug(
                "Stage 1 all probes failed "
                f"shape={shape}, classified_as={failure}, "
                f"resource_ratio={resource_overflow_ratio(error)}, "
                f"error_log:\n{self._format_search_error_log(error)}"
            )

        return results, failed_configs

    def _screen_search_param_shapes(self, *args, **kwargs):
        shape_configs = self._shape_config_pool(kwargs)
        if not shape_configs:
            raise RuntimeError("No valid search_param shapes after early_config_prune")

        measured = []
        failed_configs = []
        observed_configs = []
        seen_shape_keys = set()
        value_map = effective_search_value_map(shape_configs, self.search_params)
        initial_percentiles = stage1_initial_percentiles(self.search_params, value_map)
        total_stage1_budget = stage1_total_budget(self.search_params, shape_configs)
        initial_budget = stage1_initial_budget(self.search_params, shape_configs)
        initial = select_initial_percentile_shapes(
            shape_configs,
            self.search_params,
            limit=initial_budget,
        )
        initial_keys = {
            shape_key(
                extract_shape(config, self.search_params.params),
                self.search_params.params,
            )
            for config in initial
        }
        anchor_configs = []
        for config in shape_configs:
            key = shape_key(
                extract_shape(config, self.search_params.params),
                self.search_params.params,
            )
            if key in self._search_param_anchor_shape_keys and key not in initial_keys:
                anchor_configs.append(config)
        initial.extend(anchor_configs)
        deduped_initial = []
        deduped_initial_keys = set()
        for config in initial:
            key = shape_key(
                extract_shape(config, self.search_params.params),
                self.search_params.params,
            )
            if key in deduped_initial_keys:
                continue
            deduped_initial_keys.add(key)
            deduped_initial.append(config)
        initial = deduped_initial
        if len(initial) > total_stage1_budget:
            initial = initial[:total_stage1_budget]
        self._search_debug(
            f"Stage 1 initial cases={len(initial)} "
            f"(percentile + anchors={len(anchor_configs)}), "
            f"budget={total_stage1_budget}, "
            f"effective_value_counts="
            f"{ {name: len(values) for name, values in value_map.items()} }, "
            f"initial_percentiles={initial_percentiles}"
        )

        for config in initial:
            shape = extract_shape(config, self.search_params.params)
            seen_shape_keys.add(shape_key(shape, self.search_params.params))
            observed_configs.append(config)
        initial_results, initial_failures = self._run_stage1_probes_for_shapes(
            *args, shape_configs=initial, **kwargs
        )
        measured.extend(initial_results)
        failed_configs.extend(initial_failures)

        for round_id in range(self.search_params.shape_refine_rounds):
            candidates_for_parents = filter_shapes_by_ub_and_timing(
                measured, timing_sort_key=self._timing_sort_key
            )
            if len(candidates_for_parents) < len(measured):
                self._search_debug(
                    "Stage 1 evolve round="
                    f"{round_id + 1}, UB/timing filter dropped "
                    f"{len(measured) - len(candidates_for_parents)} entries"
                )
            parents = select_top_shape_entries(
                candidates_for_parents,
                k=self.search_params.shape_final_top_k,
                key_fn=lambda item: ub_timing_weighted_key(
                    item, timing_sort_key=self._timing_sort_key
                ),
            )
            if (
                getattr(self.operator_policy, "operator_kind", None)
                == OperatorKind.DOT_STATEFUL
                and len(self.search_params.params) == 1
                and len(parents) >= 2
            ):
                self._search_debug(
                    "Stage 1 evolve stop: DOT_STATEFUL single-param search "
                    f"already has {len(parents)} successful parent shapes"
                )
                break
            unseen_shapes = sum(
                1
                for config in shape_configs
                if shape_key(
                    extract_shape(config, self.search_params.params),
                    self.search_params.params,
                )
                not in seen_shape_keys
            )
            remaining_budget = total_stage1_budget - len(seen_shape_keys)
            remaining_rounds = self.search_params.shape_refine_rounds - round_id
            children_per_round = stage1_children_per_round(
                self.search_params,
                shape_configs,
                remaining_unseen=unseen_shapes,
                remaining_budget=remaining_budget,
                remaining_rounds=remaining_rounds,
            )
            proposals = propose_evolved_shape_configs(
                parents=parents,
                successes=measured,
                failures=failed_configs,
                failure_observations=(
                    self._search_measure_cache.stage1_failure_observations(
                        failed_configs
                    )
                ),
                observed=observed_configs,
                all_configs=shape_configs,
                spec=self.search_params,
                seen_keys=seen_shape_keys,
                limit=children_per_round,
            )
            self._search_debug(
                "Stage 1 evolve round="
                f"{round_id + 1}, proposals={len(proposals)}, "
                f"parents={len(parents)}, successes={len(measured)}, "
                f"failures={len(failed_configs)}, target_children={children_per_round}, "
                f"unseen_shapes={unseen_shapes}, "
                "proposal_shapes="
                + ", ".join(
                    str(extract_shape(config, self.search_params.params))
                    for config in proposals
                )
            )
            if not proposals:
                self._search_debug(
                    "Stage 1 evolve stop: no new proposal "
                    f"(unseen_shapes={unseen_shapes}, "
                    "all candidates may already be measured or rejected by "
                    "failure-cone/acquisition filters)"
                )
                break
            for config in proposals:
                shape = extract_shape(config, self.search_params.params)
                seen_shape_keys.add(shape_key(shape, self.search_params.params))
                observed_configs.append(config)
            proposal_results, proposal_failures = self._run_stage1_probes_for_shapes(
                *args, shape_configs=proposals, **kwargs
            )
            measured.extend(proposal_results)
            failed_configs.extend(proposal_failures)

        if not measured:
            raise RuntimeError("No valid search_param shapes after Stage 1 screening")

        selected = select_top_shape_entries(
            measured,
            k=self.search_params.shape_final_top_k,
            key_fn=lambda item: ub_timing_weighted_key(
                item, timing_sort_key=self._timing_sort_key
            ),
        )
        self._search_debug(
            "Stage 1 selected shapes: "
            + ", ".join(
                f"{item['shape']}@{self._timing_sort_key(item['time'])}"
                for item in selected
            )
        )
        self._search_stage1_measured_summary = select_top_shape_entries(
            measured,
            k=len(measured),
            key_fn=lambda item: self._timing_sort_key(item["time"]),
        )
        self._search_stage1_failed_summary = [
            extract_shape(config, self.search_params.params)
            for config in failed_configs
        ]
        return selected

    def _search_param_baseline_candidate(self):
        for config in list(self.user_configs) + list(self.gen_configs):
            if all(name in config.kwargs for name in self.search_params.params):
                profile = {
                    "mode": "BASELINE",
                    "num_stages": getattr(config, "num_stages", None),
                }
                self._search_debug(f"Stage 3 baseline guard candidate: config={config}")
                return config, profile, float("inf")
        self._search_debug(
            "Stage 3 baseline guard unavailable: no original config contains "
            f"all search params {self.search_params.params}"
        )
        return None

    def _make_stage2_searcher(self, *args, **kwargs):
        def bench_configs(configs):
            return self._bench_stage2_fast_configs(*args, configs=configs, **kwargs)

        return Stage2CompileSearcher(
            search_params=self.search_params,
            operator_policy=self.operator_policy,
            cache=self._search_measure_cache,
            apply_fixed_profile=self._apply_fixed_compile_profile,
            bench_configs=bench_configs,
            timing_sort_key=self._timing_sort_key,
            is_finite_timing=self._is_finite_timing,
            debug=self._search_debug,
            format_error_log=self._format_search_error_log,
        )

    def _run_stage3_precise_pick(
        self,
        *args,
        candidates,
        baseline_candidate=None,
        **kwargs,
    ):
        reference_fn = getattr(self.search_params, "reference_fn", None)

        ranked_candidates = []
        seen_configs = set()
        for config, profile, cost in sorted(
            candidates, key=lambda item: self._timing_sort_key(item[2])
        ):
            if config in seen_configs:
                continue
            seen_configs.add(config)
            ranked_candidates.append((config, profile, cost))

        check_queue = list(ranked_candidates[:3])
        if baseline_candidate is not None:
            baseline_config, baseline_profile, baseline_cost = baseline_candidate
            if baseline_config not in seen_configs:
                check_queue.append((baseline_config, baseline_profile, baseline_cost))
                seen_configs.add(baseline_config)
        self._search_debug(
            f"Stage 3 precise check: top-{len(check_queue)} of "
            f"{len(candidates)} candidates"
        )
        failures = []
        passing = []
        checked_configs = set()
        next_ranked_index = 3

        while True:
            while check_queue:
                config, profile, cost = check_queue.pop(0)
                if config in checked_configs:
                    continue
                checked_configs.add(config)
                rank = len(checked_configs)
                precise_timings, precise_errors = self._bench_search_configs(
                    *args, configs=[config], **kwargs
                )
                precise_cost = precise_timings.get(config, float("inf"))
                if not self._is_finite_timing(precise_cost):
                    error = precise_errors.get(config)
                    message = self._format_search_error_log(error)
                    failures.append(f"rank={rank} precise failed: {message}")
                    self._search_debug(
                        "Stage 3 precise fail "
                        f"rank={rank}, config={config}, error={message}"
                    )
                    continue

                try:
                    passed = (
                        True
                        if reference_fn is None
                        else reference_fn(self, config, *args, **kwargs)
                    )
                except Exception as exc:
                    failures.append(f"rank={rank} accuracy error: {exc}")
                    self._search_debug(
                        f"Stage 3 accuracy error rank={rank}, "
                        f"config={config}, error={exc}"
                    )
                    continue
                if not passed:
                    failures.append(f"rank={rank} accuracy failed")
                    self._search_debug(
                        f"Stage 3 accuracy fail rank={rank}, "
                        f"config={config}, cost={cost}"
                    )
                    continue
                self._search_debug(
                    "Stage 3 precise pass "
                    f"rank={rank}, config={config}, fast_cost={cost}, "
                    f"precise_cost={precise_cost}"
                )
                passing.append((config, profile, precise_cost, rank, cost))

            if passing or reference_fn is None:
                break
            while (
                next_ranked_index < len(ranked_candidates)
                and ranked_candidates[next_ranked_index][0] in checked_configs
            ):
                next_ranked_index += 1
            if next_ranked_index >= len(ranked_candidates):
                break
            extra = ranked_candidates[next_ranked_index]
            next_ranked_index += 1
            check_queue.append(extra)
            self._search_debug(
                "Stage 3 extend precise check after all checked candidates "
                f"failed correctness; next_config={extra[0]}"
            )

        if passing:
            baseline = next(
                (
                    item
                    for item in passing
                    if isinstance(item[1], dict) and item[1].get("mode") == "BASELINE"
                ),
                None,
            )
            tuned = [
                item
                for item in passing
                if not (isinstance(item[1], dict) and item[1].get("mode") == "BASELINE")
            ]
            if baseline is not None and tuned:
                best_tuned = min(tuned, key=lambda item: self._timing_sort_key(item[2]))
                baseline_time = self._timing_sort_key(baseline[2])
                tuned_time = self._timing_sort_key(best_tuned[2])
                if tuned_time < baseline_time:
                    config, profile, precise_cost, rank, fast_cost = best_tuned
                    self._search_debug(
                        "Stage 3 baseline guard accept tuned "
                        f"tuned_precise={precise_cost}, "
                        f"baseline_precise={baseline[2]}"
                    )
                else:
                    config, profile, precise_cost, rank, fast_cost = baseline
                    self._search_debug(
                        "Stage 3 baseline guard fallback "
                        f"best_tuned_precise={best_tuned[2]}, "
                        f"baseline_precise={precise_cost}"
                    )
            else:
                config, profile, precise_cost, rank, fast_cost = min(
                    passing, key=lambda item: self._timing_sort_key(item[2])
                )
            self._search_debug(
                "Stage 3 final pick "
                f"rank={rank}, config={config}, fast_cost={fast_cost}, "
                f"precise_cost={precise_cost}"
            )
            return config, profile, precise_cost

        raise RuntimeError(
            "Stage 3 top-3 candidates failed precise benchmark or accuracy check: "
            + "; ".join(failures)
        )

    def _run_search_params_autotune(self, *args, **kwargs):
        self._search_ub_cache = {}
        self._search_measure_cache = SearchMeasureCache(
            self.search_params.params,
            self._search_ub_cache,
        )
        self._search_bench_call_count = 0
        self._search_bench_config_count = 0
        all_candidates = []
        try:
            stage1_entries = self._screen_search_param_shapes(*args, **kwargs)
            if not stage1_entries:
                raise RuntimeError("No valid search_param shapes after Stage 1")
            stage1_top1_time = min(entry["time"] for entry in stage1_entries)
            stage2_searcher = self._make_stage2_searcher(*args, **kwargs)
            timings = {}
            for stage1_rank, entry in enumerate(stage1_entries, 1):
                if not getattr(self.operator_policy, "stage2_enabled", True):
                    all_candidates.append(
                        (entry["config"], entry["profile"], entry["time"])
                    )
                    timings[entry["config"]] = entry["time"]
                    continue
                shape_candidates = stage2_searcher.search(
                    stage1_entry=entry,
                    stage1_rank=stage1_rank,
                    stage1_top1_time=stage1_top1_time,
                )
                for config, profile, cost in shape_candidates:
                    all_candidates.append((config, profile, cost))
                    timings[config] = cost
            if not all_candidates:
                raise RuntimeError(
                    "No valid compile profiles after search_params compile search"
                )
            baseline_candidate = self._search_param_baseline_candidate()
            stage3_pick = self._run_stage3_precise_pick(
                *args,
                candidates=all_candidates,
                baseline_candidate=baseline_candidate,
                **kwargs,
            )
            best_config, best_profile, best_time = stage3_pick
            timings[best_config] = best_time
            self._search_debug(
                f"Stage 3 best time={best_time}, profile={best_profile}, "
                f"config={best_config} (precise-picked)"
            )
            self._search_debug(
                "Stage 1 measured summary: "
                + ", ".join(
                    f"{item['shape']}@{self._timing_sort_key(item['time'])}"
                    for item in self._search_stage1_measured_summary
                )
            )
            if self._search_stage1_failed_summary:
                self._search_debug(
                    "Stage 1 failed summary: "
                    + ", ".join(
                        str(shape) for shape in self._search_stage1_failed_summary
                    )
                )
            self._record_search_stats(
                searched=True, stage2_candidates=len(all_candidates)
            )
            return best_config, timings
        except Exception:
            self._record_search_stats(
                searched=True, stage2_candidates=len(all_candidates)
            )
            raise

    def _search_params_correctness_notice(self):
        search_params = getattr(self, "search_params", None)
        if search_params is None or getattr(search_params, "reference_fn", None):
            return None
        kind = getattr(self.operator_policy, "operator_kind", None)
        risky_kinds = {
            OperatorKind.DOT_STATEFUL,
            OperatorKind.VECTOR_DISCRETE_OR_STATEFUL,
            OperatorKind.UNKNOWN,
        }
        if kind not in risky_kinds:
            return None
        return (
            "Search params autotuning: correctness notice; "
            f"operator_kind={kind.value} has no reference_fn. "
            "Only use search_params for true performance tile knobs. "
            "Keep coverage, atomic granularity, state partition, and "
            "hand-derived UB-safe constants fixed unless a correctness gate "
            "is provided."
        )

    def _make_kernel_call(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call(warmup):
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                current.update({"warmup": warmup})
                res = self.fn.run(
                    *args,
                    **current,
                )
                if warmup:
                    return res
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(full_nargs, exception=None)

        return kernel_call

    def warmup(self, *args, **kwargs):
        _ = self.generate_key_and_configs(*args, **kwargs)
        pruned_configs = self.prune_configs(kwargs)
        ret = []
        if self.compile_parallel:
            import psutil

            max_workers = min(psutil.cpu_count(logical=False) // 2, len(pruned_configs))
            with (
                ThreadPoolExecutor(max_workers=max_workers) as executor,
                triton.AsyncCompileMode(executor),
            ):
                for config in pruned_configs:
                    ret.append(self.fn.warmup(*args, **kwargs, **config.all_kwargs()))
        else:
            for config in pruned_configs:
                ret.append(self.fn.warmup(*args, **kwargs, **config.all_kwargs()))
        self.nargs = None
        return ret

    def _profile(self, *args, config, **meta):
        from ..testing import do_bench_npu

        kernel_call = self._make_kernel_call(*args, config=config, **meta)
        fn = functools.partial(kernel_call, warmup=False)
        do_bench_npu(fn, prof_dir=self.auto_profile_dir, keep_res=True)

    def _autoparse_split_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the split axis parameters from triton kernel code.
        """
        func_ast = self._parse_ast()
        parser = SplitAxesParser(func_ast, self.keys, candidates_params)
        split_axes = parser.parse()
        self.split_axis_pid_dims = dict(getattr(parser, "split_axis_pid_dims", {}))
        self.axis_pid_dims = dict(getattr(parser, "axis_pid_dims", {}))
        if self.print_autotuning:
            print(
                f"Ascend autotuning parse split axes: {split_axes}, "
                f"split axis pid dims: {self.split_axis_pid_dims}, "
                f"axis pid dims: {self.axis_pid_dims}"
            )
        return split_axes

    def _autoparse_axis_pid_dims(self) -> Dict[str, int]:
        """
        Extract axis -> program_id dim mapping without relying on split-parameter
        classification, so fixed-grid semantics can always consume it.
        """
        func_ast = self._parse_ast()
        parser = SplitAxesParser(
            func_ast,
            self.keys,
            self._get_constexpr_candidates(),
        )
        _ = parser.parse()
        self.axis_pid_dims = dict(getattr(parser, "axis_pid_dims", {}))
        self.split_axis_pid_dims = dict(getattr(parser, "split_axis_pid_dims", {}))
        if self.print_autotuning:
            print(
                "Ascend autotuning parse axis pid dims (independent): "
                f"{self.axis_pid_dims}"
            )
        return self.axis_pid_dims

    def _get_constexpr_candidates(self) -> List[str]:
        """
        Returns all constexpr parameter names from the kernel function definition.
        """
        func_ast = self._parse_ast()
        constexpr_names = []
        for node in ast.walk(func_ast):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not isinstance(node.args, ast.arguments):
                continue
            for arg in node.args.args:
                if not isinstance(arg, ast.arg):
                    continue
                ann = arg.annotation
                if (
                    isinstance(ann, ast.Attribute)
                    and isinstance(ann.value, ast.Name)
                    and ann.value.id == "tl"
                    and ann.attr == "constexpr"
                ):
                    constexpr_names.append(arg.arg)
            break
        return constexpr_names

    def _get_fixed_grid_dim_values(
        self, grid, all_args: Dict[str, object] = None
    ) -> Dict[int, int]:
        """
        Returns fixed grid dim -> value.
        - Static tuple/list grid: direct extraction
        - Callable grid: infer fixed dims by perturbing missing constexpr params
        """
        if grid is None:
            return {}
        if callable(grid):
            return self._infer_fixed_dims_from_callable_grid(grid, all_args or {})
        return self._extract_fixed_grid_dims(grid)

    def _extract_fixed_grid_dims(self, grid) -> Dict[int, int]:
        if isinstance(grid, int):
            grid = (grid,)
        if not isinstance(grid, (tuple, list)):
            return {}
        fixed_dims = {}
        for idx, dim in enumerate(grid):
            if isinstance(dim, int) and dim > 0:
                fixed_dims[idx] = dim
        return fixed_dims

    def _normalize_grid_tuple(self, grid_out):
        if isinstance(grid_out, int):
            return (grid_out,)
        if isinstance(grid_out, (tuple, list)):
            return tuple(grid_out)
        return None

    def _infer_fixed_dims_from_callable_grid(
        self, grid_fn, all_args: Dict[str, object]
    ) -> Dict[int, int]:
        constexpr_candidates = self._get_constexpr_candidates()
        base_meta = dict(all_args or {})

        # Fill missing constexpr with stable probe defaults so grid(meta) can execute.
        for name in constexpr_candidates:
            if name not in base_meta:
                base_meta[name] = 128

        try:
            base_grid_raw = grid_fn(dict(base_meta))
        except Exception:
            return {}

        base_grid = self._normalize_grid_tuple(base_grid_raw)
        if base_grid is None:
            return {}

        dynamic_dims = set()
        # Missing constexpr are tunable candidates.
        tunable_probe_names = [
            name for name in constexpr_candidates if name not in (all_args or {})
        ]
        probe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        for name in tunable_probe_names:
            baseline = base_meta.get(name, 128)
            for probe in probe_values:
                if probe == baseline:
                    continue
                probe_meta = dict(base_meta)
                probe_meta[name] = probe
                try:
                    probe_grid_raw = grid_fn(probe_meta)
                except Exception:
                    continue
                probe_grid = self._normalize_grid_tuple(probe_grid_raw)
                if probe_grid is None:
                    continue
                if len(probe_grid) != len(base_grid):
                    dynamic_dims.update(range(min(len(probe_grid), len(base_grid))))
                    continue
                for idx, (base_dim, probe_dim) in enumerate(zip(base_grid, probe_grid)):
                    if not (isinstance(base_dim, int) and isinstance(probe_dim, int)):
                        dynamic_dims.add(idx)
                        continue
                    if base_dim != probe_dim:
                        dynamic_dims.add(idx)

        fixed_dims = {}
        for idx, dim in enumerate(base_grid):
            if idx in dynamic_dims:
                continue
            if isinstance(dim, int) and dim > 0:
                fixed_dims[idx] = dim
        return fixed_dims

    def _autoparse_tiling_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the tiling axis parameters from triton kernel code.
        """
        func_ast = self._parse_ast()
        parser = TilingAxesParser(func_ast, self.keys, candidates_params)
        tiling_axes = parser.parse()
        if self.print_autotuning:
            print(f"Ascend autotuning parse tiling axes: {tiling_axes}")
        return tiling_axes

    def _autoparse_reduction_axes(self) -> List[str]:
        """
        Extracts the reduction axis parameters from triton kernel code.
        """
        func_ast = self._parse_ast()
        parser = ReductionAxesParser(func_ast, self.keys)
        reduction_axes = parser.parse()
        for axis in reduction_axes:
            self.keys[f"r{axis}"] = self.keys.pop(axis)
        reduction_axes = [f"r{axis}" for axis in reduction_axes]

        if self.print_autotuning:
            print(
                f"Ascend autotuning parse keys: {self.keys} \n"
                f"Ascend autotuning parse reduction axes: {reduction_axes}"
            )
        return reduction_axes

    def _autoparse_low_dim_axes(self) -> List[str]:
        """
        Extracts the low dimension axis from triton kernel code.
        """
        func_ast = self._parse_ast()
        parser = LowDimsAxesParser(func_ast, self.keys)
        low_dim_axes = parser.parse()
        if len(low_dim_axes) < 1:
            if self.print_autotuning:
                print(
                    "[WARNING] Failed to parse low-dimensional axes, fallback to empty low_dim_axes."
                )
            return []
        if self.print_autotuning:
            print(f"Ascend autotuning parse low dimensional axes: {low_dim_axes}")
        return low_dim_axes

    def _autoparse_ptr_nums(self, all_args: dict) -> int:
        """
        Counts the number of pointer parameters from triton kernel code.
        """
        ptr_nums = 0
        ptr_params = list()
        for k, v in all_args.items():
            if isinstance(v, Tensor):
                ptr_nums += 1
                ptr_params.append(k)

        if self.print_autotuning:
            print(
                f"Ascend autotuning parse pointer params: {ptr_params}, pointer nums: {ptr_nums}"
            )
        return ptr_nums

    def _get_persistent_reduction_threshold(self, reduction_axis: str) -> int:
        # Keep this heuristic aligned with inductor-style policy:
        # inner reduction axis uses a larger threshold than other axes.
        if self.low_dim_axes and reduction_axis == self.low_dim_axes[0]:
            return 1024
        return 64


def autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
    *,
    auto_prof_dir=None,
    hints=None,
):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param auto_prof_dir: the specified directory to store the profiling results of the best config.
        If this parameter is None or the best config is retrieved from cache, the profiling process will be ignored.
    :type auto_prof_dir: str
    :param hints: a dict of autotune hint auguments passed to AutoTilingTuner.
    """

    def decorator(fn):
        return AutoTilingTuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
            auto_profile_dir=auto_prof_dir,
            hints=hints,
        )

    return decorator


_ALL_PARAMS = {
    "num_stages",
    "unit_flag",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
    "enable_hivm_auto_cv_balance",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
    "enable_ubuf_saving",
}

_DEFAULTS = {
    "num_stages": [2],
    "unit_flag": [False],
    "limit_auto_multi_buffer_only_for_local_buffer": [False],
    "limit_auto_multi_buffer_of_local_buffer": ["no-l0c"],
    "set_workspace_multibuffer": [2, 4],
    "enable_hivm_auto_cv_balance": [True],
    "tile_mix_vector_loop": [2, 4],
    "tile_mix_cube_loop": [2, 4],
    "enable_ubuf_saving": [True],
}

_VALID_VALUES = {
    "num_stages": [1, 2],
    "limit_auto_multi_buffer_of_local_buffer": ["no-limit", "no-l0c"],
    "set_workspace_multibuffer": [2, 4],
    "tile_mix_vector_loop": [2, 4, 8],
    "tile_mix_cube_loop": [2, 4, 8],
}

_CUBE_PARAMS = {"num_stages", "unit_flag", "limit_auto_multi_buffer_of_local_buffer"}

_MIXCV_PARAMS = {
    "num_stages",
    "unit_flag",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
    "enable_hivm_auto_cv_balance",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
    "enable_ubuf_saving",
}

_VECTOR_PARAMS = {
    "num_stages",
    "enable_ubuf_saving",
}


def _check_boolean_list(val, param_name):
    return (
        isinstance(val, (list, tuple))
        and len(val) > 0
        and all(isinstance(x, bool) for x in val)
    )


def _check_string_in_set(val, valid_set, param_name):
    return (
        isinstance(val, (list, tuple))
        and len(val) > 0
        and all(v in valid_set for v in val)
    )


def _check_int_in_set(val, valid_set, param_name):
    return (
        isinstance(val, (list, tuple))
        and len(val) > 0
        and all(isinstance(v, int) and v in valid_set for v in val)
    )


_VALIDATION_RULES = {
    "num_stages": {
        "desc": f"must be one or more of: {_VALID_VALUES['num_stages']}",
        "check": lambda val, p: _check_int_in_set(val, _VALID_VALUES["num_stages"], p),
    },
    "unit_flag": {
        "desc": "must be non-empty list/tuple of boolean values",
        "check": _check_boolean_list,
    },
    "limit_auto_multi_buffer_only_for_local_buffer": {
        "desc": "must be non-empty list/tuple of boolean values",
        "check": _check_boolean_list,
    },
    "limit_auto_multi_buffer_of_local_buffer": {
        "desc": f"must be one or more of: {_VALID_VALUES['limit_auto_multi_buffer_of_local_buffer']}",
        "check": lambda val, p: _check_string_in_set(
            val, _VALID_VALUES["limit_auto_multi_buffer_of_local_buffer"], p
        ),
    },
    "set_workspace_multibuffer": {
        "desc": f"must be one or more of: {_VALID_VALUES['set_workspace_multibuffer']}",
        "check": lambda val, p: _check_int_in_set(
            val, _VALID_VALUES["set_workspace_multibuffer"], p
        ),
    },
    "enable_hivm_auto_cv_balance": {
        "desc": "must be non-empty list/tuple of boolean values",
        "check": _check_boolean_list,
    },
    "tile_mix_vector_loop": {
        "desc": f"must be one or more of: {_VALID_VALUES['tile_mix_vector_loop']}",
        "check": lambda val, p: _check_int_in_set(
            val, _VALID_VALUES["tile_mix_vector_loop"], p
        ),
    },
    "tile_mix_cube_loop": {
        "desc": f"must be one or more of: {_VALID_VALUES['tile_mix_cube_loop']}",
        "check": lambda val, p: _check_int_in_set(
            val, _VALID_VALUES["tile_mix_cube_loop"], p
        ),
    },
    "enable_ubuf_saving": {
        "desc": "must be non-empty list/tuple of boolean values",
        "check": _check_boolean_list,
    },
}


class BaseAutotuner:
    """
    Base class for generating auto-tuning configurations without block dimensions.
    Users must provide fixed dimension parameters when calling the kernel.
    """

    def __init__(
        self, operator_name, supported_params, default_params, validation_rules
    ):
        self.operator_name = operator_name
        self.supported_params = supported_params
        self.default_params = default_params
        self.validation_rules = validation_rules

    def validate_parameters(self, **kwargs):
        # Check for unsupported parameters
        invalid_params = [k for k in kwargs.keys() if k not in _ALL_PARAMS]
        if invalid_params:
            print(
                f"[ERROR] Invalid parameters for {self.operator_name}: {invalid_params}"
            )
            return False

        for param, rule in self.validation_rules.items():
            if param in kwargs:
                if not rule["check"](kwargs[param], param):
                    print(
                        f"[ERROR] Invalid value for '{param}' in {self.operator_name}: {kwargs[param]}"
                    )
                    print(f"        Expected: {rule['desc']}")
                    return False
        return True

    def get_configs(self, **kwargs):
        """
        Generate a list of Config objects.
        Each parameter must be provided as a list (even for a single value).
        The function produces the Cartesian product of all parameter lists.
        - num_stages: each value will be set as Config.num_stages (not placed in kwargs)
        - other parameters: each value will be placed in Config.kwargs
        Returns a list of Config objects.
        """
        if not self.validate_parameters(**kwargs):
            return []

        # Collect parameter values, using defaults for missing ones
        param_values = {}
        for p in sorted(self.supported_params):
            if p in kwargs:
                param_values[p] = kwargs[p]
            else:
                param_values[p] = self.default_params.get(p, [None])

        keys = list(param_values.keys())
        values = list(param_values.values())
        combos = list(itertools.product(*values))

        configs = []
        for combo in combos:
            config_kwargs = {}
            num_stages_val = None
            for i, pname in enumerate(keys):
                val = combo[i]
                if pname == "num_stages":
                    num_stages_val = val
                else:
                    config_kwargs[pname] = val

            configs.append(
                Config(
                    kwargs=config_kwargs,
                    num_stages=num_stages_val if num_stages_val is not None else 2,
                )
            )
        return configs


CubeAutotuner = BaseAutotuner(
    operator_name="cube",
    supported_params=_CUBE_PARAMS,
    default_params=_DEFAULTS,
    validation_rules=_VALIDATION_RULES,
)

MixcvAutotuner = BaseAutotuner(
    operator_name="mixcv",
    supported_params=_MIXCV_PARAMS,
    default_params=_DEFAULTS,
    validation_rules=_VALIDATION_RULES,
)

VectorAutotuner = BaseAutotuner(
    operator_name="vector",
    supported_params=_VECTOR_PARAMS,
    default_params=_DEFAULTS,
    validation_rules=_VALIDATION_RULES,
)


def get_autotune_cube_config(**kwargs: Any) -> List[triton.Config]:
    """
    Generate autotune configuration for the cube operator.
    Supported parameters: num_stages, unit_flag, limit_auto_multi_buffer_of_local_buffer.
    """
    import triton

    return CubeAutotuner.get_configs(**kwargs)


def get_autotune_cv_config(**kwargs: Any) -> List[triton.Config]:
    """
    Generate autotune configuration for the mixcv operator.
    Supported parameters: num_stages, unit_flag, limit_auto_multi_buffer_only_for_local_buffer,
                limit_auto_multi_buffer_of_local_buffer, set_workspace_multibuffer,
                enable_hivm_auto_cv_balance, tile_mix_vector_loop, tile_mix_cube_loop, enable_ubuf_saving
    """
    import triton

    return MixcvAutotuner.get_configs(**kwargs)


def get_autotune_vector_config(**kwargs: Any) -> List[triton.Config]:
    """
    Generate autotune configuration for the vector operator.
    Supported parameters: num_stages, enable_ubuf_saving
    """
    import triton

    return VectorAutotuner.get_configs(**kwargs)


def get_max_configs(config, kernel_type="mixcv", **kwargs):
    """
    Expand a single base Config by combining it with tuning parameters.

    :param config: A triton.Config object serving as the base.
    :param kernel_type: Operator type, one of "cube", "mixcv", "vector". Default "mixcv".
    :param kwargs: Tuning parameters, each provided as a list (e.g., enable_hivm_auto_cv_balance=[True, False]).
                   If a parameter is not provided, its value is taken from the base config (if present)
                   or from the defaults.
    :return: List of expanded Config objects.
    """
    # Determine the set of parameters supported by the current kernel_type
    if kernel_type == "cube":
        supported = _CUBE_PARAMS
    elif kernel_type == "vector":
        supported = _VECTOR_PARAMS
    else:
        supported = _MIXCV_PARAMS

    # Warn about unsupported parameters provided in kwargs
    unsupported = [k for k in kwargs if k not in supported and k in _ALL_PARAMS]
    if unsupported:
        print(
            f"[WARNING] The following parameters are not supported for kernel_type '{kernel_type}': {unsupported}. They will be ignored."
        )

    # Build value lists for each parameter (priority: kwargs > base config > defaults)
    param_values = {}
    base_kwargs = config.kwargs
    base_num_stages = config.num_stages

    for param in sorted(supported):
        if param in kwargs:
            # User-provided list via tuning_params takes precedence
            val_list = kwargs[param]
        elif param == "num_stages":
            # num_stages is an attribute of Config, not part of kwargs.
            # Triton's default is 3, but Ascend only supports the local defaults.
            # Treat an unsupported base value as "not fixed" so examples that use
            # triton.Config(kwargs={...}) still follow the Ascend default table.
            val_list = (
                [base_num_stages]
                if base_num_stages in _VALID_VALUES["num_stages"]
                else _DEFAULTS[param]
            )
        elif param in base_kwargs:
            # Parameter present in base config's kwargs -> fix to that single value
            val_list = [base_kwargs[param]]
        else:
            # Otherwise fall back to defaults
            val_list = _DEFAULTS.get(param, [None])

        # Validate the value list
        if param in _VALIDATION_RULES:
            rule = _VALIDATION_RULES[param]
            if not rule["check"](val_list, param):
                raise ValueError(
                    f"Invalid value for '{param}': {val_list}. Expected: {rule['desc']}"
                )
        param_values[param] = val_list

    # Cartesian product of all parameter lists
    keys = list(param_values.keys())
    values = list(param_values.values())
    combos = list(itertools.product(*values))

    new_configs = []
    for combo in combos:
        # Start with a copy of the original config's kwargs
        new_kwargs = config.kwargs.copy()
        num_stages_val = None

        for i, pname in enumerate(keys):
            val = combo[i]
            if pname == "num_stages":
                num_stages_val = val
            else:
                # Overwrite or add the parameter to kwargs
                new_kwargs[pname] = val

        config_args = {
            "kwargs": new_kwargs,
            "num_warps": getattr(config, "num_warps", 4),
            "num_stages": (
                num_stages_val
                if num_stages_val is not None
                else getattr(config, "num_stages", 2)
            ),
            "num_ctas": getattr(config, "num_ctas", 1),
            "maxnreg": getattr(config, "maxnreg", None),
            "pre_hook": getattr(config, "pre_hook", None),
            "ir_override": getattr(config, "ir_override", None),
            "num_buffers_warp_spec": getattr(config, "num_buffers_warp_spec", None),
            "num_consumer_groups": getattr(config, "num_consumer_groups", None),
            "reg_dec_producer": getattr(config, "reg_dec_producer", None),
            "reg_inc_consumer": getattr(config, "reg_inc_consumer", None),
        }
        new_config = _make_config_compat(**config_args)
        new_configs.append(new_config)

    return new_configs


def max_autotune(
    configs,
    key,
    kernel_type="mixcv",
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
    **tuning_params,
):
    """
    Decorator that expands each base Config with tuning parameters before auto-tuning.

    Usage is similar to @triton.autotune, but allows automatic expansion of
    additional tuning parameters (e.g., enable_hivm_auto_cv_balance, tile_mix_vector_loop, ...)
    for each provided base configuration.

    :param configs: List of base triton.Config objects.
    :param key: List of argument names whose change triggers re-tuning.
    :param kernel_type: Operator type, one of "cube", "mixcv", "vector". Default "mixcv".
    :param prune_configs_by: Same as in autotune.
    :param reset_to_zero: Same as in autotune.
    :param restore_value: Same as in autotune.
    :param pre_hook: Same as in autotune.
    :param post_hook: Same as in autotune.
    :param warmup: Deprecated.
    :param rep: Deprecated.
    :param use_cuda_graph: Deprecated.
    :param do_bench: Same as in autotune.
    :param tuning_params: Additional tuning parameters as keyword arguments.
                          Each value must be a list; the Cartesian product of these lists
                          will be combined with each base config.
    """

    def decorator(fn):
        if not configs or len(configs) == 0:
            raise ValueError(
                "[max_autotune] The argument 'configs' cannot be empty. "
                "Please provide at least one base config. "
            )
        # Expand each base config with the provided tuning parameters
        expanded_configs = []
        for cfg in configs:
            expanded = get_max_configs(cfg, kernel_type=kernel_type, **tuning_params)
            expanded_configs.extend(expanded)

        # Call the original autotune decorator with the expanded configs
        return autotune(
            configs=expanded_configs,
            key=key,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
        )(fn)

    return decorator
