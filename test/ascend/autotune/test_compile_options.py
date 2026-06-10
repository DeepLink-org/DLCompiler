import pytest
import triton

from backend.ascend_autotune_runtime.compile_options import (
    expand_compile_option_configs,
    format_compile_option_result,
    parse_compile_options_hint,
)


def test_compile_options_string_hint_expands_vector_defaults():
    spec = parse_compile_options_hint("vector")
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
    )

    assert len(configs) == 4
    assert {cfg.num_stages for cfg in configs} == {1, 2}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in configs} == {True, False}
    assert {cfg.kwargs["enable_tuning_mode"] for cfg in configs} == {True}
    assert {cfg.kwargs["BLOCK_SIZE"] for cfg in configs} == {1024}


def test_compile_options_string_hint_expands_mixcv_auto_search_space_without_limit():
    spec = parse_compile_options_hint("mixcv")
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
    )

    assert spec.max_configs is None
    assert len(configs) == 156
    assert {cfg.num_stages for cfg in configs} == {1, 2}
    assert {cfg.kwargs["enable_tuning_mode"] for cfg in configs} == {True}
    assert {cfg.kwargs["unit_flag"] for cfg in configs} == {False, True}
    assert {cfg.kwargs["enable_hivm_auto_cv_balance"] for cfg in configs} == {True}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in configs} == {False, True}

    stage1_configs = [cfg for cfg in configs if cfg.num_stages == 1]
    assert len(stage1_configs) == 4
    assert all("multibuffer" not in cfg.kwargs for cfg in stage1_configs)
    assert all(
        "limit_auto_multi_buffer_only_for_local_buffer" not in cfg.kwargs
        for cfg in stage1_configs
    )
    assert all(
        "limit_auto_multi_buffer_of_local_buffer" not in cfg.kwargs
        for cfg in stage1_configs
    )
    assert all("set_workspace_multibuffer" not in cfg.kwargs for cfg in stage1_configs)
    assert all("tile_mix_vector_loop" not in cfg.kwargs for cfg in stage1_configs)
    assert all("tile_mix_cube_loop" not in cfg.kwargs for cfg in stage1_configs)
    assert all("enable_preload" not in cfg.kwargs for cfg in stage1_configs)
    assert {cfg.kwargs["enable_auto_bind_sub_block"] for cfg in stage1_configs} == {
        True
    }

    stage2_configs = [cfg for cfg in configs if cfg.num_stages == 2]
    assert len(stage2_configs) == 152
    assert all("multibuffer" not in cfg.kwargs for cfg in stage2_configs)
    assert {
        cfg.kwargs["limit_auto_multi_buffer_only_for_local_buffer"]
        for cfg in stage2_configs
    } == {False, True}
    assert {
        cfg.kwargs["limit_auto_multi_buffer_of_local_buffer"] for cfg in stage2_configs
    } == {"no-limit", "no-l0c"}
    assert {
        cfg.kwargs["set_workspace_multibuffer"]
        for cfg in stage2_configs
        if "set_workspace_multibuffer" in cfg.kwargs
    } == {2, 4}
    assert {
        cfg.kwargs["tile_mix_vector_loop"]
        for cfg in stage2_configs
        if "tile_mix_vector_loop" in cfg.kwargs
    } == {1, 2, 4}
    assert {
        cfg.kwargs["tile_mix_cube_loop"]
        for cfg in stage2_configs
        if "tile_mix_cube_loop" in cfg.kwargs
    } == {1, 2, 4}
    assert {cfg.kwargs["enable_auto_bind_sub_block"] for cfg in stage2_configs} == {
        True
    }
    assert any(
        "set_workspace_multibuffer" in cfg.kwargs
        and "tile_mix_vector_loop" in cfg.kwargs
        and "tile_mix_cube_loop" in cfg.kwargs
        for cfg in stage2_configs
    )
    assert any(
        "set_workspace_multibuffer" not in cfg.kwargs
        and "tile_mix_vector_loop" not in cfg.kwargs
        and "tile_mix_cube_loop" not in cfg.kwargs
        for cfg in stage2_configs
    )
    workspace_configs = [
        cfg for cfg in stage2_configs if "set_workspace_multibuffer" in cfg.kwargs
    ]
    assert len(workspace_configs) == 144
    assert {cfg.num_stages for cfg in workspace_configs} == {2}
    assert {
        cfg.kwargs["limit_auto_multi_buffer_only_for_local_buffer"]
        for cfg in workspace_configs
    } == {False}


def test_compile_options_workspace_pruned_when_auto_multibuffer_disabled():
    spec = parse_compile_options_hint("mixcv")
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
        fixed_options={"multibuffer": False, "num_stages": 2},
    )

    assert configs
    assert all("set_workspace_multibuffer" not in cfg.kwargs for cfg in configs)


def test_compile_options_workspace_pruned_when_workspace_limit_enabled():
    spec = parse_compile_options_hint("mixcv")
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
        fixed_options={
            "limit_auto_multi_buffer_only_for_local_buffer": True,
            "num_stages": 2,
        },
    )

    assert configs
    assert all("set_workspace_multibuffer" not in cfg.kwargs for cfg in configs)


def test_compile_options_explicit_mixcv_values_are_not_restricted_by_auto_search_space():
    spec = parse_compile_options_hint(
        {
            "kernel_type": "mixcv",
            "num_stages": [2],
            "multibuffer": [False],
            "unit_flag": [True],
            "limit_auto_multi_buffer_only_for_local_buffer": [True],
            "limit_auto_multi_buffer_of_local_buffer": ["no-limit"],
            "set_workspace_multibuffer": [4],
            "enable_hivm_auto_cv_balance": [False],
            "tile_mix_vector_loop": [8],
            "tile_mix_cube_loop": [8],
            "enable_ubuf_saving": [False],
            "enable_auto_bind_sub_block": [False],
        }
    )
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
    )

    assert len(configs) == 1
    assert configs[0].num_stages == 2
    assert configs[0].kwargs["enable_tuning_mode"] is True
    assert configs[0].kwargs["multibuffer"] is False
    assert configs[0].kwargs["unit_flag"] is True
    assert configs[0].kwargs["enable_hivm_auto_cv_balance"] is False
    assert configs[0].kwargs["enable_auto_bind_sub_block"] is False
    assert "limit_auto_multi_buffer_only_for_local_buffer" not in configs[0].kwargs
    assert "limit_auto_multi_buffer_of_local_buffer" not in configs[0].kwargs
    assert "set_workspace_multibuffer" not in configs[0].kwargs
    assert "tile_mix_vector_loop" not in configs[0].kwargs
    assert "tile_mix_cube_loop" not in configs[0].kwargs


def test_compile_options_explicit_mixcv_stage1_values_are_preserved():
    spec = parse_compile_options_hint(
        {
            "kernel_type": "mixcv",
            "num_stages": [1],
            "multibuffer": [True],
            "limit_auto_multi_buffer_only_for_local_buffer": [False],
            "limit_auto_multi_buffer_of_local_buffer": ["no-l0c"],
            "set_workspace_multibuffer": [2],
            "tile_mix_vector_loop": [4],
            "tile_mix_cube_loop": [4],
            "enable_auto_bind_sub_block": [True],
        }
    )
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
    )

    assert len(configs) == 4
    assert {cfg.num_stages for cfg in configs} == {1}
    assert {cfg.kwargs["enable_tuning_mode"] for cfg in configs} == {True}
    assert {cfg.kwargs["unit_flag"] for cfg in configs} == {False, True}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in configs} == {False, True}
    assert all(cfg.kwargs["multibuffer"] is True for cfg in configs)
    assert all("set_workspace_multibuffer" not in cfg.kwargs for cfg in configs)
    assert all("tile_mix_vector_loop" not in cfg.kwargs for cfg in configs)
    assert all("tile_mix_cube_loop" not in cfg.kwargs for cfg in configs)


def test_compile_options_auto_search_overrides_base_compile_options():
    spec = parse_compile_options_hint("mixcv")
    configs = expand_compile_option_configs(
        [
            triton.Config(
                {
                    "BLOCK_SIZE": 1024,
                    "enable_tuning_mode": False,
                    "unit_flag": True,
                    "enable_ubuf_saving": True,
                    "set_workspace_multibuffer": 4,
                    "tile_mix_vector_loop": 4,
                    "tile_mix_cube_loop": 4,
                },
                num_stages=1,
            )
        ],
        spec,
        generated_tiling=False,
    )

    assert len(configs) == 156
    assert {cfg.kwargs["enable_tuning_mode"] for cfg in configs} == {True}
    assert {cfg.kwargs["unit_flag"] for cfg in configs} == {False, True}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in configs} == {False, True}
    assert all(
        "multibuffer" not in cfg.kwargs for cfg in configs if cfg.num_stages == 1
    )
    assert all(
        "limit_auto_multi_buffer_only_for_local_buffer" not in cfg.kwargs
        for cfg in configs
        if cfg.num_stages == 1
    )
    assert all(
        "limit_auto_multi_buffer_of_local_buffer" not in cfg.kwargs
        for cfg in configs
        if cfg.num_stages == 1
    )
    assert all(
        "set_workspace_multibuffer" not in cfg.kwargs
        for cfg in configs
        if cfg.num_stages == 1
    )
    assert all(
        "tile_mix_vector_loop" not in cfg.kwargs
        for cfg in configs
        if cfg.num_stages == 1
    )
    assert all(
        "tile_mix_cube_loop" not in cfg.kwargs for cfg in configs if cfg.num_stages == 1
    )
    assert all(
        "enable_preload" not in cfg.kwargs for cfg in configs if cfg.num_stages == 1
    )
    assert {
        cfg.kwargs["enable_auto_bind_sub_block"]
        for cfg in configs
        if cfg.num_stages == 1
    } == {True}


def test_compile_options_runtime_fixed_options_are_not_redefined():
    spec = parse_compile_options_hint("mixcv")
    configs = expand_compile_option_configs(
        [
            triton.Config(
                {
                    "BLOCK_SIZE": 1024,
                    "multibuffer": False,
                },
                num_stages=2,
            )
        ],
        spec,
        generated_tiling=False,
        fixed_options={
            "multibuffer": True,
        },
    )

    assert configs
    assert all("multibuffer" not in cfg.kwargs for cfg in configs)
    assert {cfg.num_stages for cfg in configs} == {1, 2}


def test_compile_options_runtime_fixed_num_stages_limits_search():
    spec = parse_compile_options_hint("mixcv")
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=False,
        fixed_options={"num_stages": 2},
    )

    assert len(configs) == 152
    assert {cfg.num_stages for cfg in configs} == {2}


def test_compile_options_format_stage1_effective_options():
    spec = parse_compile_options_hint("mixcv")
    config = triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "enable_tuning_mode": True,
            "enable_ubuf_saving": True,
            "enable_hivm_auto_cv_balance": True,
            "unit_flag": True,
        },
        num_stages=1,
    )

    text = format_compile_option_result(config, spec)

    assert "selected_meta: BLOCK_M=32, BLOCK_N=32, num_stages=1" in text
    assert "enable_auto_multi_buffer=False" in text
    assert "set_workspace_multibuffer=<inactive: depends on auto multi-buffer>" in text
    assert "tile_mix_vector_loop=<inactive: depends on auto multi-buffer>" in text
    assert "tile_mix_cube_loop=<inactive: depends on auto multi-buffer>" in text
    assert "enable_preload" not in text


def test_compile_options_format_stage2_effective_options():
    spec = parse_compile_options_hint("mixcv")
    config = triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "enable_tuning_mode": True,
            "enable_ubuf_saving": False,
            "enable_hivm_auto_cv_balance": True,
            "limit_auto_multi_buffer_only_for_local_buffer": False,
            "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
            "set_workspace_multibuffer": 4,
            "tile_mix_vector_loop": 2,
            "tile_mix_cube_loop": 4,
            "unit_flag": False,
        },
        num_stages=2,
    )

    text = format_compile_option_result(config, spec)

    assert "selected_meta: BLOCK_M=64, BLOCK_N=128, num_stages=2" in text
    assert "enable_auto_multi_buffer=True" in text
    assert "set_workspace_multibuffer=4" in text
    assert "tile_mix_vector_loop=2" in text
    assert "tile_mix_cube_loop=4" in text
    assert "<inactive" not in text


def test_compile_options_dict_hint_limits_search_space():
    spec = parse_compile_options_hint(
        {
            "kernel_type": "vector",
            "num_stages": [2],
            "enable_ubuf_saving": [True],
        }
    )
    configs = expand_compile_option_configs(
        [triton.Config({"BLOCK_SIZE": 1024})],
        spec,
        generated_tiling=True,
    )

    assert len(configs) == 1
    assert configs[0].num_stages == 2
    assert configs[0].kwargs["enable_tuning_mode"] is True
    assert configs[0].kwargs["enable_ubuf_saving"] is True


def test_compile_options_respects_max_configs():
    spec = parse_compile_options_hint(
        {
            "kernel_type": "vector",
            "max_configs": 2,
        }
    )

    with pytest.raises(ValueError, match="generated more than 2 configs"):
        expand_compile_option_configs(
            [triton.Config({"BLOCK_SIZE": 1024})],
            spec,
            generated_tiling=True,
        )


def test_legacy_vector_config_helper_still_works():
    from backend.ascend_autotune_runtime.autotuner import get_autotune_vector_config

    configs = get_autotune_vector_config(
        num_stages=[1, 2],
        enable_ubuf_saving=[True, False],
    )

    assert len(configs) == 4
    assert {cfg.num_stages for cfg in configs} == {1, 2}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in configs} == {True, False}
