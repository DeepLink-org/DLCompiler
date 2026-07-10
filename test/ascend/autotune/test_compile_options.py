import pytest
import triton

from backend.ascend_autotune_runtime.schedule_profiles import (
    COMPILE_MODE_KEY,
    COMPILE_MODE_VECTOR,
    CompileFailureRegionSet,
    WORKSPACE_CV_AGGRESSIVE_PROBE,
    WORKSPACE_CV_LOW_RESOURCE_PROBE,
    WORKSPACE_CV_MIX1_PROBE,
    classify_compile_failure,
    compile_profile_to_config,
    effective_compile_profile_key,
    get_stage1_probe_configs,
    get_stage1_probe_profiles,
    make_stage2_seed_profiles,
    parse_compile_options_hint,
)


def test_stage1_probe_order_matches_search_params_design():
    profiles = get_stage1_probe_profiles()

    assert profiles == [
        WORKSPACE_CV_AGGRESSIVE_PROBE,
        WORKSPACE_CV_LOW_RESOURCE_PROBE,
        WORKSPACE_CV_MIX1_PROBE,
    ]
    assert {profile["unit_flag"] for profile in profiles} == {False}
    assert profiles[0]["set_workspace_multibuffer"] == 4
    assert profiles[0]["limit_auto_multi_buffer_of_local_buffer"] == "no-limit"
    assert profiles[1]["set_workspace_multibuffer"] == 2
    assert profiles[1]["limit_auto_multi_buffer_of_local_buffer"] == "no-l0c"
    assert (
        profiles[2]["tile_mix_cube_loop"],
        profiles[2]["tile_mix_vector_loop"],
    ) == (1, 1)

    configs = get_stage1_probe_configs({"BLOCK_M": 128, "BLOCK_N": 512})
    assert len(configs) == 3
    assert {cfg.num_stages for cfg in configs} == {2}
    assert {cfg.kwargs["unit_flag"] for cfg in configs} == {False}
    assert configs[0].kwargs["BLOCK_M"] == 128
    assert configs[0].kwargs["BLOCK_N"] == 512
    assert COMPILE_MODE_KEY not in configs[0].kwargs


def test_vector_compile_options_reject_mixcv_only_params():
    spec = parse_compile_options_hint({"kernel_type": "vector", "num_stages": [1, 2]})
    assert spec.params == {"num_stages": [1, 2]}

    for name, value in (
        ("unit_flag", False),
        ("tile_mix_vector_loop", 2),
        ("set_workspace_multibuffer", 2),
    ):
        with pytest.raises(ValueError, match=f"'{name}' is not supported"):
            parse_compile_options_hint({"kernel_type": "vector", name: value})


def test_stage2_seeds_keep_stage1_winner_first_and_unit_flag_disabled():
    seeds = make_stage2_seed_profiles(
        WORKSPACE_CV_MIX1_PROBE,
        shape_kwargs={"BLOCK_M": 128, "BLOCK_N": 512},
        seed_budget=8,
        allow_unit_flag=False,
    )

    assert seeds[0] == WORKSPACE_CV_MIX1_PROBE
    assert len(seeds) <= 8
    assert len({effective_compile_profile_key(seed) for seed in seeds}) == len(seeds)
    assert {seed["unit_flag"] for seed in seeds} == {False}
    assert all(
        seed[COMPILE_MODE_KEY] == WORKSPACE_CV_MIX1_PROBE[COMPILE_MODE_KEY]
        for seed in seeds
    )


def test_compile_profile_to_config_keeps_mode_internal():
    config = compile_profile_to_config(
        WORKSPACE_CV_LOW_RESOURCE_PROBE,
        shape_kwargs={"BLOCK_M": 64, "BLOCK_N": 64},
        base_config=triton.Config({"EXTRA": 1}, num_warps=8),
    )

    assert config.num_stages == 2
    assert config.num_warps == 8
    assert config.kwargs["EXTRA"] == 1
    assert config.kwargs["BLOCK_M"] == 64
    assert config.kwargs["tile_mix_cube_loop"] == 4
    assert COMPILE_MODE_KEY not in config.kwargs


def test_compile_failure_classification_and_resource_region_prune():
    assert (
        classify_compile_failure(
            "ub overflow, requires 3260416 bits while 1572864 bits available"
        )
        == "RESOURCE_UB"
    )
    assert (
        classify_compile_failure("internal error: dummyOps size is not 1")
        == "COMPILER_INTERNAL"
    )

    failed_regions = CompileFailureRegionSet()
    failed_regions.add(
        {
            **WORKSPACE_CV_LOW_RESOURCE_PROBE,
            "set_workspace_multibuffer": 4,
            "tile_mix_cube_loop": 2,
            "tile_mix_vector_loop": 2,
        },
        "RESOURCE_UB",
    )

    worse = {
        **WORKSPACE_CV_LOW_RESOURCE_PROBE,
        "set_workspace_multibuffer": 4,
        "tile_mix_cube_loop": 1,
        "tile_mix_vector_loop": 2,
        "limit_auto_multi_buffer_of_local_buffer": "no-limit",
        "enable_ubuf_saving": False,
    }
    relaxed = {
        **WORKSPACE_CV_LOW_RESOURCE_PROBE,
        "set_workspace_multibuffer": 2,
        "tile_mix_cube_loop": 4,
        "tile_mix_vector_loop": 4,
    }

    assert failed_regions.is_forbidden(worse)
    assert not failed_regions.is_forbidden(relaxed)
