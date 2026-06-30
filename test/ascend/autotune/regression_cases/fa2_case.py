from __future__ import annotations

import argparse
import json
import os
from functools import cache
from typing import Any, Dict, Tuple

import torch
import torch_npu
import triton
import triton.language as tl
from common import case_values
from triton.backends.dicp_triton.testing import do_bench_npu


torch_npu.npu.current_device()
import backend.ascend_autotune_hooks as ascend_autotune_hooks

_ASCEND_AUTOTUNE_HOOKS = ascend_autotune_hooks

os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING_TIMINGS", "1")

DEVICE = "npu"
BENCH_WARMUP = 1
BENCH_ACTIVE = 3


ATTENTION_SEARCH_HINTS = {
    "search_params": {
        "params": ["BLOCK_M", "BLOCK_N"],
    },
}


FA_TEST_CASES = [
    (1, 32, 512, 128, False, torch.float16),
    (1, 32, 1024, 128, False, torch.float16),
    (1, 32, 2048, 128, False, torch.float16),
    (1, 32, 10240, 128, False, torch.float16),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.getenv("DLC_AUTOTUNE_BENCH_WARMUP", str(BENCH_WARMUP))),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=int(os.getenv("DLC_AUTOTUNE_BENCH_ACTIVE", str(BENCH_ACTIVE))),
    )
    return parser.parse_args()


def prune_attention_configs(configs, nargs, **kwargs):
    n_ctx = kwargs.get("N_CTX", nargs.get("N_CTX"))
    if n_ctx is None:
        return configs
    pruned = [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] <= n_ctx and config.kwargs["BLOCK_N"] <= n_ctx
    ]
    return pruned or configs


@cache
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = (
        triton.runtime.driver.active.utils.get_device_properties(device)
    )

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


@triton.jit
def _attn_fwd_inner2(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)

        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1), propagate_nan=tl.PropagateNan.ALL)
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1), propagate_nan=tl.PropagateNan.ALL)
            qk = qk - m_ij[:, None]

        p = tl.math.exp(qk)
        p_cast = p.to(k.dtype)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p_cast, v)
        tl.extra.deeplink.cann.extension.compile_hint(pv, "hivm.tile_mix_cube_num", 2)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.autotune(
    configs=[],
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": prune_attention_configs},
    hints=ATTENTION_SEARCH_HINTS,
)
@triton.jit
def _attn_fwd2(
    Q,
    K,
    V,
    M,
    Out,
    sm_scale: tl.constexpr,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    NUM_BLOCKS_M = tl.cdiv(N_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    task_m_idx = 0
    task_hz_idx = 0

    for block_idx in range(pid, NUM_BLOCKS, core_step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(Q_block_ptr)

        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner2(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,
            )

        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner2(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,
            )

        m_i += tl.math.log(l_i)
        acc = acc / l_i[:, None]
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM=128, BN=None):
        head_dim_q, head_dim_k = q.shape[-1], k.shape[-1]
        head_dim_v = v.shape[-1]
        assert head_dim_q == head_dim_k and head_dim_k == head_dim_v
        assert head_dim_k in {16, 32, 64, 128, 512}

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        num_cores, _ = get_device_properties()
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        launch_meta: Dict[str, int] = {}
        if BN is not None:
            launch_meta["BLOCK_M"] = BM
            launch_meta["BLOCK_N"] = BN
        _attn_fwd2[(num_cores,)](
            q,
            k,
            v,
            M,
            o,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=head_dim_k,
            STAGE=stage,
            debug=os.environ.get("TRITON_DEBUG", "0") == "1",
            **launch_meta,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = head_dim_k
        ctx.causal = causal
        return o


attention = _attention.apply


def bench_operator(name, fn):
    print(
        f"--------------------benchmark_{name} for {BENCH_ACTIVE} times--------------------"
    )
    took = do_bench_npu(
        fn,
        warmup=BENCH_WARMUP,
        active=BENCH_ACTIVE,
        clear_l2_cache=False,
    )
    print(f"    [op time] {name}: {took:.6f} s ({took * 1000000:.3f} us)", flush=True)
    return took


def snapshot_attention_search_stats():
    current = dict(getattr(_attn_fwd2, "last_search_stats", {}))
    saved = dict(getattr(_attn_fwd2, "_last_case_search_stats", {}))
    if saved.get("searched", False) and not current.get("searched", False):
        return saved
    return current or saved


def benchmark(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM=128, BN=None):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5

    ref_out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        H,
        padding_mask=None,
        atten_mask=None,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=0,
    )[0]
    npu_time = bench_operator(
        "torch_npu.npu_fusion_attention",
        lambda: torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            H,
            padding_mask=None,
            atten_mask=None,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
        )[0],
    )

    tri_out = attention(q, k, v, causal, sm_scale, BM, BN)
    search_stats = snapshot_attention_search_stats()
    _attn_fwd2._last_case_search_stats = dict(search_stats)
    print(f"    [selected config] _attn_fwd2: {_attn_fwd2.best_config}", flush=True)
    triton_time = bench_operator(
        "attention.forward",
        lambda: attention(q, k, v, causal, sm_scale, BM, BN),
    )

    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0.0)
    return {
        "npu_time": npu_time,
        "triton_time": triton_time,
        "search_stats": search_stats,
        "best_config": str(_attn_fwd2.best_config),
    }


def load_fa_cases():
    cases = list(FA_TEST_CASES)
    selected_cases = os.getenv("TEST_FA2_CASES", "1,4")
    if selected_cases:
        selected = {int(item) for item in selected_cases.split(",") if item.strip()}
        cases = [
            case for case_index, case in enumerate(cases, 1) if case_index in selected
        ]
    return cases


def main():
    global BENCH_ACTIVE, BENCH_WARMUP

    args = parse_args()
    BENCH_WARMUP = args.warmup
    BENCH_ACTIVE = args.repeat
    results = []
    for case in case_values(load_fa_cases()):
        _attn_fwd2._last_case_search_stats = {}
        results.append(benchmark(*case))

    payload = {
        "case": "fa2",
        "ok": True,
        "triton_ms": [item["triton_time"] * 1000.0 for item in results],
        "search_time": [
            item.get("search_stats", {}).get("bench_time", 0.0) for item in results
        ],
        "bench_configs": [
            item.get("search_stats", {}).get("bench_configs", 0) for item in results
        ],
        "measurements": [
            item.get("search_stats", {}).get("measurements", 0) for item in results
        ],
        "best_config": [item.get("best_config", "") for item in results],
    }
    print(
        "DLC_AUTOTUNE_REGRESSION_RESULT=" + json.dumps(payload, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
