# SPDX-License-Identifier: Apache-2.0
# Exact-copy Triton benchmark/smoke generated from vllm_ascend source kernels.

import torch
from common import assert_close, bench, case_values, dtype_from_name, init, parse_args
from common import tl, triton


def report_smoke(name, triton_ms, models):
    print(f"op: {name}")
    print("torch_ms: smoke-only")
    print(f"triton_ms: {triton_ms:.6f}")
    print("speedup: smoke-only")
    print(f"models: {models}")


from common import safe_exp

MODELS = "FLA chunk gated delta hupdate exact-copy kernel"


@triton.autotune(
    configs=[],
    key=[],
    hints={"search_params": {"params": ["BT"]}},
)
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_hupdate_blockdim64(
    k,
    w,
    g,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int32),
        tl.load(cu_seqlens + i_n + 1).to(tl.int32),
    )
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_k = Hg * K
    stride_w = H * K

    # create b_hupd_bv1 and b_hupd_bv2
    off_hupd_1_top = tl.arange(0, 64)[:, None]
    off_hupd_2_top = tl.arange(0, 64)[None, :]

    # main recurrence
    for i_t in range(NT):
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos + i_h * T_max
        b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        w_base = w + bos * H * K + i_h * K
        # get column-sliced w [BT, 64]
        offs_w_upd1 = tl.arange(0, 64)[None, :]
        mask_w_upd1 = (offs_t_wv < T) & (offs_w_upd1 < K)
        ptr_w_upd1 = w_base + offs_t_wv * stride_w + offs_w_upd1 * 1
        b_w_upd1 = tl.load(ptr_w_upd1, mask=mask_w_upd1, other=0.0).to(tl.float32)

        offs_w_upd2 = 64 + tl.arange(0, 64)[None, :]
        mask_w_upd2 = (offs_t_wv < T) & (offs_w_upd2 < K)
        ptr_w_upd2 = w_base + offs_t_wv * stride_w + offs_w_upd2 * 1
        b_w_upd2 = tl.load(ptr_w_upd2, mask=mask_w_upd2, other=0.0).to(tl.float32)

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        # get row-sliced k [64, T]
        p_k_upd1 = tl.make_block_ptr(
            k_base, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k_upd1 = tl.load(p_k_upd1, boundary_check=(0, 1))
        p_k_upd2 = tl.make_block_ptr(
            k_base, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_k_upd2 = tl.load(p_k_upd2, boundary_check=(0, 1))

        if USE_G:
            b_w_upd1 = b_w_upd1 * b_g[:, None]
            b_w_upd2 = b_w_upd2 * b_g[:, None]

        # compute [64, BT] @ [BT, 64]
        b_hupd_local_11 = (off_hupd_1_top == off_hupd_2_top).to(tl.float32)
        b_hupd_local_22 = (off_hupd_1_top == off_hupd_2_top).to(tl.float32)

        # fp32
        if USE_G:
            b_hupd_local_11 = b_hupd_local_11 * b_g_last
            b_hupd_local_22 = b_hupd_local_22 * b_g_last

        b_hupd_local_11 -= tl.dot(b_k_upd1, b_w_upd1.to(b_k_upd1.dtype))
        b_hupd_local_22 -= tl.dot(b_k_upd2, b_w_upd2.to(b_k_upd2.dtype))
        b_hupd_local_12 = -tl.dot(b_k_upd1, b_w_upd2.to(b_k_upd1.dtype)).to(tl.float32)
        b_hupd_local_21 = -tl.dot(b_k_upd2, b_w_upd1.to(b_k_upd2.dtype)).to(tl.float32)

        hupd_base = h_update + (boh + i_t + i_n) * H * K * K + i_h * K * K
        p_hupd_11 = tl.make_block_ptr(
            hupd_base, (K, K), (K, 1), (0, 0), (64, 64), (1, 0)
        )
        b_hupd_11 = tl.load(p_hupd_11, boundary_check=(1, 0))
        p_hupd_21 = tl.make_block_ptr(
            hupd_base, (K, K), (K, 1), (64, 0), (64, 64), (1, 0)
        )
        b_hupd_21 = tl.load(p_hupd_21, boundary_check=(1, 0))
        p_hupd_12 = tl.make_block_ptr(
            hupd_base, (K, K), (K, 1), (0, 64), (64, 64), (1, 0)
        )
        b_hupd_12 = tl.load(p_hupd_12, boundary_check=(1, 0))
        p_hupd_22 = tl.make_block_ptr(
            hupd_base, (K, K), (K, 1), (64, 64), (64, 64), (1, 0)
        )
        b_hupd_22 = tl.load(p_hupd_22, boundary_check=(1, 0))

        b_hupd11_new = tl.dot(b_hupd_local_11.to(b_hupd_11.dtype), b_hupd_11).to(
            tl.float32
        )
        b_hupd11_new += tl.dot(b_hupd_local_12.to(b_hupd_21.dtype), b_hupd_21)

        b_hupd21_new = tl.dot(b_hupd_local_21.to(b_hupd_11.dtype), b_hupd_11).to(
            tl.float32
        )
        b_hupd21_new += tl.dot(b_hupd_local_22.to(b_hupd_21.dtype), b_hupd_21)

        b_hupd12_new = tl.dot(b_hupd_local_11.to(b_hupd_12.dtype), b_hupd_12).to(
            tl.float32
        )
        b_hupd12_new += tl.dot(b_hupd_local_12.to(b_hupd_22.dtype), b_hupd_22)

        b_hupd22_new = tl.dot(b_hupd_local_21.to(b_hupd_12.dtype), b_hupd_12).to(
            tl.float32
        )
        b_hupd22_new += tl.dot(b_hupd_local_22.to(b_hupd_22.dtype), b_hupd_22)

        hupd_next = h_update + (boh + i_t + i_n + 1) * H * K * K + i_h * K * K
        p_hupd_11 = tl.make_block_ptr(
            hupd_next, (K, K), (K, 1), (0, 0), (64, 64), (1, 0)
        )
        tl.store(
            p_hupd_11,
            b_hupd11_new.to(p_hupd_11.dtype.element_ty),
            boundary_check=(0, 1),
        )

        p_hupd_21 = tl.make_block_ptr(
            hupd_next, (K, K), (K, 1), (64, 0), (64, 64), (1, 0)
        )
        tl.store(
            p_hupd_21,
            b_hupd21_new.to(p_hupd_21.dtype.element_ty),
            boundary_check=(0, 1),
        )

        p_hupd_12 = tl.make_block_ptr(
            hupd_next, (K, K), (K, 1), (0, 64), (64, 64), (1, 0)
        )
        tl.store(
            p_hupd_12,
            b_hupd12_new.to(p_hupd_12.dtype.element_ty),
            boundary_check=(0, 1),
        )

        p_hupd_22 = tl.make_block_ptr(
            hupd_next, (K, K), (K, 1), (64, 64), (64, 64), (1, 0)
        )
        tl.store(
            p_hupd_22,
            b_hupd22_new.to(p_hupd_22.dtype.element_ty),
            boundary_check=(0, 1),
        )


def launch(k, w, g):
    b, t, hg, kk = k.shape
    h = g.shape[1]
    bt = 64
    nt = triton.cdiv(t, bt)
    h_update = torch.zeros((b, nt + b, h, kk, kk), device=k.device, dtype=torch.float32)
    eye = torch.eye(kk, device=k.device, dtype=torch.float32)
    h_update[:, 0, :, :, :] = eye
    cu_seqlens = torch.arange(0, (b + 1) * t, t, device=k.device, dtype=torch.int64)
    chunk_offsets = torch.arange(0, b * nt, nt, device=k.device, dtype=torch.int32)
    chunk_gated_delta_rule_fwd_kernel_hupdate_blockdim64[(nt, b * h)](
        k,
        w,
        g,
        cu_seqlens,
        chunk_offsets,
        h_update,
        t,
        h,
        hg,
        kk,
    )
    return h_update


def main():
    args = parse_args()
    init(args.seed)
    dtype = dtype_from_name(args.dtype)
    for b, t, h, hg, kk in case_values(
        ((1, 64, 1, 1, 128), (2, 128, 4, 2, 128), (4, 256, 8, 4, 128))
    ):
        k = torch.randn(b, t, hg, kk, device=args.device, dtype=dtype)
        w = torch.randn(b, t, hg, kk, device=args.device, dtype=dtype)
        g = torch.randn(b, h, t, device=args.device, dtype=torch.float32)
        launch(k, w, g)
        triton_ms, _ = bench(lambda: launch(k, w, g), args.warmup, args.repeat)
        report_smoke(
            f"chunk_gated_delta_rule_fwd_kernel_hupdate_blockdim64 shape=({b}, {t}, {h}, {hg}, {kk})",
            triton_ms,
            MODELS,
        )


def test_benchmark():
    main()


if __name__ == "__main__":
    main()
