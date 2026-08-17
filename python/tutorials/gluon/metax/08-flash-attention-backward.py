"""Single-kernel dense FlashAttention backward for MetaX C500.

This is the D128/BM32/BN128 backward main kernel.
The device kernel reconstructs P from forward LSE, forms dP/dS, atomically
accumulates dQ, and keeps dK/dV accumulators across the reverse-M traversal.
Q/dO use an explicit C500 GVM/shared synchronization pipeline.  No global P,
dP, or dS workspace is materialized.

Like the C kernel, dQ atomics target fp32 storage with SQ rounded to 128 and
physical [B,SQ128,H,D] order.  The kernel receives a logical BHSD view, so all
eight atomic transactions per thread remain unpredicated on the tail tile.

The selected configuration uses an N128 tile and the same eight C500 warps as
the reference kernel.

The dQ path consumes one full D128 operand from the untransposed K shared view
and lowers one logical GEMM. BSM permutation remains only on the packed dV/dK
B operands that use that target-specific fragment contract.
"""

import argparse
import math
import re

import torch
import triton
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
)

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


BM = 32
BN = 128
FRAGMENT_N = BN
D = 128
DQ_ACCUM_ALIGNMENT = 128
LOG2E = 1.4426950408889634
BATCH = 1
HEADS = 32
TEST_SEQ_LENS = (1024, 1023)
FLASH_ATTN_MAIN_KERNEL_PATTERN = (
    "flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel"
)


@gluon.jit(enable_gluon_layout_autotune=True)
def flash_bwd_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    dpsum_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    SQ,
    SK,
    H: gl.constexpr,
    q_stride_b,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_n,
    v_stride_d,
    do_stride_b,
    do_stride_h,
    do_stride_m,
    do_stride_d,
    lse_stride_b,
    lse_stride_h,
    lse_stride_m,
    dpsum_stride_b,
    dpsum_stride_h,
    dpsum_stride_m,
    dq_stride_b,
    dq_stride_h,
    dq_stride_m,
    dq_stride_d,
    dk_stride_b,
    dk_stride_h,
    dk_stride_n,
    dk_stride_d,
    dv_stride_b,
    dv_stride_h,
    dv_stride_n,
    dv_stride_d,
    scale,
    scale_log2,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    N_FRAGMENT: gl.constexpr,
):
    """Fuse the complete backward main while sweeping N fragments.

    One 64 KiB Shared arena follows the explicit 32x128 kernel's eight-slot
    epoch. Dot-facing roles remain independent logical roots so layout
    propagation does not conflate physical reuse with layout equivalence.
    """
    pid_n = gl.program_id(0)
    pid_bh = gl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    num_m_blocks = gl.cdiv(SQ, BLOCK_M)

    # Dot accumulator and operand layouts are intentionally left implicit.
    # The compiler derives complete target assignments from tile ownership,
    # operand continuity, BSM contracts, and the selected MMA capabilities.

    shared_storage = gl.storage_alias_spec(storage=gl.storage_kind.smem)

    # Prologue and epilogue: K/V occupy two consecutive 32 KiB halves.
    kv_tt_smem_alloc = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_D],
        num_buffers=2,
        reuse=shared_storage,
    )
    kv_tn_smem_alloc = gl.local_alloc(
        k_ptr.dtype.element_ty,
        [BLOCK_D, BLOCK_N],
        num_buffers=2,
        reuse=shared_storage,
    )

    # Reverse-M epoch: eight 8 KiB slots share the same bytes as K/V.
    # Separate roots retain independent Shared-layout constraints while the
    # leading buffer dimension fixes each role's physical slot.
    tile_epoch = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_M, BLOCK_D],
        num_buffers=4,
        reuse=shared_storage,
    )
    raw_epoch = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_M, BLOCK_D],
        num_buffers=5,
        reuse=shared_storage,
    )
    ds_direct_epoch = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_M, BLOCK_N],
        num_buffers=6,
        reuse=shared_storage,
    )
    ds_transposed_epoch = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_M],
        num_buffers=7,
        reuse=shared_storage,
    )
    p_transposed_epoch = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_M],
        num_buffers=8,
        reuse=shared_storage,
    )

    k_tt_smem = kv_tt_smem_alloc.index(0)
    v_smem = kv_tt_smem_alloc.index(1)
    k_tn_smem = kv_tn_smem_alloc.index(0)
    v_tn_smem = kv_tn_smem_alloc.index(1)
    q_smem = tile_epoch.index(0)
    do_smem = tile_epoch.index(3)
    do_raw_smem = raw_epoch.index(4)
    ds_direct_smem = ds_direct_epoch.index(5)
    ds_transposed_smem = ds_transposed_epoch.index(6)
    p_transposed_smem = p_transposed_epoch.index(7)
    # Keep the dot-facing [N,M] allocations directly lowerable. Loading through
    # memdesc_trans currently inserts an 8 KiB C500 shared staging buffer, so
    # the permuted [M,N] views are deliberately confined to stores.
    ds_transposed_writer = ds_transposed_smem.permute([1, 0])
    p_transposed_writer = p_transposed_smem.permute([1, 0])

    q_head = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    k_head = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h
    v_head = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h
    do_head = do_ptr + pid_b * do_stride_b + pid_h * do_stride_h
    lse_head = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h
    dpsum_head = (
        dpsum_ptr + pid_b * dpsum_stride_b + pid_h * dpsum_stride_h
    )
    dq_head = dq_ptr + pid_b * dq_stride_b + pid_h * dq_stride_h
    dk_head = dk_ptr + pid_b * dk_stride_b + pid_h * dk_stride_h
    dv_head = dv_ptr + pid_b * dv_stride_b + pid_h * dv_stride_h

    # Keep the fragment loop explicit even though the selected C-matching
    # configuration executes it once with N_FRAGMENT == BLOCK_N == 128.
    for fragment in range(0, BLOCK_N // N_FRAGMENT):
        vt_d = gl.arange(0, BLOCK_D)
        vt_n = (
            pid_n * BLOCK_N
            + fragment * N_FRAGMENT
            + gl.arange(0, N_FRAGMENT)
        )
        # K/V are fixed for this N fragment and dominate the reverse-M loop.
        # The D128 path loads K once as [N,D]. One physical storage slot serves
        # the direct dQ role and the transposed score role through separate
        # logical views, both materialized before the reverse-M loop.
        vt_ptrs = (
            v_head
            + vt_d[:, None] * v_stride_d
            + vt_n[None, :] * v_stride_n
        )
        vt_mask = vt_n[None, :] < SK
        vt = gl.load(vt_ptrs, mask=vt_mask, other=0.0)
        # V and K occupy disjoint prologue slots. Materialize V immediately so
        # its blocked global-load fragment does not overlap the K load.
        v_tn_smem.store(vt)
        k_n = (
            pid_n * BLOCK_N
            + fragment * N_FRAGMENT
            + gl.arange(0, N_FRAGMENT)
        )
        k_d = gl.arange(0, BLOCK_D)
        k_ptrs = (
            k_head
            + k_n[:, None] * k_stride_n
            + k_d[None, :] * k_stride_d
        )
        k_register = gl.load(k_ptrs, mask=k_n[:, None] < SK, other=0.0)

        # Match the C prologue: retain the score/dP K/V fragments, then reuse
        # the K bytes for the complete dQ B fragment contract.
        k_tn_smem.store(k_register.permute(1, 0))
        gl.metax.barrier_shared()
        kt_for_score = k_tn_smem.load()
        vt_for_dp = v_tn_smem.load()
        gl.metax.barrier_shared()

        k_tt_smem.store(k_register)
        gl.metax.barrier_shared()
        k_for_dq = k_tt_smem.load()
        # Both K/V consumer fragments are resident before the C shared union
        # is reused by the initial Q and dO async copies.
        gl.metax.barrier_shared()

        acc_dk = gl.zeros((N_FRAGMENT, BLOCK_D), gl.float32)
        acc_dv = gl.zeros((N_FRAGMENT, BLOCK_D), gl.float32)

        initial_m_block = num_m_blocks - 1
        initial_m = initial_m_block * BLOCK_M + gl.arange(0, BLOCK_M)
        initial_d = gl.arange(0, BLOCK_D)
        initial_q_ptrs = (
            q_head
            + initial_m[:, None] * q_stride_m
            + initial_d[None, :] * q_stride_d
        )
        initial_mask = initial_m[:, None] < SQ
        initial_q_raw_smem = raw_epoch.index(
            1 + initial_m_block % 2
        )
        gl.metax.async_copy_global_to_shared(
            initial_q_raw_smem,
            initial_q_ptrs,
            mask=initial_mask,
            other=0.0,
        )
        initial_do_ptrs = (
            do_head
            + initial_m[:, None] * do_stride_m
            + initial_d[None, :] * do_stride_d
        )
        gl.metax.async_copy_global_to_shared(
            do_raw_smem,
            initial_do_ptrs,
            mask=initial_mask,
            other=0.0,
        )
        # Drain the initial Q/dO copy groups before entering the reverse-M
        # loop. Later iterations retain the eight dQ atomics at the queue tail.
        gl.metax.gvm_arrive(0)
        for m_iteration in range(0, num_m_blocks):
            m_block = num_m_blocks - 1 - m_iteration
            q_raw_smem = raw_epoch.index(1 + m_block % 2)

            gl.metax.gvm_arrive(8)
            gl.metax.barrier_shared()

            # C SWIZZLE_STORE_QDO: raw Qt -> registers -> Q consumer shared.
            q_stage = q_raw_smem.load()
            q_smem.store(q_stage)
            gl.metax.barrier_shared()

            # Reconstruct P first so score and dP can reuse one fp32 boundary.
            q_score = q_smem.load()
            score = gl.dot(
                q_score,
                kt_for_score,
                gl.zeros(
                    (BLOCK_M, N_FRAGMENT),
                    gl.float32,
                ),
                input_precision="tf32",
            )

            # The score is the last consumer of the swizzled Q tile. Queue
            # Q[m-1] into the alternate raw slot while current Q remains live
            # for dK in its own slot.
            next_q_m = (m_block - 1) * BLOCK_M + gl.arange(0, BLOCK_M)
            next_q_d = gl.arange(0, BLOCK_D)
            next_q_ptrs = (
                q_head
                + next_q_m[:, None] * q_stride_m
                + next_q_d[None, :] * q_stride_d
            )
            next_q_mask = next_q_m[:, None] < SQ
            if m_block > 0:
                next_q_raw_smem = raw_epoch.index(
                    1 + (m_block - 1) % 2
                )
                gl.metax.async_copy_global_to_shared(
                    next_q_raw_smem,
                    next_q_ptrs,
                    mask=next_q_mask,
                    other=0.0,
                )

            lse_m = m_block * BLOCK_M + gl.arange(0, BLOCK_M)
            lse = gl.load(
                lse_head + lse_m * lse_stride_m,
                mask=lse_m < SQ,
                other=0.0,
            ).to(gl.float32)
            dpsum_m = m_block * BLOCK_M + gl.arange(0, BLOCK_M)
            dpsum = gl.load(
                dpsum_head + dpsum_m * dpsum_stride_m,
                mask=dpsum_m < SQ,
                other=0.0,
            ).to(gl.float32)

            # Materialize the dot-facing dO tile before publishing P^T.
            do_stage = do_raw_smem.load()
            do_smem.store(do_stage)

            p = gl.exp2(
                score * scale_log2
                - lse[:, None] * 1.4426950408889634
            )
            point_m = m_block * BLOCK_M + gl.arange(0, BLOCK_M)
            point_n = (
                pid_n * BLOCK_N
                + fragment * N_FRAGMENT
                + gl.arange(0, N_FRAGMENT)
            )
            point_mask = (point_m[:, None] < SQ) & (point_n[None, :] < SK)
            p = gl.where(point_mask, p, 0.0)
            p_element = p.to(q_ptr.dtype.element_ty)

            # P^T occupies epoch slot 7 and shares dO's publication point.
            p_transposed_writer.store(p_element)
            gl.metax.barrier_shared()

            do_for_dp = do_smem.load()
            dp = gl.dot(
                do_for_dp,
                vt_for_dp,
                gl.zeros(
                    (BLOCK_M, N_FRAGMENT),
                    gl.float32,
                ),
                input_precision="tf32",
            )

            # Retain the current P^T and raw dO fragments before forming dS.
            pt = p_transposed_smem.load()
            do_for_dv_raw = do_raw_smem.load()
            ds_element = (
                p * (dp - dpsum[:, None])
            ).to(q_ptr.dtype.element_ty)

            ds_direct_smem.store(ds_element)
            ds_transposed_writer.store(ds_element)

            do_for_dv = gl.metax.bsm_perm(do_for_dv_raw)
            acc_dv = gl.dot(pt, do_for_dv, acc_dv, input_precision="tf32")
            gl.metax.barrier_shared()

            # Queue dO[m-1] only after every current-dO reader has completed.
            if m_block > 0:
                next_do_m = (
                    (m_block - 1) * BLOCK_M + gl.arange(0, BLOCK_M)
                )
                next_do_d = gl.arange(0, BLOCK_D)
                next_do_ptrs = (
                    do_head
                    + next_do_m[:, None] * do_stride_m
                    + next_do_d[None, :] * do_stride_d
                )
                next_do_mask = next_do_m[:, None] < SQ
                gl.metax.async_copy_global_to_shared(
                    do_raw_smem,
                    next_do_ptrs,
                    mask=next_do_mask,
                    other=0.0,
                )

            # Match the explicit 32x128 order: dQ consumes direct dS, then the
            # current raw-Q fragment crosses the atomic before dK consumes dS^T.
            ds_for_dq = ds_direct_smem.load()
            dq = gl.zeros(
                (BLOCK_M, BLOCK_D),
                gl.float32,
            )
            dq = gl.dot(ds_for_dq, k_for_dq, dq, input_precision="tf32")
            dq_m = m_block * BLOCK_M + gl.arange(
                0,
                BLOCK_M,
            )
            dq_d = gl.arange(
                0,
                BLOCK_D,
            )
            dq_ptrs = (
                dq_head
                + dq_m[:, None] * dq_stride_m
                + dq_d[None, :] * dq_stride_d
            )
            dq_scaled = dq * scale

            q_for_dk_raw = q_raw_smem.load()
            gl.atomic_add(
                dq_ptrs,
                dq_scaled,
                sem="relaxed",
            )
            dst = ds_transposed_smem.load()
            q_for_dk = gl.metax.bsm_perm(q_for_dk_raw)
            acc_dk = gl.dot(dst, q_for_dk, acc_dk, input_precision="tf32")

        dk_element = (acc_dk * scale).to(dk_ptr.dtype.element_ty)
        dv_element = acc_dv.to(dv_ptr.dtype.element_ty)

        # Match C lines 719-735: reuse the disjoint K/V halves for dK/dV in
        # parallel. Outstanding dQ atomics remain ordered in the GVM FIFO and
        # are not unnecessarily drained before the epilogue.
        gl.metax.barrier_shared()
        k_tt_smem.store(dk_element)
        v_smem.store(dv_element)
        gl.metax.barrier_shared()

        out_n = (
            pid_n * BLOCK_N
            + fragment * N_FRAGMENT
            + gl.arange(0, N_FRAGMENT)
        )
        out_d = gl.arange(0, BLOCK_D)
        dk_ptrs = (
            dk_head
            + out_n[:, None] * dk_stride_n
            + out_d[None, :] * dk_stride_d
        )
        dk_out = k_tt_smem.load()
        gl.store(dk_ptrs, dk_out, mask=out_n[:, None] < SK)

        dv_ptrs = (
            dv_head
            + out_n[:, None] * dv_stride_n
            + out_d[None, :] * dv_stride_d
        )
        dv_out = v_smem.load()
        gl.store(dv_ptrs, dv_out, mask=out_n[:, None] < SK)


def _validate_inputs(q, k, v, do, lse, dpsum):
    batch, heads, seqlen_q, head_dim = q.shape
    if head_dim != D:
        raise ValueError(f"head dimension must be {D}, got {head_dim}")
    if k.shape[:2] != (batch, heads) or k.shape[3] != D:
        raise ValueError(f"invalid K shape {k.shape} for Q shape {q.shape}")
    seqlen_k = k.shape[2]
    if v.shape != (batch, heads, seqlen_k, D):
        raise ValueError(f"invalid V shape {v.shape}")
    if do.shape != q.shape:
        raise ValueError(f"invalid dO shape {do.shape}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported dtype {q.dtype}")
    if any(t.dtype != q.dtype for t in (k, v, do)):
        raise TypeError("Q/K/V/dO must share one low-precision dtype")
    if any(t.stride(-1) != 1 for t in (q, k, v, do)):
        raise ValueError("Q/K/V/dO must be contiguous in the head dimension")
    if lse.shape != (batch, heads, seqlen_q) or lse.dtype != torch.float32:
        raise ValueError("LSE must be contiguous fp32 [B,H,SQ]")
    if dpsum.shape != lse.shape or dpsum.dtype != torch.float32:
        raise ValueError("dPsum must be contiguous fp32 [B,H,SQ]")
    if not lse.is_contiguous() or not dpsum.is_contiguous():
        raise ValueError("LSE and dPsum must be contiguous")
    return batch, heads, seqlen_q, seqlen_k


def launch_flash_bwd_kernel(
    q,
    k,
    v,
    do,
    lse,
    dpsum,
    dq_accum,
    dk,
    dv,
    scale=None,
):
    """Launch one backward kernel into preallocated outputs.

    ``dq_accum`` is a logical BHSD view of zeroed physical
    [B,SQ128,H,D] storage. Multiple N tiles atomically accumulate into it.
    """
    batch, heads, seqlen_q, seqlen_k = _validate_inputs(
        q, k, v, do, lse, dpsum
    )
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    dq_seqlen = (
        triton.cdiv(seqlen_q, DQ_ACCUM_ALIGNMENT) * DQ_ACCUM_ALIGNMENT
    )
    if dq_accum.shape != (batch, heads, dq_seqlen, D):
        raise ValueError("dQ accumulator must be logical fp32 [B,H,SQ128,D]")
    if dq_accum.dtype != torch.float32:
        raise ValueError("dQ accumulator must be fp32")
    if dq_accum.stride() != (
        dq_seqlen * heads * D,
        D,
        heads * D,
        1,
    ):
        raise ValueError("dQ accumulator must be a BHSD view of BSHD storage")
    if dk.shape != k.shape or dk.dtype != k.dtype:
        raise ValueError("dK output must match K")
    if dv.shape != v.shape or dv.dtype != v.dtype:
        raise ValueError("dV output must match V")
    if any(t.stride(-1) != 1 for t in (dq_accum, dk, dv)):
        raise ValueError("dQ/dK/dV must be contiguous in the head dimension")
    grid = (triton.cdiv(seqlen_k, BN), batch * heads)
    flash_bwd_fused_kernel[grid](
        q,
        k,
        v,
        do,
        lse,
        dpsum,
        dq_accum,
        dk,
        dv,
        seqlen_q,
        seqlen_k,
        heads,
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
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        dpsum.stride(0),
        dpsum.stride(1),
        dpsum.stride(2),
        dq_accum.stride(0),
        dq_accum.stride(1),
        dq_accum.stride(2),
        dq_accum.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        float(scale),
        float(scale * LOG2E),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_D=D,
        N_FRAGMENT=FRAGMENT_N,
        num_warps=8,
        num_stages=2,
        num_ctas=1,
        pipeline="cpasync",
    )


def _make_test_inputs(seq_len, dtype):
    torch.manual_seed(0)
    shape = (BATCH, HEADS, seq_len, D)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    do = torch.randn(shape, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, do, scale


def _prepare_flash_attn_backward(q, k, v, do, scale):
    # flash_attn uses [B, S, H, D]; layout conversion and forward preparation
    # stay outside the measured backward region. The forward output supplies
    # LSE and dPsum for Gluon without materializing any [B, H, S, S] tensor.
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()
    do_fa = do.transpose(1, 2).contiguous()
    (
        _,
        q_fa,
        k_fa,
        v_fa,
        out_fa,
        lse_fa,
        _,
        rng_state,
        _,
    ) = _flash_attn_forward(
        q_fa,
        k_fa,
        v_fa,
        0.0,
        scale,
        False,
        (-1, -1),
    )
    lse = lse_fa.contiguous()
    dpsum = (
        (do_fa.float() * out_fa.float())
        .sum(dim=-1)
        .transpose(1, 2)
        .contiguous()
    )
    dq_fa = torch.empty_like(q_fa)
    dk_fa = torch.empty_like(k_fa)
    dv_fa = torch.empty_like(v_fa)

    def launch():
        _flash_attn_backward(
            do_fa,
            q_fa,
            k_fa,
            v_fa,
            out_fa,
            lse_fa,
            dq_fa,
            dk_fa,
            dv_fa,
            0.0,
            scale,
            False,
            (-1, -1),
            rng_state=rng_state,
        )

    return lse, dpsum, launch, (dq_fa, dk_fa, dv_fa)


def _profile_flash_attn_main_kernel(launch, repetitions):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as profiler:
        for _ in range(repetitions):
            launch()
        torch.cuda.synchronize()

    matches = [
        event
        for event in profiler.key_averages(group_by_input_shape=False)
        if FLASH_ATTN_MAIN_KERNEL_PATTERN in event.key
    ]
    if len(matches) != 1:
        names = [event.key for event in matches]
        raise RuntimeError(
            "expected exactly one FlashAttention main-kernel profile entry, "
            f"got {len(matches)}: {names}"
        )
    event = matches[0]
    if event.count != repetitions:
        raise RuntimeError(
            "unexpected FlashAttention main-kernel launch count: "
            f"expected {repetitions}, got {event.count}"
        )
    # torch.profiler reports device_time_total in microseconds.
    average_ms = event.device_time_total / event.count / 1000.0
    return average_ms, event.key


def _flash_kernel_description(kernel_name):
    match = re.search(
        r"Flash_bwd_kernel_traits<(\d+),\s*(\d+),\s*(\d+),\s*(\d+),",
        kernel_name,
    )
    if match is None:
        return FLASH_ATTN_MAIN_KERNEL_PATTERN
    head_dim, block_m, block_n, num_warps = match.groups()
    return f"hdim{head_dim}_{block_m}x{block_n}_{num_warps}warps"


def _measure_case(seq_len, dtype, warmup, rep, profile_rep):
    q, k, v, do, scale = _make_test_inputs(seq_len, dtype)
    lse, dpsum, flash_launch, flash_outputs = _prepare_flash_attn_backward(
        q, k, v, do, scale
    )
    dq_seqlen = (
        triton.cdiv(seq_len, DQ_ACCUM_ALIGNMENT) * DQ_ACCUM_ALIGNMENT
    )

    def allocate_outputs():
        dq_storage = torch.zeros(
            (BATCH, dq_seqlen, HEADS, D),
            device=q.device,
            dtype=torch.float32,
        )
        return (
            dq_storage.permute(0, 2, 1, 3),
            torch.empty_like(k),
            torch.empty_like(v),
        )

    # Autotuning executes every candidate multiple times, matching Triton's
    # native autotuner semantics. Use disposable outputs for that first call;
    # the selected kernel then runs once on the outputs checked below.
    tune_dq, tune_dk, tune_dv = allocate_outputs()
    launch_flash_bwd_kernel(
        q, k, v, do, lse, dpsum, tune_dq, tune_dk, tune_dv, scale
    )

    dq_accum, dk, dv = allocate_outputs()
    launch = lambda: launch_flash_bwd_kernel(
        q, k, v, do, lse, dpsum, dq_accum, dk, dv, scale
    )
    launch()
    flash_launch()
    gluon_outputs = (dq_accum[:, :, :seq_len].to(dtype), dk, dv)
    flash_outputs = tuple(x.transpose(1, 2) for x in flash_outputs)
    result = {}
    for name, actual, reference in zip(
        ("dq", "dk", "dv"), gluon_outputs, flash_outputs
    ):
        error = (actual - reference).abs()
        result[f"{name}_max_abs"] = error.max().item()
        result[f"{name}_allclose"] = torch.allclose(
            actual, reference, atol=2e-2, rtol=2e-2
        )
    result["allclose"] = all(
        result[f"{name}_allclose"] for name in ("dq", "dk", "dv")
    )

    result["gluon_ms"] = triton.testing.do_bench(
        launch, warmup=warmup, rep=rep
    )
    result["flash_main_ms"], flash_kernel_name = (
        _profile_flash_attn_main_kernel(flash_launch, profile_rep)
    )
    result["flash_main_kernel"] = _flash_kernel_description(
        flash_kernel_name
    )
    result["gluon_main_speedup"] = (
        result["flash_main_ms"] / result["gluon_ms"]
    )
    return result


def _print_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")


def run_cases(seq_lens, dtype, warmup, rep, profile_rep):
    print(
        "Timing scope: Gluon main uses precomputed LSE+dPsum (1 kernel); "
        "flash_attn main is extracted from the backward call with "
        f"torch.profiler ({profile_rep} profiled launches)."
    )
    rows = []
    for seq_len in seq_lens:
        result = _measure_case(seq_len, dtype, warmup, rep, profile_rep)
        print(
            f"Flash main dispatch for S={seq_len}: "
            f"{result['flash_main_kernel']}"
        )
        rows.append(
            [
                f"{BATCH}x{HEADS}x{seq_len}x{D}",
                dtype,
                f"{result['gluon_ms']:.6f}",
                f"{result['flash_main_ms']:.6f}",
                f"{result['gluon_main_speedup']:.3f}",
                f"{result['dq_max_abs']:.6g}",
                f"{result['dk_max_abs']:.6g}",
                f"{result['dv_max_abs']:.6g}",
                result["allclose"],
            ]
        )
    _print_table(
        [
            "BxHxSxD",
            "dtype",
            "gluon_main_ms",
            "flash_main_ms(profile)",
            "gluon_speedup_vs_flash_main",
            "dq_max_abs",
            "dk_max_abs",
            "dv_max_abs",
            "gluon_vs_flash_ok",
        ],
        rows,
    )
    if not all(row[-1] for row in rows):
        raise AssertionError(f"single-kernel accuracy failure: {rows}")


def _parse_seq_len(value):
    seq_len = int(value)
    return seq_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--profile-rep", type=int, default=25)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--seq-len",
        nargs="+",
        type=_parse_seq_len,
        default=TEST_SEQ_LENS,
        help="sequence lengths, each at least 512",
    )
    args = parser.parse_args()
    if args.profile_rep < 1:
        parser.error("--profile-rep must be at least 1")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    run_cases(
        tuple(dict.fromkeys(args.seq_len)),
        dtype,
        args.warmup,
        args.rep,
        args.profile_rep,
    )
