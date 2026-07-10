import torch
import triton
import triton.language as tl
from common import assert_close, bench, case_values, init, parse_args, report

MODELS = "Qwen3-Next and Qwen3.5 GDN metadata paths; kernel coverage does not depend on model availability"


@triton.autotune(
    configs=[],
    key=[],
    hints={"search_params": {"params": ["BLOCK_SIZE"]}},
)
@triton.jit
def _build_chunk_counts_kernel(
    cu_seqlens_ptr,
    chunk_counts_ptr,
    num_seqs: tl.constexpr,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_seqs

    bos = tl.load(cu_seqlens_ptr + offsets, mask=mask, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + offsets + 1, mask=mask, other=0).to(tl.int32)
    seq_lens = eos - bos
    chunk_counts = (seq_lens + chunk_size - 1) // chunk_size

    tl.store(chunk_counts_ptr + offsets, chunk_counts, mask=mask)


@triton.autotune(
    configs=[],
    key=[],
    hints={"search_params": {"params": ["BLOCK_SIZE"]}},
)
@triton.jit
def _build_chunk_offsets_kernel(
    chunk_counts_ptr,
    out_offsets_ptr,
    num_seqs: tl.constexpr,
    ADD_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets <= num_seqs
    prefix = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for seq_idx in range(0, num_seqs):
        chunk_count = tl.load(
            chunk_counts_ptr + seq_idx, mask=seq_idx < num_seqs, other=0
        ).to(tl.int32)
        prefix += tl.where(mask & (offsets > seq_idx), chunk_count + ADD_ONE, 0)

    tl.store(
        out_offsets_ptr + offsets,
        prefix.to(out_offsets_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.autotune(
    configs=[],
    key=[],
    hints={"search_params": {"params": ["BLOCK_SIZE"]}},
)
@triton.jit
def _build_final_chunk_indices_kernel(
    update_chunk_offsets_ptr,
    out_final_chunk_indices_ptr,
    num_seqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_seqs
    final_indices = (
        tl.load(update_chunk_offsets_ptr + offsets + 1, mask=mask, other=0).to(tl.int32)
        - 1
    )
    tl.store(
        out_final_chunk_indices_ptr + offsets,
        final_indices.to(out_final_chunk_indices_ptr.dtype.element_ty),
        mask=mask,
    )


def run_triton(cu_seqlens, chunk_size):
    num_seqs = cu_seqlens.numel() - 1
    chunk_counts = torch.empty(num_seqs, dtype=torch.int32, device=cu_seqlens.device)
    chunk_offsets = torch.empty(
        num_seqs + 1, dtype=torch.int32, device=cu_seqlens.device
    )
    update_offsets = torch.empty(
        num_seqs + 1, dtype=torch.int32, device=cu_seqlens.device
    )
    final_indices = torch.empty(num_seqs, dtype=torch.int32, device=cu_seqlens.device)

    grid_counts = lambda meta: (triton.cdiv(num_seqs, meta["BLOCK_SIZE"]),)
    _build_chunk_counts_kernel[grid_counts](
        cu_seqlens, chunk_counts, num_seqs, chunk_size
    )

    grid_offsets = lambda meta: (triton.cdiv(num_seqs + 1, meta["BLOCK_SIZE"]),)
    _build_chunk_offsets_kernel[grid_offsets](
        chunk_counts, chunk_offsets, num_seqs, ADD_ONE=0
    )
    _build_chunk_offsets_kernel[grid_offsets](
        chunk_counts, update_offsets, num_seqs, ADD_ONE=1
    )

    grid_final = lambda meta: (triton.cdiv(num_seqs, meta["BLOCK_SIZE"]),)
    _build_final_chunk_indices_kernel[grid_final](
        update_offsets, final_indices, num_seqs
    )
    return chunk_counts, chunk_offsets, update_offsets, final_indices


def torch_ref(cu_seqlens, chunk_size):
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.div(seq_lens + chunk_size - 1, chunk_size, rounding_mode="floor").to(
        torch.int32
    )
    chunk_offsets = torch.empty(
        cu_seqlens.numel(), dtype=torch.int32, device=cu_seqlens.device
    )
    update_offsets = torch.empty_like(chunk_offsets)
    chunk_offsets[0] = 0
    update_offsets[0] = 0
    chunk_offsets[1:] = torch.cumsum(counts, dim=0)
    update_offsets[1:] = torch.cumsum(counts + 1, dim=0)
    final_indices = update_offsets[1:] - 1
    return counts, chunk_offsets, update_offsets, final_indices


def main():
    args = parse_args()
    init(args.seed)
    for num_seqs, max_len, chunk_size in case_values(
        ((256, 2048, 64), (1024, 4096, 64), (2048, 8192, 128))
    ):
        lens = torch.randint(
            1, max_len, (num_seqs,), dtype=torch.int32, device=args.device
        )
        cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=args.device)
        cu_seqlens[1:] = torch.cumsum(lens, dim=0)
        tri = run_triton(cu_seqlens, chunk_size)
        ref = torch_ref(cu_seqlens, chunk_size)
        for actual, expected in zip(tri, ref):
            assert_close("gdn_chunk_meta", actual, expected, 0, 0)
        torch_ms, _ = bench(
            lambda: torch_ref(cu_seqlens, chunk_size), args.warmup, args.repeat
        )
        triton_ms, _ = bench(
            lambda: run_triton(cu_seqlens, chunk_size), args.warmup, args.repeat
        )
        report(
            f"gdn_chunk_meta_kernels shape=({num_seqs}, {max_len}, {chunk_size})",
            torch_ms,
            triton_ms,
            MODELS,
        )


def test_benchmark():
    main()


if __name__ == "__main__":
    main()
