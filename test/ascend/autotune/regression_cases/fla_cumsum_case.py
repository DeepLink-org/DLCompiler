import torch
import triton
import triton.language as tl
from common import (
    assert_close,
    bench,
    case_values,
    dtype_from_name,
    init,
    parse_args,
    report,
)

MODELS = "FLA chunk local cumsum exact-copy Triton kernel benchmark"


@triton.autotune(
    configs=[],
    key=[],
    hints={"search_params": {"params": ["NUM_CHUNKS"]}},
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,
    NUM_CHUNKS: tl.constexpr = 1,
):
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    BLOCK_T: tl.constexpr = NUM_CHUNKS * CHUNK_SIZE

    if IS_VARLEN:
        i_s, i_block = (
            tl.load(chunk_indices + i_block * 2).to(tl.int32),
            tl.load(chunk_indices + i_block * 2 + 1).to(tl.int32),
        )
        bos, eos = tl.load(cu_seqlens + i_s).to(tl.int32), tl.load(
            cu_seqlens + i_s + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        ptr_s = tl.make_block_ptr(
            s + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0)
        )
        ptr_o = tl.make_block_ptr(
            o + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0)
        )
        b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (H, NUM_CHUNKS, CHUNK_SIZE))
        b_s = tl.trans(b_s, (2, 0, 1))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (2, 0, 1))
        b_o = tl.reshape(b_o, (H, BLOCK_T))
    else:
        ptr_s = tl.make_block_ptr(
            s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0)
        )
        ptr_o = tl.make_block_ptr(
            o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0)
        )
        b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (NUM_CHUNKS, CHUNK_SIZE, H))
        b_s = tl.trans(b_s, (1, 0, 2))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (1, 0, 2))
        b_o = tl.reshape(b_o, (BLOCK_T, H))

    tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))
    return


def run_triton(
    g, chunk_size, reverse=False, scale=None, head_first=False, output_dtype=None
):
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    g_org, g_out = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = lambda meta: (triton.cdiv(T, meta["NUM_CHUNKS"] * chunk_size), B)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g_out,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        CHUNK_SIZE=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        IS_VARLEN=False,
        num_warps=8,
        num_stages=3,
    )
    return g_out


def torch_ref(g, chunk_size, reverse=False, scale=None, head_first=False):
    if head_first:
        b, h, t = g.shape
        y = g.float().reshape(b, h, t // chunk_size, chunk_size)
        y = (
            torch.flip(torch.cumsum(torch.flip(y, dims=[-1]), dim=-1), dims=[-1])
            if reverse
            else torch.cumsum(y, dim=-1)
        )
        y = y.reshape_as(g)
    else:
        b, t, h = g.shape
        y = g.float().reshape(b, t // chunk_size, chunk_size, h)
        y = (
            torch.flip(torch.cumsum(torch.flip(y, dims=[2]), dim=2), dims=[2])
            if reverse
            else torch.cumsum(y, dim=2)
        )
        y = y.reshape_as(g)
    if scale is not None:
        y = y * scale
    return y.to(g.dtype)


def main():
    args = parse_args()
    init(args.seed)
    dtype = dtype_from_name(args.dtype)
    scale = 0.5
    for B, T, H, chunk_size in case_values(
        ((2, 1024, 32, 64), (4, 2048, 64, 64), (8, 4096, 64, 64))
    ):
        g = torch.randn(B, T, H, dtype=dtype, device=args.device)
        tri = run_triton(
            g,
            chunk_size,
            reverse=False,
            scale=scale,
            head_first=False,
            output_dtype=dtype,
        )
        ref = torch_ref(g, chunk_size, reverse=False, scale=scale, head_first=False)
        assert_close("chunk_local_cumsum_scalar_kernel", tri, ref, args.rtol, args.atol)
        torch_ms, _ = bench(
            lambda: torch_ref(g, chunk_size, False, scale, False),
            args.warmup,
            args.repeat,
        )
        triton_ms, _ = bench(
            lambda: run_triton(g, chunk_size, False, scale, False, dtype),
            args.warmup,
            args.repeat,
        )
        report(
            f"chunk_local_cumsum_scalar_kernel shape=({B}, {T}, {H}, {chunk_size})",
            torch_ms,
            triton_ms,
            MODELS,
        )


def test_benchmark():
    main()


if __name__ == "__main__":
    main()
