import pytest
import torch
import triton
import triton.language as tl

fast_expf = tl.math.exp

@triton.jit
def _silu_and_mul_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """silu and mul kernel."""
    m_id = tl.program_id(0)

    up_ptr = gateup_ptr + N * stride_gun

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    gate_ptrs = gateup_ptr + m_id * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id * stride_om + offs_n * stride_on

    for _ in range(0, N, BLOCK_SIZE_N):
        gate = tl.load(gate_ptrs).to(tl.float32)
        up = tl.load(up_ptrs).to(tl.float32)

        gate = gate / (1 + fast_expf(-gate))
        out = gate * up

        tl.store(out_ptrs, out)

        gate_ptrs += BLOCK_SIZE_N * stride_gun
        up_ptrs += BLOCK_SIZE_N * stride_gun
        out_ptrs += BLOCK_SIZE_N * stride_on


@triton.jit
def _silu_and_mul_no_align_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """silu and mul kernel."""
    m_id = tl.program_id(0)

    up_ptr = gateup_ptr + N * stride_gun

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    gate_ptrs = gateup_ptr + m_id * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id * stride_om + offs_n * stride_on

    for n in range(0, N, BLOCK_SIZE_N):
        mask = n + offs_n < N
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

        gate = gate / (1 + fast_expf(-gate))
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += BLOCK_SIZE_N * stride_gun
        up_ptrs += BLOCK_SIZE_N * stride_gun
        out_ptrs += BLOCK_SIZE_N * stride_on


def silu_and_mul(gate_up: torch.Tensor, out: torch.Tensor = None):
    """silu and mul."""
    assert gate_up.dim() == 2

    M = gate_up.size(0)
    N = gate_up.size(-1) // 2
    if out is None:
        out_shape = (M, N)
        out = gate_up.new_empty(out_shape)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 1024)
    num_warps = 4
    num_stages = 2
    grid = (M, )
    if N % BLOCK_SIZE_N == 0:
        _silu_and_mul_kernel[grid](
            gate_up,
            out,
            N,
            stride_gum=gate_up.stride(0),
            stride_gun=gate_up.stride(1),
            stride_om=out.stride(0),
            stride_on=out.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _silu_and_mul_no_align_kernel[grid](
            gate_up,
            out,
            N,
            stride_gum=gate_up.stride(0),
            stride_gun=gate_up.stride(1),
            stride_om=out.stride(0),
            stride_on=out.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return out



class TestSiluAndMul:

    @pytest.fixture
    def seqlen(self):
        yield 256

    @pytest.fixture
    def feat_size(self, request):
        yield request.param

    @pytest.fixture
    def x(self, seqlen, feat_size):
        yield torch.rand(seqlen, feat_size, dtype=torch.float16, device='npu')

    @pytest.fixture
    def gt(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        yield gate * up

    @pytest.mark.parametrize('feat_size', [4096, 768], indirect=True)
    def test_silu_and_mul(self, x, gt):
        out = silu_and_mul(x)
        torch.testing.assert_close(out, gt)


def _gt(x):
    gate, up = x.chunk(2, -1)
    gate = torch.nn.functional.silu(gate)
    return gate * up


def _test_silu_and_mul(x):
    return silu_and_mul(x)


def test():
    seqlen = 256
    feat_size = 4096
    x = torch.rand(seqlen, feat_size, dtype=torch.float16, device='npu')

    gt = _gt(x)
    tt = _test_silu_and_mul(x)

    print('max diff', (gt - tt).abs().max())

    # configs = []
    # configs.append(
    #     triton.testing.Benchmark(
    #         x_names=['op'],
    #         x_vals=['fwd'],
    #         line_arg='provider',
    #         line_vals=['triton', 'pytorch'],
    #         line_names=['Triton', 'PyTorch'],
    #         ylabel='ms',
    #         plot_name='',
    #         args={},
    #     ))

    # @triton.testing.perf_report(configs)
    # def bench_fn(op, provider, device='npu'):
    #     warmup = 100
    #     rep = 200

    #     if 'triton' in provider:
    #         # fn = lambda: test_paged_attention(conti_q, blocked_kv, block_offsets, start_loc, seq_lens, history_lens, feat_dim_v)
    #         fn = lambda: silu_and_mul(x)
    #     if 'pytorch' in provider:
    #         fn = lambda: _gt(x)

    #     ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    #     return ms

    # bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
