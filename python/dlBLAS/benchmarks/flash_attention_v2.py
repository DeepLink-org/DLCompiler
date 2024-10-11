import triton
import dlblas
from dlblas.utils.gpu_helper import get_idle_device
import torch
import torch.nn.functional as F


def test():
    device_ = torch.device(get_idle_device())
    torch.cuda.set_device(device_)
    dtype = torch.float16

    seq_len, heads, dim = 25600, 32, 64
    query = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    key = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    value = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    cos = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)
    sin = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)

    tt_out = dlblas.flash_attention_v2(query, key, value)
    ref_out = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)

    print("TEST: ")
    # print(tt_out)
    print("max abs diff: ", torch.max(abs(tt_out - ref_out)))
    assert torch.allclose(tt_out, ref_out, atol=1e-2, rtol=0)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd"],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["triton", "torch"],
            ylabel="ms",
            plot_name=f"flashAttention(batchSize={1}, seqlen:{seq_len}, num_heads:{heads}, dim:{dim})",
            args={"SeqLen": seq_len},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 200

        if "triton" in provider:
            fn = lambda: dlblas.flash_attention_v2(query, key, value)

        if "torch" in provider:
            fn = lambda: F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
            ).permute(0, 2, 1, 3)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")
