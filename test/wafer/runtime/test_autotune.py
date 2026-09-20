"""Autotuner timing and cached invocation with real torch_txda tensor inputs."""
import math
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl
from triton.testing import do_bench


def test_native_autotune():
    measured = []

    def benchmark(fn, quantiles):
        times = do_bench(fn, warmup=1, rep=3, quantiles=quantiles)
        assert all(math.isfinite(t) and t > 0 for t in times)
        measured.append(times)
        return times

    @triton.autotune(configs=[triton.Config({'BLOCK': 64}), triton.Config({'BLOCK': 128})],
                     key=['N'], do_bench=benchmark)
    @triton.jit
    def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + i, tl.load(X + i, i < N, 0) + 1, i < N)

    host = torch.arange(257, dtype=torch.float32)
    x, y = (host).to("txda"), (torch.zeros_like(host)).to("txda")
    add_one[lambda meta: (triton.cdiv(257, meta['BLOCK']),)](x, y, N=257)
    assert len(measured) == 2
    assert add_one.best_config.kwargs['BLOCK'] in (64, 128)
    add_one[lambda meta: (triton.cdiv(257, meta['BLOCK']),)](x, y, N=257)
    assert len(measured) == 2  # Cache hit must not repeat benchmarking.
    torch.testing.assert_close(y.cpu(), host + 1, rtol=0, atol=0)
