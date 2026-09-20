# Adapted from FlagTree 22f4ff0: target/vendor name and host reference device only.
# flagtree tle
"""TsingMicro vendor DSA rand-family tests (tle.dsa.wafer).

Covers ``randgen`` (raw xorshift128+ i64 stream), ``rand`` (Uniform(0,1))
and ``randn`` (Normal(0,1) via Box-Muller) on the Wafer hardware TRNG.

Constraints exercised by the API:
  - ``randgen``: n_out multiple of 16, seeds are ``[16]`` i64 blocks
  - ``rand`` / ``randn``: n_out multiple of 32
"""

import pytest
import torch

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_txda():
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        return False
    target = triton.runtime.driver.active.get_current_target()
    return getattr(target, "backend", None) == "wafer"


pytestmark = pytest.mark.skipif(not _is_txda(), reason="requires TsingMicro (txda) backend")


@triton.jit
def _randgen_kernel(seed_val, out_ptr, s0_ptr, s1_ptr, N: tl.constexpr):
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed_val
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed_val + 1
    out, s0o, s1o = tle.dsa.wafer.randgen(s0, s1, N)
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, out)
    tl.store(s0_ptr + tl.arange(0, 16), s0o)
    tl.store(s1_ptr + tl.arange(0, 16), s1o)


@triton.jit
def _rand_kernel(seed_val, out_ptr, N: tl.constexpr):
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed_val
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed_val + 1
    u, s0o, s1o = tle.dsa.wafer.rand(s0, s1, N)
    tl.store(out_ptr + tl.arange(0, N), u)


@triton.jit
def _randn_kernel(seed_val, out_ptr, N: tl.constexpr):
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed_val
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed_val + 1
    n, s0o, s1o = tle.dsa.wafer.randn(s0, s1, N)
    tl.store(out_ptr + tl.arange(0, N), n)


def test_randgen_shape_and_determinism():
    n = 32
    out = torch.empty(n, device="cpu", dtype=torch.int64)
    s0o = torch.zeros(16, device="cpu", dtype=torch.int64)
    s1o = torch.zeros(16, device="cpu", dtype=torch.int64)
    out_txda = out.to("txda")
    s0o_txda = s0o.to("txda")
    s1o_txda = s1o.to("txda")
    _randgen_kernel[(1, )](42, out_txda, s0o_txda, s1o_txda, N=n)
    with torch.no_grad():
        out.copy_(out_txda.cpu())
        s0o.copy_(s0o_txda.cpu())
        s1o.copy_(s1o_txda.cpu())

    # determinism: same seed -> same stream
    out2 = torch.empty_like(out)
    out2_txda = out2.to("txda")
    s0o_txda = s0o.to("txda")
    s1o_txda = s1o.to("txda")
    _randgen_kernel[(1, )](42, out2_txda, s0o_txda, s1o_txda, N=n)
    with torch.no_grad():
        out2.copy_(out2_txda.cpu())
        s0o.copy_(s0o_txda.cpu())
        s1o.copy_(s1o_txda.cpu())
    torch.testing.assert_close(out, out2)

    # stream advances: different seed -> different values
    out3 = torch.empty_like(out)
    out3_txda = out3.to("txda")
    s0o_txda = s0o.to("txda")
    s1o_txda = s1o.to("txda")
    _randgen_kernel[(1, )](43, out3_txda, s0o_txda, s1o_txda, N=n)
    with torch.no_grad():
        out3.copy_(out3_txda.cpu())
        s0o.copy_(s0o_txda.cpu())
        s1o.copy_(s1o_txda.cpu())
    assert not torch.equal(out, out3)


def test_randgen_invalid_n_out():
    out = torch.empty(8, device="cpu", dtype=torch.int64)
    s0o = torch.zeros(16, device="cpu", dtype=torch.int64)
    s1o = torch.zeros(16, device="cpu", dtype=torch.int64)
    with pytest.raises(Exception):
        out_txda = out.to("txda")
        s0o_txda = s0o.to("txda")
        s1o_txda = s1o.to("txda")
        _randgen_kernel[(1, )](42, out_txda, s0o_txda, s1o_txda, N=8)
        with torch.no_grad():
            out.copy_(out_txda.cpu())
            s0o.copy_(s0o_txda.cpu())
            s1o.copy_(s1o_txda.cpu())  # not a multiple of 16


def test_rand_uniform_stats():
    n = 16384
    out = torch.empty(n, device="cpu", dtype=torch.float32)
    out_txda = out.to("txda")
    _rand_kernel[(1, )](7, out_txda, N=n)
    with torch.no_grad():
        out.copy_(out_txda.cpu())

    assert out.min().item() >= 0.0 and out.max().item() < 1.0
    mean, std = out.mean().item(), out.std().item()
    # Uniform(0,1): mean 0.5, std sqrt(1/12) ~= 0.2887; 16K samples of a
    # single fixed-seed hardware stream, tolerance ~3 sigma.
    assert abs(mean - 0.5) < 0.03, f"uniform mean off: {mean}"
    assert abs(std - 0.2887) < 0.01, f"uniform std off: {std}"


def test_randn_normal_stats():
    n = 16384
    out = torch.empty(n, device="cpu", dtype=torch.float32)
    out_txda = out.to("txda")
    _randn_kernel[(1, )](11, out_txda, N=n)
    with torch.no_grad():
        out.copy_(out_txda.cpu())

    mean, std = out.mean().item(), out.std().item()
    assert abs(mean) < 0.05, f"normal mean off: {mean}"
    assert abs(std - 1.0) < 0.05, f"normal std off: {std}"
    # heavier tails than uniform
    assert out.abs().max().item() > 3.0


def test_rand_invalid_n_out():
    out = torch.empty(16, device="cpu", dtype=torch.float32)
    with pytest.raises(Exception):
        out_txda = out.to("txda")
        _rand_kernel[(1, )](7, out_txda, N=16)
        with torch.no_grad():
            out.copy_(out_txda.cpu())  # not a multiple of 32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
