import argparse

import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


BENCHMARK_SHAPES = [(1024, 1024), (2048, 2048), (4096, 4096), (4096, 8192)]


def is_maca():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "maca"


@gluon.jit
def layer_norm_fwd_kernel(X, Y, W, B, stride, N, eps, BLOCK_SIZE: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offset = row * stride
    x_ptrs = X + row_offset + cols
    y_ptrs = Y + row_offset + cols

    x = gl.load(x_ptrs, mask=mask, other=0.0).to(gl.float32)
    x = gl.where(mask, x, 0.0)
    mean = gl.sum(x, axis=0) / N

    x_centered = gl.where(mask, x - mean, 0.0)
    var = gl.sum(x_centered * x_centered, axis=0) / N
    rstd = gl.rsqrt(var + eps)

    w = gl.load(W + cols, mask=mask, other=0.0).to(gl.float32)
    b = gl.load(B + cols, mask=mask, other=0.0).to(gl.float32)
    y = x_centered * rstd * w + b

    gl.store(y_ptrs, y, mask=mask)


@gluon.jit
def layer_norm_fwd_chunked_kernel(X, Y, W, B, stride, N, eps, BLOCK_SIZE: gl.constexpr, CHUNK_SIZE: gl.constexpr):
    row = gl.program_id(0)
    row_offset = row * stride

    sum_acc = 0.0
    for off in gl.static_range(0, BLOCK_SIZE, CHUNK_SIZE):
        cols = off + gl.arange(0, CHUNK_SIZE)
        mask = cols < N
        x = gl.load(X + row_offset + cols, mask=mask, other=0.0).to(gl.float32)
        x = gl.where(mask, x, 0.0)
        sum_acc += gl.sum(x, axis=0)
    mean = sum_acc / N

    var_acc = 0.0
    for off in gl.static_range(0, BLOCK_SIZE, CHUNK_SIZE):
        cols = off + gl.arange(0, CHUNK_SIZE)
        mask = cols < N
        x = gl.load(X + row_offset + cols, mask=mask, other=0.0).to(gl.float32)
        x_centered = gl.where(mask, x - mean, 0.0)
        var_acc += gl.sum(x_centered * x_centered, axis=0)
    rstd = gl.rsqrt(var_acc / N + eps)

    for off in gl.static_range(0, BLOCK_SIZE, CHUNK_SIZE):
        cols = off + gl.arange(0, CHUNK_SIZE)
        mask = cols < N
        x = gl.load(X + row_offset + cols, mask=mask, other=0.0).to(gl.float32)
        w = gl.load(W + cols, mask=mask, other=0.0).to(gl.float32)
        b = gl.load(B + cols, mask=mask, other=0.0).to(gl.float32)
        y = (x - mean) * rstd * w + b
        gl.store(Y + row_offset + cols, y, mask=mask)


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    assert normalized_shape == (N,)

    max_fused_size = 65536 // x.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(N))
    if N > block_size:
        raise RuntimeError("This Gluon layer norm doesn't support feature dim >= 64KB.")

    num_warps = min(max(block_size // 256, 1), 8)

    layer_norm_fwd_kernel[(M,)](
        x_arg,
        y,
        weight,
        bias,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_ctas=1,
    )
    return y


def layer_norm_chunked(x, normalized_shape, weight, bias, eps=1e-5, chunk_size=1024):
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    assert normalized_shape == (N,)

    max_fused_size = 65536 // x.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(N))
    if N > block_size:
        raise RuntimeError("This Gluon layer norm doesn't support feature dim >= 64KB.")
    assert block_size % chunk_size == 0

    num_warps = min(max(chunk_size // 256, 1), 8)
    layer_norm_fwd_chunked_kernel[(M,)](
        x_arg,
        y,
        weight,
        bias,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=block_size,
        CHUNK_SIZE=chunk_size,
        num_warps=num_warps,
        num_ctas=1,
    )
    return y


@pytest.mark.skipif(not is_maca(), reason="Requires MetaX/MACA target")
@pytest.mark.parametrize("M, N", [(128, 1024), (1024, 4096), (4096, 8192)])
def test_maca_layer_norm_fwd(M, N):
    torch.manual_seed(0)
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, dtype=torch.float16, device="cuda")
    bias = torch.rand(w_shape, dtype=torch.float16, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=torch.float16, device="cuda")

    y = layer_norm(x, w_shape, weight, bias, 1e-5)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, 1e-5).to(torch.float16)
    diff = (y - y_ref).abs()
    rel = diff / y_ref.abs().clamp_min(1e-6)
    print(
        f"[accuracy] shape={M}x{N} "
        f"max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g} "
        f"max_rel={rel.max().item():.6g} "
        f"allclose={torch.allclose(y, y_ref, atol=1e-2, rtol=0)}"
    )
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)


def _ms_to_gbs(M, N, ms):
    bytes_moved = M * N * 2 * 3 + N * 2 * 2 + M * 4 * 2
    return bytes_moved / (ms * 1e-3) / 1e9


def _shape_name(M, N):
    return f"{M}x{N}"


def _make_inputs(M, N):
    torch.manual_seed(0)
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, dtype=torch.float16, device="cuda")
    bias = torch.rand(w_shape, dtype=torch.float16, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=torch.float16, device="cuda")
    return x, w_shape, weight, bias


def _measure_accuracy(x, w_shape, weight, bias):
    return _measure_accuracy_with(lambda: layer_norm(x, w_shape, weight, bias, 1e-5), x, w_shape, weight, bias)


def _measure_accuracy_with(fn, x, w_shape, weight, bias):
    y = fn()
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, 1e-5).to(torch.float16)
    diff = (y - y_ref).abs()
    rel = diff / y_ref.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "allclose": torch.allclose(y, y_ref, atol=1e-2, rtol=0),
    }


def _print_markdown_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def run_accuracy_cases(shapes=((128, 1024), (1024, 4096))):
    rows = []
    for M, N in shapes:
        x, w_shape, weight, bias = _make_inputs(M, N)
        result = _measure_accuracy(x, w_shape, weight, bias)
        rows.append([
            _shape_name(M, N),
            f"{result['max_abs']:.6g}",
            f"{result['mean_abs']:.6g}",
            f"{result['max_rel']:.6g}",
            result["allclose"],
        ])
    _print_markdown_table(["shape", "max_abs", "mean_abs", "max_rel", "allclose"], rows)


def run_benchmark(warmup=25, rep=100, check_correctness=True, compare_chunked=False):
    rows = []
    for M, N in BENCHMARK_SHAPES:
        x, w_shape, weight, bias = _make_inputs(M, N)
        accuracy = _measure_accuracy(x, w_shape, weight, bias)
        if check_correctness and not accuracy["allclose"]:
            y = layer_norm(x, w_shape, weight, bias, 1e-5)
            y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, 1e-5).to(torch.float16)
            torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)

        torch_ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, w_shape, weight, bias, 1e-5),
                                           warmup=warmup, rep=rep)
        gluon_ms = triton.testing.do_bench(lambda: layer_norm(x, w_shape, weight, bias, 1e-5), warmup=warmup,
                                           rep=rep)
        row = [
            _shape_name(M, N),
            f"{torch_ms:.4f}",
            f"{gluon_ms:.4f}",
            f"{_ms_to_gbs(M, N, torch_ms):.2f}",
            f"{_ms_to_gbs(M, N, gluon_ms):.2f}",
            f"{torch_ms / gluon_ms:.3f}",
            f"{accuracy['max_abs']:.6g}",
            f"{accuracy['mean_abs']:.6g}",
            f"{accuracy['max_rel']:.6g}",
            accuracy["allclose"],
        ]
        if compare_chunked:
            chunked_accuracy = _measure_accuracy_with(
                lambda: layer_norm_chunked(x, w_shape, weight, bias, 1e-5),
                x,
                w_shape,
                weight,
                bias,
            )
            if check_correctness and not chunked_accuracy["allclose"]:
                y = layer_norm_chunked(x, w_shape, weight, bias, 1e-5)
                y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, 1e-5).to(torch.float16)
                torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)
            chunked_ms = triton.testing.do_bench(lambda: layer_norm_chunked(x, w_shape, weight, bias, 1e-5),
                                                 warmup=warmup, rep=rep)
            row.extend([
                f"{chunked_ms:.4f}",
                f"{_ms_to_gbs(M, N, chunked_ms):.2f}",
                f"{torch_ms / chunked_ms:.3f}",
                f"{chunked_accuracy['max_abs']:.6g}",
                f"{chunked_accuracy['mean_abs']:.6g}",
                f"{chunked_accuracy['max_rel']:.6g}",
                chunked_accuracy["allclose"],
            ])
        rows.append(row)

    headers = ["shape", "torch_ms", "gluon_ms", "torch_GB/s", "gluon_GB/s", "speedup", "max_abs", "mean_abs",
               "max_rel", "allclose"]
    if compare_chunked:
        headers.extend(["chunked_ms", "chunked_GB/s", "chunked_speedup", "chunked_max_abs", "chunked_mean_abs",
                        "chunked_max_rel", "chunked_allclose"])
    _print_markdown_table(headers, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="benchmark selected layer norm forward shapes")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-check", action="store_true", help="skip correctness check during benchmark")
    parser.add_argument("--compare-chunked", action="store_true", help="also benchmark experimental chunked kernel")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(warmup=args.warmup, rep=args.rep, check_correctness=not args.no_check,
                      compare_chunked=args.compare_chunked)
    else:
        run_accuracy_cases()

# shape        torch_ms    gluon_ms    speedup
# ━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━
# 1024x1024      0.0271      0.0239      1.131
# ───────────  ──────────  ──────────  ─────────
# 2048x2048      0.0453      0.0508      0.892
# ───────────  ──────────  ──────────  ─────────
# 4096x4096      0.1115      0.1307      0.853
# ───────────  ──────────  ──────────  ─────────
# 4096x8192      0.1810      0.2955      0.612
