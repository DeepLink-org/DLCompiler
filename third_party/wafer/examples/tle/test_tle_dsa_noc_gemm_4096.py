import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

TILE_NUM = 16
M = 4096
K = 1024
N = 4096
BLOCK_M = M // TILE_NUM
BLOCK_K = K
SUB_N = N // TILE_NUM

TILE_PHYSICAL_RELATION = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 9, 10, 6, 5, 4]

MESH = tle.device_mesh(
    None,
    _shape=(TILE_NUM, ),
    _dim_names=("tile", ),
    _physical_ids=tuple(TILE_PHYSICAL_RELATION),
)


@triton.jit
def dsa_shift_n_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    send_next_tile_lut_ptr,
    ring_index_lut_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SUB_N: tl.constexpr,
    TILE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    send_next_tile = tl.load(send_next_tile_lut_ptr + pid)
    ring_index = tl.load(ring_index_lut_ptr + pid)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    a = tl.load(a_ptrs)

    shard_idx = ring_index
    offs_sub_n = shard_idx * SUB_N + tl.arange(0, SUB_N)
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_sub_n[None, :]
    b_init = tl.load(b_ptrs)

    send_buf = tle.dsa.alloc((BLOCK_K, SUB_N), tl.float16)
    recv_buf = tle.dsa.alloc((BLOCK_K, SUB_N), tl.float16)

    offs_buf_k = tl.arange(0, BLOCK_K)[:, None] + tl.zeros((1, SUB_N), dtype=tl.int32)
    offs_buf_n = tl.arange(0, SUB_N)[None, :] + tl.zeros((BLOCK_K, 1), dtype=tl.int32)

    send_ptr = tle.dsa.local_ptr(send_buf, [offs_buf_k, offs_buf_n])
    recv_ptr = tle.dsa.local_ptr(recv_buf, [offs_buf_k, offs_buf_n])

    remote_recv_buf = tle.remote(recv_buf, send_next_tile)
    remote_recv_ptr = tle.dsa.local_ptr(remote_recv_buf, [offs_buf_k, offs_buf_n])

    tl.store(send_ptr, b_init)

    for step in range(TILE_NUM):
        b_cur = tl.load(send_ptr)
        c_part = tl.dot(a, b_cur, out_dtype=tl.float32)

        offs_n = shard_idx * SUB_N + tl.arange(0, SUB_N)
        c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, c_part.to(tl.float16))

        if step < TILE_NUM - 1:
            tl.store(remote_recv_ptr, tl.load(send_ptr))
            # tle.distributed_barrier(MESH)
            tl.store(send_ptr, tl.load(recv_ptr))
            # tle.distributed_barrier(MESH)

            shard_idx = tl.where(shard_idx == 0, TILE_NUM - 1, shard_idx - 1)


def build_ring_luts(mesh, device):
    phys = mesh.physical_ids
    n = mesh.size
    send_next = torch.empty(n, dtype=torch.int32)
    ring_index = torch.empty(n, dtype=torch.int32)
    for i in range(n):
        cur = phys[i]
        nxt = phys[(i + 1) % n]
        send_next[cur] = nxt
        ring_index[cur] = i
    return send_next.to(device), ring_index.to(device)


def run(m=M, n=N, k=K, device="cpu", pattern="random", seed=0):
    """Execute with explicit torch_txda tensors; the CRT ring has 16 members."""
    if m <= 0 or n <= 0 or m % TILE_NUM or n % TILE_NUM or k <= 0:
        raise ValueError("M and N must be divisible by the 16-tile ring size")
    torch.manual_seed(seed)
    if pattern == "structured":
        # Exact FP16 values exercise tile/shard routing without reduction noise.
        a = torch.zeros((m, k), device="cpu", dtype=torch.float16)
        a[torch.arange(m, device="cpu"), torch.arange(m, device="cpu") % k] = 1
        rows = torch.arange(k, device="cpu")[:, None]
        cols = torch.arange(n, device="cpu")[None, :]
        b = (((rows * 7 + cols * 11 + seed * 13) % 1024).float() / 1024).half()
    elif pattern == "random":
        a = torch.randn((m, k), device="cpu", dtype=torch.float16)
        b = torch.randn((k, n), device="cpu", dtype=torch.float16)
    else:
        raise ValueError(f"Unknown input pattern: {pattern}")
    c = torch.full((m, n), float("nan"), device="cpu", dtype=torch.float16)
    send_next_lut, ring_index_lut = build_ring_luts(MESH, device)
    from triton.backends.dicp_triton.wafer_runtime import initialize_noc
    initialize_noc()
    a_txda = a.to("txda")
    b_txda = b.to("txda")
    c_txda = c.to("txda")
    send_next_lut_txda = send_next_lut.to("txda")
    ring_index_lut_txda = ring_index_lut.to("txda")
    dsa_shift_n_gemm_kernel[(TILE_NUM,)](
        a_txda, b_txda, c_txda, send_next_lut_txda, ring_index_lut_txda,
        M=m, N=n, K=k, BLOCK_M=m // TILE_NUM, BLOCK_K=k,
        SUB_N=n // TILE_NUM, TILE_NUM=TILE_NUM,
        launch_mode="cluster",
    )
    with torch.no_grad():
        c.copy_(c_txda.cpu())
    ref = a.cpu().float() @ b.cpu().float()
    result = c.cpu().float()
    tolerance = 0.0 if pattern == "structured" else 1e-1
    torch.testing.assert_close(result, ref, atol=tolerance, rtol=tolerance)
    max_diff = (result - ref).abs().max().item()
    print(f"PASS NoC ring GEMM: M={m}, N={n}, K={k}, tiles={TILE_NUM}, "
          f"pattern={pattern}, seed={seed}, max_abs_diff={max_diff:.8g}", flush=True)


@pytest.mark.parametrize("m,n,k", [(256, 256, 64), (4096, 4096, 1024)])
@pytest.mark.parametrize("pattern", ["structured", "random"])
def test_noc_gemm(m, n, k, pattern, device):
    for seed in (0, 1):
        run(m, n, k, device=device, pattern=pattern, seed=seed)


if __name__ == "__main__":
    raise SystemExit("Run this example with scripts/wafer/run_wafer_example_suite.py "
                     "--suite examples --select tle/test_tle_dsa_noc_gemm_4096.py "
                     "--output-dir /tmp/wafer-noc-results")
