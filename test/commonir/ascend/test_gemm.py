import tilelang
import tilelang.language as T
import torch

device = torch.npu.current_device()
dtype = torch.float16


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_K, block_M), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[k * block_K, by * block_M], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_A=True,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def test_gemm():
    func = matmul(1024, 1024, 1024, 128, 128, 32)
    kernel = tilelang.compile(func)
    SIZEALL = 1024

    torch.manual_seed(0)
    a = torch.rand((SIZEALL, SIZEALL), dtype=dtype, device=device) - 0.5
    b = torch.rand((SIZEALL, SIZEALL), dtype=dtype, device=device) - 0.5
    result = torch.zeros((SIZEALL, SIZEALL), dtype=dtype, device=device)

    kernel(a, b, result)
    golden = torch.transpose(a, 0, 1) @ torch.transpose(b, 0, 1)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2**-6

    torch.testing.assert_close(result[mask], golden[mask], atol=tmpatol, rtol=0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=tmprtol)
    print("run matmul success")


if __name__ == "__main__":
    test_gemm()
