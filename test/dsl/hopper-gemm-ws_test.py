import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = "npu"

M, N, K = (8192, 8192, 8192)

# 简化的hook函数
def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]

# 直接使用一个配置，不使用autotune
@triton.jit
def matmul_kernel_tlx_ws(
    a_desc, b_desc, c_desc,  #
    M, N, K,  #
    BM: tl.constexpr,  #
    BN: tl.constexpr,  #
    BK: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    # Need NUM_STAGES sets of SMEM buffers for A and B
    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    # Need NUM_STAGES sets of mbarriers for A and B
    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 1

            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                # Async load to a[buf]
                empty_a_1st = tlx.local_view(bars_empty_a, buf)
                full_a_1st = tlx.local_view(bars_full_a, buf)
                tlx.barrier_wait(bar=empty_a_1st, phase=p)
                tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                data_a_1st = tlx.local_view(a, buf)
                tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                # Async load to b[buf]
                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                # Async load to a[buf+NUM_STAGES]
                empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES - 1))

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 0
            acc = tl.zeros([BM // 2, BN], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

                # Wait for TMA load
                full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id())
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=full_a, phase=p)
                tlx.barrier_wait(bar=full_b, phase=p)

                # async_dot
                data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id())
                data_b = tlx.local_view(b, buf)
                acc = tlx.async_dot(
                    data_a,
                    data_b,
                    acc,
                )
                # async_wait
                acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                # Release buffers
                empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id())
                empty_b = tlx.local_view(bars_empty_b, buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES - 1))

            offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn], c0)
                c1 = acc1.to(tlx.dtype_of(c_desc))
                c_desc.store([offset_cm, offset_bn + BN // 2], c1)
            else:
                c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros(
        (M, N),
        dtype=torch.float16,
        device=DEVICE,
    )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(
        a,
        shape=[M, K],
        strides=[K, 1],
        block_shape=dummy_block,
    )

    desc_in_2 = TensorDescriptor(
        b,
        shape=[K, N],
        strides=[N, 1],
        block_shape=dummy_block,
    )
    desc_out = TensorDescriptor(
        c,
        shape=[M, N],
        strides=[N, 1],
        block_shape=dummy_block,
    )

    # 直接使用固定配置参数
    config = {
        "BM": 128,
        "BN": 256,
        "BK": 64,
        "GROUP_SIZE_M": 8,
        "NUM_STAGES": 4,
        "NUM_MMA_WARPS": 8,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": True,
    }
    
    # 设置block_shape
    matmul_tma_set_block_size_hook({
        "BM": config["BM"],
        "BN": config["BN"],
        "BK": config["BK"],
        "NUM_MMA_GROUPS": config["NUM_MMA_GROUPS"],
        "EPILOGUE_SUBTILE": config["EPILOGUE_SUBTILE"],
        "a_desc": desc_in_1,
        "b_desc": desc_in_2,
        "c_desc": desc_out,
    })

    grid = (triton.cdiv(M, config['BM']) * triton.cdiv(N, config['BN']), )
    
    # 直接调用内核
    matmul_kernel_tlx_ws[grid](
        desc_in_1, desc_in_2, desc_out,
        M, N, K,
        BM=config["BM"],
        BN=config["BN"],
        BK=config["BK"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        NUM_STAGES=config["NUM_STAGES"],
        NUM_MMA_WARPS=config["NUM_MMA_WARPS"],
        NUM_MMA_GROUPS=config["NUM_MMA_GROUPS"],
        EPILOGUE_SUBTILE=config["EPILOGUE_SUBTILE"],
        num_stages=1,
        num_warps=4,
    )
    return c


def main():
    print(f"Running matmul test on {DEVICE}...")
    
    # 创建随机输入矩阵
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    
    # 运行Triton kernel
    print(f"Input shapes: a={a.shape}, b={b.shape}")
    
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    
    start.record()
    c_triton = matmul(a, b)
    end.record()
    torch.npu.synchronize()
    
    elapsed_time = start.elapsed_time(end)
    print(f"Triton matmul time: {elapsed_time:.2f} ms")
    
    # 使用PyTorch验证结果（可选）
    if M <= 4096 and N <= 4096 and K <= 4096:  # 限制大小避免OOM
        print("Verifying with PyTorch matmul...")
        c_torch = torch.matmul(a.float(), b.float()).half()
        
        # 计算误差
        error = torch.abs(c_triton - c_torch).max().item()
        print(f"Maximum absolute error: {error:.6f}")
        
        if error < 1e-2:
            print("Result verification PASSED!")
        else:
            print("Result verification FAILED!")
    
    print(f"Output shape: {c_triton.shape}")
    print(f"Test completed!")


if __name__ == "__main__":
    main()