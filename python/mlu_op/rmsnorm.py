import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--Z", type=int, dest="Z", default=1)
    parser.add_argument("--H", type=int, dest="H", default=48)
    parser.add_argument("--wl", type=int, default=1024)
    parser.add_argument("--dh", type=int, dest="D_HEAD", default=64)

    args = parser.parse_args()
    return args

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.math.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def call_tt(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                )
    return out


def rms_norm_pytorch(x: torch.Tensor, rms_w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms_w


if __name__ == '__main__':
    drl_config = parse_args()

    random.seed(drl_config.seed)
    np.random.seed(drl_config.seed)
    torch.manual_seed(drl_config.seed)

    batch, heads, seq_len, dim = drl_config.Z, drl_config.H, drl_config.wl, drl_config.D_HEAD
    K=heads*dim

    embeddings_load = torch.randn([batch, seq_len, heads * dim], dtype=torch.float16, device="mlu")
    rms_weights = torch.randn([heads * dim], dtype=torch.float16, device="mlu") * 0.2


    drl_config.total_flops = batch * seq_len * heads * dim

    rms_out_tt = call_tt(x=embeddings_load, rms_w=rms_weights, eps=1e-6)
    rms_out_pt = rms_norm_pytorch(embeddings_load, rms_weights, eps=1e-6)

    assert torch.allclose(rms_out_tt, rms_out_pt, atol=1e-1), \
            f"max diff: {torch.max(torch.abs(rms_out_tt, rms_out_pt))}"

    print('BENCH...')

    # ms = triton.testing.do_bench(lambda: call(_cuasmrl, load_dir, embeddings_load, rms_weights), warmup=100, rep=100)
    ms_tt = triton.testing.do_bench(lambda: call_tt(x=embeddings_load, rms_w=rms_weights), warmup=100, rep=100)
    ms_pt = triton.testing.do_bench(lambda: rms_norm_pytorch(embeddings_load, rms_weights, eps=1e-6), warmup=100, rep=100)

    print(f'tt: {ms_tt}; pt: {ms_pt}')
    # data = {
    #     'cuasmrl': ms,
    #     'tt': ms_tt,
    # }
    # print(data)

    # fp = f"data/{GPU}/rmsnorm/{batch}_{heads}_{seq_len}_{dim}/bench_{drl_config.seed}.pkl"
    # with open(fp, 'wb') as f:
    #     pickle.dump(data, f)