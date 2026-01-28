import math
import pytest
import torch

from triton.backends.dicp_triton.device_utils import infer_device

import torch
import enum
from typing import Tuple

import triton
import triton.language as tl


class BackendType(enum.Enum):
    """Backend type."""

    TORCH = enum.auto()
    TRITON = enum.auto()


def lightning_attention_prefill_forward_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    slope_rate: torch.Tensor,
    BLOCK_SIZE=64,
    in_place=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform lightning attention prefill.
    modify from: https://github.com/MiniMax-AI/MiniMax-M1/blob/main/modeling_minimax_m1.py

    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
        slope_rate: Decay rate tensor
        BLOCK_SIZE: Size of blocks for processing
        BLOCK_MODEL: Size of blocks for parallel processing

    Returns:
        output: Attention output tensor [B, H, N, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert q.ndim == 4
    assert past_key_value.shape == (b, h, d, e)

    s = slope_rate.to(torch.float32)
    NUM_BLOCK = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    array = torch.arange(BLOCK_SIZE).to(q) + 1
    q_decay = torch.exp(-s * array.reshape(-1, 1))
    k_decay = torch.exp(-s * (BLOCK_SIZE - array.reshape(-1, 1)))
    index = array[:, None] - array[None, :]
    s_index = (
        s
        * index[
            None,
            None,
        ]
    )
    s_index = torch.where(index >= 0, -s_index, float("-inf"))
    diag_decay = torch.exp(s_index)

    if past_key_value is not None:
        kv = past_key_value
    else:
        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
    output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

    for i in range(NUM_BLOCK):
        si = i * BLOCK_SIZE
        ei = min(si + BLOCK_SIZE, n)
        m = ei - si
        qi = q[:, :, si:ei].contiguous()
        ki = k[:, :, si:ei].contiguous()
        vi = v[:, :, si:ei].contiguous()
        qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)
        qk = (
            torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32)
            * diag_decay[:, :, :m, :m]
        )
        qkv_diag = torch.matmul(qk, vi.to(torch.float32))
        output[:, :, si:ei] = qkv_none_diag + qkv_diag
        block_decay = torch.exp(-s * m)
        kv = block_decay * kv + torch.matmul(
            (ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi
        )
    if in_place:
        past_key_value.copy_(kv)
        return output, past_key_value
    else:
        return output, kv


@triton.jit
def _fwd_loop_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    slope_rate,
    kv_cache_ptr,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    """
    Kernel for lightning attention prefill with KV cache.
    """
    # get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    e_offset = off_e * BLOCK_MODEL
    kv_offset = off_bh * d * e

    # get block ptr
    Q_block_ptr = q_ptr + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = k_ptr + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = v_ptr + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = output_ptr + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    S_block_ptr = slope_rate + off_h

    # init decay
    s = tl.load(S_block_ptr).to(tl.float32)
    off_block = tl.arange(0, BLOCK)
    q_decay = tl.exp(-s.to(tl.float32) * (off_block[:, None] + 1))
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - 1 - off_block[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    index = off_block[:, None] - off_block[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)

    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    # loop compute
    for i in range(NUM_BLOCK):
        if n < BLOCK * (i + 1):
            block_decay = tl.exp(-s.to(tl.float32) * (n - BLOCK * i))
            # (BLOCK - 1 - off_block[None, :] + n - BLOCK)
            k_trans_decay = tl.exp(-s.to(tl.float32) * (n - 1 - off_block[None, :]))
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q * q_decay, kv)
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        off_block += BLOCK

    KV_block_ptr = (
        kv_cache_ptr
        + kv_offset
        + e_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, BLOCK_MODEL)[None, :]
    )
    tl.store(
        KV_block_ptr,
        kv.to(KV_block_ptr.dtype.element_ty),
    )


def lightning_attention_prefill_forward_triton_loop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    slope_rate: torch.Tensor,
    BLOCK_SIZE=64,
    BLOCK_MODEL=32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform lightning attention prefill.
    modify from: https://github.com/OpenNLPLab/lightning-attention/blob/main/lightning_attn/ops/triton/lightning_attn2.py

    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
        slope_rate: Decay rate tensor
        BLOCK_SIZE: Size of blocks for processing
        BLOCK_MODEL: Size of blocks for parallel processing

    Returns:
        output: Attention output tensor [B, H, N, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = slope_rate.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
    assert past_key_value.shape == (b, h, d, e)
    assert o.shape == v.shape
    assert o.dtype == v.dtype

    NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK_SIZE)
    # parallel over channel
    BLOCK_M = min(triton.next_power_of_2(e), BLOCK_MODEL)
    assert e % BLOCK_M == 0
    grid = (b * h, triton.cdiv(e, BLOCK_M))

    if past_key_value is not None:
        kv = past_key_value
        assert kv.dtype == torch.float32
    else:
        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)

    _fwd_loop_kernel[grid](
        q,
        k,
        v,
        o,
        s,
        kv,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK_SIZE,
        NUM_BLOCK=NUM_BLOCK,
        BLOCK_MODEL=BLOCK_M,
    )
    return o, kv


def lightning_attention_decode_forward_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    slope_rate: torch.Tensor,
    in_place=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform lightning attention decoding.
    modify from: https://github.com/MiniMax-AI/MiniMax-M1/blob/main/modeling_minimax_m1.py

    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
        slope_rate: Decay rate tensor

    Returns:
        output: Attention output tensor [B, H, 1, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
    """
    assert q.ndim == 4
    B, H, _, D = q.shape
    E = v.shape[-1]
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, E)
    assert past_key_value.shape == (B, H, D, E)
    kv = past_key_value
    s = torch.exp(-slope_rate)
    kv = (
        torch.einsum(
            "... n d, ... n e -> ... d e",
            k,
            v,
        )
        + s * kv
    )
    qkv = torch.einsum("... n d, ... d e -> ... n e", q, kv.to(q.dtype))
    past_key_value.copy_(kv)
    if in_place:
        past_key_value.copy_(kv)
        return qkv, past_key_value
    else:
        return qkv, kv


@triton.jit
def _lightningattn_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate,
    output_ptr,
    D: tl.constexpr,
    qkv_b_stride,
    qkv_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d_stride,
    cache_e_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for lightning attention decoding with KV cache.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    batch_id = pid_b
    head_id = pid_h

    # Load decay rate for the current head
    ratio = tl.load(slope_rate + pid_h)

    # Calculate offsets for dimensions
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = (
        qk_d_offsets[:, None] * cache_d_stride + v_d_offsets[None, :] * cache_e_stride
    )

    # Calculate offsets for the current batch and head
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = batch_id * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # Load query, key, and value tensors
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    # Compute key-value outer product
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # Apply decay to previous KV cache
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old

    # Compute attention output
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)


def lightning_attention_decode_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    slope_rate: torch.Tensor,
    BLOCK_SIZE: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform lightning attention decoding using Triton kernels.
    modify from: https://github.com/vllm-project/vllm/vllm/model_executor/layers/lightning_attn.py

    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, E]
        kv_caches: Key-value cache tensor
        slope_rate: Decay rate tensor
        BLOCK_SIZE: Size of blocks for processing

    Returns:
        output: Attention output tensor [B, H, 1, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
    """
    assert q.ndim == 4
    B, H, _, D = q.shape
    E = v.shape[-1]
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, E)
    assert past_key_value.shape == (B, H, D, E)

    # Initialize output tensor
    o = torch.empty((B, H, 1, E), dtype=q.dtype, device=q.device)

    # Set grid dimensions for the kernel
    grid = (B, H, D // BLOCK_SIZE)

    # Calculate strides for tensors
    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)

    cache_b_stride = past_key_value.stride(0)
    cache_h_stride = past_key_value.stride(1)
    cache_d_stride = past_key_value.stride(2)
    cache_e_stride = past_key_value.stride(3)

    # Launch the kernel
    _lightningattn_attn_decode_kernel[grid](
        q,
        k,
        v,
        past_key_value,
        slope_rate,
        o,
        D,
        qkv_b_stride,
        qkv_h_stride,
        cache_b_stride,
        cache_h_stride,
        cache_d_stride,
        cache_e_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return o, past_key_value


def lightning_attention_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    slope_rate: torch.Tensor,
    BLOCK_SIZE=64,
    BackendType: int = BackendType.TORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
        slope_rate: Decay rate tensor
        BLOCK_SIZE: Size of blocks for processing
        BLOCK_MODEL: Size of blocks for parallel processing

    Returns:
        output: Attention output tensor [B, H, N, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
    """
    if BackendType == BackendType.TRITON:
        return lightning_attention_prefill_forward_triton_loop(
            q, k, v, past_key_value, slope_rate, BLOCK_SIZE
        )
    else:
        return lightning_attention_prefill_forward_torch(
            q,
            k,
            v,
            past_key_value,
            slope_rate,
        )


def lightning_attention_decode_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    slope_rate: torch.Tensor,
    BLOCK_SIZE: int = 128,
    BackendType: int = BackendType.TORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
        slope_rate: Decay rate tensor
        BLOCK_SIZE: Size of blocks for processing in triton
        BackendType: torch or triton

    Returns:
        output: Attention output tensor [B, H, 1, E]
        kv_caches: Key-value cache tensor [B, H, D, E]
    """
    if BackendType == BackendType.TRITON:
        return lightning_attention_decode_forward_triton(
            q, k, v, past_key_value, slope_rate, BLOCK_SIZE
        )
    else:
        return lightning_attention_decode_forward_torch(
            q, k, v, past_key_value, slope_rate
        )


class TestLightningAttn:

    @pytest.fixture
    def B(self, request):
        yield request.param

    @pytest.fixture
    def H(self, request):
        yield request.param

    @pytest.fixture
    def N(self, request):
        yield request.param

    @pytest.fixture
    def D(self, request):
        yield request.param

    @pytest.fixture
    def E(self, request):
        yield request.param

    @pytest.fixture
    def dtype(self, request):
        yield request.param

    @pytest.fixture
    def BLOCK_SIZE(self, request):
        yield request.param

    @pytest.fixture
    def q_states(self, B, H, N, D, dtype):
        yield torch.randn([B, H, N, D], dtype=dtype, device=infer_device())

    @pytest.fixture
    def k_states(self, B, H, N, D, dtype):
        yield torch.randn([B, H, N, D], dtype=dtype, device=infer_device())

    @pytest.fixture
    def v_states(self, B, H, N, E, dtype):
        yield torch.randn([B, H, N, E], dtype=dtype, device=infer_device())

    @pytest.fixture
    def past_key_value(self, B, H, D, E, dtype):
        yield torch.randn([B, H, D, E], dtype=dtype, device=infer_device())

    @pytest.fixture
    def slope_rate(self, H, dtype):
        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slope_rate = torch.tensor(
            get_slopes(H), dtype=dtype, device=infer_device()
        ).reshape(H, 1, 1)
        yield slope_rate * (1 + 1e-5)

    # float32 only
    @pytest.mark.parametrize(
        ["B", "H", "N", "D", "E", "dtype", "BLOCK_SIZE"],
        [
            (1, 64, 5, 64, 128, torch.float32, 8),
            (1, 64, 72, 64, 64, torch.float32, 8),
            # (1, 64, 72, 64, 64, torch.float32, 16),
        ],
        indirect=True,
    )
    def test_lightning_attention_prefill(
        self,
        q_states,
        k_states,
        v_states,
        slope_rate,
        past_key_value,
        BLOCK_SIZE,
        dtype,
    ):
        past_key_value_torch = torch.zeros_like(past_key_value)
        past_key_value_triton = torch.zeros_like(past_key_value)
        out_torch, _ = lightning_attention_prefill_forward(
            q_states,
            k_states,
            v_states,
            past_key_value_torch,
            slope_rate,
            BLOCK_SIZE,
            BackendType=BackendType.TORCH,
        )
        out_triton, _ = lightning_attention_prefill_forward(
            q_states,
            k_states,
            v_states,
            past_key_value_triton,
            slope_rate,
            BLOCK_SIZE,
            BackendType=BackendType.TRITON,
        )

        if dtype == torch.float32:
            rtol = 1e-03
            atol = 1
        else:
            rtol = 1e-03
            atol = 1

        kv_check = torch.allclose(
            past_key_value_torch,
            past_key_value_triton,
            rtol=rtol,
            atol=atol,
        )
        output_check = torch.allclose(
            out_torch,
            out_triton,
            rtol=rtol,
            atol=atol,
        )

        assert (
            kv_check
        ), f"past_key_value torch:{past_key_value_torch}, past_key_value triton:{past_key_value_triton}"
        assert output_check, f"output torch:{out_torch}, output triton:{out_triton}"
        print(
            f"zmz debug torch kv_check:{past_key_value_torch}, triton :{past_key_value_triton}"
        )

    # float32 only
    @pytest.mark.parametrize(
        ["B", "H", "N", "D", "E", "dtype", "BLOCK_SIZE"],
        [
            (8, 64, 1, 128, 128, torch.float32, 64),
            (16, 64, 1, 128, 128, torch.float32, 64),
        ],
        indirect=True,
    )
    def test_lightning_attention_decode(
        self,
        q_states,
        k_states,
        v_states,
        slope_rate,
        past_key_value,
        BLOCK_SIZE,
        dtype,
    ):
        past_key_value_torch = torch.zeros_like(past_key_value)
        past_key_value_triton = torch.zeros_like(past_key_value)
        out_torch, _ = lightning_attention_decode_forward(
            q_states,
            k_states,
            v_states,
            past_key_value_torch,
            slope_rate,
            BLOCK_SIZE,
            BackendType=BackendType.TORCH,
        )
        out_triton, _ = lightning_attention_decode_forward(
            q_states,
            k_states,
            v_states,
            past_key_value_triton,
            slope_rate,
            BLOCK_SIZE,
            BackendType=BackendType.TRITON,
        )

        if dtype == torch.float32:
            rtol = 1e-03
            atol = 1e-02
        else:
            rtol = 1e-03
            atol = 1e-02

        kv_check = torch.allclose(
            past_key_value_torch,
            past_key_value_triton,
            rtol=rtol,
            atol=atol,
        )
        output_check = torch.allclose(
            out_torch,
            out_triton,
            rtol=rtol,
            atol=atol,
        )

        assert (
            kv_check
        ), f"past_key_value torch:{past_key_value_torch}, past_key_value triton:{past_key_value_triton}"
        assert output_check, f"output torch:{out_torch}, output triton:{out_triton}"


if __name__ == "__main__":
    pytest.main([__file__])
