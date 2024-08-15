# modify from: https://github.com/InternLM/lmdeploy
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def per_channel_quant(x, n_bits, dtype):
    """Quantize the input tensor 'x' channel-wise using the given number of
    bits.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be a
            2-dimensional tensor.
        n_bits (int): The number of bits to use for quantization.
        dtype (torch.dtype): The data type to which the quantized tensor should
            be converted.

    Returns:
        tuple: A tuple containing two items -- the quantized tensor and
            the scale used for quantization.
    """
    assert x.ndim == 2
    x = x.to(torch.float32)
    x_absmax = x.view(x.shape[0], -1).abs().max(dim=1, keepdim=True)[0]
    q_max = 2**(n_bits - 1) - 1
    q_min = -2**(n_bits - 1)
    scale = x_absmax / (2**(n_bits - 1) - 1)
    x_q = torch.round(x / scale).clamp(q_min, q_max).to(dtype)
    return x_q, scale


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_N': 64,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4)
    ],
    key=['N', 'K'],
)
@triton.jit
def _linear(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rms_scale_ptr,
    linear_scale_ptr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B`, and store the result in output
    tensor `C`.

    The function applies auto-tuning for optimal performance and uses Just-in-
    Time compilation.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_N': 64,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4)
    ],
    key=['N', 'K'],
)
@triton.jit
def _linear_add(
    A,
    B,
    C,
    residual_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rms_scale_ptr,
    linear_scale_ptr,
):
    """Triton-accelerated function used to perform a linear operation (dot
    product) on input tensors `A` and `B`, with addition of residual.

    The result is stored in tensor `C`. The function applies auto-tuning for
    optimal performance and uses Just-in-Time compilation.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale
    c = c.to(residual_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    residual_ptrs = (residual_ptr + stride_cm * offs_cm[:, None] +
                     stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    residual = tl.load(residual_ptrs, mask=c_mask, other=0.)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c + residual, mask=c_mask)


def matmul_kernel_dynamic_quant(a,
                                b,
                                rms_scale,
                                linear_scale,
                                residual=None,
                                bias=None,
                                output_dtype=torch.float16):
    """This function performs matrix multiplication with dynamic quantization.

    It takes two input tensors `a` and `b`, scales them with `rms_scale` and
    `linear_scale`, and optionally adds a `residual` tensor and a `bias`. The
    output is returned in the specified `output_dtype`.
    """

    assert a.shape[-1] == b.shape[-1]
    assert b.ndim == 2 and b.is_contiguous()
    M = a.numel() // a.shape[-1]
    N, K = b.shape
    c_shape = a.shape[:-1] + (N, )
    if residual is not None:
        assert residual.shape == c_shape
        assert residual.is_contiguous()
    c = a.new_empty(c_shape, dtype=output_dtype)

    BLOCK_M = 128
    if M < BLOCK_M:
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_M = max(BLOCK_M, 16)

    def grid(META):
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, META['BLOCK_N']), )

    if residual is not None:
        _linear_add[grid](a,
                          b,
                          c,
                          residual,
                          M,
                          N,
                          K,
                          a.stride(-2),
                          a.stride(-1),
                          b.stride(1),
                          b.stride(0),
                          c.stride(-2),
                          c.stride(-1),
                          BLOCK_M=BLOCK_M,
                          GROUP_SIZE_M=8,
                          rms_scale_ptr=rms_scale,
                          linear_scale_ptr=linear_scale)
    else:
        _linear[grid](a,
                      b,
                      c,
                      M,
                      N,
                      K,
                      a.stride(-2),
                      a.stride(-1),
                      b.stride(1),
                      b.stride(0),
                      c.stride(-2),
                      c.stride(-1),
                      BLOCK_M=BLOCK_M,
                      GROUP_SIZE_M=8,
                      rms_scale_ptr=rms_scale,
                      linear_scale_ptr=linear_scale)
    if bias is not None:
        c += bias

    return c


@triton.jit
def _per_token_quant_int8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token quantization on a
    tensor.

    This function converts the tensor values into signed 8-bit integers.
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    y_ptr += row * y_stride
    y_q_ptr += row * y_stride
    y_s_ptr += row

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / 127
    y_q = tl.math.round(y / y_s).to(tl.int8)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_quant_int8(x, eps):
    """Function to perform per-token quantization on an input tensor `x`.

    It converts the tensor values into signed 8-bit integers and returns the
    quantized tensor along with the scaling factor used for quantization.
    """

    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_s = torch.empty(x.shape[:-1] + (1, ),
                      device=x.device,
                      dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    # enqueue kernel
    _per_token_quant_int8[(M, )](x,
                                 x_q,
                                 x_s,
                                 x.stride(-2),
                                 N,
                                 eps,
                                 BLOCK=BLOCK,
                                 num_warps=num_warps)

    return x_q, x_s


@triton.jit
def _rms_norm_fwd_fused_dynamic_symmetric(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Scale,  # pointer to the scales of the output activation
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    """A Triton kernel that calculates Root Mean Square (RMS) normalization
    with fused dynamic symmetric quantization."""
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    _var = x * x
    var = tl.sum(_var, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    w = tl.load(W + cols, mask=mask)
    x_hat = x * rstd
    y = x_hat * w

    scale = tl.max(tl.abs(y)).to(tl.float32) / 127
    tl.store(Scale + row, scale)

    y = tl.math.round(y / scale)
    y = tl.minimum(y, 127)
    y = tl.maximum(y, -128)
    tl.store(Y + cols, y, mask=mask)


def rms_norm_dynamic_quant(x, w, eps):
    """Performs RMS normalization with dynamic quantization.

    The function reshapes the input tensor `x`, creates an empty tensor `y`
    with the same shape as `x`, and calculates RMS normalization on the
    reshaped `x` using a Triton kernel `_rms_norm_fwd_fused_dynamic_symmetric`.
    """

    x_arg = x.flatten(0, -2)
    y = torch.empty_like(x, dtype=torch.int8)
    M, K = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(K))
    if K > BLOCK_SIZE:
        raise RuntimeError(
            "This rms norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    scale = x.new_empty(x.shape[:-1] + (1, ), dtype=torch.float32)
    _rms_norm_fwd_fused_dynamic_symmetric[(M, )](x_arg,
                                                 y,
                                                 w,
                                                 scale,
                                                 x_arg.stride(0),
                                                 K,
                                                 eps,
                                                 BLOCK_SIZE=BLOCK_SIZE,
                                                 num_warps=num_warps)
    return y, scale

