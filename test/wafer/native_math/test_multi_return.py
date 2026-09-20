# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""Original Ascend cross-entropy/gradient kernels with native Wafer tanh.

Host reductions and the independent autograd reference run on CPU; all Triton
kernels receive native TXDA tensors.
"""
import pytest
import torch
import torch_txda  # noqa: F401 -- registers the TXDA device
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,  # set it as constexpr since reduction is always known at compile time
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    RETURN_Z_LOSS (int): The boolean value to decide whether storing z loss to z_loss_ptr or not. It must be 0 or 1.
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    """

    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(
        tl.float32
    )  # we need to store the original value of X_y for the loss calculation
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * libdevice.tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * libdevice.tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(
                    tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0)
                )
            else:
                scaled_x_sum += tl.sum(
                    tl.where(X_offsets < n_cols, -eps * X_block, 0.0)
                )
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
    #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
    #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
    lse = m + tl.log(d)

    # 4. [Online Softmax] Second pass: compute gradients
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            intermediate = libdevice.tanh(X_block / softcap)
            X_block = softcap * intermediate

        if not HAS_WEIGHT:
            X_block = tl.exp(X_block - m) / d
            # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
            X_block += 2 * lse_square_scale * lse * X_block
            # smoothing term
            X_block += -eps
            # special handle dx_y
            X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
            # reduction scale
            if reduction == "mean":
                X_block = X_block / n_non_ignore
        else:
            weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
            softmax_X = tl.exp(X_block - m) / d
            # derivative of original_loss
            dloss_ori = (1 - label_smoothing) * softmax_X
            # specially handle dx_y
            dloss_ori = tl.where(
                X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing)
            )
            dloss_ori = dloss_ori * weight_y
            # derivative of smooth_loss
            dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
            # derivative of z-loss
            dz_loss = 2 * lse_square_scale * lse * softmax_X
            # reduction scale
            if reduction == "mean":
                dloss_ori = dloss_ori / sum_non_ignore_weight
                dloss_smooth = dloss_smooth / sum_non_ignore_weight
                dz_loss = dz_loss / n_non_ignore
            # derivative of total_loss
            X_block = dloss_ori + dloss_smooth + dz_loss

        # chain rule softcapping
        # d(softcap * libdevice.tanh(x / softcap)) = (1 - tanh^2(x / softcap))
        if HAS_SOFTCAPPING:
            X_block = X_block * (1 - intermediate * intermediate)

        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    #      = X_y - m - log d = X_y - lse
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    # An auxiliary loss, z_loss
    z_loss = lse_square_scale * lse * lse
    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)

@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def run_cross_entropy(host, target, grad):
    rows, cols = host.shape
    x, y = (host).to("txda"), (target).to("txda")
    loss = (torch.zeros(rows, dtype=host.dtype)).to("txda")
    z_loss = (torch.zeros(rows, dtype=host.dtype)).to("txda")
    non_ignore = int((target != 0).sum())
    # Preserve the original kernel, grid, softcap, smoothing and reduction.
    # Framework bookkeeping/reductions are CPU-side, not another NPU OP.
    liger_cross_entropy_kernel[(rows,)](
        x, x.stride(0), y, y.stride(0), None, loss, z_loss, 1, cols,
        non_ignore, non_ignore, 0.0, 0, 1e-4, 0.1, "mean", 30.0, True,
        min(32768, triton.next_power_of_2(cols)), False, True,
    )
    # The original backward multiplies its in-place gradient by grad_output.
    element_mul_kernel[(rows,)](x, x.stride(0), (grad).to("txda"), cols,
                               min(32768, triton.next_power_of_2(cols)))
    return loss.cpu().sum(), z_loss.cpu().sum(), x.cpu()


def reference_cross_entropy(host, target, grad):
    # Independent CPU autograd oracle, including softcap, label smoothing,
    # ignored labels and z-loss. Round per-row stores as in the original ABI.
    x = host.float().requires_grad_(True)
    capped = 30.0 * torch.tanh(x / 30.0)
    lse = torch.logsumexp(capped, dim=-1)
    loss = torch.nn.functional.cross_entropy(capped, target, ignore_index=0,
                                             label_smoothing=0.1, reduction="none")
    mask = target != 0
    z = torch.where(mask, 1e-4 * lse.square(), 0.0)
    losses = (loss + z) / mask.sum()
    losses.sum().backward()
    # The kernel first stores the unscaled gradient in the input's dtype;
    # the separate backward kernel then multiplies it by the scalar gradient.
    expected_grad = (x.grad.to(host.dtype) * grad).to(host.dtype)
    return losses.to(host.dtype).sum(), (z / mask.sum()).to(host.dtype).sum(), expected_grad


@pytest.mark.parametrize("B,T,V", [(2, 512, 4096)])
@pytest.mark.parametrize("scalar,dtype,atol,rtol", [
    (1.0, torch.bfloat16, 1e-8, 5e-2),
    (1.0, torch.float32, 1e-8, 1e-6),
])
def test_correctness_functional(B, T, V, scalar, dtype, atol, rtol):
    torch.manual_seed(0)
    host = torch.randn(B * T, V, dtype=dtype) * scalar
    target = torch.randint(0, V, (B * T,), dtype=torch.long)
    grad = torch.randn((), dtype=dtype)
    first = run_cross_entropy(host, target, grad)
    second = run_cross_entropy(host, target, grad)
    reference = reference_cross_entropy(host, target, grad)
    for actual, repeat, expected in zip(first, second, reference):
        # Keep the source's two-execution comparisons and add independent truth.
        assert torch.allclose(actual, repeat, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
