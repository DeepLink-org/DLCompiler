import math
import pytest
import triton
import triton.language as tl
import torch
import test_common


@triton.jit
def _conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    output_ptr,
    bias_ptr,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    h_in,
    w_in,
    h_out,
    w_out,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(h_out * w_out, BLOCK_SIZE)
    pid_b = pid // (num_pid_n * out_channels)
    pid_c = (pid // num_pid_n) % out_channels
    pid_n = pid % num_pid_n

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_h = offs_n // w_out
    offs_w = offs_n % w_out
    mask = offs_n < (h_out * w_out)

    bias = tl.load(bias_ptr + pid_c) if bias_ptr is not None else 0.0
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + bias

    x_offset = pid_b * stride_xn
    w_offset = pid_c * stride_wc

    for c_in in range(in_channels):
        w_c_offset = w_offset + c_in * stride_wn
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                h_in_val = offs_h + padding_h - kh * dilation_h
                w_in_val = offs_w + padding_w - kw * dilation_w

                cond_h = (h_in_val >= 0) & (h_in_val < h_in * stride_h)
                cond_w = (w_in_val >= 0) & (w_in_val < w_in * stride_w)
                cond = cond_h & cond_w & mask

                cond_h_stride = (h_in_val % stride_h) == 0
                cond_w_stride = (w_in_val % stride_w) == 0
                final_cond = cond & cond_h_stride & cond_w_stride

                h_in_idx = tl.where(final_cond, h_in_val // stride_h, 0)
                w_in_idx = tl.where(final_cond, w_in_val // stride_w, 0)

                x_offsets = (
                    x_offset
                    + c_in * stride_xc
                    + h_in_idx * stride_xh
                    + w_in_idx * stride_xw
                )

                w_val = tl.load(w_ptr + w_c_offset + kh * stride_wh + kw * stride_ww)
                x_vals = tl.load(x_ptr + x_offsets, mask=final_cond, other=0.0)
                acc += x_vals * w_val

    out_offset = (
        pid_b * stride_on + pid_c * stride_oc + offs_h * stride_oh + offs_w * stride_ow
    )
    tl.store(output_ptr + out_offset, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = torch.nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch, in_channels, h_in, w_in = x.shape
        h_out = (
            (h_in - 1) * self.stride
            - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + 1
        )
        w_out = (
            (w_in - 1) * self.stride
            - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + 1
        )

        out = torch.empty(
            (batch, self.out_channels, h_out, w_out), device=x.device, dtype=x.dtype
        )

        stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
        stride_wn, stride_wc, stride_wh, stride_ww = self.weight.stride()
        stride_on, stride_oc, stride_oh, stride_ow = out.stride()

        total_blocks = batch * self.out_channels * math.ceil((h_out * w_out) / 64)

        _conv_transpose2d_kernel[(total_blocks,)](
            x,
            self.weight,
            out,
            self.bias,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            h_in,
            w_in,
            h_out,
            w_out,
            stride_xn,
            stride_xc,
            stride_xh,
            stride_xw,
            stride_wn,
            stride_wc,
            stride_wh,
            stride_ww,
            stride_on,
            stride_oc,
            stride_oh,
            stride_ow,
            64,
        )
        return out


# ===========================================================
# Parameterize ALL inputs via pytest.mark.parametrize
# ===========================================================
@pytest.mark.parametrize(
    "batch,in_ch,out_ch,kernel,H,W,stride,padding,dilation,dtype",
    [
        # your requested single-configuration repeated for two dtypes
        # (2, 4, 4, 3, 8, 8, 2, 1, 1, "float32"),
        # TODO bisheng-compiler 修复
        (2, 4, 4, 3, 8, 8, 2, 1, 1, "float16"),
        (2, 32, 32, 3, 32, 32, 5, 1, 2, "float16"),
    ],
)
def test_conv_transpose2d_param(
    batch, in_ch, out_ch, kernel, H, W, stride, padding, dilation, dtype
):
    device = torch.device("npu")

    # generate input on npu using test_common helper
    x = test_common.generate_tensor((batch, in_ch, H, W), dtype).npu()

    model = ModelNew(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )

    model.to(device)
    model.weight.data = (
        model.weight.data.to(device).to(eval("torch." + dtype)).contiguous()
    )
    if model.bias is not None:
        model.bias.data = (
            model.bias.data.to(device).to(eval("torch." + dtype)).contiguous()
        )

    ref = (
        torch.nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        .to(device)
        .to(eval("torch." + dtype))
    )

    with torch.no_grad():
        ref.weight.data.copy_(model.weight.data)
        ref.bias.data.copy_(model.bias.data)

    with torch.no_grad():
        y_cal = model(x)
        y_ref = ref(x)

    test_common.validate_cmp(dtype, y_cal, y_ref)
