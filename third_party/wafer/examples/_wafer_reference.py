"""Independent CPU decoding for the scaled-dot numerical oracle."""

import torch


def upcast_mxfp_cpu(packed, scale, format_name, dtype):
    """Decode E2M1/E4M3/E5M2 plus E8M0 scales, rounding like the original oracle.

    packed is contiguous and packed along its last dimension. For E2M1, the low
    nibble precedes the high nibble. A scale byte of 0 maps to zero and 255 to NaN,
    matching the bitcast/where logic in the original Triton reference kernel.
    """
    if packed.device.type != "cpu" or scale.device.type != "cpu":
        raise ValueError("The Wafer numerical oracle must run on CPU")
    packed = packed.contiguous()
    if format_name == "e2m1":
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=dtype)
        values = magnitudes[(codes & 7).long()]
        values = torch.where((codes & 8) != 0, -values, values)
    else:
        fp8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[format_name]
        values = packed.view(fp8).to(dtype)
    scales = (scale.to(torch.int32) << 23).view(torch.float32).to(dtype)
    scales = scales.repeat_interleave(32, dim=-1)
    result = values * scales
    return torch.where(scale.repeat_interleave(32, dim=-1) == 255, float("nan"), result)
