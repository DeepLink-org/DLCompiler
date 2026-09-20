#!/usr/bin/env python3
"""Validate the paired torch/torch_txda/txops release with native TXDNN calls."""

import argparse
import importlib.metadata

import torch
import torch_txda  # noqa: F401
import txops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matmul",
        action="store_true",
        help="Also reproduce the vendor BF16 matmul case (use an external timeout)",
    )
    args = parser.parse_args()
    for name in ("torch", "torch-txda", "txops", "triton"):
        print(f"{name}: {importlib.metadata.version(name)}", flush=True)
    assert torch.txda.is_available() and torch.txda.device_count() > 0
    host = torch.arange(256, dtype=torch.float32).reshape(1, 256)
    torch.testing.assert_close(host.to("txda").cpu(), host, rtol=0, atol=0)
    handle = txops.txdnn.create()
    layout = txops.txdnn.txLayout.NCX
    # Use TXDNN's descriptor API and compare its returned host tensor.
    lhs = txops.txdnn.tensor_like(handle, host, layout)
    rhs = txops.txdnn.tensor_like(handle, torch.ones_like(host) * 2, layout)
    result = txops.txdnn.add(handle, lhs, rhs).tensor()
    torch.testing.assert_close(result, host + 2, rtol=0, atol=0)
    print("PASS native txdnn.add: 256 FP32 elements", flush=True)
    try:
        txops.txdnn.add(handle, object(), rhs)
    except TypeError:
        print("PASS native txdnn invalid argument propagation", flush=True)
    else:
        raise AssertionError("TXDNN accepted an invalid tensor descriptor")
    if not args.matmul:
        return
    a = torch.ones((128, 64), dtype=torch.bfloat16)
    b = torch.ones((64, 256), dtype=torch.bfloat16)
    da, db = (txops.txdnn.tensor_like(handle, value, layout) for value in (a, b))
    print("Launching vendor txdnn.matmul", flush=True)
    result = txops.txdnn.matmul(handle, da, db).tensor()
    torch.testing.assert_close(result, a @ b, rtol=0, atol=0)
    print("PASS native txdnn.matmul: BF16 128x64 @ 64x256", flush=True)


if __name__ == "__main__":
    main()
