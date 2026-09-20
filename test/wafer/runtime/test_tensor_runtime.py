"""Native counterpart to Ascend load/store cases; exercises the tensor ABI."""
import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def copy_kernel(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, tl.load(X + i, i < N, 0), i < N)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16,
                                    torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("size", [1, 31, 256, 257])
def test_native_copy(dtype, size):
    host = (torch.arange(size) % 61 - 30).to(dtype)
    x = host.to("txda")
    # Keep the offset/boundary check explicit after removing device_tensor.
    storage = torch.full((size + 128,), -83, dtype=dtype).to("txda")
    y = storage[64:64 + size]
    copy_kernel[(triton.cdiv(size, 256),)](x, y, size, 256)
    torch.testing.assert_close(y.cpu(), host, rtol=0, atol=0)
    actual = storage.cpu()
    torch.testing.assert_close(actual[:64], torch.full((64,), -83, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(actual[64 + size:], torch.full((64,), -83, dtype=dtype), rtol=0, atol=0)


def test_explicit_stream_and_cache():
    # Framework events and the compiler launcher must use the same stream.
    stream = torch.txda.Stream()
    host = torch.arange(257, dtype=torch.float32)
    storage = torch.cat((host, torch.zeros(15), torch.zeros_like(host))).to("txda")
    x, y = storage[:257], storage[272:]
    assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    with torch.txda.stream(stream):
        driver = triton.runtime.driver.active
        assert driver.get_current_stream(torch.txda.current_device()) == stream.txda_stream
        start, end = torch.txda.Event(enable_timing=True), torch.txda.Event(enable_timing=True)
        start.record()
        first = copy_kernel[(2,)](x, y, 257, 256)
        # The intermediate stays on TXDA between launches; both views retain
        # their shared allocation and the destination's nonzero offset.
        second = copy_kernel[(2,)](y, x, 257, 256)
        end.record()
    end.synchronize()
    assert first is second
    assert start.elapsed_time(end) >= 0
    torch.testing.assert_close(y.cpu(), host, rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), host, rtol=0, atol=0)
