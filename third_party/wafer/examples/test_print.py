import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.load(Y + tl.arange(0, BLOCK))
    tl.device_print("x: ", x, hex=True)
    tl.device_print("constant : ", tl.constexpr(42))
    tl.store(Y + tl.arange(0, BLOCK), x)


def test_print():
    x = torch.arange(16, dtype=torch.int32)
    x.reshape(4, 4)
    y = torch.zeros_like(x)
    x_txda = x.to("txda")
    y_txda = y.to("txda")
    kernel_device_print[(1, )](x_txda, y_txda, BLOCK=16)
    with torch.no_grad():
        y.copy_(y_txda.cpu())
    torch.testing.assert_close(y, x)


if __name__ == "__main__":
    # Run the test with pytest
    test_print()
