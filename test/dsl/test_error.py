import triton
import triton.language as tl
import torch
import re

def test_sqrtt_error():
    @triton.jit
    def test_kernel(X, Y, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offsets)
        y = tl.sqrtt(x)                     # ERROR
        tl.store(Y + offsets, y)

    x = torch.randn(16, device='npu')
    y = torch.zeros_like(x)

    try:
        test_kernel[(1,)](x, y, BLOCK=16)
        assert False, "Expected a compilation error, but no error was raised"
    except Exception as e:
        error_msg = str(e)
        print("Error message:", error_msg)
        # CHECK
        assert "ERROR LINE 12, COL 8" in error_msg, "Error line not found in message"
        assert "y = tl.sqrtt(x)" in error_msg, "Expected source line not found"
        assert "        ^" in error_msg, "Pointer ^ not found in error message"

        print("✅ Test passed: Correct error message detected.")

# 运行测试
if __name__ == "__main__":
    test_sqrtt_error()
