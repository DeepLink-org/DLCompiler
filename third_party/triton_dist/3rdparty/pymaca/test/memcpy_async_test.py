import torch
import triton.pymaca.maca as maca

src_tensor = torch.randn(100, device="cuda", dtype=torch.float)
dst_tensor = torch.randn(100, device="cuda", dtype=torch.float)
size = 100 * 4
stream = torch.cuda.current_stream().cuda_stream
src_ptr = src_tensor
print("origin dst: ", dst_tensor)
print("src", src_tensor)
(err, ) = maca.mcMemcpyAsync(dst_tensor.data_ptr(), src_tensor.data_ptr(), size, maca.mcMemcpyKind.mcMemcpyDefault,
                             stream)
print(err)

if isinstance(err, maca.mcError_t):
    if err != maca.mcError_t.mcSuccess:
        raise RuntimeError(f"maca Error: {err}: {maca.mcGetErrorString(err)}")
else:
    raise RuntimeError(f"Unknown error type: {err}")

print("final dst: ", dst_tensor)
