import torch
import triton
import triton.language as tl

@triton.jit
def _fused_kernel1(
    logits1,
    ones,
    stride_s,
    stride_b,
    BLOCK_S: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    pid = tl.program_id(axis=0)
    
    offs_s = pid * BLOCK_S  + tl.arange(0, BLOCK_S)[:,None]
    logits1_col = tl.arange(0, BLOCK_E)[None,:]

    logits1_ptrs = logits1 + offs_s * stride_s + logits1_col
    logits1_data = tl.load(logits1_ptrs, mask=logits1_col < EXPERTS, other=-float("inf"))
    ones_ptr = ones + offs_s * stride_b + tl.arange(0, 1)[None,:]
    ones_data = tl.max(logits1_data, axis=1) #tl.load(ones_ptr)
    ones_data2 = tl.broadcast_to(tl.reshape(ones_data, (BLOCK_S, 1)), (BLOCK_S, BLOCK_E))
    # tl.expand_dims
    a = tl.where(logits1_data == ones_data2, 0.0,logits1_data)
    tl.store(logits1_ptrs, a, mask=logits1_col < EXPERTS)

def test():
    a = torch.randn((4, 4)).cuda()
    b = torch.randn((4, 1)).cuda()
    print(f"a:{a}")
    print(f"b:{b}")
    _fused_kernel1[1,](a, b, a.stride(0), b.stride(0), a.shape[0], a.shape[1], a.shape[1])
    print(f"res:{a}")

test()