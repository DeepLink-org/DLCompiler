from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import time

from dlblas import get_op, get_list_op_names

def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity

def fused_topkgating(
    logits: Tensor,
    k: int,
    capacity_factor: float = 1.0, 
    min_capacity: int = 2
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
    # everything is in fp32 in this function
   
    gates = F.softmax(logits, dim=1)
    
    num_experts = int(gates.shape[1])

    capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))

    # Create a mask by top-k experts
    indices_s = torch.topk(gates, k, dim=1).indices
    indices_s = indices_s.permute(1, 0).reshape(-1)
    masks = F.one_hot(indices_s, num_classes=num_experts)
    
    # Compute locations in capacity buffer
    locations = torch.cumsum(masks, dim=0) - 1
    # reshape (s,e) to (k,s,e)
    masks = masks.reshape(-1, gates.shape[0], num_experts)
    locations = locations.reshape(-1, gates.shape[0], num_experts)

    # gating decisions
    exp_counts = torch.sum(masks[0], dim=0).detach().to("cpu")

    # print(f"gates:{gates}")    

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # print(f"l_aux:{l_aux}")
    # print(f"ce:{ce}")
    # print(f"masks:{masks}")
    # print(f"locations:{locations.shape}")
    
    # Remove locations outside capacity from mask
    masks *= torch.lt(locations, capacity)

    # Store the capacity location for each token
    locations_s = torch.sum(locations * masks, dim=2)
   
    

    # Normalize gate probabilities
    mask_float = masks.type_as(logits)
    gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    denom_s = torch.sum(gate_s, dim=0)
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gate_s /= denom_s

    

    # Calculate combine_weights and dispatch_mask
    gate_all = torch.einsum("ks,kse->kse", gate_s, mask_float)
    
    # print(f"capacity:{capacity}")
    
    
    # ---- test begin ----
    # k, s, e, c= locations_s.shape[0], locations_s.shape[1], logits.shape[1], capacity
    # combine_weights_test = torch.zeros((s, e, c), device=logits.device, dtype=logits.dtype)
    # for idx_k in range(k):
    #     for idx_s in range(s):
    #         combine_weights_test[idx_s,:,locations_s[idx_k][idx_s]] += gate_all[idx_k, idx_s,:]
    # --replace---
    locations_sc = F.one_hot(locations_s, num_classes=capacity).type_as(logits)
    combine_sec = torch.einsum("kse,ksc->ksec", gate_all, locations_sc)
    combine_weights = torch.sum(combine_sec, dim=0)
    
    # assert torch.allclose(combine_weights, combine_weights_test)
    # --- test end ----
    
    # torch.cuda.synchronize(logits.device)
    # t0 = time.time()

    dispatch_mask = combine_weights.bool()

    # torch.cuda.synchronize(logits.device)
    # print(f"torch time:{(time.time() - t0) * 1000.0}")
    
    # return l_aux, masks, locations_s, exp_counts
    return l_aux, combine_weights, dispatch_mask, exp_counts



def test():
    device_ = torch.device('cuda:6')
    torch.cuda.set_device(device_)
    k, SeqLen, NumberExperts = 8, 33, 64
    # k, SeqLen, NumberExperts = 2, 4096, 64
    shape = (SeqLen, NumberExperts)
    logits_torch = torch.randn(shape, device=device_, requires_grad=True)
    capacity_factor: float = 1.0
    min_capacity: int = 2
    
    with torch.no_grad():
        logits_triton = logits_torch.clone()

    logits_triton.requires_grad = True
    
    model_torch = fused_topkgating

    model_triton = get_op("topk_gating", (logits_triton, k, capacity_factor, min_capacity))
    
    output1_torch, output2_torch, output3_torch, output4_torch = model_torch(logits_torch, k, capacity_factor, min_capacity)
    output1_triton, output2_triton, output3_triton, output4_triton = model_triton(logits_triton, k, capacity_factor, min_capacity)

    
    
    assert output1_torch.shape == output1_triton.shape
    assert torch.allclose(output1_torch, output1_triton)
    assert output2_torch.shape == output2_triton.shape
    assert torch.allclose(output2_torch, output2_triton)
    assert output3_torch.shape == output3_triton.shape
    assert torch.allclose(output3_torch, output3_triton)
    assert torch.allclose(output4_torch, output4_triton)
   
    loss_torch = torch.sum(torch.mean(output1_torch * output2_torch))
    loss_triton = torch.sum(torch.mean(output1_triton * output2_triton))
    
    assert torch.allclose(loss_torch, loss_triton)

    # for backward 
    # loss_torch.backward()
    # loss_triton.backward()
    
    # assert logits_torch.grad.shape == logits_triton.grad.shape
    # assert torch.allclose(logits_torch.grad, logits_triton.grad)
    


    
    # vary seq length for fixed head and batch=4
    configs = []

    configs.append(
        triton.testing.Benchmark(
            x_names=["Experts"],
            x_vals=[NumberExperts],
            line_arg="provider",

            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],

            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"top{k}-gating-seqLen:{SeqLen}",
            args={
                "SeqLen": SeqLen
            },
        ))
    @triton.testing.perf_report(configs)
    def bench_top2gating(SeqLen, Experts, provider, device=device_):
        warmup = 10
        rep = 10
        shape = (SeqLen, Experts)
        logits = torch.randn(shape, device=device, requires_grad=True)


        if "triton" in provider:
            fn = lambda: model_triton(logits, k)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        if "pytorch" in provider:
            fn = lambda : model_torch(logits, k)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        return ms
    
    bench_top2gating.run(show_plots=True, print_data=True)





if __name__ == '__main__':
    test()
    print("sucessfully!")
