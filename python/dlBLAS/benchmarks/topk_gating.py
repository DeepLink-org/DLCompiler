from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import time
from torch.profiler import profile, record_function, ProfilerActivity
import dlblas
from python.dlBLAS.dlblas.utils.device_utils import get_idle_device


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
    logits: Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
    # everything is in fp32 in this function

    gates = F.softmax(logits, dim=1)

    num_experts = int(gates.shape[1])

    capacity = _capacity(
        gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)
    )

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

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts
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

    # ---- test begin ----
    # k, s, e, c= locations_s.shape[0], locations_s.shape[1], logits.shape[1], capacity
    # combine_weights_test = torch.zeros((s, e, c), device=logits.device, dtype=logits.dtype)
    # for idx_k in range(k):
    #     for idx_s in range(s):
    #         combine_weights_test[idx_s,:,locations_s[idx_k][idx_s]] += gate_all[idx_k, idx_s,:]
    # dispatch_mask = combine_weights_test.bool()
    # return l_aux, combine_weights_test, dispatch_mask, exp_counts
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


from functorch.compile import aot_module, make_boxed_func, aot_function
from torch._dynamo.backends.common import aot_autograd


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    # print(">>> FX graph:")
    # gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return make_boxed_func(gm.forward)  # return a python callable


my_aot_backend = aot_autograd(fw_compiler=my_compiler)

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.3.x
    from tutel import moe as tutel_moe

    TUTEL_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass
from collections import namedtuple

GatingTokenRearrangeInfo = namedtuple(
    "GatingTokenRearrangeInfo",
    ["token_rearranged_ec_idx", "token_exp_weights", "expert_select_token_idx"],
)


# @torch.compile(backend=my_aot_backend)
def fused_topkgating_opt(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    enable_token_rearrange_opt: bool = True,
    use_tutel: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    capacity = _capacity(
        gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)
    )
    # Create a mask by top-k experts
    indices_s = torch.topk(gates, k, dim=1).indices.t()
    masks = F.one_hot(indices_s.reshape(-1), num_classes=num_experts)

    # Compute locations in capacity buffer
    if use_tutel and TUTEL_INSTALLED:
        locations = tutel_moe.fast_cumsum_sub_one(masks)
    else:
        locations = torch.cumsum(masks, dim=0) - 1

    # reshape (s,e) to (k,s,e)
    masks = masks.reshape(-1, gates.shape[0], num_experts)
    locations = locations.reshape(-1, gates.shape[0], num_experts)

    # gating decisions
    # exp_counts = torch.sum(masks[0], dim=0).detach()

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    masks *= torch.lt(locations, capacity)

    # Store the capacity location for each token
    locations_s = torch.sum(locations * masks, dim=2)

    # Normalize gate probabilities
    mask_float = masks.type_as(logits)
    # gate_s = einsum("se,kse->ks", gates, mask_float)
    gate_s, indices_s = torch.max(gates * mask_float, dim=2)
    denom_s = torch.sum(gate_s, dim=0)
    # Avoid divide-by-zero
    clamp_denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gate_s /= clamp_denom_s

    # if enable_token_rearrange_opt:
    token_rearranged_ec_idx = indices_s.int() * capacity + locations_s.int()
    # shape：[S, E]->[C, E]->[E, C]->[E*C]
    # import pdb; pdb.set_trace()
    token_sel_exp_int_mask = masks * torch.arange(
        k, 0, -1, device=masks.device
    ).reshape(k, 1, 1)
    expert_sel_top_c_token_idx = torch.topk(
        torch.sum(token_sel_exp_int_mask, dim=0), k=capacity, dim=0, sorted=True
    )[1]
    expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(
        num_experts * capacity
    )
    token_rearranged_ec_idx = token_rearranged_ec_idx.reshape(-1)
    token_exp_weights = gate_s.reshape(-1)

    top2_gating_token_infos = GatingTokenRearrangeInfo(
        token_rearranged_ec_idx=token_rearranged_ec_idx,
        token_exp_weights=token_exp_weights,
        expert_select_token_idx=expert_select_token_idx,
    )
    return l_aux, top2_gating_token_infos
    # else:
    #     # Calculate combine_weights and dispatch_mask
    #     gate_all = torch.einsum("ks,kse->kse", gate_s, mask_float)
    #     locations_sc = F.one_hot(locations_s, num_classes=capacity).type_as(logits)
    #     combine_sec = torch.einsum("kse,ksc->ksec", gate_all, locations_sc)
    #     combine_weights = torch.sum(combine_sec, dim=0)
    #     dispatch_mask = combine_weights.bool()

    #     return l_aux, combine_weights, dispatch_mask


device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)


def test():
    # k, SeqLen, NumberExperts = 4, 16, 8
    k, SeqLen, NumberExperts = 8, 4096, 64
    shape = (SeqLen, NumberExperts)
    logits_torch = torch.randn(shape, device=device_, requires_grad=True)
    capacity_factor: float = 1.0
    min_capacity: int = 2
    enable_token_rearrange_opt = True

    with torch.no_grad():
        logits_triton = logits_torch.clone()
        logits_test = logits_torch.clone()

    logits_triton.requires_grad = True
    logits_test.requires_grad = True

    model_torch = fused_topkgating_opt
    model_triton = dlblas.topk_gating

    output1_torch, out_torch_pack = model_torch(
        logits_torch, k, capacity_factor, min_capacity, enable_token_rearrange_opt
    )
    output2_torch = out_torch_pack.token_rearranged_ec_idx
    output3_torch = out_torch_pack.token_exp_weights
    output4_torch = out_torch_pack.expert_select_token_idx
    output1_triton, output2_triton, output3_triton, output4_triton = model_triton(
        logits_triton, k, capacity_factor, min_capacity, False
    )

    assert output1_torch.shape == output1_triton.shape
    assert torch.allclose(output1_torch, output1_triton)
    assert output2_torch.shape == output2_triton.shape
    assert torch.allclose(output2_torch, output2_triton)
    assert output3_torch.shape == output3_triton.shape
    assert torch.allclose(output3_torch, output3_triton)
    assert torch.allclose(output4_torch, output4_triton)

    loss_torch = torch.sum(torch.mean(output1_torch * output3_torch))
    loss_triton = torch.sum(torch.mean(output1_triton * output3_triton))

    assert torch.allclose(loss_torch, loss_triton)

    # # for backward
    dout_torch = torch.randn_like(loss_torch)
    with torch.no_grad():
        dout_triton = dout_torch.clone()
    loss_torch.backward(dout_torch, retain_graph=True)
    loss_triton.backward(dout_triton, retain_graph=True)

    print(
        f"logits grad max diff: {(logits_torch.grad - logits_triton.grad).abs().max().item()}"
    )
    print(
        f"logits grad mean diff: {(logits_torch.grad - logits_triton.grad).abs().mean().item()}"
    )

    assert logits_torch.grad.shape == logits_triton.grad.shape
    assert torch.allclose(logits_torch.grad, logits_triton.grad, rtol=1e-8, atol=1e-8)

    if False:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
        ) as prof:
            output1_torch, out_torch_pack = model_torch(
                logits_torch,
                k,
                capacity_factor,
                min_capacity,
                enable_token_rearrange_opt,
            )
            output1_triton, output2_triton, output3_triton, output4_triton = (
                model_triton(logits_triton, k, capacity_factor, min_capacity, True)
            )
            loss_torch = torch.sum(torch.mean(output1_torch * output3_torch))
            loss_triton = torch.sum(torch.mean(output1_triton * output3_triton))
            dout_torch = torch.randn_like(loss_torch)
            with torch.no_grad():
                dout_triton = dout_torch.clone()
            loss_torch.backward(dout_torch, retain_graph=True)
            loss_triton.backward(dout_triton, retain_graph=True)
            assert torch.allclose(
                logits_torch.grad, logits_triton.grad, rtol=1e-8, atol=1e-8
            )
        prof.export_chrome_trace(f"./trace_{time.time_ns()}.json")

    # vary seq length for fixed head and batch=4
    configs = []

    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd", "bwd"],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"Experts{NumberExperts}-top{k}-gating-seqLen:{SeqLen}",
            args={"SeqLen": SeqLen},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_top2gating(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 200
        shape = (SeqLen, NumberExperts)
        logits = torch.randn(shape, device=device, requires_grad=True)

        if "triton" in provider:
            if "fwd" == op:
                fn = lambda: model_triton(
                    logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt
                )
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif "bwd" == op:
                # def iter(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt):
                out0, out1, out2, _ = model_triton(
                    logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt
                )
                loss = torch.sum(torch.mean(out0 * out2))
                # loss.backward(retain_graph=True)
                bwd_fn = lambda: loss.backward(retain_graph=True)
                # bwd_fn = lambda: iter(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        if "pytorch" in provider:
            if "fwd" == op:
                fn = lambda: model_torch(
                    logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt
                )
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif "bwd" == op:
                # def iter(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt):
                out0, out_torch_pack = model_torch(
                    logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt
                )
                output2_torch = out_torch_pack.token_rearranged_ec_idx
                output3_torch = out_torch_pack.token_exp_weights
                output4_torch = out_torch_pack.expert_select_token_idx
                loss = torch.sum(torch.mean(out0 * output3_torch))
                # loss.backward(retain_graph=True)

                bwd_fn = lambda: loss.backward(retain_graph=True)
                # bwd_fn = lambda: iter(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        return ms

    bench_top2gating.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")
