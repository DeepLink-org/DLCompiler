import torch
import torch.nn.functional as F
import triton
import dlblas

from einops import rearrange, repeat

def torch_layernorm_gated(x_ref, weight_ref, bias_ref, z_ref, group_size):
    out_ref = rearrange(F.layer_norm(rearrange(x_ref, "... (g d) -> ... g d", d=group_size), (group_size,), eps=1e-5), "... g d -> ... (g d)") * weight_ref
    out_ref = out_ref + bias_ref
    out_ref = out_ref * F.silu(z_ref)
    return out_ref
 

def torch_rmsnorm_gated(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


def benchmark():
    device_ = 'cuda'
    group_size = 64
    # set seed
    torch.random.manual_seed(0)
    batch = 16
    seqlen = 1024
    d = 2048
    dtype, wtype = torch.float32, torch.float32
    x = torch.randn(batch, seqlen, d, dtype=dtype, device=device_, requires_grad=True)
    z = torch.randn(batch, seqlen, d, dtype=dtype, device=device_, requires_grad=True)
    weight = torch.randn(d, dtype=wtype, device=device_, requires_grad=True)
    bias = torch.randn(d, dtype=wtype, device=device_, requires_grad=True)
   
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    z_pt = z.detach().clone().requires_grad_() if z is not None else None
    weight_ref = weight.detach().clone().requires_grad_()
    weight_pt = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    bias_pt = bias.detach().clone().requires_grad_() if bias is not None else None
    out_tri_layernorm = dlblas.layernorm_gated(x, weight, bias, z=z, eps=1e-5, group_size=group_size, norm_before_gate=True,
                       is_rms_norm=False)
    out_pt_layernorm = torch_layernorm_gated(x_ref, weight_ref, bias_ref, z_ref, group_size)
    assert torch.allclose(out_tri_layernorm, out_pt_layernorm, rtol=1e-3, atol=1e-3)
    out_pt_rmsnorm = torch_rmsnorm_gated(x_pt, weight_pt, bias_pt, z=z_pt, eps=1e-5, group_size=group_size,
                              norm_before_gate=True, upcast=False)
    out_tri_rmsnorm = dlblas.layernorm_gated(x, weight, bias, z=z, eps=1e-5, group_size=group_size, norm_before_gate=True,
                       is_rms_norm=True)
    assert torch.allclose(out_tri_rmsnorm, out_pt_rmsnorm, rtol=1e-3, atol=1e-3)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["kernel"],
            x_vals=["layernorm_gated", "rmsnorm_gated"],
            
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],

            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"norm-seqLen:{seqlen}",
            args={
                "group_size": group_size,
                "seqlen": seqlen,
                "batch":batch,
                "d":d,
                "dtype":dtype,
            },
        ))
    @triton.testing.perf_report(configs)
    def bench_layernorm_gated(kernel, group_size, seqlen, batch, d, dtype, provider):
        warmup = 100
        rep = 100
        x = torch.randn(batch, seqlen, d, dtype=dtype, device=device_, requires_grad=True)
        z = torch.randn(batch, seqlen, d, dtype=dtype, device=device_, requires_grad=True)
        weight = torch.randn(d, dtype=wtype, device=device_, requires_grad=True)
        bias = torch.randn(d, dtype=wtype, device=device_, requires_grad=True)
        is_rmsnorm = (kernel == "rmsnorm_gated")
        if "triton" in provider:
            fn = lambda: dlblas.layernorm_gated(x, weight, bias, z=z, eps=1e-5, group_size=group_size, 
                                                norm_before_gate=True, is_rms_norm=is_rmsnorm)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        if "pytorch" in provider:
            if is_rmsnorm:
                fn = lambda: torch_rmsnorm_gated(x_pt, weight_pt, bias_pt, z=z_pt, eps=1e-5, group_size=group_size,
                              norm_before_gate=True, upcast=False)
            else:
                fn = lambda : torch_layernorm_gated(x, weight, bias, z, group_size)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    
    bench_layernorm_gated.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    benchmark()
    print("sucessfully!")
