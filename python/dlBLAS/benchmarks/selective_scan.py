import torch
from einops import einsum, rearrange, repeat
import triton
import dlblas
from dlblas.utils.gpu_helper import get_idle_device 



# credit: https://github.com/johnma2006/mamba-minimal/blob/master/model.py#L275
def ref_selective_scan(u, delta, A, B, C, D, initial_state):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
        
    """
    original_dtype = u.dtype
    u, delta, A, B, C, D = map(lambda x: x.float(), (u, delta, A, B, C, D))
    (b, l, d_in) = u.shape
    n = A.shape[1]
    
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
    
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    x = torch.zeros((b, d_in, n), device=deltaA.device)
    x += initial_state
    ys = []    
    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
    
    y = y + u * D[None, None, :]

    return y.to(original_dtype), x 


device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)
def test():
    B = 2
    T = 16
    D = 512
    K = 16
    dtype = torch.float32
    A = (-(torch.rand(D, K, dtype=dtype)).exp().cuda()).requires_grad_(True)
    x = torch.randn(B, T, D, dtype=dtype).cuda().requires_grad_(True)
    delta = torch.randn(B, T, D, dtype=dtype).sigmoid().cuda().requires_grad_(True)
    B2 = torch.randn(B, T, K, dtype=dtype).cuda().requires_grad_(True)
    C = torch.randn(B, T, K, dtype=dtype).cuda().requires_grad_(True)
    D2 = torch.randn(D, dtype=dtype).cuda().requires_grad_(True)

    initial_state = torch.randn(B, D, K, dtype=dtype).cuda().requires_grad_(False)

    tri, tri_final = dlblas.selective_scan(x, delta, A, B2, C, D2, initial_state)
    do = torch.randn_like(tri)
    tri.backward(do)

    tri_dc, C.grad = C.grad.clone(), None
    tri_dx, x.grad = x.grad.clone(), None
    tri_db, B2.grad = B2.grad.clone(), None
    tri_delta, delta.grad = delta.grad.clone(), None
    tri_A, A.grad = A.grad.clone(), None

    ref, ref_final = ref_selective_scan(x, delta, A, B2, C, D2, initial_state)

    print(f"max diff {(tri-ref).abs().max()}")
    assert torch.allclose(tri, ref, rtol = 1e-05, atol = 1e-05)
    print(f"max diff {(tri_final-ref_final).abs().max()}")
    assert torch.allclose(tri_final, ref_final, rtol = 1e-05, atol = 1e-05)

    ref.backward(do)
    ref_dc, C.grad = C.grad.clone(), None
    ref_dx, x.grad = x.grad.clone(), None
    ref_db, B2.grad = B2.grad.clone(), None
    ref_delta, delta.grad = delta.grad.clone(), None
    ref_A, A.grad = A.grad.clone(), None

    print(f"max diff {(tri_dc-ref_dc).abs().max()}")    
    assert torch.allclose(tri_dc, ref_dc, rtol = 1e-05, atol = 1e-05)
    print(f"max diff {(tri_dx-ref_dx).abs().max()}")
    assert torch.allclose(tri_dx, ref_dx, rtol = 1e-05, atol = 1e-05)
    print(f"max diff {(tri_db-ref_db).abs().max()}")  
    assert torch.allclose(tri_db, ref_db, rtol = 1e-05, atol = 1e-05)
    print(f"max diff {(tri_delta-ref_delta).abs().max()}")
    assert torch.allclose(tri_delta, ref_delta, rtol = 1e-05, atol = 1e-05)
    print(f"max diff {(tri_A-ref_A).abs().max()}")
    assert torch.allclose(tri_A, ref_A, rtol = 1e-05, atol = 1e-05)
    
    configs = []

    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=['fwd', 'bwd'],
            line_arg="provider",

            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],

            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"selective_scan-B:{B}-T:{T}-D:{D}-K:{K}",
            args={
            },
        ))
    @triton.testing.perf_report(configs)
    def bench(op, provider):
        warmup = 100
        rep = 200
        if "triton" in provider:
            if 'fwd' == op:
                fn = lambda: dlblas.selective_scan(x, delta, A, B2, C, D2, initial_state)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif 'bwd' == op:
                tri, tri_final = dlblas.selective_scan(x, delta, A, B2, C, D2, initial_state)
                do = torch.randn_like(tri)
                bwd_fn = lambda: tri.backward(do, retain_graph=True)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()
        
        if "pytorch" in provider:
            if 'fwd' == op:
                fn = lambda: ref_selective_scan(x, delta, A, B2, C, D2, initial_state)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif 'bwd' == op:
                tri, tri_final = ref_selective_scan(x, delta, A, B2, C, D2, initial_state)
                do = torch.randn_like(tri)
                bwd_fn = lambda: tri.backward(do, retain_graph=True)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()
        return ms
    
    bench.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print("sucessfully!")