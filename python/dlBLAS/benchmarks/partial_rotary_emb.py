from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import triton
import dlblas
from dlblas.utils.gpu_helper import get_idle_device

from functorch.compile import aot_module, make_boxed_func, aot_function
from torch._dynamo.backends.common import aot_autograd


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    # print(">>> FX graph:")
    # gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return make_boxed_func(gm.forward)  # fwd do not need boxed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# my_aot_backend = aot_autograd(fw_compiler=my_compiler)


# @torch.compile(backend=my_aot_backend)
def partial_rotary_emb(q, k_pe, kv, cos, sin):
    bsz, q_len, num_heads, q_head_dim = q.shape
    assert bsz == k_pe.shape[0] and q_len == k_pe.shape[1] and 1 == k_pe.shape[2]
    qk_rope_head_dim = k_pe.shape[3]
    qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    assert bsz == kv.shape[0] and q_len == kv.shape[1] and num_heads == kv.shape[2]
    v_head_dim = kv.shape[3] - qk_nope_head_dim
    assert q_len == cos.shape[1] and qk_rope_head_dim == cos.shape[2]
    assert cos.shape == sin.shape

    q = q.transpose(1, 2)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)
    kv = kv.transpose(1, 2)
    k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
    # return q_pe, k_pe

    q_out = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    q_out[:, :, :, :qk_nope_head_dim] = q_nope
    q_out[:, :, :, qk_nope_head_dim:] = q_pe
    k = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    k[:, :, :, :qk_nope_head_dim] = k_nope
    k[:, :, :, qk_nope_head_dim:] = k_pe

    if q_head_dim != v_head_dim:
        v = F.pad(v, [0, q_head_dim - v_head_dim])

    q_out = q_out.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    kv_out = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
    return q_out, kv_out


device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)


def testt(qq, k, cos, sin):
    b, h, s, d = 1, 32, 33, 64
    qq = qq.transpose(1, 2)
    q0, q1 = torch.split(qq, [d, d], dim=-1)
    q1, out_k = apply_rotary_pos_emb(q1, k, cos, sin)
    q_out = k.new_empty(b, h, s, d * 2)
    q_out[:, :, :, :d] = q0
    q_out[:, :, :, d:] = q1
    q_out = q_out.transpose(1, 2)
    return q_out, out_k


def test2():
    b, h, s, d = 1, 32, 33, 64
    qq = torch.randn((b, s, h, d * 2), dtype=torch.float16, device=device_)
    k = torch.randn((b, h, s, d), dtype=torch.float16, device=device_)
    cos = torch.randn((b, s, d), dtype=torch.float16, device=device_)
    sin = torch.randn((b, s, d), dtype=torch.float16, device=device_)
    qq.requires_grad = True
    k.requires_grad = True
    cos.requires_grad = True
    sin.requires_grad = True
    q_out, k_out = testt(qq, k, cos, sin)
    loss_torch = torch.sum(torch.mean(q_out))
    dout_torch = torch.randn_like(loss_torch)
    loss_torch.backward(dout_torch, retain_graph=True)
    print(qq.grad)
    import pdb

    pdb.set_trace()
    # a = torch.randn((4, 8), dtype=torch.float16, device=device_)
    # a.requires_grad = True
    # a0, a1 = torch.split(a, [4, 4], dim=-1)
    # b = a1 * a1
    # c = b.new_empty(4, 8)
    # c[:, :4] = a0
    # c[:, 4:] = a1

    # loss = torch.sum(torch.mean(c))
    # loss.backward()
    # print(a.grad)


def test():
    num_heads = 128
    nope_head_dim = 128
    rope_head_dim = 64
    v_head_dim = 128
    q_head_dim = nope_head_dim + rope_head_dim
    bsz, q_len = 1, 4096
    q = torch.randn(
        (bsz, q_len, num_heads, q_head_dim), dtype=torch.bfloat16, device=device_
    )
    k_pe = torch.randn(
        (bsz, q_len, 1, rope_head_dim), dtype=torch.bfloat16, device=device_
    )
    kv = torch.randn(
        (bsz, q_len, num_heads, nope_head_dim + v_head_dim),
        dtype=torch.bfloat16,
        device=device_,
    )
    cos = torch.randn((bsz, q_len, rope_head_dim), dtype=torch.bfloat16, device=device_)
    sin = torch.randn((bsz, q_len, rope_head_dim), dtype=torch.bfloat16, device=device_)
    with torch.no_grad():
        q_tri, k_pe_tri, kv_tri, cos_tri, sin_tri = (
            q.clone(),
            k_pe.clone(),
            kv.clone(),
            cos.clone(),
            sin.clone(),
        )
    q.requires_grad = True
    k_pe.requires_grad = True
    kv.requires_grad = True
    cos.requires_grad = True
    sin.requires_grad = True
    out_q, out_kv = partial_rotary_emb(q, k_pe, kv, cos, sin)

    # loss_torch = torch.sum(torch.mean(out_q))
    # dout_torch = torch.randn_like(loss_torch)
    # loss_torch.backward(dout_torch, retain_graph=True)
    # print(torch.max(q.grad))
    # import pdb

    # pdb.set_trace()

    out_tri_q, out_tri_kv = dlblas.partial_rotary_emb(
        q_tri, k_pe_tri, kv_tri, cos_tri, sin_tri
    )

    assert torch.allclose(out_q, out_tri_q, rtol=0.01, atol=0.01)
    assert torch.allclose(
        out_kv[:, :, 0, :, :nope_head_dim], out_tri_kv[:, :, 0, :, :nope_head_dim]
    )
    assert torch.allclose(
        out_kv[:, :, 0, :, nope_head_dim:],
        out_tri_kv[:, :, 0, :, nope_head_dim:],
        rtol=0.01,
        atol=0.01,
    )
    assert torch.allclose(
        out_kv[:, :, 1, :, :v_head_dim], out_tri_kv[:, :, 1, :, :v_head_dim]
    )
    assert torch.allclose(
        out_kv,
        out_tri_kv,
        rtol=0.01,
        atol=0.01,
    )
    # loss_torch = torch.sum(torch.mean(out_q))
    # dout_torch = torch.randn_like(loss_torch)
    # loss_torch.backward(dout_torch, retain_graph=True)
    # print(torch.max(q.grad))
    # import pdb

    # pdb.set_trace()

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=[
                "fwd",
            ],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],
            ylabel="ms",
            plot_name=f"fused_partial_mla(batchSize={bsz}, seqlen:{q_len}, num_heads:{num_heads}, nope_head_dim:{nope_head_dim}, rope_head_dim:{rope_head_dim})",
            args={"SeqLen": q_len},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 200

        if "triton" in provider:
            if "fwd" == op:
                fn = lambda: dlblas.partial_rotary_emb(
                    q_tri, k_pe_tri, kv_tri, cos_tri, sin_tri
                )
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

            else:
                raise Exception()

        if "pytorch" in provider:
            if "fwd" == op:
                fn = lambda: partial_rotary_emb(q, k_pe, kv, cos, sin)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")
