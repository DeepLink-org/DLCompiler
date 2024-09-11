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


def rotate_half_conj(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


my_aot_backend = aot_autograd(fw_compiler=my_compiler)


@torch.compile(backend=my_aot_backend)
def _apply_rotary_compile(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos1: torch.Tensor,
    cos2: torch.Tensor,
    sin1: torch.Tensor,
    sin2: torch.Tensor,
):
    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin2 + x2 * cos2
    return out1, out2


@torch.compile(backend=my_aot_backend)
def _test_compile(q: torch.Tensor):
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return q


def _test_func(q: torch.Tensor):
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return q


def apply_rotary_pos_emb(q, k, cos, sin, conj=False, unsqueeze_dim=1):
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

    if conj:
        # out1.copy_(x1 * cos + x2 * sin)
        # out2.copy_(x2 * cos - x1 * sin)
        q_embed = (q * cos) + (rotate_half_conj(q) * sin)
        k_embed = (k * cos) + (rotate_half_conj(k) * sin)
    else:
        # out1.copy_(x1 * cos - x2 * sin)
        # out2.copy_(x2 * cos + x1 * sin)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# @torch.compile()
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


def test_emb(q, k_pe, kv, cos, sin):
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

    q_pe_out, k_pe_out = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

    q_out = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    q_out[:, :, :, :qk_nope_head_dim] = q_nope
    q_out[:, :, :, qk_nope_head_dim:] = q_pe_out
    k = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    k[:, :, :, :qk_nope_head_dim] = k_nope
    k[:, :, :, qk_nope_head_dim:] = k_pe_out

    if q_head_dim != v_head_dim:
        v = F.pad(v, [0, q_head_dim - v_head_dim])

    q_out = q_out.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    kv_out = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
    return q_out, kv_out, q_pe, k_pe


device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k_pe, kv, cos, sin):

        # x1 = q[..., : q.shape[-1] // 2]
        # x2 = q[..., q.shape[-1] // 2 :]
        # out = _torch_apply_rotary_func(x1, x2, cos, sin, False)
        # return out
        # cos = torch.cat((cos, cos), dim=-1)
        # sin = torch.cat((sin, sin), dim=-1)
        # out_q, _ = apply_rotary_pos_emb(q, q, cos, sin, False)
        # (
        #     transpose_9,
        #     cat_2,
        #     unsqueeze,
        #     unsqueeze_1,
        #     _unsafe_view,
        #     _unsafe_view_1,
        #     cat,
        #     cat_1,
        # ) = emb_fwd(q, k_pe, kv, cos, sin)
        # ctx.save_for_backward(
        #     unsqueeze, unsqueeze_1, _unsafe_view, _unsafe_view_1, cat, cat_1
        # )
        # return transpose_9, cat_2
        out_q, out_kv, q_pe, k_pe = test_emb(q, k_pe, kv, cos, sin)
        ctx.save_for_backward(kv, cos, sin, q_pe, k_pe)
        return out_q, out_kv

    @staticmethod
    def backward(ctx, d_q, do_kv):
        kv, cos, sin, q_pe, k_pe = ctx.saved_tensors
        bsz, seq_len, heads, q_head_dim = d_q.shape
        rope_head_dim = cos.shape[-1]
        nope_head_dim = q_head_dim - rope_head_dim
        v_head_dim = kv.shape[3] - nope_head_dim
        do_k = do_kv[:, :, 0, :, :]
        do_v = do_kv[:, :, 1, :, :][:, :, :, :v_head_dim]
        do_k_nope = do_k[:, :, :, :nope_head_dim]
        do_k_pe = do_k[:, :, :, nope_head_dim:]
        d_q_nope = d_q[:, :, :, :nope_head_dim]
        d_q_pe = d_q[:, :, :, nope_head_dim:]

        # bwd for
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        # import pdb

        # pdb.set_trace()
        dx_q_pe = rotary_bwd(q_pe.transpose(1, 2), d_q_pe, cos, sin)
        dx_k_pe = rotary_bwd(k_pe.transpose(1, 2), do_k_pe, cos, sin)

        b, h, s, d = dx_q_pe.shape
        dx_q_pe = dx_q_pe.view(b, h, s, 2, d // 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = dx_k_pe.shape
        dx_k_pe = dx_k_pe.view(b, h, s, 2, d // 2).transpose(4, 3).reshape(b, h, s, d)

        dx_kv = torch.concat([do_k_nope, do_v], dim=-1)
        dx_q = torch.concat((d_q_nope, dx_q_pe), dim=-1)
        return dx_q, dx_k_pe, dx_kv, None, None


def rotary_bwd(q_pe, do_q, cos, sin):
    do0 = do_q[..., : do_q.shape[-1] // 2]
    do1 = do_q[..., do_q.shape[-1] // 2 :]
    x0 = q_pe[..., : q_pe.shape[-1] // 2]
    x1 = q_pe[..., q_pe.shape[-1] // 2 :]
    cos0 = cos[:, :, :, : cos.shape[-1] // 2]
    cos1 = cos[:, :, :, cos.shape[-1] // 2 :]
    sin0 = sin[:, :, :, : sin.shape[-1] // 2]
    sin1 = sin[:, :, :, sin.shape[-1] // 2 :]
    dx_q_0 = do1 * sin1 + do0 * cos0
    dx_q_1 = do1 * cos1 - do0 * sin0
    return torch.concat((dx_q_0, dx_q_1), dim=-1)
    # mul_4 = torch.ops.aten.mul.Tensor(do1, x1)
    mul_5 = torch.ops.aten.mul.Tensor(do1, cos1)
    # mul_6 = torch.ops.aten.mul.Tensor(do1, x0)
    mul_7 = torch.ops.aten.mul.Tensor(do1, sin1)
    neg = torch.ops.aten.neg.default(do0)
    # mul_8 = torch.ops.aten.mul.Tensor(neg, x1)
    mul_9 = torch.ops.aten.mul.Tensor(neg, sin0)
    add_1 = torch.ops.aten.add.Tensor(mul_5, mul_9)
    # mul_10 = torch.ops.aten.mul.Tensor(do0, x0)
    mul_11 = torch.ops.aten.mul.Tensor(do0, cos0)
    q = torch.ops.aten.add.Tensor(mul_7, mul_11)
    dx_q = torch.concat((q, add_1), dim=-1)
    return dx_q


def test():
    num_heads = 2  # 128
    nope_head_dim = 4  # 128
    rope_head_dim = 4  # 64
    v_head_dim = 4  # 128
    q_head_dim = nope_head_dim + rope_head_dim
    bsz, q_len = 1, 2  # 1, 4096
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
        q_tri, k_pe_tri, kv_tri = (
            q.clone(),
            k_pe.clone(),
            kv.clone(),
        )
        q_test, k_pe_test, kv_test = (
            q.clone(),
            k_pe.clone(),
            kv.clone(),
        )
    q_tri.requires_grad = True
    k_pe_tri.requires_grad = True
    kv_tri.requires_grad = True

    q.requires_grad = True
    k_pe.requires_grad = True
    kv.requires_grad = True
    q_test.requires_grad = True
    k_pe_test.requires_grad = True
    kv_test.requires_grad = True

    out_q, out_kv = partial_rotary_emb(q, k_pe, kv, cos, sin)
    loss_torch = torch.sum(torch.mean(out_q) * torch.mean(out_kv))
    loss_torch.backward(retain_graph=True)
    out_q_test, out_kv_test = ApplyRotaryEmb.apply(q_test, k_pe_test, kv_test, cos, sin)
    loss_test = torch.sum(torch.mean(out_q_test) * torch.mean(out_kv_test))
    loss_test.backward(retain_graph=True)
    print(f"q grad max diff: {(q.grad - q_test.grad).abs().max().item()}")
    print(f"k_pe grad mean diff: {(k_pe.grad - k_pe_test.grad).abs().mean().item()}")
    print(f"kv grad mean diff: {(kv.grad - kv_test.grad).abs().mean().item()}")

    assert torch.allclose(q.grad, q_test.grad)

    out_tri_q, out_tri_kv = dlblas.partial_rotary_emb(q_tri, k_pe_tri, kv_tri, cos, sin)
    assert torch.allclose(out_q, out_tri_q)
    # print(out_kv)
    # print(out_tri_kv)
    out_k = out_kv[:, :, 0, :, :]
    out_v = out_kv[:, :, 1, :, :][:, :, :, :v_head_dim]
    out_tri_k = out_tri_kv[:, :, 0, :, :]
    out_tri_v = out_tri_kv[:, :, 1, :, :][:, :, :, :v_head_dim]
    assert torch.allclose(out_k, out_tri_k)
    assert torch.allclose(out_v, out_tri_v)
    assert torch.allclose(out_kv, out_tri_kv)
    loss_tri = torch.sum(torch.mean(out_tri_q) * torch.mean(out_tri_kv))
    loss_tri.backward(retain_graph=True)

    print(k_pe.grad)
    print(k_pe_tri.grad)

    print(f"q grad max diff: {(q.grad - q_tri.grad).abs().max().item()}")
    print(f"k_pe grad max diff: {(k_pe.grad - k_pe_tri.grad).abs().max().item()}")

    # assert torch.allclose(q.grad, q_tri.grad)
    # return

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
                    q_tri, k_pe_tri, kv_tri, cos, sin
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
    # test2()
    print("sucessfully!")
