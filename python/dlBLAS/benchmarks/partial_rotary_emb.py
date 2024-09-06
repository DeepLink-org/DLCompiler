import torch
from torch import nn
import torch.nn.functional as F
import triton
import dlblas
from dlblas.utils.gpu_helper import get_idle_device


class PartialRotaryEmb(nn.Module):
    def __init__(
        self,
        num_heads: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
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

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, q, k_pe, kv, cos, sin):
        assert q.shape[2] == self.num_heads
        assert q.shape[3] == self.q_head_dim
        bsz, q_len, self.num_heads, self.q_head_dim = q.shape

        q = q.transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        assert (
            bsz == kv.shape[0]
            and q_len == kv.shape[1]
            and self.num_heads == kv.shape[2]
        )
        assert self.qk_nope_head_dim + self.v_head_dim == kv.shape[3]
        kv = kv.transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        q = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        q[:, :, :, : self.qk_nope_head_dim] = q_nope
        q[:, :, :, self.qk_nope_head_dim :] = q_pe
        k = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        k[:, :, :, : self.qk_nope_head_dim] = k_nope
        k[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.q_head_dim - self.v_head_dim])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
        return q, kv


device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)


def test():
    num_heads = 32
    nope_head_dim = 128
    rope_head_dim = 64
    v_head_dim = 128
    q_head_dim = nope_head_dim + rope_head_dim
    bsz, q_len = 2, 4096
    q = torch.randn(
        (bsz, q_len, num_heads, q_head_dim), dtype=torch.float16, device=device_
    )
    k_pe = torch.randn(
        (bsz, q_len, 1, rope_head_dim), dtype=torch.float16, device=device_
    )
    kv = torch.randn(
        (bsz, q_len, num_heads, nope_head_dim + v_head_dim),
        dtype=torch.float16,
        device=device_,
    )
    cos = torch.randn((bsz, q_len, rope_head_dim), dtype=torch.float16, device=device_)
    sin = torch.randn((bsz, q_len, rope_head_dim), dtype=torch.float16, device=device_)
    q_tri, k_pe_tri, kv_tri, cos_tri, sin_tri = (
        q.clone(),
        k_pe.clone(),
        kv.clone(),
        cos.clone(),
        sin.clone(),
    )
    out_q, out_kv = PartialRotaryEmb(
        num_heads, v_head_dim, nope_head_dim, rope_head_dim
    ).forward(q, k_pe, kv, cos, sin)
    # print(f"out_kv.shape:{out_kv.shape}")
    out_tri_q, out_tri_kv = dlblas.partial_rotary_emb(
        q_tri, k_pe_tri, kv_tri, cos_tri, sin_tri
    )
    # print(f"out_tri_kv.shape:{out_tri_kv.shape}")
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
                fn = lambda: PartialRotaryEmb(
                    num_heads, v_head_dim, nope_head_dim, rope_head_dim
                ).forward(q, k_pe, kv, cos, sin)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")
