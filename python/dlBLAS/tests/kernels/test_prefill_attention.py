import math
import unittest
import torch
import torch.nn.functional as F
from dlblas.kernels.prefill_attention import (
    context_attention_fwd,
)


class TestPrefillAttention(unittest.TestCase):
    def _torch_att(self, xq, xk, xv, bs, seqlen, num_head, head_dim):
        xq = xq.view(bs, seqlen, num_head, head_dim)
        xk = xk.view(bs, seqlen, num_head, head_dim)
        xv = xv.view(bs, seqlen, num_head, head_dim)
        mask = (
            torch.tril(torch.ones(seqlen, seqlen), diagonal=0)
            .unsqueeze(0)
            .unsqueeze(0)
            .cuda()
        )
        mask[mask == 0.0] = -100000000.0
        mask = mask.repeat(bs, num_head, 1, 1)
        keys = xk
        values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
        scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
        output = (
            torch.matmul(scores, values)
            .transpose(1, 2)
            .contiguous()
            .reshape(-1, num_head, head_dim)
        )
        return output

    def test_prefill_attention(self):
        Z, H, N_CTX, D_HEAD = 4, 6, 1024, 128
        dtype = torch.float16
        Z = 3
        q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
            mean=0.1, std=0.2
        )
        k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
            mean=0.4, std=0.2
        )
        v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
            mean=0.3, std=0.2
        )
        o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
            mean=0.3, std=0.2
        )

        max_input_len = N_CTX
        Z = 4
        b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
        b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")

        b_seq_len[0] = 512
        b_seq_len[1] = 1024
        b_seq_len[2] = 512
        b_seq_len[3] = 1024

        for i in range(1, Z):
            b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]

        torch_out = []
        start = 0
        for i in range(Z):
            end = start + b_seq_len[i]
            torch_o = self._torch_att(
                q[start:end], k[start:end], v[start:end], 1, b_seq_len[i], H, D_HEAD
            )
            start = end
            torch_out.append(torch_o)
        torch_out = torch.cat(torch_out, dim=0)
        context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len)
        print(o.shape, torch_out.shape)

        print("max diff", torch.max(torch.abs(torch_out - o)))
        print("mean diff", torch.mean(torch.abs(torch_out - o)))
        assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    unittest.main()
