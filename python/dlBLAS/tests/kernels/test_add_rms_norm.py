import torch
import dlblas
from torch.nn.parameter import Parameter


class FuseRMSNorm(torch.nn.Module):
    def __init__(self, weights, eps=1e-6):
        super().__init__()

        self.weight = weights
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


def compare_tensor(a, b, prec):
    epsilon = 1.0 / 16384

    diff = a - b
    diff = diff.abs().pow(2).sum()
    a_pow_sum = a.pow(2).sum()
    if diff <= (2 * epsilon) * (2 * epsilon):
        diff = 0.0
    if a_pow_sum <= epsilon:
        a_pow_sum = a_pow_sum + epsilon
    diff = torch.div(diff, (a_pow_sum * 1.0))
    return diff.sqrt().item() <= prec


def test_add_rms_norm0():
    H, C = 4096, 4096
    eps = 1e-6
    input = torch.randn(H, C, device="cuda", dtype=torch.half)
    ref_input = input.clone()
    residual = torch.randn(H, C, device="cuda", dtype=torch.half)
    ref_residual = residual.clone()
    weight = Parameter(torch.randn(C, device="cuda", dtype=torch.half))
    ref_normed_out, ref_added_out = dlblas.add_rms_norm(
        ref_input, weight, ref_residual, eps
    )
    rms_norm = FuseRMSNorm(weight, eps=eps)
    normed_out, added_out = rms_norm(input, residual)
    print("max abs diff: ", torch.max(abs(ref_normed_out - normed_out)))
    print("max abs diff: ", torch.max(abs(ref_added_out - added_out)))
    assert compare_tensor(normed_out, ref_normed_out, 0.003)
    assert compare_tensor(added_out, ref_added_out, 0.003)
    print("test_add_rms_norm: pass")


def test_rms_norm0():
    H, C = 4096, 4096
    eps = 1e-6
    input = torch.randn(H, C, device="cuda", dtype=torch.half)
    ref_input = input.clone()
    weight = Parameter(torch.randn(C, device="cuda", dtype=torch.half))
    ref_normed_out = dlblas.rms_norm(ref_input, weight, eps)
    rms_norm = FuseRMSNorm(weight, eps=eps)
    normed_out, added_out = rms_norm(input)
    print("max abs diff: ", torch.max(abs(ref_normed_out - normed_out)))
    assert compare_tensor(normed_out, ref_normed_out, 0.003)
    print("test_rms_norm: pass")


if __name__ == "__main__":
    test_rms_norm0()
    test_add_rms_norm0()
