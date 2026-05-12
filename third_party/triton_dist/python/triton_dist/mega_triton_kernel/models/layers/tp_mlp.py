################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import torch
from torch import nn


def shard_local(tensor: torch.Tensor, world_size: int, dim: int, local_rank: int):
    tensor_dim = tensor.shape[dim]
    tensor_slice = tensor_dim // world_size
    if tensor_dim % world_size != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} is not divisible by world size {world_size}.")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"Local rank {local_rank} is out of bounds for world size {world_size}.")
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor.dim()} dimensions.")
    if tensor_slice == 0:
        raise ValueError(f"Tensor slice size is zero for tensor dimension {tensor_dim} and world size {world_size}.")
    return tensor.split(tensor_slice, dim=dim)[local_rank].contiguous()


# adapt from triton_dist/layers/nvidia/tp_mlp.py
class TPMLPBuilder:
    """
    Tensor Parallel MLP.
    This MLP uses a common TP strategy:
    1. First linear layer (gate/up) weights are column-parallel.
    2. Second linear layer (down) weights are row-parallel.
    """

    def __init__(self, builder, rank=0, world_size=8, group=None):
        self._builder = builder
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.act_fn = None
        self.gate_up_proj = None
        self.down_proj = None

    def _init_parameters(self, mlp: nn.Module, verbose=False):
        """
        Initializes and shards MLP parameters for Tensor Parallelism.
        mlp: A standard nn.Module MLP (e.g., from HuggingFace Transformers).
             Expected to have mlp.gate_proj, mlp.up_proj, mlp.down_proj.
        """
        gate_proj = shard_local(mlp.gate_proj.weight.detach(), self.world_size, 0, self.rank)
        up_proj = shard_local(mlp.up_proj.weight.detach(), self.world_size, 0, self.rank)
        self.gate_up_proj = torch.cat((gate_proj, up_proj),
                                      dim=0).to("cuda", non_blocking=True)  # [MLP_size * 2 // world_size, hidden_size]
        self.down_proj = shard_local(mlp.down_proj.weight.detach(), self.world_size, 1,
                                     self.rank).to("cuda", non_blocking=True)  # [MLP_size // world_size, hidden_size]

        self.act_fn = mlp.act_fn
        self.ag_N_per_rank = self.gate_up_proj.shape[0]
        self.K = self.gate_up_proj.shape[1]
        self.dtype = self.gate_up_proj.dtype

        assert mlp.gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] MLP initialized with parameters: gate_up_proj shape: {self.gate_up_proj.shape}, down_proj shape: {self.down_proj.shape}"
            )

    def build_fwd(self, x, fc1_output=None, act_out=None, fc2_out=None, ar_out=None):
        """
        x: input tensor, shape [batch_size, seq_len, hidden_size], dtype bfloat16
        """
        assert len(x.shape) == 3
        assert x.dtype == torch.bfloat16
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x = x.reshape(-1, hidden_size)
        fc1_output = torch.empty(num_tokens, self.gate_up_proj.shape[0], dtype=x.dtype,
                                 device=x.device) if fc1_output is None else fc1_output
        act_out = torch.empty(num_tokens, self.gate_up_proj.shape[0] //
                              2, dtype=x.dtype, device=x.device) if act_out is None else act_out
        if self.world_size > 1:
            fc2_out = self._builder.create_symm_tensor(
                (num_tokens, hidden_size), x.dtype) if fc2_out is None else fc2_out
            ar_out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device) if ar_out is None else ar_out
        else:
            assert ar_out is None
            fc2_out = torch.empty(num_tokens, hidden_size, dtype=x.dtype,
                                  device=x.device) if fc2_out is None else fc2_out
        self._builder.make_fc1(x, self.gate_up_proj, fc1_output)
        self._builder.make_silu_mul_up(fc1_output, act_out)
        self._builder.make_fc2(act_out, self.down_proj, fc2_out)
        if self.world_size > 1:
            self._builder.make_allreduce(fc2_out, ar_out, double_input_buffer=True)
            return ar_out.reshape(batch_size, seq_len, hidden_size)
        return fc2_out.reshape(batch_size, seq_len, hidden_size)
