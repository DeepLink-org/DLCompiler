## Install dlblas
refer to [README](../../README.md)

## Modify InternEvo
internlm/model/moe/gshard_layer.py

replace

```
class TopKGate(Module):
    def forward(...):
        ...
        if self.use_fused_gating or self.k > 2:
            assert self.noisy_gate_policy != "RSample", "RSample noisy is not supported by fused_gating policy"
            gate_output = fused_topkgating(
                logits, self.k, self.capacity_factor if self.training else self.eval_capacity_factor, self.min_capacity
            )
        # deepspeed-style code
        elif self.k == 1:
            gate_output = top1gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens,
                self.use_rts,
            )

        elif self.k == 2:
            gate_output = top2gating(
                logits, self.capacity_factor if self.training else self.eval_capacity_factor, self.min_capacity
            )
        else:
            assert False, "Unsupported gating policy"
```

to

```
import dlblas
class TopKGate(Module):
    def forward(...):
        ...
        gate_output = dlblas.topk_gating(logits, self.k, self.capacity_factor if self.training else self.eval_capacity_factor, self.min_capacity)
```

## Launch Model Training
bash srun_moe.sh



## Problems to Be Solved
add 
```
locations_s[locations_s>=capacity] = capacity - 1
locations_s[locations_s<0] = 0
```

in def call(gates, masks, locations, k, capacity)  in python/dlBLAS/dlblas/kernels/topk_gating_fwd_part3.py