import torch

from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.fx.experimental.symbolic_shapes import ShapeEnv
'''
play around how to convert torch.Tensor -> FakeTensor

e.g. 
https://github.com/pytorch/pytorch/blob/90d5a6f001ef3ea40ef91ae20e050e39a6d550de/torch/_functorch/aot_autograd.py#L496
'''

device = 'cpu'

a = torch.randn(16, 128, device=device)
b = torch.randn(128, 16, device=device)
c = torch.randn(16, 128, device=device)
d = torch.randn(128, 16, device=device)
flat_args = (a, b, c, d)


class AOTConfig:
    is_export = False
    num_params_buffers = -10
    dynamic_shapes = False


aot_config = AOTConfig()


class Config:
    static_weight_shapes = False


config = Config()

fake_mode = detect_fake_mode(flat_args)
if fake_mode is None:
    shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
    fake_mode = FakeTensorMode(shape_env=shape_env)
else:
    shape_env = fake_mode.shape_env


def process_inputs(flat_args):

    def convert(idx, x):
        if shape_env is not None:
            from torch._dynamo.source import ConstantSource
            if isinstance(x, int):
                # We always specialize on scalar values in export.
                if aot_config.is_export:
                    return x
                source = ConstantSource(f"sym_{idx}")
                return shape_env.create_symintnode(shape_env.create_symbol(
                    x, source),
                                                   hint=x,
                                                   source=source)
        if not isinstance(x, torch.Tensor):
            return x
        if isinstance(x, FakeTensor):
            assert x.fake_mode is fake_mode
            return x
        if is_traceable_wrapper_subclass(x):
            attrs, _ = x.__tensor_flatten__()
            if all(isinstance(getattr(x, attr), FakeTensor) for attr in attrs):
                assert all(
                    getattr(x, attr).fake_mode is fake_mode for attr in attrs)
                return x

        # see note [Tensor Fakification and Symbol Caching]
        symbolic_context = None
        source = None
        if tracing_context := torch._guards.TracingContext.try_get():
            if x in tracing_context.tensor_to_context:
                symbolic_context = tracing_context.tensor_to_context[x]
                source = symbolic_context.tensor_source
        if (idx < aot_config.num_params_buffers and config.static_weight_shapes
                and not symbolic_context):
            # TODO: Ensure that this codepath is never exercised from
            # Dynamo
            return fake_mode.from_tensor(x, static_shapes=True)

        # hgl: symbolic_context is None, and source is None
        return fake_mode.from_tensor(x,
                                     static_shapes=False,
                                     symbolic_context=symbolic_context,
                                     source=source)

    return [convert(idx, x) for idx, x in enumerate(flat_args)]


out = process_inputs(flat_args)
for x in out:
    print(x)
