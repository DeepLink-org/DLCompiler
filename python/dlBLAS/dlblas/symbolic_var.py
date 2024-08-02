# import torch
# from torch import SymInt
#
# from torch._dynamo.source import ConstantSource
# from torch.fx.experimental.sym_node import to_node, SymNode, method_to_operator
# from torch._C import _disabled_torch_function_impl
# from torch.fx.experimental.symbolic_shapes import (
#     DimConstraints,
#     DimDynamic,
#     expect_true,
#     guard_bool,
#     guard_float,
#     guard_int,
#     GuardOnDataDependentSymNode,
#     ShapeEnv,
#     is_symbolic,
#     StatelessSymbolicContext,
#     statically_known_true,
#     _constrain_range_for_size,
# )
#
# '''
# steal from:
#     https://github.com/pytorch/pytorch/blob/7854d84acbfb7a4e3e807951188535a0316b585e/test/test_dynamic_shapes.py#L110
#
# rationale:
#     symbolic reasoning is needed because when we register a kernel, we don't know the concrete shape of the inputs, i.e. we only know when user pass in args; thus we want the kernel can handle input tensors with symbolic shapes at register time. There are 2 potential ways of doing it:
#
#     1. write kernel as (jinja2) template, e.g.
#
#     https://github.com/pytorch/pytorch/blob/7854d84acbfb7a4e3e807951188535a0316b585e/torch/_inductor/kernel/mm_plus_mm.py#L22
#
#     In this way when users pass in args, we can render the kernels and compile them at (offline `runtime`.
#
#     2. register the kernels and express shape as symbolic variables. In this way, when users pass in args, we can reason about whether the args' shape is contained by our symbolic expression, and if so, the kernel is applicable to users' case.
#
#     This file is mainly intented to support 2
# '''
#
# global_shape_env = ShapeEnv()
#
# meta_funcs = {}
#
# def create_contiguous(shape):
#     strides = [1]
#     for dim in reversed(shape[:-1]):
#         strides.append(dim * strides[-1])
#     return list(reversed(strides))
#
#
#
# class FakeSymbolicTensor(torch.Tensor):
#     @staticmethod
#     def __new__(cls, sym_shape, sym_strides, dtype, layout, requires_grad, device, storage_offset=0):
#         # TODO: this is wrong in general
#         sym_stride = create_contiguous(sym_shape)
#         r = torch.Tensor._make_wrapper_subclass(
#             cls, sym_shape,
#             sym_stride, storage_offset,
#             dtype=dtype, layout=layout, requires_grad=requires_grad,
#             device=device,
#         )
#         return r
#
#     __torch_function__ = _disabled_torch_function_impl
#
#     def new_empty(self, shape):
#         return FakeSymbolicTensor(shape, None, self.dtype, self.layout, self.requires_grad, self.device)
#
#     @classmethod
#     def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
#         if func_overload in meta_funcs:
#             return meta_funcs[func_overload](*args, **kwargs)
#
#         if func_overload == torch.ops.aten.new_empty.default:
#             self = args[0]
#             shape = args[1]
#             return FakeSymbolicTensor(shape, self.stride(), self.dtype, self.layout, self.requires_grad, self.device)
#
#         raise RuntimeError(f"operator {func_overload} not supported")
#
#
# def create_symbolic_tensor(name, arg, shape_env=None, source=None, dynamic_dims=None):
#     if shape_env is None:
#         global global_shape_env
#         shape_env = global_shape_env
#
#     if source is None:
#         source = ConstantSource(name)
#     constraint_dims = [None] * arg.dim()
#     if dynamic_dims is None:
#         dynamic_dims = [DimDynamic.DUCK] * arg.dim()
#     sym_shapes, sym_strides, sym_storage_offset = \
#         shape_env.create_symbolic_sizes_strides_storage_offset(
#             arg,
#             source=source,
#             symbolic_context=StatelessSymbolicContext(
#                 dynamic_sizes=dynamic_dims,
#                 constraint_sizes=constraint_dims
#             ),
#         )
#     return FakeSymbolicTensor(sym_shapes, sym_strides, arg.dtype, arg.layout, arg.requires_grad, arg.device, sym_storage_offset)
#
# def create_symtype(cls, pytype, shape_env, val):
#     from torch._dynamo.source import ConstantSource
#     symbol = shape_env.create_symbol(
#         val,
#         source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
#         # dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
#         dynamic_dim=DimDynamic.DYNAMIC,
#         constraint_dim=None,
#     )
#     return cls(SymNode(
#         symbol,
#         shape_env,
#         pytype,
#         hint=val,
#     ))
#
# def create_symint(i, shape_env=None):
#     if shape_env is None:
#         global global_shape_env
#         shape_env = global_shape_env
#     return create_symtype(SymInt, int, shape_env, i)
'''
we want to reuse torch symbolic varialbe as much as possible, however it seems requires a lot of work.
    we just define our own for now...
'''


import torch


class SymVar:
    '''
    this could be replaced by smypy
        which can be used to solve for complex shape constraints
    '''

    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name

    def __eq__(self, value) -> bool:
        if not isinstance(value, SymVar):
            return False
        return self.name == value.name

    def __hash__(self) -> int:
        return hash(self.name)


class Tensor:
    '''
    this just mimic the torch.Tensor
        if need to do analysis, e.g. dry run tensor operation, we should probably subclass torch.Tensor, and implement __torch_dispatch__
        for more: https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
    '''

    def __init__(self, shape, *, device:str, dtype:torch.dtype):
        for arg in shape:
            assert isinstance(arg, SymVar), f'expect a SymVar, but got {arg}'
        self._shape = shape
        self._device = device
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype
