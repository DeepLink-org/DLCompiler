import torch

from sympy import symbols

x, y = symbols('x y')
expr = x + 2 * y

print(type(x), '; ', x)
print(type(expr), '; ', expr)

print(x == x)
print(x == y)

a, b = symbols('1 2')
expr = a + b

print(type(a), '; ', a)
print(type(expr), '; ', expr)

print(a == a)
print(a == b)

print()
print('======================')
print('======================')
print()

# NOTE: torch won't accept sympy's symbol input
# torch has its own symbolic shape system
# a = torch.randn((x, y))
# print(a)

from torch import SymInt, sym_int

a = SymInt(1)
b = SymInt(2)
print(type(a), '; ', a)

# operation is NOT ok
# c = a + b
# d = b - a
# print(type(c), '; ', c)
# print(type(d), '; ', d)

# e = sym_int(1)
# print(type(e), '; ', e)
# print(a == e)
# print(b == e)

print()
print('======================')
print('======================')
print()

from torch.fx.experimental.sym_node import to_node, SymNode, method_to_operator
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.symbolic_shapes import (
    DimConstraints,
    DimDynamic,
    expect_true,
    guard_bool,
    guard_float,
    guard_int,
    GuardOnDataDependentSymNode,
    ShapeEnv,
    is_symbolic,
    StatelessSymbolicContext,
    statically_known_true,
    _constrain_range_for_size,
)

meta_funcs = {}


def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))


class FakeSymbolicTensor(torch.Tensor):

    @staticmethod
    def __new__(cls,
                sym_shape,
                sym_strides,
                dtype,
                layout,
                requires_grad,
                device,
                storage_offset=0):
        # TODO: this is wrong in general
        sym_stride = create_contiguous(sym_shape)
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sym_shape,
            sym_stride,
            storage_offset,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            device=device,
        )
        return r

    __torch_function__ = _disabled_torch_function_impl

    def new_empty(self, shape):
        return FakeSymbolicTensor(shape, None, self.dtype, self.layout,
                                  self.requires_grad, self.device)

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(shape, self.stride(), self.dtype,
                                      self.layout, self.requires_grad,
                                      self.device)

        raise RuntimeError(f"operator {func_overload} not supported")


def create_symbolic_tensor(name,
                           arg,
                           shape_env,
                           source=None,
                           dynamic_dims=None):
    from torch._dynamo.source import ConstantSource

    if source is None:
        source = ConstantSource(name)
    constraint_dims = [None] * arg.dim()
    if dynamic_dims is None:
        dynamic_dims = [DimDynamic.DUCK] * arg.dim()
    sym_shapes, sym_strides, sym_storage_offset = \
        shape_env.create_symbolic_sizes_strides_storage_offset(
            arg,
            source=source,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=dynamic_dims,
                constraint_sizes=constraint_dims
            ),
        )
    return FakeSymbolicTensor(sym_shapes, sym_strides, arg.dtype, arg.layout,
                              arg.requires_grad, arg.device,
                              sym_storage_offset)


shape_env = ShapeEnv()
x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
y = create_symbolic_tensor("y", torch.randn(5, 4, 3), shape_env)

assert (not isinstance(x.shape[0], SymNode))
assert (isinstance(x.shape[0], SymInt))

print(x.shape[0] == 5)
print(x.shape[0] == y.shape[0])
print(x.size())
print(x.size() == y.size())

z = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
print(id(z), id(x))

print()
print('======================')
print('======================')
print()


def create_symtype(cls, pytype, shape_env, val):
    from torch._dynamo.source import ConstantSource
    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        # dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        dynamic_dim=DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    return cls(SymNode(
        symbol,
        shape_env,
        pytype,
        hint=val,
    ))


def create_symint(shape_env, i: int):
    return create_symtype(SymInt, int, shape_env, i)


a = create_symint(shape_env, 1)
b = create_symint(shape_env, 1)
c = create_symint(shape_env, 2)
print(a, b, c)
print(type(a), type(b), type(c))
# torch.rand((a, b), requires_grad=False)  # however, cannot construct Tensor!!
print('ok')
