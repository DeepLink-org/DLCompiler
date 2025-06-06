from triton.language import core, math
from triton.language.standard import max, sum
from triton.runtime.jit import jit

@core._tensor_member_fn
@jit
def flip(x, dim=None):
    """
    Flips a tensor `x` along the dimension `dim`.

    :param x: the first input tensor
    :type x: Block
    :param dim: the dimension to flip along (currently only final dimension supported)
    :type dim: int
    """
    core.static_print("tl.flip is unsupported for now. Use libdevice.flip instead.")
    core.static_assert(False)
    return x

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    assert core.constexpr(x.dtype.is_floating()), "Unexpected dtype"
    return 1 / (1 + math.exp(-x))

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    assert core.constexpr(x.dtype.is_floating()), "Unexpected dtype"
    z = x - max(x, 0)
    num = math.exp(z)
    den = sum(num, 0)
    return math.fdiv(num, den, ieee_rounding)
