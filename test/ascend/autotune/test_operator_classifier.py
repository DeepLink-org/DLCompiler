import ast

from backend.ascend_autotune_runtime.kernel_archetype import (
    MemoryAccessKind,
    OperatorKind,
    analyze_operator_ast,
    classify_operator,
)


def _classify(source: str):
    features = analyze_operator_ast(ast.parse(source))
    return features, classify_operator(features)


def test_tl_arange_only_is_not_reduction():
    features, kind = _classify(
        """
def kernel(x, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(x + offs, mask=mask)
    tl.store(y + offs, vals + 1, mask=mask)
"""
    )

    assert not features.has_explicit_reduction
    assert kind == OperatorKind.VECTOR_AFFINE


def test_explicit_tl_sum_is_vector_reduction():
    features, kind = _classify(
        """
def kernel(x, y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tl.load(x + offs, mask=offs < n, other=0.0)
    total = tl.sum(vals)
    tl.store(y, total)
"""
    )

    assert features.has_explicit_reduction
    assert kind == OperatorKind.VECTOR_REDUCTION


def test_pointer_chasing_is_vector_discrete_or_stateful():
    features, kind = _classify(
        """
def kernel(ptrs, y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = tl.load(ptrs + pid)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(base + offs)
    tl.store(y + pid * BLOCK + offs, vals)
"""
    )

    assert features.memory_access_kind == MemoryAccessKind.INDIRECT
    assert kind == OperatorKind.VECTOR_DISCRETE_OR_STATEFUL


def test_dot_with_indirect_load_is_dot_stateful():
    _, kind = _classify(
        """
def kernel(ptrs, b, c, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = tl.load(ptrs + pid)
    offs = tl.arange(0, BLOCK)
    x = tl.load(base + offs)
    y = tl.load(b + offs)
    z = tl.dot(x[:, None], y[None, :])
    tl.store(c + pid * BLOCK + offs, z)
"""
    )

    assert kind == OperatorKind.DOT_STATEFUL
