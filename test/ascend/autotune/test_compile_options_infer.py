import ast

import pytest
from triton.runtime.autotuner import Config

from backend.ascend_autotune_runtime.autoparser import DotCallParser
from backend.ascend_autotune_runtime.autotuner import AutoTilingTuner


def _func_ast(source):
    return ast.parse(source)


def _make_dummy_kernel(source=None, parse_exc=None, scope=None):
    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []

    def _parse():
        if parse_exc is not None:
            raise parse_exc
        return _func_ast(source)

    _dummy_kernel.parse = _parse
    _dummy_kernel.get_capture_scope = lambda: scope or {}
    return _dummy_kernel


def _make_tuner(fn, hints=None):
    return AutoTilingTuner(
        fn,
        fn.arg_names,
        [Config({})],
        [],
        None,
        None,
        hints=hints,
    )


def test_dot_call_parser_detects_tl_dot():
    func_ast = _func_ast("""
def kernel(a, b, c):
    acc = tl.dot(a, b)
    return acc
""")

    assert DotCallParser(func_ast).parse() is True


def test_dot_call_parser_detects_tl_dot_scaled():
    func_ast = _func_ast("""
def kernel(a, b, scales):
    acc = tl.dot_scaled(a, b, scales)
    return acc
""")

    assert DotCallParser(func_ast).parse() is True


def test_dot_call_parser_ignores_non_tl_dot_alias():
    func_ast = _func_ast("""
def kernel(a, b):
    acc = dot(a, b)
    return acc
""")

    assert DotCallParser(func_ast).parse() is False


def test_dot_call_parser_detects_dot_in_called_jit_helper():
    helper = _make_dummy_kernel("""
def helper(a, b):
    acc = tl.dot(a, b)
    return acc
""")
    func_ast = _func_ast("""
def kernel(a, b):
    return helper(a, b)
""")

    assert DotCallParser(func_ast, {"helper": helper}).parse() is True


def test_compile_options_infers_mixcv_when_kernel_has_dot():
    fn = _make_dummy_kernel("""
def kernel(a, b):
    acc = tl.dot(a, b)
    return acc
""")

    tuner = _make_tuner(fn)

    assert tuner.hints["compile_options"] == "mixcv"
    assert tuner.compile_options.enabled is True
    assert tuner.compile_options.kernel_type == "mixcv"


def test_compile_options_infers_mixcv_when_called_helper_has_dot():
    helper = _make_dummy_kernel("""
def helper(a, b):
    acc = tl.dot(a, b)
    return acc
""")
    fn = _make_dummy_kernel("""
def kernel(a, b):
    return helper(a, b)
""", scope={"helper": helper})

    tuner = _make_tuner(fn)

    assert tuner.hints["compile_options"] == "mixcv"
    assert tuner.compile_options.kernel_type == "mixcv"


def test_compile_options_infers_vector_when_kernel_has_no_dot():
    fn = _make_dummy_kernel("""
def kernel(a, b):
    acc = a + b
    return acc
""")

    tuner = _make_tuner(fn)

    assert tuner.hints["compile_options"] == "vector"
    assert tuner.compile_options.enabled is True
    assert tuner.compile_options.kernel_type == "vector"


def test_compile_options_infers_through_libentry_like_wrapper():
    jit_fn = _make_dummy_kernel("""
def kernel(a, b):
    acc = tl.dot(a, b)
    return acc
""")

    class LibEntryLike:
        def __init__(self, fn):
            self.fn = fn
            self.jit_function = fn
            self.arg_names = fn.arg_names

    tuner = _make_tuner(LibEntryLike(jit_fn))

    assert tuner.ast_fn is jit_fn
    assert tuner.hints["compile_options"] == "mixcv"
    assert tuner.compile_options.kernel_type == "mixcv"


def test_compile_options_explicit_hint_is_not_overridden():
    fn = _make_dummy_kernel(parse_exc=RuntimeError("parse should not run"))

    tuner = _make_tuner(fn, hints={"compile_options": "vector"})

    assert tuner.hints["compile_options"] == "vector"
    assert tuner.compile_options.kernel_type == "vector"


def test_compile_options_inference_failure_requires_explicit_hint():
    fn = _make_dummy_kernel(parse_exc=RuntimeError("parse failed"))

    with pytest.raises(ValueError, match="Cannot infer Ascend compile_options"):
        _make_tuner(fn)
