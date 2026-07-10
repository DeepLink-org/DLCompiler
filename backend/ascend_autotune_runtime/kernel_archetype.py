from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set


class OperatorKind(str, Enum):
    DOT_REGULAR = "DOT_REGULAR"
    DOT_STATEFUL = "DOT_STATEFUL"
    VECTOR_REDUCTION = "VECTOR_REDUCTION"
    VECTOR_AFFINE = "VECTOR_AFFINE"
    VECTOR_DISCRETE_OR_STATEFUL = "VECTOR_DISCRETE_OR_STATEFUL"
    UNKNOWN = "UNKNOWN"


class MemoryAccessKind(str, Enum):
    AFFINE = "AFFINE"
    INDIRECT = "INDIRECT"
    UNKNOWN = "UNKNOWN"


class ControlKind(str, Enum):
    STATIC_OR_RUNTIME = "STATIC_OR_RUNTIME"
    DATA_DEPENDENT = "DATA_DEPENDENT"
    UNKNOWN = "UNKNOWN"


class MemoryEffectKind(str, Enum):
    NORMAL = "NORMAL"
    SCATTER_OR_UNKNOWN = "SCATTER_OR_UNKNOWN"
    UNKNOWN = "UNKNOWN"


class _Taint(str, Enum):
    INDEX = "INDEX"
    MEMORY = "MEMORY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OperatorFeatures:
    has_dot: bool = False
    has_explicit_reduction: bool = False
    memory_access_kind: MemoryAccessKind = MemoryAccessKind.AFFINE
    control_kind: ControlKind = ControlKind.STATIC_OR_RUNTIME
    memory_effect_kind: MemoryEffectKind = MemoryEffectKind.NORMAL
    analysis_failed: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SearchPolicy:
    operator_kind: OperatorKind
    compile_options_kernel_type: str
    stage1_profile_family: str
    stage2_enabled: bool
    allow_final_unit_flag: bool = False
    mixcv_quick_gate: bool = False
    no_dot_shape_values: bool = False


def analyze_operator_ast(
    func_ast: ast.AST,
    scope: Optional[Dict[str, Any]] = None,
    seen: Optional[Set[int]] = None,
) -> OperatorFeatures:
    analyzer = _OperatorFeatureAnalyzer(scope=scope or {}, seen=seen or set())
    try:
        analyzer.visit(func_ast)
    except Exception as exc:  # noqa: BLE001 - classifier must fail closed.
        return OperatorFeatures(
            memory_access_kind=MemoryAccessKind.UNKNOWN,
            control_kind=ControlKind.UNKNOWN,
            memory_effect_kind=MemoryEffectKind.UNKNOWN,
            analysis_failed=True,
            reason=str(exc),
        )
    return analyzer.features()


def classify_operator(features: OperatorFeatures) -> OperatorKind:
    if features.analysis_failed:
        return OperatorKind.UNKNOWN

    if features.has_dot:
        if features.memory_access_kind != MemoryAccessKind.AFFINE:
            return OperatorKind.DOT_STATEFUL
        if features.control_kind != ControlKind.STATIC_OR_RUNTIME:
            return OperatorKind.DOT_STATEFUL
        if features.memory_effect_kind != MemoryEffectKind.NORMAL:
            return OperatorKind.DOT_STATEFUL
        return OperatorKind.DOT_REGULAR

    if features.memory_access_kind != MemoryAccessKind.AFFINE:
        return OperatorKind.VECTOR_DISCRETE_OR_STATEFUL
    if features.control_kind == ControlKind.DATA_DEPENDENT:
        return OperatorKind.VECTOR_DISCRETE_OR_STATEFUL
    if features.memory_effect_kind != MemoryEffectKind.NORMAL:
        return OperatorKind.VECTOR_DISCRETE_OR_STATEFUL
    if features.has_explicit_reduction:
        return OperatorKind.VECTOR_REDUCTION
    return OperatorKind.VECTOR_AFFINE


def make_search_policy(kind: OperatorKind) -> SearchPolicy:
    if kind == OperatorKind.DOT_REGULAR:
        return SearchPolicy(
            operator_kind=kind,
            compile_options_kernel_type="mixcv",
            stage1_profile_family="mixcv",
            stage2_enabled=True,
            allow_final_unit_flag=True,
            mixcv_quick_gate=True,
        )
    if kind == OperatorKind.DOT_STATEFUL:
        return SearchPolicy(
            operator_kind=kind,
            compile_options_kernel_type="mixcv",
            stage1_profile_family="conservative_mixcv",
            stage2_enabled=True,
            allow_final_unit_flag=False,
            mixcv_quick_gate=True,
        )
    if kind == OperatorKind.VECTOR_REDUCTION:
        return SearchPolicy(
            operator_kind=kind,
            compile_options_kernel_type="vector",
            stage1_profile_family="vector",
            stage2_enabled=True,
            no_dot_shape_values=True,
        )
    if kind == OperatorKind.VECTOR_AFFINE:
        return SearchPolicy(
            operator_kind=kind,
            compile_options_kernel_type="vector",
            stage1_profile_family="vector",
            stage2_enabled=True,
            no_dot_shape_values=True,
        )
    if kind == OperatorKind.VECTOR_DISCRETE_OR_STATEFUL:
        return SearchPolicy(
            operator_kind=kind,
            compile_options_kernel_type="vector",
            stage1_profile_family="vector",
            stage2_enabled=False,
            no_dot_shape_values=True,
        )
    return SearchPolicy(
        operator_kind=OperatorKind.UNKNOWN,
        compile_options_kernel_type="vector",
        stage1_profile_family="vector",
        stage2_enabled=False,
        no_dot_shape_values=True,
    )


def analyze_operator_policy(
    func_ast: ast.AST,
    scope: Optional[Dict[str, Any]] = None,
    seen: Optional[Set[int]] = None,
) -> tuple[OperatorFeatures, SearchPolicy]:
    features = analyze_operator_ast(func_ast, scope=scope, seen=seen)
    kind = classify_operator(features)
    return features, make_search_policy(kind)


class _OperatorFeatureAnalyzer(ast.NodeVisitor):
    _REDUCTION_CALLS = {"sum", "max", "min", "prod"}
    _DOT_CALLS = {"dot", "dot_scaled"}
    _AFFINE_TL_CALLS = {
        "arange",
        "program_id",
        "cdiv",
        "minimum",
        "maximum",
        "where",
        "zeros",
        "full",
        "broadcast_to",
        "expand_dims",
    }

    def __init__(self, *, scope: Dict[str, Any], seen: Set[int]):
        self.scope = scope
        self.seen = seen
        self.env: Dict[str, _Taint] = {}
        self.has_dot = False
        self.has_explicit_reduction = False
        self.memory_access_kind = MemoryAccessKind.AFFINE
        self.control_kind = ControlKind.STATIC_OR_RUNTIME
        self.memory_effect_kind = MemoryEffectKind.NORMAL

    def features(self) -> OperatorFeatures:
        return OperatorFeatures(
            has_dot=self.has_dot,
            has_explicit_reduction=self.has_explicit_reduction,
            memory_access_kind=self.memory_access_kind,
            control_kind=self.control_kind,
            memory_effect_kind=self.memory_effect_kind,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Assign(self, node: ast.Assign):
        taint = self._expr_taint(node.value)
        for target in node.targets:
            self._assign_target(target, taint)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        taint = (
            self._expr_taint(node.value) if node.value is not None else _Taint.UNKNOWN
        )
        self._assign_target(node.target, taint)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        taint = self._combine_taint(
            self._expr_taint(node.target), self._expr_taint(node.value)
        )
        self._assign_target(node.target, taint)
        self.visit(node.value)

    def visit_For(self, node: ast.For):
        self._update_control_kind(self._expr_taint(node.iter))
        self._assign_target(node.target, _Taint.INDEX)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_If(self, node: ast.If):
        self._update_control_kind(self._expr_taint(node.test))
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While):
        self._update_control_kind(self._expr_taint(node.test))
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Call(self, node: ast.Call):
        if self._is_tl_call(node.func, self._DOT_CALLS):
            self.has_dot = True
        if self._is_tl_call(node.func, self._REDUCTION_CALLS):
            self.has_explicit_reduction = True
        if self._is_tl_call(node.func, {"load"}):
            self._record_load_address(node)
        elif self._is_tl_call(node.func, {"store"}):
            self._record_store_address(node)
        else:
            self._merge_called_jit_features(node.func)
        self.generic_visit(node)

    def _record_load_address(self, node: ast.Call):
        if not node.args:
            self.memory_access_kind = MemoryAccessKind.UNKNOWN
            return
        taint = self._expr_taint(node.args[0])
        if taint == _Taint.MEMORY:
            self.memory_access_kind = MemoryAccessKind.INDIRECT
        elif (
            taint == _Taint.UNKNOWN
            and self.memory_access_kind != MemoryAccessKind.INDIRECT
        ):
            self.memory_access_kind = MemoryAccessKind.UNKNOWN

    def _record_store_address(self, node: ast.Call):
        if not node.args:
            self.memory_effect_kind = MemoryEffectKind.UNKNOWN
            return
        taint = self._expr_taint(node.args[0])
        if taint in {_Taint.MEMORY, _Taint.UNKNOWN}:
            self.memory_effect_kind = MemoryEffectKind.SCATTER_OR_UNKNOWN

    def _merge_called_jit_features(self, func: ast.AST):
        if not isinstance(func, ast.Name):
            return
        callee = self.scope.get(func.id)
        if callee is None or not callable(getattr(callee, "parse", None)):
            return
        callee_id = id(callee)
        if callee_id in self.seen:
            return
        self.seen.add(callee_id)
        callee_scope = (
            callee.get_capture_scope()
            if callable(getattr(callee, "get_capture_scope", None))
            else self.scope
        )
        features = analyze_operator_ast(
            callee.parse(), scope=callee_scope, seen=self.seen
        )
        self.has_dot = self.has_dot or features.has_dot
        self.has_explicit_reduction = (
            self.has_explicit_reduction or features.has_explicit_reduction
        )
        self.memory_access_kind = _merge_memory_access(
            self.memory_access_kind, features.memory_access_kind
        )
        self.control_kind = _merge_control(self.control_kind, features.control_kind)
        self.memory_effect_kind = _merge_memory_effect(
            self.memory_effect_kind, features.memory_effect_kind
        )

    def _assign_target(self, target: ast.AST, taint: _Taint):
        if isinstance(target, ast.Name):
            self.env[target.id] = taint
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign_target(elt, taint)

    def _update_control_kind(self, taint: _Taint):
        if taint == _Taint.MEMORY:
            self.control_kind = ControlKind.DATA_DEPENDENT
        elif (
            taint == _Taint.UNKNOWN and self.control_kind != ControlKind.DATA_DEPENDENT
        ):
            self.control_kind = ControlKind.UNKNOWN

    def _expr_taint(self, node: Optional[ast.AST]) -> _Taint:
        if node is None:
            return _Taint.INDEX
        if isinstance(node, ast.Name):
            return self.env.get(node.id, _Taint.INDEX)
        if isinstance(node, ast.Constant):
            return _Taint.INDEX
        if isinstance(node, ast.Attribute):
            return self._expr_taint(node.value)
        if isinstance(node, ast.Subscript):
            return self._combine_taint(
                self._expr_taint(node.value), self._expr_taint(node.slice)
            )
        if isinstance(node, ast.Slice):
            return self._combine_all(
                self._expr_taint(node.lower),
                self._expr_taint(node.upper),
                self._expr_taint(node.step),
            )
        if isinstance(node, ast.UnaryOp):
            return self._expr_taint(node.operand)
        if isinstance(node, ast.BinOp):
            return self._combine_taint(
                self._expr_taint(node.left), self._expr_taint(node.right)
            )
        if isinstance(node, ast.BoolOp):
            return self._combine_all(
                *(self._expr_taint(value) for value in node.values)
            )
        if isinstance(node, ast.Compare):
            return self._combine_all(
                self._expr_taint(node.left),
                *(self._expr_taint(comp) for comp in node.comparators),
            )
        if isinstance(node, ast.IfExp):
            return self._combine_all(
                self._expr_taint(node.test),
                self._expr_taint(node.body),
                self._expr_taint(node.orelse),
            )
        if isinstance(node, ast.Call):
            return self._call_taint(node)
        if isinstance(node, (ast.Tuple, ast.List)):
            return self._combine_all(*(self._expr_taint(elt) for elt in node.elts))
        return _Taint.UNKNOWN

    def _call_taint(self, node: ast.Call) -> _Taint:
        arg_taint = self._combine_all(
            *(self._expr_taint(arg) for arg in node.args),
            *(self._expr_taint(keyword.value) for keyword in node.keywords),
        )
        if self._is_tl_call(node.func, {"load"}):
            return _Taint.MEMORY
        if self._is_tl_call(node.func, self._DOT_CALLS | self._REDUCTION_CALLS):
            return arg_taint
        if self._is_tl_call(node.func, self._AFFINE_TL_CALLS):
            return arg_taint
        if isinstance(node.func, ast.Name) and node.func.id in {
            "range",
            "min",
            "max",
            "int",
        }:
            return arg_taint
        if arg_taint == _Taint.MEMORY:
            return _Taint.MEMORY
        return _Taint.UNKNOWN

    @staticmethod
    def _is_tl_call(func: ast.AST, names: Set[str]) -> bool:
        return (
            isinstance(func, ast.Attribute)
            and func.attr in names
            and isinstance(func.value, ast.Name)
            and func.value.id == "tl"
        )

    @staticmethod
    def _combine_taint(left: _Taint, right: _Taint) -> _Taint:
        if _Taint.MEMORY in (left, right):
            return _Taint.MEMORY
        if _Taint.UNKNOWN in (left, right):
            return _Taint.UNKNOWN
        return _Taint.INDEX

    def _combine_all(self, *values: _Taint) -> _Taint:
        result = _Taint.INDEX
        for value in values:
            result = self._combine_taint(result, value)
        return result


def _merge_memory_access(
    left: MemoryAccessKind, right: MemoryAccessKind
) -> MemoryAccessKind:
    if MemoryAccessKind.INDIRECT in (left, right):
        return MemoryAccessKind.INDIRECT
    if MemoryAccessKind.UNKNOWN in (left, right):
        return MemoryAccessKind.UNKNOWN
    return MemoryAccessKind.AFFINE


def _merge_control(left: ControlKind, right: ControlKind) -> ControlKind:
    if ControlKind.DATA_DEPENDENT in (left, right):
        return ControlKind.DATA_DEPENDENT
    if ControlKind.UNKNOWN in (left, right):
        return ControlKind.UNKNOWN
    return ControlKind.STATIC_OR_RUNTIME


def _merge_memory_effect(
    left: MemoryEffectKind, right: MemoryEffectKind
) -> MemoryEffectKind:
    if MemoryEffectKind.SCATTER_OR_UNKNOWN in (left, right):
        return MemoryEffectKind.SCATTER_OR_UNKNOWN
    if MemoryEffectKind.UNKNOWN in (left, right):
        return MemoryEffectKind.UNKNOWN
    return MemoryEffectKind.NORMAL
