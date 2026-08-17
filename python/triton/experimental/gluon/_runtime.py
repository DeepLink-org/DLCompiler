from __future__ import annotations
from triton.backends.compiler import Language
from triton.compiler.compiler import ASTSource
from triton.runtime.jit import JITFunction, constexpr_function
from typing import Callable, Iterable, Optional, TypeVar, Union
from triton._C.libtriton import ir

from ._layout_autotune import LayoutAutotuneRuntime

T = TypeVar("T")

__all__ = ["constexpr_function", "jit"]


class GluonASTSource(ASTSource):

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        super().__init__(fn, signature, constexprs, attrs)
        self.language = Language.GLUON
        self.ext = "ttgir"

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.compiler.compiler import make_backend
        from triton.compiler.code_generator import ast_to_ttir

        builder = ir.builder(context)
        module = builder.create_module()

        # Assign module attributes eagerly, as they are needed to verify layouts
        backend = make_backend(target)
        target = backend.get_target_name(options)

        module.set_attr("ttg.target", builder.get_string_attr(target))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))

        is_cuda = options.backend_name == "cuda"
        if is_cuda and options.maxnreg is not None:
            module.set_attr("ttg.maxnreg", builder.get_int32_attr(options.maxnreg))

        module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                             module_map=module_map, module=module)
        return module


class GluonJITFunction(JITFunction[T]):

    def __init__(
        self,
        *args,
        enable_gluon_layout_autotune=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_gluon_layout_autotune = bool(enable_gluon_layout_autotune)
        self._layout_autotune_dispatch_args = tuple(
            param.name for param in self.params if param.do_not_specialize
        )
        self._layout_autotune = LayoutAutotuneRuntime()

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True

    def run(self, *args, grid, warmup, **kwargs):
        # Layout autotune is a JIT policy, not a kernel launch argument. Keep
        # the backend option in the compilation key without exposing it at
        # every call site.
        kwargs["enable_gluon_layout_autotune"] = self.enable_gluon_layout_autotune
        return super().run(*args, grid=grid, warmup=warmup, **kwargs)

    def _prepare_kernel_for_launch(self, kernel, **context):
        return self._layout_autotune.prepare(
            kernel,
            jit_key=context.pop("key"),
            do_not_specialize=self._layout_autotune_dispatch_args,
            **context,
        )


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
    enable_gluon_layout_autotune: bool = False,
) -> Union[GluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return GluonJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
            enable_gluon_layout_autotune=enable_gluon_layout_autotune,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
