"""
Declarative compiler flag descriptors for NPU bishengir compilation.

Replaces repetitive imperative::

    if metadata.get("key") is not None:
        options.append(f"--flag={metadata['key']}")

patterns with a data-driven ``CompilerFlag`` mapping whose ``emit()`` method
uniformly yields CLI argument strings via generator semantics.
"""
from dataclasses import dataclass
from itertools import chain
from types import GeneratorType
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Getter = Callable[[dict, Any], Any]
Builder = Callable[[dict, Any], Iterable[str]]


# ---------------------------------------------------------------------------
# CompilerFlag
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompilerFlag:
    """Describes a mapping from compilation metadata to CLI flags.

    Use the three factory constructors rather than the raw constructor:

    * ``CompilerFlag.simple(key, template)``
    * ``CompilerFlag.derived(key, template, getter)``
    * ``CompilerFlag.custom(key, builder)``

    They are mutually exclusive, preventing invalid combinations.
    """

    key: str
    template: str = ""
    getter: Optional[Getter] = None
    builder: Optional[Builder] = None

    def emit(self, metadata: dict, opt: Any) -> Iterable[str]:
        """Yield zero or more CLI argument strings."""
        if self.builder is not None:
            result = self.builder(metadata, opt)
            if result is not None:
                yield from result
            return

        value = (
            self.getter(metadata, opt)
            if self.getter is not None
            else metadata.get(self.key)
        )

        if value is not None:
            if isinstance(value, GeneratorType):
                raise TypeError(
                    f"Getter for CompilerFlag '{self.key}' returned a generator. "
                    f"Use CompilerFlag.custom() with a builder instead of "
                    f"CompilerFlag.derived() for functions that yield."
                )
            yield self.template.format(value=value)

    # -- factories -----------------------------------------------------------

    @classmethod
    def simple(cls, key: str, template: str) -> "CompilerFlag":
        """``metadata[key]``, if not None, is formatted into *template*."""
        return cls(key=key, template=template)

    @classmethod
    def derived(cls, key: str, template: str, getter: Getter) -> "CompilerFlag":
        """Use *getter(metadata, opt)* instead of ``metadata[key]``."""
        return cls(key=key, template=template, getter=getter)

    @classmethod
    def custom(cls, key: str, builder: Builder) -> "CompilerFlag":
        """*builder(metadata, opt)* yields zero or more flag strings."""
        return cls(key=key, builder=builder)


# ---------------------------------------------------------------------------
# Option list builder
# ---------------------------------------------------------------------------


def build_compile_options(
    metadata: dict,
    opt: Any,
    flags: Iterable[CompilerFlag],
    initial: Optional[list[str]] = None,
) -> list[str]:
    """Apply *flags* to produce a flat list of CLI option strings.

    Parameters
    ----------
    metadata : dict
        Compilation metadata (typically includes NPUOptions fields).
    opt : NPUOptions
        The parsed options dataclass.
    flags : iterable of CompilerFlag
        Flag descriptors to process in order.
    initial : list[str] or None
        Pre-populated option strings placed before the flag-derived ones.

    Returns
    -------
    list[str]
    """
    return [
        *(initial or []),
        *chain.from_iterable(flag.emit(metadata, opt) for flag in flags),
    ]
