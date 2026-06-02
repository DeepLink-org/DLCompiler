"""Ascend auto-tune runtime hooks: module-level proxy switch.

Called from DICPDriver.__init__ when backend="ascend".

Why module-level replacement instead of subclassing at import time?
  - ``@triton.autotune()`` creates tuner instances at decoration time
  - Driver init may happen AFTER decoration, so static subclassing won't
    affect existing instances.
  - Module-level replacement of ``triton.autotune`` / ``triton.max_autotune``
    affects ALL subsequently decorated kernels regardless of creation order.

Design: flat module, no classes, no registry.
  - ``_is_ascend_backend()`` dynamically checks whether the active driver
    target is ``"ascend"``.
  - Proxy functions check this at call time and dispatch to either
    stock-triton or ascend-enhanced (lazy-import on first use).
  - ``triton.autotune`` / ``triton.max_autotune`` are replaced **on import**
    of this module, so the proxy is always in place before any user code
    runs ``@triton.autotune``.
"""

import triton

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

# Lazy-loaded ascend implementations (filled on first ascend call).
_ASCEND_AUTOTUNE = None
_ASCEND_MAX_AUTOTUNE = None


def _is_ascend_backend():
    """Return True when the active driver target is 'ascend'.

    ``driver.active`` is a lazy property — the first access triggers
    ``_create_driver()``, which discovers the NPU and constructs a
    ``DICPDriver(target='ascend')``.  We check the ``target`` attribute
    rather than the driver type so the logic is duck-type-safe.
    """
    try:
        return getattr(triton.runtime.driver.active, 'target', None) == 'ascend'
    except Exception:
        return False


def _ascend_autotune_fn():
    """Lazy singleton accessor for ascend ``autotune``."""
    global _ASCEND_AUTOTUNE
    if _ASCEND_AUTOTUNE is None:
        from .ascend_autotune_runtime.autotuner import autotune as _fn
        _ASCEND_AUTOTUNE = _fn
    return _ASCEND_AUTOTUNE


def _ascend_max_autotune_fn():
    """Lazy singleton accessor for ascend ``max_autotune``."""
    global _ASCEND_MAX_AUTOTUNE
    if _ASCEND_MAX_AUTOTUNE is None:
        from .ascend_autotune_runtime.autotuner import max_autotune as _fn
        _ASCEND_MAX_AUTOTUNE = _fn
    return _ASCEND_MAX_AUTOTUNE


# ------------------------------------------------------------------
# Proxies (installed once on import of this module)
# ------------------------------------------------------------------
def _autotune_proxy(configs, key, **kwargs):
    if _is_ascend_backend():
        return _ascend_autotune_fn()(configs=configs, key=key, **kwargs)
    from triton.runtime.autotuner import autotune as _stock
    return _stock(configs=configs, key=key, **kwargs)


def _max_autotune_proxy(configs, key, kernel_type="mixcv", **kwargs):
    if _is_ascend_backend():
        return _ascend_max_autotune_fn()(
            configs=configs, key=key, kernel_type=kernel_type, **kwargs
        )
    # Non-ascend: max_autotune is ascend-only; fall back to plain autotune
    # (ascend-specific params such as kernel_type are silently ignored).
    return triton.autotune(configs=configs, key=key, **kwargs)


# Replace immediately — proxy is always in place, check decides the path.
triton.autotune = _autotune_proxy
triton.max_autotune = _max_autotune_proxy


# ------------------------------------------------------------------
# Public API — kept as no-ops for backward compatibility.
# ------------------------------------------------------------------
def hook_autotune_for_ascend():
    """No-op: autotune auto-detects the ascend backend at call time.

    Retained for backward compatibility — callers such as
    ``DICPDriver.__init__`` and test conftest files still invoke this,
    but the proxy now checks ``driver.active.target`` dynamically.
    """
    pass


def unhook_autotune_for_ascend():
    """No-op: autotune auto-detects the ascend backend at call time.

    Retained for backward compatibility with test teardown code.
    """
    pass
