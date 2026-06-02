# The proxy replaces triton.autotune / triton.max_autotune at import time.
# We import this early so every test module sees the proxy (which auto-detects
# ascend via triton.runtime.driver.active.target at call time).
import backend.ascend_autotune_hooks  # noqa: F401 — side-effect import
