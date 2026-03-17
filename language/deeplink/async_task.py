from triton.language import core


class async_task:
    """
    Context manager to run code fragments asynchronously.
    Supports both NPU (with scope) and GPGPU (without scope) architectures.
    """

    # Class attributes for NPU vector/cube operations
    vector = "vector"
    cube = "cube"

    def __init__(self, *args, _builder=None, **kwargs):
        """
        Initialize async_task for either NPU or GPGPU.
        
        NPU mode: uses 'scope' parameter (e.g., async_task(scope="vector"))
        GPGPU mode: uses task_ids or resource specifications
        """
        self.builder = _builder
        self.scope = core._unwrap_if_constexpr(kwargs.get("scope", None))
        
        # NPU mode: scope is provided
        if self.scope is not None:
            # NPU-specific initialization (simple case)
            pass
        else:
            # GPGPU mode initialization
            self._init_gpgpu(args, kwargs)

    def _init_gpgpu(self, args, kwargs):
        """Initialize for GPGPU architecture."""
        self.is_default = False
        self.is_explict = False
        self.task_ids = None
        self.num_warps = None
        self.num_regs = None
        self.replicate = None
        
        if args:
            # Case 1: Explicit task IDs provided
            assert len(args) == 1, "Expected exactly one argument"
            if isinstance(args[0], core.constexpr) and args[0] == "default":
                # Case 1a: Default task specification
                self.is_explict = True
                self.is_default = True
                self.num_regs = core._unwrap_if_constexpr(
                    kwargs.get("num_regs", kwargs.get("registers", None))
                )
                self.replicate = core._unwrap_if_constexpr(
                    kwargs.get("replicate", 1)
                )
            else:
                # Case 1b: Specific task IDs
                self.task_ids = list({
                    core._unwrap_if_constexpr(tid) for tid in args[0]
                })
        else:
            # Case 2: Resource-based specification
            self.is_explict = True
            self.num_warps = core._unwrap_if_constexpr(
                kwargs.get("num_warps", None)
            )
            self.num_regs = core._unwrap_if_constexpr(
                kwargs.get("num_regs", kwargs.get("registers", None))
            )
            self.replicate = core._unwrap_if_constexpr(
                kwargs.get("replicate", 1)
            )

    def __enter__(self):
        """Enter the asynchronous execution context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the asynchronous execution context."""
        pass


class async_tasks:
    """
    Context manager for grouping multiple async operations (GPGPU only).
    """
    
    def __init__(self):
        """Initialize async tasks group."""
        pass

    def __enter__(self):
        """Enter the grouped asynchronous execution context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the grouped asynchronous execution context."""
        pass