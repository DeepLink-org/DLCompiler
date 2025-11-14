from triton.language import core


class async_task:
    """
    Context manager to run code fragments asynchronously.
    """
    vector = 'vector'
    cube = 'cube'

    def __init__(self, *args, _builder=None, **kwargs):
        self.builder = _builder
        self.scope = core._unwrap_if_constexpr(kwargs.get("scope", None))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
